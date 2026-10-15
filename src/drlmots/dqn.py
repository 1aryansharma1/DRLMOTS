"""CPU DQN replay, masked targets, target synchronization and live control."""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random

import torch
from torch.nn import functional as F

from drlmots.entities import ResourceTier, Task
from drlmots.normalization import _digest
from drlmots.q_network import MaskedSelector, NetworkConfig, NetworkContext, QNetwork, _seed
from drlmots.rl_state import RLRewardConfig, RLStateSchema, VARIANTS, encode_frame, frame_reward, variant_mask


def _positive_int(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")


def _vector(values):
    result = []
    for value in values:
        Task._validate_number("normalized replay value", value, allow_zero=True)
        if value > 1:
            raise ValueError("normalized replay values must be in [0,1]")
        result.append(float(value))
    if not result:
        raise ValueError("replay state must not be empty")
    return tuple(result)


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 20
    seed: int = 42
    training_seed_start: int = 2000
    calibration_episodes: int = 3
    evaluation_seeds: tuple[int, ...] = (3000, 3001, 3002)
    hidden_widths: tuple[int, ...] = (50, 50)
    learning_rate: float = 0.001
    gamma: float = 0.95
    replay_capacity: int = 2000
    batch_size: int = 32
    target_sync_updates: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000
    max_grad_norm: float = 10.0
    latency_scale_ms: float = 10.0
    rejection_penalty: float = 1.0

    def __post_init__(self):
        for name in ("episodes", "calibration_episodes", "replay_capacity", "batch_size",
                     "target_sync_updates", "epsilon_decay_steps"):
            _positive_int(getattr(self, name), name)
        _seed(self.seed)
        _seed(self.training_seed_start)
        if self.training_seed_start + max(self.episodes, self.calibration_episodes) >= 2**63:
            raise ValueError("training seed range is too large")
        widths = NetworkConfig(self.hidden_widths, self.seed).hidden_widths
        seeds = tuple(self.evaluation_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("evaluation seeds must be nonempty and unique")
        for seed in seeds:
            _seed(seed)
            if self.training_seed_start <= seed < self.training_seed_start + max(self.episodes, self.calibration_episodes):
                raise ValueError("training/calibration and evaluation seeds must be disjoint")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch size must fit replay capacity")
        for name in ("learning_rate", "max_grad_norm", "latency_scale_ms", "rejection_penalty"):
            Task._validate_number(name, getattr(self, name), allow_zero=False)
        for name in ("gamma", "epsilon_start", "epsilon_end"):
            value = getattr(self, name)
            Task._validate_number(name, value, allow_zero=True)
            if value > 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon end must not exceed epsilon start")
        object.__setattr__(self, "evaluation_seeds", seeds)
        object.__setattr__(self, "hidden_widths", widths)

    def manifest(self):
        return asdict(self)

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != set(cls().manifest()):
            raise ValueError("training config must contain exactly the documented fields")
        return cls(**data)


@dataclass(frozen=True)
class Experience:
    state: tuple[float, ...]
    mask: tuple[bool, ...]
    action: int
    reward: float
    next_state: tuple[float, ...] | None
    next_mask: tuple[bool, ...] | None
    bootstrap_discount: float
    discount_steps: int
    done: bool

    def __post_init__(self):
        state, mask = _vector(self.state), tuple(self.mask)
        if not mask or any(type(value) is not bool for value in mask):
            raise ValueError("replay mask must contain booleans")
        if type(self.action) is not int or not 0 <= self.action < len(mask) or not mask[self.action]:
            raise ValueError("replay action must be legal")
        if type(self.done) is not bool:
            raise ValueError("terminal flag must be bool")
        Task._validate_number("replay reward magnitude", abs(self.reward) if type(self.reward) in (int, float) else self.reward,
                              allow_zero=True)
        Task._validate_number("bootstrap_discount", self.bootstrap_discount, allow_zero=True)
        _positive_int(self.discount_steps, "discount_steps")
        if self.bootstrap_discount > 1:
            raise ValueError("bootstrap discount must be in [0,1]")
        if self.done:
            if self.next_state is not None or self.next_mask is not None or self.bootstrap_discount != 0:
                raise ValueError("terminal transition cannot bootstrap")
        else:
            if self.next_state is None or self.next_mask is None:
                raise ValueError("nonterminal transition requires next state and mask")
            following, following_mask = _vector(self.next_state), tuple(self.next_mask)
            if (len(following) != len(state) or len(following_mask) != len(mask)
                    or any(type(value) is not bool for value in following_mask) or not any(following_mask)):
                raise ValueError("nonterminal state/mask must match dimensions and have a legal action")
            object.__setattr__(self, "next_state", following)
            object.__setattr__(self, "next_mask", following_mask)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "mask", mask)


def restore_rng(rng, state):
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (tuple, list)) else value
    try:
        rng.setstate(tuples(state))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid local RNG state") from error


class ReplayBuffer:
    def __init__(self, capacity, dimension, actions, context, seed):
        for value, name in ((capacity, "capacity"), (dimension, "dimension"), (actions, "actions")):
            _positive_int(value, name)
        self.capacity, self.dimension, self.actions, self.context = capacity, dimension, actions, context
        self.items = deque(maxlen=capacity)
        self.rng = random.Random(_seed(seed))

    def add(self, item, *, context):
        if context != self.context:
            raise ValueError("incompatible replay context")
        if not isinstance(item, Experience) or len(item.state) != self.dimension or len(item.mask) != self.actions:
            raise ValueError("incompatible replay transition shape")
        self.items.append(item)

    def sample(self, size):
        _positive_int(size, "sample size")
        if size > len(self.items):
            raise ValueError("not enough replay samples")
        return tuple(self.rng.sample(tuple(self.items), size))

    def __len__(self):
        return len(self.items)


def bellman_targets(target, experiences, context):
    if not experiences:
        raise ValueError("target batch must not be empty")
    rewards = torch.tensor([item.reward for item in experiences], dtype=torch.float32, device="cpu")
    indices = [i for i, item in enumerate(experiences) if not item.done and item.bootstrap_discount > 0]
    with torch.no_grad():
        if indices:
            states = torch.tensor([experiences[i].next_state for i in indices], dtype=torch.float32, device="cpu")
            masks = torch.tensor([experiences[i].next_mask for i in indices], dtype=torch.bool, device="cpu")
            values = target(states, context=context).masked_fill(~masks, -torch.inf).max(dim=1).values
            discounts = torch.tensor([experiences[i].bootstrap_discount for i in indices], dtype=torch.float32)
            rewards[indices] += discounts * values
    if not torch.isfinite(rewards).all():
        raise ValueError("Bellman targets must be finite")
    return rewards


class Learner:
    def __init__(self, normalizer, config, variant):
        if variant not in VARIANTS or not isinstance(normalizer.context.schema, RLStateSchema):
            raise ValueError("learner requires a contract-2 scaler and a supported variant")
        self.normalizer, self.config, self.variant = normalizer, config, variant
        self.reward = RLRewardConfig(config.latency_scale_ms, config.rejection_penalty)
        self.context = NetworkContext.from_artifacts(normalizer, self.reward)
        self.context_sha = _digest(self.context.manifest())
        self.online = QNetwork(self.context, NetworkConfig(config.hidden_widths, config.seed))
        self.target = deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        self.selector = MaskedSelector((config.seed + 1) % 2**63)
        self.replay = ReplayBuffer(config.replay_capacity, normalizer.context.schema.dimension,
                                   len(normalizer.context.schema.vm_ids), self.context_sha, (config.seed + 2) % 2**63)
        self.updates, self.decision_steps, self.episodes_completed = 0, 0, 0
        self.losses, self.history = [], []
        self.training_task_hashes = set()
        self.run_config = None

    @property
    def epsilon(self):
        fraction = min(self.decision_steps / self.config.epsilon_decay_steps, 1)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def remember(self, item):
        self.replay.add(item, context=self.context_sha)
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device="cpu")
        actions = torch.tensor([item.action for item in batch], dtype=torch.int64, device="cpu")
        targets = bellman_targets(self.target, batch, self.context)
        predictions = self.online(states, context=self.context).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise ValueError("training loss must be finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.online.parameters()):
            raise ValueError("updated parameters must be finite")
        self.updates += 1
        value = float(loss.detach())
        self.losses.append(value)
        if self.updates % self.config.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict(), strict=True)
        return value


def baseline_action(frame, policy):
    legal = [i for i, vm in enumerate(frame.vms) if vm.estimate is not None]
    if not legal:
        return None
    if policy == "first_fit_fcfs":
        return legal[0]
    if policy == "fog_first_completion":
        return min(legal, key=lambda i: (frame.vms[i].estimate.predicted_finish_time_ms,
                                         frame.vms[i].tier is ResourceTier.CLOUD, frame.vms[i].vm_id))
    if policy != "least_queue_fcfs":
        raise ValueError("unknown calibration/baseline policy")
    return min(legal, key=lambda i: (len(frame.vms[i].queued) + int(frame.vms[i].running_remaining_ms > 0),
                                     frame.vms[i].vm_id))


class BaselineController:
    def __init__(self, schema, reward, policy="least_queue_fcfs"):
        self.schema, self.reward_config, self.policy = schema, reward, policy
        self.records, self.states = [], []

    def __call__(self, frame):
        state = encode_frame(self.schema, frame)
        action = baseline_action(frame, self.policy)
        reward = frame_reward(frame, action, self.reward_config)
        self.states.append(state)
        self.records.append({"task_id": frame.task_id, "time_ms": frame.time_ms, "action": action,
                             "vm_id": frame.vms[action].vm_id if action is not None else None,
                             "legal_mask": frame.mask, "action_mask": frame.mask, "reward": reward})
        return self.records[-1]["vm_id"]


class EpisodeController:
    def __init__(self, learner, *, learn):
        self.learner, self.learn = learner, learn
        self.records, self.pending = [], None
        self.transitions_added = 0

    def _close(self, next_state=None, next_mask=None):
        if self.pending is None:
            return
        pending = self.pending
        terminal = next_state is None
        item = Experience(pending["state"], pending["mask"], pending["action"], pending["reward"],
                          next_state, next_mask, 0.0 if terminal else self.learner.config.gamma ** pending["steps"],
                          pending["steps"], terminal)
        self.learner.remember(item)
        self.transitions_added += 1
        self.pending = None

    def __call__(self, frame):
        agent = self.learner
        raw = encode_frame(agent.normalizer.context.schema, frame)
        state = agent.normalizer.transform(agent.normalizer.context, raw)
        mask = variant_mask(frame, agent.variant)
        action, values = None, None
        epsilon = agent.epsilon if self.learn else 0.0
        if any(mask):
            if self.learn:
                self._close(state, mask)
                epsilon = agent.epsilon
            with torch.no_grad():
                q = agent.online(torch.tensor([state], dtype=torch.float32, device="cpu"), context=agent.context)[0]
            values = q.tolist()
            action = agent.selector.select(q, mask, epsilon=epsilon)
            if self.learn:
                agent.decision_steps += 1
        reward = frame_reward(frame, action, agent.reward)
        if self.learn:
            if action is not None:
                self.pending = {"state": state, "mask": mask, "action": action, "reward": reward, "steps": 1}
            elif self.pending is not None:
                self.pending["reward"] = math.fsum((self.pending["reward"],
                                                     agent.config.gamma ** self.pending["steps"] * reward))
                self.pending["steps"] += 1
        vm_id = frame.vms[action].vm_id if action is not None else None
        estimate = frame.vms[action].estimate if action is not None else None
        self.records.append({"task_id": frame.task_id, "time_ms": frame.time_ms, "action": action,
                             "vm_id": vm_id, "legal_mask": frame.mask, "action_mask": mask,
                             "raw_state": raw, "normalized_state": state, "q_values": values,
                             "epsilon": epsilon, "reward": reward,
                             "predicted_finish_time_ms": estimate.predicted_finish_time_ms if estimate else None})
        return vm_id

    def finish(self):
        if self.learn:
            self._close()


def reward_summary(records):
    total = math.fsum(item["reward"] for item in records)
    Task._validate_number("episode return magnitude", abs(total), allow_zero=True)
    return {"all_arrival_count": len(records), "undiscounted_return": total,
            "average_reward": total / len(records) if records else None,
            "forced_rejections": sum(item["action"] is None for item in records)}
