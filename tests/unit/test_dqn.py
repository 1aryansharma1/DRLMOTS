from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import random

import pytest
import torch

from drlmots.dqn import Experience, Learner, ReplayBuffer, TrainingConfig, bellman_targets
from drlmots.q_network import QNetwork
from drlmots.run_config import load_run_config
from drlmots.train import cpu_execution, initialize


ROOT = Path(__file__).resolve().parents[2]


def experience(**changes):
    return Experience(**(dict(state=(0.0, 0.5), mask=(True, False), action=0, reward=1.0,
                              next_state=(0.5, 1.0), next_mask=(True, False), bootstrap_discount=0.5,
                              discount_steps=1, done=False) | changes))


def test_replay_eviction_local_sampling_and_detached_immutable_records():
    before = random.getstate()
    first, second = ReplayBuffer(2, 2, 2, "context", 42), ReplayBuffer(2, 2, 2, "context", 42)
    state, mask = [0, 0.5], [True, False]
    item = experience(state=state, mask=mask)
    state[0], mask[0] = 1, False
    assert item.state == (0, 0.5) and item.mask == (True, False)
    with pytest.raises(FrozenInstanceError):
        item.action = 1
    for reward in (1, 2, 3):
        for buffer in (first, second):
            buffer.add(experience(reward=reward), context="context")
    assert [item.reward for item in first.items] == [2, 3]
    assert first.sample(2) == second.sample(2)
    assert random.getstate() == before
    for size in (0, 3, True):
        with pytest.raises(ValueError):
            first.sample(size)
    with pytest.raises(ValueError, match="context"):
        first.add(item, context="different")
    with pytest.raises(ValueError, match="shape"):
        ReplayBuffer(1, 3, 2, "context", 42).add(item, context="context")


@pytest.mark.parametrize("changes", [
    {"action": 1}, {"action": None}, {"action": True}, {"mask": (1, 0)},
    {"state": (0, float("nan"))}, {"state": (0, 1.1)}, {"state": (True, 0)},
    {"reward": float("inf")}, {"reward": True}, {"bootstrap_discount": 1.1},
    {"discount_steps": 0}, {"next_mask": (False, False)}, {"next_state": None},
    {"next_state": (1,)}, {"done": True}, {"done": 1},
])
def test_invalid_experience_cannot_enter_replay(changes):
    with pytest.raises((ValueError, TypeError)):
        experience(**changes)


def test_hand_masked_bellman_targets_and_terminal_no_bootstrap():
    class FixedTarget:
        def __call__(self, states, *, context):
            assert not torch.is_grad_enabled()
            assert len(states) == 2
            return torch.tensor([[2, 100], [-3, 200]], dtype=torch.float32)
    samples = [experience(), experience(reward=-1, bootstrap_discount=0.25, discount_steps=2),
               experience(reward=-2, next_state=None, next_mask=None, bootstrap_discount=0, done=True),
               experience(reward=4, bootstrap_discount=0)]
    assert bellman_targets(FixedTarget(), samples, None).tolist() == [2, -1.75, -2, 4]
    with pytest.raises(ValueError, match="finite"):
        bellman_targets(FixedTarget(), [experience(reward=1e308, bootstrap_discount=0)], None)


@pytest.mark.parametrize("changes", [
    {"gamma": -1}, {"gamma": True}, {"learning_rate": 0}, {"episodes": 0},
    {"batch_size": 2001}, {"target_sync_updates": False}, {"epsilon_start": 0.01, "epsilon_end": 0.1},
    {"evaluation_seeds": (2000,)}, {"evaluation_seeds": (3000, 3000)},
    {"max_grad_norm": float("nan")}, {"hidden_widths": (0, 50)},
])
def test_explicit_training_settings_validate(changes):
    with pytest.raises((ValueError, TypeError)):
        replace(TrainingConfig(), **changes)


def test_online_updates_are_finite_and_target_is_independent_until_sync():
    config = load_run_config(ROOT / "configs/normalization_train.yaml")
    config = replace(config, workload=replace(config.workload, task_count=8))
    settings = TrainingConfig(episodes=2, calibration_episodes=1, hidden_widths=(8, 8),
                              batch_size=1, replay_capacity=8, target_sync_updates=2)
    with cpu_execution():
        learner = initialize(config, settings, "standard")
        initial = {key: value.clone() for key, value in learner.online.state_dict().items()}
        dimension = learner.normalizer.context.schema.dimension
        actions = len(learner.normalizer.context.schema.vm_ids)
        item = Experience((0.5,) * dimension, (True,) * actions, 0, 1, None, None, 0, 1, True)
        learner.remember(item)
        assert learner.updates == 1 and learner.losses[0] >= 0
        assert any(not torch.equal(initial[key], value) for key, value in learner.online.state_dict().items())
        assert all(torch.equal(initial[key], value) for key, value in learner.target.state_dict().items())
        assert all(not value.requires_grad and value.grad is None for value in learner.target.parameters())
        learner.remember(item)
        assert learner.updates == 2
        assert all(torch.equal(learner.online.state_dict()[key], value) for key, value in learner.target.state_dict().items())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in learner.online.parameters())
        assert learner.epsilon == settings.epsilon_start
        learner.decision_steps = settings.epsilon_decay_steps * 2
        assert learner.epsilon == pytest.approx(settings.epsilon_end)
