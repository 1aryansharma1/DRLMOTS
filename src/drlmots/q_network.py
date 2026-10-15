"""Context-bound CPU Q-network and legal selection for synthetic prototype A-025."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random

import torch
from torch import nn

from drlmots.normalization import (
    NormalizationContext, StateNormalizer, _canonical, _digest, _read_json, capture_batch,
)
from drlmots.reward import RewardConfig, load_reward_config
from drlmots.run_config import load_run_config
from drlmots.state import encode_state


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return value


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("artifact identifiers must be lowercase SHA-256 strings")


@dataclass(frozen=True)
class NetworkConfig:
    hidden_widths: tuple[int, ...] = (50, 50)
    seed: int = 42

    def __post_init__(self):
        widths = tuple(self.hidden_widths)
        if len(widths) != 2 or any(type(width) is not int or width <= 0 for width in widths):
            raise ValueError("hidden_widths must contain two positive integers")
        _seed(self.seed)
        object.__setattr__(self, "hidden_widths", widths)

    def manifest(self):
        return {"hidden_widths": list(self.hidden_widths), "seed": self.seed,
                "activation": "relu", "dtype": "float32", "device": "cpu",
                "initialization": "torch_linear_default"}


@dataclass(frozen=True)
class NetworkContext:
    normalization: NormalizationContext
    normalizer_sha256: str
    reward_sha256: str

    def __post_init__(self):
        if not isinstance(self.normalization, NormalizationContext):
            raise TypeError("network context requires NormalizationContext")
        if not self.normalization.schema.vm_ids:
            raise ValueError("Q-network requires at least one VM action")
        _sha(self.normalizer_sha256)
        _sha(self.reward_sha256)

    @classmethod
    def from_artifacts(cls, normalizer, reward):
        if not isinstance(normalizer, StateNormalizer) or not isinstance(reward, RewardConfig):
            raise TypeError("network context requires StateNormalizer and RewardConfig")
        return cls(normalizer.context, normalizer.manifest()["sha256"], reward.manifest()["sha256"])

    def manifest(self):
        return {"normalization": self.normalization.manifest(),
                "normalizer_sha256": self.normalizer_sha256, "reward_sha256": self.reward_sha256,
                "action_vm_ids": list(self.normalization.schema.vm_ids)}


class QNetwork(nn.Module):
    def __init__(self, context, config=NetworkConfig()):
        super().__init__()
        if not isinstance(context, NetworkContext) or not isinstance(config, NetworkConfig):
            raise TypeError("QNetwork requires NetworkContext and NetworkConfig")
        self.context, self.config = context, config
        width1, width2 = config.hidden_widths
        # Linear initializes with the global CPU generator; fork restores its exact state.
        # Explicit CPU float32 also avoids depending on caller default dtype/device.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.layers = nn.Sequential(
                nn.Linear(context.normalization.schema.dimension, width1, device="cpu", dtype=torch.float32),
                nn.ReLU(), nn.Linear(width1, width2, device="cpu", dtype=torch.float32),
                nn.ReLU(), nn.Linear(width2, len(context.normalization.schema.vm_ids),
                                     device="cpu", dtype=torch.float32),
            )

    def forward(self, states, *, context):
        if not isinstance(context, NetworkContext) or context != self.context:
            raise ValueError("incompatible network schema/topology/normalizer/reward context")
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a float32 CPU tensor")
        if states.dtype != torch.float32 or states.device.type != "cpu":
            raise ValueError("states must be a float32 CPU tensor")
        if states.ndim != 2 or states.shape[1] != context.normalization.schema.dimension:
            raise ValueError("states must have shape [batch, schema dimension]")
        if not torch.isfinite(states).all() or (states < 0).any() or (states > 1).any():
            raise ValueError("normalized states must be finite and in [0,1]")
        for parameter in self.parameters():
            if (parameter.device.type != "cpu" or parameter.dtype != torch.float32
                    or not torch.isfinite(parameter).all()):
                raise ValueError("network parameters must be finite float32 CPU tensors")
        result = self.layers(states)
        if not torch.isfinite(result).all():
            raise ValueError("Q outputs must be finite")
        return result

    def infer(self, observation, *, normalizer, reward):
        """One actual observation through its frozen scaler; no gradient or simulation."""
        context = NetworkContext.from_artifacts(normalizer, reward)
        raw = encode_state(context.normalization.schema, observation)
        scaled = normalizer.transform(context.normalization, raw)
        with torch.no_grad():
            return self(torch.tensor([scaled], dtype=torch.float32, device="cpu"),
                        context=context)[0].detach().clone()

    def artifact(self):
        weights = {}
        for name, value in self.state_dict().items():
            if value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError("checkpoint weights must be finite float32 CPU tensors")
            weights[name] = value.detach().tolist()
        payload = {"schema_version": 1, "kind": "synthetic_q_network",
                   "architecture": self.config.manifest(), "context": self.context.manifest(),
                   "runtime": {"torch": str(torch.__version__)}, "weights": weights}
        artifact = {**payload, "sha256": _digest(payload)}
        return artifact

    def save(self, path):
        artifact = self.artifact()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_canonical(artifact) + "\n", encoding="utf-8")
        return artifact["sha256"]

    @classmethod
    def load(cls, path, *, context):
        return cls.from_artifact(_read_json(Path(path).read_text(encoding="utf-8")), context=context)

    @classmethod
    def from_artifact(cls, artifact, *, context):
        if not isinstance(context, NetworkContext):
            raise TypeError("load requires NetworkContext")
        fields = {"schema_version", "kind", "architecture", "context", "runtime", "weights", "sha256"}
        if not isinstance(artifact, dict) or set(artifact) != fields:
            raise ValueError("invalid Q-network checkpoint fields")
        if (type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1
                or artifact["kind"] != "synthetic_q_network"):
            raise ValueError("unsupported Q-network checkpoint version/kind")
        if artifact["sha256"] != _digest({k: v for k, v in artifact.items() if k != "sha256"}):
            raise ValueError("Q-network checkpoint checksum mismatch")
        if _canonical(artifact["context"]) != _canonical(context.manifest()):
            raise ValueError("incompatible network schema/topology/normalizer/reward context")
        architecture = artifact["architecture"]
        if not isinstance(architecture, dict) or set(architecture) != set(NetworkConfig().manifest()):
            raise ValueError("invalid network architecture")
        config = NetworkConfig(architecture["hidden_widths"], architecture["seed"])
        if architecture != config.manifest():
            raise ValueError("unsupported network architecture")
        runtime = artifact["runtime"]
        if not isinstance(runtime, dict) or set(runtime) != {"torch"} or not isinstance(runtime["torch"], str):
            raise ValueError("invalid checkpoint runtime")
        result = cls(context, config)
        weights, expected = artifact["weights"], result.state_dict()
        if not isinstance(weights, dict) or set(weights) != set(expected):
            raise ValueError("checkpoint parameter names do not match architecture")
        tensors = {}
        for name, target in expected.items():
            values = weights[name]
            if (not isinstance(values, list) or len(values) != target.shape[0]
                    or (target.ndim == 2 and any(not isinstance(row, list) or len(row) != target.shape[1]
                                               for row in values))):
                raise ValueError("checkpoint parameter shape does not match architecture")
            flat = [v for row in values for v in row] if target.ndim == 2 else values
            if any(type(v) not in (int, float) or not math.isfinite(v)
                   or abs(v) > torch.finfo(torch.float32).max for v in flat):
                raise ValueError("checkpoint parameters must be finite float32 numbers")
            tensors[name] = torch.tensor(values, dtype=torch.float32, device="cpu")
        result.load_state_dict(tensors, strict=True)
        return result


class MaskedSelector:
    """Local seeded epsilon-greedy; no fake action for forced rejection."""

    def __init__(self, seed):
        self._rng = random.Random(_seed(seed))

    def select(self, q_values, mask, *, epsilon):
        if (type(epsilon) not in (int, float) or not math.isfinite(epsilon)
                or not 0 <= epsilon <= 1):
            raise ValueError("epsilon must be finite and in [0,1]")
        if not isinstance(q_values, torch.Tensor):
            raise TypeError("Q values must be a float32 CPU tensor")
        if (q_values.dtype != torch.float32 or q_values.device.type != "cpu"
                or q_values.ndim != 1 or not torch.isfinite(q_values).all()):
            raise ValueError("Q values must be a finite float32 CPU vector")
        mask = tuple(mask)
        if len(mask) != q_values.numel() or any(type(value) is not bool for value in mask):
            raise ValueError("mask must contain one boolean per action")
        legal = [index for index, allowed in enumerate(mask) if allowed]
        if not legal:
            return None
        if epsilon == 1 or (epsilon > 0 and self._rng.random() < epsilon):
            return self._rng.choice(legal)
        # max returns the first equal element; stable schema order defines ties.
        return max(legal, key=lambda index: q_values[index].item())


def main():
    parser = argparse.ArgumentParser(description="Untrained Q-network inference on held-out baseline observations")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-widths", type=int, nargs=2, default=(50, 50))
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()
    try:
        inputs = {p.resolve() for p in (args.config, args.normalizer, args.reward_config)}
        outputs = {p.resolve() for p in (args.checkpoint, args.output)}
        if len(outputs) != 2 or inputs & outputs:
            raise ValueError("checkpoint/report must be distinct and must not overwrite inputs")
        config = load_run_config(args.config)
        batch, report = capture_batch(config, split="test")
        normalizer = StateNormalizer.load(args.normalizer, context=batch.context)
        scaled = normalizer.transform_batch(batch)
        reward = load_reward_config(args.reward_config)
        context = NetworkContext.from_artifacts(normalizer, reward)
        network = QNetwork(context, NetworkConfig(args.hidden_widths, args.seed))
        selector = MaskedSelector(args.seed)
        states = torch.tensor(scaled, dtype=torch.float32, device="cpu").reshape(-1, batch.context.schema.dimension)
        with torch.no_grad():
            values = network(states, context=context)
        records = []
        # Validate epsilon even if there are no arrival records.
        selector.select(torch.zeros(len(context.normalization.schema.vm_ids)),
                        (False,) * len(context.normalization.schema.vm_ids), epsilon=args.epsilon)
        for decision, q in zip(report["decision_trace"]["decisions"], values):
            mask = tuple(decision["action_mask"])
            action = selector.select(q, mask, epsilon=args.epsilon)
            records.append({"task_id": decision["observation"]["task_id"], "q_values": q.tolist(),
                            "action_mask": list(mask), "selected_action": action,
                            "selected_vm_id": None if action is None else context.normalization.schema.vm_ids[action]})
        checkpoint_sha = network.save(args.checkpoint)
        restored = QNetwork.load(args.checkpoint, context=context)
        with torch.no_grad():
            if not torch.equal(values, restored(states, context=context)):
                raise ValueError("checkpoint reload changed Q values")
        output = {"scope": "Untrained inference on baseline observations; choices were not executed in the simulator.",
                  "context": context.manifest(), "architecture": network.config.manifest(),
                  "trace_source_policy": report["policy"], "run_configuration": report["run_configuration"],
                  "workload_sha256": report["workload"]["sha256"],
                  "checkpoint_sha256": checkpoint_sha, "checkpoint_reload_equal": True,
                  "epsilon": args.epsilon, "selection_seed": args.seed, "decisions": records}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_canonical(output) + "\n", encoding="utf-8")
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Wrote {len(records)} untrained decisions to {args.output}; checkpoint reload matched")


if __name__ == "__main__":
    main()
