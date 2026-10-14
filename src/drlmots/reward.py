"""Causal synthetic placement reward; not the paper's multi-objective Eq. (25)."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import yaml

from drlmots.actions import ActionMap, action_mask
from drlmots.entities import Task
from drlmots.run_config import _UniqueKeyLoader
from drlmots.state import DecisionObservation
from drlmots.workload import processing_time_ms


ALGORITHM = "projected_latency_v1"


@dataclass(frozen=True)
class RewardConfig:
    latency_scale_ms: float
    rejection_penalty: float

    def __post_init__(self):
        Task._validate_number("latency_scale_ms", self.latency_scale_ms, allow_zero=False)
        Task._validate_number("rejection_penalty", self.rejection_penalty, allow_zero=False)

    def manifest(self):
        payload = {"schema_version": 1, "kind": "synthetic_reward", "algorithm": ALGORITHM,
                   "latency_scale_ms": self.latency_scale_ms, "rejection_penalty": self.rejection_penalty}
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def parse_reward_config(data):
    fields = {"schema_version", "kind", "algorithm", "latency_scale_ms", "rejection_penalty"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("reward config must contain exactly: " + ", ".join(sorted(fields)))
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["kind"] != "synthetic_reward" or data["algorithm"] != ALGORITHM):
        raise ValueError("unsupported reward schema/kind/algorithm")
    return RewardConfig(data["latency_scale_ms"], data["rejection_penalty"])


def load_reward_config(path):
    try:
        data = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid reward YAML: {error}") from error
    return parse_reward_config(data)


@dataclass(frozen=True)
class RewardResult:
    task_id: str
    available_time_ms: float
    selected_action: int | None
    selected_vm_id: str | None
    projected_wait_ms: float | None
    service_ms: float | None
    projected_completion_ms: float | None
    value: float


def score_action(observation: DecisionObservation, selected_action: int | None, config: RewardConfig):
    """Use only decision-time observations, without eventual task outcomes."""
    if not isinstance(observation, DecisionObservation) or not isinstance(config, RewardConfig):
        raise TypeError("score_action requires DecisionObservation and RewardConfig")
    mask = action_mask(observation)
    if selected_action is None:
        if any(mask):
            raise ValueError("null action is permitted only for a forced all-false-mask rejection")
        return RewardResult(observation.task_id, observation.time_ms, None, None, None, None, None,
                            -float(config.rejection_penalty))
    actions = ActionMap(tuple(vm.vm_id for vm in observation.vms))
    vm_id = actions.vm_id(selected_action)
    if not mask[selected_action]:
        raise ValueError("cannot reward a masked VM action")
    vm = observation.vms[selected_action]
    service = processing_time_ms(observation.length_bytes, length_unit="bytes")
    try:
        waiting = math.fsum((vm.remaining_service_ms, vm.queued_service_ms))
        latency = math.fsum((waiting, service))
    except OverflowError as error:
        raise ValueError("projected latency must be representable and finite") from error
    Task._validate_number("projected latency", latency, allow_zero=False)
    scale = config.latency_scale_ms
    if latency >= scale:
        ratio = scale / latency
        value = ratio / (1 + ratio)
    else:
        value = 1 / (1 + latency / scale)
    return RewardResult(observation.task_id, observation.time_ms, selected_action, vm_id,
                        waiting, service, latency, value)
