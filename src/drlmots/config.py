
"""Loading and validation for DRLMOTS YAML configuration."""

from pathlib import Path
from typing import Any
import math

import yaml


class ConfigError(ValueError):
    """Raised when a DRLMOTS configuration is invalid."""


def _require_mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require_keys(mapping: dict, keys: set[str], name: str) -> None:
    missing = keys - mapping.keys()
    if missing:
        raise ConfigError(f"{name} is missing keys: {sorted(missing)}")


def _nullable_positive_int(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer or null")


def validate_config(config: Any) -> dict:
    """Validate required configuration sections and basic field types."""
    config = _require_mapping(config, "configuration")
    _require_keys(
        config,
        {"schema_version", "seed", "topology", "workload", "dqn", "unresolved"},
        "configuration",
    )

    if (
        isinstance(config["schema_version"], bool)
        or not isinstance(config["schema_version"], int)
        or config["schema_version"] != 1
    ):
        raise ConfigError("schema_version must be 1")

    seed = config["seed"]
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ConfigError("seed must be a non-negative integer or null")

    topology = _require_mapping(config["topology"], "topology")
    topology_keys = {
        "access_point_count",
        "end_node_count",
        "fog_node_count",
        "cloud_datacenter_count",
        "vms_per_fog_node",
        "cloud_vm_count",
    }
    _require_keys(topology, topology_keys, "topology")
    for key in topology_keys:
        _nullable_positive_int(topology[key], f"topology.{key}")

    if "paper_experiment" in config:
        experiment = _require_mapping(config["paper_experiment"], "paper_experiment")
        _require_keys(experiment, {
            "vm_count", "host_storage_tb", "host_memory_gb",
            "host_bandwidth_reported", "vm_bandwidth_reported",
        }, "paper_experiment")
        if experiment["vm_count"] is not None:
            validate_paper_vm_count(experiment["vm_count"])
        for key in (
            "host_storage_tb", "host_memory_gb", "host_bandwidth_reported", "vm_bandwidth_reported"
        ):
            value = experiment[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ConfigError(f"paper_experiment.{key} must be finite and positive")

    workload = _require_mapping(config["workload"], "workload")
    _require_keys(
        workload,
        {
            "task_count",
            "task_size_kb_range",
            "task_length_unit",
            "arrival_distribution",
            "data_source",
        },
        "workload",
    )
    _nullable_positive_int(workload["task_count"], "workload.task_count")

    size_range = workload["task_size_kb_range"]
    if (
        not isinstance(size_range, list)
        or len(size_range) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in size_range
        )
        or size_range[0] > size_range[1]
    ):
        raise ConfigError(
            "workload.task_size_kb_range must contain two positive "
            "numbers in ascending order"
        )

    for key in ("task_length_unit", "arrival_distribution"):
        if workload[key] is not None and not isinstance(workload[key], str):
            raise ConfigError(f"workload.{key} must be a string or null")

    if not isinstance(workload["data_source"], str) or not workload["data_source"]:
        raise ConfigError("workload.data_source must be a non-empty string")

    dqn = _require_mapping(config["dqn"], "dqn")
    dqn_keys = {
        "gamma",
        "epsilon_start",
        "epsilon_min",
        "epsilon_decay",
        "learning_rate",
        "replay_capacity",
        "batch_size",
        "target_sync_interval",
        "episodes",
    }
    _require_keys(dqn, dqn_keys, "dqn")

    for key in ("gamma", "epsilon_min", "epsilon_decay"):
        value = dqn[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"dqn.{key} must be numeric")
        if not 0 <= value <= 1:
            raise ConfigError(f"dqn.{key} must be between 0 and 1")

    if dqn["epsilon_start"] is not None:
        value = dqn["epsilon_start"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError("dqn.epsilon_start must be numeric or null")
        if not 0 <= value <= 1:
            raise ConfigError("dqn.epsilon_start must be between 0 and 1")
        if value < dqn["epsilon_min"]:
            raise ConfigError("dqn.epsilon_start must be at least epsilon_min")

    rate = dqn["learning_rate"]
    if rate is not None and (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise ConfigError("dqn.learning_rate must be finite and positive or null")

    for key in (
        "replay_capacity",
        "batch_size",
        "target_sync_interval",
        "episodes",
    ):
        _nullable_positive_int(dqn[key], f"dqn.{key}")

    if (
        dqn["batch_size"] is not None
        and dqn["replay_capacity"] is not None
        and dqn["batch_size"] > dqn["replay_capacity"]
    ):
        raise ConfigError("dqn.batch_size cannot exceed replay_capacity")

    if not isinstance(config["unresolved"], list) or any(
        not isinstance(item, str) for item in config["unresolved"]
    ):
        raise ConfigError("unresolved must be a list of strings")

    return config


def validate_paper_vm_count(value: int) -> int:
    """Validate the recorded experiment range, separately from tiny fixtures."""
    _nullable_positive_int(value, "paper VM count")
    if value is None or not 20 <= value <= 200:
        raise ConfigError("paper VM count must be between 20 and 200")
    return value


def load_config(path: str | Path) -> dict:
    """Load YAML and validate its structure."""
    with Path(path).open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    return validate_config(config)
