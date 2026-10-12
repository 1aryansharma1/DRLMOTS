"""Runnable synthetic configurations, separate from unresolved paper settings."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import yaml

from drlmots.config import ConfigError
from drlmots.topology import Topology, build_topology
from drlmots.workload import SyntheticWorkloadSettings


def _exact_keys(value, keys, name):
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    if set(value) != set(keys):
        raise ConfigError(f"{name} must contain exactly: " + ", ".join(sorted(keys)))


@dataclass(frozen=True)
class RunConfig:
    topology: Topology
    workload: SyntheticWorkloadSettings

    def manifest(self) -> dict:
        """Record effective settings, excluding local file paths/formatting."""
        topology = asdict(self.topology)
        for vm in topology["vms"]:
            del vm["queue"]  # Runnable config has no preloaded task queues.
        payload = {
            "schema_version": 1,
            "kind": "synthetic_baseline",
            "topology": topology,
            "workload": {
                "generator": "uniform_interarrival",
                "length_unit": "bytes",
                "arrival_time_unit": "milliseconds",
                **asdict(self.workload),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        return {**json.loads(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("run YAML mapping keys must be strings")
            if key in seen:
                raise ConfigError(f"duplicate run YAML key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_run_config(data) -> RunConfig:
    _exact_keys(data, {"schema_version", "kind", "topology", "workload"}, "run configuration")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ConfigError("run schema_version must be integer 1")
    if data["kind"] != "synthetic_baseline":
        raise ConfigError("run kind must be synthetic_baseline")
    workload = data["workload"]
    setting_keys = {
        "task_count", "seed", "interarrival_ms_range", "length_bytes_range", "data_size_kb_range",
    }
    _exact_keys(workload, setting_keys | {"generator", "length_unit", "arrival_time_unit"}, "run workload")
    if workload["generator"] != "uniform_interarrival":
        raise ConfigError("run workload.generator must be uniform_interarrival")
    if workload["length_unit"] != "bytes" or workload["arrival_time_unit"] != "milliseconds":
        raise ConfigError("run workload requires bytes and milliseconds explicitly")
    try:
        topology = build_topology(data["topology"])
        settings = SyntheticWorkloadSettings(**{key: workload[key] for key in setting_keys})
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
    if settings.task_count and not topology.vms:
        raise ConfigError("non-empty workload requires at least one VM")
    return RunConfig(topology, settings)


def load_run_config(path: str | Path) -> RunConfig:
    try:
        with Path(path).open("r", encoding="utf-8-sig") as file:
            data = yaml.load(file, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid run YAML: {error}") from error
    return parse_run_config(data)
