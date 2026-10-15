"""Runnable synthetic configurations, separate from unresolved paper settings."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import yaml

from drlmots.config import ConfigError
from drlmots.topology import Topology, build_topology
from drlmots.workload import SyntheticWorkloadSettings
from drlmots.resources import require_declared_units, validate_resource_units
from drlmots.fog_cloud import FogCloudSettings, parse_placement_settings


def _exact_keys(value, keys, name):
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    if set(value) != set(keys):
        raise ConfigError(f"{name} must contain exactly: " + ", ".join(sorted(keys)))


@dataclass(frozen=True)
class RunConfig:
    topology: Topology
    workload: SyntheticWorkloadSettings
    resource_units: dict[str, str] = field(default_factory=dict)
    schema_version: int = 1
    placement: FogCloudSettings | None = None

    def __post_init__(self):
        object.__setattr__(self, "resource_units", validate_resource_units(self.resource_units))
        self._validate_schema()

    def _validate_schema(self):
        if type(self.schema_version) is not int or self.schema_version not in (1, 2, 3):
            raise ConfigError("run schema_version must be integer 1, 2 or 3")
        if self.schema_version == 3:
            if not isinstance(self.placement, FogCloudSettings):
                raise ConfigError("run schema 3 requires placement/network settings")
            self.placement.candidates(self.topology)
        elif self.placement is not None:
            raise ConfigError("placement/network settings require run schema 3")
        self.topology.validate()
        if self.schema_version == 1 and (
            self.resource_units or self.workload.required_resources
            or any(vm.resource_capacities is not None for vm in self.topology.vms)
        ):
            raise ConfigError("resource-aware settings require run schema 2")
        require_declared_units(self.workload.required_resources, self.resource_units)
        for vm in self.topology.vms:
            require_declared_units(vm.resource_capacities or {}, self.resource_units)

    def manifest(self) -> dict:
        """Record effective settings, excluding local file paths/formatting."""
        self._validate_schema()
        topology = asdict(self.topology)
        for vm in topology["vms"]:
            del vm["queue"]  # Runnable config has no preloaded task queues.
            if self.schema_version == 1:
                del vm["resource_capacities"]
        payload = {
            "schema_version": self.schema_version,
            "kind": "synthetic_baseline",
            "topology": topology,
            "workload": {
                "generator": "uniform_interarrival",
                "length_unit": "bytes",
                "arrival_time_unit": "milliseconds",
                **self.workload.manifest_settings(),
            },
        }
        if self.schema_version >= 2:
            payload["resource_units"] = dict(self.resource_units)
            payload["workload"]["required_resources"] = dict(self.workload.required_resources)
        if self.placement is not None:
            payload["placement"] = self.placement.manifest()
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
    if not isinstance(data, dict):
        raise ConfigError("run configuration must be a mapping")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2, 3):
        raise ConfigError("run schema_version must be integer 1, 2 or 3")
    resource_aware = schema_version >= 2
    root_keys = {"schema_version", "kind", "topology", "workload"}
    if resource_aware:
        root_keys.add("resource_units")
    if schema_version == 3:
        root_keys.add("placement")
    _exact_keys(data, root_keys, "run configuration")
    if data["kind"] != "synthetic_baseline":
        raise ConfigError("run kind must be synthetic_baseline")
    workload = data["workload"]
    setting_keys = {
        "task_count", "seed", "interarrival_ms_range", "length_bytes_range", "data_size_kb_range",
    }
    if resource_aware:
        setting_keys.add("required_resources")
    _exact_keys(workload, setting_keys | {"generator", "length_unit", "arrival_time_unit"}, "run workload")
    if workload["generator"] != "uniform_interarrival":
        raise ConfigError("run workload.generator must be uniform_interarrival")
    if workload["length_unit"] != "bytes" or workload["arrival_time_unit"] != "milliseconds":
        raise ConfigError("run workload requires bytes and milliseconds explicitly")
    try:
        topology = build_topology(data["topology"], resource_aware=resource_aware)
        settings = SyntheticWorkloadSettings(**{key: workload[key] for key in setting_keys})
        units = validate_resource_units(data["resource_units"] if resource_aware else {})
        require_declared_units(settings.required_resources, units)
        for vm in topology.vms:
            require_declared_units(vm.resource_capacities or {}, units)
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
    if settings.task_count and not topology.vms:
        raise ConfigError("non-empty workload requires at least one VM")
    try:
        placement = parse_placement_settings(data["placement"]) if schema_version == 3 else None
        return RunConfig(topology, settings, units, schema_version, placement)
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error


def load_run_config(path: str | Path) -> RunConfig:
    try:
        with Path(path).open("r", encoding="utf-8-sig") as file:
            data = yaml.load(file, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid run YAML: {error}") from error
    return parse_run_config(data)
