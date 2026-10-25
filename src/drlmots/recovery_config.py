"""Versioned explicit node outages and bounded restart recovery (A-029).

This mode is independent of legacy schema-1/2/3 simulators and RL contracts.
A saved outage trace is exogenous: scheduling never draws failure randomness.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import yaml

from drlmots.entities import Task
from drlmots.normalization import _digest
from drlmots.run_config import _UniqueKeyLoader


def number(name, value, *, positive=False):
    Task._validate_number(name, value, allow_zero=not positive)
    return float(value)


@dataclass(frozen=True)
class Outage:
    node_id: str
    start_ms: float
    end_ms: float

    def __post_init__(self):
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("outage node ID must be nonempty")
        number("outage start", self.start_ms)
        number("outage end", self.end_ms, positive=True)
        if self.end_ms <= self.start_ms:
            raise ValueError("outage end must follow start")


@dataclass(frozen=True)
class RecoveryConfig:
    horizon_ms: float
    max_retries: int
    recovery_delay_ms: float
    recovery_wait_ms: float
    recovery_power_w: float
    idle_power_w: tuple[tuple[str, float], ...]
    outages: tuple[Outage, ...]

    def __post_init__(self):
        number("observation horizon", self.horizon_ms, positive=True)
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        for name in ("recovery_delay_ms", "recovery_wait_ms", "recovery_power_w"):
            number(name, getattr(self, name))
        idle = tuple(self.idle_power_w)
        if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in idle):
            raise ValueError("idle power requires VM/value pairs")
        if len({item[0] for item in idle}) != len(idle):
            raise ValueError("duplicate idle power VM ID")
        for vm, value in idle:
            if not isinstance(vm, str) or not vm.strip():
                raise ValueError("idle power VM ID must be nonempty")
            number("idle power", value)
        outages = tuple(self.outages)
        if any(not isinstance(item, Outage) for item in outages):
            raise TypeError("outages must contain Outage values")
        outages = tuple(sorted(outages, key=lambda item: (item.node_id, item.start_ms)))
        ends = {}
        for item in outages:
            if item.end_ms > self.horizon_ms:
                raise ValueError("outages must end within the observation horizon")
            if item.start_ms <= ends.get(item.node_id, -1):
                raise ValueError("outages for a node cannot overlap or touch")
            ends[item.node_id] = item.end_ms
        object.__setattr__(self, "idle_power_w", tuple(sorted(tuple(item) for item in idle)))
        object.__setattr__(self, "outages", outages)

    @classmethod
    def parse(cls, data):
        fields = {"schema_version", "kind", "horizon_ms", "max_retries", "recovery_delay_ms",
                  "recovery_wait_ms", "recovery_power_w", "idle_power_w", "outages"}
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("recovery config must contain exactly its documented fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["kind"] != "node_restart_recovery":
            raise ValueError("unsupported recovery schema/kind")
        if not isinstance(data["idle_power_w"], dict) or not isinstance(data["outages"], list):
            raise ValueError("invalid idle power/outage containers")
        items = []
        for item in data["outages"]:
            if not isinstance(item, dict) or set(item) != {"node_id", "start_ms", "end_ms"}:
                raise ValueError("invalid outage fields")
            items.append(Outage(**item))
        return cls(*(data[name] for name in ("horizon_ms", "max_retries", "recovery_delay_ms",
                    "recovery_wait_ms", "recovery_power_w")), tuple(data["idle_power_w"].items()), tuple(items))

    def manifest(self):
        payload = {"schema_version": 1, "kind": "node_restart_recovery", "horizon_ms": self.horizon_ms,
                   "max_retries": self.max_retries, "recovery_delay_ms": self.recovery_delay_ms,
                   "recovery_wait_ms": self.recovery_wait_ms, "recovery_power_w": self.recovery_power_w,
                   "idle_power_w": dict(self.idle_power_w), "outages": [asdict(item) for item in self.outages]}
        return payload

    def trace_manifest(self):
        payload = {"horizon_ms": self.horizon_ms, "outages": [asdict(item) for item in self.outages]}
        return {**payload, "sha256": _digest(payload)}


def load_recovery_config(path):
    return RecoveryConfig.parse(yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader))


def node_membership(topology):
    topology.validate()
    return {vm: prefix + identifier for prefix, nodes, field in (
        ("fog:", topology.fog_nodes, "fog_node_id"), ("cloud:", topology.cloud_datacenters, "datacenter_id"))
        for node in nodes for identifier in (getattr(node, field),) for vm in node.vm_ids}


def sample_outages(*, horizon_ms, seed, node_rates_per_s, repair_ms):
    """Optional alternating up/down process, materialized before any policy runs.

    Each node has its own seed-derived stream; adding another node never changes
    an existing node's trace. No independence across real-world nodes is claimed.
    """
    number("horizon", horizon_ms, positive=True)
    number("repair duration", repair_ms, positive=True)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("failure seed must be an integer in [0,2**63)")
    if not isinstance(node_rates_per_s, dict):
        raise ValueError("node rates must be a mapping")
    result = []
    for node, rate in sorted(node_rates_per_s.items()):
        if not isinstance(node, str) or not node.strip():
            raise ValueError("failure domain must be nonempty")
        number("failure rate", rate)
        if not rate:
            continue
        rng = random.Random(int(_digest({"seed": seed, "node": node}), 16))
        time = 0.0
        while True:
            start = time + rng.expovariate(rate) * 1000
            if start >= horizon_ms:
                break
            end = min(start + repair_ms, horizon_ms)
            if start <= time or end <= start:
                raise ValueError("outage sampling lost timestamp precision")
            result.append(Outage(node, start, end))
            if len(result) > 100000:
                raise ValueError("too many outages; reduce failure rates or horizon")
            time = end
    return tuple(sorted(result, key=lambda item: (item.node_id, item.start_ms)))
