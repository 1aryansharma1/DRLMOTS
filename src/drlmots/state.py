"""Provisional synthetic arrival-time observation and raw encoding (A-022)."""

from dataclasses import dataclass
import hashlib
import json

from drlmots.entities import ResourceTier, Task
from drlmots.resources import require_declared_units, validate_resource_units


def _ids(values):
    values = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("VM IDs must be nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError("duplicate VM ID in schema")
    return tuple(sorted(values))


@dataclass(frozen=True)
class VMObservation:
    vm_id: str
    tier: ResourceTier
    running: bool
    remaining_service_ms: float
    waiting_count: int
    waiting_capacity: int
    queued_service_ms: float
    capacities: tuple[float | None, ...]
    available: tuple[float | None, ...]

    def __post_init__(self):
        _ids((self.vm_id,))
        if not isinstance(self.tier, ResourceTier) or type(self.running) is not bool:
            raise TypeError("tier and running must be ResourceTier and bool")
        if type(self.waiting_count) is not int or type(self.waiting_capacity) is not int:
            raise TypeError("waiting count/capacity must be integers")
        if self.waiting_capacity <= 0 or not 0 <= self.waiting_count <= self.waiting_capacity:
            raise ValueError("invalid waiting count/capacity")
        Task._validate_number("remaining service", self.remaining_service_ms, allow_zero=True)
        Task._validate_number("queued service", self.queued_service_ms, allow_zero=True)
        if self.running != (self.remaining_service_ms > 0):
            raise ValueError("running flag must agree with remaining service")
        if (self.waiting_count > 0) != (self.queued_service_ms > 0):
            raise ValueError("waiting count must agree with queued service")
        if not self.running and self.waiting_count:
            raise ValueError("idle decision snapshot cannot have waiting tasks")
        object.__setattr__(self, "capacities", tuple(self.capacities))
        object.__setattr__(self, "available", tuple(self.available))
        if len(self.capacities) != len(self.available):
            raise ValueError("capacity/available columns must match")
        for capacity, available in zip(self.capacities, self.available):
            if (capacity is None) != (available is None):
                raise ValueError("capacity/available known flags must match")
            if capacity is not None:
                Task._validate_number("capacity", capacity, allow_zero=False)
                Task._validate_number("available", available, allow_zero=True)
                if available > capacity:
                    raise ValueError("available exceeds capacity")
                if not self.running and available != capacity:
                    raise ValueError("idle resources must be fully available")


@dataclass(frozen=True)
class DecisionObservation:
    task_id: str
    time_ms: float
    length_bytes: float
    data_size_kb: float
    resource_units: tuple[tuple[str, str], ...]
    demands: tuple[float, ...]
    vms: tuple[VMObservation, ...]

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task ID must be nonempty")
        Task._validate_number("decision time", self.time_ms, allow_zero=True)
        Task._validate_number("length", self.length_bytes, allow_zero=False)
        Task._validate_number("data size", self.data_size_kb, allow_zero=False)
        object.__setattr__(self, "demands", tuple(self.demands))
        object.__setattr__(self, "vms", tuple(self.vms))
        items = tuple(tuple(item) for item in self.resource_units)
        units = validate_resource_units(dict(items))
        if items != tuple(sorted(units.items())) or len(items) != len(self.demands):
            raise ValueError("snapshot resource units must be unique, sorted and match demands")
        object.__setattr__(self, "resource_units", items)
        for demand in self.demands:
            Task._validate_number("demand", demand, allow_zero=True)
        if any(not isinstance(vm, VMObservation) for vm in self.vms):
            raise TypeError("snapshot VMs must be VMObservation values")
        if tuple(vm.vm_id for vm in self.vms) != _ids(vm.vm_id for vm in self.vms):
            raise ValueError("snapshot VMs must follow sorted action order")
        if any(len(vm.capacities) != len(self.demands) for vm in self.vms):
            raise ValueError("VM resource columns must match task demands")


@dataclass(frozen=True)
class StateSchema:
    vm_ids: tuple[str, ...]
    resource_units: tuple[tuple[str, str], ...]

    def __post_init__(self):
        object.__setattr__(self, "vm_ids", _ids(self.vm_ids))
        items = tuple(tuple(item) for item in self.resource_units)
        units = validate_resource_units(dict(items))
        if len(items) != len(units):
            raise ValueError("duplicate resource name in schema")
        object.__setattr__(self, "resource_units", tuple(sorted(units.items())))

    @classmethod
    def from_topology(cls, topology, resource_units=None):
        topology.validate()
        units = validate_resource_units({} if resource_units is None else resource_units)
        for vm in topology.vms:
            require_declared_units(vm.resource_capacities or {}, units)
        return cls(tuple(vm.vm_id for vm in topology.vms), tuple(units.items()))

    @property
    def resource_names(self):
        return tuple(name for name, _ in self.resource_units)

    @property
    def features(self):
        fields = [("task.decision_time", "ms"), ("task.length", "bytes"), ("task.data_size", "kB")]
        fields.extend((f"task.demand.{name}", unit) for name, unit in self.resource_units)
        for index in range(len(self.vm_ids)):
            prefix = f"vm[{index}]"
            fields.extend((f"{prefix}.{name}", unit) for name, unit in (
                ("is_cloud", "binary"), ("running", "binary"), ("remaining_service", "ms"),
                ("waiting_count", "tasks"), ("waiting_capacity", "tasks"), ("queued_service", "ms"),
            ))
            for name, unit in self.resource_units:
                fields.extend(((f"{prefix}.capacity_known.{name}", "binary"),
                               (f"{prefix}.capacity.{name}", unit),
                               (f"{prefix}.available_known.{name}", "binary"),
                               (f"{prefix}.available.{name}", unit)))
        return tuple(fields)

    @property
    def dimension(self):
        return 3 + len(self.resource_units) + len(self.vm_ids) * (6 + 4 * len(self.resource_units))

    def manifest(self):
        payload = {
            "schema_version": 1, "kind": "synthetic_arrival_observation",
            "vm_action_ids": list(self.vm_ids), "resource_units": dict(self.resource_units),
            "features": [{"name": name, "unit": unit} for name, unit in self.features],
            "dimension": self.dimension, "normalization": "none; raw finite Python floats",
            "boundary": "after same-time completions/queued starts, before each arrival placement",
        }
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def encode_state(schema: StateSchema, observation: DecisionObservation) -> tuple[float, ...]:
    """Pure fixed-shape encoding; unknown amounts carry a separate known flag."""
    if tuple(vm.vm_id for vm in observation.vms) != schema.vm_ids:
        raise ValueError("snapshot VM order does not match schema")
    if observation.resource_units != schema.resource_units:
        raise ValueError("snapshot resource columns do not match schema")
    values = [observation.time_ms, observation.length_bytes, observation.data_size_kb, *observation.demands]
    for vm in observation.vms:
        values.extend((int(vm.tier is ResourceTier.CLOUD), int(vm.running), vm.remaining_service_ms,
                       vm.waiting_count, vm.waiting_capacity, vm.queued_service_ms))
        for capacity, available in zip(vm.capacities, vm.available):
            values.extend((int(capacity is not None), capacity if capacity is not None else 0,
                           int(available is not None), available if available is not None else 0))
    for value in values:
        Task._validate_number("state feature", value, allow_zero=True)
    return tuple(float(value) for value in values)
