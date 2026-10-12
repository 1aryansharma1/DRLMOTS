"""Explicit per-run units and per-VM reservation accounting, without host inference."""

from collections.abc import Mapping
from dataclasses import dataclass

from drlmots.entities import Task, VM


def validate_resource_units(value) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("resource_units must be a mapping")
    result = {}
    for name, unit in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("resource_units names must be non-empty strings")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"resource_units[{name!r}] must be a non-empty unit string")
        result[name] = unit
    return result


def require_declared_units(names, units) -> None:
    missing = set(names) - set(units)
    if missing:
        raise ValueError(f"resource_units must explicitly declare: {sorted(missing)}")


def fits_resources(task: Task, vm: VM) -> bool:
    capacities = vm.resource_capacities or {}
    return all(name in capacities and amount <= capacities[name]
               for name, amount in task.required_resources.items())


@dataclass(frozen=True)
class ResourceEvent:
    vm_id: str
    task_id: str
    time_ms: float
    action: str
    amounts: dict[str, float]
    available_after: dict[str, float]


class ResourceLedger:
    """One running reservation per VM; queued tasks never consume capacity."""

    def __init__(self, vms):
        self.capacities = {vm.vm_id: None if vm.resource_capacities is None
                           else dict(vm.resource_capacities) for vm in vms}
        self.reservations = {}
        self.events = []

    def available(self, vm_id):
        capacity = self.capacities[vm_id]
        if capacity is None:
            return None
        allocated = self.reservations.get(vm_id, (None, {}))[1]
        return {name: amount - allocated.get(name, 0) for name, amount in capacity.items()}

    def reserve(self, vm_id: str, task: Task, time_ms: float) -> None:
        if vm_id in self.reservations:
            raise RuntimeError(f"VM {vm_id!r} already has a running reservation")
        available = self.available(vm_id) or {}
        amounts = dict(task.required_resources)
        if any(name not in available or amount > available[name] for name, amount in amounts.items()):
            raise RuntimeError("resource reservation would exceed capacity")
        self.reservations[vm_id] = (task.task_id, amounts)
        if amounts:
            self.events.append(ResourceEvent(
                vm_id, task.task_id, time_ms, "reserve", dict(amounts), self.available(vm_id)
            ))

    def release(self, vm_id: str, task: Task, time_ms: float) -> None:
        reservation = self.reservations.get(vm_id)
        if reservation is None or reservation[0] != task.task_id:
            raise RuntimeError("resource release does not match the running task")
        _, amounts = self.reservations.pop(vm_id)
        if amounts:
            self.events.append(ResourceEvent(
                vm_id, task.task_id, time_ms, "release", dict(amounts), self.available(vm_id)
            ))

    def final_balances(self) -> dict:
        if self.reservations:
            raise RuntimeError("simulation ended with unreleased resource reservations")
        return {vm_id: self.available(vm_id) for vm_id in self.capacities}
