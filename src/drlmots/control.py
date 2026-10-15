"""Detached, causal decision frames for live scheduler callbacks (RL contract 2)."""

from dataclasses import dataclass
import math

from drlmots.entities import ResourceTier, Task
from drlmots.fog_cloud import CompletionEstimate


@dataclass(frozen=True)
class QueuedWork:
    upload_remaining_ms: float
    service_ms: float
    demands: tuple[float, ...]


@dataclass(frozen=True)
class VMFrame:
    vm_id: str
    tier: ResourceTier
    reachable: bool
    running_remaining_ms: float
    waiting_capacity: int
    queued: tuple[QueuedWork, ...]
    capacities: tuple[float | None, ...]
    available: tuple[float | None, ...]
    upload_ms: float
    estimate: CompletionEstimate | None


@dataclass(frozen=True)
class DecisionFrame:
    task_id: str
    time_ms: float
    length_bytes: float
    data_size_kb: float
    resource_units: tuple[tuple[str, str], ...]
    demands: tuple[float, ...]
    vms: tuple[VMFrame, ...]
    arrivals_remaining: int

    @property
    def mask(self):
        return tuple(vm.estimate is not None for vm in self.vms)


def decision_frame(task, now, vms, units, queues, running, durations, ledger, estimates,
                   *, remaining, reachable=None, ready=None, uploads=None):
    names = tuple(sorted(units))
    estimates = {item.vm_id: item for item in estimates}
    reachable = {vm.vm_id for vm in vms} if reachable is None else set(reachable)
    ready, uploads = {} if ready is None else ready, {} if uploads is None else uploads
    snapshots = []
    for vm in sorted(vms, key=lambda item: item.vm_id):
        capacities, available = vm.resource_capacities or {}, ledger.available(vm.vm_id) or {}
        queued = tuple(QueuedWork(max(0.0, ready.get(item.task_id, now) - now), durations[item.task_id],
                                  tuple(float(item.required_resources.get(name, 0)) for name in names))
                       for item in queues[vm.vm_id])
        snapshots.append(VMFrame(
            vm.vm_id, vm.tier, vm.vm_id in reachable,
            running[vm.vm_id][2] - now if vm.vm_id in running else 0.0,
            vm.queue_capacity, queued, tuple(capacities.get(name) for name in names),
            tuple(available.get(name) for name in names), uploads.get(vm.vm_id, 0.0), estimates.get(vm.vm_id),
        ))
    return DecisionFrame(task.task_id, float(now), float(task.length), float(task.data_size_kb),
                         tuple((name, units[name]) for name in names),
                         tuple(float(task.required_resources.get(name, 0)) for name in names),
                         tuple(snapshots), remaining)


def zero_network_estimates(task, now, eligible, queues, running, durations):
    result = []
    for vm in sorted(eligible, key=lambda item: item.vm_id):
        tail = math.fsum((running[vm.vm_id][2] if vm.vm_id in running else now,
                          *(durations[item.task_id] for item in queues[vm.vm_id])))
        finish = math.fsum((tail, durations[task.task_id]))
        Task._validate_number("projected finish", finish, allow_zero=False)
        result.append(CompletionEstimate(vm.vm_id, vm.tier, 0.0, now, tail,
                                         durations[task.task_id], tail, finish, finish - now))
    return tuple(result)


def checked_action(controller, frame):
    vm_id = controller(frame)
    eligible = {vm.vm_id for vm in frame.vms if vm.estimate is not None}
    if not eligible:
        if vm_id is not None:
            raise ValueError("controller must return None for forced rejection")
    elif not isinstance(vm_id, str) or vm_id not in eligible:
        raise ValueError("controller must select an eligible VM; voluntary rejection is unsupported")
    return vm_id
