"""Explicit uploads and fog-preferred completion-time scheduling (A-026)."""

from collections import deque
from dataclasses import asdict, dataclass, replace
import math

import simpy

from drlmots.baselines import BASELINE_POLICIES, FOG_FIRST_POLICY, choose_vm
from drlmots.entities import ResourceTier, Task, TaskStatus
from drlmots.resources import ResourceLedger, fits_resources, require_declared_units, validate_resource_units
from drlmots.workload import processing_time_ms


def _sum(*values):
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError("network timing must be finite and representable") from error
    Task._validate_number("network timing", result, allow_zero=True)
    return result


def _after(time, duration):
    result = _sum(time, duration)
    if duration > 0 and result <= time:
        raise ValueError("positive network/service duration must advance its timestamp")
    return result


@dataclass(frozen=True)
class TransmissionLink:
    vm_id: str
    bandwidth_mbps: float
    propagation_ms: float

    def __post_init__(self):
        if not isinstance(self.vm_id, str) or not self.vm_id.strip():
            raise ValueError("link VM ID must be nonempty")
        Task._validate_number("bandwidth_mbps", self.bandwidth_mbps, allow_zero=False)
        Task._validate_number("propagation_ms", self.propagation_ms, allow_zero=True)

    def upload_ms(self, data_size_kb):
        Task._validate_number("data_size_kb", data_size_kb, allow_zero=False)
        transmission = 8 * (data_size_kb / self.bandwidth_mbps)
        Task._validate_number("serialization time", transmission, allow_zero=False)
        return _sum(self.propagation_ms, transmission)


@dataclass(frozen=True)
class FogCloudSettings:
    source_end_node_id: str
    links: tuple[TransmissionLink, ...]

    def __post_init__(self):
        if not isinstance(self.source_end_node_id, str) or not self.source_end_node_id.strip():
            raise ValueError("source_end_node_id must be nonempty")
        links = tuple(self.links)
        if any(not isinstance(link, TransmissionLink) for link in links):
            raise TypeError("links must contain TransmissionLink values")
        if len({link.vm_id for link in links}) != len(links):
            raise ValueError("duplicate transmission link VM ID")
        object.__setattr__(self, "links", tuple(sorted(links, key=lambda link: link.vm_id)))

    def candidates(self, topology):
        topology.validate()
        origin = next((end for end in topology.end_nodes if end.end_node_id == self.source_end_node_id), None)
        if origin is None:
            raise ValueError("source end node does not exist in topology")
        local = {vm_id for fog in topology.fog_nodes if fog.fog_node_id in origin.fog_node_ids
                 for vm_id in fog.vm_ids}
        candidates = tuple(sorted((vm for vm in topology.vms
                                   if vm.tier is ResourceTier.CLOUD or vm.vm_id in local),
                                  key=lambda vm: vm.vm_id))
        if {link.vm_id for link in self.links} != {vm.vm_id for vm in candidates}:
            raise ValueError("transmission links must cover exactly the reachable fog/cloud VM IDs")
        return candidates

    def manifest(self):
        return {"policy": FOG_FIRST_POLICY, "source_end_node_id": self.source_end_node_id,
                "data_size_unit": "decimal_kB", "upload_model": "parallel_no_contention",
                "links": [asdict(link) for link in self.links]}


def parse_placement_settings(data):
    fields = {"policy", "source_end_node_id", "data_size_unit", "upload_model", "links"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("placement must contain exactly: " + ", ".join(sorted(fields)))
    if (data["policy"] != FOG_FIRST_POLICY or data["data_size_unit"] != "decimal_kB"
            or data["upload_model"] != "parallel_no_contention"):
        raise ValueError("unsupported placement policy, data unit or upload model")
    if not isinstance(data["links"], list):
        raise ValueError("placement links must be a list")
    if any(not isinstance(link, dict) or set(link) != {"vm_id", "bandwidth_mbps", "propagation_ms"}
           for link in data["links"]):
        raise ValueError("each link requires vm_id, bandwidth_mbps and propagation_ms")
    return FogCloudSettings(data["source_end_node_id"], tuple(TransmissionLink(**link) for link in data["links"]))


@dataclass(frozen=True)
class CompletionEstimate:
    vm_id: str
    tier: ResourceTier
    upload_ms: float
    upload_ready_time_ms: float
    queue_tail_finish_time_ms: float
    service_ms: float
    predicted_start_time_ms: float
    predicted_finish_time_ms: float
    completion_duration_ms: float


def choose_completion_vm(estimates):
    """Earliest finish within tiers; fog wins an exact cross-tier tie."""
    if not estimates:
        raise ValueError("cannot choose from empty completion estimates")
    return min(estimates, key=lambda item: (item.predicted_finish_time_ms,
                                          item.tier is ResourceTier.CLOUD, item.vm_id))


@dataclass(frozen=True)
class PlacementDecision:
    task_id: str
    time_ms: float
    selected_vm_id: str | None
    reason: str
    candidates: tuple[CompletionEstimate, ...]


def simulate_network(topology, tasks, *, task_length_unit, resource_units, policy, placement):
    # Local import avoids a cycle while returning the shared result/outcome contract.
    from drlmots.environment import SimulationResult, TaskOutcome

    if not isinstance(placement, FogCloudSettings):
        raise TypeError("network simulation requires FogCloudSettings")
    if policy not in (*BASELINE_POLICIES, FOG_FIRST_POLICY):
        raise ValueError("unknown network scheduling policy")
    if task_length_unit != "bytes":
        raise ValueError("task_length_unit must explicitly be 'bytes' for this model")
    candidates = placement.candidates(topology)
    links = {link.vm_id: link for link in placement.links}
    units = validate_resource_units({} if resource_units is None else resource_units)
    for vm in topology.vms:
        require_declared_units(vm.resource_capacities or {}, units)
        if vm.queue_capacity is None or vm.queue:
            raise ValueError("network simulation requires explicit queue capacity and empty initial queues")
    copied, durations, ids = [], {}, set()
    for original in tasks:
        if not isinstance(original, Task):
            raise TypeError("every workload item must be a Task")
        task = replace(original)
        if task.task_id in ids or task.status is not TaskStatus.CREATED:
            raise ValueError("network tasks require unique IDs and CREATED status")
        require_declared_units(task.required_resources, units)
        ids.add(task.task_id)
        durations[task.task_id] = processing_time_ms(task.length, length_unit=task_length_unit)
        copied.append(task)
    ledger = ResourceLedger(topology.vms)
    env = simpy.Environment()
    incoming = sorted(copied, key=lambda task: task.arrival_time)
    queues = {vm.vm_id: deque() for vm in topology.vms}
    running, ready, outcomes, decisions, transfers = {}, {}, {}, [], []

    def start(vm_id, task):
        finish = _after(env.now, durations[task.task_id])
        task.transition_to(TaskStatus.RUNNING)
        ledger.reserve(vm_id, task, env.now)
        running[vm_id] = (task, env.now, finish)

    def estimate(vm, task):
        tail = running[vm.vm_id][2] if vm.vm_id in running else env.now
        for waiting in queues[vm.vm_id]:
            tail = _after(max(tail, ready[waiting.task_id]), durations[waiting.task_id])
        upload = links[vm.vm_id].upload_ms(task.data_size_kb)
        upload_ready = _after(env.now, upload)
        start_time = max(tail, upload_ready)
        finish = _after(start_time, durations[task.task_id])
        return CompletionEstimate(vm.vm_id, vm.tier, upload, upload_ready, tail,
                                  durations[task.task_id], start_time, finish, finish - env.now)

    def drive():
        index = 0
        while index < len(incoming) or running or any(queues.values()):
            arrivals = [incoming[index].arrival_time] if index < len(incoming) else []
            finishes = [item[2] for item in running.values()]
            uploads = [ready[queue[0].task_id] for vm_id, queue in queues.items()
                       if queue and vm_id not in running]
            yield env.timeout(min(arrivals + finishes + uploads) - env.now)
            # All finishes/releases and ready queued starts precede same-time arrivals.
            for vm_id in sorted(tuple(running)):
                task, started, finish = running[vm_id]
                if finish > env.now:
                    continue
                task.transition_to(TaskStatus.COMPLETED)
                ledger.release(vm_id, task, env.now)
                outcomes[task.task_id] = TaskOutcome(task.task_id, vm_id, task.status,
                                                    task.arrival_time, started, finish)
                del running[vm_id]
            for vm_id in sorted(queues):
                if vm_id not in running and queues[vm_id] and ready[queues[vm_id][0].task_id] <= env.now:
                    start(vm_id, queues[vm_id].popleft())
            while index < len(incoming) and incoming[index].arrival_time <= env.now:
                task = incoming[index]
                index += 1
                feasible = [vm for vm in candidates if fits_resources(task, vm)]
                eligible = [vm for vm in feasible if len(queues[vm.vm_id]) < vm.queue_capacity]
                estimates = tuple(estimate(vm, task) for vm in eligible)
                if not eligible:
                    task.transition_to(TaskStatus.FAILED)
                    reason = ("no reachable VM can satisfy task resources" if not feasible
                              else "all reachable resource-feasible VM queues are full")
                    outcomes[task.task_id] = TaskOutcome(task.task_id, None, task.status,
                                                        task.arrival_time, None, None, reason)
                    decisions.append(PlacementDecision(task.task_id, env.now, None, reason, ()))
                    continue
                if policy == FOG_FIRST_POLICY:
                    selected = choose_completion_vm(estimates)
                    has_fog = any(item.tier is ResourceTier.FOG for item in estimates)
                    has_cloud = any(item.tier is ResourceTier.CLOUD for item in estimates)
                    reason = ("fog_no_slower_than_cloud" if selected.tier is ResourceTier.FOG and has_cloud
                              else "only_fog_available" if selected.tier is ResourceTier.FOG
                              else "cloud_finishes_sooner" if has_fog else "fog_unavailable")
                else:
                    vm = choose_vm(policy, eligible, {vm.vm_id: len(queues[vm.vm_id]) + int(vm.vm_id in running)
                                                      for vm in eligible})
                    selected = next(item for item in estimates if item.vm_id == vm.vm_id)
                    reason = policy
                decisions.append(PlacementDecision(task.task_id, env.now, selected.vm_id, reason, estimates))
                ready[task.task_id] = selected.upload_ready_time_ms
                transfers.append({"task_id": task.task_id, "vm_id": selected.vm_id,
                                  "upload_ms": selected.upload_ms, "upload_ready_time_ms": ready[task.task_id]})
                task.transition_to(TaskStatus.QUEUED)
                queues[selected.vm_id].append(task)

    env.process(drive())
    env.run()
    if len(outcomes) != len(copied) or running or any(queues.values()):
        raise RuntimeError("network simulation did not account for every task")
    return SimulationResult(tuple(outcomes[task.task_id] for task in copied),
                            tuple(ledger.events), ledger.final_balances(),
                            placement_config=placement.manifest(), placement_decisions=tuple(decisions),
                            transfers=tuple(transfers))
