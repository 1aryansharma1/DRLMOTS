"""Small deterministic SimPy baseline, independent of a learned policy."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import math

import simpy

from drlmots.entities import Task, TaskStatus
from drlmots.topology import Topology
from drlmots.resources import (
    ResourceEvent, ResourceLedger, fits_resources, require_declared_units, validate_resource_units,
)
from drlmots.workload import processing_time_ms
from drlmots.baselines import POLICIES, FOG_FIRST_POLICY, choose_vm
from drlmots.actions import ActionMap, action_mask
from drlmots.state import DecisionObservation, StateSchema, VMObservation, encode_state
from drlmots.reward import RewardConfig, RewardResult, score_action


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    vm_id: str | None
    status: TaskStatus
    arrival_time_ms: float
    start_time_ms: float | None
    finish_time_ms: float | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    observation: DecisionObservation
    state: tuple[float, ...]
    action_mask: tuple[bool, ...]
    selected_action: int | None
    selected_vm_id: str | None
    failure_reason: str | None
    reward: RewardResult | None = None


@dataclass(frozen=True)
class SimulationResult:
    outcomes: tuple[TaskOutcome, ...]
    resource_events: tuple[ResourceEvent, ...] = ()
    available_resources: dict = field(default_factory=dict)
    decision_schema: dict | None = None
    decisions: tuple[DecisionRecord, ...] = ()
    reward_config: RewardConfig | None = None
    placement_config: dict | None = None
    placement_decisions: tuple = ()
    transfers: tuple = ()

    @property
    def completed_count(self) -> int:
        return sum(item.status is TaskStatus.COMPLETED for item in self.outcomes)

    @property
    def failed_count(self) -> int:
        return sum(item.status is TaskStatus.FAILED for item in self.outcomes)

    @property
    def makespan_ms(self) -> float:
        """Latest successful finish measured from simulation time zero."""
        return max((item.finish_time_ms for item in self.outcomes
                    if item.status is TaskStatus.COMPLETED), default=0.0)

    @property
    def average_completion_time_ms(self) -> float | None:
        completed = [item for item in self.outcomes if item.status is TaskStatus.COMPLETED]
        if not completed:
            return None
        return sum(item.finish_time_ms - item.arrival_time_ms for item in completed) / len(completed)


def simulate(
    topology: Topology,
    tasks: Sequence[Task],
    *,
    task_length_unit: str,
    resource_units=None,
    policy: str = "least_queue_fcfs",
    capture_decisions: bool = False,
    reward_config: RewardConfig | None = None,
    placement_config=None,
) -> SimulationResult:
    """Place tasks using a selected baseline, then execute each VM's queue FCFS.

    Queue capacity counts waiting tasks, excluding the one running task. Ties
    between VMs use VM ID; simultaneous arrivals retain input order. Completions
    precede arrivals at the same timestamp. A full cluster rejects the arriving
    task explicitly. This is a software baseline, not the paper's DQN/threshold
    policy. Caller-owned tasks and VM queues are never changed.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose one of {POLICIES}")
    if type(capture_decisions) is not bool:
        raise TypeError("capture_decisions must be bool")
    if reward_config is not None and not isinstance(reward_config, RewardConfig):
        raise TypeError("reward_config must be RewardConfig or None")
    if placement_config is not None:
        if capture_decisions or reward_config is not None:
            raise ValueError("network-aware runs require a new state/reward contract; legacy RL capture is unsupported")
        from drlmots.fog_cloud import simulate_network
        return simulate_network(topology, tasks, task_length_unit=task_length_unit,
                                resource_units=resource_units, policy=policy, placement=placement_config)
    if policy == FOG_FIRST_POLICY:
        raise ValueError("fog_first_completion requires schema-3 placement/network settings")
    capture_decisions = capture_decisions or reward_config is not None
    topology.validate()
    units = validate_resource_units({} if resource_units is None else resource_units)
    for vm in topology.vms:
        require_declared_units(vm.resource_capacities or {}, units)
    ledger = ResourceLedger(topology.vms)
    schema = StateSchema.from_topology(topology, units) if capture_decisions else None
    actions = ActionMap(schema.vm_ids) if schema is not None else None
    decisions = []
    if task_length_unit != "bytes":
        raise ValueError("task_length_unit must explicitly be 'bytes' for this model")
    copied = []
    task_ids = set()
    durations = {}
    for task in tasks:
        if not isinstance(task, Task):
            raise TypeError("every workload item must be a Task")
        task = replace(task)
        if task.task_id in task_ids:
            raise ValueError(f"duplicate task_id {task.task_id!r}")
        if task.status is not TaskStatus.CREATED:
            raise ValueError("simulation inputs must have CREATED status")
        require_declared_units(task.required_resources, units)
        task_ids.add(task.task_id)
        durations[task.task_id] = processing_time_ms(task.length, length_unit=task_length_unit)
        copied.append(task)
    if not copied:
        return SimulationResult((), available_resources=ledger.final_balances(),
                                decision_schema=schema.manifest() if schema is not None else None,
                                reward_config=reward_config)
    if not topology.vms:
        raise ValueError("at least one VM is required for a non-empty workload")
    for vm in topology.vms:
        if vm.queue_capacity is None:
            raise ValueError(f"explicit queue_capacity required for VM {vm.vm_id!r}")
        if vm.queue:
            raise ValueError("initial VM queues must be empty for this baseline")

    incoming = sorted(copied, key=lambda task: task.arrival_time)
    env = simpy.Environment()
    queues = {vm.vm_id: deque() for vm in topology.vms}
    running: dict[str, tuple[Task, float, float]] = {}
    outcomes = {}

    def observe(task):
        snapshots = []
        for vm in sorted(topology.vms, key=lambda item: item.vm_id):
            capacities = vm.resource_capacities or {}
            available = ledger.available(vm.vm_id) or {}
            snapshots.append(VMObservation(
                vm.vm_id, vm.tier, vm.vm_id in running,
                running[vm.vm_id][2] - env.now if vm.vm_id in running else 0,
                len(queues[vm.vm_id]), vm.queue_capacity,
                math.fsum(durations[item.task_id] for item in queues[vm.vm_id]),
                tuple(capacities.get(name) for name in schema.resource_names),
                tuple(available.get(name) for name in schema.resource_names),
            ))
        return DecisionObservation(task.task_id, env.now, task.length, task.data_size_kb,
                                   schema.resource_units,
                                   tuple(task.required_resources.get(name, 0) for name in schema.resource_names),
                                   tuple(snapshots))

    def start(vm_id, task):
        finish = env.now + durations[task.task_id]
        if not math.isfinite(finish) or finish <= env.now:
            raise ValueError("finish timestamp must be finite and greater than start")
        task.transition_to(TaskStatus.RUNNING)
        ledger.reserve(vm_id, task, env.now)
        running[vm_id] = (task, env.now, finish)

    def drive():
        index = 0
        while index < len(incoming) or running:
            next_arrival = incoming[index].arrival_time if index < len(incoming) else math.inf
            next_finish = min((item[2] for item in running.values()), default=math.inf)
            yield env.timeout(min(next_arrival, next_finish) - env.now)

            # Release all completed work before admitting this timestamp's arrivals.
            for vm_id in sorted(tuple(running)):
                task, started, finished = running[vm_id]
                if finished > env.now:
                    continue
                task.transition_to(TaskStatus.COMPLETED)
                ledger.release(vm_id, task, env.now)
                outcomes[task.task_id] = TaskOutcome(
                    task.task_id, vm_id, task.status, task.arrival_time, started, finished
                )
                del running[vm_id]
                if queues[vm_id]:
                    start(vm_id, queues[vm_id].popleft())

            while index < len(incoming) and incoming[index].arrival_time <= env.now:
                task = incoming[index]
                index += 1
                feasible = [vm for vm in topology.vms if fits_resources(task, vm)]
                eligible = [vm for vm in feasible
                            if vm.vm_id not in running or len(queues[vm.vm_id]) < vm.queue_capacity]
                observation = observe(task) if capture_decisions else None
                mask = action_mask(observation) if observation is not None else None
                if mask is not None and {actions.vm_id(i) for i, valid in enumerate(mask) if valid} != {
                    vm.vm_id for vm in eligible
                }:
                    raise RuntimeError("action mask disagrees with simulator eligibility")
                if not eligible:
                    task.transition_to(TaskStatus.FAILED)
                    outcomes[task.task_id] = TaskOutcome(
                        task.task_id, None, task.status, task.arrival_time, None, None,
                        ("no VM can satisfy task resources" if not feasible
                         else "all resource-feasible VM queues are full" if task.required_resources
                         else "all VM queues are full"),
                    )
                    if observation is not None:
                        decisions.append(DecisionRecord(observation, encode_state(schema, observation), mask,
                                                        None, None, outcomes[task.task_id].failure_reason,
                                                        score_action(observation, None, reward_config)
                                                        if reward_config is not None else None))
                    continue
                vm = choose_vm(policy, eligible, {
                    item.vm_id: len(queues[item.vm_id]) + int(item.vm_id in running)
                    for item in eligible
                })
                if observation is not None:
                    decisions.append(DecisionRecord(observation, encode_state(schema, observation), mask,
                                                    actions.index(vm.vm_id), vm.vm_id, None,
                                                    score_action(observation, actions.index(vm.vm_id), reward_config)
                                                    if reward_config is not None else None))
                task.transition_to(TaskStatus.QUEUED)
                if vm.vm_id in running:
                    queues[vm.vm_id].append(task)
                else:
                    start(vm.vm_id, task)

    env.process(drive())
    env.run()
    if len(outcomes) != len(copied) or any(queues.values()) or running:
        raise RuntimeError("simulation did not account for every task")
    return SimulationResult(
        tuple(outcomes[task.task_id] for task in copied), tuple(ledger.events), ledger.final_balances(),
        schema.manifest() if schema is not None else None, tuple(decisions), reward_config,
    )
