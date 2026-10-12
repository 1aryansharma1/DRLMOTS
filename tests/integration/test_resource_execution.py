from dataclasses import asdict, replace
from pathlib import Path

import pytest

from drlmots.__main__ import configured_report
from drlmots.entities import Host, ResourceTier, Task, TaskStatus, VM
from drlmots.environment import simulate
from drlmots.run_config import load_run_config
from drlmots.topology import FogNode, Topology


UNITS = {"cpu": "abstract units", "memory_gb": "GB", "memory_mb": "MB"}
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "resources.yaml"


def topology(*capacities, queue_capacity=1):
    vms = tuple(VM(f"vm-{index}", ResourceTier.FOG, 200, queue_capacity=queue_capacity,
                   resource_capacities=capacity) for index, capacity in enumerate(capacities))
    return Topology((Host("host", 3, 32, 1000),), vms,
                    (FogNode("fog", ("host",), tuple(vm.vm_id for vm in vms)),))


def task(task_id="task", arrival=0, **requirements):
    return Task(task_id, arrival, 1000, 20, required_resources=requirements)


def run(resources, tasks):
    return simulate(resources, tasks, task_length_unit="bytes", resource_units=UNITS)


def test_exact_fit_only_running_tasks_reserve_and_release_before_next_start():
    resources = topology({"cpu": 2, "memory_gb": 4})
    tasks = [task("a", cpu=2, memory_gb=4), task("b", cpu=2, memory_gb=4)]
    original = asdict(resources)
    result = run(resources, tasks)
    assert [outcome.finish_time_ms for outcome in result.outcomes] == [2, 4]
    assert [(event.task_id, event.action, event.time_ms) for event in result.resource_events] == [
        ("a", "reserve", 0), ("a", "release", 2), ("b", "reserve", 2), ("b", "release", 4),
    ]
    assert result.resource_events[0].available_after == {"cpu": 0, "memory_gb": 0}
    assert result.available_resources == {"vm-0": {"cpu": 2, "memory_gb": 4}}
    assert asdict(resources) == original
    assert all(work.status is TaskStatus.CREATED for work in tasks)
    assert run(resources, tasks) == result


def test_same_time_completion_frees_resources_and_waiting_slot_before_arrival():
    result = run(topology({"cpu": 2}), [task("a", cpu=2), task("b", cpu=2), task("c", arrival=2, cpu=2)])
    assert result.completed_count == 3
    assert result.failed_count == 0
    assert [outcome.start_time_ms for outcome in result.outcomes] == [0, 2, 4]
    assert result.available_resources == {"vm-0": {"cpu": 2}}


def test_idle_infeasible_vm_is_skipped_for_a_feasible_vm():
    result = run(topology({"cpu": 1}, {"cpu": 2}), [task(cpu=2)])
    assert result.outcomes[0].vm_id == "vm-1"
    assert result.completed_count == 1


def test_full_feasible_queue_does_not_fall_back_to_an_infeasible_idle_vm():
    result = run(topology({"cpu": 2}, {"cpu": 1}), [task(name, cpu=2) for name in ("a", "b", "c")])
    assert (result.completed_count, result.failed_count) == (2, 1)
    assert result.outcomes[-1].failure_reason == "all resource-feasible VM queues are full"
    assert result.outcomes[-1].vm_id is None
    assert len(result.resource_events) == 4


def test_full_queue_falls_back_to_another_feasible_vm():
    result = run(topology({"cpu": 2}, {"cpu": 2}), [task(str(index), cpu=2) for index in range(4)])
    assert [outcome.vm_id for outcome in result.outcomes] == ["vm-0", "vm-1", "vm-0", "vm-1"]
    assert result.completed_count == 4
    assert len(result.resource_events) == 8


@pytest.mark.parametrize("capacities,requirements", [
    (None, {"cpu": 1}), ({}, {"cpu": 1}), ({"cpu": 1}, {"cpu": 2}),
    ({"cpu": 2}, {"cpu": 1, "memory_gb": 1}), ({"memory_gb": 128}, {"memory_mb": 1}),
])
def test_unsatisfiable_or_unknown_capacity_has_explicit_terminal_failure(capacities, requirements):
    result = run(topology(capacities), [task(**requirements)])
    assert result.completed_count == 0
    assert result.failed_count == 1
    assert result.outcomes[0].status is TaskStatus.FAILED
    assert result.outcomes[0].failure_reason == "no VM can satisfy task resources"
    assert result.makespan_ms == 0
    assert result.average_completion_time_ms is None
    assert result.resource_events == ()


def test_known_capacity_with_missing_units_is_invalid_even_for_empty_workload():
    with pytest.raises(ValueError, match="resource_units"):
        simulate(topology({"cpu": 2}), [], task_length_unit="bytes")


def test_empty_workload_reports_full_capacity_without_reservations():
    result = run(topology({"cpu": 2}), [])
    assert result.outcomes == ()
    assert result.resource_events == ()
    assert result.available_resources == {"vm-0": {"cpu": 2}}


def test_unrequested_resources_remain_available():
    result = run(topology({"cpu": 2, "memory_gb": 4}), [task(cpu=1)])
    assert result.resource_events[0].available_after == {"cpu": 1, "memory_gb": 4}


def test_resource_config_example_and_replay():
    config = load_run_config(CONFIG)
    report = configured_report(config)
    assert report == configured_report(config)
    assert (report["completed_count"], report["failed_count"]) == (4, 2)
    assert report["makespan_ms"] == 4
    assert report["average_completion_time_ms"] == 3
    assert [outcome["vm_id"] for outcome in report["outcomes"]] == [
        "vm-cloud", "vm-fog-a", "vm-cloud", "vm-fog-a", None, None,
    ]
    assert len(report["resource_events"]) == 8
    for vm in config.topology.vms:
        assert report["available_resources"][vm.vm_id] == vm.resource_capacities
    assert report["workload"]["generator"]["version"] == 2
    assert report["workload"]["resource_units"] == config.resource_units


def test_heterogeneous_tasks_are_filtered_independently_and_all_are_accounted_for():
    resources = topology({"cpu": 1, "memory_gb": 2}, {"cpu": 4, "memory_gb": 8}, queue_capacity=2)
    tasks = [task(str(index), arrival=(index % 13) / 10, cpu=(index % 5) + 1,
                  memory_gb=(index % 4) + 1) for index in range(100)]
    result = run(resources, tasks)
    assert result.completed_count + result.failed_count == 100
    assert len({outcome.task_id for outcome in result.outcomes}) == 100
    by_id = {work.task_id: work for work in tasks}
    capacities = {vm.vm_id: vm.resource_capacities for vm in resources.vms}
    for outcome in result.outcomes:
        if outcome.status is TaskStatus.COMPLETED:
            assert all(amount <= capacities[outcome.vm_id][name]
                       for name, amount in by_id[outcome.task_id].required_resources.items())
    assert len(result.resource_events) == 2 * result.completed_count
    assert all(amount >= 0 for event in result.resource_events for amount in event.available_after.values())
    assert result.available_resources == capacities
    assert result == run(resources, tasks)
