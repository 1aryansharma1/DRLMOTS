from dataclasses import replace

import pytest

from drlmots.__main__ import smoke_report
from drlmots.entities import Host, ResourceTier, Task, TaskStatus, VM
from drlmots.environment import simulate
from drlmots.topology import FogNode, Topology


def topology(vm_count=1, capacity=1):
    vms = tuple(VM(f"vm-{index}", ResourceTier.FOG, 200, queue_capacity=capacity)
                for index in range(vm_count))
    return Topology(
        hosts=(Host("host", 3, 32, 1000),), vms=vms,
        fog_nodes=(FogNode("fog", ("host",), tuple(vm.vm_id for vm in vms)),),
    )


def task(task_id="t1", arrival=0, length=1000):
    return Task(task_id, arrival, length, 20)


def run(resources, tasks):
    return simulate(resources, tasks, task_length_unit="bytes")


def test_one_task_hand_calculated_timeline():
    result = run(topology(), [task(arrival=3)])
    outcome = result.outcomes[0]
    assert (outcome.start_time_ms, outcome.finish_time_ms) == (3, 5)
    assert result.makespan_ms == 5
    assert result.average_completion_time_ms == 2


def test_fcfs_queue_and_explicit_overflow_without_task_loss():
    result = run(topology(), [task("a"), task("b"), task("c")])
    assert result.completed_count == 2
    assert result.failed_count == 1
    assert [item.finish_time_ms for item in result.outcomes] == [2, 4, None]
    assert result.outcomes[2].failure_reason == "all VM queues are full"
    assert result.outcomes[2].status is TaskStatus.FAILED


def test_completion_before_same_timestamp_arrival():
    result = run(topology(), [task("a"), task("b"), task("c", arrival=2)])
    assert result.failed_count == 0
    assert [item.start_time_ms for item in result.outcomes] == [0, 2, 4]
    assert result.makespan_ms == 6


def test_two_vms_run_concurrently_and_fallback_when_queue_full():
    result = run(topology(2), [task(str(index)) for index in range(5)])
    assert [item.vm_id for item in result.outcomes] == ["vm-0", "vm-1", "vm-0", "vm-1", None]
    assert [item.finish_time_ms for item in result.outcomes] == [2, 2, 4, 4, None]
    assert result.completed_count + result.failed_count == 5


def test_inputs_remain_unchanged_and_reruns_are_identical():
    resources = topology()
    tasks = [task("later", arrival=10), task("first")]
    initial_queue = resources.vms[0].queue
    first = run(resources, tasks)
    assert run(resources, tasks) == first
    assert resources.vms[0].queue is initial_queue
    assert all(item.status is TaskStatus.CREATED for item in tasks)
    assert [item.task_id for item in first.outcomes] == ["later", "first"]
    assert first.makespan_ms == 12


def test_empty_workload():
    result = run(Topology((), ()), [])
    assert result.outcomes == ()
    assert result.makespan_ms == 0
    assert result.average_completion_time_ms is None


def test_nonempty_workload_without_vms_is_rejected():
    with pytest.raises(ValueError, match="at least one VM"):
        run(Topology((), ()), [task()])


def test_unknown_queue_capacity_is_not_interpreted_as_unlimited():
    with pytest.raises(ValueError, match="explicit queue_capacity"):
        run(topology(capacity=None), [task()])


def test_duplicate_tasks_are_rejected():
    with pytest.raises(ValueError, match="duplicate task_id"):
        run(topology(), [task(), task()])


def test_nonempty_initial_queues_rejected():
    resources = topology()
    resources.vms[0].queue.append(task("queued"))
    with pytest.raises(ValueError, match="initial VM queues"):
        run(resources, [task()])


def test_unsupported_resource_requirements_are_not_silently_ignored():
    demanding = replace(task(), required_resources={"cpu": 1})
    with pytest.raises(ValueError, match="resource feasibility"):
        run(topology(), [demanding])


def test_noncreated_tasks_rejected():
    with pytest.raises(ValueError, match="CREATED"):
        run(topology(), [replace(task(), status=TaskStatus.COMPLETED)])


def test_ambiguous_time_model_rejected():
    with pytest.raises(ValueError, match="bytes"):
        simulate(topology(), [task()], task_length_unit="unknown")


def test_smoke_example_matches_hand_calculation():
    report = smoke_report()
    assert report["completed_count"] == 3
    assert report["failed_count"] == 0
    assert report["makespan_ms"] == 4
    assert report["average_completion_time_ms"] == pytest.approx(8 / 3)
    assert [item["finish_time_ms"] for item in report["outcomes"]] == [2, 4, 3]
    assert report == smoke_report()


def test_fallback_to_a_different_fog_node():
    resources = Topology(
        hosts=(Host("h0", 3, 32, 1000), Host("h1", 3, 32, 1000)),
        vms=(VM("vm-0", ResourceTier.FOG, 200, queue_capacity=1),
             VM("vm-1", ResourceTier.FOG, 200, queue_capacity=1)),
        fog_nodes=(FogNode("f0", ("h0",), ("vm-0",)),
                   FogNode("f1", ("h1",), ("vm-1",))),
    )
    # vm-0 is busy with a full waiting queue; vm-1 still has a free queue slot.
    result = run(resources, [task(str(index)) for index in range(4)])
    assert result.outcomes[-1].vm_id == "vm-1"
    assert result.completed_count == 4


def test_larger_fractional_workload_preserves_serial_execution_and_accounting():
    resources = topology(4, capacity=3)
    tasks = [task(str(index), arrival=(index % 17) / 10, length=100 + index * 7)
             for index in range(100)]
    result = run(resources, tasks)
    assert result.completed_count + result.failed_count == len(tasks)
    assert len({outcome.task_id for outcome in result.outcomes}) == len(tasks)
    assert result == run(resources, tasks)
    for vm in resources.vms:
        finished = sorted((outcome for outcome in result.outcomes if outcome.vm_id == vm.vm_id),
                          key=lambda outcome: outcome.start_time_ms)
        for item in finished:
            assert item.start_time_ms >= item.arrival_time_ms
            assert item.finish_time_ms > item.start_time_ms
        for previous, following in zip(finished, finished[1:]):
            assert previous.finish_time_ms <= following.start_time_ms
