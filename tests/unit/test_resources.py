import pytest

from drlmots.entities import Host, ResourceTier, Task, VM


def make_task(task_id="task-1"):
    return Task(
        task_id=task_id,
        arrival_time=0,
        length=1000,
        data_size_kb=20,
    )


def test_host_accepts_positive_reported_capacities():
    host = Host(
        host_id="host-1",
        storage_tb=3,
        memory_gb=32,
        bandwidth_reported=1000,
    )

    assert host.host_id == "host-1"
    assert host.storage_tb == 3
    assert host.memory_gb == 32
    assert host.bandwidth_reported == 1000


@pytest.mark.parametrize("host_id", ["", "   "])
def test_host_rejects_empty_id(host_id):
    with pytest.raises(ValueError, match="host_id"):
        Host(host_id, 3, 32, 1000)


@pytest.mark.parametrize("field,value", [
    ("storage_tb", 0),
    ("storage_tb", -1),
    ("memory_gb", 0),
    ("memory_gb", -1),
    ("bandwidth_reported", 0),
    ("bandwidth_reported", -1),
    ("bandwidth_reported", float("inf")),
    ("bandwidth_reported", float("nan")),
])
def test_host_rejects_invalid_capacity(field, value):
    values = {
        "storage_tb": 3,
        "memory_gb": 32,
        "bandwidth_reported": 1000,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        Host("host-1", **values)


def test_vm_accepts_fog_tier_and_unspecified_queue_capacity():
    vm = VM(
        vm_id="vm-1",
        tier=ResourceTier.FOG,
        bandwidth_reported=200,
    )

    assert vm.vm_id == "vm-1"
    assert vm.tier is ResourceTier.FOG
    assert vm.queue == []
    assert vm.queue_capacity is None


def test_vm_accepts_cloud_tier():
    vm = VM("vm-2", ResourceTier.CLOUD, 200)

    assert vm.tier is ResourceTier.CLOUD


@pytest.mark.parametrize("vm_id", ["", "   "])
def test_vm_rejects_empty_id(vm_id):
    with pytest.raises(ValueError, match="vm_id"):
        VM(vm_id, ResourceTier.FOG, 200)


def test_vm_rejects_invalid_tier():
    with pytest.raises(TypeError, match="tier"):
        VM("vm-3", "fog", 200)


@pytest.mark.parametrize("bandwidth", [0, -1, float("inf"), float("nan")])
def test_vm_rejects_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="bandwidth_reported"):
        VM("vm-4", ResourceTier.FOG, bandwidth)


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_vm_rejects_invalid_queue_capacity(capacity):
    with pytest.raises((ValueError, TypeError), match="queue_capacity"):
        VM("vm-5", ResourceTier.FOG, 200, queue_capacity=capacity)


def test_vm_accepts_queue_within_capacity():
    task = make_task()
    vm = VM(
        "vm-6",
        ResourceTier.FOG,
        200,
        queue=[task],
        queue_capacity=2,
    )

    assert vm.queue == [task]
    assert vm.queue_capacity == 2


def test_vm_rejects_queue_over_capacity():
    with pytest.raises(ValueError, match="exceeds"):
        VM(
            "vm-7",
            ResourceTier.FOG,
            200,
            queue=[make_task("task-1"), make_task("task-2")],
            queue_capacity=1,
        )


def test_vm_copies_queue_list():
    tasks = [make_task()]
    vm = VM("vm-8", ResourceTier.FOG, 200, queue=tasks)

    tasks.clear()

    assert len(vm.queue) == 1


def test_vm_rejects_non_task_queue_items():
    with pytest.raises(TypeError, match="Task"):
        VM("vm-9", ResourceTier.FOG, 200, queue=["not-a-task"])


def test_vm_rejects_non_list_queue():
    with pytest.raises(TypeError, match="queue"):
        VM("vm-10", ResourceTier.FOG, 200, queue=())
