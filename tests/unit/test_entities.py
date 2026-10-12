
import pytest

from drlmots.entities import Task, TaskStatus


def test_task_creation_with_valid_fields():
    task = Task(
        task_id="task-001",
        arrival_time=0,
        length=1000,
        data_size_kb=20,
        required_resources={"cpu": 1.0, "memory_gb": 2.0},
    )

    assert task.task_id == "task-001"
    assert task.arrival_time == 0
    assert task.length == 1000
    assert task.data_size_kb == 20
    assert task.required_resources == {"cpu": 1.0, "memory_gb": 2.0}
    assert task.status is TaskStatus.CREATED


def test_task_can_have_unspecified_resource_requirements():
    task = Task(
        task_id="task-002",
        arrival_time=5,
        length=100,
        data_size_kb=1,
    )

    assert task.required_resources == {}


@pytest.mark.parametrize("task_id", ["", "   "])
def test_empty_task_id_is_rejected(task_id):
    with pytest.raises(ValueError, match="task_id"):
        Task(task_id, 0, 100, 20)


@pytest.mark.parametrize("arrival_time", [-1, float("inf"), float("nan")])
def test_invalid_arrival_time_is_rejected(arrival_time):
    with pytest.raises(ValueError, match="arrival_time"):
        Task("task-003", arrival_time, 100, 20)


@pytest.mark.parametrize("length", [0, -1, float("inf"), float("nan")])
def test_invalid_task_length_is_rejected(length):
    with pytest.raises(ValueError, match="length"):
        Task("task-004", 0, length, 20)


@pytest.mark.parametrize("data_size_kb", [0, -1, float("inf"), float("nan")])
def test_invalid_data_size_is_rejected(data_size_kb):
    with pytest.raises(ValueError, match="data_size_kb"):
        Task("task-005", 0, 100, data_size_kb)


def test_invalid_resource_requirement_is_rejected():
    with pytest.raises(ValueError, match="required_resources"):
        Task(
            "task-006",
            0,
            100,
            20,
            required_resources={"cpu": 0},
        )


def test_resource_mapping_is_copied():
    requirements = {"cpu": 1.0}
    task = Task("task-007", 0, 100, 20, required_resources=requirements)

    requirements["cpu"] = 8.0

    assert task.required_resources["cpu"] == 1.0


def test_invalid_status_type_is_rejected():
    with pytest.raises(TypeError, match="status"):
        Task("task-008", 0, 100, 20, status="running")


def test_task_lifecycle():
    task = Task("lifecycle", 0, 100, 20)
    for status in (TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.COMPLETED):
        task.transition_to(status)
        assert task.status is status


@pytest.mark.parametrize("source", list(TaskStatus))
@pytest.mark.parametrize("target", list(TaskStatus))
def test_transition_rules(source, target):
    allowed = {
        (TaskStatus.CREATED, TaskStatus.QUEUED),
        (TaskStatus.CREATED, TaskStatus.FAILED),
        (TaskStatus.QUEUED, TaskStatus.RUNNING),
        (TaskStatus.QUEUED, TaskStatus.FAILED),
        (TaskStatus.RUNNING, TaskStatus.COMPLETED),
        (TaskStatus.RUNNING, TaskStatus.FAILED),
    }
    task = Task("transition", 0, 100, 20, status=source)
    if (source, target) in allowed:
        task.transition_to(target)
        assert task.status is target
    else:
        with pytest.raises(ValueError, match="invalid task transition"):
            task.transition_to(target)
        assert task.status is source


def test_transition_rejects_raw_string():
    with pytest.raises(TypeError, match="TaskStatus"):
        Task("transition", 0, 100, 20).transition_to("queued")
