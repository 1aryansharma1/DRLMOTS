
"""Core data structures for DRLMOTS.

Unspecified paper units are kept generic rather than guessed.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from collections.abc import Mapping


class TaskStatus(str, Enum):
    """Lifecycle states used by the task model."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """A schedulable task in the cloud-fog environment.

    Attributes:
        task_id: Unique identifier supplied by the workload.
        arrival_time: Non-negative arrival timestamp in model time units.
        length: Positive task length; its physical unit remains unresolved.
        data_size_kb: Positive input data size in kilobytes.
        required_resources: Resource names mapped to positive requirements.
        status: Current lifecycle state.
    """

    task_id: str
    arrival_time: float
    length: float
    data_size_kb: float
    required_resources: Mapping[str, float] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.CREATED

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")

        self._validate_number("arrival_time", self.arrival_time, allow_zero=True)
        self._validate_number("length", self.length, allow_zero=False)
        self._validate_number("data_size_kb", self.data_size_kb, allow_zero=False)

        if not isinstance(self.required_resources, Mapping):
            raise TypeError("required_resources must be a mapping")

        validated_resources: dict[str, float] = {}
        for name, amount in self.required_resources.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("resource names must be non-empty strings")
            self._validate_number(
                f"required_resources[{name!r}]",
                amount,
                allow_zero=False,
            )
            validated_resources[name] = amount

        # Copy the mapping so later changes to the caller's dictionary
        # do not silently change this task's requirements.
        self.required_resources = validated_resources

        if not isinstance(self.status, TaskStatus):
            raise TypeError("status must be a TaskStatus value")

    @staticmethod
    def _validate_number(
        name: str,
        value: float,
        *,
        allow_zero: bool,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

        if value < 0 or (not allow_zero and value == 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}")

class ResourceTier(str, Enum):
    """Execution tier for a VM."""

    FOG = "fog"
    CLOUD = "cloud"


@dataclass
class Host:
    """Physical host with capacities reported by the paper.

    Values are supplied explicitly; the model does not assume defaults.
    The paper's bandwidth notation is retained as a reported value
    because its precise unit interpretation is not yet confirmed.
    """

    host_id: str
    storage_tb: float
    memory_gb: float
    bandwidth_reported: float

    def __post_init__(self) -> None:
        if not isinstance(self.host_id, str) or not self.host_id.strip():
            raise ValueError("host_id must be a non-empty string")

        self._validate_positive("storage_tb", self.storage_tb)
        self._validate_positive("memory_gb", self.memory_gb)
        self._validate_positive("bandwidth_reported", self.bandwidth_reported)

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


@dataclass
class VM:
    """Virtual machine used as a scheduling resource.

    Queue capacity is None when the paper's capacity is unspecified.
    None means unknown, not unlimited.
    """

    vm_id: str
    tier: ResourceTier
    bandwidth_reported: float
    queue: list[Task] = field(default_factory=list)
    queue_capacity: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.vm_id, str) or not self.vm_id.strip():
            raise ValueError("vm_id must be a non-empty string")

        if not isinstance(self.tier, ResourceTier):
            raise TypeError("tier must be a ResourceTier value")

        Host._validate_positive("bandwidth_reported", self.bandwidth_reported)

        if not isinstance(self.queue, list):
            raise TypeError("queue must be a list of Task objects")

        if any(not isinstance(task, Task) for task in self.queue):
            raise TypeError("every queue item must be a Task")

        # Copy the queue to avoid unintended external list mutation.
        self.queue = list(self.queue)

        if self.queue_capacity is not None:
            if (
                isinstance(self.queue_capacity, bool)
                or not isinstance(self.queue_capacity, int)
            ):
                raise TypeError("queue_capacity must be an integer or None")

            if self.queue_capacity <= 0:
                raise ValueError("queue_capacity must be positive")

            if len(self.queue) > self.queue_capacity:
                raise ValueError("initial queue exceeds queue_capacity")
