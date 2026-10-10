
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
