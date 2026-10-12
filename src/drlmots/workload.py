"""Fixed and seeded synthetic inputs with explicit units and provenance."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import platform
import random

from drlmots.entities import Task, validate_resource_amounts
from drlmots.resources import require_declared_units, validate_resource_units


def _validated_range(value, name: str, *, integers=False, allow_zero=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain a lower and upper bound")
    for bound in value:
        types = (int,) if integers else (int, float)
        if isinstance(bound, bool) or not isinstance(bound, types):
            raise TypeError(f"{name} bounds must be {'integers' if integers else 'numbers'}")
        try:
            finite = math.isfinite(bound)
        except OverflowError:
            finite = False
        if not finite or bound < 0 or (not allow_zero and bound == 0):
            raise ValueError(f"{name} bounds must be finite and {'nonnegative' if allow_zero else 'positive'}")
    if value[0] > value[1]:
        raise ValueError(f"{name} bounds must be in ascending order")
    return tuple(value)


@dataclass(frozen=True)
class SyntheticWorkloadSettings:
    """Explicit synthetic settings; none are asserted as paper defaults."""

    task_count: int
    seed: int
    interarrival_ms_range: tuple[float, float]
    length_bytes_range: tuple[int, int]
    data_size_kb_range: tuple[float, float]
    required_resources: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("task_count", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a nonnegative integer")
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        object.__setattr__(self, "interarrival_ms_range", _validated_range(
            self.interarrival_ms_range, "interarrival_ms_range", allow_zero=True
        ))
        object.__setattr__(self, "length_bytes_range", _validated_range(
            self.length_bytes_range, "length_bytes_range", integers=True
        ))
        object.__setattr__(self, "data_size_kb_range", _validated_range(
            self.data_size_kb_range, "data_size_kb_range"
        ))
        object.__setattr__(self, "required_resources", validate_resource_amounts(
            self.required_resources, "required_resources"
        ))

    def manifest_settings(self) -> dict:
        result = asdict(self)
        if not self.required_resources:
            del result["required_resources"]  # Preserve demand-free generator-v1 provenance.
        return result


def seeded_workload(settings: SyntheticWorkloadSettings) -> list[Task]:
    """Generate exactly task_count tasks with a local, reproducible RNG.

    The first arrival is zero. Subsequent uniform interarrival intervals, integer
    byte lengths and uniform data sizes follow assumption A-018, not paper claims.
    """
    if not isinstance(settings, SyntheticWorkloadSettings):
        raise TypeError("settings must be SyntheticWorkloadSettings")
    rng = random.Random(settings.seed)
    tasks = []
    arrival = 0.0
    for index in range(settings.task_count):
        if index:
            arrival += rng.uniform(*settings.interarrival_ms_range)
            if not math.isfinite(arrival):
                raise ValueError("accumulated arrival timestamp must remain finite")
        length = rng.randint(*settings.length_bytes_range)
        processing_time_ms(length, length_unit="bytes")
        tasks.append(Task(
            task_id=f"synthetic-{index + 1:06d}",
            arrival_time=arrival,
            length=length,
            data_size_kb=rng.uniform(*settings.data_size_kb_range),
            required_resources=settings.required_resources,
        ))
    return tasks


def processing_time_ms(length: float, *, length_unit: str) -> float:
    """Recorded Eq. (8) interpretation: 1000 bytes take 2 milliseconds.

    Only explicitly byte-valued synthetic/test inputs may use this function.
    This does not resolve the paper's unexplained task length of 700,000.
    """
    if length_unit != "bytes":
        raise ValueError("length_unit must explicitly be 'bytes' for this model")
    Task._validate_number("length", length, allow_zero=False)
    duration = length * (2.0 / 1000.0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("processing duration must be finite and positive")
    return duration


def fixed_workload(records: Sequence[Mapping], *, length_unit: str) -> list[Task]:
    """Construct independent tasks, rejecting duplicate IDs and ambiguous units."""
    if length_unit != "bytes":
        raise ValueError("length_unit must explicitly be 'bytes' for this model")
    tasks = []
    ids = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each task record must be a mapping")
        task = Task(**record)
        processing_time_ms(task.length, length_unit=length_unit)
        if task.task_id in ids:
            raise ValueError(f"duplicate task_id {task.task_id!r}")
        ids.add(task.task_id)
        tasks.append(task)
    return tasks


def workload_manifest(
    tasks: Sequence[Task],
    *,
    length_unit: str,
    generator_settings: SyntheticWorkloadSettings | None = None,
    resource_units=None,
) -> dict:
    """Save exact inputs and their stable checksum; no external dataset claim."""
    units = validate_resource_units({} if resource_units is None else resource_units)
    for task in tasks:
        require_declared_units(task.required_resources, units)
    payload = {
        "data_source": "synthetic_fixed_list",
        "seed": None,  # The fixed-list generator uses no randomness.
        "length_unit": length_unit,
        "arrival_time_unit": "milliseconds",
        "tasks": [asdict(task) for task in tasks],
    }
    if generator_settings is not None:
        if not isinstance(generator_settings, SyntheticWorkloadSettings):
            raise TypeError("generator_settings must be SyntheticWorkloadSettings")
        require_declared_units(generator_settings.required_resources, units)
        if length_unit != "bytes":
            raise ValueError("seeded workloads must explicitly use bytes")
        if len(tasks) != generator_settings.task_count:
            raise ValueError("manifest task count does not match generator settings")
        if any(task.required_resources != generator_settings.required_resources for task in tasks):
            raise ValueError("manifest demands do not match the generator resource profile")
        payload.update({
            "data_source": "synthetic_seeded",
            "seed": generator_settings.seed,
            "generator": {
                "name": "uniform_interarrival",
                "version": 2 if generator_settings.required_resources else 1,
                "rng": "python.random.Random",
                "python_version": platform.python_version(),
                "first_arrival_ms": 0,
                "length_distribution": "integer_uniform_inclusive",
                "data_size_distribution": "uniform",
                "settings": generator_settings.manifest_settings(),
            },
        })
    if units:
        payload["resource_units"] = units
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}
