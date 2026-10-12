"""Deterministic synthetic inputs with explicit units and provenance."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math

from drlmots.entities import Task


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


def workload_manifest(tasks: Sequence[Task], *, length_unit: str) -> dict:
    """Save exact inputs and their stable checksum; no external dataset claim."""
    payload = {
        "data_source": "synthetic_fixed_list",
        "seed": None,  # The fixed-list generator uses no randomness.
        "length_unit": length_unit,
        "arrival_time_unit": "milliseconds",
        "tasks": [asdict(task) for task in tasks],
    }
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}
