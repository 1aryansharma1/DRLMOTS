from copy import deepcopy
from dataclasses import replace
import random

import pytest

from drlmots.workload import (
    SyntheticWorkloadSettings, fixed_workload, processing_time_ms,
    seeded_workload, workload_manifest,
)


RECORD = {"task_id": "t1", "arrival_time": 0, "length": 1000, "data_size_kb": 20}


@pytest.mark.parametrize("length,expected", [(1000, 2), (500, 1), (2000, 4)])
def test_explicit_byte_conversion(length, expected):
    assert processing_time_ms(length, length_unit="bytes") == expected


@pytest.mark.parametrize("unit", [None, "unknown", "instructions", "kilobytes"])
def test_ambiguous_units_rejected(unit):
    with pytest.raises(ValueError, match="bytes"):
        processing_time_ms(1000, length_unit=unit)


@pytest.mark.parametrize("length", [True, 0, -1, float("nan"), float("inf"), 5e-324])
def test_invalid_service_time_inputs(length):
    with pytest.raises((TypeError, ValueError)):
        processing_time_ms(length, length_unit="bytes")


def test_fixed_records_round_trip_without_aliasing():
    record = {**RECORD, "required_resources": {"cpu": 1}}
    tasks = fixed_workload([record], length_unit="bytes")
    record["required_resources"]["cpu"] = 5
    assert tasks[0].required_resources == {"cpu": 1}
    assert tasks[0].task_id == RECORD["task_id"]
    assert tasks[0].length == RECORD["length"]


def test_duplicate_task_ids_rejected():
    with pytest.raises(ValueError, match="duplicate task_id"):
        fixed_workload([RECORD, RECORD], length_unit="bytes")


def test_manifest_reproducibility_and_input_sensitivity():
    first = workload_manifest(fixed_workload([RECORD], length_unit="bytes"), length_unit="bytes")
    second = workload_manifest(fixed_workload([deepcopy(RECORD)], length_unit="bytes"), length_unit="bytes")
    different = workload_manifest(fixed_workload([{**RECORD, "arrival_time": 1}], length_unit="bytes"), length_unit="bytes")
    assert first == second
    assert first["sha256"] != different["sha256"]
    assert first["data_source"] == "synthetic_fixed_list"


def settings(**overrides):
    values = dict(task_count=50, seed=42, interarrival_ms_range=(0, 3),
                  length_bytes_range=(500, 2000), data_size_kb_range=(20, 150))
    return SyntheticWorkloadSettings(**{**values, **overrides})


def test_seeded_workload_is_repeatable_and_leaves_global_rng_unchanged():
    before = random.getstate()
    first = seeded_workload(settings())
    assert first == seeded_workload(settings())
    assert before == random.getstate()
    assert first != seeded_workload(settings(seed=43))


def test_generated_count_ids_bounds_and_arrival_order():
    tasks = seeded_workload(settings())
    assert len(tasks) == 50
    assert len({task.task_id for task in tasks}) == 50
    assert tasks[0].arrival_time == 0
    assert all(500 <= task.length <= 2000 and isinstance(task.length, int) for task in tasks)
    assert all(20 <= task.data_size_kb <= 150 for task in tasks)
    assert all(task.required_resources == {} for task in tasks)
    for previous, following in zip(tasks, tasks[1:]):
        assert 0 <= following.arrival_time - previous.arrival_time <= 3 + 1e-12


def test_constant_bounds_produce_exact_expected_fields():
    tasks = seeded_workload(settings(task_count=4, interarrival_ms_range=(2, 2),
                                     length_bytes_range=(1000, 1000), data_size_kb_range=(20, 20)))
    assert [task.arrival_time for task in tasks] == [0, 2, 4, 6]
    assert [task.length for task in tasks] == [1000] * 4
    assert [task.data_size_kb for task in tasks] == [20] * 4


def test_zero_intervals_are_simultaneous_arrivals():
    tasks = seeded_workload(settings(interarrival_ms_range=(0, 0)))
    assert all(task.arrival_time == 0 for task in tasks)


def test_empty_generated_workload_retains_settings_in_manifest():
    config = settings(task_count=0, seed=0)
    tasks = seeded_workload(config)
    manifest = workload_manifest(tasks, length_unit="bytes", generator_settings=config)
    assert tasks == []
    assert manifest["seed"] == 0
    assert manifest["data_source"] == "synthetic_seeded"
    assert manifest["generator"]["settings"]["task_count"] == 0
    assert manifest == workload_manifest(seeded_workload(config), length_unit="bytes", generator_settings=config)


@pytest.mark.parametrize("changes", [
    {"seed": 43}, {"task_count": 49}, {"interarrival_ms_range": (0, 4)},
    {"length_bytes_range": (500, 3000)}, {"data_size_kb_range": (20, 160)},
])
def test_manifest_records_all_settings_and_changes_checksum(changes):
    first_config = settings()
    first = workload_manifest(seeded_workload(first_config), length_unit="bytes", generator_settings=first_config)
    other_config = replace(first_config, **changes)
    other = workload_manifest(seeded_workload(other_config), length_unit="bytes", generator_settings=other_config)
    assert first["sha256"] != other["sha256"]
    assert first["generator"]["name"] == "uniform_interarrival"
    assert first["generator"]["rng"] == "python.random.Random"
    assert first["generator"]["python_version"]
    assert first["generator"]["settings"]["length_bytes_range"] == (500, 2000)


def test_manifest_hashes_settings_even_if_no_tasks_were_generated():
    config = settings(task_count=0)
    first = workload_manifest([], length_unit="bytes", generator_settings=config)
    other = workload_manifest([], length_unit="bytes", generator_settings=replace(config, seed=43))
    assert first["sha256"] != other["sha256"]


@pytest.mark.parametrize("field", ["task_count", "seed"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "4", None])
def test_invalid_count_or_seed(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        settings(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("interarrival_ms_range", (-1, 3)),
    ("interarrival_ms_range", (4, 3)),
    ("interarrival_ms_range", (0, float("inf"))),
    ("interarrival_ms_range", (0, float("nan"))),
    ("interarrival_ms_range", (True, 3)),
    ("interarrival_ms_range", (0,)),
    ("interarrival_ms_range", "0,3"),
    ("length_bytes_range", (0, 1000)),
    ("length_bytes_range", (1000, 500)),
    ("length_bytes_range", (1.5, 1000)),
    ("length_bytes_range", (True, 1000)),
    ("length_bytes_range", (1, 10**400)),
    ("data_size_kb_range", (0, 150)),
    ("data_size_kb_range", (-1, 150)),
    ("data_size_kb_range", (150, 20)),
    ("data_size_kb_range", (20, float("nan"))),
    ("data_size_kb_range", (20, float("inf"))),
    ("data_size_kb_range", (20, None)),
])
def test_invalid_generator_ranges(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        settings(**{field: value})


def test_settings_do_not_alias_caller_range_lists():
    bounds = [0, 3]
    config = settings(interarrival_ms_range=bounds)
    bounds[1] = 10
    assert config.interarrival_ms_range == (0, 3)


def test_accumulated_time_overflow_is_explicit():
    config = settings(task_count=4, interarrival_ms_range=(1e308, 1e308))
    with pytest.raises(ValueError, match="arrival timestamp"):
        seeded_workload(config)


def test_manifest_rejects_mismatched_count_or_units():
    config = settings(task_count=1)
    with pytest.raises(ValueError, match="task count"):
        workload_manifest([], length_unit="bytes", generator_settings=config)
    with pytest.raises(ValueError, match="bytes"):
        workload_manifest(seeded_workload(config), length_unit="unknown", generator_settings=config)
