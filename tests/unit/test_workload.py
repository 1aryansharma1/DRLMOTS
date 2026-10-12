from copy import deepcopy

import pytest

from drlmots.workload import fixed_workload, processing_time_ms, workload_manifest


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
