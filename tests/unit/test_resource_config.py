from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from drlmots.__main__ import configured_report
from drlmots.config import ConfigError
from drlmots.run_config import RunConfig, load_run_config, parse_run_config
from drlmots.workload import seeded_workload, workload_manifest


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "resources.yaml"


def spec():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_schema_two_effective_manifest_replays_every_constraint_and_result():
    config = load_run_config(CONFIG)
    manifest = config.manifest()
    assert manifest["schema_version"] == 2
    assert manifest["resource_units"] == {"cpu_units": "abstract CPU units", "memory_gb": "GB"}
    assert manifest["workload"]["required_resources"] == {"cpu_units": 2, "memory_gb": 4}
    effective = {key: value for key, value in manifest.items() if key != "sha256"}
    assert configured_report(parse_run_config(effective)) == configured_report(config)


@pytest.mark.parametrize("where", ["units", "capacities", "requirements"])
def test_missing_schema_two_fields_cannot_be_inferred(where):
    data = spec()
    if where == "units":
        del data["resource_units"]
    elif where == "capacities":
        del data["topology"]["vms"][0]["resource_capacities"]
    else:
        del data["workload"]["required_resources"]
    with pytest.raises(ConfigError):
        parse_run_config(data)


@pytest.mark.parametrize("value", [None, 0, True, "", " "])
def test_invalid_declared_units_rejected(value):
    data = spec()
    data["resource_units"]["cpu_units"] = value
    with pytest.raises(ConfigError, match="resource_units"):
        parse_run_config(data)


def test_missing_units_for_a_declared_capacity_or_demand_are_rejected():
    data = spec()
    del data["resource_units"]["memory_gb"]
    with pytest.raises(ConfigError, match="memory_gb"):
        parse_run_config(data)


@pytest.mark.parametrize("value", [0, -1, True, None, float("nan"), float("inf")])
def test_invalid_workload_resource_profile_rejected(value):
    data = spec()
    data["workload"]["required_resources"]["cpu_units"] = value
    with pytest.raises(ConfigError, match="required_resources"):
        parse_run_config(data)


def test_explicitly_unknown_capacities_are_not_made_up():
    data = spec()
    for vm in data["topology"]["vms"]:
        vm["resource_capacities"] = None
    report = configured_report(parse_run_config(data))
    assert report["completed_count"] == 0
    assert report["failed_count"] == 6
    assert report["resource_events"] == []
    assert all(balance is None for balance in report["available_resources"].values())


def test_units_capacities_and_profiles_do_not_alias_the_input_dict():
    data = spec()
    config = parse_run_config(data)
    data["resource_units"]["cpu_units"] = "different"
    data["topology"]["vms"][0]["resource_capacities"]["cpu_units"] = 100
    data["workload"]["required_resources"]["cpu_units"] = 99
    assert config.resource_units["cpu_units"] == "abstract CPU units"
    assert config.topology.vms[0].resource_capacities["cpu_units"] == 2
    assert config.workload.required_resources["cpu_units"] == 2
    tasks = seeded_workload(config.workload)
    tasks[0].required_resources["cpu_units"] = 50
    assert tasks[1].required_resources["cpu_units"] == 2
    assert config.workload.required_resources["cpu_units"] == 2


@pytest.mark.parametrize("change", ["units", "capacity", "demands"])
def test_resource_provenance_changes_with_each_effective_setting(change):
    data = spec()
    original = configured_report(parse_run_config(data))
    if change == "units":
        data["resource_units"]["cpu_units"] = "another abstract unit"
    elif change == "capacity":
        data["topology"]["vms"][0]["resource_capacities"]["cpu_units"] = 3
    else:
        data["workload"]["required_resources"]["cpu_units"] = 1
    other = configured_report(parse_run_config(data))
    assert other["run_configuration"]["sha256"] != original["run_configuration"]["sha256"]
    if change in ("units", "demands"):
        assert other["workload"]["sha256"] != original["workload"]["sha256"]


def test_schema_one_manifest_cannot_drop_resource_constraints():
    config = load_run_config(CONFIG)
    with pytest.raises(ConfigError, match="schema 2"):
        RunConfig(config.topology, config.workload, config.resource_units, schema_version=1)


def test_demand_profiles_do_not_change_arrival_length_or_size_draws():
    config = load_run_config(CONFIG)
    first = seeded_workload(config.workload)
    second = seeded_workload(replace(config.workload, required_resources={}))
    assert [(task.arrival_time, task.length, task.data_size_kb) for task in first] == [
        (task.arrival_time, task.length, task.data_size_kb) for task in second
    ]


def test_manifest_rejects_missing_units_or_inconsistent_generator_demands():
    config = load_run_config(CONFIG)
    tasks = seeded_workload(config.workload)
    with pytest.raises(ValueError, match="resource_units"):
        workload_manifest(tasks, length_unit="bytes", generator_settings=config.workload)
    tasks[0].required_resources["cpu_units"] = 1
    with pytest.raises(ValueError, match="resource profile"):
        workload_manifest(tasks, length_unit="bytes", generator_settings=config.workload,
                          resource_units=config.resource_units)


def test_empty_generated_profile_still_requires_declared_units():
    config = load_run_config(CONFIG)
    empty_settings = replace(config.workload, task_count=0)
    with pytest.raises(ValueError, match="resource_units"):
        workload_manifest([], length_unit="bytes", generator_settings=empty_settings)
