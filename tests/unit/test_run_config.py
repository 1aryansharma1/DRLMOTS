from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from drlmots.config import ConfigError
from drlmots.run_config import load_run_config, parse_run_config
from drlmots.topology import build_topology


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "synthetic.yaml"


def spec():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_explicit_topology_matches_configured_memberships_and_resources():
    config = load_run_config(CONFIG)
    topology = config.topology
    assert len(topology.hosts) == 3
    assert len(topology.vms) == 3
    assert len(topology.fog_nodes) == 2
    assert len(topology.cloud_datacenters) == 1
    assert topology.end_nodes[0].fog_node_ids == ("fog-a", "fog-b")
    assert [vm.queue_capacity for vm in topology.vms] == [1, 1, 1]
    assert config.workload.task_count == 6


def test_builder_does_not_alias_or_mutate_input_records():
    data = spec()
    original = deepcopy(data)
    config = parse_run_config(data)
    assert data == original
    data["topology"]["hosts"][0]["memory_gb"] = 99
    data["topology"]["fog_nodes"][0]["vm_ids"].append("other")
    assert config.topology.hosts[0].memory_gb == 32
    assert config.topology.fog_nodes[0].vm_ids == ("vm-fog-a",)


def test_effective_config_is_reusable_and_ignores_yaml_formatting_or_path(tmp_path):
    first = load_run_config(CONFIG).manifest()
    payload = {key: value for key, value in first.items() if key != "sha256"}
    assert parse_run_config(payload).manifest() == first
    reformatted = tmp_path / "elsewhere.yaml"
    reformatted.write_text(yaml.safe_dump(spec(), sort_keys=True), encoding="utf-8")
    assert load_run_config(reformatted).manifest() == first
    first["topology"]["vms"][0]["queue_capacity"] = 50
    assert load_run_config(CONFIG).topology.vms[0].queue_capacity == 1


def test_changed_effective_settings_change_config_checksum():
    data = spec()
    original = parse_run_config(data).manifest()
    data["topology"]["vms"][0]["queue_capacity"] = 2
    assert parse_run_config(data).manifest()["sha256"] != original["sha256"]
    data = spec()
    data["workload"]["seed"] = 7
    assert parse_run_config(data).manifest()["sha256"] != original["sha256"]


@pytest.mark.parametrize("group", [
    "hosts", "vms", "fog_nodes", "cloud_datacenters", "access_points", "end_nodes"
])
def test_missing_explicit_topology_groups_rejected(group):
    data = spec()
    del data["topology"][group]
    with pytest.raises(ConfigError, match="topology"):
        parse_run_config(data)


@pytest.mark.parametrize("field", ["vm_id", "tier", "bandwidth_reported", "queue_capacity"])
def test_missing_vm_fields_rejected(field):
    data = spec()
    del data["topology"]["vms"][0][field]
    with pytest.raises(ConfigError, match="vms"):
        parse_run_config(data)


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5])
def test_invalid_queue_limits_rejected(value):
    data = spec()
    data["topology"]["vms"][0]["queue_capacity"] = value
    with pytest.raises(ConfigError, match="queue_capacity"):
        parse_run_config(data)


@pytest.mark.parametrize("value", [None, "vms", {}, [None]])
def test_invalid_vm_list_structure_rejected(value):
    data = spec()
    data["topology"]["vms"] = value
    with pytest.raises(ConfigError, match="vms"):
        parse_run_config(data)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 1.0), ("schema_version", 3),
    ("kind", "paper"), ("kind", None),
])
def test_wrong_schema_or_kind_rejected(field, value):
    data = spec()
    data[field] = value
    with pytest.raises(ConfigError):
        parse_run_config(data)


@pytest.mark.parametrize("field,value", [
    ("length_unit", "instructions"), ("arrival_time_unit", "seconds"),
    ("generator", "poisson"), ("seed", None), ("task_count", True),
])
def test_invalid_units_generator_or_settings_rejected(field, value):
    data = spec()
    data["workload"][field] = value
    with pytest.raises(ConfigError):
        parse_run_config(data)


def test_unknown_fields_cannot_be_silently_ignored():
    for destination in ("root", "topology", "vm", "workload"):
        data = spec()
        target = {"root": data, "topology": data["topology"],
                  "vm": data["topology"]["vms"][0], "workload": data["workload"]}[destination]
        target["typo"] = 1
        with pytest.raises(ConfigError, match="exactly"):
            parse_run_config(data)


@pytest.mark.parametrize("case,error", [
    ("duplicate", "duplicate vm_id"), ("tier", "wrong tier"),
    ("invalid_tier", "tier must"), ("unknown_vm", "unknown VM"),
    ("unknown_host", "unknown host"), ("unknown_end", "unknown end node"),
    ("unknown_fog", "unknown fog node"), ("orphan", "exactly one node"),
])
def test_loaded_topology_retains_existing_invariants(case, error):
    data = spec()
    topology = data["topology"]
    if case == "duplicate":
        topology["vms"].append(deepcopy(topology["vms"][0]))
    elif case == "tier":
        topology["vms"][0]["tier"] = "cloud"
    elif case == "invalid_tier":
        topology["vms"][0]["tier"] = "edge"
    elif case == "unknown_vm":
        topology["fog_nodes"][0]["vm_ids"] = ["missing"]
    elif case == "unknown_host":
        topology["fog_nodes"][0]["host_ids"] = ["missing"]
    elif case == "unknown_end":
        topology["access_points"][0]["end_node_id"] = "missing"
    elif case == "unknown_fog":
        topology["end_nodes"][0]["fog_node_ids"] = ["missing"]
    elif case == "orphan":
        topology["hosts"].append({"host_id": "orphan", "storage_tb": 3,
                                  "memory_gb": 32, "bandwidth_reported": 1000})
    with pytest.raises(ConfigError, match=error):
        parse_run_config(data)


def test_explicit_empty_topology_requires_empty_workload():
    data = spec()
    data["topology"] = {name: [] for name in data["topology"]}
    with pytest.raises(ConfigError, match="at least one VM"):
        parse_run_config(data)
    data["workload"]["task_count"] = 0
    assert parse_run_config(data).topology.vms == ()


@pytest.mark.parametrize("content,error", [
    ("schema_version: [", "invalid run YAML"),
    ("schema_version: 1\nschema_version: 2", "duplicate run YAML key"),
    ("true: 1", "keys must be strings"),
    ("!!python/object:danger {}", "invalid run YAML"),
])
def test_invalid_or_ambiguous_yaml_rejected(content, error, tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=error):
        load_run_config(path)


def test_paper_planning_config_is_not_a_runnable_baseline():
    with pytest.raises(ConfigError, match="run configuration"):
        load_run_config(CONFIG.with_name("default.yaml"))


def test_builder_rejects_non_mapping():
    with pytest.raises(TypeError, match="topology must be a mapping"):
        build_topology(None)
