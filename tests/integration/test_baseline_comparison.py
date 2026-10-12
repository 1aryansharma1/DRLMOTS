from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drlmots.entities import Host, ResourceTier, Task, VM
from drlmots.environment import simulate
from drlmots.evaluation import compare_policies
from drlmots.run_config import load_run_config, parse_run_config
from drlmots.topology import FogNode, Topology


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "resources.yaml"


def test_hand_calculated_policy_difference_and_unchanged_inputs():
    vms = (VM("b", ResourceTier.FOG, 200, queue_capacity=2),
           VM("a", ResourceTier.FOG, 200, queue_capacity=2))
    topology = Topology(hosts=(Host("host", 3, 32, 1000),), vms=vms,
                        fog_nodes=(FogNode("fog", ("host",), ("b", "a")),))
    tasks = [Task("long", 0, 5000, 20), Task("short-1", 0, 1000, 20), Task("short-2", 0, 1000, 20)]
    original = deepcopy((topology, tasks))
    least = simulate(topology, tasks, task_length_unit="bytes")
    first = simulate(topology, tasks, task_length_unit="bytes", policy="first_fit_fcfs")
    assert [item.vm_id for item in least.outcomes] == ["a", "b", "a"]
    assert [item.finish_time_ms for item in least.outcomes] == [10, 2, 12]
    assert [item.vm_id for item in first.outcomes] == ["a", "a", "a"]
    assert [item.finish_time_ms for item in first.outcomes] == [10, 12, 14]
    assert (least.makespan_ms, first.makespan_ms) == (12, 14)
    assert (topology, tasks) == original


def test_unknown_policy_rejected_even_for_empty_workload():
    with pytest.raises(ValueError, match="unknown policy"):
        simulate(Topology((), ()), [], task_length_unit="bytes", policy="typo")


@pytest.mark.parametrize("policies", [(), ("least_queue_fcfs", "least_queue_fcfs"), ("unknown",)])
def test_invalid_comparison_policies(policies):
    with pytest.raises(ValueError):
        compare_policies(load_run_config(CONFIG), policies)


def test_paired_inputs_resource_safety_and_policy_replay():
    config = load_run_config(CONFIG)
    original = deepcopy(config)
    comparison = compare_policies(config)
    for report in comparison["reports"]:
        assert report["workload"]["sha256"] == comparison["workload_sha256"]
        assert report["run_configuration"]["sha256"] == comparison["configuration_sha256"]
        assert (report["completed_count"], report["failed_count"]) == (4, 2)
        assert report["average_waiting_time_ms"] == 1
        assert sum(vm["busy_ms"] for vm in report["vm_usage"]) == 8
        assert [vm["occupancy_pct"] for vm in report["vm_usage"]] == [100, 100, 0]
        assert len(report["resource_events"]) == 8
        assert report["available_resources"] == {vm.vm_id: vm.resource_capacities for vm in config.topology.vms}
        manifest = dict(report["run_configuration"])
        manifest.pop("sha256")
        replay = compare_policies(parse_run_config(manifest), (report["policy"],))
        assert replay["reports"][0] == report
    assert compare_policies(config) == comparison
    assert config == original


@pytest.mark.parametrize("count,profile", [(0, {"cpu_units": 2, "memory_gb": 4}), (3, {"cpu_units": 100})])
def test_no_success_has_zero_occupancy_and_null_wait(count, profile):
    config = load_run_config(CONFIG)
    config = replace(config, workload=replace(config.workload, task_count=count, required_resources=profile))
    for report in compare_policies(config)["reports"]:
        assert report["completed_count"] == 0
        assert report["failed_count"] == count
        assert report["makespan_ms"] == 0
        assert report["average_waiting_time_ms"] is None
        assert all(vm["occupancy_pct"] == 0 for vm in report["vm_usage"])


def test_actual_cli_policy_parameter_is_recorded(tmp_path):
    output = tmp_path / "first-fit.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(CONFIG),
                             "--policy", "first_fit_fcfs", "--output", str(output)],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["policy"] == "first_fit_fcfs"
    assert (report["completed_count"], report["failed_count"]) == (4, 2)
