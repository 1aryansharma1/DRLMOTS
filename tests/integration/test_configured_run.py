import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from drlmots.__main__ import configured_report
from drlmots.run_config import load_run_config, parse_run_config


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "synthetic.yaml"


def test_configured_example_matches_exact_hand_calculation():
    config = load_run_config(CONFIG)
    report = configured_report(config)
    assert report == configured_report(config)
    assert report["completed_count"] == 6
    assert report["failed_count"] == 0
    assert report["makespan_ms"] == 4
    assert report["average_completion_time_ms"] == 3
    assert [item["start_time_ms"] for item in report["outcomes"]] == [0, 0, 0, 2, 2, 2]
    assert [item["finish_time_ms"] for item in report["outcomes"]] == [2, 2, 2, 4, 4, 4]
    assert len({item["vm_id"] for item in report["outcomes"]}) == 3
    assert report["topology_counts"] == {
        "hosts": 3, "vms": 3, "fog_nodes": 2, "cloud_datacenters": 1,
        "access_points": 1, "end_nodes": 1,
    }
    assert all(vm.queue == [] for vm in config.topology.vms)


def test_saved_effective_configuration_reproduces_the_same_run():
    report = configured_report(load_run_config(CONFIG))
    effective = {key: value for key, value in report["run_configuration"].items() if key != "sha256"}
    assert configured_report(parse_run_config(effective)) == report


def test_cli_saves_configured_run(tmp_path):
    output = tmp_path / "configured.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(CONFIG), "--output", str(output)],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["completed_count"] == 6
    assert report["topology_counts"]["vms"] == 3
    assert report["run_configuration"]["sha256"]


def test_changing_queue_capacity_changes_observed_overflow():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["workload"]["task_count"] = 9
    first = configured_report(parse_run_config(data))
    assert (first["completed_count"], first["failed_count"]) == (6, 3)
    for vm in data["topology"]["vms"]:
        vm["queue_capacity"] = 2
    second = configured_report(parse_run_config(data))
    assert (second["completed_count"], second["failed_count"]) == (9, 0)
    assert second["makespan_ms"] == 6
    assert first["run_configuration"]["sha256"] != second["run_configuration"]["sha256"]


@pytest.mark.parametrize("flags", [
    ["--seed", "42"], ["--task-count", "10"], ["--length-bytes", "500", "1000"],
])
def test_config_and_workload_flags_cannot_silently_override_each_other(flags, tmp_path):
    output = tmp_path / "invalid.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(CONFIG), *flags,
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
    assert not output.exists()


def test_invalid_topology_produces_cli_error_without_output(tmp_path):
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data["topology"]["vms"][0]["queue_capacity"] = None
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    output = tmp_path / "invalid.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(path), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "queue_capacity" in result.stderr
    assert not output.exists()


def test_missing_config_is_a_readable_cli_error(tmp_path):
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(tmp_path / "missing.yaml")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
