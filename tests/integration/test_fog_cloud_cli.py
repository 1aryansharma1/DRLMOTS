from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drlmots.__main__ import configured_report
from drlmots.demo import demo_scenarios, generate_demo
from drlmots.evaluation import compare_policies
from drlmots.normalization import capture_batch
from drlmots.run_config import load_run_config, parse_run_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/fog_cloud.yaml"


def test_real_network_report_forecasts_accounting_and_pairing():
    config = load_run_config(CONFIG)
    before = deepcopy(config)
    comparison = compare_policies(config)
    assert comparison["policies"] == ["least_queue_fcfs", "first_fit_fcfs", "fog_first_completion"]
    for report in comparison["reports"]:
        assert report["completed_count"] + report["failed_count"] == 100
        assert report["network_delay_ms"] is None
        assert len(report["placement"]["decisions"]) == 100
        assert report["workload"]["sha256"] == comparison["workload_sha256"]
        assert report["run_configuration"]["sha256"] == comparison["configuration_sha256"]
        outcomes = {item["task_id"]: item for item in report["outcomes"]}
        for decision in report["placement"]["decisions"]:
            if decision["selected_vm_id"] is None:
                continue
            selected = next(item for item in decision["candidates"] if item["vm_id"] == decision["selected_vm_id"])
            assert selected["predicted_finish_time_ms"] == pytest.approx(outcomes[decision["task_id"]]["finish_time_ms"])
            if report["policy"] == "fog_first_completion":
                best = min(decision["candidates"], key=lambda item: (item["predicted_finish_time_ms"], item["tier"] == "cloud", item["vm_id"]))
                assert selected == best
        for vm in config.topology.vms:
            events = [event for event in report["resource_events"] if event["vm_id"] == vm.vm_id]
            assert len(events) % 2 == 0
            assert [event["action"] for event in events] == ["reserve", "release"] * (len(events) // 2)
    assert config == before
    report = configured_report(config)
    assert report["policy"] == "fog_first_completion"
    manifest = report["run_configuration"].copy()
    manifest.pop("sha256")
    assert configured_report(parse_run_config(manifest)) == report
    assert any(d["reason"] == "fog_no_slower_than_cloud" for d in report["placement"]["decisions"])


def test_old_baselines_and_legacy_schema_remain_zero_network():
    config = load_run_config(ROOT / "configs/resources.yaml")
    report = configured_report(config)
    assert (report["completed_count"], report["failed_count"], report["makespan_ms"]) == (4, 2, 4)
    assert report["network_delay_ms"] == 0 and "placement" not in report
    assert compare_policies(config)["policies"] == ["least_queue_fcfs", "first_fit_fcfs"]
    assert report["run_configuration"]["sha256"] == "76e284813d5c9166d3fef29c454ce08d28d10acaf549dbc9a0b9bc84ec2b520f"


def test_legacy_learning_path_rejects_network_configs():
    with pytest.raises(ValueError, match="new state/reward contract"):
        capture_batch(load_run_config(CONFIG), split="train")


def test_network_scenarios_keep_identical_links_for_every_policy():
    scenarios = demo_scenarios(load_run_config(CONFIG))
    assert len(scenarios) == 3
    for scenario in scenarios:
        reports = scenario["comparison"]["reports"]
        assert len(reports) == 3
        assert all(report["placement"]["configuration"] == reports[0]["placement"]["configuration"] for report in reports)


def test_network_demo_contains_three_real_policies_and_network_figures(tmp_path):
    pytest.importorskip("matplotlib")
    path = generate_demo(CONFIG, tmp_path)
    html = path.read_text(encoding="utf-8")
    bundle = json.loads((tmp_path / "comparison.json").read_text())
    assert bundle["default_policy"] == "fog_first_completion"
    assert set(bundle["policy_labels"]) == {"least_queue_fcfs", "first_fit_fcfs", "fog_first_completion"}
    assert all(len(s["comparison"]["reports"]) == 3 for s in bundle["scenarios"])
    assert "placement-details" in html and "Best fog total ms" in html
    assert len(list((tmp_path / "figures").glob("*.png"))) == 21
    assert len(list((tmp_path / "figures").glob("*.svg"))) == 21


def test_cli_uses_network_config_and_default_fog_policy(tmp_path):
    output = tmp_path / "network.json"
    args = [sys.executable, "-m", "drlmots", "--config", str(CONFIG), "--output", str(output)]
    first = subprocess.run(args, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    report = json.loads(output.read_text())
    assert report["policy"] == "fog_first_completion" and report["placement"]["transfers"]
    before = output.read_bytes()
    second = subprocess.run(args, capture_output=True, text=True)
    assert second.returncode == 0 and before == output.read_bytes()


@pytest.mark.parametrize("extra", [["--capture-decisions"], ["--reward-config", str(ROOT / "configs/reward.yaml")]])
def test_cli_rejects_legacy_capture_without_creating_report(tmp_path, extra):
    output = tmp_path / "bad.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(CONFIG),
                             "--output", str(output), *extra], capture_output=True, text=True)
    assert result.returncode != 0 and "new state/reward contract" in result.stderr
    assert not output.exists()


def test_cli_fog_policy_requires_explicit_network_settings(tmp_path):
    result = subprocess.run([sys.executable, "-m", "drlmots", "--policy", "fog_first_completion"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "requires schema-3" in result.stderr
