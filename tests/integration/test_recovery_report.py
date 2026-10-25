from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import pytest

from drlmots.__main__ import configured_report
from drlmots.efficiency import load_objectives
from drlmots.entities import Task
from drlmots.recovery import RECOVERY_POLICIES, simulate_recovery
from drlmots.recovery_config import Outage, load_recovery_config
from drlmots.recovery_report import compare_recovery
from drlmots.run_config import load_run_config
from drlmots.train import seeded_config

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    return (load_run_config(ROOT / "configs/rl_network.yaml"),
            load_objectives(ROOT / "configs/objectives.yaml"),
            load_recovery_config(ROOT / "configs/failure_recovery.yaml"))


def test_correlated_node_failure_interrupts_all_its_vms_and_queued_tasks():
    config, objective, recovery = fixture()
    recovery = replace(recovery, outages=(Outage("fog:fog-a", 1, 20),), max_retries=0)
    tasks = [Task(str(i), 0, 5000, 1, {"cpu_units": 2, "memory_gb": 4}) for i in range(12)]
    result = simulate_recovery(config, objective, recovery, tasks=tasks, policy="first_fit_fcfs")
    failed = [a for a in result["attempts"] if a["status"] == "interrupted"]
    assert {a["vm_id"] for a in failed} == {"fog-a1", "fog-a2"}
    assert len(failed) == 4  # two running + two waiting/uploaded
    assert all(a["end_ms"] == 1 for a in failed)
    assert result["metrics"]["permanent_failure_count"] == 4
    for vm in config.topology.vms:
        assert result["available_resources"][vm.vm_id] == vm.resource_capacities


@pytest.mark.parametrize("policy", RECOVERY_POLICIES)
def test_random_workloads_are_reproducible_and_fully_accounted(policy):
    config, objective, recovery = fixture()
    before_rng = random.getstate()
    for seed in range(5300, 5310):
        current = seeded_config(config, seed)
        result = simulate_recovery(current, objective, recovery, policy=policy)
        assert result == simulate_recovery(current, objective, recovery, policy=policy)
        metrics = result["metrics"]
        assert metrics["completed_count"] + metrics["rejected_count"] + metrics["permanent_failure_count"] == 100
        assert metrics["total_energy_j"] == pytest.approx(sum(metrics[key] for key in (
            "compute_energy_j", "transmission_energy_j", "idle_energy_j", "recovery_energy_j")))
        assert sum(r["attributed_energy_j"] for r in result["outcomes"]) + metrics["idle_energy_j"] == pytest.approx(metrics["total_energy_j"])
        assert len({r["task_id"] for r in result["outcomes"]}) == 100
        attempts = {a["attempt_id"]: a for a in result["attempts"]}
        for row in result["outcomes"]:
            assert row["attempt_elapsed_ms"] + row["recovery_blocked_ms"] == pytest.approx(row["latency_ms"])
            assert len(row["attempt_ids"]) <= recovery.max_retries + 1
            own = [attempts[key] for key in row["attempt_ids"]]
            assert all(a["node_id"] != b["node_id"] for a, b in zip(own, own[1:]))
            assert all(a["status"] != "completed" for a in own[:-1])
        for decision in result["decisions"]:
            own = attempts[decision["attempt_id"]]
            assert own["node_id"] in decision["observed_up_nodes"]
            assert own["node_id"] != decision["excluded_node"]
            assert "outage_trace" not in decision
    assert random.getstate() == before_rng


def test_future_outage_changes_do_not_change_earlier_decisions():
    config, objective, recovery = fixture()
    early = simulate_recovery(config, objective, replace(recovery, outages=()))
    late = simulate_recovery(config, objective, replace(recovery, outages=(Outage("fog:fog-a", 80, 100),)))
    assert [d for d in early["decisions"] if d["time_ms"] < 80] == [d for d in late["decisions"] if d["time_ms"] < 80]


def test_report_pairing_artifacts_and_actual_recovery_results(tmp_path):
    pytest.importorskip("matplotlib")
    config, objective, recovery = fixture()
    result = compare_recovery(config, objective, recovery, (5200, 5201), tmp_path)
    assert len(result["rows"]) == 24
    assert len(result["summaries"]) == 12
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []
        def handle_starttag(self, tag, attrs):
            self.paths.extend(value for key, value in attrs if key in ("href", "src"))
    parser = Links()
    parser.feed((tmp_path / "recovery-report.html").read_text(encoding="utf-8"))
    assert all((tmp_path / name).is_file() for name in parser.paths)
    for seed in (5200, 5201):
        records = [r for r in result["rows"] if r["seed"] == seed]
        assert len({r["workload_sha256"] for r in records}) == 1
        assert len({r["outage_sha256"] for r in records if r["mode"] != "no_failures"}) == 1
    assert any(r["metrics"]["recovered_task_count"] > 0 for r in result["rows"])
    assert all(r["metrics"]["recovered_task_count"] == 0 for r in result["rows"] if r["mode"] != "failures_with_recovery")
    assert (tmp_path / "recovery-report.zip").is_file()
    with zipfile.ZipFile(tmp_path / "recovery-report.zip") as archive:
        assert archive.testzip() is None
        assert all(name in archive.namelist() for name in parser.paths)
        assert "metrics.csv" in archive.namelist()
    with pytest.raises(ValueError, match="different experiment"):
        compare_recovery(config, objective, replace(recovery, max_retries=3), (5200, 5201), tmp_path)


def test_cli_invalid_config_is_clear_without_writing_results(tmp_path):
    result = subprocess.run([sys.executable, "-m", "drlmots.recovery_report", "--config", "configs/rl_network.yaml",
        "--objectives", "configs/objectives.yaml", "--recovery-config", "missing-recovery.yaml",
        "--output-dir", str(tmp_path / "output")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "error:" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("policy", RECOVERY_POLICIES[:3])
def test_preserved_six_vm_baseline_is_unchanged_without_failures(policy):
    config, objective, recovery = fixture()
    for seed in (5200, 5201):
        config = seeded_config(config, seed)
        old = configured_report(config, policy=policy)
        new = simulate_recovery(config, objective, replace(recovery, outages=()), policy=policy)
        assert old["completed_count"] == new["metrics"]["completed_count"]
        assert old["makespan_ms"] == pytest.approx(new["metrics"]["makespan_ms"])
        attempts = {a["task_id"]: a for a in new["attempts"]}
        for previous, outcome in zip(old["outcomes"], new["outcomes"]):
            assert previous["finish_time_ms"] == outcome["finish_time_ms"]
            if outcome["status"] == "completed":
                assert previous["vm_id"] == attempts[outcome["task_id"]]["vm_id"]
