from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import zipfile

import pytest
import torch

from drlmots.dqn import TrainingConfig
from drlmots.efficiency import (aggregate, load_objectives, metrics_for, paired_changes,
                                pareto_policies, run_experiment)
from drlmots.normalization import _digest
from drlmots.objectives import evaluate_objectives
from drlmots.run_config import load_run_config
from drlmots.train import (checkpoint_artifact, continue_training, cpu_execution, initialize,
                           load_checkpoint, run_episode, save_checkpoint, seeded_config)

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    config = load_run_config(ROOT / "configs/rl_network.yaml")
    config = replace(config, workload=replace(config.workload, task_count=8))
    objective = load_objectives(ROOT / "configs/objectives.yaml")
    settings = TrainingConfig(episodes=2, calibration_episodes=1, batch_size=4, replay_capacity=32,
                              hidden_widths=(8, 8), evaluation_seeds=(3000, 3001))
    return config, objective, settings


@pytest.fixture(autouse=True)
def single_thread():
    with cpu_execution():
        yield


def test_live_objectives_match_realized_task_accounting_and_update_weights(tmp_path):
    config, objective, settings = fixture()
    agent = initialize(config, replace(settings, multi_objective=objective), "standard")
    before = _digest(agent.online.artifact()["weights"])
    summary = continue_training(agent)
    assert summary["updates"] > 0 and before != _digest(agent.online.artifact()["weights"])
    fingerprint = checkpoint_artifact(agent)["sha256"]
    report = run_episode(agent, seeded_config(config, 3000))
    evidence = evaluate_objectives(report, objective)
    assert checkpoint_artifact(agent)["sha256"] == fingerprint
    for decision, actual in zip(report["rl"]["decisions"], evidence["tasks"]):
        assert decision["task_id"] == actual["task_id"]
        assert decision["reward"] == pytest.approx(actual["utility"])
        if actual["admitted"]:
            assert decision["objectives"]["total_energy_j"] == pytest.approx(actual["total_energy_j"])
            assert decision["objectives"]["failure_risk"] == pytest.approx(actual["failure_risk"])
    metrics = evidence["metrics"]
    assert metrics["total_energy_j"] == pytest.approx(metrics["compute_energy_j"] + metrics["transmission_energy_j"])
    assert metrics["expected_successful_tasks"] + metrics["expected_execution_failures"] == pytest.approx(report["completed_count"])
    assert metrics["expected_success_rate_all_arrivals"] <= metrics["completion_rate"]
    path = tmp_path / "checkpoint.json"
    save_checkpoint(agent, path)
    loaded = load_checkpoint(path, for_resume=True)
    assert checkpoint_artifact(loaded)["sha256"] == fingerprint


def test_multiobjective_resume_is_exact_and_coefficients_are_context_bound(tmp_path):
    config, objective, settings = fixture()
    settings = replace(settings, multi_objective=objective)
    whole = initialize(config, settings, "fog_preferred")
    continue_training(whole)
    partial = initialize(config, settings, "fog_preferred")
    continue_training(partial, target_episodes=1)
    path = tmp_path / "checkpoint.json"
    save_checkpoint(partial, path)
    resumed = load_checkpoint(path, for_resume=True)
    continue_training(resumed)
    assert checkpoint_artifact(resumed)["sha256"] == checkpoint_artifact(whole)["sha256"]
    data = checkpoint_artifact(whole)
    data["training_config"]["multi_objective"]["vms"][0]["active_power_w"] += 1
    data["sha256"] = _digest({key: value for key, value in data.items() if key != "sha256"})
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="context"):
        load_checkpoint(path)


def test_all_rejected_and_empty_workloads_do_not_fabricate_efficiency():
    config, objective, settings = fixture()
    agent = initialize(config, settings, "standard")
    impossible = replace(config, workload=replace(config.workload, required_resources={"cpu_units": 99, "memory_gb": 4}))
    report = run_episode(agent, impossible)
    metrics = evaluate_objectives(report, objective)["metrics"]
    assert metrics["total_energy_j"] == metrics["expected_success_rate_all_arrivals"] == 0
    assert metrics["energy_per_expected_success_j"] is None
    assert metrics["mean_all_arrival_utility"] == -1
    empty = run_episode(agent, replace(config, workload=replace(config.workload, task_count=0)))
    values = evaluate_objectives(empty, objective)["metrics"]
    assert values["completion_rate"] is values["mean_all_arrival_utility"] is None


def test_complete_report_runs_with_paired_inputs_valid_links_and_resume(tmp_path):
    pytest.importorskip("matplotlib")
    config, objective, settings = fixture()
    result = run_experiment(config, settings, objective, tmp_path, model_seeds=(42, 43), scenarios=("normal", "burst"))
    assert len(result["rows"]) == 2 * 2 * 2 * 7
    assert len(result["models"]) == 6
    for summary in result["summaries"]:
        if summary["policy"].endswith("dqn"):
            assert summary["metrics"]["completion_rate"]["model_seed_std"] is not None
        else:
            assert summary["model_seeds"] == []
            assert summary["metrics"]["completion_rate"]["model_seed_std"] is None
    for trace in {r["trace_file"] for r in result["rows"]}:
        reports = json.loads((tmp_path / trace).read_text())["reports"]
        assert len({r["workload"]["sha256"] for r in reports}) == 1
        assert len({r["run_configuration"]["sha256"] for r in reports}) == 1
        assert len({r["multi_objective_evaluation"]["model"]["energy_scale_j"] for r in reports}) == 1
        for report in reports[-3:]:
            assert all(r["epsilon"] == 0 for r in report["rl"]["decisions"])
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []
        def handle_starttag(self, tag, attrs):
            self.paths.extend(value for key, value in attrs if key in ("src", "href"))
    links = Links()
    links.feed((tmp_path / "efficiency-report.html").read_text(encoding="utf-8"))
    assert all((tmp_path / link).is_file() for link in links.paths)
    (tmp_path / "secret.txt").write_text("not an artifact")
    resumed = run_experiment(config, settings, objective, tmp_path, model_seeds=(42, 43), scenarios=("normal", "burst"), resume=True)
    assert resumed["rows"] == result["rows"]
    with zipfile.ZipFile(tmp_path / "efficiency-report.zip") as archive:
        assert archive.testzip() is None
        assert "secret.txt" not in archive.namelist()
        assert "models/balanced_dqn-42/history.json" in archive.namelist()
        assert "models/balanced_dqn-42/last-training-episode.json" in archive.namelist()
        assert "efficiency-report.html" in archive.namelist()
        assert all(not name.startswith(("/", "..")) for name in archive.namelist())
    with pytest.raises(ValueError, match="fresh"):
        run_experiment(config, settings, objective, tmp_path, model_seeds=(42, 43), scenarios=("normal", "burst"))
    with pytest.raises(ValueError, match="different experiment"):
        run_experiment(config, settings, replace(objective, energy_scale_j=.2), tmp_path, resume=True)


def test_aggregate_separates_initialization_and_workload_variation_and_signed_changes():
    config, objective, settings = fixture()
    agent = initialize(config, settings, "standard")
    report = run_episode(agent, seeded_config(config, 3000))
    metrics = metrics_for(report, objective, settings)
    rows = []
    for model in (42, 43):
        for workload in (3000, 3001):
            for policy in ("least_queue_fcfs", "balanced_dqn"):
                values = dict(metrics)
                values["total_energy_j"] = 10 if policy == "least_queue_fcfs" else (12 if model == 42 else 16)
                rows.append({"scenario": "normal", "policy": policy, "model_seed": model,
                             "workload_seed": workload, "metrics": values})
    changes = [r for r in paired_changes(rows) if r["policy"] == "balanced_dqn" and r["metric"] == "total_energy_j"]
    assert [r["improvement_pct"] for r in changes] == [-20, -20, -60, -60]
    summary = next(r for r in aggregate(rows) if r["policy"] == "balanced_dqn")
    assert summary["metrics"]["total_energy_j"]["mean"] == 14
    assert summary["metrics"]["total_energy_j"]["model_seed_std"] == pytest.approx(2.8284271247)
    assert summary["metrics"]["total_energy_j"]["workload_seed_std"] == 0
    assert pareto_policies(aggregate(rows), "normal")
