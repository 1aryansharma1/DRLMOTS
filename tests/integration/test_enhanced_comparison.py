from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import pytest
import torch

from drlmots.enhanced_dqn import EnhancedDQNConfig
from drlmots.enhanced_comparison import (ComparisonPlan, MODES, POLICIES, compare_frozen,
    load_frozen_pair, paired_effects, summarize)
from drlmots.enhanced_sanity import SanityPlan, run_sanity
from drlmots.normalization import _digest
from drlmots.objective_model import failure_trace, generate_work, load_objective_model
from drlmots.objective_sim import MODEL_POLICIES, simulate_objectives
from drlmots.run_config import load_run_config


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen-study")
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    model.data["horizon_ms"] = 60
    model.data["workload"].update(task_count=4, interarrival_ms=[1, 2])
    settings = EnhancedDQNConfig(batch_size=2, warmup_transitions=2, replay_capacity=32,
        target_sync_updates=2, epsilon_decay_decisions=8)
    training = SanityPlan(training_episodes=2, validation_every_episodes=2,
        evaluation_workload_seeds=(81000,), evaluation_failure_seeds=(91000,))
    directory = root / "training"
    run_sanity(run, model, settings, training, directory, render_figures=False)
    plan = ComparisonPlan((82000, 82001), (92000, 92001))
    output = root / "comparison"
    original = deepcopy((run, model.manifest()))
    checkpoint_files = (directory / "initialization.json", directory / "trained-final.json")
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files]
    threads = torch.get_num_threads()
    result = compare_frozen(run, model, plan, directory, output)
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files]
    assert original == (run, model.manifest()) and torch.get_num_threads() == threads
    return run, model, plan, directory, output, result


def test_six_policies_share_work_and_fault_inputs_without_duplicated_no_fault_cases(experiment):
    run, model, plan, directory, output, result = experiment
    assert len(result["rows"]) == 36 and len(result["summaries"]) == 12
    for seed in plan.workload_seeds:
        group = [r for r in result["rows"] if r["workload_seed"] == seed]
        assert len({r["workload_sha256"] for r in group}) == 1
        for mode, failure in [("no_failures", None), *[("failures_with_recovery", f) for f in plan.failure_seeds]]:
            case = [r for r in group if r["mode"] == mode and r["failure_seed"] == failure]
            assert len(case) == 6 and {r["policy"] for r in case} == set(POLICIES)
            assert len({r["outage_sha256"] for r in case}) == 1
    assert result["frozen_audit"]["unchanged"] and result["frozen_audit"]["before"] == result["frozen_audit"]["after"]
    for row in result["rows"]:
        trace = json.loads((output / row["trace_file"]).read_text())
        assert trace["recovery_enabled"] is True
        assert trace["metrics"] == {k: v for k, v in row["metrics"].items() if k != "penalized_return"}
        if row["policy"].endswith("dqn"):
            assert trace["rl"]["learn"] is False and all(d["epsilon"] == 0 for d in trace["rl"]["decisions"])
            assert trace["rl"]["total_reward"] == pytest.approx(row["metrics"]["penalized_return"])
        else:
            assert "rl" not in trace


def test_common_return_is_independently_accounted_for_every_policy(experiment):
    _, model, _, _, _, result = experiment
    o = model.data["objective"]
    for row in result["rows"]:
        m = row["metrics"]
        n = m["submitted_count"]
        unsuccessful = sum(m[k + "_count"] for k in ("rejected", "execution_failure", "deadline_missed", "unfinished"))
        time = m["successful_makespan_ms"] if m["completed_count"] else model.data["horizon_ms"]
        penalty = 10 * max(0, o["minimum_completion_rate"] - m["completed_count"] / n)
        if m["deadline_eligible_count"]:
            penalty += 10 * max(0, o["minimum_deadline_success_rate"] - m["deadline_success_count"] / m["deadline_eligible_count"])
        cost = o["weights"]["time"] * time / o["scales"]["time_ms"] + o["weights"]["energy"] * m["total_energy_j"] / o["scales"]["energy_j"] + o["weights"]["failure"] * unsuccessful / n + penalty
        assert m["penalized_return"] == pytest.approx(-cost)


def test_native_heuristic_paths_preserve_original_full_report_hashes(experiment):
    run, model, _, _, _, result = experiment
    for row in result["rows"]:
        if row["policy"] not in MODEL_POLICIES:
            continue
        report = simulate_objectives(run, model, policy=row["policy"], tasks=generate_work(model, row["workload_seed"]),
            outages=[] if row["failure_seed"] is None else failure_trace(model, row["failure_seed"]),
            failure_seed=row["failure_seed"] or 0, recovery_enabled=True)
        assert _digest(report) == row["original_report_sha256"]


def test_failure_confidence_intervals_count_seed_clusters_not_crossed_runs(experiment):
    result = experiment[-1]
    for row in result["summaries"]:
        assert row["run_count"] == (2 if row["mode"] == "no_failures" else 4)
        assert row["uncertainty"]["completed_count"]["n"] == 2
    altered = deepcopy(result["rows"])
    baseline = {(r["mode"], r["workload_seed"], r["failure_seed"]): r for r in altered if r["policy"] == "untrained_dqn"}
    for row in altered:
        if row["policy"] == "trained_dqn" and row["mode"] == "failures_with_recovery":
            old = baseline[(row["mode"], row["workload_seed"], row["failure_seed"])]
            row["metrics"]["completed_count"] = old["metrics"]["completed_count"] + (2 if row["failure_seed"] == 92000 else 4)
    effect = next(e for e in paired_effects(altered) if e["mode"] == "failures_with_recovery" and e["baseline"] == "untrained_dqn")
    assert len(effect["pairs"]) == 4
    assert effect["uncertainty"]["completed_count"]["n"] == 2
    assert effect["uncertainty"]["completed_count"]["mean"] == 3


def test_infeasible_runs_cannot_win_by_losing_work_or_undefined_makespan(experiment):
    altered = deepcopy(experiment[-1]["rows"])
    row = next(r for r in altered if r["policy"] == "trained_dqn" and r["mode"] == "no_failures")
    row["objective"].update(feasible=False, score=None)
    row["metrics"].update(total_energy_j=0, completed_count=0, successful_makespan_ms=None, observed_unsuccessful_fraction=1)
    summary = next(r for r in summarize(altered) if r["policy"] == "trained_dqn" and r["mode"] == "no_failures")
    assert not summary["pareto_efficient"] and not summary["eligible_for_recommendation"]
    assert summary["objective_score_mean"] is None and summary["metric_contributing_runs"]["successful_makespan_ms"] == 1


def test_missing_timing_pairs_keep_undefined_intervals_instead_of_zero(experiment):
    altered = deepcopy(experiment[-1]["rows"])
    for row in altered:
        if row["policy"] == "trained_dqn" and row["mode"] == "failures_with_recovery":
            row["metrics"]["successful_makespan_ms"] = None
    for effect in paired_effects(altered):
        if effect["mode"] == "failures_with_recovery":
            assert effect["uncertainty"]["successful_makespan_ms"] == dict(mean=None, n=0, std=None, ci95=None)
            assert effect["uncertainty"]["completed_count"]["n"] == 2


def test_portable_report_bundle_contains_every_link_and_verified_artifact(experiment):
    output, result = experiment[-2:]
    html = (output / "comparison-report.html").read_text(encoding="utf-8")
    assert "<script" not in html and "{{" not in html
    assert "Information access differs" in html and "before delayed failure detection" in html
    for policy in ("Untrained DQN", "Trained DQN", "Least queue", "First fit", "Fog-first", "Multi-objective"):
        assert policy in html
    links = re.findall(r'(?:href|src)="([^"]+)"', html)
    with zipfile.ZipFile(output / "comparison-report.zip") as archive:
        assert archive.testzip() is None
        for target in links:
            assert (output / target).is_file() and target in archive.namelist()
        for path in archive.namelist():
            assert not Path(path).is_absolute() and ".." not in Path(path).parts
            assert archive.read(path) == (output / path).read_bytes()
        assert len([n for n in archive.namelist() if n.startswith("runs/")]) == 36
    manifest = json.loads((output / "artifacts.json").read_text())
    assert all(hashlib.sha256((output / path).read_bytes()).hexdigest() == checksum for path, checksum in manifest["sha256"].items())
    for filename, count in (("metrics.csv", 36), ("paired-deltas.csv", 30)):
        with (output / filename).open() as handle:
            assert len(list(csv.DictReader(handle))) == count
    assert len(list((output / "figures").glob("*.png"))) == 3
    assert len(list((output / "figures").glob("*.svg"))) == 3


@pytest.mark.parametrize("values", [(), (1, 1), (True,), (-1,)])
def test_invalid_seed_sets_are_rejected(values):
    with pytest.raises((ValueError, TypeError)):
        ComparisonPlan(workload_seeds=values)


@pytest.mark.parametrize("work,fail", [((10000,), (92000,)), ((81000,), (92000,)), ((81002,), (92000,)),
    ((82000,), (20001,)), ((82000,), (91000,))])
def test_training_and_sanity_diagnostic_leakage_is_rejected(work, fail, experiment):
    with pytest.raises(ValueError, match="disjoint"):
        load_frozen_pair(experiment[0], experiment[1], experiment[3], ComparisonPlan(work, fail))


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_plan_schema_is_strict(version):
    plan = ComparisonPlan().manifest()
    plan["schema_version"] = version
    with pytest.raises(ValueError, match="kind/version"):
        ComparisonPlan.parse(plan)


def test_plan_roundtrip_and_unknown_fields():
    plan = ComparisonPlan()
    assert ComparisonPlan.parse(json.loads(json.dumps(plan.manifest()))) == plan
    with pytest.raises(ValueError, match="documented"):
        ComparisonPlan.parse({**plan.manifest(), "select_best_model": True})


def test_output_protection_and_incompatible_model(experiment, tmp_path):
    run, model, plan, directory, _, _ = experiment
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(ValueError, match="empty output"):
        compare_frozen(run, model, plan, directory, tmp_path)
    assert sentinel.read_text() == "original"
    changed = deepcopy(model)
    changed.data["horizon_ms"] += 1
    with pytest.raises(ValueError, match="differs"):
        load_frozen_pair(run, changed, directory, plan)


@pytest.mark.parametrize("mutation", ["source", "missing_source", "fingerprint", "untrained_counter"])
def test_bad_provenance_or_modified_checkpoints_are_rejected(mutation, experiment, tmp_path):
    run, model, plan, directory, _, _ = experiment
    for name in ("initialization.json", "trained-final.json", "experiment.json", "result.json"):
        (tmp_path / name).write_bytes((directory / name).read_bytes())
    provenance = json.loads((tmp_path / "experiment.json").read_text())
    result = json.loads((tmp_path / "result.json").read_text())
    if mutation == "source":
        provenance["source_sha256"]["enhanced_dqn.py"] = "0" * 64
    elif mutation == "missing_source":
        provenance["source_sha256"].pop("objective_sim.py")
    elif mutation == "fingerprint":
        result["final_fingerprint"] = "0" * 64
    else:
        checkpoint = json.loads((tmp_path / "initialization.json").read_text())
        checkpoint["decisions"] = 1
        checkpoint.pop("sha256")
        checkpoint["sha256"] = _digest(checkpoint)
        result["initial_fingerprint"] = checkpoint["sha256"]
        (tmp_path / "initialization.json").write_text(json.dumps(checkpoint))
    (tmp_path / "experiment.json").write_text(json.dumps(provenance))
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load_frozen_pair(run, model, tmp_path, plan)


def test_mismatched_pair_hashes_are_rejected(experiment):
    altered = deepcopy(experiment[-1]["rows"])
    next(r for r in altered if r["policy"] == "untrained_dqn")["outage_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="input hashes"):
        paired_effects(altered)
