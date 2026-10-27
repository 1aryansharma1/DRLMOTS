from copy import deepcopy
from dataclasses import replace
import gzip
import json
import hashlib
import re
import zipfile

import pytest

from drlmots.enhanced_dqn import EnhancedDQNConfig
from drlmots.normalization import _digest
from drlmots.objective_model import load_objective_model
from drlmots.predictive_dqn import PredictiveDQN
from drlmots.predictive_training import TrainingPlan, run_study, validation_rank
from drlmots.run_config import load_run_config


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    model.data["horizon_ms"] = 60
    model.data["workload"].update(task_count=4, interarrival_ms=[1, 2])
    settings = EnhancedDQNConfig(batch_size=2, warmup_transitions=2, replay_capacity=32,
        epsilon_decay_decisions=8, target_sync_updates=2)
    plan = TrainingPlan(model_seeds=(42, 43), training_budgets=(2, 4), validation_every_episodes=2,
        validation_workload_seeds=(84000,), validation_failure_seeds=(94000,),
        test_workload_seeds=(85000, 85001), test_failure_seeds=(95000, 95001))
    output = tmp_path_factory.mktemp("predictive-study")/"run"
    original = deepcopy((run, model.manifest()))
    import drlmots.predictive_training as module
    original_panel = module.panel
    def checked_panel(*args, **kwargs):
        if kwargs["crossed"]:
            for seed in plan.model_seeds:
                saved = json.loads((output/f"models/seed-{seed}/final.json").read_text())
                assert saved["episodes"] == max(plan.training_budgets)
                assert (output/f"models/seed-{seed}/selection.json").is_file()
                with gzip.open(output/f"models/seed-{seed}/training-traces.jsonl.gz", "rt", encoding="utf-8") as f:
                    assert sum(1 for line in f) == max(plan.training_budgets)
        return original_panel(*args, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "panel", checked_panel)
        result = run_study(run, model, settings, plan, output)
    assert original == (run, model.manifest())
    return run, model, plan, output, result


def test_actual_training_and_validation_selected_nested_budget_checkpoints(study):
    run, model, plan, output, result = study
    assert result["training_episode_count"] == 8
    assert len(result["test_rows"]) == 42
    assert len(result["validation_stages"]) == 6
    assert all(a["updates"] > 0 and a["episodes"] == 4 and a["final_epsilon"] == pytest.approx(.05) for a in result["training_audits"])
    for seed in plan.model_seeds:
        for budget in plan.training_budgets:
            selected = result["selections"][str(seed)][budget]
            candidates = [s for s in result["validation_stages"] if s["model_seed"] == seed and 0 < s["episode"] <= budget]
            best = min(candidates, key=lambda s: s["selection_rank"])
            assert selected["episode"] == best["episode"] and selected["fingerprint"] == best["frozen_fingerprint"]
            agent = PredictiveDQN.from_configs(run, model)
            restored = PredictiveDQN.load(output/selected["file"], agent.contract, host_nodes=agent.host_nodes)
            assert restored.episodes == best["episode"] and restored.fingerprint == selected["fingerprint"]
            with gzip.open(output/f"models/seed-{seed}/training-traces.jsonl.gz", "rt", encoding="utf-8") as f:
                available = {_digest(t) for line in f for t in json.loads(line)["rl"]["transitions"]}
            from dataclasses import asdict
            assert {_digest(asdict(t)) for t in restored.replay} <= available
    assert all(a["unchanged"] for a in result["frozen_audits"])


def test_fresh_test_pairing_and_original_reward_accounting(study):
    result = study[-1]
    cases = {(r["mode"], r["workload_seed"], r["failure_seed"]) for r in result["test_rows"]}
    assert len(cases) == 6
    for key in cases:
        rows = [r for r in result["test_rows"] if (r["mode"], r["workload_seed"], r["failure_seed"]) == key]
        assert len(rows) == 7
        assert len({r["workload_sha256"] for r in rows}) == len({r["outage_sha256"] for r in rows}) == 1
    for r in result["test_rows"]:
        if r["feasible"]:
            assert r["metrics"]["penalized_return"] == pytest.approx(-r["objective_score"])
    assert result["test_generated_after_selection"]
    locked = json.loads((study[3]/"selection-locked.json").read_text())
    assert locked["all_training_complete"] and locked["test_inputs_generated"] is False


def test_complete_compressed_training_traces_and_learning_counts(study):
    _, _, plan, output, result = study
    for seed in plan.model_seeds:
        with gzip.open(output/f"models/seed-{seed}/training-traces.jsonl.gz", "rt", encoding="utf-8") as f:
            traces = [json.loads(line) for line in f]
        assert len(traces) == 4
        assert all(t["rl"]["network_feature_count"] == 325 and t["rl"]["learn"] for t in traces)
        assert all(t["rl"]["total_reward"] == pytest.approx(t["rl"]["expected_reward"]) for t in traces)
        assert traces[-1]["rl"]["learner_updates"] == next(a["updates"] for a in result["training_audits"] if a["model_seed"] == seed)


def test_validation_selection_prefers_service_feasibility_then_cost_not_last_episode(study):
    model = study[1]
    base = deepcopy(study[-1]["validation_stages"][0]["rows"])
    for r in base:
        r["feasible"] = True
        r["metrics"].update(completion_rate=1, deadline_success_rate=1, penalized_return=-.5)
    worse = deepcopy(base)
    for r in worse:
        r["metrics"]["penalized_return"] = -.7
    assert validation_rank(base, 2, model) < validation_rank(worse, 4, model)
    infeasible = deepcopy(base)
    infeasible[0]["feasible"] = False
    infeasible[0]["metrics"].update(completion_rate=.5, deadline_success_rate=.5, penalized_return=-.1)
    assert validation_rank(base, 2, model) < validation_rank(infeasible, 4, model)
    assert validation_rank(base, 2, model) < validation_rank(base, 4, model)


@pytest.mark.parametrize("change", [
    dict(test_workload_seeds=(30000,)), dict(test_failure_seeds=(40000,)),
    dict(validation_workload_seeds=(85000,)), dict(test_workload_seeds=(82000,)),
    dict(training_workload_seed_start=82000), dict(test_failure_seeds=(92000,)),
    dict(training_budgets=(3, 2)), dict(training_budgets=(0,)),
    dict(model_seeds=(42, 42)), dict(model_seeds=(True,)),
    dict(validation_every_episodes=0), dict(validation_workload_seeds=()),
])
def test_plan_rejects_leakage_seen_panels_and_invalid_budgets(change):
    with pytest.raises(ValueError):
        replace(TrainingPlan(), **change)


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_strict_plan_version_and_round_trip(version):
    plan = TrainingPlan()
    assert TrainingPlan.parse(json.loads(json.dumps(plan.manifest()))) == plan
    data = plan.manifest()
    data["schema_version"] = version
    with pytest.raises(ValueError):
        TrainingPlan.parse(data)


def test_existing_evidence_is_not_overwritten(study):
    with pytest.raises(ValueError, match="empty output"):
        run_study(study[0], study[1], EnhancedDQNConfig(), study[2], study[3])


def test_training_report_preserves_metrics_and_portable_links(study):
    from drlmots.predictive_report import render
    output, result = study[-2:]
    before = (output/"result.json").read_bytes()
    summary = render(output)
    assert (output/"result.json").read_bytes() == before
    assert len(summary) == 8
    for s in summary:
        matching = [r for r in result["test_rows"] if r["method"] == s["method"] and r["mode"] == s["mode"]]
        assert s["metrics"]["total_energy_j"] == pytest.approx(sum(r["metrics"]["total_energy_j"] for r in matching)/len(matching))
        assert s["eligible"] == all(r["feasible"] for r in matching)
    html = (output/"report.html").read_text(encoding="utf-8")
    assert "<script" not in html and "{{" not in html and "325 inputs" in html
    links = [p for p in re.findall(r'(?:href|src)="([^"]+)"', html) if not p.startswith("#")]
    with zipfile.ZipFile(output/"report.zip") as z:
        assert z.testzip() is None and all(p in z.namelist() for p in links)
        assert all(z.read(n) == (output/n).read_bytes() for n in z.namelist())
    evidence = json.loads((output/"evidence.json").read_text())
    assert all(hashlib.sha256((output/n).read_bytes()).hexdigest() == s for n, s in evidence["sha256"].items())
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 3


def test_predictive_inference_cli_reproduces_selected_frozen_metrics(study, tmp_path):
    import yaml
    from drlmots.predictive_inference import main
    run, model, plan, output, result = study
    model_file = tmp_path/"model.yaml"
    model_file.write_text(yaml.safe_dump(model.manifest()))
    selected = result["selections"]["42"][2]
    args = ["--model-config", str(model_file), "--checkpoint", str(output/selected["file"]),
            "--workload-seed", "85000", "--failure-seed", "95000", "--output", str(tmp_path/"inference.json")]
    main(args)
    actual = json.loads((tmp_path/"inference.json").read_text())
    expected = next(r for r in result["test_rows"] if r["method"] == "predictive_budget_2" and r["model_seed"] == 42
                    and r["workload_seed"] == 85000 and r["failure_seed"] == 95000)
    assert actual["metrics"] == {k: v for k, v in expected["metrics"].items() if k != "penalized_return"}
    assert actual["rl"]["checkpoint_fingerprint"] == selected["fingerprint"]
    assert not actual["rl"]["learn"] and actual["rl"]["network_feature_count"] == 325
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
