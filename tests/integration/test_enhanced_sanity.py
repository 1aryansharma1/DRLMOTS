from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from drlmots.enhanced_dqn import EnhancedDQN, EnhancedDQNConfig, run_enhanced_episode
from drlmots.enhanced_sanity import (SanityPlan, diagnostic_scenarios, main,
                                    network_digest, run_sanity)
from drlmots.normalization import _digest
from drlmots.objective_model import load_objective_model
from drlmots.run_config import load_run_config


def fixture():
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    model.data["horizon_ms"] = 60
    model.data["workload"].update(task_count=4, interarrival_ms=[1, 2])
    settings = EnhancedDQNConfig(batch_size=2, warmup_transitions=2, replay_capacity=32,
        target_sync_updates=2, epsilon_decay_decisions=8)
    plan = SanityPlan(training_episodes=2, validation_every_episodes=1,
        evaluation_workload_seeds=(81000,), evaluation_failure_seeds=(91000,))
    return run, model, settings, plan


@pytest.mark.parametrize("changes", [
    dict(training_episodes=0), dict(training_episodes=True), dict(validation_every_episodes=0),
    dict(evaluation_workload_seeds=()), dict(evaluation_workload_seeds=(81000, 81000)),
    dict(evaluation_failure_seeds=(True,)), dict(evaluation_workload_seeds=(10000,)),
    dict(evaluation_failure_seeds=(20000,)), dict(recovery_probe_workload_seed=81000),
    dict(evaluation_workload_seeds=(81000,)), dict(training_workload_seed_start=2**63 - 1)])
def test_invalid_or_leaking_seed_plans_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        SanityPlan(**changes)


@pytest.mark.parametrize("version", [True, 2, "1"])
def test_plan_version_is_strict(version):
    data = SanityPlan().manifest()
    data["schema_version"] = version
    with pytest.raises(ValueError, match="kind/version"):
        SanityPlan.parse(data)


def test_plan_roundtrip_and_no_unknown_fields():
    plan = SanityPlan()
    assert SanityPlan.parse(json.loads(json.dumps(plan.manifest()))) == plan
    with pytest.raises(ValueError, match="documented"):
        SanityPlan.parse({**plan.manifest(), "pick_best_validation": True})


def test_real_training_and_frozen_behavior_are_reproducible_and_restore_exactly(tmp_path):
    run, model, settings, plan = fixture()
    originals = deepcopy((run, model.manifest(), settings, plan))
    threads = torch.get_num_threads()
    a = run_sanity(run, model, settings, plan, tmp_path / "a", render_figures=False)
    b = run_sanity(run, model, settings, plan, tmp_path / "b", render_figures=False)
    assert a == b
    assert a["learning_updates_verified"] and a["optimizer_updates"] > 0
    assert a["initial_online_sha256"] != a["final_online_sha256"] and a["parameter_l2_change"] > 0
    assert a["fixed_probe_counts"]["initial"] > 0 and a["fixed_probe_counts"]["recovery"] > 0
    assert [s["episode"] for s in a["validation_stages"]] == [0, 1, 2]
    assert all(s["frozen_verified"] for s in a["validation_stages"])
    assert a["checkpoint_selection"].startswith("last fixed-budget")
    actual_transitions = [t for file in sorted((tmp_path / "a/training").glob("*.json"))
                          for t in json.loads(file.read_text())["rl"]["transitions"]]
    assert a["training_decisions"] == len(actual_transitions) == sum(a["training_transition_kinds"].values())
    agent = EnhancedDQN.from_configs(run, model, settings)
    loaded = EnhancedDQN.load(tmp_path / "a/trained-final.json", agent.contract, host_nodes=agent.host_nodes)
    assert loaded.episodes == 2 and loaded.updates == a["optimizer_updates"]
    assert network_digest(loaded.online) == a["final_online_sha256"]
    name, kwargs = diagnostic_scenarios(model, plan)[-1]
    actual = run_enhanced_episode(loaded, run, model, **kwargs)
    saved = json.loads((tmp_path / f"a/validation/episode-002-{name}.json").read_text())
    assert _digest(actual) == _digest(saved)
    probes = json.loads((tmp_path / "a/fixed-observation-probes.json").read_text())
    for stage in a["validation_stages"]:
        for observation, result in zip(probes, stage["fixed_observation_results"]):
            assert observation["observation"]["legal_mask"][result["action"]]
            assert result["changed"] == (result["action"] != observation["initial_action"])
    assert torch.get_num_threads() == threads and (run, model.manifest(), settings, plan) == originals
    assert "figures/" not in (tmp_path / "a/REPORT.md").read_text()
    assert json.loads((tmp_path / "a/experiment.json").read_text())["status"] == "complete"


def test_too_small_budget_records_missing_updates_without_claiming_training_success(tmp_path):
    run, model, _, plan = fixture()
    result = run_sanity(run, model, EnhancedDQNConfig(), replace(plan, training_episodes=1), tmp_path, render_figures=False)
    assert result["optimizer_updates"] == 0 and not result["learning_updates_verified"]
    assert not result["behavior_changed"] and result["parameter_l2_change"] == 0
    assert not result["checks"]["optimizer_updates"]


def test_existing_evidence_is_never_overwritten(tmp_path):
    run, model, settings, plan = fixture()
    sentinel = tmp_path / "do-not-overwrite.txt"
    sentinel.write_text("original")
    with pytest.raises(ValueError, match="empty output"):
        run_sanity(run, model, settings, plan, tmp_path)
    assert sentinel.read_text() == "original" and len(list(tmp_path.iterdir())) == 1


def test_partial_budget_last_episode_is_evaluated_even_if_not_interval_multiple(tmp_path):
    run, model, settings, plan = fixture()
    result = run_sanity(run, model, settings, replace(plan, validation_every_episodes=3), tmp_path, render_figures=False)
    assert [s["episode"] for s in result["validation_stages"]] == [0, 2]
    assert result["validation_stages"][-1]["updates"] == result["optimizer_updates"]


def test_plotting_uses_actual_result_and_exports_both_formats(tmp_path):
    run, model, settings, plan = fixture()
    result = run_sanity(run, model, settings, replace(plan, training_episodes=1), tmp_path)
    assert result["optimizer_updates"] > 0
    for name in ("learning", "validation"):
        assert (tmp_path / f"figures/{name}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (tmp_path / f"figures/{name}.svg").read_text()


def test_cli_rejects_bad_plan_without_starting_an_experiment(tmp_path):
    plan = tmp_path / "bad.yaml"
    plan.write_text("schema_version: 99\nkind: enhanced_dqn_training_sanity\n")
    with pytest.raises(SystemExit) as error:
        main(["--plan-config", str(plan), "--output-dir", str(tmp_path / "output")])
    assert error.value.code == 2 and not (tmp_path / "output").exists()
