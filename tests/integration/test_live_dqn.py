from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random

import pytest
import torch

from drlmots.__main__ import configured_report
from drlmots.dqn import EpisodeController, TrainingConfig
from drlmots.entities import Task
from drlmots.environment import simulate
from drlmots.normalization import TraceBatch, _digest, fit_normalizer
from drlmots.rl_state import RLRewardConfig, encode_frame, learning_context, variant_mask
from drlmots.run_config import load_run_config
from drlmots.train import (checkpoint_artifact, continue_training, cpu_execution, evaluate, initialize,
                           load_checkpoint, run_episode, save_checkpoint, seeded_config)


ROOT = Path(__file__).resolve().parents[2]


def config(network=True, tasks=12):
    cfg = load_run_config(ROOT / ("configs/rl_network.yaml" if network else "configs/normalization_train.yaml"))
    return replace(cfg, workload=replace(cfg.workload, task_count=tasks))


def settings(**changes):
    return replace(TrainingConfig(episodes=4, calibration_episodes=1, batch_size=4, replay_capacity=64,
                                   hidden_widths=(8, 8), target_sync_updates=3, evaluation_seeds=(3000, 3001)), **changes)


@pytest.fixture(autouse=True)
def one_thread():
    with cpu_execution():
        yield


def test_live_network_choices_actually_change_placements_and_preserve_resources():
    cfg, spec = config(), settings()
    learner = initialize(cfg, spec, "standard")
    with torch.no_grad():
        for p in learner.online.parameters():
            p.zero_()
        learner.online.layers[-1].bias[0] = 10  # choose cloud-1 even though fog is faster
    before = deepcopy(cfg)
    report = run_episode(learner, seeded_config(cfg, 3000))
    first = report["rl"]["decisions"][0]
    assert first["vm_id"] == report["outcomes"][0]["vm_id"] == "cloud-1"
    assert first["action"] == 0 and first["q_values"][0] == 10
    assert configured_report(seeded_config(cfg, 3000))["outcomes"][0]["vm_id"].startswith("fog")
    assert cfg == before
    assert learner.updates == 0 and not len(learner.replay)
    assert report["completed_count"] + report["failed_count"] == cfg.workload.task_count
    for vm in cfg.topology.vms:
        assert report["available_resources"][vm.vm_id] == vm.resource_capacities
    outcomes = {item["task_id"]: item for item in report["outcomes"]}
    for record in report["rl"]["decisions"]:
        if record["action"] is not None:
            assert record["predicted_finish_time_ms"] == pytest.approx(outcomes[record["task_id"]]["finish_time_ms"])


def test_fog_variant_keeps_preference_as_explicit_constraint_and_has_real_choices():
    cfg = config(tasks=40)
    learner = initialize(cfg, settings(), "fog_preferred")
    with torch.no_grad():
        for p in learner.online.parameters():
            p.zero_()
        learner.online.layers[-1].bias[0] = 100
    report = run_episode(learner, seeded_config(cfg, 3000))
    assert report["outcomes"][0]["vm_id"].startswith("fog")
    assert any(sum(row["action_mask"]) > 1 for row in report["rl"]["decisions"])
    for record, placement in zip(report["rl"]["decisions"], report["placement"]["decisions"]):
        action = record["action"]
        if action is None:
            assert not any(record["legal_mask"])
            continue
        assert record["action_mask"][action] and record["legal_mask"][action]
        fog = [item["predicted_finish_time_ms"] for item in placement["candidates"] if item["tier"] == "fog"]
        cloud = [item["predicted_finish_time_ms"] for item in placement["candidates"] if item["tier"] == "cloud"]
        if fog and cloud:
            if min(fog) <= min(cloud):
                assert record["vm_id"].startswith("fog") and record["predicted_finish_time_ms"] <= min(cloud)
            else:
                assert record["vm_id"].startswith("cloud") and record["predicted_finish_time_ms"] < min(fog)


def test_network_contract_represents_idle_upload_queue_and_frozen_scaling():
    cfg = config()
    agent = initialize(cfg, settings(), "standard")
    states = []
    def choose(frame):
        states.append((frame, encode_frame(agent.normalizer.context.schema, frame)))
        return next((vm.vm_id for vm in frame.vms if vm.estimate), None)
    simultaneous = replace(cfg, workload=replace(cfg.workload, interarrival_ms_range=(0, 0)))
    configured_report(simultaneous, policy="test_upload_state", controller=choose)
    assert any(vm.queued and vm.running_remaining_ms == 0 for frame, _ in states for vm in frame.vms)
    before = agent.normalizer.manifest()
    for frame, raw in states:
        row = agent.normalizer.transform(agent.normalizer.context, raw)
        assert len(row) == agent.normalizer.context.schema.dimension
        assert all(0 <= value <= 1 for value in row)
    assert agent.normalizer.manifest() == before
    assert agent.normalizer.context.schema.manifest()["schema_version"] == 2
    assert agent.normalizer.context.schema.dimension == 168


@pytest.mark.parametrize("network", [False, True])
def test_actual_training_updates_both_environment_modes(network):
    cfg, spec = config(network), settings(episodes=2)
    learner = initialize(cfg, spec, "standard")
    before = learner.normalizer.manifest()
    summary = continue_training(learner)
    assert summary["updates"] > 0
    assert summary["initial_weights_sha256"] != summary["final_weights_sha256"]
    assert learner.episodes_completed == 2 and len(learner.history) == 2
    assert learner.normalizer.manifest() == before
    assert len(learner.losses) == learner.updates
    assert sum(row["completed_count"] + row["failed_count"] for row in learner.history) == 24


@pytest.mark.parametrize("gamma", [0, 0.5, 1])
def test_forced_prefix_and_suffix_credit_has_no_fake_replay_action(gamma):
    cfg = config(False)
    learner = initialize(cfg, settings(gamma=gamma, batch_size=32), "standard")
    controller = EpisodeController(learner, learn=True)
    # Forced prefix, legal placement, then two forced rejections.
    tasks = [Task("prefix", 0, 1000, 20, {"cpu_units": 99, "memory_gb": 4}),
             Task("legal", 0, 1000, 20, {"cpu_units": 2, "memory_gb": 4}),
             Task("forced1", 0, 1000, 20, {"cpu_units": 99, "memory_gb": 4}),
             Task("forced2", 0, 1000, 20, {"cpu_units": 99, "memory_gb": 4})]
    result = simulate(cfg.topology, tasks, task_length_unit="bytes", resource_units=cfg.resource_units,
                      policy="standard_dqn", controller=controller)
    controller.finish()
    assert (result.completed_count, result.failed_count) == (1, 3)
    assert len(learner.replay) == 1
    item = learner.replay.items[0]
    assert item.action is not None and item.done and item.bootstrap_discount == 0
    assert item.discount_steps == 3
    assert item.reward == pytest.approx(5 / 6 - gamma - gamma**2)
    assert sum(record["reward"] for record in controller.records) == pytest.approx(5 / 6 - 3)


def test_all_rejected_episode_has_negative_return_and_empty_replay():
    cfg = config()
    learner = initialize(cfg, settings(), "standard")
    cfg = replace(cfg, workload=replace(cfg.workload, required_resources={"cpu_units": 99, "memory_gb": 4}))
    report = run_episode(learner, cfg, learn=True)
    assert report["failed_count"] == 12
    assert report["rl"]["summary"]["undiscounted_return"] == -12
    assert not learner.replay.items and learner.updates == learner.decision_steps == 0
    with pytest.raises(ValueError, match="no legal placement"):
        initialize(cfg, settings(), "standard")


@pytest.mark.parametrize("gamma", [0, 0.5, 1])
def test_forced_middle_chain_bootstraps_only_at_next_actionable_arrival(gamma):
    cfg = config(False)
    learner = initialize(cfg, settings(gamma=gamma, batch_size=32), "standard")
    controller = EpisodeController(learner, learn=True)
    good = {"cpu_units": 2, "memory_gb": 4}
    bad = {"cpu_units": 99, "memory_gb": 4}
    tasks = [Task("first", 0, 1000, 20, good), Task("forced1", 0, 1000, 20, bad),
             Task("forced2", 1, 1000, 20, bad), Task("next-legal", 3, 1000, 20, good)]
    simulate(cfg.topology, tasks, task_length_unit="bytes", resource_units=cfg.resource_units,
             policy="standard_dqn", controller=controller)
    controller.finish()
    first, last = learner.replay.items
    assert not first.done and first.discount_steps == 3
    assert first.reward == pytest.approx(5 / 6 - gamma - gamma**2)
    assert first.bootstrap_discount == gamma**3 and any(first.next_mask)
    assert first.next_state == last.state and last.done


def test_full_checkpoint_roundtrip_and_resume_matches_uninterrupted_training(tmp_path):
    cfg, spec = config(tasks=10), settings()
    uninterrupted = initialize(cfg, spec, "standard")
    continue_training(uninterrupted)
    interrupted = initialize(cfg, spec, "standard")
    continue_training(interrupted, target_episodes=2)
    path = tmp_path / "checkpoint.json"
    save_checkpoint(interrupted, path)
    state, python = torch.random.get_rng_state().clone(), random.getstate()
    resumed = load_checkpoint(path, for_resume=True)
    assert checkpoint_artifact(resumed)["sha256"] == checkpoint_artifact(interrupted)["sha256"]
    assert torch.equal(state, torch.random.get_rng_state()) and python == random.getstate()
    continue_training(resumed)
    assert checkpoint_artifact(resumed)["sha256"] == checkpoint_artifact(uninterrupted)["sha256"]


def test_held_out_evaluation_is_paired_greedy_frozen_and_rejects_overlap():
    cfg, spec = config(), settings(episodes=2)
    standard, preferred = initialize(cfg, spec, "standard"), initialize(cfg, spec, "fog_preferred")
    continue_training(standard)
    continue_training(preferred)
    before = checkpoint_artifact(standard)["sha256"], checkpoint_artifact(preferred)["sha256"]
    evaluation = evaluate({"standard_dqn": standard, "fog_preferred_dqn": preferred}, cfg, spec.evaluation_seeds)
    assert len(evaluation["runs"]) == 2 and len(evaluation["summaries"]) == 5
    assert before == (checkpoint_artifact(standard)["sha256"], checkpoint_artifact(preferred)["sha256"])
    for run in evaluation["runs"]:
        assert len({report["workload"]["sha256"] for report in run["reports"]}) == 1
        for report in run["reports"][-2:]:
            assert all(row["epsilon"] == 0 for row in report["rl"]["decisions"])
            assert report["rl"]["learn"] is False
    with pytest.raises(ValueError, match="overlaps"):
        evaluate({"standard_dqn": standard}, cfg, (2000,))
    changed = deepcopy(cfg)
    changed.topology.vms[0].queue_capacity += 1
    with pytest.raises(ValueError, match="environment"):
        evaluate({"standard_dqn": standard}, changed, (3000,))


def test_exact_task_overlap_rejected_even_with_different_seeds():
    cfg = config(False, tasks=6)
    cfg = replace(cfg, workload=replace(cfg.workload, interarrival_ms_range=(0, 0),
                                        length_bytes_range=(1000, 1000), data_size_kb_range=(20, 20)))
    with pytest.raises(ValueError, match="tasks overlap"):
        initialize(cfg, settings(episodes=1), "standard")


@pytest.mark.parametrize("mutation", ["checksum", "version", "counter", "losses", "moments", "replay", "provenance", "context"])
def test_corrupt_or_incompatible_training_checkpoints_rejected(tmp_path, mutation):
    learner = initialize(config(tasks=8), settings(episodes=1), "standard")
    continue_training(learner)
    data = checkpoint_artifact(learner)
    if mutation == "checksum":
        data["updates"] += 1
    elif mutation == "version":
        data["schema_version"] = True
    elif mutation == "counter":
        data["decision_steps"] = -1
    elif mutation == "losses":
        data["losses"] = []
    elif mutation == "moments":
        name = next(iter(data["optimizer"]["moments"]))
        data["optimizer"]["moments"][name]["exp_avg_sq"][0][0] = -1
    elif mutation == "replay":
        data["replay"][0]["action"] = None
    elif mutation == "provenance":
        data["training_task_hashes"] = []
    else:
        data["context_sha256"] = "0" * 64
    if mutation != "checksum":
        data["sha256"] = _digest({k: v for k, v in data.items() if k != "sha256"})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        load_checkpoint(path, for_resume=True)


def test_controller_cannot_choose_illegal_vm_or_voluntarily_reject():
    cfg = config(False)
    for chooser in (lambda frame: "missing", lambda frame: None):
        with pytest.raises(ValueError, match="eligible"):
            configured_report(cfg, policy="invalid_agent", controller=chooser)
