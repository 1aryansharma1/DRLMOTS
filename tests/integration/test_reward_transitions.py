from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drlmots.__main__ import configured_report, smoke_report
from drlmots.entities import Host, ResourceTier, Task, VM
from drlmots.environment import simulate
from drlmots.normalization import capture_batch, fit_normalizer
from drlmots.reward import RewardConfig, load_reward_config, parse_reward_config
from drlmots.run_config import load_run_config, parse_run_config
from drlmots.topology import FogNode, Topology
from drlmots.transitions import build_learning_transitions, build_transitions, summarize_rewards


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "configs" / "resources.yaml"
REWARD = ROOT / "configs" / "reward.yaml"


def topology():
    return Topology((Host("h", 3, 32, 1000),),
                    (VM("vm", ResourceTier.FOG, 200, queue_capacity=1, resource_capacities={"cpu": 2}),),
                    fog_nodes=(FogNode("f", ("h",), ("vm",)),))


def task(name, arrival=0, demand=2):
    return Task(name, arrival, 1000, 20, {"cpu": demand})


def run(tasks, config=RewardConfig(10, 1)):
    return simulate(topology(), tasks, task_length_unit="bytes", resource_units={"cpu": "units"}, reward_config=config)


def test_worked_report_rewards_return_and_forced_terminal_boundary():
    config = load_run_config(RUN)
    original = deepcopy(config)
    scored = configured_report(config, reward_config=load_reward_config(REWARD))
    reward = scored.pop("reward")
    for decision in scored["decision_trace"]["decisions"]:
        decision.pop("reward")
    assert scored == configured_report(config, capture_decisions=True)
    assert config == original
    assert [row["reward"] for row in reward["transitions"]] == pytest.approx([5/6, 5/6, 5/7, 5/7, -1, -1])
    assert reward["summary"]["return"] == pytest.approx(23 / 21)
    assert reward["summary"]["average_reward"] == pytest.approx(23 / 126)
    assert (reward["summary"]["admitted_count"], reward["summary"]["rejected_count"]) == (4, 2)
    assert [t["done"] for t in reward["transitions"]] == [False] * 5 + [True]
    assert reward["transitions"][-2]["action"] is None
    assert reward["transitions"][-2]["next_action_mask"] == (False, False, False)
    assert reward["transitions"][-1]["next_state"] is None


def test_prediction_matches_completion_and_no_double_credit_during_drain():
    config = load_run_config(ROOT / "configs" / "normalization_train.yaml")
    report = configured_report(config, reward_config=RewardConfig(10, 1))
    outcomes = {row["task_id"]: row for row in report["outcomes"]}
    assert len(report["reward"]["transitions"]) == 100
    for decision in report["decision_trace"]["decisions"]:
        result = decision["reward"]
        outcome = outcomes[result["task_id"]]
        assert result["available_time_ms"] == outcome["arrival_time_ms"]
        if outcome["vm_id"] is not None:
            assert result["projected_completion_ms"] == pytest.approx(outcome["finish_time_ms"] - outcome["arrival_time_ms"])
        else:
            assert result["value"] == -1
    assert report["reward"]["summary"]["return"] == pytest.approx(sum(d["reward"]["value"] for d in report["decision_trace"]["decisions"]))


def test_reward_is_causal_and_inputs_remain_unchanged():
    inputs = [task("a"), task("b")]
    original = deepcopy(inputs)
    prefix = run(inputs)
    extended = run(inputs + [task("future", 0), task("much-later", 1000)])
    assert [r.reward for r in prefix.decisions] == [r.reward for r in extended.decisions[:2]]
    assert inputs == original
    plain = simulate(topology(), inputs, task_length_unit="bytes", resource_units={"cpu": "units"}, capture_decisions=True)
    assert replace(prefix, reward_config=None, decisions=tuple(replace(d, reward=None) for d in prefix.decisions)) == plain


def test_empty_and_all_rejected_episodes_account_for_every_arrival():
    empty = run([])
    assert build_transitions(empty.decisions) == ()
    assert summarize_rewards(empty.decisions)["return"] == 0
    assert summarize_rewards(empty.decisions)["average_reward"] is None
    rejected = run([task("x", demand=3), task("y", demand=3)])
    assert rejected.completed_count == 0 and rejected.makespan_ms == 0
    assert summarize_rewards(rejected.decisions)["return"] == -2
    assert summarize_rewards(rejected.decisions)["average_reward"] == -1
    assert build_learning_transitions(rejected.decisions, gamma=0.9) == ()
    assert [r.done for r in build_transitions(rejected.decisions)] == [False, True]


@pytest.mark.parametrize("gamma", [0, 0.5, 1])
def test_forced_chains_are_discounted_before_next_actionable_bootstrap(gamma):
    result = run([task("a"), task("b"), task("c"), task("d"), task("e", 2)])
    learning = build_learning_transitions(result.decisions, gamma=gamma)
    assert [record.task_id for record in learning] == ["a", "b", "e"]
    assert learning[0].reward == pytest.approx(5 / 6)
    assert learning[0].bootstrap_discount == gamma
    assert learning[1].reward == pytest.approx(5 / 7 - gamma - gamma ** 2)
    assert learning[1].discount_steps == 3
    assert learning[1].bootstrap_discount == gamma ** 3
    assert not learning[1].done and any(learning[1].next_action_mask)
    assert learning[2].done and learning[2].bootstrap_discount == 0
    assert learning[2].next_state is None


def test_forced_suffix_is_terminal_and_forced_prefix_has_no_fake_action():
    suffix = run([task("a"), task("b"), task("c"), task("d")])
    records = build_learning_transitions(suffix.decisions, gamma=0.5)
    assert len(records) == 2 and records[-1].done
    assert records[-1].reward == pytest.approx(5 / 7 - 0.5 - 0.25)
    assert records[-1].bootstrap_discount == 0 and records[-1].next_action_mask is None
    prefix = run([task("no-fit", demand=3), task("admit")])
    learning = build_learning_transitions(prefix.decisions, gamma=0.9)
    assert len(learning) == 1 and learning[0].task_id == "admit"
    assert summarize_rewards(prefix.decisions)["return"] == pytest.approx(-1 + 5 / 6)


@pytest.mark.parametrize("gamma", [-1, 2, True, float("nan"), float("inf")])
def test_invalid_transition_discounts_rejected(gamma):
    with pytest.raises((TypeError, ValueError)):
        build_learning_transitions(run([task("a")]).decisions, gamma=gamma)


def test_transition_identity_state_and_scoring_mismatches_rejected():
    record = run([task("a")]).decisions[0]
    for bad in (replace(record, reward=None), replace(record, state=(0,)),
                replace(record, selected_vm_id="other", reward=replace(record.reward, selected_vm_id="other")),
                replace(record, reward=replace(record.reward, available_time_ms=100))):
        with pytest.raises(ValueError):
            build_transitions([bad])
    with pytest.raises(ValueError, match="unique"):
        build_transitions([record, record])


def test_nonrepresentable_return_is_rejected_instead_of_exporting_infinity():
    result = run([task("x", demand=3), task("y", demand=3)], RewardConfig(10, 1e308))
    with pytest.raises(ValueError, match="representable"):
        summarize_rewards(result.decisions)


def test_reward_replay_policy_and_normalizer_preservation():
    config = load_run_config(ROOT / "configs" / "normalization_train.yaml")
    training, _ = capture_batch(config, split="train")
    normalizer = fit_normalizer([training])
    original = normalizer.manifest()
    reward_config = RewardConfig(10, 2)
    report = configured_report(config, policy="first_fit_fcfs", reward_config=reward_config)
    for decision in report["decision_trace"]["decisions"]:
        assert len(normalizer.transform(training.context, decision["state"])) == 47
    assert normalizer.manifest() == original
    saved_run, saved_reward = dict(report["run_configuration"]), dict(report["reward"]["configuration"])
    saved_run.pop("sha256")
    saved_reward.pop("sha256")
    assert report == configured_report(parse_run_config(saved_run), policy=report["policy"], reward_config=parse_reward_config(saved_reward))


def test_actual_reward_cli_and_invalid_config_no_output(tmp_path):
    output = tmp_path / "reward.json"
    command = [sys.executable, "-m", "drlmots", "--config", str(RUN), "--reward-config", str(REWARD), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text())
    assert report["reward"]["summary"]["return"] == pytest.approx(23 / 21)
    assert len(report["reward"]["transitions"]) == 6
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(REWARD.read_text().replace("latency_scale_ms: 10", "latency_scale_ms: 0"))
    bad_output = tmp_path / "missing.json"
    bad = subprocess.run([*command[:-4], "--reward-config", str(invalid), "--output", str(bad_output)], capture_output=True, text=True)
    assert bad.returncode != 0 and not bad_output.exists()


def test_unscored_legacy_report_is_unchanged_in_shape():
    report = smoke_report(capture_decisions=True)
    assert "reward" not in report
    assert all("reward" not in d for d in report["decision_trace"]["decisions"])


def test_seeded_cli_nonrepresentable_reward_return_is_a_clean_error(tmp_path):
    reward = tmp_path / "huge-penalty.yaml"
    reward.write_text(REWARD.read_text().replace("rejection_penalty: 1", "rejection_penalty: 1.0e+308"))
    output = tmp_path / "not-written.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--task-count", "100", "--seed", "42",
                             "--reward-config", str(reward), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode != 0 and "representable" in result.stderr
    assert "Traceback" not in result.stderr and not output.exists()
