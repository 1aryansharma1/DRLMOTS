from dataclasses import replace
import math

import pytest

from drlmots.entities import ResourceTier
from drlmots.reward import RewardConfig, load_reward_config, parse_reward_config, score_action
from drlmots.state import DecisionObservation, VMObservation


def observation(*, waiting=0, remaining=0, length=1000, demand=1, capacity=2, full=False):
    vm = VMObservation("vm", ResourceTier.FOG, remaining > 0, remaining,
                       1 if waiting or full else 0, 1, waiting if waiting else 2 if full else 0,
                       (capacity,), (0 if remaining else capacity,))
    return DecisionObservation("task", 3, length, 20, (("cpu", "units"),), (demand,), (vm,))


def test_hand_rewards_prefer_shorter_wait_and_explicit_rejection_is_negative():
    config = RewardConfig(10, 1)
    immediate = score_action(observation(), 0, config)
    queued = score_action(observation(remaining=2), 0, config)
    full = score_action(observation(remaining=2, full=True), None, config)
    assert immediate.value == pytest.approx(5 / 6)
    assert queued.value == pytest.approx(5 / 7)
    assert immediate.value > queued.value > full.value == -1
    assert (queued.projected_wait_ms, queued.service_ms, queued.projected_completion_ms) == (2, 2, 4)
    assert immediate.available_time_ms == queued.available_time_ms == full.available_time_ms == 3
    assert full.projected_completion_ms is None and full.selected_vm_id is None


def test_other_queued_work_is_counted_and_service_length_changes_reward():
    reward = score_action(observation(remaining=2, waiting=4), None, RewardConfig(10, 1))
    assert reward.value == -1  # Waiting capacity full, regardless of forecast cost.
    short = score_action(observation(remaining=2), 0, RewardConfig(10, 1))
    long = score_action(observation(remaining=2, length=2000), 0, RewardConfig(10, 1))
    assert long.projected_completion_ms == 6
    assert short.value > long.value == pytest.approx(10 / 16)


def test_multiple_waiting_tasks_contribute_to_projected_latency():
    original = observation(remaining=3, waiting=5)
    vm = replace(original.vms[0], waiting_capacity=2)
    obs = replace(original, vms=(vm,))
    reward = score_action(obs, 0, RewardConfig(10, 1))
    assert (reward.projected_wait_ms, reward.projected_completion_ms, reward.value) == (8, 10, 0.5)


@pytest.mark.parametrize("index", [-1, 1, True, 0.0, "0"])
def test_invalid_action_indices_rejected(index):
    with pytest.raises((TypeError, ValueError)):
        score_action(observation(), index, RewardConfig(10, 1))


def test_masked_and_voluntary_rejection_actions_rejected():
    with pytest.raises(ValueError, match="masked"):
        score_action(observation(capacity=1, demand=2), 0, RewardConfig(10, 1))
    with pytest.raises(ValueError, match="null action"):
        score_action(observation(), None, RewardConfig(10, 1))


@pytest.mark.parametrize("value", [0, -1, True, "10", float("inf"), float("nan"), 10**400])
@pytest.mark.parametrize("field", ["latency_scale_ms", "rejection_penalty"])
def test_invalid_reward_parameters(value, field):
    settings = {"latency_scale_ms": 10, "rejection_penalty": 1} | {field: value}
    with pytest.raises((ValueError, TypeError)):
        RewardConfig(**settings)


def test_ratio_extremes_stay_finite_without_overflow():
    slow = observation(remaining=1e308)
    assert score_action(slow, 0, RewardConfig(5e-324, 1)).value == 0
    balanced = score_action(slow, 0, RewardConfig(1e308, 1)).value
    assert balanced == pytest.approx(0.5)
    assert math.isfinite(balanced)
    assert score_action(observation(), 0, RewardConfig(1e308, 1)).value == 1


def test_nonrepresentable_projection_is_rejected():
    original = observation(remaining=1e308, waiting=1e308)
    obs = replace(original, vms=(replace(original.vms[0], waiting_capacity=2),))
    with pytest.raises(ValueError, match="representable"):
        score_action(obs, 0, RewardConfig(10, 1))


def test_config_manifest_replay_and_parameter_fingerprint():
    config = RewardConfig(10, 1)
    saved = config.manifest()
    sha = saved.pop("sha256")
    assert parse_reward_config(saved) == config
    assert RewardConfig(20, 1).manifest()["sha256"] != sha
    assert RewardConfig(10, 2).manifest()["sha256"] != sha


@pytest.mark.parametrize("change", [{"schema_version": True}, {"algorithm": "paper_eq25"},
                                   {"kind": "paper_reward"}, {"extra": 1}])
def test_unsupported_or_unknown_config_fields_rejected(change):
    saved = RewardConfig(10, 1).manifest()
    saved.pop("sha256")
    with pytest.raises(ValueError):
        parse_reward_config(saved | change)


def test_duplicate_reward_yaml_and_missing_fields_rejected(tmp_path):
    path = tmp_path / "reward.yaml"
    path.write_text("latency_scale_ms: 10\nlatency_scale_ms: 20\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_reward_config(path)
    with pytest.raises(ValueError, match="exactly"):
        parse_reward_config({"latency_scale_ms": 10})
