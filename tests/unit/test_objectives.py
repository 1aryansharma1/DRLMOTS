from dataclasses import replace
import math

import pytest

from drlmots.dqn import TrainingConfig
from drlmots.objectives import (MultiObjectiveConfig, VMObjectiveProfile, normalized_cost,
                                objective_score, task_objectives)
from drlmots.rl_state import RLRewardConfig


def spec():
    return MultiObjectiveConfig(.4, .4, .2, .15, (VMObjectiveProfile("v", 100, .002, 2),))


def test_joules_convert_ms_to_seconds_and_decimal_kb_coefficients():
    values = task_objectives(spec().vms[0], service_ms=2, data_size_kb=20, latency_ms=10, network=True)
    assert values["compute_energy_j"] == pytest.approx(.2)
    assert values["transmission_energy_j"] == pytest.approx(.04)
    assert values["total_energy_j"] == pytest.approx(.24)
    assert values["failure_risk"] == pytest.approx(1 - math.exp(-.02))
    assert values["success_probability"] + values["failure_risk"] == pytest.approx(1)


def test_zero_network_excludes_transmission_energy_even_with_nonzero_coefficient():
    values = task_objectives(spec().vms[0], service_ms=2, data_size_kb=20, latency_ms=2, network=False)
    assert values["transmission_energy_j"] == 0
    assert values["total_energy_j"] == pytest.approx(.2)


def test_queue_and_upload_exposure_increase_risk_not_active_compute_energy():
    a = task_objectives(spec().vms[0], service_ms=2, data_size_kb=20, latency_ms=2, network=True)
    b = task_objectives(spec().vms[0], service_ms=2, data_size_kb=20, latency_ms=20, network=True)
    assert a["total_energy_j"] == b["total_energy_j"]
    assert b["failure_risk"] > a["failure_risk"]


def test_score_responds_to_each_objective_and_weights_are_explicit():
    values = {"latency_ms": 10, "total_energy_j": .15, "failure_risk": .1}
    result = objective_score(values, spec(), 10)
    assert result["weighted_cost"] == pytest.approx(.42)
    assert result["utility"] == pytest.approx(.58)
    for key in values:
        worse = dict(values)
        worse[key] *= 2
        assert objective_score(worse, spec(), 10)["utility"] < result["utility"]
    latency_only = replace(spec(), latency_weight=1, energy_weight=0, failure_risk_weight=0)
    assert objective_score(values, latency_only, 10)["utility"] == .5


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_physical_coefficients_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        VMObjectiveProfile("v", bad, 0, 0)


@pytest.mark.parametrize("change", [dict(latency_weight=.5), dict(energy_scale_j=0),
                                     dict(vms=()), dict(vms=(VMObjectiveProfile("v", 1, 0, 0),) * 2)])
def test_invalid_weights_scale_or_profiles_rejected(change):
    with pytest.raises(ValueError):
        replace(spec(), **change)


def test_extreme_cost_is_bounded_without_infinite_denominator():
    assert normalized_cost(1e308, 1e308) == .5
    assert normalized_cost(0, 10) == 0
    profile = VMObjectiveProfile("v", 0, 0, 1e308)
    assert task_objectives(profile, service_ms=1, data_size_kb=1, latency_ms=1e308, network=True)["failure_risk"] == 1
    with pytest.raises((ValueError, OverflowError)):
        task_objectives(VMObjectiveProfile("v", 1e308, 1, 0), service_ms=1e308, data_size_kb=1e308, latency_ms=1e308, network=True)


def test_profile_schema_roundtrip_and_exact_vm_catalogue():
    assert MultiObjectiveConfig.parse(spec().manifest()) == spec()
    spec().validate_vm_ids(["v"])
    with pytest.raises(ValueError, match="exactly match"):
        spec().validate_vm_ids(["v", "extra"])
    bad = spec().manifest()
    bad["vms"][0]["power"] = 100
    with pytest.raises(ValueError):
        MultiObjectiveConfig.parse(bad)


def test_legacy_training_and_reward_manifests_stay_unchanged():
    legacy = TrainingConfig().manifest()
    assert "multi_objective" not in legacy
    assert TrainingConfig.parse(legacy) == TrainingConfig()
    reward = RLRewardConfig(10, 1)
    assert reward.manifest()["schema_version"] == 2
    multi = replace(TrainingConfig(), multi_objective=spec())
    assert TrainingConfig.parse(multi.manifest()) == multi
    assert RLRewardConfig(10, 1, spec()).manifest()["schema_version"] == 3
    assert RLRewardConfig(10, 1, spec()).manifest()["sha256"] != reward.manifest()["sha256"]
