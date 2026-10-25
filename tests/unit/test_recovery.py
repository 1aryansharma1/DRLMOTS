from copy import deepcopy
from dataclasses import replace
import math
import random

import pytest

from drlmots.__main__ import configured_report
from drlmots.entities import Task
from drlmots.objectives import MultiObjectiveConfig, VMObjectiveProfile
from drlmots.recovery import simulate_recovery
from drlmots.recovery_config import Outage, RecoveryConfig, sample_outages
from drlmots.run_config import parse_run_config


def fixture(*, outages=(Outage("fog:f", 5, 20),), retries=1, horizon=30, idle=0, task_count=1):
    topology = {"hosts": [dict(host_id=n, storage_tb=3, memory_gb=32, bandwidth_reported=1000) for n in ("hf", "hc")],
        "vms": [dict(vm_id=v, tier=t, bandwidth_reported=200, queue_capacity=2,
                     resource_capacities={"cpu": 2}) for v, t in (("a-fog", "fog"), ("z-cloud", "cloud"))],
        "fog_nodes": [dict(fog_node_id="f", host_ids=["hf"], vm_ids=["a-fog"])],
        "cloud_datacenters": [dict(datacenter_id="c", host_ids=["hc"], vm_ids=["z-cloud"])],
        "access_points": [dict(access_point_id="ap", end_node_id="end")],
        "end_nodes": [dict(end_node_id="end", fog_node_ids=["f"])]}
    config = parse_run_config({"schema_version": 3, "kind": "synthetic_baseline", "resource_units": {"cpu": "units"},
        "topology": topology, "placement": {"policy": "fog_first_completion", "source_end_node_id": "end",
            "data_size_unit": "decimal_kB", "upload_model": "parallel_no_contention",
            "links": [dict(vm_id="a-fog", bandwidth_mbps=64, propagation_ms=0),
                      dict(vm_id="z-cloud", bandwidth_mbps=48, propagation_ms=0)]},
        "workload": dict(generator="uniform_interarrival", length_unit="bytes", arrival_time_unit="milliseconds",
            task_count=task_count, seed=42, interarrival_ms_range=[0, 0], length_bytes_range=[2500, 2500],
            data_size_kb_range=[24, 24], required_resources={"cpu": 2})})
    profile = MultiObjectiveConfig(.4, .4, .2, .15, (VMObjectiveProfile("a-fog", 100, .01, 0),
                                                   VMObjectiveProfile("z-cloud", 200, .02, 0)))
    recovery = RecoveryConfig(horizon, retries, 1, 8, 10, (("a-fog", idle), ("z-cloud", idle)), outages)
    return config, profile, recovery


def test_hand_calculated_failure_restart_latency_and_joules():
    config, profile, recovery = fixture()
    before = deepcopy(config)
    result = simulate_recovery(config, profile, recovery)
    first, second = result["attempts"]
    assert (first["dispatch_ms"], first["start_ms"], first["end_ms"]) == (0, 3, 5)
    assert (second["dispatch_ms"], second["start_ms"], second["end_ms"]) == (6, 10, 15)
    assert first["compute_energy_j"] == pytest.approx(.2)
    assert first["transmission_energy_j"] == pytest.approx(.24)
    assert second["compute_energy_j"] == pytest.approx(1)
    assert second["transmission_energy_j"] == pytest.approx(.48)
    metrics = result["metrics"]
    assert metrics["recovery_energy_j"] == pytest.approx(.01)
    assert metrics["total_energy_j"] == pytest.approx(1.93)
    assert metrics["makespan_ms"] == metrics["successful_mean_latency_ms"] == 15
    assert metrics["completed_count"] == metrics["recovered_task_count"] == 1
    assert result["outcomes"][0]["attempt_elapsed_ms"] == 14
    assert result["outcomes"][0]["recovery_blocked_ms"] == 1
    assert result["available_resources"] == {"a-fog": {"cpu": 2}, "z-cloud": {"cpu": 2}}
    assert config == before


def test_no_recovery_records_execution_failure_not_admission_rejection():
    config, profile, recovery = fixture(retries=0)
    result = simulate_recovery(config, profile, recovery)
    assert result["metrics"]["permanent_failure_count"] == 1
    assert result["metrics"]["rejected_count"] == 0
    assert result["metrics"]["total_energy_j"] == pytest.approx(.44)
    assert result["metrics"]["energy_per_completed_task_j"] is None
    assert result["outcomes"][0]["terminal_time_ms"] == 5


def test_partial_input_upload_is_charged_by_bytes_sent():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 1, 20),))
    result = simulate_recovery(config, profile, recovery)
    failed = result["attempts"][0]
    assert failed["start_ms"] is None and failed["executed_ms"] == 0
    assert failed["transmitted_kb"] == pytest.approx(8)
    assert failed["transmission_energy_j"] == pytest.approx(.08)
    assert result["metrics"]["makespan_ms"] == 11
    assert result["metrics"]["total_energy_j"] == pytest.approx(1.57)


def test_failure_during_propagation_charges_all_input_but_no_compute():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 4, 20),), retries=0)
    links = tuple(replace(link, propagation_ms=2 if link.vm_id == "a-fog" else 10) for link in config.placement.links)
    config = replace(config, placement=replace(config.placement, links=links))
    result = simulate_recovery(config, profile, recovery)
    failed = result["attempts"][0]
    assert failed["transmitted_kb"] == 24
    assert failed["compute_energy_j"] == 0
    assert failed["total_energy_j"] == pytest.approx(.24)


def test_completion_at_failure_boundary_wins_and_does_not_retry():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 8, 20),))
    result = simulate_recovery(config, profile, recovery)
    assert result["metrics"]["completed_count"] == 1
    assert result["metrics"]["node_interrupted_task_count"] == 0
    assert len(result["attempts"]) == 1


def test_horizon_finalizes_inflight_and_pending_and_releases_resources():
    config, profile, recovery = fixture(horizon=9, outages=(Outage("fog:f", 5, 8),))
    result = simulate_recovery(config, profile, recovery)
    assert result["outcomes"][0]["reason"] == "observation_horizon"
    assert result["attempts"][-1]["transmitted_kb"] == pytest.approx(18)
    assert result["metrics"]["permanent_failure_count"] == 1
    assert result["metrics"]["settlement_time_ms"] == 9
    pending = simulate_recovery(config, profile, replace(recovery, recovery_delay_ms=10))
    assert len(pending["attempts"]) == 1
    assert pending["outcomes"][0]["recovery_blocked_ms"] == 4
    assert pending["metrics"]["recovery_energy_j"] == pytest.approx(.04)


def test_idle_energy_uses_same_fixed_horizon_and_excludes_downtime():
    config, profile, recovery = fixture(idle=10)
    result = simulate_recovery(config, profile, recovery)
    assert result["vm_usage"][0]["up_ms"] == 15
    assert result["vm_usage"][0]["busy_ms"] == 2
    assert result["vm_usage"][1]["busy_ms"] == 5
    assert result["metrics"]["idle_energy_j"] == pytest.approx(.38)
    assert result["metrics"]["total_energy_j"] == pytest.approx(2.31)


def test_backup_waits_for_other_node_repair_without_future_knowledge():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 5, 20), Outage("cloud:c", 0, 9)))
    result = simulate_recovery(config, profile, recovery)
    assert result["attempts"][1]["dispatch_ms"] == 9
    assert result["outcomes"][0]["recovery_blocked_ms"] == 4
    assert result["metrics"]["makespan_ms"] == 18
    assert all("outages" not in row and "repair_time" not in row for row in result["decisions"])


def test_backup_wait_expires_and_original_failed_domain_is_excluded():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 5, 7), Outage("cloud:c", 0, 25)))
    result = simulate_recovery(config, profile, recovery)
    assert len(result["attempts"]) == 1
    assert result["outcomes"][0]["terminal_time_ms"] == 14
    assert result["outcomes"][0]["reason"] == "backup_wait_expired"


def test_retry_budget_and_repeated_failure_have_one_terminal_outcome():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 5, 7), Outage("cloud:c", 12, 20)))
    result = simulate_recovery(config, profile, recovery)
    assert len(result["attempts"]) == 2 and len(result["outcomes"]) == 1
    assert result["metrics"]["permanent_failure_count"] == 1
    assert result["metrics"]["interrupted_attempt_count"] == 2
    resumed = simulate_recovery(config, profile, replace(recovery, max_retries=2))
    assert len(resumed["attempts"]) == 3
    assert resumed["metrics"]["recovered_task_count"] == 1


def test_repair_allows_new_arrivals_but_never_resurrects_failed_task():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 5, 7),), retries=0)
    tasks = [Task("one", 0, 2500, 24, {"cpu": 2}), Task("two", 8, 2500, 24, {"cpu": 2})]
    result = simulate_recovery(config, profile, recovery, tasks=tasks)
    assert [r["status"] for r in result["outcomes"]] == ["permanent_failure", "completed"]
    assert result["attempts"][1]["vm_id"] == "a-fog"


def test_simultaneous_failures_apply_before_backup_selection():
    config, profile, recovery = fixture(outages=(Outage("fog:f", 5, 15), Outage("cloud:c", 5, 15)))
    result = simulate_recovery(config, profile, recovery)
    assert result["outcomes"][0]["reason"] == "backup_wait_expired"
    assert len(result["decisions"]) == 1


@pytest.mark.parametrize("policy", ["least_queue_fcfs", "first_fit_fcfs", "fog_first_completion"])
def test_no_failure_matches_preserved_network_baseline(policy):
    config, profile, recovery = fixture(outages=(), task_count=12, horizon=60)
    config = replace(config, workload=replace(config.workload, interarrival_ms_range=(0, 2),
                                            length_bytes_range=(500, 2000), data_size_kb_range=(5, 24)))
    old = configured_report(config, policy=policy)
    new = simulate_recovery(config, profile, recovery, policy=policy)
    assert new["metrics"]["completed_count"] == old["completed_count"]
    assert new["metrics"]["rejected_count"] == old["failed_count"]
    by_id = {a["task_id"]: a for a in new["attempts"]}
    for previous, actual in zip(old["outcomes"], new["outcomes"]):
        assert previous["finish_time_ms"] == pytest.approx(actual["finish_time_ms"]) if actual["finish_time_ms"] is not None else previous["finish_time_ms"] is None
        if previous["vm_id"] is not None:
            assert previous["vm_id"] == by_id[actual["task_id"]]["vm_id"]


def test_empty_workload_charges_idle_power_and_has_no_success_ratio():
    config, profile, recovery = fixture(outages=(), task_count=0, idle=10)
    result = simulate_recovery(config, profile, recovery)
    assert result["metrics"]["total_energy_j"] == pytest.approx(.6)
    assert result["metrics"]["completion_rate"] is None
    assert result["metrics"]["recovery_success_rate"] is None


@pytest.mark.parametrize("mutation", [dict(max_retries=-1), dict(max_retries=True), dict(horizon_ms=0),
    dict(recovery_wait_ms=float("nan")), dict(outages=(Outage("fog:f", 1, 10), Outage("fog:f", 10, 20)))])
def test_invalid_recovery_settings_rejected(mutation):
    config, profile, recovery = fixture()
    with pytest.raises((ValueError, TypeError)):
        replace(recovery, **mutation)


def test_strict_schema_profiles_and_legacy_dqn_guard():
    config, profile, recovery = fixture()
    assert RecoveryConfig.parse(recovery.manifest()) == recovery
    bad = recovery.manifest()
    bad["unknown"] = 1
    with pytest.raises(ValueError):
        RecoveryConfig.parse(bad)
    with pytest.raises(ValueError, match="unknown failure"):
        simulate_recovery(config, profile, replace(recovery, outages=(Outage("unknown", 1, 2),)))
    with pytest.raises(ValueError, match="idle power"):
        simulate_recovery(config, profile, replace(recovery, idle_power_w=(("a-fog", 999), ("z-cloud", 0))))
    with pytest.raises(ValueError, match="DQN"):
        simulate_recovery(config, profile, recovery, policy="standard_dqn")
    with pytest.raises(ValueError, match="strictly before"):
        simulate_recovery(config, profile, recovery, tasks=[Task("late", 30, 1000, 2)])


def test_outage_sampler_is_reproducible_local_and_node_stable():
    state = random.getstate()
    first = sample_outages(horizon_ms=100, seed=42, node_rates_per_s={"a": 100}, repair_ms=3)
    extended = sample_outages(horizon_ms=100, seed=42, node_rates_per_s={"a": 100, "b": 100}, repair_ms=3)
    assert first == tuple(item for item in extended if item.node_id == "a")
    assert first == sample_outages(horizon_ms=100, seed=42, node_rates_per_s={"a": 100}, repair_ms=3)
    assert first != sample_outages(horizon_ms=100, seed=43, node_rates_per_s={"a": 100}, repair_ms=3)
    assert random.getstate() == state
    assert all(0 <= o.start_ms < o.end_ms <= 100 for o in extended)
