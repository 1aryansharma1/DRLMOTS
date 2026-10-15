from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from drlmots.entities import Host, ResourceTier, Task, VM
from drlmots.environment import simulate
from drlmots.fog_cloud import FogCloudSettings, TransmissionLink, parse_placement_settings
from drlmots.run_config import load_run_config, parse_run_config
from drlmots.topology import CloudDatacenter, EndNode, FogNode, Topology


ROOT = Path(__file__).resolve().parents[2]


def infrastructure(capacity=2, cloud=True):
    hosts = [Host("h-fog", 3, 32, 1000)]
    vms = [VM("z-fog", ResourceTier.FOG, 200, queue_capacity=capacity,
              resource_capacities={"cpu": 2, "memory": 4})]
    clouds = ()
    if cloud:
        hosts.append(Host("h-cloud", 3, 32, 1000))
        vms.append(VM("a-cloud", ResourceTier.CLOUD, 200, queue_capacity=capacity,
                      resource_capacities={"cpu": 4, "memory": 8}))
        clouds = (CloudDatacenter("cloud", ("h-cloud",), ("a-cloud",)),)
    return Topology(tuple(hosts), tuple(vms),
                    fog_nodes=(FogNode("local", ("h-fog",), ("z-fog",)),),
                    cloud_datacenters=clouds, end_nodes=(EndNode("origin", ("local",)),))


def settings(cloud=True, cloud_delay=4):
    links = [TransmissionLink("z-fog", 8, 0)]
    if cloud:
        links.append(TransmissionLink("a-cloud", 8, cloud_delay))
    return FogCloudSettings("origin", tuple(links))


def task(name="t", arrival=0, length=1000, data=1, demands=None):
    return Task(name, arrival, length, data, {"cpu": 2, "memory": 4} if demands is None else demands)


def run(tasks, *, topology=None, placement=None, policy="fog_first_completion"):
    return simulate(infrastructure() if topology is None else topology, tasks,
                    task_length_unit="bytes", resource_units={"cpu": "units", "memory": "GB"},
                    policy=policy, placement_config=settings() if placement is None else placement)


def test_idle_fog_and_busy_fog_are_preferred_if_they_finish_sooner():
    result = run([task("first"), task("busy-fog", 1), task("third", 1), task("overflow-to-cloud", 1)])
    assert [out.vm_id for out in result.outcomes] == ["z-fog", "z-fog", "z-fog", "a-cloud"]
    assert [out.start_time_ms for out in result.outcomes] == [1, 3, 5, 6]
    assert [out.finish_time_ms for out in result.outcomes] == [3, 5, 7, 8]
    decision = result.placement_decisions[1]
    fog = next(item for item in decision.candidates if item.vm_id == "z-fog")
    cloud = next(item for item in decision.candidates if item.vm_id == "a-cloud")
    assert (fog.queue_tail_finish_time_ms, fog.upload_ready_time_ms) == (3, 2)
    assert fog.predicted_finish_time_ms == 5 < cloud.predicted_finish_time_ms == 8
    assert decision.reason == "fog_no_slower_than_cloud"
    assert result.placement_decisions[-1].reason == "fog_unavailable"


def test_cloud_is_selected_when_busy_fog_would_finish_later():
    result = run([task("long", length=10000), task("short", 1)])
    assert [item.vm_id for item in result.outcomes] == ["z-fog", "a-cloud"]
    estimates = {item.vm_id: item.predicted_finish_time_ms for item in result.placement_decisions[1].candidates}
    assert estimates == {"a-cloud": 8, "z-fog": 23}
    assert result.placement_decisions[1].reason == "cloud_finishes_sooner"


def test_equal_finish_prefers_fog_even_if_cloud_id_sorts_first():
    result = run([task()], placement=settings(cloud_delay=0))
    assert result.outcomes[0].vm_id == "z-fog"
    assert {item.predicted_finish_time_ms for item in result.placement_decisions[0].candidates} == {3}


def test_network_size_and_bandwidth_can_change_the_selected_tier():
    # Fog has low fixed delay but slower serialization; large inputs favor cloud.
    links = FogCloudSettings("origin", (TransmissionLink("z-fog", 8, 0), TransmissionLink("a-cloud", 80, 2)))
    small, large = run([task(data=1)], placement=links), run([task(data=100)], placement=links)
    assert small.outcomes[0].vm_id == "z-fog"
    assert large.outcomes[0].vm_id == "a-cloud"
    assert large.outcomes[0].finish_time_ms == 14


def test_uploads_overlap_execution_and_do_not_get_added_twice():
    result = run([task("a"), task("b", 1)])
    estimate = next(item for item in result.placement_decisions[1].candidates if item.vm_id == "z-fog")
    assert estimate.completion_duration_ms == 4  # max(2ms backlog, 1ms upload) + 2ms service
    assert result.outcomes[1].finish_time_ms - result.outcomes[1].arrival_time_ms == 4


def test_queue_capacity_includes_uploading_tasks_and_rejects_without_losing_tasks():
    result = run([task("a"), task("b"), task("c")], topology=infrastructure(capacity=1))
    assert [item.vm_id for item in result.outcomes] == ["z-fog", "a-cloud", None]
    assert (result.completed_count, result.failed_count) == (2, 1)
    assert "queues are full" in result.outcomes[2].failure_reason
    assert result.placement_decisions[2].candidates == ()
    assert min(event.time_ms for event in result.resource_events) == 1


def test_head_of_line_upload_wait_preserves_fcfs_and_reservations():
    result = run([task("slow-upload", data=10), task("fast-upload", data=1)],
                 topology=infrastructure(cloud=False), placement=settings(cloud=False))
    assert [out.start_time_ms for out in result.outcomes] == [10, 12]
    assert [out.finish_time_ms for out in result.outcomes] == [12, 14]
    assert [event.time_ms for event in result.resource_events] == [10, 12, 12, 14]
    assert all(result.available_resources[vm.vm_id] == vm.resource_capacities
               for vm in infrastructure(cloud=False).vms)
    assert result.placement_decisions[0].reason == "only_fog_available"


def test_independent_uploads_and_upload_start_event_precede_same_time_arrivals():
    # First upload becomes execution at t=1, freeing a waiting slot before b arrives.
    result = run([task("a"), task("b", 1)], topology=infrastructure(capacity=1, cloud=False),
                 placement=settings(cloud=False))
    assert result.failed_count == 0
    assert [out.start_time_ms for out in result.outcomes] == [1, 3]


def test_same_time_finish_releases_then_starts_waiter_before_admitting_another():
    result = run([task("a"), task("b", 1), task("c", 3)],
                 topology=infrastructure(capacity=1, cloud=False), placement=settings(cloud=False))
    assert result.failed_count == 0
    assert [out.start_time_ms for out in result.outcomes] == [1, 3, 5]


def test_resource_infeasible_fog_falls_back_to_cloud_and_all_infeasible_rejects():
    result = run([task(demands={"cpu": 3, "memory": 4})])
    assert result.outcomes[0].vm_id == "a-cloud"
    rejected = run([task(demands={"cpu": 5, "memory": 4})])
    assert rejected.failed_count == 1
    assert "no reachable VM" in rejected.outcomes[0].failure_reason


def test_unlinked_fog_is_not_treated_as_local_or_available():
    original = infrastructure()
    remote = VM("remote", ResourceTier.FOG, 200, queue_capacity=3,
                resource_capacities={"cpu": 10, "memory": 100})
    topology = replace(original, hosts=(*original.hosts, Host("h-remote", 3, 32, 1000)),
                       vms=(*original.vms, remote),
                       fog_nodes=(*original.fog_nodes, FogNode("far", ("h-remote",), ("remote",))))
    result = run([task(demands={"cpu": 3, "memory": 4})], topology=topology)
    assert result.outcomes[0].vm_id == "a-cloud"
    assert "remote" not in {item.vm_id for item in result.placement_decisions[0].candidates}
    assert result.available_resources["remote"] == remote.resource_capacities


def test_forecasts_match_actual_finishes_without_future_information_and_inputs_stay_unchanged():
    topology, placement = infrastructure(), settings()
    tasks = [task("a"), task("b", 1), task("c", 2), task("d", 3)]
    before = deepcopy((topology, tasks, placement))
    prefix = run(tasks[:2], topology=topology, placement=placement)
    result = run(tasks, topology=topology, placement=placement)
    assert result.placement_decisions[:2] == prefix.placement_decisions
    assert (topology, tasks, placement) == before
    assert result == run(tasks, topology=topology, placement=placement)
    for decision, outcome in zip(result.placement_decisions, result.outcomes):
        selected = next(item for item in decision.candidates if item.vm_id == decision.selected_vm_id)
        assert selected.predicted_start_time_ms == pytest.approx(outcome.start_time_ms)
        assert selected.predicted_finish_time_ms == pytest.approx(outcome.finish_time_ms)


def test_empty_network_workload_has_restored_balances():
    result = run([])
    assert result.outcomes == () and result.makespan_ms == 0
    assert result.placement_config == settings().manifest()


@pytest.mark.parametrize("bandwidth,delay", [(0, 0), (-1, 0), (True, 0), (float("nan"), 0),
                                              (float("inf"), 0), (1, -1), (1, float("inf"))])
def test_invalid_link_values_rejected(bandwidth, delay):
    with pytest.raises((ValueError, TypeError)):
        TransmissionLink("vm", bandwidth, delay)


def test_decimal_units_and_unrepresentable_network_times():
    assert TransmissionLink("vm", 8, 2).upload_ms(20) == 22
    with pytest.raises(ValueError):
        TransmissionLink("vm", 1e-308, 0).upload_ms(1e308)
    with pytest.raises(ValueError, match="advance"):
        run([task(arrival=1e20)])


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "origin", "unit", "model", "unknown", "negative"])
def test_bad_placement_or_links_are_rejected(change):
    config = load_run_config(ROOT / "configs/fog_cloud.yaml").manifest()
    config.pop("sha256")
    data = config["placement"]
    if change == "missing":
        data["links"].pop()
    elif change == "extra":
        data["links"].append({"vm_id": "missing", "bandwidth_mbps": 10, "propagation_ms": 0})
    elif change == "duplicate":
        data["links"].append(deepcopy(data["links"][0]))
    elif change == "origin":
        data["source_end_node_id"] = "missing"
    elif change == "unit":
        data["data_size_unit"] = "KiB"
    elif change == "model":
        data["upload_model"] = "shared"
    elif change == "unknown":
        data["links"][0]["typo"] = 1
    else:
        data["links"][0]["propagation_ms"] = -1
    with pytest.raises((ValueError, TypeError)):
        parse_run_config(config)


def test_placement_config_round_trip_and_network_settings_change_manifest():
    config = load_run_config(ROOT / "configs/fog_cloud.yaml")
    data = config.manifest()
    checksum = data.pop("sha256")
    assert parse_run_config(data).manifest() == config.manifest()
    data["placement"]["links"][0]["bandwidth_mbps"] += 1
    assert parse_run_config(data).manifest()["sha256"] != checksum
    assert parse_placement_settings(config.placement.manifest()) == config.placement


@pytest.mark.parametrize("keyword", ["capture_decisions", "reward_config"])
def test_legacy_rl_capture_is_blocked_for_network_timing(keyword):
    from drlmots.reward import RewardConfig
    value = True if keyword == "capture_decisions" else RewardConfig(10, 1)
    with pytest.raises(ValueError, match="new state/reward contract"):
        simulate(infrastructure(), [task()], task_length_unit="bytes", policy="fog_first_completion",
                 placement_config=settings(), **{keyword: value})
