from copy import deepcopy
import json
import math
import random
from pathlib import Path
import zipfile

import pytest

from drlmots.objective_model import (ObjectiveModel, WorkItem, conditional_failure_probability,
    failure_trace, generate_work, load_objective_model, power_w)
from drlmots.objective_sim import MODEL_POLICIES, simulate_objectives
from drlmots.run_config import parse_run_config, load_run_config


def setup_model():
    config = parse_run_config(dict(schema_version=3, kind="synthetic_baseline", resource_units={"cpu": "units", "memory_gb": "GB"},
        topology=dict(hosts=[dict(host_id=h, storage_tb=1, memory_gb=16, bandwidth_reported=100) for h in ("hf", "hc")],
            vms=[dict(vm_id=v, tier=t, bandwidth_reported=100, queue_capacity=3, resource_capacities={"cpu": 1,"memory_gb":4})
                 for v,t in (("a", "fog"), ("b", "fog"), ("z", "cloud"))],
            fog_nodes=[dict(fog_node_id="f", host_ids=["hf"], vm_ids=["a","b"])],
            cloud_datacenters=[dict(datacenter_id="c", host_ids=["hc"], vm_ids=["z"])],
            access_points=[dict(access_point_id="ap", end_node_id="e")],
            end_nodes=[dict(end_node_id="e", fog_node_ids=["f"])]),
        placement=dict(policy="fog_first_completion", source_end_node_id="e", data_size_unit="decimal_kB", upload_model="parallel_no_contention",
                       links=[dict(vm_id=v, bandwidth_mbps=8, propagation_ms=0) for v in ("a","b","z")]),
        workload=dict(generator="uniform_interarrival", length_unit="bytes", arrival_time_unit="milliseconds", task_count=1, seed=0,
                      interarrival_ms_range=[0,0], length_bytes_range=[1,1], data_size_kb_range=[1,1],required_resources={"cpu":1})))
    model = ObjectiveModel(dict(schema_version=1, kind="host_network_objectives", coefficient_provenance="test coefficients", horizon_ms=100,
        workload=dict(task_count=2, seed=0, interarrival_ms=[0,0], compute_mi=[10,10], input_kb=[2,2], output_kb=[1,1],
                      relative_deadline_ms=None, required_resources={"cpu":1}),
        hosts={"hf":dict(capacity_mips=2000,power_curve=[[0,10],[1,110]],down_power_w=1,reboot_power_w=30),
               "hc":dict(capacity_mips=4000,power_curve=[[0,20],[1,220]],down_power_w=2,reboot_power_w=60)},
        vms={"a":dict(host_id="hf",mips=1000,link_id="f"),"b":dict(host_id="hf",mips=1000,link_id="f"),"z":dict(host_id="hc",mips=4000,link_id="c")},
        links={"f":dict(bandwidth_mbps=8,propagation_ms=0,joules_per_kb=.01,idle_power_w=0),
               "c":dict(bandwidth_mbps=8,propagation_ms=0,joules_per_kb=.02,idle_power_w=0)},
        failures={n:dict(distribution="exponential",scale_ms=100,shape=1,repair_ms=10,reboot_ms=2,detection_ms=2) for n in ("fog:f","cloud:c")},
        recovery=dict(max_retries=2,restart_ms=1,wait_ms=10,coordinator_power_w=10),
        objective=dict(weights=dict(time=.4,energy=.4,failure=.2),scales=dict(time_ms=100,energy_j=10,decision_time_ms=20,decision_energy_j=1),
                       minimum_completion_rate=.8, minimum_deadline_success_rate=.8)))
    return config,model


def task(name="t", arrival=0, mi=10, inputs=2, outputs=1, deadline=None, demands=None):
    return WorkItem(name,arrival,mi,inputs,outputs,deadline,{"cpu":1} if demands is None else demands)


def outage(start=5, repair=20, end=22, detect=7, node="fog:f"):
    return dict(node_id=node,start_ms=start,repair_end_ms=repair,end_ms=end,detection_ms=detect)


def run(c,m,tasks=None,**kwargs):
    return simulate_objectives(c,m,tasks=[task()] if tasks is None else tasks,policy=kwargs.pop("policy","first_fit_fcfs"),outages=kwargs.pop("outages",[]),**kwargs)


def test_mi_mips_host_energy_and_output_hand_calculation():
    c,m=setup_model()
    r=run(c,m)
    a=r["attempts"][0]
    assert (a["start_ms"],a["compute_end_ms"],a["end_ms"]) == (2,12,13)
    assert a["executed_mi"] == 10
    assert a["phase_ms"] == dict(upload=2,upload_propagation=0,queue=0,compute=10,download=1,download_propagation=0)
    assert a["host_dynamic_energy_j"] == pytest.approx(.5)
    assert a["network_energy_j"] == pytest.approx(.03)
    assert r["metrics"]["total_energy_j"] == pytest.approx(3.53)
    assert r["metrics"]["batch_makespan_ms"] == 13
    cloud=run(c,m,policy="fog_first_completion")
    assert cloud["decisions"][0]["selected_vm_id"] == "z"
    assert cloud["attempts"][0]["phase_ms"]["compute"] == 2.5
    assert cloud["metrics"]["batch_makespan_ms"] == 5.5


def test_two_vms_share_host_idle_and_link_bandwidth():
    c,m=setup_model()
    r=run(c,m,[task("one"),task("two")],policy="least_queue_fcfs")
    assert [a["vm_id"] for a in r["attempts"]] == ["a","b"]
    assert all(a["start_ms"] == 4 and a["end_ms"] == 16 for a in r["attempts"])
    assert r["host_energy"]["hf"]["idle_j"] == pytest.approx(1)
    assert r["host_energy"]["hf"]["dynamic_j"] == pytest.approx(1)
    assert r["metrics"]["total_energy_j"] == pytest.approx(4.06)
    assert all(s["transmitted_kb"] <= s["capacity_kb"]+1e-8 for s in r["link_segments"])


def test_piecewise_power_interpolation_and_integrated_joules():
    assert power_w([[0,10],[.5,50],[1,70]],.75) == 60
    c,m=setup_model()
    m.data["hosts"]["hf"]["power_curve"]=[[0,10],[.5,50],[1,70]]
    r=run(c,m)
    assert r["host_energy"]["hf"]["dynamic_j"] == pytest.approx(.4)
    assert sum(s["energy_j"] for s in r["power_segments"]) == pytest.approx(r["metrics"]["host_energy_j"])


def test_separate_detection_restart_and_repair_with_partial_attempt_energy():
    c,m=setup_model()
    r=run(c,m,outages=[outage()])
    a,b=r["attempts"]
    assert a["end_ms"] == 5 and a["phase_ms"]["compute"] == 3
    assert b["dispatch_ms"] == 8 and b["vm_id"] == "z" and b["end_ms"] == 13.5
    block=r["recovery_intervals"][0]
    assert (block["detection_ms"],block["restart_ms"],block["backup_wait_ms"]) == (2,1,0)
    assert block["energy_j"] == pytest.approx(.03)
    assert r["host_energy"]["hf"]["down_j"] == pytest.approx(.015)
    assert r["host_energy"]["hf"]["reboot_j"] == pytest.approx(.06)
    assert r["host_energy"]["hf"]["idle_j"] == pytest.approx(.83)
    # 0.83+0.15+0.015+0.06 fog, 2+0.5 cloud, .02+.06 bytes, .03 coordinator.
    assert r["metrics"]["total_energy_j"] == pytest.approx(3.665)
    assert r["metrics"]["recovered_task_count"] == 1
    assert r["metrics"]["mean_recovery_to_delivery_ms"] == 8.5
    assert r["availability"][1]["observed_availability"] == .83


def test_stale_health_does_not_peek_at_physical_failure():
    c,m=setup_model()
    r=run(c,m,[task(arrival=6)],outages=[outage()])
    first=r["decisions"][0]
    assert first["time_ms"] == 6 and first["selected_vm_id"] == "a"
    assert "fog:f" in first["observed_up_nodes"]
    assert r["attempts"][0]["end_ms"] == 6
    assert r["attempts"][1]["dispatch_ms"] == 8
    assert r["recovery_intervals"][0]["detection_ms"] == 1


def test_deadline_and_censoring_are_distinct_and_resources_released():
    c,m=setup_model()
    m.data["horizon_ms"]=10
    r=run(c,m)
    assert r["metrics"]["unfinished_count"] == 1
    assert r["metrics"]["execution_failure_count"] == 0
    assert r["metrics"]["batch_makespan_ms"] is None
    assert r["metrics"]["successful_makespan_ms"] is None
    deadline=run(c,m,[task(deadline=8)])
    assert deadline["metrics"]["deadline_missed_count"] == 1
    assert deadline["metrics"]["deadline_success_rate"] == 0
    unmatured=run(c,m,[task(deadline=20)])
    assert unmatured["metrics"]["deadline_success_rate"] is None
    assert unmatured["metrics"]["unfinished_count"] == 1
    for report in (r,deadline,unmatured):
        assert report["available_resources"] == {v.vm_id:dict(v.resource_capacities) for v in c.topology.vms}


def test_exact_delivery_deadline_and_fault_boundary_wins():
    c,m=setup_model()
    m.data["horizon_ms"]=13
    r=run(c,m,[task(deadline=13)],outages=[outage(start=13,repair=20,end=22,detect=15)])
    assert r["metrics"]["completed_count"] == 1
    assert r["metrics"]["deadline_success_rate"] == 1
    assert r["metrics"]["interrupted_attempt_count"] == 0


def test_output_failure_restarts_and_partial_bytes_charged():
    c,m=setup_model()
    r=run(c,m,outages=[outage(start=12.5,repair=20,end=22,detect=14.5)])
    a=r["attempts"][0]
    assert a["compute_end_ms"] == 12 and a["download_sent_kb"] == .5
    assert a["network_energy_j"] == pytest.approx(.025)
    assert r["attempts"][1]["executed_mi"] == 10
    assert r["metrics"]["completed_count"] == 1


def test_full_upload_during_propagation_but_no_computation():
    c,m=setup_model()
    m.data["links"]["f"]["propagation_ms"]=10
    r=run(c,m,outages=[outage()])
    a=r["attempts"][0]
    assert a["upload_sent_kb"] == 2 and a["executed_mi"] == 0
    assert a["phase_ms"]["upload_propagation"] == 3


def test_zero_input_and_output_have_no_serialization_but_keep_propagation():
    c,m=setup_model()
    m.data["links"]["f"]["propagation_ms"]=1
    r=run(c,m,[task(inputs=0,outputs=0)])
    assert r["attempts"][0]["start_ms"] == 1
    assert r["metrics"]["batch_makespan_ms"] == 12
    assert r["metrics"]["network_traffic_energy_j"] == 0


def test_no_recovery_waits_for_detection_and_fails_once():
    c,m=setup_model()
    r=run(c,m,outages=[outage()],recovery_enabled=False)
    assert r["metrics"]["execution_failure_count"] == 1
    assert r["outcomes"][0]["terminal_ms"] == 7
    assert len(r["attempts"]) == 1
    assert r["metrics"]["recovery_energy_j"] == pytest.approx(.02)


def test_shared_failure_domain_interrupts_both_vms():
    c,m=setup_model()
    r=run(c,m,[task("one"),task("two")],policy="least_queue_fcfs",outages=[outage()])
    interrupted=[a for a in r["attempts"] if a["status"] == "interrupted"]
    assert {a["vm_id"] for a in interrupted} == {"a","b"}
    assert all(a["end_ms"] == 5 for a in interrupted)


def test_recovery_waits_without_using_future_repair_timestamp():
    c,m=setup_model()
    trace=[outage(),outage(0,8,10,0,"cloud:c")]
    r=run(c,m,outages=trace)
    assert r["attempts"][1]["dispatch_ms"] == 10
    assert r["recovery_intervals"][0]["backup_wait_ms"] == 2
    assert r["decisions"][1]["selected_vm_id"] == "z"


def test_short_repair_before_detection_does_not_leave_node_falsely_down():
    c,m=setup_model()
    r=run(c,m,[task("first"),task("later",arrival=9)],outages=[outage(1,2,3,8)])
    decision=next(d for d in r["decisions"] if d["task_id"] == "later")
    assert "fog:f" in decision["observed_up_nodes"]


def test_future_failure_cannot_change_earlier_placement():
    c,m=setup_model()
    a=run(c,m)
    b=run(c,m,outages=[outage(70,80,82,72)])
    assert a["decisions"] == b["decisions"]
    assert a["outcomes"] == b["outcomes"]


def test_infeasible_and_empty_workloads_never_rank_as_energy_winners():
    c,m=setup_model()
    rejected=run(c,m,[task(demands={"cpu":99})])
    assert rejected["metrics"]["rejected_count"] == 1
    assert rejected["metrics"]["execution_failure_count"] == 0
    assert rejected["objective"]["score"] is None
    assert not rejected["objective"]["feasible"]
    empty=run(c,m,[])
    assert empty["metrics"]["completion_rate"] is None and empty["objective"]["score"] is None
    assert empty["metrics"]["total_energy_j"] == 3


def test_normalization_and_weights_are_fixed_linear_ratios():
    c,m=setup_model()
    r=run(c,m)
    assert r["objective"]["score"] == pytest.approx(.4*13/100+.4*3.53/10)
    m.data["objective"]["scales"]["energy_j"]=1
    r=run(c,m)
    assert r["objective"]["normalized"]["energy"] == pytest.approx(3.53) # no clipping


def test_distribution_reproducibility_local_rng_and_per_node_streams():
    c,m=setup_model()
    random.seed(989)
    state=random.getstate()
    a=failure_trace(m,77)
    assert a == failure_trace(m,77)
    assert a != failure_trace(m,78)
    generate_work(m,99)
    assert random.getstate() == state
    n=deepcopy(m.manifest())
    n["failures"]["extra"]=deepcopy(n["failures"]["fog:f"])
    assert [o for o in failure_trace(ObjectiveModel(n),77) if o["node_id"] != "extra"] == a
    for node in ("fog:f","cloud:c"):
        group=[o for o in a if o["node_id"] == node]
        assert all(x["end_ms"] < y["start_ms"] for x,y in zip(group,group[1:]))


def test_exponential_and_age_dependent_weibull_probabilities():
    c,m=setup_model()
    f=m.data["failures"]["fog:f"]
    assert conditional_failure_probability(f,40,10) == pytest.approx(1-math.exp(-.1))
    f.update(distribution="weibull",shape=2)
    assert conditional_failure_probability(f,40,10) == pytest.approx(1-math.exp(-(.5**2-.4**2)))
    assert conditional_failure_probability(f,50,10) > conditional_failure_probability(f,0,10)


def test_long_run_exponential_availability_and_mle_are_sane():
    c,m=setup_model()
    m.data["horizon_ms"]=100000
    r=simulate_objectives(c,m,tasks=[],failure_seed=515)
    for a in r["availability"]:
        assert a["configured_steady_state_availability"] == pytest.approx(100/112)
        assert abs(a["observed_availability"]-100/112) < .03
        assert abs(a["exponential_rate_mle_per_s"]-10) < 1


@pytest.mark.parametrize("mutation", [
    lambda d:d.update(unknown=True),
    lambda d:d["vms"]["a"].update(mips=3000),
    lambda d:d["vms"]["a"].update(host_id="missing"),
    lambda d:d["hosts"]["hf"].update(power_curve=[[0,30],[1,20]]),
    lambda d:d["links"]["f"].update(bandwidth_mbps=0),
    lambda d:d["failures"]["fog:f"].update(shape=2),
    lambda d:d["workload"].update(compute_mi=[0,1]),
    lambda d:d["workload"].update(seed=True),
    lambda d:d["objective"]["weights"].update(time=1),
    lambda d:d["objective"].update(minimum_completion_rate=1.1),
])
def test_invalid_contracts_fail_instead_of_silent_fallback(mutation):
    c,m=setup_model()
    d=m.manifest()
    mutation(d)
    with pytest.raises((ValueError,TypeError)):
        ObjectiveModel(d)


def test_cross_node_host_mapping_and_memory_overcommit_are_rejected():
    c,m=setup_model()
    m.data["vms"]["a"]["host_id"]="hc"
    m.data["hosts"]["hc"]["capacity_mips"]=10000
    with pytest.raises(ValueError,match="own node"):
        run(c,m)
    c,m=setup_model()
    c.topology.vms[0].resource_capacities["memory_gb"]=20
    with pytest.raises(ValueError,match="memory exceeds"):
        run(c,m)


@pytest.mark.parametrize("policy", MODEL_POLICIES)
def test_randomized_conservation_determinism_and_input_immutability(policy):
    c=load_run_config("configs/rl_network.yaml")
    m=load_objective_model("configs/objective_model.yaml")
    before=deepcopy((c,m.manifest()))
    for seed in range(5):
        a=simulate_objectives(c,m,policy=policy,workload_seed=6200+seed,failure_seed=7100+seed)
        assert a == simulate_objectives(c,m,policy=policy,workload_seed=6200+seed,failure_seed=7100+seed)
        metrics=a["metrics"]
        assert sum(metrics[k+"_count"] for k in ("completed","rejected","execution_failure","deadline_missed","unfinished")) == 60
        assert metrics["total_energy_j"] == pytest.approx(sum(metrics[k] for k in ("host_energy_j","network_traffic_energy_j","network_idle_energy_j","recovery_energy_j")))
        assert metrics["host_energy_j"] == pytest.approx(sum(s["energy_j"] for s in a["power_segments"]))
        assert sum(x["host_dynamic_energy_j"] for x in a["attempts"]) == pytest.approx(sum(e["dynamic_j"] for e in a["host_energy"].values()))
        assert sum(x["network_energy_j"] for x in a["attempts"]) == pytest.approx(metrics["network_traffic_energy_j"])
        assert all(0 <= s["utilization"] <= 1 for s in a["power_segments"])
        assert all(s["transmitted_kb"] <= s["capacity_kb"]+1e-8 for s in a["link_segments"])
        for o in a["outcomes"]:
            assert o["attempt_elapsed_ms"]+o["recovery_blocked_ms"] == pytest.approx(o["terminal_ms"]-o["arrival_ms"])
        assert a["available_resources"] == {v.vm_id:dict(v.resource_capacities) for v in c.topology.vms}
    assert (c,m.manifest()) == before


def test_retry_exhaustion_and_backup_wait_expiry_do_not_resurrect_tasks():
    c,m=setup_model()
    m.data["recovery"]["max_retries"]=1
    r=run(c,m,outages=[outage(),outage(11,30,32,13,"cloud:c")])
    assert len(r["attempts"]) == 2
    assert r["outcomes"][0]["status"] == "execution_failure"
    assert r["outcomes"][0]["terminal_ms"] == 13
    blocked=run(c,m,outages=[outage(),outage(0,50,52,0,"cloud:c")])
    assert blocked["outcomes"][0]["reason"] == "backup_wait_expired"
    assert blocked["outcomes"][0]["terminal_ms"] == 18


def test_cutoff_during_detection_or_restart_is_censored_and_billed_partially():
    c,m=setup_model()
    m.data["horizon_ms"]=6
    r=run(c,m,outages=[outage()])
    assert r["metrics"]["unfinished_count"] == 1
    assert r["recovery_intervals"][0]["detection_ms"] == 1
    assert r["metrics"]["recovery_energy_j"] == pytest.approx(.01)
    m.data["horizon_ms"]=7.5
    r=run(c,m,outages=[outage()])
    assert r["recovery_intervals"][0]["restart_ms"] == .5
    assert r["metrics"]["recovery_energy_j"] == pytest.approx(.025)


def test_upload_and_download_compete_on_the_same_shared_link():
    c,m=setup_model()
    # First output starts at 12, second upload joins at 12; each gets 0.5 kB/ms.
    r=run(c,m,[task("one"),task("two",arrival=12,inputs=2,outputs=0)],policy="least_queue_fcfs")
    first,second=r["attempts"]
    assert first["end_ms"] == 14
    assert second["start_ms"] == 15
    assert any(s["active_transfers"] == 2 and s["start_ms"] == 12 for s in r["link_segments"])


def test_empty_link_idle_and_down_host_power_are_still_accounted():
    c,m=setup_model()
    m.data["links"]["f"]["idle_power_w"]=5
    r=run(c,m,[],outages=[outage()])
    assert r["metrics"]["network_idle_energy_j"] == pytest.approx(.5)
    assert r["host_energy"]["hf"]["down_j"] > 0
    assert r["host_energy"]["hf"]["reboot_j"] > 0


def test_legacy_dqn_and_malformed_work_or_outages_are_rejected():
    c,m=setup_model()
    with pytest.raises(ValueError,match="DQN"):
        run(c,m,policy="dqn")
    with pytest.raises(ValueError,match="duplicate task"):
        run(c,m,[task(),task()])
    with pytest.raises(ValueError,match="cutoff"):
        run(c,m,[task(arrival=100)])
    with pytest.raises(ValueError,match="unknown outage"):
        run(c,m,outages=[outage(node="wrong")])
    with pytest.raises(ValueError,match="overlapping"):
        run(c,m,outages=[outage(),outage()])
    with pytest.raises(ValueError,match="ordering"):
        run(c,m,outages=[outage(detect=1)])


def test_weibull_shape_one_matches_exponential_reliability_and_mean():
    c,m=setup_model()
    f=m.data["failures"]["fog:f"]
    exp=conditional_failure_probability(f,30,20)
    f["distribution"]="weibull"
    assert conditional_failure_probability(f,30,20) == pytest.approx(exp)
    r=run(c,m)
    assert r["availability"][1]["configured_steady_state_availability"] == pytest.approx(100/112)


def test_duplicate_yaml_keys_rejected(tmp_path):
    p=tmp_path/"bad.yaml"
    p.write_text("schema_version: 1\nschema_version: 1\n",encoding="utf-8")
    with pytest.raises(ValueError,match="duplicate"):
        load_objective_model(p)


def test_fully_serialized_output_survives_host_failure_during_propagation():
    c,m=setup_model()
    m.data["links"]["f"]["propagation_ms"]=2
    # Input: 0-2 serialized, 2-4 propagation; compute 4-14; output 14-15, delivery 17.
    r=run(c,m,outages=[outage(16,20,22,18)])
    assert r["metrics"]["completed_count"] == 1
    assert r["metrics"]["batch_makespan_ms"] == 17
    assert r["metrics"]["interrupted_task_count"] == 0
    assert r["attempts"][0]["download_sent_kb"] == 1
