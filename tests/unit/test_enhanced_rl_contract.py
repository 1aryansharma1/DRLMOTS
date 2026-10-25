from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from drlmots.enhanced_rl_contract import (EnhancedRLContract, PlacementRequest, QueuedTelemetry,
    VMTelemetry, HostTelemetry, LinkTelemetry, NodeTelemetry, ObservationSnapshot, RewardSnapshot,
    reward_delta, terminal_reward_snapshot)
from drlmots.objective_model import WorkItem, failure_trace, load_objective_model
from drlmots.objective_sim import MODEL_POLICIES, simulate_objectives
from drlmots.run_config import load_run_config


def settings(**kwargs):
    run=load_run_config("configs/rl_network.yaml")
    model=load_objective_model("configs/objective_model.yaml")
    return run,model,EnhancedRLContract.from_configs(run,model,**kwargs)


def snapshot(contract,**changes):
    c=contract.context
    request=PlacementRequest.from_work(WorkItem("current",0,7,30,5,80,{"cpu_units":2,"memory_gb":4}))
    result=ObservationSnapshot(10,1,0,0,request,
        tuple(VMTelemetry(v,0,None,(),tuple(sorted(spec["capacities"].items()))) for v,spec in c["vms"].items()),
        tuple(HostTelemetry(h,0,0.0) for h in c["hosts"]),
        tuple(LinkTelemetry(l,0,0,0,0,0) for l in c["links"]),
        tuple(NodeTelemetry(n,0,True,0) for n in c["failures"]))
    return replace(result,**changes)


def features(contract,observation):
    return dict(zip((f["name"] for f in contract.context["features"]),observation.values))


def change_vm(s,identifier,**changes):
    return replace(s,vms=tuple(replace(v,**changes) if v.vm_id==identifier else v for v in s.vms))


def test_frozen_versioned_observation_layout_and_stable_sorted_actions():
    run,model,c=settings()
    obs=c.encode(snapshot(c))
    assert c.vm_ids == ("cloud-1","cloud-2","fog-a1","fog-a2","fog-b1","fog-b2")
    assert c.context["contract_version"] == 4
    assert len(obs.values) == len(c.context["features"]) == 253
    assert obs.legal_mask == (True,)*6
    assert c.manifest()["sha256"] == obs.context_sha256
    assert all(math.isfinite(v) and 0<=v<=1 for v in obs.values)
    assert len({f["name"] for f in c.context["features"]}) == 253
    with pytest.raises(FrozenInstanceError):
        obs.values=()
    detached=c.context
    detached["objective"]["weights"]["time"]=99
    assert c.context["objective"]["weights"]["time"] == .4
    digest=c.sha256
    model.data["vms"]["fog-a1"]["mips"]=1100
    assert c.sha256 == digest


def test_normalization_uses_fixed_public_scales_without_held_out_fitting():
    _,_,c=settings()
    s=snapshot(c)
    f=features(c,c.encode(s))
    assert f["task.compute_mi"] == pytest.approx(7/(7+12))
    assert f["task.has_deadline"] == 1
    huge=replace(s,request=replace(s.request,compute_mi=1e300,input_kb=1e300))
    encoded=c.encode(huge)
    assert all(math.isfinite(x) and 0<=x<=1 for x in encoded.values)
    assert c.context["observation_scales"]["compute_mi"] == 12


def test_unknown_telemetry_has_explicit_flags_and_blocks_health_unknown_nodes():
    _,_,c=settings()
    s=snapshot(c)
    s=replace(s,nodes=tuple(replace(n,available=None,up_since_ms=None) for n in s.nodes),
              hosts=tuple(replace(h,utilization=None) for h in s.hosts))
    obs=c.encode(s)
    assert obs.legal_mask == (False,)*6
    f=features(c,obs)
    assert f["node[fog:fog-a].health_known"] == 0
    assert f["host[hf-a].utilization_known"] == 0
    assert c.validate_action(obs,None) is None


def test_resource_fit_uses_total_capacity_busy_vm_can_queue():
    _,_,c=settings()
    s=snapshot(c)
    s=change_vm(s,"fog-a1",running_remaining_ms=5,available_resources=(("cpu_units",0),("memory_gb",0)))
    obs=c.encode(s)
    assert obs.legal_mask[c.vm_ids.index("fog-a1")]
    demanding=replace(s,request=replace(s.request,resources=(("cpu_units",3),("memory_gb",4))))
    assert c.encode(demanding).legal_mask == (True,True,False,False,False,False)


def test_full_queue_and_ordered_slot_features():
    _,_,c=settings()
    s=snapshot(c)
    a=QueuedTelemetry("upload",8,0,3,(("cpu_units",2),))
    b=QueuedTelemetry("ready",0,0,10,(("memory_gb",4),))
    first=c.encode(change_vm(s,"fog-a1",waiting=(a,b)))
    second=c.encode(change_vm(s,"fog-a1",waiting=(b,a)))
    assert not first.legal_mask[c.vm_ids.index("fog-a1")]
    assert first.values != second.values
    f=features(c,first)
    assert f["vm[fog-a1].waiting[0].upload"] == 1
    assert f["vm[fog-a1].waiting[1].ready"] == 1
    with pytest.raises(ValueError,match="waiting queue"):
        c.encode(change_vm(s,"fog-a1",waiting=(a,a,a)))


def test_recovery_excludes_previous_node_and_preserves_both_tier_choices():
    _,_,c=settings()
    s=snapshot(c)
    s=replace(s,request=replace(s.request,attempts_used=1,previous_node_id="fog:fog-a",recovery_wait_remaining_ms=5))
    obs=c.encode(s)
    assert obs.placement_kind == "recovery"
    assert obs.legal_mask == (True,True,False,False,True,True)
    assert c.validate_action(obs,0) == "cloud-1"
    with pytest.raises(ValueError,match="eligible"):
        c.validate_action(obs,2)


def test_unreachable_vms_are_masked_by_static_source_topology():
    run,model,_=settings()
    run=replace(run,topology=replace(run.topology,end_nodes=(replace(run.topology.end_nodes[0],fog_node_ids=("fog-a",)),)),
                placement=replace(run.placement,links=tuple(l for l in run.placement.links if not l.vm_id.startswith("fog-b"))))
    c=EnhancedRLContract.from_configs(run,model)
    obs=c.encode(snapshot(c))
    assert obs.legal_mask == (True,True,True,True,False,False)


def test_expired_deadline_and_no_legal_action_are_environment_owned():
    _,_,c=settings()
    s=snapshot(c)
    obs=c.encode(replace(s,request=replace(s.request,deadline_ms=10)))
    assert not any(obs.legal_mask)
    assert c.validate_action(obs,None) is None
    with pytest.raises(ValueError,match="voluntary"):
        c.validate_action(c.encode(s),None)


@pytest.mark.parametrize("action",[True,-1,6,1.0,"fog-a1"])
def test_illegal_action_types_and_indices_are_rejected(action):
    _,_,c=settings()
    with pytest.raises(ValueError,match="eligible"):
        c.validate_action(c.encode(snapshot(c)),action)


def test_context_binds_reward_topology_and_normalization_but_not_seeds():
    run,model,c=settings()
    model.data["workload"]["seed"]=999
    run=replace(run,workload=replace(run.workload,seed=888))
    assert EnhancedRLContract.from_configs(run,model).sha256 == c.sha256
    assert "seed" not in json.dumps(c.context)
    model.data["objective"]["weights"].update(time=.5,energy=.3)
    other=EnhancedRLContract.from_configs(run,model)
    assert other.sha256 != c.sha256
    with pytest.raises(ValueError,match="incompatible"):
        other.validate_action(c.encode(snapshot(c)),0)


def test_future_failure_traces_have_no_observation_input_channel():
    _,model,c=settings()
    one,two=failure_trace(model,7100),failure_trace(model,7101)
    assert one != two
    s=snapshot(c)
    before=c.encode(s)
    # Neither trace is accepted by the strict telemetry API. Physics extraction is step 2.
    with pytest.raises(TypeError):
        ObservationSnapshot(**{**s.__dict__,"outages":one})
    with pytest.raises(TypeError):
        NodeTelemetry("fog:fog-a",0,True,0,physical_available=False)
    assert c.encode(s) == before
    assert not any("repair_end" in f["name"] or "failure_seed" in f["name"] for f in c.context["features"])


@pytest.mark.parametrize("change",[
    lambda s:replace(s,time_ms=400),
    lambda s:replace(s,request=replace(s.request,arrival_ms=11)),
    lambda s:replace(s,nodes=(replace(s.nodes[0],reported_at_ms=11),)+s.nodes[1:]),
    lambda s:replace(s,nodes=(replace(s.nodes[0],available=False,up_since_ms=0),)+s.nodes[1:]),
    lambda s:replace(s,nodes=s.nodes[:-1]),
    lambda s:replace(s,hosts=(replace(s.hosts[0],utilization=1.1),)+s.hosts[1:]),
    lambda s:replace(s,links=(replace(s.links[0],remaining_upload_kb=5),)+s.links[1:]),
    lambda s:replace(s,completed_count=1),
    lambda s:replace(s,request=replace(s.request,attempts_used=3,previous_node_id="fog:fog-a")),
    lambda s:replace(s,request=replace(s.request,previous_node_id="fog:fog-a")),
    lambda s:replace(s,request=replace(s.request,attempts_used=1)),
    lambda s:replace(s,request=replace(s.request,resources=(("unknown",1),))),
    lambda s:change_vm(s,"fog-a1",available_resources=(("cpu_units",0),("memory_gb",0))),
    lambda s:change_vm(s,"fog-a1",waiting=(QueuedTelemetry("ready",8,0,3),)),
])
def test_inconsistent_or_future_observations_rejected(change):
    _,_,c=settings()
    with pytest.raises((ValueError,TypeError)):
        c.encode(change(snapshot(c)))


def test_reward_components_have_a_hand_calculated_total():
    _,_,c=settings(planned_task_count=2)
    a=RewardSnapshot()
    b=RewardSnapshot(time_ms=20,total_energy_j=1,arrived_count=2,completed_count=1,successful_makespan_ms=15)
    r=reward_delta(c,a,b)
    assert r.energy_reward == pytest.approx(-.4/50)
    assert r.time_reward == pytest.approx(-.4*15/400)
    assert r.failure_reward == r.service_penalty_reward == 0
    assert r.total == pytest.approx(-.023)
    assert r.bootstrap_discount == 1 and not r.terminated
    z=replace(b,time_ms=400,total_energy_j=2,rejected_count=1,deadline_eligible_count=2,deadline_success_count=1)
    tail=reward_delta(c,b,z,terminated=True)
    assert tail.energy_reward == pytest.approx(-.008)
    assert tail.failure_reward == pytest.approx(-.1)
    assert tail.service_penalty_reward == pytest.approx(-6) # 10*(.8-.5) twice
    assert tail.bootstrap_discount == 0 and tail.terminated
    assert r.total+tail.total == pytest.approx(-6.131)


def test_interval_partitioning_and_delayed_failures_telescope():
    _,_,c=settings(planned_task_count=2)
    snapshots=[RewardSnapshot(),RewardSnapshot(5,.3,2),RewardSnapshot(30,1,2,1,successful_makespan_ms=25),
        RewardSnapshot(400,2,2,1,execution_failure_count=1,deadline_eligible_count=2,deadline_success_count=1,successful_makespan_ms=25)]
    pieces=[reward_delta(c,a,b,terminated=b.time_ms==400) for a,b in zip(snapshots,snapshots[1:])]
    total=reward_delta(c,snapshots[0],snapshots[-1],terminated=True)
    assert sum(p.total for p in pieces) == pytest.approx(total.total)
    assert pieces[0].failure_reward == pieces[1].failure_reward == 0
    assert pieces[-1].failure_reward == pytest.approx(-.1)
    assert pieces[0].energy_reward < 0 # costs remain even without a completion


def test_no_completions_gets_explicit_time_surrogate_and_floor_penalty():
    _,_,c=settings(planned_task_count=1)
    final=RewardSnapshot(400,1,1,rejected_count=1)
    delta=reward_delta(c,RewardSnapshot(),final,terminated=True)
    assert delta.time_reward == pytest.approx(-.4)
    assert delta.failure_reward == pytest.approx(-.2)
    assert delta.service_penalty_reward == -8
    assert final.successful_makespan_ms == 0 # report makespans are not fabricated


def test_cutoff_censoring_penalty_is_not_an_execution_failure():
    _,_,c=settings(planned_task_count=1)
    final=RewardSnapshot(400,2,1,unfinished_count=1)
    delta=reward_delta(c,RewardSnapshot(),final,terminated=True)
    assert final.execution_failure_count == 0
    assert delta.failure_reward == pytest.approx(-.2)
    assert delta.bootstrap_discount == 0


@pytest.mark.parametrize("before,after,terminated",[
    (RewardSnapshot(),RewardSnapshot(400,1,1,completed_count=1,successful_makespan_ms=20),False),
    (RewardSnapshot(),RewardSnapshot(10,1,1),True),
    (RewardSnapshot(20,2,1),RewardSnapshot(10,3,1),False),
    (RewardSnapshot(10,2,1),RewardSnapshot(20,1,1),False),
    (RewardSnapshot(10,2,1),RewardSnapshot(10,3,1),False),
    (RewardSnapshot(),RewardSnapshot(400,1,1),True),
    (RewardSnapshot(),RewardSnapshot(20,1,1,unfinished_count=1),False),
    (RewardSnapshot(),RewardSnapshot(20,1,1,completed_count=2),False),
    (RewardSnapshot(),RewardSnapshot(20,1,1,successful_makespan_ms=10),False),
    (RewardSnapshot(),RewardSnapshot(20,1,1,completed_count=1,successful_makespan_ms=21),False),
])
def test_invalid_reward_ledgers_or_episode_boundaries_are_rejected(before,after,terminated):
    _,_,c=settings(planned_task_count=1)
    with pytest.raises((ValueError,TypeError)):
        reward_delta(c,before,after,terminated=terminated)


@pytest.mark.parametrize("policy",MODEL_POLICIES)
def test_actual_enhanced_report_reward_equals_documented_objective(policy):
    run,model,c=settings()
    original=deepcopy((run,model.manifest()))
    report=simulate_objectives(run,model,policy=policy,workload_seed=6200,failure_seed=7100)
    final=terminal_reward_snapshot(c,report)
    delta=reward_delta(c,RewardSnapshot(),final,terminated=True)
    assert report["objective"]["feasible"]
    assert delta.total == pytest.approx(-report["objective"]["score"])
    assert final.total_energy_j == report["metrics"]["total_energy_j"]
    assert (run,model.manifest()) == original
    bad=deepcopy(report)
    bad["model_configuration"]["objective"]["weights"].update(time=.5,energy=.3)
    with pytest.raises(ValueError,match="contexts differ"):
        terminal_reward_snapshot(c,bad)


def test_contract_rejects_empty_training_episode_and_invalid_penalties():
    with pytest.raises(ValueError,match="positive declared"):
        settings(planned_task_count=0)
    with pytest.raises((ValueError,TypeError)):
        settings(completion_penalty=float("nan"))


@pytest.mark.parametrize("text",["{}","not-json",'{"contract_version":3,"kind":"live_scheduler_observation"}'])
def test_legacy_or_malformed_context_cannot_be_rehydrated_as_contract_four(text):
    with pytest.raises(ValueError):
        EnhancedRLContract(text)


def test_canonical_context_round_trip_and_invalid_feature_layout():
    _,_,c=settings()
    assert EnhancedRLContract(c._context_json).manifest() == c.manifest()
    malformed=c.context
    malformed["features"][0]["name"]="hidden_next_failure_ms"
    from drlmots.normalization import _canonical
    with pytest.raises(ValueError,match="layout"):
        EnhancedRLContract(_canonical(malformed))


def test_completed_reward_snapshot_cannot_hide_its_makespan():
    _,_,c=settings(planned_task_count=1)
    with pytest.raises(ValueError,match="makespan"):
        reward_delta(c,RewardSnapshot(),RewardSnapshot(20,1,1,completed_count=1))


@pytest.mark.parametrize("mutate",[
    lambda c:c["objective"]["weights"].update(time=-.4,energy=1.2),
    lambda c:c["failures"]["fog:fog-a"].update(next_failure_ms=50),
    lambda c:c.update(normalization="fit_on_test"),
    lambda c:c.update(discount=True),
])
def test_rehydrated_context_revalidates_public_semantics(mutate):
    _,_,c=settings()
    data=c.context
    mutate(data)
    from drlmots.normalization import _canonical
    with pytest.raises(ValueError):
        EnhancedRLContract(_canonical(data))
