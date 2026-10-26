from copy import deepcopy
from dataclasses import replace
import json
import math
import random

import pytest
import torch

from drlmots.enhanced_dqn import (EnhancedDQN,EnhancedDQNConfig,EnhancedExperience,
    EnhancedQNetwork,masked_targets,run_enhanced_episode,validate_experience)
from drlmots.enhanced_rl_contract import (EncodedObservation,RewardDelta,RewardSnapshot,
    reward_delta,terminal_reward_snapshot)
from drlmots.enhanced_telemetry import ObservedTelemetry
from drlmots.objective_model import WorkItem,load_objective_model
from drlmots.objective_sim import simulate_objectives,MODEL_POLICIES
from drlmots.run_config import load_run_config


def setup(*,count=1,horizon=60,period=1,**settings):
    run=load_run_config("configs/rl_network.yaml")
    model=load_objective_model("configs/objective_model.yaml")
    model.data["horizon_ms"]=horizon
    agent=EnhancedDQN.from_configs(run,model,EnhancedDQNConfig(telemetry_period_ms=period,**settings),planned_task_count=count)
    # Explicit fixture policy: highest initial preference fog-a1, then cloud-1.
    with torch.no_grad():
        for p in agent.online.parameters(): p.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([5.,4.,10.,8.,3.,2.]))
        agent.target.load_state_dict(agent.online.state_dict())
    return run,model,agent


def work(name="one",arrival=0,mi=12,deadline=None,resources=None):
    return WorkItem(name,arrival,mi,0,0,deadline,{"cpu_units":2,"memory_gb":4} if resources is None else resources)


def outage(start=5,detect=7,repair=20,end=24,node="fog:fog-a"):
    return dict(node_id=node,start_ms=start,repair_end_ms=repair,end_ms=end,detection_ms=detect)


def inference(run,model,agent,tasks=None,**kwargs):
    return run_enhanced_episode(agent,run,model,tasks=[work()] if tasks is None else tasks,
                                outages=kwargs.pop("outages",[]),**kwargs)


def feature(obs,agent,name):
    index=next(i for i,f in enumerate(agent.contract.context["features"]) if f["name"]==name)
    return obs["values"][index]


def test_dqn_controls_initial_and_recovery_placement_with_frozen_learner():
    run,model,a=setup()
    before=a.fingerprint
    report=inference(run,model,a,outages=[outage()])
    assert [d["selected_vm_id"] for d in report["decisions"]] == ["fog-a1","cloud-1"]
    assert [d["retry"] for d in report["decisions"]] == [False,True]
    assert report["attempts"][0]["end_ms"] == 5
    assert report["attempts"][1]["dispatch_ms"] == 8
    assert report["metrics"]["completed_count"] == 1
    assert report["metrics"]["recovered_task_count"] == 1
    assert a.fingerprint==before and a.updates==a.decisions==a.episodes==0
    assert len(report["rl"]["transitions"])==2
    assert report["rl"]["transitions"][-1]["feedback"]["bootstrap_discount"]==0
    assert report["rl"]["total_reward"]==pytest.approx(report["rl"]["expected_reward"])
    terminal=terminal_reward_snapshot(a.contract,report)
    assert report["rl"]["total_reward"]==pytest.approx(reward_delta(a.contract,RewardSnapshot(),terminal,terminated=True).total)
    assert report["available_resources"]=={v.vm_id:dict(v.resource_capacities) for v in run.topology.vms}


def test_hidden_failure_does_not_clear_queues_until_delivered_information_changes():
    run,model,agent=setup(count=2,period=5)
    # No poll between 0 and 3. First task is queued in the reported view in both worlds.
    tasks=[work("first"),work("second",3)]
    normal=inference(run,model,agent,tasks)
    hidden=inference(run,model,agent,tasks,outages=[outage(start=1,detect=9,repair=20,end=24)])
    before=next(d for d in normal["rl"]["decisions"] if d["observation"]["task_id"]=="second")
    after=next(d for d in hidden["rl"]["decisions"] if d["observation"]["task_id"]=="second")
    assert before==after
    assert feature(after["observation"],agent,"vm[fog-a1].queue_count")>0
    assert feature(after["observation"],agent,"node[fog:fog-a].observed_up")==1
    assert hidden["attempts"][1]["end_ms"]==3 # physics stops a stale-health placement immediately
    assert not any(m["event"]=="failure_notice" and m["time_ms"]<9 for m in hidden["rl"]["telemetry_messages"])


def test_future_faults_cannot_change_earlier_decisions():
    run,model,a=setup(count=2)
    tasks=[work("first"),work("second",2)]
    normal=inference(run,model,a,tasks)
    future=inference(run,model,a,tasks,outages=[outage(start=40,detect=42,repair=50,end=54)])
    assert normal["rl"]["decisions"]==future["rl"]["decisions"]
    assert normal["outcomes"]==future["outcomes"]


def test_failure_detection_and_reboot_allow_future_placements_without_resurrection():
    run,model,a=setup(count=3)
    report=inference(run,model,a,[work("one"),work("two",9),work("three",26)],outages=[outage()])
    decisions={d["observation"]["task_id"]:d for d in report["rl"]["decisions"] if d["observation"]["placement_kind"]=="initial"}
    assert not decisions["two"]["observation"]["legal_mask"][2]
    assert decisions["three"]["observation"]["legal_mask"][2]
    assert feature(decisions["three"]["observation"],a,"node[fog:fog-a].detected_failures")==.5
    assert len(report["outcomes"])==3


def test_delayed_notice_after_repair_does_not_mark_a_repaired_node_down():
    run,model,a=setup(count=2)
    report=inference(run,model,a,[work("one"),work("two",12)],outages=[outage(1,10,2,3)])
    second=next(d for d in report["rl"]["decisions"] if d["observation"]["task_id"]=="two")
    assert second["observation"]["legal_mask"][2]
    assert any(m["event"]=="failure_notice" and m["stale"] for m in report["rl"]["telemetry_messages"])


def test_open_detection_and_restart_energy_is_accrued_before_interval_closes():
    run,model,a=setup(count=2)
    tasks=[work("first"),work("second",6)]
    report=inference(run,model,a,tasks,outages=[outage()])
    transition=report["rl"]["transitions"][0]
    assert transition["feedback"]["end_ms"]==6
    host=sum(s["energy_j"] for s in report["power_segments"] if s["end_ms"]<=6)
    # No bytes; network idle is 1.9 W. One active recovery has accrued 1 ms at 2 W.
    energy_to_six=host+1.9*6/1000+2*1/1000
    expected=-.4*energy_to_six/50
    assert transition["feedback"]["energy_reward"]==pytest.approx(expected)


def test_forced_prefix_is_logged_without_an_invented_action_or_replay_item():
    run,model,a=setup(count=2)
    report=inference(run,model,a,[work("impossible",resources={"cpu_units":99}),work("possible",10)])
    assert report["rl"]["decisions"][0]["action"] is None
    assert len(report["rl"]["transitions"])==1
    assert report["rl"]["prefix_reward"]["failure_reward"]==pytest.approx(-.1)
    assert report["rl"]["prefix_reward"]["energy_reward"]<0
    assert report["rl"]["total_reward"]==pytest.approx(report["rl"]["prefix_reward"]["energy_reward"]+
        report["rl"]["prefix_reward"]["failure_reward"]+sum(sum(x["feedback"][k] for k in
        ("energy_reward","time_reward","failure_reward","service_penalty_reward")) for x in report["rl"]["transitions"]))


def test_zero_legal_actions_has_a_terminal_prefix_and_zero_optimizer_updates():
    run,model,a=setup()
    report=inference(run,model,a,[work(resources={"cpu_units":99})],learn=True)
    assert report["rl"]["learnable_decisions"]==0 and not report["rl"]["transitions"]
    assert report["rl"]["prefix_reward"]["terminated"]
    assert a.updates==0 and len(a.replay)==0 and a.episodes==1


def test_forced_recovery_waits_are_folded_until_a_legal_target_or_terminal():
    run,model,a=setup(count=1)
    # Fog A fails; cloud and fog B also unavailable. The only eligible backups recover at 12.
    report=inference(run,model,a,outages=[outage(),outage(0,0,10,12,"cloud:cloud"),outage(0,0,10,12,"fog:fog-b")])
    forced=[d for d in report["rl"]["decisions"] if d["action"] is None]
    assert forced and all(not any(d["observation"]["legal_mask"]) for d in forced)
    assert len(report["rl"]["transitions"])==2
    first=report["rl"]["transitions"][0]
    assert first["next_state"]["time_ms"]==12
    assert any(first["next_state"]["legal_mask"])


def test_cutoff_during_recovery_preserves_censoring_and_releases_resources():
    run,model,a=setup(horizon=6)
    report=inference(run,model,a,outages=[outage()])
    assert report["metrics"]["unfinished_count"]==1
    assert report["metrics"]["execution_failure_count"]==0
    assert report["metrics"]["recovery_energy_j"]==pytest.approx(.002)
    assert report["rl"]["transitions"][-1]["feedback"]["terminated"]


def test_admission_cache_prevents_queue_overflow_between_sensor_reports():
    run,model,a=setup(count=20,period=10)
    report=inference(run,model,a,[work(str(i)) for i in range(20)])
    assert len(report["outcomes"])==20
    assert report["metrics"]["rejected_count"]==8 # two waiting admissions on each of six VMs at time zero
    assert report["available_resources"]=={v.vm_id:dict(v.resource_capacities) for v in run.topology.vms}


def observation(a,*,time=0,mask=None):
    return EncodedObservation(a.contract.sha256,time,"fixture","initial",(0.,)*len(a.contract.context["features"]),
                              (True,)*6 if mask is None else mask)


def experience(a,*,terminal=False,mask=None):
    s=observation(a)
    n=None if terminal else observation(a,time=1,mask=mask)
    r=RewardDelta(a.contract.sha256,0,60 if terminal else 1,-1,0,0,0,terminal,0 if terminal else 1)
    return EnhancedExperience(s,0,r,n)


def test_masks_apply_to_greedy_exploratory_and_bellman_choices():
    _,_,a=setup()
    mask=(True,False,False,False,False,False)
    obs=observation(a,mask=mask)
    assert a.select(obs,0)[0]==0
    assert all(a.select(obs,1)[0]==0 for _ in range(30))
    values=masked_targets(a.target,[experience(a,mask=mask),experience(a,terminal=True)])
    assert values.tolist()==pytest.approx([4,-1]) # illegal fog Q=10 cannot enter target
    with pytest.raises(ValueError,match="legal next"):
        validate_experience(a.contract,experience(a,mask=(False,)*6))


def test_optimizer_target_sync_and_warmup_are_connected():
    _,_,a=setup(batch_size=2,warmup_transitions=3,target_sync_updates=2)
    start=deepcopy(a.online.state_dict())
    for i in range(2): assert a.remember(experience(a)) is None
    assert a.updates==0
    assert a.remember(experience(a)) is not None
    assert a.updates==1
    assert any(not torch.equal(start[k],a.online.state_dict()[k]) for k in start)
    assert any(not torch.equal(a.target.state_dict()[k],a.online.state_dict()[k]) for k in start)
    a.remember(experience(a))
    assert a.updates==2
    assert all(torch.equal(a.target.state_dict()[k],a.online.state_dict()[k]) for k in start)


def test_frozen_inference_and_checkpoint_round_trip(tmp_path):
    run,model,a=setup()
    path=tmp_path/"agent.json"
    a.save(path)
    before=a.fingerprint
    report=inference(run,model,a,outages=[outage()])
    loaded=EnhancedDQN.load(path,a.contract,host_nodes=a.host_nodes)
    assert loaded.fingerprint==before==a.fingerprint
    assert inference(run,model,loaded,outages=[outage()])==report
    bad=json.loads(path.read_text())
    bad["kind"]="legacy_dqn_checkpoint"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="checkpoint"):
        EnhancedDQN.load(path,a.contract,host_nodes=a.host_nodes)


def test_optimizer_replay_and_rng_resume_exactly(tmp_path):
    _,_,a=setup(batch_size=2,warmup_transitions=2,target_sync_updates=3)
    for _ in range(4): a.remember(experience(a))
    a.decisions=4
    path=tmp_path/"trained-fixture.json"
    a.save(path)
    b=EnhancedDQN.load(path,a.contract,host_nodes=a.host_nodes)
    assert a.fingerprint==b.fingerprint
    assert a.select(observation(a),1)==b.select(observation(b),1)
    assert a.remember(experience(a))==pytest.approx(b.remember(experience(b)))
    assert a.fingerprint==b.fingerprint


def test_controller_cannot_be_reused_or_saved_mid_episode():
    run,model,a=setup()
    ctrl=a.controller()
    simulate_objectives(run,model,policy="masked_dqn",controller=ctrl,tasks=[work()],outages=[])
    with pytest.raises(ValueError,match="reuse"):
        simulate_objectives(run,model,policy="masked_dqn",controller=ctrl,tasks=[work()],outages=[])
    ctrl=a.controller()
    ctrl.begin(a.contract,a.host_nodes,True)
    with pytest.raises(ValueError,match="between"):
        a.artifact()


@pytest.mark.parametrize("changes",[
    dict(batch_size=0),dict(batch_size=20,warmup_transitions=10),dict(warmup_transitions=3000),
    dict(telemetry_period_ms=0),dict(epsilon_start=.01,epsilon_end=.5),dict(recovery_enabled=1),dict(seed=True)])
def test_invalid_settings(changes):
    with pytest.raises((ValueError,TypeError)):
        EnhancedDQNConfig(**changes)


def test_initialization_and_frozen_selection_preserve_global_rngs():
    random.seed(77);torch.manual_seed(77)
    py_state=random.getstate();torch_state=torch.random.get_rng_state().clone()
    run,model,a=setup()
    assert random.getstate()==py_state and torch.equal(torch.random.get_rng_state(),torch_state)
    inference(run,model,a)
    assert random.getstate()==py_state and torch.equal(torch.random.get_rng_state(),torch_state)


def test_original_four_rules_are_available_and_controller_is_opt_in():
    run,model,a=setup()
    for policy in MODEL_POLICIES:
        report=simulate_objectives(run,model,policy=policy,tasks=[work()],outages=[])
        assert "rl" not in report
    with pytest.raises(ValueError,match="requires"):
        simulate_objectives(run,model,policy="masked_dqn",tasks=[work()],outages=[])
    with pytest.raises(ValueError,match="requires"):
        simulate_objectives(run,model,controller=a.controller(),tasks=[work()],outages=[])


def test_telemetry_settings_and_recovery_mode_are_checkpoint_bound(tmp_path):
    run,model,a=setup()
    with pytest.raises(ValueError,match="mismatch"):
        simulate_objectives(run,model,policy="masked_dqn",controller=a.controller(),tasks=[work()],outages=[],recovery_enabled=False)
    b=EnhancedDQN(a.contract,replace(a.settings,telemetry_period_ms=5),host_nodes=a.host_nodes)
    assert a.environment_sha256!=b.environment_sha256
    path=tmp_path/"agent.json";a.save(path)
    other=EnhancedDQN.from_configs(run,model,planned_task_count=2)
    with pytest.raises(ValueError,match="context mismatch"):
        EnhancedDQN.load(path,other.contract,host_nodes=other.host_nodes)


@pytest.mark.parametrize("mutate",[
    lambda d:d.update(unknown=True),
    lambda d:d["optimizer"]["param_groups"][0].update(lr=4),
    lambda d:d["online"]["layers.0.weight"].update(dtype="torch.float64"),
    lambda d:d.update(updates=1,losses=[1]),
])
def test_rehashed_malformed_checkpoints_are_rejected(tmp_path,mutate):
    _,_,a=setup()
    data=a.artifact()
    mutate(data)
    data.pop("sha256")
    from drlmots.normalization import _digest
    data["sha256"]=_digest(data)
    path=tmp_path/"malformed.json";path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        EnhancedDQN.load(path,a.contract,host_nodes=a.host_nodes)


def test_mutable_replay_and_incompatible_context_are_rejected():
    _,_,a=setup()
    item=experience(a)
    with pytest.raises(ValueError,match="immutable"):
        a.remember(replace(item,state=replace(item.state,values=list(item.state.values))))
    with pytest.raises(ValueError,match="contract"):
        a.remember(replace(item,state=replace(item.state,context_sha256="0"*64)))
    with pytest.raises(ValueError):
        a.remember(replace(item,feedback=replace(item.feedback,terminated=1)))


@pytest.mark.parametrize("period",[.5,3,7])
def test_connected_randomized_conservation_and_immutability(period):
    run=load_run_config("configs/rl_network.yaml")
    model=load_objective_model("configs/objective_model.yaml")
    agent=EnhancedDQN.from_configs(run,model,EnhancedDQNConfig(telemetry_period_ms=period))
    before=deepcopy((run,model.manifest()))
    fingerprint=agent.fingerprint
    report=run_enhanced_episode(agent,run,model,workload_seed=6201,failure_seed=7101)
    m=report["metrics"]
    assert m["submitted_count"]==sum(m[k+"_count"] for k in ("completed","rejected","execution_failure","deadline_missed","unfinished"))
    assert m["host_energy_j"]==pytest.approx(sum(s["energy_j"] for s in report["power_segments"]))
    assert m["total_energy_j"]==pytest.approx(sum(m[k] for k in ("host_energy_j","network_traffic_energy_j","network_idle_energy_j","recovery_energy_j")))
    assert all(x["transmitted_kb"]<=x["capacity_kb"]+1e-8 for x in report["link_segments"])
    assert report["rl"]["total_reward"]==pytest.approx(report["rl"]["expected_reward"])
    for d in report["rl"]["decisions"]:
        assert d["action"] is None or d["observation"]["legal_mask"][d["action"]]
    assert (run,model.manifest())==before and agent.fingerprint==fingerprint


def test_unused_host_is_mapped_and_sensor_reported_explicitly():
    run,model,_=setup()
    host=replace(run.topology.hosts[0],host_id="hf-unused")
    fog=replace(run.topology.fog_nodes[0],host_ids=run.topology.fog_nodes[0].host_ids+("hf-unused",))
    run=replace(run,topology=replace(run.topology,hosts=run.topology.hosts+(host,),fog_nodes=(fog,)+run.topology.fog_nodes[1:]))
    model.data["hosts"]["hf-unused"]=deepcopy(model.data["hosts"]["hf-a"])
    agent=EnhancedDQN.from_configs(run,model,planned_task_count=1)
    report=inference(run,model,agent,outages=[outage()])
    assert "hf-unused" in report["host_energy"]
    assert agent.host_nodes["hf-unused"]=="fog:fog-a"


def test_cli_is_frozen_and_exports_a_complete_report(tmp_path):
    from drlmots.enhanced_dqn import main
    path=tmp_path/"inference.json"
    main(["--output",str(path)])
    report=json.loads(path.read_text())
    assert report["policy"]=="masked_dqn"
    assert report["rl"]["learner_updates"]==0 and not report["rl"]["learn"]
    assert report["rl"]["learnable_decisions"]>0
    assert "untrained" in report["rl"]["inference_scope"]


def test_one_episode_learning_plumbing_accepts_both_request_kinds_without_a_benchmark():
    # Deliberately below warmup: validate live replay wiring, not a training experiment.
    run,model,a=setup(epsilon_start=0,epsilon_end=0)
    report=inference(run,model,a,outages=[outage()],learn=True)
    assert a.decisions==2 and len(a.replay)==2 and a.updates==0 and a.episodes==1
    assert {x.state.placement_kind for x in a.replay}=={"initial","recovery"}
    assert report["rl"]["learn"]


def test_empty_workload_and_imprecise_sensor_budget_rejected_before_episode_start():
    run,model,a=setup()
    with pytest.raises(ValueError,match="positive"):
        inference(run,model,a,[])
    assert not a.episode_active
    b=EnhancedDQN(a.contract,replace(a.settings,telemetry_period_ms=1e-12),host_nodes=a.host_nodes)
    with pytest.raises(ValueError,match="polls"):
        inference(run,model,b)
    assert not b.episode_active


def test_shared_link_hidden_cleanup_is_not_read_into_the_next_observation():
    run,model,_=setup(count=3,period=5)
    model.data["vms"]["cloud-1"]["link_id"]="fog-a-access"
    agent=EnhancedDQN.from_configs(run,model,EnhancedDQNConfig(telemetry_period_ms=5),planned_task_count=3)
    with torch.no_grad():
        for p in agent.online.parameters(): p.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([5.,4.,10.,8.,3.,2.]))
    tasks=[WorkItem("cloud",0,12,50,10,None,{"cpu_units":3,"memory_gb":4}),
           WorkItem("fog",0,12,50,10,None,{"cpu_units":2,"memory_gb":4}),work("later",2)]
    healthy=inference(run,model,agent,tasks)
    failed=inference(run,model,agent,tasks,outages=[outage(1,7,20,24)])
    a=next(d for d in healthy["rl"]["decisions"] if d["observation"]["task_id"]=="later")
    b=next(d for d in failed["rl"]["decisions"] if d["observation"]["task_id"]=="later")
    assert a==b # actual sharing changed at 1, but no new sensor response was delivered yet


def test_completed_episode_checkpoint_saves_new_replay_and_round_trips(tmp_path):
    run,model,a=setup(epsilon_start=0,epsilon_end=0)
    inference(run,model,a,outages=[outage()],learn=True)
    path=tmp_path/"episode.json"
    a.save(path)
    b=EnhancedDQN.load(path,a.contract,host_nodes=a.host_nodes)
    assert b.fingerprint==a.fingerprint
    assert b.decisions==2 and b.episodes==1 and len(b.replay)==2 and b.updates==0
