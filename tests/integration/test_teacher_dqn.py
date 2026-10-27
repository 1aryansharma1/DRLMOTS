from dataclasses import replace
import json

import pytest
import torch

from drlmots.enhanced_dqn import EnhancedDQNConfig, _observation
from drlmots.objective_model import WorkItem, load_objective_model
from drlmots.predictive_dqn import PREDICTIONS, PredictiveDQN
from drlmots.predictive_training import episode
from drlmots.run_config import load_run_config
from drlmots.teacher_dqn import ObservedTeacher, behavior_clone, dataset_from_reports, teacher_choice


def setup():
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    model.data["horizon_ms"] = 60
    model.data["workload"].update(task_count=4, interarrival_ms=[1, 2])
    settings = EnhancedDQNConfig(batch_size=2, warmup_transitions=2, replay_capacity=32, target_sync_updates=2)
    return run, model, settings


@pytest.fixture(scope="module")
def collection():
    torch.set_num_threads(1)
    run, model, settings = setup()
    teacher = ObservedTeacher.from_configs(run, model, settings)
    reports = [episode(teacher, run, model, workload_seed=50000+i, failure_seed=60000+i) for i in range(4)]
    return run, model, settings, teacher, reports, dataset_from_reports(reports)


def test_teacher_equation_deadline_filter_and_legal_actions(collection):
    _, _, _, teacher, _, data = collection
    for obs, expected in data:
        p = teacher.online.predictions(torch.tensor(obs.values)).reshape(6, len(PREDICTIONS))
        index = {f[0]: i for i, f in enumerate(PREDICTIONS)}
        c = teacher.contract.context
        weights, scales = c["objective"]["weights"], c["objective"]["scales"]
        costs = [weights["time"]*float(row[index["latency_ms"]])/scales["decision_time_ms"]
            + weights["energy"]*float(row[index["incremental_energy_j"]])/scales["decision_energy_j"]
            + weights["failure"]*float(row[index["failure_probability"]]) for row in p]
        legal = [i for i, b in enumerate(obs.legal_mask) if b]
        candidates = [i for i in legal if p[i, index["deadline_met"]] != 0] or legal
        best = min(candidates, key=lambda i:(costs[i], c["vms"][teacher.contract.vm_ids[i]]["tier"] == "cloud", teacher.contract.vm_ids[i]))
        action, q = teacher.select(obs)
        assert action == expected == best
        assert q == pytest.approx([-x for x in costs])


def test_teacher_rejects_training_and_exploration(collection):
    teacher, data = collection[3], collection[5]
    with pytest.raises(ValueError, match="without learning|must not update"):
        teacher.controller(learn=True)
    with pytest.raises(ValueError, match="deterministic"):
        teacher.select(data[0][0], .1)


def test_teacher_has_no_learned_weight_dependency_and_no_mutations(collection):
    teacher, data = collection[3], collection[5]
    before = teacher.fingerprint
    assert [teacher_choice(teacher, s)[0] for s, a in data] == [a for s, a in data]
    assert teacher.fingerprint == before
    other = ObservedTeacher(teacher.contract, replace(teacher.settings, seed=123), host_nodes=teacher.host_nodes)
    with torch.no_grad():
        for p in other.online.parameters():
            p.fill_(100)
    assert [other.select(s)[0] for s, a in data] == [a for s, a in data]


def test_hidden_future_outage_does_not_change_teacher_initial_decision():
    run, model, settings = setup()
    teacher = ObservedTeacher.from_configs(run, model, settings)
    work = [WorkItem("a",0,12,20,2,55,{"cpu_units":2,"memory_gb":4}),
            WorkItem("b",1,12,20,2,55,{"cpu_units":2,"memory_gb":4}),
            WorkItem("c",2,12,20,2,55,{"cpu_units":2,"memory_gb":4}),
            WorkItem("d",3,12,20,2,55,{"cpu_units":2,"memory_gb":4})]
    first = episode(teacher, run, model, tasks=work, outages=[])
    vm = teacher.contract.vm_ids[first["rl"]["decisions"][0]["action"]]
    node = teacher.contract.context["vms"][vm]["node_id"]
    faults = [dict(node_id=node,start_ms=2,repair_end_ms=20,end_ms=24,detection_ms=4)]
    second = episode(teacher, run, model, tasks=work, outages=faults)
    assert first["rl"]["decisions"][0] == second["rl"]["decisions"][0]
    recovery = [r for r in second["rl"]["decisions"] if r["observation"]["placement_kind"] == "recovery"]
    assert recovery
    for r in recovery:
        assert r["action"] is None or teacher.contract.context["vms"][teacher.contract.vm_ids[r["action"]]]["node_id"] != node
    assert second["rl"]["total_reward"] == pytest.approx(second["rl"]["expected_reward"])


def test_bc_changes_weights_and_agreement_without_polluting_rl_counters(collection, tmp_path):
    run, model, settings, _, _, data = collection
    a = PredictiveDQN.from_configs(run, model, settings)
    result = behavior_clone(a, data, epochs=30, batch_size=4)
    assert result["initial_online_sha256"] != result["final_online_sha256"]
    assert result["training_agreement_after"]["all"]["accuracy"] > result["training_agreement_before"]["all"]["accuracy"]
    assert result["history"][-1]["cross_entropy"] < result["history"][0]["cross_entropy"]
    assert (a.updates, a.decisions, a.episodes, len(a.replay), len(a.losses)) == (0,0,0,0,0)
    assert a.optimizer.state_dict()["state"] == {}
    assert all(torch.equal(v,a.target.state_dict()[k]) for k,v in a.online.state_dict().items())
    a.save(tmp_path/"cloned.json")
    b = PredictiveDQN.load(tmp_path/"cloned.json", a.contract, host_nodes=a.host_nodes)
    assert a.fingerprint == b.fingerprint
    before = a.fingerprint
    episode(a,run,model,workload_seed=51000,failure_seed=61000)
    assert a.fingerprint == before
    report_a = episode(a,run,model,learn=True,workload_seed=51000,failure_seed=61000)
    report_b = episode(b,run,model,learn=True,workload_seed=51000,failure_seed=61000)
    assert a.fingerprint == b.fingerprint and report_a == report_b
    assert a.updates > 0 and a.episodes == 1


@pytest.mark.parametrize("kwargs", [dict(epochs=0),dict(batch_size=0),dict(learning_rate=0)])
def test_bc_rejects_invalid_training_settings(collection, kwargs):
    run, model, settings, *_ = collection
    with pytest.raises(ValueError):
        behavior_clone(PredictiveDQN.from_configs(run,model,settings),collection[-1],**kwargs)


def test_bc_is_seeded_and_rejects_empty_illegal_or_used_agent(collection):
    run,model,settings,_,_,data = collection
    a = PredictiveDQN.from_configs(run,model,settings)
    b = PredictiveDQN.from_configs(run,model,settings)
    assert behavior_clone(a,data,epochs=2) == behavior_clone(b,data,epochs=2)
    assert a.fingerprint == b.fingerprint
    with pytest.raises(ValueError,match="nonempty"):
        behavior_clone(a,[])
    with pytest.raises(ValueError):
        behavior_clone(a,[(data[0][0],None)])
    episode(a,run,model,learn=True,workload_seed=51000,failure_seed=61000)
    with pytest.raises(ValueError,match="fresh"):
        behavior_clone(a,data)


def test_teacher_dataset_excludes_forced_events(collection):
    reports,data = collection[-2:]
    assert len(data) == sum(len(r["rl"]["transitions"]) for r in reports)
    assert all(s.legal_mask[a] for s,a in data)


def test_teacher_deadline_priority_tie_break_and_all_false_mask(collection, monkeypatch):
    teacher,data=collection[3],collection[-1]
    obs=data[0][0]
    width=len(PREDICTIONS)
    columns={f[0]:i for i,f in enumerate(PREDICTIONS)}
    p=torch.zeros((6,width))
    p[:,columns["latency_ms"]]=1
    p[1,columns["latency_ms"]]=100
    p[1,columns["deadline_met"]]=1
    monkeypatch.setattr(teacher.online,"predictions",lambda x:p.flatten())
    assert teacher.select(obs)[0]==1  # Deadline-feasible beats cheaper predicted miss.
    p[:,columns["deadline_met"]]=0
    p[:,columns["latency_ms"]]=1
    assert teacher.contract.vm_ids[teacher.select(obs)[0]]=="fog-a1"
    only_cloud=replace(obs,legal_mask=(True,False,False,False,False,False))
    assert teacher.select(only_cloud)[0]==0
    none=replace(obs,legal_mask=(False,)*6)
    assert teacher.select(none)==(None,())
