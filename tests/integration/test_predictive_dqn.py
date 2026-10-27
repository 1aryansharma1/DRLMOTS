from dataclasses import replace
import json
import math

import pytest
import torch

from drlmots.enhanced_dqn import EnhancedDQN, EnhancedDQNConfig, run_enhanced_episode
from drlmots.enhanced_rl_contract import (EnhancedRLContract, HostTelemetry, LinkTelemetry,
    NodeTelemetry, ObservationSnapshot, PlacementRequest, QueuedTelemetry, VMTelemetry)
from drlmots.normalization import _digest
from drlmots.objective_model import WorkItem, conditional_failure_probability, load_objective_model, power_w
from drlmots.predictive_dqn import PREDICTIONS, PredictiveDQN
from drlmots.run_config import load_run_config


def setup():
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    contract = EnhancedRLContract.from_configs(run, model)
    agent = PredictiveDQN.from_configs(run, model)
    c = contract.context
    snap = ObservationSnapshot(10, 1, 0, 0,
        PlacementRequest.from_work(WorkItem("current", 0, 7, 30, 5, 80, {"cpu_units": 2, "memory_gb": 4})),
        tuple(VMTelemetry(v, 0, None, (), tuple(sorted(s["capacities"].items()))) for v, s in c["vms"].items()),
        tuple(HostTelemetry(h, 0, 0) for h in c["hosts"]),
        tuple(LinkTelemetry(l, 0, 0, 0, 0, 0) for l in c["links"]),
        tuple(NodeTelemetry(n, 0, True, 0) for n in c["failures"]))
    return run, model, agent, snap


def predictions(agent, snapshot):
    observation = agent.contract.encode(snapshot)
    values = torch.tensor(observation.values, dtype=torch.float32)
    matrix = agent.online.predictions(values).reshape(len(agent.contract.vm_ids), len(PREDICTIONS)).tolist()
    return {v: dict(zip((p[0] for p in PREDICTIONS), row)) for v, row in zip(agent.contract.vm_ids, matrix)}


@pytest.mark.parametrize("vm", ("cloud-1", "cloud-2", "fog-a1", "fog-a2", "fog-b1", "fog-b2"))
def test_idle_predictions_match_independent_physical_equations(vm):
    _, model, agent, snap = setup()
    c = agent.contract.context
    spec = c["vms"][vm]
    link, host = c["links"][spec["link_id"]], c["hosts"][spec["host_id"]]
    p = predictions(agent, snap)[vm]
    upload = 8*30/link["bandwidth_mbps"] + link["propagation_ms"]
    compute = 1000*7/spec["mips"]
    download = 8*5/link["bandwidth_mbps"] + link["propagation_ms"]
    latency = upload + compute + download
    energy = (power_w(host["power_curve"], spec["mips"]/host["capacity_mips"])-power_w(host["power_curve"], 0))*compute/1000 + 35*link["joules_per_kb"]
    expected = dict(upload_ms=upload, queue_wait_ms=0, compute_ms=compute, download_ms=download,
        latency_ms=latency, finish_ms=10+latency, incremental_energy_j=energy,
        deadline_slack_ms=70-latency, deadline_lateness_ms=0, deadline_met=1,
        failure_probability=conditional_failure_probability(c["failures"][spec["node_id"]], 10, latency), estimate_available=1)
    assert p == pytest.approx(expected, rel=2e-5, abs=2e-5)


def test_feature_layout_and_batch_encoding_leave_base_contract_and_masks_unchanged():
    _, _, agent, snap = setup()
    obs = agent.contract.encode(snap)
    x = torch.tensor(obs.values)
    before = x.clone()
    y = agent.online.augmented(x)
    assert len(obs.values) == 253 and len(y) == 325
    assert len(agent.features["features"]) == 72
    assert torch.equal(x, before) and torch.equal(y[:253], x)
    assert torch.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
    batch = agent.online.augmented(torch.stack([x, x]))
    assert torch.equal(batch[0], y) and torch.equal(batch[1], y)
    assert obs.legal_mask == (True,) * 6
    recovered = replace(snap, request=replace(snap.request, attempts_used=1, previous_node_id="fog:fog-a"))
    recovery_obs = agent.contract.encode(recovered)
    action, _ = agent.select(recovery_obs)
    assert recovery_obs.legal_mask[action] and action not in (2, 3)


def test_queue_and_half_duplex_backlog_predictions_are_sample_and_hold():
    _, _, agent, snap = setup()
    queue = (QueuedTelemetry("ready", 0, 0, 6),)
    snap = replace(snap, vms=tuple(replace(v, running_remaining_ms=20, waiting=queue,
        available_resources=(("cpu_units", 0), ("memory_gb", 0))) if v.vm_id == "fog-a1" else v for v in snap.vms),
        links=tuple(replace(l, upload_flows=1, download_flows=1, remaining_upload_kb=20, remaining_download_kb=10)
                    if l.link_id == "fog-a-access" else l for l in snap.links))
    p = predictions(agent, snap)["fog-a1"]
    c = agent.contract.context
    speed = c["vms"]["fog-a1"]["mips"]
    upload = 8*(20+10+30)/100 + .3
    tail = 20+1000*6/speed
    assert p["upload_ms"] == pytest.approx(upload)
    assert p["queue_wait_ms"] == pytest.approx(tail-upload, rel=2e-5)
    later = predictions(agent, replace(snap, time_ms=11))["fog-a1"]
    assert later["queue_wait_ms"] == pytest.approx(p["queue_wait_ms"], rel=2e-5)
    assert later["finish_ms"]-p["finish_ms"] == pytest.approx(1, abs=1e-4)


def test_unknown_telemetry_and_no_deadline_have_explicit_semantics():
    _, _, agent, snap = setup()
    snap = replace(snap, request=replace(snap.request, deadline_ms=None),
        nodes=tuple(replace(n, available=None, up_since_ms=None) for n in snap.nodes),
        hosts=tuple(replace(h, utilization=None) for h in snap.hosts))
    assert not any(agent.contract.encode(snap).legal_mask)
    for p in predictions(agent, snap).values():
        assert p["estimate_available"] == 0
        assert p["deadline_met"] == 1 and p["deadline_slack_ms"] == p["deadline_lateness_ms"] == 0


def test_none_failure_distribution_produces_zero_probability():
    run, model, _, snap = setup()
    for f in model.data["failures"].values():
        f["distribution"] = "none"
    agent = PredictiveDQN.from_configs(run, model)
    assert all(p["failure_probability"] == 0 for p in predictions(agent, snap).values())


def test_saturated_float32_telemetry_is_rejected_instead_of_silently_clipped():
    _, _, agent, snap = setup()
    x = torch.tensor(agent.contract.encode(snap).values)
    x[agent.online.index["task.compute_mi"]] = 1
    with pytest.raises(ValueError, match="saturates"):
        agent.online(x)


def test_hidden_failure_does_not_change_predictive_inputs_before_any_notice_or_sensor():
    run, model, _, _ = setup()
    model.data["workload"]["task_count"] = 2
    agent = PredictiveDQN.from_configs(run, model, EnhancedDQNConfig(telemetry_period_ms=10))
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        agent.online.layers[-1].bias[2] = 1
    tasks = [WorkItem(f"hidden-{i}", float(i), 7, 30, 5, 80, {"cpu_units": 2, "memory_gb": 4}) for i in range(2)]
    outage = dict(node_id="fog:fog-a", start_ms=.25, detection_ms=5, repair_end_ms=12, end_ms=16)
    a = run_enhanced_episode(agent, run, model, tasks=tasks, outages=[])
    b = run_enhanced_episode(agent, run, model, tasks=tasks, outages=[outage])
    for i in (0, 1):
        assert a["rl"]["decisions"][i]["observation"] == b["rl"]["decisions"][i]["observation"]
        assert a["rl"]["decisions"][i]["q_values"] == b["rl"]["decisions"][i]["q_values"]
    recovery = [d for d in b["rl"]["decisions"] if d["observation"]["placement_kind"] == "recovery" and d["action"] is not None]
    assert recovery and all(d["action"] not in (2, 3) for d in recovery)
    assert b["metrics"]["recovered_task_count"] == 2


def test_predictive_checkpoint_restores_learning_exactly_and_rejects_legacy(tmp_path):
    run, model, _, _ = setup()
    model.data["horizon_ms"] = 60
    model.data["workload"].update(task_count=4, interarrival_ms=[1, 2])
    settings = EnhancedDQNConfig(batch_size=2, warmup_transitions=2, replay_capacity=32)
    agent = PredictiveDQN.from_configs(run, model, settings)
    report = run_enhanced_episode(agent, run, model, learn=True, workload_seed=30000, failure_seed=40000)
    assert agent.updates > 0 and math.isfinite(report["rl"]["total_reward"])
    path = tmp_path / "predictive.json"
    agent.save(path)
    restored = PredictiveDQN.load(path, agent.contract, host_nodes=agent.host_nodes)
    assert agent.fingerprint == restored.fingerprint
    a = run_enhanced_episode(agent, run, model, learn=True, workload_seed=30001, failure_seed=40001)
    b = run_enhanced_episode(restored, run, model, learn=True, workload_seed=30001, failure_seed=40001)
    assert _digest(a) == _digest(b) and agent.fingerprint == restored.fingerprint
    before = agent.fingerprint
    run_enhanced_episode(agent, run, model, workload_seed=84000, failure_seed=94000)
    assert agent.fingerprint == before
    with pytest.raises(ValueError, match="kind/version"):
        EnhancedDQN.load(path, agent.contract)
    legacy = EnhancedDQN.from_configs(run, model, settings)
    legacy.save(tmp_path / "legacy.json")
    with pytest.raises(ValueError, match="kind/version"):
        PredictiveDQN.load(tmp_path / "legacy.json", agent.contract)
    data = json.loads(path.read_text())
    data["online"]["bandwidth"]["tensor"][0] += 1
    data.pop("sha256")
    data["sha256"] = _digest(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="constants"):
        PredictiveDQN.load(path, agent.contract)
