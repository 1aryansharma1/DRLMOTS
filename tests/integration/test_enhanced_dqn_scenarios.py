"""Milestone 3: independently calculated connected scenarios, not learning results."""
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from drlmots.enhanced_dqn import (EnhancedDQN, EnhancedDQNConfig, EnhancedExperience,
                                 run_enhanced_episode, validate_experience)
from drlmots.enhanced_rl_contract import EncodedObservation, RewardDelta
from drlmots.normalization import _digest
from drlmots.objective_model import load_objective_model, WorkItem
from drlmots.objective_sim import simulate_objectives
from drlmots.run_config import load_run_config


def fixture(count=1, *, horizon=100, period=1, recovery=True, propagation=0):
    """Six real configured VMs; simple coefficients permit pencil-and-paper totals.

    Two fog hosts: 2000 MIPS, 10->110 W. Cloud: 4000 MIPS, 20->220 W.
    Fog VMs: 1000 MIPS each; cloud VMs: 2000 each. Every link: 1 kB/ms.
    All links have zero idle power; transmission costs .01 J/kB fog, .02 cloud.
    """
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    d = model.data
    d["horizon_ms"] = horizon
    for host_id, host in d["hosts"].items():
        cloud = host_id == "hc"
        host.update(capacity_mips=4000 if cloud else 2000,
                    power_curve=[[0, 20 if cloud else 10], [1, 220 if cloud else 110]],
                    down_power_w=2 if cloud else 1, reboot_power_w=60 if cloud else 30)
    for vm_id, vm in d["vms"].items():
        vm["mips"] = 2000 if vm_id.startswith("cloud") else 1000
    for link_id, link in d["links"].items():
        link.update(bandwidth_mbps=8, propagation_ms=propagation,
                    joules_per_kb=.02 if link_id == "cloud-wan" else .01, idle_power_w=0)
    d["recovery"].update(max_retries=2, restart_ms=1, wait_ms=10, coordinator_power_w=10)
    d["objective"]["scales"].update(time_ms=100, energy_j=10)
    settings = EnhancedDQNConfig(telemetry_period_ms=period, recovery_enabled=recovery,
                                 epsilon_start=0, epsilon_end=0)
    agent = EnhancedDQN.from_configs(run, model, settings, planned_task_count=count)
    # Explicit test weights: exercise the live Q-network with predictable choices.
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([5., 4., 10., 8., 3., 2.]))
        agent.target.load_state_dict(agent.online.state_dict())
    return run, model, agent


def task(name="one", arrival=0, *, mi=10, inputs=2, outputs=1, deadline=None, cpu=2):
    return WorkItem(name, arrival, mi, inputs, outputs, deadline, {"cpu_units": cpu, "memory_gb": 4})


def fault(start=5, detection=7, repair=20, end=22, node="fog:fog-a"):
    return dict(node_id=node, start_ms=start, detection_ms=detection, repair_end_ms=repair, end_ms=end)


def episode(run, model, agent, tasks=None, *, outages=(), **kwargs):
    return run_enhanced_episode(agent, run, model, tasks=[task()] if tasks is None else tasks,
                                outages=outages, **kwargs)


def check_accounting(report, run):
    """Independent ledger/reward audit, without calling the reward implementation."""
    m, rl = report["metrics"], report["rl"]
    n = m["submitted_count"]
    unsuccessful = sum(m[k + "_count"] for k in ("rejected", "execution_failure", "deadline_missed", "unfinished"))
    assert m["completed_count"] + unsuccessful == n == len(report["outcomes"])
    assert len({o["task_id"] for o in report["outcomes"]}) == n
    host_j = math.fsum(s["power_w"] * (s["end_ms"] - s["start_ms"]) / 1000 for s in report["power_segments"])
    traffic_j = math.fsum(a["network_energy_j"] for a in report["attempts"])
    d = report["model_configuration"]
    idle_j = sum(l["idle_power_w"] for l in d["links"].values()) * d["horizon_ms"] / 1000
    recovery_j = math.fsum((b["detection_ms"] + b["restart_ms"]) * d["recovery"]["coordinator_power_w"] / 1000 for b in report["recovery_intervals"])
    assert m["host_energy_j"] == pytest.approx(host_j)
    assert m["total_energy_j"] == pytest.approx(host_j + traffic_j + idle_j + recovery_j)
    assert report["available_resources"] == {v.vm_id: dict(v.resource_capacities) for v in run.topology.vms}
    assert all(s["transmitted_kb"] <= s["capacity_kb"] + 1e-8 for s in report["link_segments"])
    objective = d["objective"]
    makespan = m["successful_makespan_ms"] if m["completed_count"] else d["horizon_ms"]
    components = dict(
        energy_reward=-objective["weights"]["energy"] * m["total_energy_j"] / objective["scales"]["energy_j"],
        time_reward=-objective["weights"]["time"] * makespan / objective["scales"]["time_ms"],
        failure_reward=-objective["weights"]["failure"] * unsuccessful / n,
        service_penalty_reward=-10 * max(0, objective["minimum_completion_rate"] - m["completed_count"] / n))
    if m["deadline_eligible_count"]:
        components["service_penalty_reward"] -= 10 * max(0, objective["minimum_deadline_success_rate"] -
            m["deadline_success_count"] / m["deadline_eligible_count"])
    for key, expected in components.items():
        actual = rl["prefix_reward"][key] + math.fsum(t["feedback"][key] for t in rl["transitions"])
        assert actual == pytest.approx(expected, abs=1e-10)
    assert rl["total_reward"] == pytest.approx(math.fsum(components.values()))
    for transition in rl["transitions"]:
        state, action, feedback = transition["state"], transition["action"], transition["feedback"]
        assert type(action) is int and state["legal_mask"][action]
        assert feedback["start_ms"] == state["time_ms"]
        if feedback["terminated"]:
            assert feedback["end_ms"] == d["horizon_ms"] and transition["next_state"] is None
            assert feedback["bootstrap_discount"] == 0
        else:
            assert feedback["end_ms"] == transition["next_state"]["time_ms"]
            assert any(transition["next_state"]["legal_mask"]) and feedback["bootstrap_discount"] == 1


def test_no_fault_time_energy_and_return_are_independently_calculated():
    run, model, agent = fixture()
    report = episode(run, model, agent)
    attempt = report["attempts"][0]
    assert (attempt["start_ms"], attempt["compute_end_ms"], attempt["end_ms"]) == (2, 12, 13)
    # 40 W system idle for .1 s + 50 W dynamic for .01 s + 3 kB * .01 J/kB.
    assert report["metrics"]["total_energy_j"] == pytest.approx(4 + .5 + .03)
    assert report["rl"]["total_reward"] == pytest.approx(-.2332)
    check_accounting(report, run)


def test_retry_partial_work_and_delayed_reward_have_closed_form_totals():
    run, model, agent = fixture()
    report = episode(run, model, agent, outages=[fault()])
    assert [a["vm_id"] for a in report["attempts"]] == ["fog-a1", "cloud-1"]
    assert [(a["dispatch_ms"], a["end_ms"]) for a in report["attempts"]] == [(0, 5), (8, 16)]
    assert report["attempts"][0]["executed_mi"] == 3
    assert report["attempts"][1]["executed_mi"] == 10
    # Fog A .83 idle + .15 partial dynamic + .015 down + .06 reboot;
    # Fog B 1 idle; cloud 2 idle + .5 dynamic; bytes .02+.06; recovery .03.
    assert report["metrics"]["total_energy_j"] == pytest.approx(4.665)
    first = report["rl"]["transitions"][0]["feedback"]
    assert first["end_ms"] == 8
    assert first["energy_reward"] == pytest.approx(-.4 * .493 / 10)
    assert first["time_reward"] == first["failure_reward"] == 0
    assert report["rl"]["total_reward"] == pytest.approx(-.2506)
    check_accounting(report, run)


@pytest.mark.parametrize("fraction", [.1, .2, .3, .7])
def test_fractional_recovery_clock_cannot_exceed_its_wait_budget_by_roundoff(fraction):
    run, model, agent = fixture()
    report = episode(run, model, agent, outages=[fault(start=5 + fraction, detection=7 + fraction)])
    assert report["metrics"]["recovered_task_count"] == 1
    assert report["attempts"][1]["dispatch_ms"] == pytest.approx(8 + fraction)
    features = agent.contract.context["features"]
    index = next(i for i, f in enumerate(features) if f["name"] == "task.recovery_wait_remaining_ms")
    recovery = next(d["observation"] for d in report["rl"]["decisions"] if d["observation"]["placement_kind"] == "recovery")
    assert recovery["values"][index] <= 10 / (10 + features[index]["scale"])
    check_accounting(report, run)


def test_simultaneous_recoveries_have_exactly_zero_energy_between_same_time_choices():
    run, model, agent = fixture(count=3)
    report = episode(run, model, agent, [task("first"), task("second", .5), task("third", 1)],
                     outages=[fault(start=5.1, detection=7.1)])
    same_time = [t for t in report["rl"]["transitions"]
                 if t["state"]["placement_kind"] == "recovery" and t["feedback"]["start_ms"] == t["feedback"]["end_ms"]]
    assert len(same_time) == 2
    assert all(t["feedback"]["energy_reward"] == 0 for t in same_time)
    check_accounting(report, run)


@pytest.mark.parametrize("period", [.3, 1, 7])
def test_reward_and_physics_do_not_depend_on_sensor_partition_for_one_task(period):
    run, model, agent = fixture(period=period)
    report = episode(run, model, agent, outages=[fault()])
    assert report["metrics"]["total_energy_j"] == pytest.approx(4.665)
    assert report["metrics"]["successful_makespan_ms"] == pytest.approx(16)
    assert report["rl"]["total_reward"] == pytest.approx(-.2506)
    check_accounting(report, run)


@pytest.mark.parametrize("start,phase", [(1, "upload"), (2.5, "upload_propagation"),
                                         (5, "compute"), (13.5, "download")])
def test_faults_during_each_live_phase_restart_on_another_domain(start, phase):
    run, model, agent = fixture(propagation=1)
    report = episode(run, model, agent, outages=[fault(start=start, detection=start + 2, repair=30, end=32)])
    first, second = report["attempts"]
    assert first["status"] == "interrupted" and first["phase_ms"][phase] > 0
    assert second["node_id"] != first["node_id"] and second["executed_mi"] == 10
    assert report["metrics"]["recovered_task_count"] == 1
    check_accounting(report, run)


def test_fully_serialized_output_survives_its_host_failure():
    run, model, agent = fixture(propagation=1)
    report = episode(run, model, agent, outages=[fault(start=14.5, detection=16.5, repair=30, end=32)])
    assert len(report["attempts"]) == 1 and report["outcomes"][0]["finish_ms"] == 15
    assert report["metrics"]["interrupted_attempt_count"] == 0
    check_accounting(report, run)


def test_queued_and_running_work_share_failure_domain_but_reserve_only_while_computing():
    run, model, agent = fixture(count=2)
    report = episode(run, model, agent, [task("first"), task("queued", .5)], outages=[fault(start=8, detection=10)])
    queued = next(a for a in report["attempts"] if a["task_id"] == "queued" and a["number"] == 1)
    assert queued["phase_ms"]["queue"] > 0 and queued["executed_mi"] == 0
    assert report["metrics"]["interrupted_attempt_count"] == 2
    assert report["metrics"]["recovered_task_count"] == 2
    check_accounting(report, run)


@pytest.mark.parametrize("third_fault", [False, True])
def test_repeated_failures_obey_retry_budget_and_finalize_each_task_once(third_fault):
    run, model, agent = fixture()
    outages = [fault(), fault(start=12, detection=14, repair=40, end=42, node="cloud:cloud")]
    if third_fault:
        outages.append(fault(start=20, detection=22, repair=40, end=42, node="fog:fog-b"))
    report = episode(run, model, agent, outages=outages)
    assert [a["node_id"] for a in report["attempts"]] == ["fog:fog-a", "cloud:cloud", "fog:fog-b"]
    assert report["metrics"]["retry_attempt_count"] == 2
    assert report["outcomes"][0]["status"] == ("execution_failure" if third_fault else "completed")
    assert len(report["outcomes"][0]["attempt_ids"]) == 3
    check_accounting(report, run)


def test_disabled_recovery_charges_only_partial_work_and_detection():
    run, model, agent = fixture(recovery=False)
    report = episode(run, model, agent, outages=[fault()])
    assert len(report["attempts"]) == 1 and report["outcomes"][0]["terminal_ms"] == 7
    assert report["metrics"]["total_energy_j"] == pytest.approx(4.095)
    assert report["metrics"]["recovery_energy_j"] == pytest.approx(.02)
    assert report["rl"]["total_reward"] == pytest.approx(-8.7638)
    check_accounting(report, run)


@pytest.mark.parametrize("horizon,deadline,status", [(100, 8, "deadline_missed"), (8, 20, "unfinished"),
                                                   (8, 8, "deadline_missed"), (100, 0, "deadline_missed")])
def test_deadline_cutoff_and_no_completion_surrogate_remain_distinct(horizon, deadline, status):
    run, model, agent = fixture(horizon=horizon)
    report = episode(run, model, agent, [task(deadline=deadline)])
    assert report["outcomes"][0]["status"] == status
    assert report["metrics"]["successful_makespan_ms"] is None
    assert report["metrics"]["deadline_eligible_count"] == int(deadline <= horizon)
    check_accounting(report, run)


@pytest.mark.parametrize("period", [.3, 1, 7])
def test_exact_output_deadline_cutoff_and_failure_tie_prefers_delivery(period):
    run, model, agent = fixture(horizon=13, period=period)
    report = episode(run, model, agent, [task(deadline=13)], outages=[fault(start=13, detection=15)])
    assert report["metrics"]["completed_count"] == 1 and report["metrics"]["interrupted_attempt_count"] == 0
    check_accounting(report, run)


def test_failure_and_completion_costs_are_not_assigned_before_they_occur():
    run, model, agent = fixture(count=3, recovery=False)
    report = episode(run, model, agent, [task("failed"), task("before_notice", 6), task("after_notice", 9)], outages=[fault()])
    transitions = report["rl"]["transitions"]
    assert transitions[0]["feedback"]["end_ms"] == 6
    assert transitions[0]["feedback"]["failure_reward"] == 0
    assert transitions[1]["feedback"]["end_ms"] == 9
    assert transitions[1]["feedback"]["failure_reward"] == pytest.approx(-.2 * 2 / 3)
    assert all(t["feedback"]["time_reward"] == 0 for t in transitions[:-1])
    check_accounting(report, run)


@pytest.mark.parametrize("changes", [dict(repair=30, end=35), dict(repair=60, end=70)])
def test_future_repair_changes_cannot_change_an_earlier_recovery_choice(changes):
    run, model, agent = fixture()
    a = episode(run, model, agent, outages=[fault()])
    b = episode(run, model, agent, outages=[fault(**changes)])
    # Same delivered information at initial/recovery decisions, different future repair schedule.
    assert a["rl"]["decisions"] == b["rl"]["decisions"]
    assert a["outcomes"] == b["outcomes"]
    assert a["metrics"]["total_energy_j"] != b["metrics"]["total_energy_j"]


def test_an_old_notice_during_a_newer_failure_does_not_restore_health():
    run, model, agent = fixture(count=3)
    report = episode(run, model, agent, [task("old"), task("new", 4), task("after", 11)], outages=[
        fault(start=1, detection=10, repair=2, end=3), fault(start=5, detection=7, repair=20, end=22)])
    later = next(d for d in report["rl"]["decisions"] if d["observation"]["task_id"] == "after")
    assert not later["observation"]["legal_mask"][2]
    check_accounting(report, run)


@pytest.mark.parametrize("period", [1, 5])
def test_running_resources_and_host_power_follow_delivered_reports_not_hidden_cleanup(period):
    run, model, agent = fixture(count=2, period=period)
    tasks = [task("running", mi=30, inputs=0), task("later", 7)]
    normal = episode(run, model, agent, tasks)
    hidden = episode(run, model, agent, tasks, outages=[fault(start=6, detection=12)])
    a = next(d for d in normal["rl"]["decisions"] if d["observation"]["task_id"] == "later")
    b = next(d for d in hidden["rl"]["decisions"] if d["observation"]["task_id"] == "later")
    names = [f["name"] for f in agent.contract.context["features"]]
    value = lambda observation, name: observation["values"][names.index(name)]
    assert value(b["observation"], "vm[fog-a1].running") == 1
    assert value(b["observation"], "vm[fog-a1].resource[cpu_units].available") == 0
    assert value(b["observation"], "host[hf-a].observed_utilization") > 0
    if period == 5:
        # Both worlds last reported at 5: identical complete policy inputs and Q/action.
        assert a == b
    else:
        # Missing responses are allowed evidence: healthy at 7 versus failed last at 5.
        assert a["observation"] != b["observation"]
        assert value(a["observation"], "node[fog:fog-a].telemetry_age_ms") == 0
        assert value(b["observation"], "node[fog:fog-a].telemetry_age_ms") > 0
    check_accounting(hidden, run)


@pytest.mark.parametrize("horizon,deadline,status,end", [(100, None, "execution_failure", 18),
    (100, 12, "deadline_missed", 12), (10, None, "unfinished", 10)])
def test_recovery_with_no_backup_ends_at_wait_expiry_deadline_or_cutoff(horizon, deadline, status, end):
    run, model, agent = fixture(horizon=horizon)
    report = episode(run, model, agent, [task(deadline=deadline)], outages=[fault(),
        fault(0, 0, 30, 32, "cloud:cloud"), fault(0, 0, 30, 32, "fog:fog-b")])
    assert len(report["attempts"]) == 1
    assert (report["outcomes"][0]["status"], report["outcomes"][0]["terminal_ms"]) == (status, end)
    assert len(report["rl"]["transitions"]) == 1
    assert report["metrics"]["recovery_energy_j"] == pytest.approx(.03)
    check_accounting(report, run)


def test_live_initial_and_recovery_masks_respect_source_reachability():
    run, model, _ = fixture()
    end = replace(run.topology.end_nodes[0], fog_node_ids=("fog-b",))
    placement = replace(run.placement, links=tuple(l for l in run.placement.links if not l.vm_id.startswith("fog-a")))
    run = replace(run, topology=replace(run.topology, end_nodes=(end,)), placement=placement)
    agent = EnhancedDQN.from_configs(run, model, planned_task_count=1)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([5., 4., 10., 8., 3., 2.]))
    report = episode(run, model, agent, outages=[fault(node="cloud:cloud")])
    assert [a["vm_id"] for a in report["attempts"]] == ["cloud-1", "fog-b1"]
    assert all(not any(d["observation"]["legal_mask"][2:4]) for d in report["rl"]["decisions"])
    check_accounting(report, run)


def test_recovery_never_uses_resource_infeasible_fog_when_only_cloud_can_fit():
    run, model, agent = fixture()
    report = episode(run, model, agent, [task(cpu=3)], outages=[fault(node="cloud:cloud")])
    assert len(report["attempts"]) == 1 and report["attempts"][0]["vm_id"] == "cloud-1"
    assert report["outcomes"][0]["status"] == "execution_failure"
    assert all(not any(d["observation"]["legal_mask"][2:]) for d in report["rl"]["decisions"])
    check_accounting(report, run)


def test_busy_but_total_capacity_feasible_vm_can_queue_another_task():
    run, model, agent = fixture(count=2)
    report = episode(run, model, agent, [task("first", inputs=0), task("second", 2, inputs=0)])
    assert [a["vm_id"] for a in report["attempts"]] == ["fog-a1", "fog-a1"]
    assert [(a["start_ms"], a["compute_end_ms"]) for a in report["attempts"]] == [(0, 10), (10, 20)]
    check_accounting(report, run)


def test_same_time_initial_choices_have_no_fabricated_integrated_cost():
    run, model, agent = fixture(count=2)
    report = episode(run, model, agent, [task("first"), task("second")])
    first = report["rl"]["transitions"][0]["feedback"]
    assert first["start_ms"] == first["end_ms"] == 0
    assert all(first[k] == 0 for k in ("energy_reward", "time_reward", "failure_reward", "service_penalty_reward"))
    check_accounting(report, run)


@pytest.mark.parametrize("seed", [2, 17])
def test_live_exploration_respects_initial_and_recovery_masks_without_optimizer_updates(seed):
    run, model, initial = fixture()
    settings = replace(initial.settings, seed=seed, epsilon_start=1, epsilon_end=1)
    agent = EnhancedDQN(initial.contract, settings, host_nodes=initial.host_nodes)
    report = episode(run, model, agent, learn=True, outages=[
        fault(5, 7, 8, 10, node) for node in ("fog:fog-a", "fog:fog-b", "cloud:cloud")])
    assert len(report["attempts"]) == 2 and agent.updates == 0
    assert report["attempts"][0]["node_id"] != report["attempts"][1]["node_id"]
    for decision in report["rl"]["decisions"]:
        if decision["action"] is not None:
            assert decision["epsilon"] == 1 and decision["observation"]["legal_mask"][decision["action"]]
    check_accounting(report, run)


def test_forced_no_action_is_never_a_valid_replay_experience():
    _, _, agent = fixture()
    obs = EncodedObservation(agent.contract.sha256, 0, "forced", "initial", (0.,) * 253, (False,) * 6)
    feedback = RewardDelta(agent.contract.sha256, 0, 100, -1, -.4, -.2, -8, True, 0)
    with pytest.raises(ValueError, match="legal VM action"):
        validate_experience(agent.contract, EnhancedExperience(obs, None, feedback, None))


def test_nonterminal_replay_cannot_inject_terminal_service_penalties():
    _, _, agent = fixture()
    obs = EncodedObservation(agent.contract.sha256, 0, "one", "initial", (0.,) * 253, (True,) * 6)
    nxt = replace(obs, time_ms=1)
    feedback = RewardDelta(agent.contract.sha256, 0, 1, -.01, 0, 0, -8, False, 1)
    with pytest.raises(ValueError, match="terminal service"):
        validate_experience(agent.contract, EnhancedExperience(obs, 0, feedback, nxt))


def test_replay_rejects_energy_in_zero_elapsed_time():
    _, _, agent = fixture()
    obs = EncodedObservation(agent.contract.sha256, 0, "one", "initial", (0.,) * 253, (True,) * 6)
    feedback = RewardDelta(agent.contract.sha256, 0, 0, -.01, 0, 0, 0, False, 1)
    with pytest.raises(ValueError, match="zero elapsed time"):
        validate_experience(agent.contract, EnhancedExperience(obs, 0, feedback, obs))


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_interrupted_frozen_episode_releases_guard_without_fabricating_terminal_feedback(monkeypatch, error):
    run, model, agent = fixture()
    before = agent.fingerprint
    select = agent.select

    def interrupted(*args, **kwargs):
        raise error("test interruption")

    monkeypatch.setattr(agent, "select", interrupted)
    with pytest.raises(error):
        episode(run, model, agent)
    assert not agent.episode_active and agent.episodes == 0
    assert agent.fingerprint == before
    monkeypatch.setattr(agent, "select", select)
    assert episode(run, model, agent)["metrics"]["completed_count"] == 1


def test_direct_simulator_interruption_releases_only_its_own_controller(monkeypatch):
    run, model, agent = fixture()
    controller = agent.controller()

    def interrupted(*args, **kwargs):
        raise RuntimeError("test interruption")

    monkeypatch.setattr(agent, "select", interrupted)
    with pytest.raises(RuntimeError, match="test interruption"):
        simulate_objectives(run, model, policy="masked_dqn", controller=controller, tasks=[task()], outages=[])
    assert controller.aborted and not agent.episode_active and not controller.transitions
    with pytest.raises(ValueError):
        controller.finish(None)


def test_refused_nested_controller_does_not_abort_an_existing_episode():
    run, model, agent = fixture()
    controller = agent.controller()
    controller.begin(agent.contract, agent.host_nodes, True)
    with pytest.raises(ValueError, match="reuse"):
        simulate_objectives(run, model, policy="masked_dqn", controller=controller, tasks=[task()], outages=[])
    assert agent.episode_active and not controller.aborted
    controller.abort()
    assert not agent.episode_active


def test_learning_abort_preserves_only_closed_nonterminal_feedback_and_can_checkpoint(monkeypatch, tmp_path):
    run, model, agent = fixture(count=2)
    controller = agent.controller(learn=True)
    select = agent.select
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("test interruption")
        return select(*args, **kwargs)

    monkeypatch.setattr(agent, "select", interrupted)
    with pytest.raises(RuntimeError, match="test interruption"):
        simulate_objectives(run, model, policy="masked_dqn", controller=controller,
                            tasks=[task("first"), task("second", 1)], outages=[])
    assert controller.aborted and controller.pending is None and not agent.episode_active
    assert agent.decisions == 1 and agent.episodes == agent.updates == 0
    assert len(agent.replay) == 1 and not agent.replay[0].feedback.terminated
    assert agent.replay[0].feedback.end_ms == 1
    path = tmp_path / "aborted-learner.json"
    agent.save(path)
    restored = EnhancedDQN.load(path, agent.contract, host_nodes=agent.host_nodes)
    assert restored.fingerprint == agent.fingerprint


@pytest.mark.parametrize("corruption", ["forced_action", "early_penalty"])
def test_rehashed_checkpoints_cannot_restore_invalid_replay(corruption, tmp_path):
    run, model, agent = fixture(count=2)
    episode(run, model, agent, [task("first"), task("second", 1)], learn=True)
    payload = agent.artifact()
    replay = payload["replay"][0]
    if corruption == "forced_action":
        replay["state"]["legal_mask"] = [False] * 6
        replay["action"] = None
    else:
        replay["feedback"]["service_penalty_reward"] = -8
    payload.pop("sha256")
    payload["sha256"] = _digest(payload)
    path = tmp_path / "malformed-replay.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="legal VM action|terminal service"):
        EnhancedDQN.load(path, agent.contract, host_nodes=agent.host_nodes)


@pytest.mark.parametrize("load", ["light", "burst", "tight_deadline"])
@pytest.mark.parametrize("period", [.3, 3, 11])
@pytest.mark.parametrize("recovery", [False, True])
def test_frozen_seeded_scenario_matrix_conserves_resources_and_all_reward_components(load, period, recovery):
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    model.data["workload"].update(task_count=12, interarrival_ms=[0, 0] if load == "burst" else [2, 5],
                                 relative_deadline_ms=[4, 9] if load == "tight_deadline" else [45, 90])
    for node, scale in [("fog:fog-a", 20), ("fog:fog-b", 30), ("cloud:cloud", 40)]:
        model.data["failures"][node]["scale_ms"] = scale
    agent = EnhancedDQN.from_configs(run, model, EnhancedDQNConfig(telemetry_period_ms=period, recovery_enabled=recovery))
    before = agent.fingerprint
    for work_seed, fail_seed in [(6200, 7100), (6201, 7101)]:
        report = run_enhanced_episode(agent, run, model, workload_seed=work_seed, failure_seed=fail_seed)
        check_accounting(report, run)
        assert agent.fingerprint == before


def test_all_four_reference_rules_match_complete_pre_fix_report_hashes():
    golden = json.loads(Path("tests/fixtures/enhanced_rules_fac9513.json").read_text())
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    assert run.manifest()["sha256"] == golden["run_sha256"] and _digest(model.manifest()) == golden["model_sha256"]
    for spec in golden["reports"]:
        report = simulate_objectives(run, model, policy=spec["policy"], workload_seed=spec["workload_seed"], failure_seed=spec["failure_seed"])
        assert _digest(report) == spec["report_sha256"]
