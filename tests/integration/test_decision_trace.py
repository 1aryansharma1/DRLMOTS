from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from drlmots.__main__ import configured_report
from drlmots.entities import Host, ResourceTier, Task, VM
from drlmots.environment import simulate
from drlmots.run_config import load_run_config, parse_run_config
from drlmots.topology import FogNode, Topology


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "resources.yaml"


def topology():
    return Topology(hosts=(Host("host", 3, 32, 1000),),
                    vms=(VM("vm", ResourceTier.FOG, 200, queue_capacity=1, resource_capacities={"cpu": 2}),),
                    fog_nodes=(FogNode("fog", ("host",), ("vm",)),))


def task(name, arrival=0):
    return Task(name, arrival, 1000, 20, {"cpu": 2})


def run(tasks, capture=True):
    return simulate(topology(), tasks, task_length_unit="bytes", resource_units={"cpu": "units"},
                    capture_decisions=capture)


def test_hand_checked_snapshots_before_each_arrival_and_after_same_time_completion():
    tasks = [task("a"), task("b"), task("rejected"), task("c", 2), task("d", 4)]
    result = run(tasks)
    vms = [d.observation.vms[0] for d in result.decisions]
    assert [d.observation.task_id for d in result.decisions] == [t.task_id for t in tasks]
    assert [d.observation.time_ms for d in result.decisions] == [0, 0, 0, 2, 4]
    assert [v.running for v in vms] == [False, True, True, True, True]
    assert [v.waiting_count for v in vms] == [0, 0, 1, 0, 0]
    assert [v.remaining_service_ms for v in vms] == [0, 2, 2, 2, 2]
    assert [v.queued_service_ms for v in vms] == [0, 0, 2, 0, 0]
    assert [v.available for v in vms] == [(2,), (0,), (0,), (0,), (0,)]
    assert [d.action_mask for d in result.decisions] == [(True,), (True,), (False,), (True,), (True,)]
    assert [d.selected_action for d in result.decisions] == [0, 0, None, 0, 0]
    assert result.decisions[2].failure_reason == "all resource-feasible VM queues are full"
    assert result.available_resources == {"vm": {"cpu": 2}}
    assert vms[1].available == (0,)  # Prior snapshot survives final ledger release.
    assert (result.completed_count, result.failed_count, result.makespan_ms) == (4, 1, 8)
    assert replace(result, decision_schema=None, decisions=()) == run(tasks, False)
    assert run(tasks) == result


def test_unknown_capacity_rejection_and_empty_trace():
    resources = topology()
    resources.vms[0].resource_capacities = None
    result = simulate(resources, [task("no-fit")], task_length_unit="bytes",
                      resource_units={"cpu": "units"}, capture_decisions=True)
    decision = result.decisions[0]
    assert decision.action_mask == (False,)
    assert decision.selected_action is None and decision.selected_vm_id is None
    assert decision.failure_reason == "no VM can satisfy task resources"
    assert decision.observation.vms[0].capacities == (None,)
    empty = run([])
    assert empty.decisions == () and empty.decision_schema["dimension"] == 14
    assert empty.available_resources == {"vm": {"cpu": 2}}


@pytest.mark.parametrize("policy", ["least_queue_fcfs", "first_fit_fcfs"])
def test_resource_fixture_exact_mapping_and_unchanged_report(policy):
    config = load_run_config(CONFIG)
    original = deepcopy(config)
    captured = configured_report(config, policy=policy, capture_decisions=True)
    trace = captured.pop("decision_trace")
    assert captured == configured_report(config, policy=policy)
    assert trace["schema"]["vm_action_ids"] == ["vm-cloud", "vm-fog-a", "vm-fog-b"]
    assert trace["schema"]["dimension"] == 47
    assert [d["action_mask"] for d in trace["decisions"]] == [
        (True, True, False), (True, True, False),
        (True, True, False) if policy == "least_queue_fcfs" else (False, True, False),
        (False, True, False), (False, False, False), (False, False, False),
    ]
    assert [d["selected_action"] for d in trace["decisions"]] == (
        [0, 1, 0, 1, None, None] if policy == "least_queue_fcfs" else [0, 0, 1, 1, None, None])
    assert config == original


def test_seeded_trace_accounts_for_every_decision_and_replays_without_input_mutation():
    config = load_run_config(CONFIG)
    config = replace(config, workload=replace(config.workload, task_count=100,
                     interarrival_ms_range=(0, 3), length_bytes_range=(500, 2000), data_size_kb_range=(20, 150)))
    original = deepcopy(config)
    report = configured_report(config, capture_decisions=True)
    trace = report["decision_trace"]
    assert len(trace["decisions"]) == len(report["outcomes"]) == 100
    outcomes = {o["task_id"]: o for o in report["outcomes"]}
    for d in trace["decisions"]:
        assert len(d["state"]) == trace["schema"]["dimension"]
        assert all(math.isfinite(value) for value in d["state"])
        assert d["selected_vm_id"] == outcomes[d["observation"]["task_id"]]["vm_id"]
        if d["selected_action"] is not None:
            assert d["action_mask"][d["selected_action"]]
        else:
            assert not any(d["action_mask"])
    manifest = dict(report["run_configuration"])
    manifest.pop("sha256")
    assert report == configured_report(parse_run_config(manifest), policy=report["policy"], capture_decisions=True)
    assert config == original


def test_actual_cli_trace_output_and_default_opt_out(tmp_path):
    output = tmp_path / "decisions.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(CONFIG),
                             "--capture-decisions", "--output", str(output)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text())
    assert len(report["decision_trace"]["decisions"]) == 6
    assert report["decision_trace"]["schema"]["dimension"] == 47
    assert "decision_trace" not in configured_report(load_run_config(CONFIG))
