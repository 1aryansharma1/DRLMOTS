"""Actual outage/restart simulator with attempt-level time/energy accounting.

No legacy DQN controller is accepted: its state/reward omits outages, pending
recoveries and delayed outcomes. This is the frozen environment before RL adaptation.
"""

from collections import deque
from dataclasses import asdict, replace
import math

import simpy

from drlmots.entities import ResourceTier, Task, TaskStatus
from drlmots.fog_cloud import _after, _sum
from drlmots.normalization import _digest
from drlmots.objectives import normalized_cost
from drlmots.recovery_config import RecoveryConfig, node_membership
from drlmots.resources import ResourceLedger, fits_resources, require_declared_units
from drlmots.workload import processing_time_ms, seeded_workload, workload_manifest


RECOVERY_POLICIES = ("least_queue_fcfs", "first_fit_fcfs", "fog_first_completion", "energy_latency")


def simulate_recovery(config, objectives, recovery, *, policy="fog_first_completion", tasks=None):
    if policy not in RECOVERY_POLICIES:
        raise ValueError("recovery mode supports reference rules only; legacy DQN checkpoints require a new fault-aware state/reward contract")
    if config.placement is None or not isinstance(recovery, RecoveryConfig):
        raise ValueError("recovery requires a network config and RecoveryConfig")
    config.manifest()  # Revalidate potentially mutable topology/resources.
    topology, horizon = config.topology, float(recovery.horizon_ms)
    objectives.validate_vm_ids(vm.vm_id for vm in topology.vms)
    nodes = node_membership(topology)
    if set(nodes) != {vm.vm_id for vm in topology.vms}:
        raise ValueError("all VMs need an explicit failure-domain membership")
    if {outage.node_id for outage in recovery.outages} - set(nodes.values()):
        raise ValueError("outage references an unknown failure domain")
    idle = dict(recovery.idle_power_w)
    if set(idle) != set(nodes):
        raise ValueError("idle power must cover exactly all VM IDs")
    if any(idle[vm] > objectives.profile(vm).active_power_w for vm in idle):
        raise ValueError("idle power cannot exceed total active power")
    if any(vm.queue or vm.queue_capacity is None for vm in topology.vms):
        raise ValueError("recovery requires empty queues and explicit queue capacities")
    generated = tasks is None
    originals = seeded_workload(config.workload) if generated else list(tasks)
    copied, seen = [], set()
    for task in originals:
        if not isinstance(task, Task) or task.status is not TaskStatus.CREATED or task.task_id in seen:
            raise ValueError("recovery workload requires unique CREATED tasks")
        if task.arrival_time >= horizon:
            raise ValueError("every task must arrive strictly before the observation horizon")
        require_declared_units(task.required_resources, config.resource_units)
        copied.append(replace(task))
        seen.add(task.task_id)
    workload = workload_manifest(copied, length_unit="bytes", resource_units=config.resource_units,
                                 generator_settings=config.workload if generated else None)
    incoming = sorted(range(len(copied)), key=lambda i: copied[i].arrival_time)
    service = {i: processing_time_ms(task.length, length_unit="bytes") for i, task in enumerate(copied)}
    reachable = config.placement.candidates(topology)
    links = {link.vm_id: link for link in config.placement.links}
    queues = {vm.vm_id: deque() for vm in topology.vms}
    running, pending, outcomes = {}, {}, {}
    attempts, by_task, recovery_intervals, decisions, events = [], {i: [] for i in incoming}, [], [], []
    affected = set()
    env = simpy.Environment()
    ledger = ResourceLedger(topology.vms)
    up = {node: True for node in nodes.values()}
    # At a common timestamp, complete service first, then apply all outages,
    # start pre-existing ready work, dispatch recoveries, then external arrivals.
    edges = sorted([(o.start_ms, 1, o.node_id) for o in recovery.outages]
                   + [(o.end_ms, 0, o.node_id) for o in recovery.outages])

    def terminal(index, status, reason):
        if index in outcomes:
            raise RuntimeError("task finalized twice")
        outcomes[index] = {"task_id": copied[index].task_id, "status": status, "reason": reason,
                           "arrival_time_ms": copied[index].arrival_time, "terminal_time_ms": float(env.now),
                           "finish_time_ms": float(env.now) if status == "completed" else None,
                           "attempt_ids": [a["attempt_id"] for a in by_task[index]],
                           "recovered": status == "completed" and index in affected}

    def close_pending(index):
        item = pending.pop(index)
        elapsed = env.now - item["failed_at_ms"]
        recovery_intervals.append({"task_id": copied[index].task_id, "start_ms": item["failed_at_ms"],
                                   "end_ms": float(env.now), "elapsed_ms": elapsed,
                                   "coordinator_active_ms": min(elapsed, recovery.recovery_delay_ms),
                                   "energy_j": recovery.recovery_power_w * min(elapsed, recovery.recovery_delay_ms) / 1000})

    def finish_attempt(attempt, status, reason):
        if attempt["end_ms"] is not None:
            raise RuntimeError("attempt ended twice")
        index, vm = attempt["task_index"], attempt["vm_id"]
        if attempt["start_ms"] is not None:
            ledger.release(vm, replace(copied[index], task_id=attempt["attempt_id"]), env.now)
        elapsed = env.now - attempt["dispatch_ms"]
        run_ms = 0.0 if attempt["start_ms"] is None else env.now - attempt["start_ms"]
        upload_elapsed = min(elapsed, attempt["upload_ms"])
        before_execution = attempt["start_ms"] if attempt["start_ms"] is not None else env.now
        wait = max(0.0, before_execution - attempt["upload_ready_ms"])
        # Serialization occurs first, followed by propagation. A failed upload
        # consumes energy only for bytes already serialized, even during propagation.
        serialization = 8 * copied[index].data_size_kb / links[vm].bandwidth_mbps
        transmitted = copied[index].data_size_kb * min(1.0, elapsed / serialization)
        profile = objectives.profile(vm)
        compute_j = profile.active_power_w * run_ms / 1000
        transfer_j = transmitted * profile.transmission_j_per_kb
        attempt.update(end_ms=float(env.now), status=status, reason=reason, upload_elapsed_ms=upload_elapsed,
                       post_upload_wait_ms=wait, executed_ms=run_ms, transmitted_kb=transmitted,
                       compute_energy_j=compute_j, transmission_energy_j=transfer_j,
                       total_energy_j=_sum(compute_j, transfer_j))
        if not math.isclose(elapsed, _sum(upload_elapsed, wait, run_ms), rel_tol=1e-10, abs_tol=1e-9):
            raise RuntimeError("attempt time accounting mismatch")

    def interrupt(attempt):
        index = attempt["task_index"]
        finish_attempt(attempt, "interrupted", "node_failure")
        affected.add(index)
        if len(by_task[index]) - 1 >= recovery.max_retries:
            terminal(index, "permanent_failure", "retry_budget_exhausted")
            return
        not_before = _after(env.now, recovery.recovery_delay_ms)
        pending[index] = {"failed_at_ms": float(env.now), "not_before_ms": not_before,
                          "deadline_ms": _after(not_before, recovery.recovery_wait_ms),
                          "excluded_node": attempt["node_id"]}

    def candidate_estimates(index, excluded=None):
        task = copied[index]
        result = []
        for vm in reachable:
            vm_id = vm.vm_id
            if not up[nodes[vm_id]] or nodes[vm_id] == excluded or not fits_resources(task, vm) or len(queues[vm_id]) >= vm.queue_capacity:
                continue
            tail = running[vm_id]["planned_finish_ms"] if vm_id in running else env.now
            for item in queues[vm_id]:
                tail = _after(max(tail, item["upload_ready_ms"]), service[item["task_index"]])
            upload = links[vm_id].upload_ms(task.data_size_kb)
            ready = _after(env.now, upload)
            finish = _after(max(tail, ready), service[index])
            energy = (objectives.profile(vm_id).active_power_w - idle[vm_id]) * service[index] / 1000 + task.data_size_kb * objectives.profile(vm_id).transmission_j_per_kb
            result.append({"vm_id": vm_id, "node_id": nodes[vm_id], "tier": vm.tier.value,
                           "load": len(queues[vm_id]) + int(vm_id in running), "upload_ms": upload,
                           "predicted_finish_ms": finish, "predicted_incremental_energy_j": energy})
        return result

    def dispatch(index, *, excluded=None, is_retry=False):
        candidates = candidate_estimates(index, excluded)
        if not candidates:
            return False
        if policy == "first_fit_fcfs":
            chosen = min(candidates, key=lambda a: a["vm_id"])
        elif policy == "least_queue_fcfs":
            chosen = min(candidates, key=lambda a: (a["load"], a["vm_id"]))
        elif policy == "fog_first_completion":
            chosen = min(candidates, key=lambda a: (a["predicted_finish_ms"], a["tier"] == "cloud", a["vm_id"]))
        else:
            # Current information only. Future outage times and old analytical
            # hazard coefficients are deliberately absent from the decision.
            chosen = min(candidates, key=lambda a: (
                .5 * normalized_cost(a["predicted_finish_ms"] - env.now, 10)
                + .5 * normalized_cost(a["predicted_incremental_energy_j"], objectives.energy_scale_j), a["vm_id"]))
        if is_retry:
            close_pending(index)
        number = len(by_task[index]) + 1
        attempt = {"attempt_id": f"attempt-{index + 1}-{number}", "task_id": copied[index].task_id,
                   "task_index": index, "number": number, "vm_id": chosen["vm_id"], "node_id": chosen["node_id"],
                   "dispatch_ms": float(env.now), "upload_ms": chosen["upload_ms"],
                   "upload_ready_ms": _after(env.now, chosen["upload_ms"]),
                   "start_ms": None, "end_ms": None, "status": "queued"}
        attempts.append(attempt)
        by_task[index].append(attempt)
        queues[chosen["vm_id"]].append(attempt)
        decisions.append({"task_id": copied[index].task_id, "attempt_id": attempt["attempt_id"],
                          "time_ms": float(env.now), "is_retry": is_retry, "excluded_node": excluded,
                          "selected_vm_id": chosen["vm_id"], "candidates": candidates,
                          "observed_up_nodes": sorted(node for node, available in up.items() if available)})
        return True

    def start_ready():
        for vm in sorted(queues):
            if up[nodes[vm]] and vm not in running and queues[vm] and queues[vm][0]["upload_ready_ms"] <= env.now:
                attempt = queues[vm].popleft()
                attempt["start_ms"] = float(env.now)
                attempt["status"] = "running"
                attempt["planned_finish_ms"] = _after(env.now, service[attempt["task_index"]])
                ledger.reserve(vm, replace(copied[attempt["task_index"]], task_id=attempt["attempt_id"]), env.now)
                running[vm] = attempt

    def drive():
        index, edge = 0, 0
        while True:
            for vm in sorted(tuple(running)):
                attempt = running[vm]
                if attempt["planned_finish_ms"] <= env.now:
                    finish_attempt(attempt, "completed", None)
                    terminal(attempt["task_index"], "completed", None)
                    del running[vm]
            failed = []
            while edge < len(edges) and edges[edge][0] <= env.now:
                _, is_down, node = edges[edge]
                edge += 1
                up[node] = not bool(is_down)
                events.append({"time_ms": float(env.now), "node_id": node, "state": "down" if is_down else "up"})
                if is_down:
                    for vm in sorted(queues):
                        if nodes[vm] != node:
                            continue
                        if vm in running:
                            failed.append(running.pop(vm))
                        failed.extend(queues[vm])
                        queues[vm].clear()
            # Apply all simultaneous node changes before any backup selection.
            for attempt in sorted(failed, key=lambda item: item["task_index"]):
                interrupt(attempt)
            if env.now >= horizon:
                for attempt in list(running.values()) + [a for q in queues.values() for a in q]:
                    finish_attempt(attempt, "deadline", "observation_horizon")
                    terminal(attempt["task_index"], "permanent_failure", "observation_horizon")
                running.clear()
                for q in queues.values():
                    q.clear()
                for task_index in sorted(tuple(pending)):
                    close_pending(task_index)
                    terminal(task_index, "permanent_failure", "observation_horizon")
                break
            start_ready()
            for task_index in sorted(tuple(pending)):
                item = pending[task_index]
                if item["not_before_ms"] <= env.now:
                    if dispatch(task_index, excluded=item["excluded_node"], is_retry=True):
                        continue
                    if item["deadline_ms"] <= env.now:
                        close_pending(task_index)
                        terminal(task_index, "permanent_failure", "backup_wait_expired")
            while index < len(incoming) and copied[incoming[index]].arrival_time <= env.now:
                task_index = incoming[index]
                index += 1
                if not dispatch(task_index):
                    terminal(task_index, "rejected", "no_up_reachable_feasible_queue")
            # Network uploads are positive; new admissions cannot start immediately.
            future = [horizon]
            if index < len(incoming):
                future.append(copied[incoming[index]].arrival_time)
            if edge < len(edges):
                future.append(edges[edge][0])
            future.extend(a["planned_finish_ms"] for a in running.values())
            future.extend(q[0]["upload_ready_ms"] for vm, q in queues.items() if q and vm not in running)
            for item in pending.values():
                future.extend(t for t in (item["not_before_ms"], item["deadline_ms"]) if t > env.now)
            next_time = min(t for t in future if t > env.now)
            yield env.timeout(next_time - env.now)

    env.process(drive())
    env.run()
    if len(outcomes) != len(copied) or any(a["end_ms"] is None for a in attempts):
        raise RuntimeError("every task/attempt must have one terminal outcome")
    balances = ledger.final_balances()
    task_rows = []
    for index, task in enumerate(copied):
        row = outcomes[index]
        task_attempts = by_task[index]
        intervals = [item for item in recovery_intervals if item["task_id"] == task.task_id]
        row.update(latency_ms=row["terminal_time_ms"] - task.arrival_time,
                   attempt_elapsed_ms=math.fsum(a["end_ms"] - a["dispatch_ms"] for a in task_attempts),
                   recovery_blocked_ms=math.fsum(item["elapsed_ms"] for item in intervals),
                   compute_energy_j=math.fsum(a["compute_energy_j"] for a in task_attempts),
                   transmission_energy_j=math.fsum(a["transmission_energy_j"] for a in task_attempts),
                   recovery_energy_j=math.fsum(item["energy_j"] for item in intervals))
        row["attributed_energy_j"] = _sum(row["compute_energy_j"], row["transmission_energy_j"], row["recovery_energy_j"])
        if not math.isclose(row["latency_ms"], _sum(row["attempt_elapsed_ms"], row["recovery_blocked_ms"]), rel_tol=1e-10, abs_tol=1e-9):
            raise RuntimeError("task latency accounting mismatch")
        task_rows.append(row)
    usage = []
    for vm in sorted(nodes):
        downtime = math.fsum(o.end_ms - o.start_ms for o in recovery.outages if o.node_id == nodes[vm])
        busy = math.fsum(a["executed_ms"] for a in attempts if a["vm_id"] == vm)
        idle_ms = horizon - downtime - busy
        if idle_ms < -1e-8:
            raise RuntimeError("VM busy time exceeds uptime")
        usage.append({"vm_id": vm, "node_id": nodes[vm], "up_ms": horizon - downtime,
                      "busy_ms": busy, "idle_ms": max(0.0, idle_ms), "idle_energy_j": idle[vm] * max(0.0, idle_ms) / 1000})
    completed = [row for row in task_rows if row["status"] == "completed"]
    rejected = sum(row["status"] == "rejected" for row in task_rows)
    compute_j = math.fsum(a["compute_energy_j"] for a in attempts)
    upload_j = math.fsum(a["transmission_energy_j"] for a in attempts)
    idle_j = math.fsum(u["idle_energy_j"] for u in usage)
    recovery_j = math.fsum(item["energy_j"] for item in recovery_intervals)
    total_j = _sum(compute_j, upload_j, idle_j, recovery_j)
    recovered = sum(row["recovered"] for row in completed)
    metrics = {"submitted_count": len(copied), "completed_count": len(completed), "rejected_count": rejected,
               "permanent_failure_count": len(copied) - len(completed) - rejected,
               "node_interrupted_task_count": len(affected), "interrupted_attempt_count": sum(a["status"] == "interrupted" for a in attempts),
               "recovered_task_count": recovered, "retry_attempt_count": sum(a["number"] > 1 for a in attempts),
               "completion_rate": len(completed) / len(copied) if copied else None,
               "observed_failure_rate_all_arrivals": (len(copied) - len(completed)) / len(copied) if copied else None,
               "recovery_success_rate": recovered / len(affected) if affected else None,
               "makespan_ms": max((row["finish_time_ms"] for row in completed), default=0.0),
               "settlement_time_ms": max((row["terminal_time_ms"] for row in task_rows), default=0.0),
               "successful_mean_latency_ms": math.fsum(row["latency_ms"] for row in completed) / len(completed) if completed else None,
               "mean_recovered_latency_ms": math.fsum(row["latency_ms"] for row in completed if row["recovered"]) / recovered if recovered else None,
               "compute_energy_j": compute_j, "transmission_energy_j": upload_j, "idle_energy_j": idle_j,
               "recovery_energy_j": recovery_j, "total_energy_j": total_j,
               "failed_attempt_energy_j": math.fsum(a["total_energy_j"] for a in attempts if a["status"] != "completed"),
               "retry_attempt_energy_j": math.fsum(a["total_energy_j"] for a in attempts if a["number"] > 1),
               "energy_per_completed_task_j": total_j / len(completed) if completed else None}
    for name, value in metrics.items():
        if value is not None:
            Task._validate_number(name, value, allow_zero=True)
    return {"schema_version": 1, "kind": "observed_node_recovery_simulation", "policy": policy,
            "run_configuration": config.manifest(), "workload": workload, "recovery_config": recovery.manifest(),
            "outage_trace": recovery.trace_manifest(), "objective_coefficients": objectives.manifest(),
            "accounting": {"time_unit": "ms", "energy_unit": "J", "power_unit": "W",
                "energy_window": "fixed [0,horizon_ms] for all policies", "active_power": "total running VM power; idle charged only while up and not running",
                "transfer": "serialization then propagation; interrupted transmission charged proportionally to bytes sent",
                "restart": "full input retained at durable source; full service restarted on a different failure domain",
                "coordinator": "per-task power during fixed recovery delay; waiting after delay has no coordinator active charge",
                "tie_order": "service completions, node changes, existing queued starts, recoveries, external arrivals; horizon finalizes unfinished work",
                "limits": "no output transfers, link contention, checkpoints, source/controller failures, shared host power or hardware calibration",
                "analytical_risk": "legacy failure_rate_per_s coefficients are retained only for provenance; actual node outages determine outcomes",
                "rl": "legacy DQN checkpoints are not compatible with this environment; fault-aware learning contract remains a separate step"},
            "observed_objectives": {"directions": "minimize makespan, fixed-window energy and all-arrival unsuccessful fraction",
                "makespan_ms": metrics["makespan_ms"], "energy_j": total_j,
                "unsuccessful_fraction": metrics["observed_failure_rate_all_arrivals"],
                "qualification": "read admission/rejection counts alongside makespan; this is an objective vector, not the old proxy reward"},
            "metrics": metrics, "outcomes": task_rows, "attempts": [{k: v for k, v in a.items() if k != "task_index"} for a in attempts],
            "recovery_intervals": recovery_intervals, "node_events": events, "decisions": decisions,
            "vm_usage": usage, "resource_events": [asdict(event) for event in ledger.events], "available_resources": balances}
