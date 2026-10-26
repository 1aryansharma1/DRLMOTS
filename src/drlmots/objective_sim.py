"""Causal A-030 event simulation. All model time is milliseconds.

Shared half-duplex bottlenecks use equal processor sharing among active transfers.
A VM reserves its fixed MIPS and requested resources only during computation.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import math

from drlmots.entities import Task
from drlmots.normalization import _digest
from drlmots.objective_model import (ObjectiveModel, WorkItem, conditional_failure_probability,
                                    failure_trace, fields, generate_work, number, power_w)
from drlmots.resources import ResourceLedger, fits_resources, require_declared_units


MODEL_POLICIES = ("least_queue_fcfs", "first_fit_fcfs", "fog_first_completion", "multi_objective")
PHASES = ("upload", "upload_propagation", "queue", "compute", "download", "download_propagation")


def simulate_objectives(config, model, *, policy="multi_objective", workload_seed=None,
                        failure_seed=0, tasks=None, outages=None, recovery_enabled=True, controller=None):
    if policy not in MODEL_POLICIES and policy != "masked_dqn":
        raise ValueError("objective model supports reference rules or enhanced masked_dqn; old DQN checkpoints are incompatible")
    if (controller is not None) != (policy == "masked_dqn"):
        raise ValueError("masked_dqn requires an enhanced controller; rules keep their original path")
    if not isinstance(model, ObjectiveModel) or type(recovery_enabled) is not bool:
        raise ValueError("expected ObjectiveModel and boolean recovery flag")
    # Revalidate mutable caller settings and make the simulation independent of later mutation.
    model = ObjectiveModel(model.manifest())
    d = model.data
    nodes, host_nodes = model.validate_topology(config)
    vms = {v.vm_id: v for v in config.topology.vms}
    reachable = {v.vm_id for v in config.placement.candidates(config.topology)}
    originals = generate_work(model, workload_seed) if tasks is None else list(tasks)
    work = []
    for item in originals:
        if not isinstance(item, WorkItem):
            raise ValueError("new model needs explicit MI WorkItems, not byte-length Tasks")
        item = WorkItem(**asdict(item))
        require_declared_units(item.required_resources, config.resource_units)
        if item.arrival_ms >= d["horizon_ms"]:
            raise ValueError("all workload arrivals must precede the observation cutoff")
        work.append(item)
    if len({w.task_id for w in work}) != len(work):
        raise ValueError("duplicate task ID")
    trace = failure_trace(model, failure_seed) if outages is None else deepcopy(list(outages))
    last = {}
    for o in sorted(trace, key=lambda o: (o["node_id"], o["start_ms"])):
        fields(o, "node_id start_ms repair_end_ms end_ms detection_ms", "outage")
        if o["node_id"] not in d["failures"]:
            raise ValueError("unknown outage node")
        for key in ("start_ms", "repair_end_ms", "end_ms", "detection_ms"):
            number(key, o[key])
        if not (o["start_ms"] <= o["repair_end_ms"] <= o["end_ms"] and o["end_ms"] > o["start_ms"] and o["detection_ms"] >= o["start_ms"]):
            raise ValueError("invalid outage clock ordering")
        if o["start_ms"] <= last.get(o["node_id"], -1):
            raise ValueError("overlapping or touching node outages")
        last[o["node_id"]] = o["end_ms"]
    trace.sort(key=lambda o: (o["start_ms"], o["node_id"]))
    horizon, now = d["horizon_ms"], 0.0
    physical = {n: "up" for n in d["failures"]}
    known = {n: True for n in physical}
    last_up = {n: 0.0 for n in physical}
    current_outage = {}
    edges = sorted((o[key], rank, oi) for oi, o in enumerate(trace)
                   for key, rank in (("start_ms", 0), ("repair_end_ms", 1), ("end_ms", 2), ("detection_ms", 3)))
    incoming = sorted(range(len(work)), key=lambda i: (work[i].arrival_ms, i))
    queues = {v: deque() for v in vms}
    running, active, pending, outcomes = {}, {}, {}, {}
    attempts, decisions, node_events, blocks, power_segments, link_segments = [], [], [], [], [], []
    by_task = {i: [] for i in incoming}
    ledger = ResourceLedger(config.topology.vms)
    host_energy = {h: dict(idle_j=0.0, dynamic_j=0.0, down_j=0.0, reboot_j=0.0, up_ms=0.0, busy_ms=0.0, utilization_ms=0.0) for h in d["hosts"]}
    link_energy = {l: dict(traffic_j=0.0, idle_j=0.0, transmitted_kb=0.0) for l in d["links"]}
    affected = set()

    telemetry = None
    poll_index = 0
    if controller is not None:
        from drlmots.enhanced_rl_contract import (EnhancedRLContract, HostTelemetry, LinkTelemetry,
            QueuedTelemetry, RewardSnapshot, VMTelemetry)
        from drlmots.enhanced_telemetry import ObservedTelemetry
        if not isinstance(getattr(controller,"contract",None),EnhancedRLContract):
            raise ValueError("enhanced contract-4 controller required")
        public = controller.contract.context
        expected = EnhancedRLContract.from_configs(config,model,planned_task_count=len(work),
            completion_penalty=public["completion_penalty"],deadline_penalty=public["deadline_penalty"])
        controller.begin(expected,host_nodes,recovery_enabled)
        telemetry = ObservedTelemetry(expected,host_nodes)
        period = controller.settings.telemetry_period_ms

    def reward_cursor():
        # These evaluator measurements are not observation features. Open recovery
        # intervals accrue continuously, rather than being billed early or at close.
        host_j=sum(e[k] for e in host_energy.values() for k in ("idle_j","dynamic_j","down_j","reboot_j"))
        net_j=sum(e["traffic_j"]+e["idle_j"] for e in link_energy.values())
        recovery_j=sum(b["energy_j"] for b in blocks)
        for p in pending.values():
            detection=min(now,p["detect_at_ms"])-p["failed_at_ms"]
            restart=max(0,min(now,p["ready_ms"])-p["detect_at_ms"])
            recovery_j+=(detection+restart)*d["recovery"]["coordinator_power_w"]/1000
        counts={status:sum(o["status"]==status for o in outcomes.values()) for status in
                ("completed","rejected","execution_failure","deadline_missed","unfinished")}
        matured=[i for i in incoming[:arrival_index] if work[i].deadline_ms is not None and work[i].deadline_ms<=now]
        return RewardSnapshot(now,host_j+net_j+recovery_j,arrival_index,counts["completed"],counts["rejected"],
            counts["execution_failure"],counts["deadline_missed"],counts["unfinished"],len(matured),
            sum(i in outcomes and outcomes[i]["status"]=="completed" for i in matured),
            max((o["finish_ms"] for o in outcomes.values() if o["status"]=="completed"),default=0))

    def deliver_sensor_reports():
        # A request is periodic and public. A failed node sends no report: retain
        # its previous cache until a real response or an explicit failure notice.
        # Raw physical dictionaries are never passed to the controller.
        for node in sorted(physical):
            if physical[node] != "up":
                continue
            vm_views=[]
            for vm in sorted(v for v in vms if nodes[v]==node):
                waiting=[]
                for a in queues[vm]:
                    phase={"upload":"upload","upload_propagation":"propagation","queue":"ready"}[a["phase"]]
                    waiting.append(QueuedTelemetry(phase,a["remaining_kb"] if phase=="upload" else 0,
                        max(0,a["phase_end_ms"]-now) if phase=="propagation" else 0,
                        work[a["task_index"]].compute_mi,tuple(sorted(work[a["task_index"]].required_resources.items()))))
                vm_views.append(VMTelemetry(vm,now,max(0,running[vm]["phase_end_ms"]-now) if vm in running else None,
                                             tuple(waiting),tuple(sorted(ledger.available(vm).items()))))
            host_views=[]
            for host in sorted(h for h in d["hosts"] if host_nodes[h]==node):
                used=sum(d["vms"][vm]["mips"] for vm in running if d["vms"][vm]["host_id"]==host)
                host_views.append(HostTelemetry(host,now,used/d["hosts"][host]["capacity_mips"]))
            traffic=[]
            for link in sorted(d["links"]):
                flows=[a for a in active.values() if a["node_id"]==node and d["vms"][a["vm_id"]]["link_id"]==link]
                up=[a for a in flows if a["phase"]=="upload"]
                down=[a for a in flows if a["phase"]=="download"]
                traffic.append(LinkTelemetry(link,now,len(up),len(down),sum(a["remaining_kb"] for a in up),sum(a["remaining_kb"] for a in down)))
            telemetry.receive(node,now,vms=tuple(vm_views),hosts=tuple(host_views),traffic=tuple(traffic))

    def service(i, vm):
        return 1000 * work[i].compute_mi / d["vms"][vm]["mips"]

    def resource_task(a):
        w = work[a["task_index"]]
        return Task(a["attempt_id"], w.arrival_ms, w.compute_mi, max(w.input_kb, 1e-30), w.required_resources)

    def flow_groups():
        result = {l: [] for l in d["links"]}
        for a in active.values():
            if a["phase"] in ("upload", "download"):
                result[d["vms"][a["vm_id"]]["link_id"]].append(a)
        return result

    def advance(end):
        nonlocal now
        dt = end - now
        if dt <= 0:
            raise RuntimeError("simulation clock did not advance")
        for a in active.values():
            a["phase_ms"][a["phase"]] += dt
            if a["phase"] == "compute":
                a["executed_mi"] += dt * d["vms"][a["vm_id"]]["mips"] / 1000
        for link_id, flows in flow_groups().items():
            link, total_sent = d["links"][link_id], 0.0
            if flows:
                rate = link["bandwidth_mbps"] / (8 * len(flows))
                for a in flows:
                    sent = min(a["remaining_kb"], dt * rate)
                    a["remaining_kb"] = max(0.0, a["remaining_kb"] - sent)
                    a[a["phase"] + "_sent_kb"] += sent
                    a["network_energy_j"] += sent * link["joules_per_kb"]
                    total_sent += sent
            e = link_energy[link_id]
            e["traffic_j"] += total_sent * link["joules_per_kb"]
            e["transmitted_kb"] += total_sent
            e["idle_j"] += link["idle_power_w"] * dt / 1000
            if flows:
                link_segments.append(dict(link_id=link_id, start_ms=now, end_ms=end, active_transfers=len(flows),
                                          transmitted_kb=total_sent, capacity_kb=link["bandwidth_mbps"] * dt / 8))
        for host_id, host in d["hosts"].items():
            state = physical[host_nodes[host_id]]
            members = [a for vm, a in running.items() if d["vms"][vm]["host_id"] == host_id]
            mips = sum(d["vms"][a["vm_id"]]["mips"] for a in members)
            utilization = mips / host["capacity_mips"] if state == "up" else 0.0
            power = power_w(host["power_curve"], utilization) if state == "up" else host[state + "_power_w"]
            e = host_energy[host_id]
            if state == "up":
                idle = host["power_curve"][0][1]
                dynamic = (power-idle) * dt / 1000
                e["idle_j"] += idle * dt / 1000
                e["dynamic_j"] += dynamic
                e["up_ms"] += dt
                e["busy_ms"] += dt if members else 0
                e["utilization_ms"] += utilization * dt
                for a in members:
                    a["host_dynamic_energy_j"] += dynamic * d["vms"][a["vm_id"]]["mips"] / mips
            else:
                e[state + "_j"] += power * dt / 1000
            power_segments.append(dict(host_id=host_id, node_id=host_nodes[host_id], start_ms=now, end_ms=end,
                                       state=state, utilization=utilization, power_w=power, energy_j=power*dt/1000))
        now = end

    def close_attempt(a, status, reason):
        i, vm = a["task_index"], a["vm_id"]
        if a["end_ms"] is not None:
            raise RuntimeError("attempt finalized twice")
        if running.get(vm) is a:
            ledger.release(vm, resource_task(a), now)
            del running[vm]
        if a in queues[vm]:
            queues[vm].remove(a)
        active.pop(i, None)
        a.update(end_ms=now, status=status, reason=reason,
                 attributable_energy_j=a["host_dynamic_energy_j"] + a["network_energy_j"])
        if not math.isclose(sum(a["phase_ms"].values()), now-a["dispatch_ms"], abs_tol=1e-8):
            raise RuntimeError("attempt phase accounting mismatch")

    def close_pending(i):
        p = pending.pop(i)
        detection = min(now, p["detect_at_ms"]) - p["failed_at_ms"]
        restart = max(0, min(now, p["ready_ms"]) - p["detect_at_ms"])
        blocks.append(dict(task_id=work[i].task_id, start_ms=p["failed_at_ms"], end_ms=now,
                           detection_ms=detection, restart_ms=restart,
                           backup_wait_ms=max(0, now-p["ready_ms"]),
                           energy_j=(detection+restart)*d["recovery"]["coordinator_power_w"]/1000))

    def terminal(i, status, reason):
        if i in outcomes:
            raise RuntimeError("task finalized twice")
        if i in active:
            close_attempt(active[i], status, reason)
        if i in pending:
            close_pending(i)
        outcomes[i] = dict(task_id=work[i].task_id, status=status, reason=reason, arrival_ms=work[i].arrival_ms,
                           terminal_ms=now, finish_ms=now if status == "completed" else None,
                           deadline_ms=work[i].deadline_ms, recovered=i in affected and status == "completed",
                           attempt_ids=[a["attempt_id"] for a in by_task[i]])

    def interrupt(a, detect_at):
        i = a["task_index"]
        affected.add(i)
        close_attempt(a, "interrupted", "node_failure")
        ready = max(now, detect_at) + d["recovery"]["restart_ms"]
        pending[i] = dict(failed_at_ms=now, detect_at_ms=max(now, detect_at), ready_ms=ready,
                          expires_ms=ready+d["recovery"]["wait_ms"], excluded_node=a["node_id"])

    def estimates(i, excluded=None):
        w, result = work[i], []
        groups = flow_groups()
        for vm in sorted(reachable):
            spec = d["vms"][vm]
            node = nodes[vm]
            if not known[node] or node == excluded or len(queues[vm]) >= vms[vm].queue_capacity:
                continue
            proxy = Task(w.task_id, w.arrival_ms, w.compute_mi, max(w.input_kb, 1e-30), w.required_resources)
            if not fits_resources(proxy, vms[vm]):
                continue
            link = d["links"][spec["link_id"]]
            backlog = sum(a["remaining_kb"] for a in groups[spec["link_id"]])
            upload = 8 * (backlog + w.input_kb) / link["bandwidth_mbps"] + link["propagation_ms"]
            tail = running[vm]["phase_end_ms"] if vm in running else now
            for a in queues[vm]:
                # Causal approximation: current transfer/propagation work only, no future arrivals/failures.
                ready = now if a["phase"] == "queue" else max(now, a.get("phase_end_ms", now))
                if a["phase"] == "upload":
                    ready = now + 8 * (backlog + a["remaining_kb"]) / link["bandwidth_mbps"] + link["propagation_ms"]
                tail = max(tail, ready) + service(a["task_index"], vm)
            finish = max(tail, now+upload) + service(i, vm) + 8*w.output_kb/link["bandwidth_mbps"] + link["propagation_ms"]
            host = d["hosts"][spec["host_id"]]
            current_mips = sum(d["vms"][v]["mips"] for v in running if d["vms"][v]["host_id"] == spec["host_id"] and v != vm)
            u0, u1 = current_mips/host["capacity_mips"], (current_mips+spec["mips"])/host["capacity_mips"]
            energy = (power_w(host["power_curve"], u1)-power_w(host["power_curve"], u0))*service(i, vm)/1000 + (w.input_kb+w.output_kb)*link["joules_per_kb"]
            risk = conditional_failure_probability(d["failures"][node], max(0, now-last_up[node]), finish-now)
            o = d["objective"]
            score = o["weights"]["time"]*(finish-now)/o["scales"]["decision_time_ms"] + o["weights"]["energy"]*energy/o["scales"]["decision_energy_j"] + o["weights"]["failure"]*risk
            result.append(dict(vm_id=vm, node_id=node, tier=vms[vm].tier.value,
                               load=len(queues[vm])+int(vm in running), predicted_finish_ms=finish,
                               predicted_incremental_energy_j=energy, predicted_failure_probability=risk,
                               predicted_deadline_met=w.deadline_ms is None or finish <= w.deadline_ms, score=score))
        return result

    def dispatch(i, retry=False):
        excluded = pending[i]["excluded_node"] if retry else None
        if controller is not None:
            current=reward_cursor()
            observation=controller.contract.encode(telemetry.snapshot(now,work[i],arrived=arrival_index,
                completed=current.completed_count,unsuccessful=current.unsuccessful_count,
                attempts_used=len(by_task[i]),previous_node=excluded,
                wait_remaining=max(0,pending[i]["expires_ms"]-now) if retry else 0))
            action=controller(observation,current)
            selected=controller.contract.validate_action(observation,action)
            if selected is None:
                return False
            if len(queues[selected])>=vms[selected].queue_capacity:
                raise RuntimeError("observed admission undercounted the physical queue")
            candidates=[]  # Do not expose private placement predictions to the agent.
            chosen={"vm_id":selected}
        else:
            candidates = estimates(i, excluded)
            if not candidates:
                return False
            # Hard task deadlines are also used by multi-objective placement when a predicted feasible VM exists.
            selectable = candidates
            if policy == "multi_objective" and any(c["predicted_deadline_met"] for c in candidates):
                selectable = [c for c in candidates if c["predicted_deadline_met"]]
            if policy == "first_fit_fcfs":
                chosen = min(selectable, key=lambda c: c["vm_id"])
            elif policy == "least_queue_fcfs":
                chosen = min(selectable, key=lambda c: (c["load"], c["vm_id"]))
            elif policy == "fog_first_completion":
                chosen = min(selectable, key=lambda c: (c["predicted_finish_ms"], c["tier"] == "cloud", c["vm_id"]))
            else:
                chosen = min(selectable, key=lambda c: (c["score"], c["tier"] == "cloud", c["vm_id"]))
        if retry:
            close_pending(i)
        vm, n = chosen["vm_id"], len(by_task[i])+1
        a = dict(attempt_id=f"attempt-{i+1}-{n}", task_id=work[i].task_id, task_index=i, number=n,
                 vm_id=vm, node_id=nodes[vm], host_id=d["vms"][vm]["host_id"],
                 dispatch_ms=now, start_ms=None, compute_end_ms=None, end_ms=None,
                 phase="upload", remaining_kb=work[i].input_kb, status="active",
                 phase_ms={p: 0.0 for p in PHASES}, executed_mi=0.0, upload_sent_kb=0.0,
                 download_sent_kb=0.0, network_energy_j=0.0, host_dynamic_energy_j=0.0)
        attempts.append(a)
        by_task[i].append(a)
        active[i] = a
        queues[vm].append(a)
        if telemetry is not None:
            telemetry.admitted(vm,work[i])
        decisions.append(dict(time_ms=now, task_id=work[i].task_id, attempt_id=a["attempt_id"], retry=retry,
                              selected_vm_id=vm, observed_up_nodes=sorted(n for n in known if known[n]) if telemetry is None else
                              sorted(n for n,v in telemetry.nodes.items() if v.available is True), candidates=candidates))
        if physical[nodes[vm]] != "up":
            # A stale health view can route into a failed node. Physics rejects it;
            # the scheduler learns at detection, never by reading future outage edges.
            interrupt(a, current_outage[nodes[vm]]["detection_ms"])
        return True

    def settle_phases():
        changed = False
        for i, a in sorted(tuple(active.items())):
            phase = a["phase"]
            if phase in ("upload", "download") and a["remaining_kb"] <= 1e-9:
                a["phase"] = phase + "_propagation"
                a["phase_end_ms"] = now + d["links"][d["vms"][a["vm_id"]]["link_id"]]["propagation_ms"]
                changed = True
            elif phase in ("upload_propagation", "download_propagation", "compute") and a["phase_end_ms"] <= now:
                if phase == "upload_propagation":
                    a["phase"] = "queue"
                elif phase == "compute":
                    vm = a["vm_id"]
                    ledger.release(vm, resource_task(a), now)
                    del running[vm]
                    a.update(compute_end_ms=now, phase="download", remaining_kb=work[i].output_kb)
                else:
                    terminal(i, "completed", None)
                changed = True
        return changed

    def start_ready():
        changed = False
        for vm in sorted(queues):
            if physical[nodes[vm]] == "up" and vm not in running and queues[vm] and queues[vm][0]["phase"] == "queue":
                a = queues[vm].popleft()
                a.update(phase="compute", start_ms=now, phase_end_ms=now+service(a["task_index"], vm))
                ledger.reserve(vm, resource_task(a), now)
                running[vm] = a
                changed = True
        return changed

    arrival_index = edge_index = 0
    while True:
        # Output delivery at the exact failure/deadline/cutoff wins. Zero-length
        # serialization/propagation resolves at the same timestamp without a time step.
        while settle_phases():
            pass
        while edge_index < len(edges) and edges[edge_index][0] <= now:
            _, rank, oi = edges[edge_index]
            edge_index += 1
            o, node = trace[oi], trace[oi]["node_id"]
            if rank == 0:
                physical[node] = "down"
                current_outage[node] = o
                for a in sorted(tuple(active.values()), key=lambda a: a["task_index"]):
                    # A fully serialized output has already left the failed host.
                    # Its in-flight propagation survives; partial output does not.
                    if a["node_id"] == node and a["phase"] != "download_propagation":
                        interrupt(a, o["detection_ms"])
            elif rank == 1:
                physical[node] = "reboot"
            elif rank == 2:
                physical[node], known[node], last_up[node] = "up", True, now
                if telemetry is not None:
                    telemetry.ready(node,now)
            elif current_outage.get(node) is o and physical[node] != "up":
                known[node] = False
            if rank == 3 and telemetry is not None:
                telemetry.detected(node,now,o["start_ms"])
            node_events.append(dict(time_ms=now, node_id=node, event=("failure", "repair_finished", "reboot_finished", "failure_detected")[rank], observed_available=known[node]))
        # Deadlines are terminal application requirements; the cutoff is censoring.
        for i in tuple(active.keys() | pending.keys()):
            if work[i].deadline_ms is not None and work[i].deadline_ms <= now:
                terminal(i, "deadline_missed", "task_deadline")
        if now >= horizon:
            for i in tuple(active.keys() | pending.keys()):
                terminal(i, "unfinished", "observation_cutoff")
            break
        start_ready()
        if telemetry is not None and now >= poll_index*period:
            deliver_sensor_reports()
            poll_index += 1
        for i, p in sorted(tuple(pending.items())):
            retries_used = len(by_task[i])-1
            max_retries = d["recovery"]["max_retries"] if recovery_enabled else 0
            if now >= p["detect_at_ms"] and retries_used >= max_retries:
                terminal(i, "execution_failure", "retry_budget_exhausted")
            elif now >= p["ready_ms"]:
                if not dispatch(i, retry=True) and now >= p["expires_ms"]:
                    terminal(i, "execution_failure", "backup_wait_expired")
        while arrival_index < len(incoming) and work[incoming[arrival_index]].arrival_ms <= now:
            i = incoming[arrival_index]
            arrival_index += 1
            if work[i].deadline_ms is not None and work[i].deadline_ms <= now:
                terminal(i, "deadline_missed", "task_deadline")
            elif not dispatch(i):
                terminal(i, "rejected", "no_observed_eligible_VM")
        # Resolve new zero-byte uploads before computing the next timestamp.
        while True:
            changed = settle_phases()
            changed = start_ready() or changed
            if not changed:
                break
        future = [horizon]
        if telemetry is not None:
            future.append(poll_index*period)
        if arrival_index < len(incoming):
            future.append(work[incoming[arrival_index]].arrival_ms)
        if edge_index < len(edges):
            future.append(edges[edge_index][0])
        for link_id, flows in flow_groups().items():
            if flows:
                rate = d["links"][link_id]["bandwidth_mbps"]/(8*len(flows))
                future.append(now + min(a["remaining_kb"] for a in flows)/rate)
        for a in active.values():
            if a["phase"] in ("compute", "upload_propagation", "download_propagation"):
                future.append(a["phase_end_ms"])
        for i, p in pending.items():
            future.extend(t for t in (p["detect_at_ms"], p["ready_ms"], p["expires_ms"]) if t > now)
        for i in active.keys() | pending.keys():
            if work[i].deadline_ms is not None:
                future.append(work[i].deadline_ms)
        end = min(t for t in future if t > now)
        advance(end)
    if len(outcomes) != len(work):
        raise RuntimeError("missing task outcomes")
    metrics = {s+"_count": sum(o["status"] == s for o in outcomes.values()) for s in (
        "completed", "rejected", "execution_failure", "deadline_missed", "unfinished")}
    n, completed = len(work), metrics["completed_count"]
    deadline_work = [i for i, w in enumerate(work) if w.deadline_ms is not None and w.deadline_ms <= horizon]
    deadline_success = sum(outcomes[i]["status"] == "completed" for i in deadline_work)
    success = [o for o in outcomes.values() if o["status"] == "completed"]
    latencies = [o["finish_ms"]-o["arrival_ms"] for o in success]
    recovery_delays = [o["finish_ms"] - next(a["end_ms"] for a in by_task[i] if a["status"] == "interrupted")
                       for i, o in outcomes.items() if o["recovered"]]
    host_j = sum(e[k] for e in host_energy.values() for k in ("idle_j", "dynamic_j", "down_j", "reboot_j"))
    traffic_j = sum(e["traffic_j"] for e in link_energy.values())
    net_idle_j = sum(e["idle_j"] for e in link_energy.values())
    recovery_j = sum(b["energy_j"] for b in blocks)
    total = host_j + traffic_j + net_idle_j + recovery_j
    metrics.update(submitted_count=n, completion_rate=completed/n if n else None,
        successful_makespan_ms=max((o["finish_ms"] for o in success), default=None),
        successful_mean_latency_ms=sum(latencies)/len(latencies) if latencies else None,
        batch_makespan_ms=max((o["finish_ms"] for o in success), default=0) if completed == n and n else None,
        deadline_eligible_count=len(deadline_work), deadline_success_count=deadline_success,
        deadline_success_rate=deadline_success/len(deadline_work) if deadline_work else None,
        interrupted_task_count=len(affected), interrupted_attempt_count=sum(a["status"] == "interrupted" for a in attempts),
        retry_attempt_count=sum(a["number"] > 1 for a in attempts), recovered_task_count=sum(o["recovered"] for o in success),
        recovery_success_rate=sum(o["recovered"] for o in success)/len(affected) if affected else None,
        mean_recovery_to_delivery_ms=sum(recovery_delays)/len(recovery_delays) if recovery_delays else None,
        observed_unsuccessful_fraction=1-completed/n if n else None,
        known_terminal_failure_fraction=(metrics["rejected_count"]+metrics["execution_failure_count"]+metrics["deadline_missed_count"])/n if n else None,
        host_energy_j=host_j, network_traffic_energy_j=traffic_j, network_idle_energy_j=net_idle_j,
        recovery_energy_j=recovery_j, total_energy_j=total,
        energy_per_completed_task_j=total/completed if completed else None,
        nonsuccessful_attempt_attributable_energy_j=sum(a["attributable_energy_j"] for a in attempts if a["status"] != "completed"))
    availability = []
    for node, f in sorted(d["failures"].items()):
        down = sum(max(0, min(horizon, o["end_ms"])-o["start_ms"]) for o in trace if o["node_id"] == node and o["start_ms"] < horizon)
        uptime = horizon-down
        failures = sum(o["node_id"] == node and o["start_ms"] < horizon for o in trace)
        mean_up = f["scale_ms"] * (math.gamma(1+1/f["shape"]) if f["distribution"] == "weibull" else 1)
        theoretical = 1.0 if f["distribution"] == "none" else mean_up/(mean_up+f["repair_ms"]+f["reboot_ms"])
        availability.append(dict(node_id=node, observed_up_ms=uptime, observed_availability=uptime/horizon,
                                 failures=failures, exponential_rate_mle_per_s=1000*failures/uptime if uptime and f["distribution"] == "exponential" else None,
                                 configured_steady_state_availability=theoretical,
                                 theoretical_applies_to_trace=outages is None))
    o = d["objective"]
    reasons = []
    if n == 0:
        reasons.append("empty_workload")
    if metrics["completion_rate"] is not None and metrics["completion_rate"] < o["minimum_completion_rate"]:
        reasons.append("completion_constraint")
    if metrics["deadline_success_rate"] is not None and metrics["deadline_success_rate"] < o["minimum_deadline_success_rate"]:
        reasons.append("deadline_constraint")
    if not completed:
        reasons.append("no_successful_makespan")
    normalized = dict(time=metrics["successful_makespan_ms"]/o["scales"]["time_ms"] if completed else None,
                      energy=total/o["scales"]["energy_j"], failure=metrics["observed_unsuccessful_fraction"])
    score = sum(o["weights"][k]*v for k, v in normalized.items()) if not reasons else None
    for i, outcome in outcomes.items():
        elapsed = sum(a["end_ms"]-a["dispatch_ms"] for a in by_task[i])
        blocked = sum(b["end_ms"]-b["start_ms"] for b in blocks if b["task_id"] == work[i].task_id)
        outcome.update(attempt_elapsed_ms=elapsed, recovery_blocked_ms=blocked)
        if not math.isclose(elapsed+blocked, outcome["terminal_ms"]-work[i].arrival_ms, abs_tol=1e-8):
            raise RuntimeError("original task time accounting mismatch")
    balances = ledger.final_balances()
    payload = [asdict(w) for w in work]
    report = dict(schema_version=1, kind="host_network_objective_simulation", policy=policy,
        run_configuration=config.manifest(), model_configuration=d, recovery_enabled=recovery_enabled,
        workload=dict(tasks=payload, sha256=_digest(payload)),
        outage_trace=dict(outages=trace, sha256=_digest(trace), failure_seed=failure_seed if outages is None else None),
        metrics=metrics, objective=dict(normalized=normalized, weights=o["weights"], scales=o["scales"],
                                       feasible=not reasons, infeasibility_reasons=reasons, score=score),
        attempts=attempts, outcomes=list(outcomes.values()), recovery_intervals=blocks,
        decisions=decisions, node_events=node_events, availability=availability,
        host_energy=host_energy, link_energy=link_energy, power_segments=power_segments, link_segments=link_segments,
        resource_events=[asdict(e) for e in ledger.events], available_resources=balances,
        accounting=dict(cutoff_ms=horizon, energy_scope="all hosts and links over fixed observation window plus coordinator",
            host_idle_allocation="charged once per host, never duplicated per VM",
            task_attribution="dynamic host energy shared by active VM MIPS plus its actual transferred bytes; excludes shared idle",
            predictions="causal approximations using present queues and traffic; no future failure trace or arrivals",
            censoring="unfinished at cutoff is distinct from terminal failure; eventual batch makespan is undefined unless all complete"))
    if controller is not None:
        report["rl"]=controller.finish(reward_cursor())
        report["rl"]["telemetry_messages"]=telemetry.messages
    return report
