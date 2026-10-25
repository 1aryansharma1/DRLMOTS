"""A-030 explicit MI/MIPS, host power, shared network and renewal failures.

A separate contract: no reinterpretation of legacy bytes or DQN checkpoints.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import random

import yaml

from drlmots.entities import Task, validate_resource_amounts
from drlmots.normalization import _digest
from drlmots.recovery_config import node_membership, number
from drlmots.run_config import _UniqueKeyLoader


def fields(value, names, context):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"{context} requires exactly: {names}")


def integer(value, context):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{context} must be an integer in [0, 2**63)")


@dataclass(frozen=True)
class WorkItem:
    task_id: str
    arrival_ms: float
    compute_mi: float
    input_kb: float
    output_kb: float
    deadline_ms: float | None = None
    required_resources: dict = field(default_factory=dict)

    def __post_init__(self):
        # Reuse the established resource and identifier validation, never its byte runtime.
        Task(self.task_id, self.arrival_ms, self.compute_mi, max(self.input_kb, 1e-30), self.required_resources)
        number("input kB", self.input_kb)
        number("output kB", self.output_kb)
        if self.deadline_ms is not None:
            number("absolute deadline", self.deadline_ms)
            if self.deadline_ms < self.arrival_ms:
                raise ValueError("deadline cannot precede arrival")
        object.__setattr__(self, "required_resources", validate_resource_amounts(self.required_resources, "demands"))


def power_w(curve, utilization):
    """Piecewise linear interpolation; exactly two endpoints gives linear power."""
    if not 0 <= utilization <= 1 + 1e-12:
        raise ValueError("host utilization outside [0,1]")
    utilization = min(1.0, utilization)
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if utilization <= x1:
            return y0 + (y1-y0) * (utilization-x0) / (x1-x0)
    return float(curve[-1][1])


class ObjectiveModel:
    def __init__(self, data):
        self.data = deepcopy(data)
        d = self.data
        fields(d, "schema_version kind coefficient_provenance horizon_ms workload hosts vms links failures recovery objective", "objective model")
        if type(d["schema_version"]) is not int or d["schema_version"] != 1 or d["kind"] != "host_network_objectives":
            raise ValueError("unsupported objective model contract")
        if not isinstance(d["coefficient_provenance"], str) or not d["coefficient_provenance"].strip():
            raise ValueError("coefficient provenance must be documented")
        number("horizon", d["horizon_ms"], positive=True)
        w = d["workload"]
        fields(w, "task_count seed interarrival_ms compute_mi input_kb output_kb relative_deadline_ms required_resources", "workload")
        integer(w["task_count"], "task count")
        integer(w["seed"], "workload seed")
        for key in ("interarrival_ms", "compute_mi", "input_kb", "output_kb", "relative_deadline_ms"):
            pair = w[key]
            if key == "relative_deadline_ms" and pair is None:
                continue
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"{key} requires a two element range")
            for value in pair:
                number(key, value, positive=key == "compute_mi")
            if pair[0] > pair[1]:
                raise ValueError(f"{key} range is reversed")
        validate_resource_amounts(w["required_resources"], "workload resources")
        for group in ("hosts", "vms", "links", "failures"):
            if not isinstance(d[group], dict) or not d[group] or any(not isinstance(k, str) or not k.strip() for k in d[group]):
                raise ValueError(f"{group} must be a nonempty ID mapping")
        for host in d["hosts"].values():
            fields(host, "capacity_mips power_curve down_power_w reboot_power_w", "host")
            number("host MIPS", host["capacity_mips"], positive=True)
            for key in ("down_power_w", "reboot_power_w"):
                number(key, host[key])
            curve = host["power_curve"]
            if not isinstance(curve, list) or len(curve) < 2:
                raise ValueError("power curve needs endpoints at 0 and 1")
            for point in curve:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("power curve requires utilization/power pairs")
                number("utilization", point[0])
                number("watts", point[1])
            if curve[0][0] != 0 or curve[-1][0] != 1 or any(a[0] >= b[0] or a[1] > b[1] for a, b in zip(curve, curve[1:])):
                raise ValueError("power curve must increase in utilization and not decrease in power")
        for vm in d["vms"].values():
            fields(vm, "host_id mips link_id", "VM")
            if vm["host_id"] not in d["hosts"] or vm["link_id"] not in d["links"]:
                raise ValueError("unknown VM host/link")
            number("VM MIPS", vm["mips"], positive=True)
        for host_id, host in d["hosts"].items():
            if sum(v["mips"] for v in d["vms"].values() if v["host_id"] == host_id) > host["capacity_mips"]:
                raise ValueError("VM MIPS exceed physical host capacity")
        for link in d["links"].values():
            fields(link, "bandwidth_mbps propagation_ms joules_per_kb idle_power_w", "shared link")
            for key, value in link.items():
                number(key, value, positive=key == "bandwidth_mbps")
        for failure in d["failures"].values():
            fields(failure, "distribution scale_ms shape repair_ms reboot_ms detection_ms", "failure process")
            if failure["distribution"] not in ("exponential", "weibull", "none"):
                raise ValueError("unknown uptime distribution")
            for key in ("scale_ms", "shape", "repair_ms", "reboot_ms", "detection_ms"):
                number(key, failure[key], positive=key in ("scale_ms", "shape"))
            if failure["distribution"] == "exponential" and failure["shape"] != 1:
                raise ValueError("exponential shape must equal one")
            if failure["distribution"] != "none" and failure["repair_ms"] + failure["reboot_ms"] <= 0:
                raise ValueError("positive repair/reboot time required")
        r = d["recovery"]
        fields(r, "max_retries restart_ms wait_ms coordinator_power_w", "recovery")
        integer(r["max_retries"], "max retries")
        for key in ("restart_ms", "wait_ms", "coordinator_power_w"):
            number(key, r[key])
        o = d["objective"]
        fields(o, "weights scales minimum_completion_rate minimum_deadline_success_rate", "objective")
        fields(o["weights"], "time energy failure", "weights")
        fields(o["scales"], "time_ms energy_j decision_time_ms decision_energy_j", "fixed scales")
        for value in o["weights"].values():
            number("weight", value)
        if not math.isclose(sum(o["weights"].values()), 1, abs_tol=1e-12):
            raise ValueError("objective weights must sum to one")
        for value in o["scales"].values():
            number("scale", value, positive=True)
        for key in ("minimum_completion_rate", "minimum_deadline_success_rate"):
            number(key, o[key])
            if o[key] > 1:
                raise ValueError("service constraints must lie in [0,1]")

    def manifest(self):
        return deepcopy(self.data)

    def validate_topology(self, config):
        config.manifest()
        d, topology = self.data, config.topology
        if config.placement is None:
            raise ValueError("objective model requires a source/reachability topology")
        nodes = node_membership(topology)
        if set(d["vms"]) != set(nodes) or set(d["hosts"]) != {h.host_id for h in topology.hosts} or set(d["failures"]) != set(nodes.values()):
            raise ValueError("model must cover exactly all topology hosts, VMs and nodes")
        host_nodes = {h: prefix + getattr(node, attr) for prefix, group, attr in (
            ("fog:", topology.fog_nodes, "fog_node_id"), ("cloud:", topology.cloud_datacenters, "datacenter_id"))
            for node in group for h in node.host_ids}
        for vm_id, vm in d["vms"].items():
            if host_nodes[vm["host_id"]] != nodes[vm_id]:
                raise ValueError("VM must map to a physical host in its own node")
        if config.resource_units.get("memory_gb") != "GB" or any(v.resource_capacities is None or "memory_gb" not in v.resource_capacities for v in topology.vms):
            raise ValueError("physical memory accounting requires declared memory_gb in GB on every VM")
        for host in topology.hosts:
            members = [v for v in topology.vms if d["vms"][v.vm_id]["host_id"] == host.host_id]
            if any(v.queue or v.queue_capacity is None or v.resource_capacities is None for v in members):
                raise ValueError("VMs require empty queues, capacities and queue limits")
            if sum(v.resource_capacities.get("memory_gb", 0) for v in members) > host.memory_gb:
                raise ValueError("VM memory exceeds physical host capacity")
        return nodes, host_nodes


def load_objective_model(path):
    return ObjectiveModel(yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader))


def generate_work(model, seed=None):
    w = model.data["workload"]
    seed = w["seed"] if seed is None else seed
    integer(seed, "workload seed")
    rng, now, result = random.Random(seed), 0.0, []
    for index in range(w["task_count"]):
        if index:
            now += rng.uniform(*w["interarrival_ms"])
        work, inputs, outputs = (rng.uniform(*w[key]) for key in ("compute_mi", "input_kb", "output_kb"))
        deadline = None if w["relative_deadline_ms"] is None else now + rng.uniform(*w["relative_deadline_ms"])
        result.append(WorkItem(f"work-{index+1:06d}", now, work, inputs, outputs, deadline, w["required_resources"]))
    return result


def failure_trace(model, seed):
    """Policy-independent per-node renewal streams; perfect repair resets age."""
    integer(seed, "failure seed")
    result, horizon = [], model.data["horizon_ms"]
    for node, f in sorted(model.data["failures"].items()):
        if f["distribution"] == "none":
            continue
        rng = random.Random(int(_digest({"failure_seed": seed, "node": node}), 16))
        now = 0.0
        while now < horizon:
            uptime = rng.expovariate(1 / f["scale_ms"]) if f["distribution"] == "exponential" else rng.weibullvariate(f["scale_ms"], f["shape"])
            start = now + uptime
            if start >= horizon:
                break
            repair = start + f["repair_ms"]
            end = repair + f["reboot_ms"]
            if not now < start < end:
                raise ValueError("failure process lost clock precision")
            result.append(dict(node_id=node, start_ms=start, repair_end_ms=repair, end_ms=end,
                               detection_ms=start + f["detection_ms"]))
            if len(result) > 100000:
                raise ValueError("too many failures")
            now = end
    return sorted(result, key=lambda o: (o["start_ms"], o["node_id"]))


def conditional_failure_probability(f, age_ms, exposure_ms):
    if f["distribution"] == "none":
        return 0.0
    shape = 1 if f["distribution"] == "exponential" else f["shape"]
    delta = ((age_ms + exposure_ms) / f["scale_ms"])**shape - (age_ms / f["scale_ms"])**shape
    return -math.expm1(-delta)
