"""A-031 / contract 4: standalone enhanced observation/action/reward boundary.

No simulator controller or training integration is installed by this module.
Telemetry is an observed view supplied by a future adapter, never a physics dump.
"""
from dataclasses import asdict, dataclass
import json
import math

from drlmots.normalization import _canonical, _digest
from drlmots.objective_model import ObjectiveModel, WorkItem, fields, integer, number, power_w


def _count(value, name, maximum=None):
    integer(value, name)
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds its budget")


def _pairs(value, name):
    if not isinstance(value, tuple) or any(not isinstance(p, tuple) or len(p) != 2 for p in value):
        raise ValueError(f"{name} must be immutable name/value pairs")
    result = dict(value)
    if len(result) != len(value) or any(not isinstance(k, str) or not k for k in result):
        raise ValueError(f"duplicate/invalid {name} identifier")
    for amount in result.values():
        number(name, amount)
    return result


@dataclass(frozen=True)
class PlacementRequest:
    task_id: str
    arrival_ms: float
    compute_mi: float
    input_kb: float
    output_kb: float
    deadline_ms: float | None
    resources: tuple[tuple[str, float], ...]
    attempts_used: int = 0
    previous_node_id: str | None = None
    recovery_wait_remaining_ms: float = 0.0

    @classmethod
    def from_work(cls, work, **recovery):
        if not isinstance(work, WorkItem):
            raise TypeError("explicit MI WorkItem required")
        return cls(work.task_id, work.arrival_ms, work.compute_mi, work.input_kb,
                   work.output_kb, work.deadline_ms, tuple(sorted(work.required_resources.items())), **recovery)


@dataclass(frozen=True)
class QueuedTelemetry:
    phase: str  # upload, propagation, ready
    remaining_upload_kb: float
    remaining_propagation_ms: float
    compute_mi: float
    resources: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class VMTelemetry:
    vm_id: str
    reported_at_ms: float
    running_remaining_ms: float | None
    waiting: tuple[QueuedTelemetry, ...]
    available_resources: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class HostTelemetry:
    host_id: str
    reported_at_ms: float
    utilization: float | None


@dataclass(frozen=True)
class LinkTelemetry:
    link_id: str
    reported_at_ms: float
    upload_flows: int
    download_flows: int
    remaining_upload_kb: float
    remaining_download_kb: float


@dataclass(frozen=True)
class NodeTelemetry:
    node_id: str
    reported_at_ms: float
    available: bool | None
    up_since_ms: float | None
    detected_failures: int = 0


@dataclass(frozen=True)
class ObservationSnapshot:
    time_ms: float
    arrived_count: int
    completed_count: int
    unsuccessful_count: int
    request: PlacementRequest
    vms: tuple[VMTelemetry, ...]
    hosts: tuple[HostTelemetry, ...]
    links: tuple[LinkTelemetry, ...]
    nodes: tuple[NodeTelemetry, ...]


@dataclass(frozen=True)
class EncodedObservation:
    context_sha256: str
    time_ms: float
    task_id: str
    placement_kind: str
    values: tuple[float, ...]
    legal_mask: tuple[bool, ...]


def _layout(c):
    result = []
    def add(prefix, names, unit, scale):
        result.extend((prefix+name, unit, scale) for name in names.split())
    h, s = c["horizon_ms"], c["observation_scales"]
    add("episode.", "remaining_ms", "ms", h)
    add("episode.", "arrived completed unsuccessful", "tasks", c["planned_task_count"])
    add("task.", "age_ms deadline_remaining_ms recovery_wait_remaining_ms", "ms", h)
    add("task.", "has_deadline deadline_expired is_recovery", "binary", None)
    add("task.", "attempts_used", "attempts", c["recovery"]["max_retries"]+1)
    add("task.", "compute_mi", "MI", s["compute_mi"])
    add("task.", "input_kb", "decimal_kB", s["input_kb"])
    add("task.", "output_kb", "decimal_kB", s["output_kb"])
    for r, unit in c["resource_units"].items():
        result.append(("task.demand."+r, unit, s["resources"][r]))
    for node in c["failures"]:
        p = f"node[{node}]."
        add(p, "health_known observed_up age_known", "binary", None)
        add(p, "telemetry_age_ms observed_up_age_ms", "ms", h)
        add(p, "detected_failures", "events", 1)
    for host, spec in c["hosts"].items():
        p = f"host[{host}]."
        add(p, "utilization_known", "binary", None)
        add(p, "observed_utilization", "fraction", 1)
        add(p, "estimated_up_power_w", "W", max(1, spec["power_curve"][-1][1]))
        add(p, "telemetry_age_ms", "ms", h)
    for link in c["links"]:
        p = f"link[{link}]."
        add(p, "upload_flows download_flows", "flows", len(c["vms"]))
        add(p, "remaining_upload_kb remaining_download_kb", "decimal_kB", max(s["input_kb"], s["output_kb"]))
        add(p, "telemetry_age_ms", "ms", h)
    for vm, spec in c["vms"].items():
        p = f"vm[{vm}]."
        add(p, "is_fog reachable previous_failed_node running", "binary", None)
        add(p, "mips", "MIPS", s["mips"])
        add(p, "queue_count", "tasks", spec["queue_capacity"])
        add(p, "running_remaining_ms telemetry_age_ms", "ms", h)
        for r, unit in c["resource_units"].items():
            add(p+f"resource[{r}].", "capacity_known", "binary", None)
            add(p+f"resource[{r}].", "capacity available", unit, s["resources"][r])
        for slot in range(spec["queue_capacity"]):
            q = p+f"waiting[{slot}]."
            add(q, "present upload propagation ready", "binary", None)
            add(q, "remaining_upload_kb", "decimal_kB", s["input_kb"])
            add(q, "remaining_propagation_ms", "ms", h)
            add(q, "compute_mi", "MI", s["compute_mi"])
            for r, unit in c["resource_units"].items():
                result.append((q+"demand."+r, unit, s["resources"][r]))
    return result


@dataclass(frozen=True)
class EnhancedRLContract:
    """Immutable canonical public configuration; action order is sorted VM ID."""
    _context_json: str

    def __post_init__(self):
        if not isinstance(self._context_json,str):
            raise TypeError("canonical contract context text required")
        try:
            c=json.loads(self._context_json)
            expected=set("contract_version kind horizon_ms planned_task_count source_end_node_id resource_units vms hosts links failures recovery objective observation_scales normalization discount completion_penalty deadline_penalty boundary termination features".split())
            if not isinstance(c,dict) or set(c)!=expected or type(c["contract_version"]) is not int or c["contract_version"]!=4 or c["kind"]!="enhanced_scheduler_rl_contract":
                raise ValueError("unsupported enhanced RL contract context")
            if _canonical(c)!=self._context_json or c["discount"]!=1.0:
                raise ValueError("contract requires canonical context and undiscounted finite episodes")
            number("contract horizon",c["horizon_ms"],positive=True)
            _count(c["planned_task_count"],"planned task count")
            if not c["planned_task_count"]:
                raise ValueError("RL episodes require a positive task count")
            for name in ("completion_penalty","deadline_penalty"):
                number(name,c[name],positive=True)
            if any(not isinstance(c[name],dict) or not c[name] or list(c[name])!=sorted(c[name]) for name in ("vms","hosts","links","failures")):
                raise ValueError("contract IDs must be nonempty sorted mappings")
            for spec in c["vms"].values():
                _count(spec["queue_capacity"],"queue capacity")
                if not spec["queue_capacity"]:
                    raise ValueError("positive VM queue limit required")
            if c["normalization"]!="x/(x+fixed_public_scale); binary unchanged; no fitting or clipping" or type(c["discount"]) not in (int,float):
                raise ValueError("unsupported normalization/discount semantics")
            units=c["resource_units"]
            if not isinstance(units,dict) or any(not isinstance(k,str) or not k.strip() or not isinstance(v,str) or not v.strip() for k,v in units.items()):
                raise ValueError("invalid public resource units")
            fields(c["observation_scales"],"compute_mi input_kb output_kb mips resources","observation scales")
            if set(c["observation_scales"]["resources"])!=set(units):
                raise ValueError("resource scale catalogue differs from declared units")
            for spec in c["vms"].values():
                fields(spec,"host_id mips link_id node_id tier queue_capacity capacities reachable","contract VM")
                if type(spec["reachable"]) is not bool or spec["tier"] not in ("fog","cloud") or spec["node_id"] not in c["failures"]:
                    raise ValueError("invalid contract VM reachability/tier/node")
                if not isinstance(spec["capacities"],dict) or set(spec["capacities"])-set(units):
                    raise ValueError("undeclared resource capacity")
                for amount in spec["capacities"].values():
                    number("VM resource capacity",amount,positive=True)
            # Reuse strict physical/reward schemas rather than accepting extra fault-trace fields.
            ObjectiveModel(dict(schema_version=1,kind="host_network_objectives",coefficient_provenance="public RL context",
                horizon_ms=c["horizon_ms"],workload=dict(task_count=c["planned_task_count"],seed=0,interarrival_ms=[0,0],
                    compute_mi=[1,1],input_kb=[0,0],output_kb=[0,0],relative_deadline_ms=None,required_resources={}),
                hosts=c["hosts"],vms={v:{k:spec[k] for k in ("host_id","mips","link_id")} for v,spec in c["vms"].items()},
                links=c["links"],failures=c["failures"],recovery=c["recovery"],objective=c["objective"]))
            layout=[dict(name=name,unit=unit,scale=scale) for name,unit,scale in _layout(c)]
            if c["features"]!=layout or len({f["name"] for f in layout})!=len(layout):
                raise ValueError("contract feature layout differs from version 4")
            for feature in layout:
                if feature["scale"] is not None:
                    number("observation scale",feature["scale"],positive=True)
        except (KeyError,TypeError,json.JSONDecodeError) as error:
            raise ValueError("malformed enhanced RL contract context") from error

    @classmethod
    def from_configs(cls, run, model, *, planned_task_count=None,
                     completion_penalty=10.0, deadline_penalty=10.0):
        model = ObjectiveModel(model.manifest())
        nodes, _ = model.validate_topology(run)
        d = model.manifest()
        n = d["workload"]["task_count"] if planned_task_count is None else planned_task_count
        _count(n, "planned task count")
        if n == 0:
            raise ValueError("RL episodes need a positive declared task budget")
        for name, value in (("completion penalty", completion_penalty), ("deadline penalty", deadline_penalty)):
            number(name, value, positive=True)
        eligible = {v.vm_id for v in run.placement.candidates(run.topology)}
        vms = {v.vm_id: {**d["vms"][v.vm_id], "node_id": nodes[v.vm_id], "tier": v.tier.value,
                       "queue_capacity": v.queue_capacity, "capacities": dict(v.resource_capacities),
                       "reachable": v.vm_id in eligible} for v in sorted(run.topology.vms, key=lambda v: v.vm_id)}
        units = dict(sorted(run.resource_units.items()))
        scales = {k: max(1.0, d["workload"][k][1]) for k in ("compute_mi", "input_kb", "output_kb")}
        scales["mips"] = max(v["mips"] for v in vms.values())
        scales["resources"] = {r: max([1.0]+[v["capacities"].get(r, 0) for v in vms.values()]) for r in units}
        c = dict(contract_version=4, kind="enhanced_scheduler_rl_contract", horizon_ms=d["horizon_ms"],
                 planned_task_count=n, source_end_node_id=run.placement.source_end_node_id,
                 resource_units=units, vms=vms, hosts=dict(sorted(d["hosts"].items())),
                 links=dict(sorted(d["links"].items())), failures=dict(sorted(d["failures"].items())),
                 recovery=d["recovery"], objective=d["objective"], observation_scales=scales,
                 normalization="x/(x+fixed_public_scale); binary unchanged; no fitting or clipping",
                 discount=1.0, completion_penalty=completion_penalty, deadline_penalty=deadline_penalty,
                 boundary="before initial/recovery dispatch; after same-time settlement and observed telemetry updates",
                 termination="finite-horizon objective at H; no bootstrap at H; report unfinished remains censored")
        c["features"] = [dict(name=name, unit=unit, scale=scale) for name, unit, scale in _layout(c)]
        return cls(_canonical(c))

    @property
    def context(self):
        return json.loads(self._context_json)

    @property
    def sha256(self):
        return _digest(self.context)

    @property
    def vm_ids(self):
        return tuple(self.context["vms"])

    def manifest(self):
        return {**self.context, "sha256": self.sha256}

    def encode(self, snapshot):
        if not isinstance(snapshot, ObservationSnapshot) or not isinstance(snapshot.request, PlacementRequest):
            raise TypeError("observed telemetry snapshot and placement request required")
        c, raw = self.context, {}
        t, h, req = snapshot.time_ms, c["horizon_ms"], snapshot.request
        number("decision time", t)
        if t >= h:
            raise ValueError("no placement decision at/after the episode horizon")
        for name in ("arrived_count", "completed_count", "unsuccessful_count"):
            _count(getattr(snapshot, name), name, c["planned_task_count"])
        if snapshot.completed_count+snapshot.unsuccessful_count >= snapshot.arrived_count or not snapshot.arrived_count:
            raise ValueError("inconsistent observed task counts")
        demands = _pairs(req.resources, "request resources")
        WorkItem(req.task_id, req.arrival_ms, req.compute_mi, req.input_kb, req.output_kb, req.deadline_ms, demands)
        if req.arrival_ms > t or set(demands)-set(c["resource_units"]):
            raise ValueError("future task arrival or undeclared task resource")
        _count(req.attempts_used, "attempts used", c["recovery"]["max_retries"])
        number("recovery wait remaining", req.recovery_wait_remaining_ms)
        if req.recovery_wait_remaining_ms > c["recovery"]["wait_ms"]:
            raise ValueError("recovery wait exceeds configured limit")
        if req.attempts_used == 0:
            if req.previous_node_id is not None or req.recovery_wait_remaining_ms != 0:
                raise ValueError("initial placement cannot contain recovery state")
        elif req.previous_node_id not in c["failures"]:
            raise ValueError("recovery needs a known previous failed node")
        raw.update({"episode.remaining_ms":h-t, "episode.arrived":snapshot.arrived_count,
                    "episode.completed":snapshot.completed_count, "episode.unsuccessful":snapshot.unsuccessful_count,
                    "task.age_ms":t-req.arrival_ms, "task.deadline_remaining_ms":max(0, req.deadline_ms-t) if req.deadline_ms is not None else 0,
                    "task.recovery_wait_remaining_ms":req.recovery_wait_remaining_ms,
                    "task.has_deadline":int(req.deadline_ms is not None), "task.deadline_expired":int(req.deadline_ms is not None and req.deadline_ms <= t),
                    "task.is_recovery":int(req.attempts_used > 0), "task.attempts_used":req.attempts_used,
                    "task.compute_mi":req.compute_mi, "task.input_kb":req.input_kb, "task.output_kb":req.output_kb})
        for r in c["resource_units"]:
            raw["task.demand."+r] = demands.get(r, 0)
        def index(items, kind, id_field, expected):
            if not isinstance(items, tuple) or any(type(v) is not kind for v in items):
                raise TypeError("telemetry must contain immutable typed tuples")
            result = {getattr(v, id_field):v for v in items}
            if len(result) != len(items) or set(result) != set(expected):
                raise ValueError("telemetry IDs must cover their context exactly once")
            for item in items:
                number("reported time", item.reported_at_ms)
                if item.reported_at_ms > t:
                    raise ValueError("future telemetry is forbidden")
            return result
        nv = index(snapshot.nodes, NodeTelemetry, "node_id", c["failures"])
        hv = index(snapshot.hosts, HostTelemetry, "host_id", c["hosts"])
        lv = index(snapshot.links, LinkTelemetry, "link_id", c["links"])
        vv = index(snapshot.vms, VMTelemetry, "vm_id", c["vms"])
        for node, view in nv.items():
            if view.available is not None and type(view.available) is not bool:
                raise ValueError("observed health must be bool or unknown")
            _count(view.detected_failures, "detected failures")
            if view.up_since_ms is not None:
                number("observed up since", view.up_since_ms)
                if view.available is not True or view.up_since_ms > view.reported_at_ms:
                    raise ValueError("up age requires a known available observation")
            p=f"node[{node}]."
            raw.update({p+"health_known":int(view.available is not None),p+"observed_up":int(view.available is True),
                        p+"age_known":int(view.up_since_ms is not None),p+"telemetry_age_ms":t-view.reported_at_ms,
                        p+"observed_up_age_ms":t-view.up_since_ms if view.up_since_ms is not None else 0,
                        p+"detected_failures":view.detected_failures})
        for host, view in hv.items():
            if view.utilization is not None:
                number("observed utilization",view.utilization)
                if view.utilization > 1:
                    raise ValueError("observed utilization exceeds one")
            p=f"host[{host}]."
            raw.update({p+"utilization_known":int(view.utilization is not None),p+"observed_utilization":view.utilization or 0,
                        p+"estimated_up_power_w":power_w(c["hosts"][host]["power_curve"],view.utilization) if view.utilization is not None else 0,
                        p+"telemetry_age_ms":t-view.reported_at_ms})
        for link, view in lv.items():
            p=f"link[{link}]."
            for name in ("upload_flows","download_flows"):
                _count(getattr(view,name),name)
            for direction in ("upload","download"):
                size=getattr(view,"remaining_"+direction+"_kb")
                number("remaining traffic",size)
                if not getattr(view,direction+"_flows") and size:
                    raise ValueError("remaining bytes require an observed active transfer")
            for name in ("upload_flows","download_flows","remaining_upload_kb","remaining_download_kb"):
                raw[p+name]=getattr(view,name)
            raw[p+"telemetry_age_ms"]=t-view.reported_at_ms
        masks=[]
        for vm, spec in c["vms"].items():
            view, p = vv[vm], f"vm[{vm}]."
            if view.running_remaining_ms is not None:
                number("observed remaining computation",view.running_remaining_ms)
            if not isinstance(view.waiting,tuple) or len(view.waiting)>spec["queue_capacity"]:
                raise ValueError("invalid observed VM waiting queue")
            available=_pairs(view.available_resources,"available resources")
            if set(available)!=set(spec["capacities"]) or any(available[r]>spec["capacities"][r] for r in available):
                raise ValueError("available resources disagree with declared capacities")
            if view.running_remaining_ms is None and available != spec["capacities"]:
                raise ValueError("idle VM telemetry must report released resources")
            raw.update({p+"is_fog":int(spec["tier"]=="fog"),p+"reachable":int(spec["reachable"]),
                        p+"previous_failed_node":int(spec["node_id"]==req.previous_node_id),
                        p+"running":int(view.running_remaining_ms is not None),p+"mips":spec["mips"],
                        p+"queue_count":len(view.waiting),p+"running_remaining_ms":view.running_remaining_ms or 0,
                        p+"telemetry_age_ms":t-view.reported_at_ms})
            for r in c["resource_units"]:
                raw.update({p+f"resource[{r}].capacity_known":int(r in spec["capacities"]),
                            p+f"resource[{r}].capacity":spec["capacities"].get(r,0),
                            p+f"resource[{r}].available":available.get(r,0)})
            for slot in range(spec["queue_capacity"]):
                q=view.waiting[slot] if slot<len(view.waiting) else None
                if q is not None:
                    if type(q) is not QueuedTelemetry or q.phase not in ("upload","propagation","ready"):
                        raise ValueError("unknown observed queue phase")
                    number("queued compute",q.compute_mi,positive=True)
                    number("queued upload",q.remaining_upload_kb)
                    number("queued propagation",q.remaining_propagation_ms)
                    if (q.phase!="upload" and q.remaining_upload_kb) or (q.phase!="propagation" and q.remaining_propagation_ms):
                        raise ValueError("queue remaining work contradicts its phase")
                    qr=_pairs(q.resources,"queued demands")
                    if set(qr)-set(c["resource_units"]) or any(amount<=0 or amount>spec["capacities"].get(r,0) for r,amount in qr.items()):
                        raise ValueError("queued demands must fit the destination VM")
                else:
                    qr={}
                qp=p+f"waiting[{slot}]."
                raw[qp+"present"]=int(q is not None)
                for phase in ("upload","propagation","ready"):
                    raw[qp+phase]=int(q is not None and q.phase==phase)
                for name in ("remaining_upload_kb","remaining_propagation_ms","compute_mi"):
                    raw[qp+name]=getattr(q,name) if q is not None else 0
                for r in c["resource_units"]:
                    raw[qp+"demand."+r]=qr.get(r,0)
            fits=all(amount<=spec["capacities"].get(r,0) for r,amount in demands.items())
            masks.append(bool(spec["reachable"] and nv[spec["node_id"]].available is True and
                              spec["node_id"]!=req.previous_node_id and fits and
                              len(view.waiting)<spec["queue_capacity"] and not raw["task.deadline_expired"]))
        if set(raw)!={f["name"] for f in c["features"]}:
            raise RuntimeError("observation layout mismatch")
        values=[]
        for f in c["features"]:
            value,scale=raw[f["name"]],f["scale"]
            number("observation feature",value)
            if scale is None:
                if value not in (0,1):
                    raise ValueError("binary feature must be zero or one")
                values.append(float(value))
            else:
                values.append(value/(value+scale) if value<=scale else 1/(1+scale/value))
        return EncodedObservation(self.sha256,t,req.task_id,"recovery" if req.attempts_used else "initial",tuple(values),tuple(masks))

    def validate_action(self, observation, action):
        if not isinstance(observation,EncodedObservation) or observation.context_sha256!=self.sha256:
            raise ValueError("incompatible observation contract")
        if len(observation.values)!=len(self.context["features"]) or len(observation.legal_mask)!=len(self.vm_ids):
            raise ValueError("observation dimension mismatch")
        if any(type(x) is not bool for x in observation.legal_mask):
            raise ValueError("action mask must contain booleans")
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in observation.values):
            raise ValueError("invalid encoded observation")
        if action is None:
            if any(observation.legal_mask):
                raise ValueError("voluntary rejection/waiting is not a learned action")
            return None
        if type(action) is not int or not 0<=action<len(self.vm_ids) or not observation.legal_mask[action]:
            raise ValueError("action must select an eligible VM")
        return self.vm_ids[action]


@dataclass(frozen=True)
class RewardSnapshot:
    time_ms: float = 0.0
    total_energy_j: float = 0.0
    arrived_count: int = 0
    completed_count: int = 0
    rejected_count: int = 0
    execution_failure_count: int = 0
    deadline_missed_count: int = 0
    unfinished_count: int = 0
    deadline_eligible_count: int = 0
    deadline_success_count: int = 0
    successful_makespan_ms: float = 0.0

    @property
    def unsuccessful_count(self):
        return self.rejected_count+self.execution_failure_count+self.deadline_missed_count+self.unfinished_count


@dataclass(frozen=True)
class RewardDelta:
    context_sha256: str
    start_ms: float
    end_ms: float
    energy_reward: float
    time_reward: float
    failure_reward: float
    service_penalty_reward: float
    terminated: bool
    bootstrap_discount: float

    @property
    def total(self):
        return math.fsum((self.energy_reward,self.time_reward,self.failure_reward,self.service_penalty_reward))


def _validate_reward_snapshot(c,s):
    if type(s) is not RewardSnapshot:
        raise TypeError("typed cumulative reward snapshot required")
    n,h=c["planned_task_count"],c["horizon_ms"]
    for name in ("time_ms","total_energy_j","successful_makespan_ms"):
        number(name,getattr(s,name))
    if s.time_ms>h or s.successful_makespan_ms>s.time_ms:
        raise ValueError("reward snapshot contains future time")
    for name in ("arrived_count","completed_count","rejected_count","execution_failure_count","deadline_missed_count","unfinished_count","deadline_eligible_count","deadline_success_count"):
        _count(getattr(s,name),name,n)
    if s.completed_count+s.unsuccessful_count>s.arrived_count or s.deadline_eligible_count>s.arrived_count or s.deadline_success_count>min(s.completed_count,s.deadline_eligible_count):
        raise ValueError("inconsistent cumulative reward outcomes")
    if bool(s.completed_count)!=bool(s.successful_makespan_ms):
        raise ValueError("positive makespan and observed completions must agree")
    if s.unfinished_count and s.time_ms!=h:
        raise ValueError("unfinished is classified only at the observation cutoff")


def reward_delta(contract,before,after,*,terminated=False):
    """Actual cumulative differences; gamma=1 makes interval partitions telescope."""
    c=contract.context
    if type(terminated) is not bool:
        raise ValueError("termination flag must be boolean")
    _validate_reward_snapshot(c,before)
    _validate_reward_snapshot(c,after)
    if before.time_ms>=c["horizon_ms"] or terminated!=(after.time_ms==c["horizon_ms"]):
        raise ValueError("exactly the finite-horizon end must terminate")
    for name,value in asdict(before).items():
        if getattr(after,name)<value:
            raise ValueError("cumulative reward quantities cannot decrease")
    if after.time_ms==before.time_ms and after.total_energy_j!=before.total_energy_j:
        raise ValueError("actual integrated energy cannot increase in zero elapsed time")
    w,s=c["objective"]["weights"],c["objective"]["scales"]
    time_cost=after.successful_makespan_ms-before.successful_makespan_ms
    penalty=0.0
    if terminated:
        if after.arrived_count!=c["planned_task_count"] or after.completed_count+after.unsuccessful_count!=after.arrived_count:
            raise ValueError("terminal snapshot must account for the whole declared workload")
        if not after.completed_count:
            time_cost+=c["horizon_ms"] # Explicit training surrogate, never fake a batch makespan.
        completion=after.completed_count/after.arrived_count
        deadline=after.deadline_success_count/after.deadline_eligible_count if after.deadline_eligible_count else None
        penalty=c["completion_penalty"]*max(0,c["objective"]["minimum_completion_rate"]-completion)
        if deadline is not None:
            penalty+=c["deadline_penalty"]*max(0,c["objective"]["minimum_deadline_success_rate"]-deadline)
    result=RewardDelta(contract.sha256,before.time_ms,after.time_ms,
        -w["energy"]*(after.total_energy_j-before.total_energy_j)/s["energy_j"],
        -w["time"]*time_cost/s["time_ms"],
        -w["failure"]*(after.unsuccessful_count-before.unsuccessful_count)/c["planned_task_count"],
        -penalty,terminated,0.0 if terminated else 1.0)
    if not math.isfinite(result.total):
        raise ValueError("reward arithmetic must remain finite")
    return result


def terminal_reward_snapshot(contract,report):
    """Offline accounting bridge only; never feed a completed report into observations."""
    if report.get("kind")!="host_network_objective_simulation":
        raise ValueError("enhanced objective report required")
    from drlmots.run_config import parse_run_config
    # Reconstruct and compare the full public context, excluding only seed-only workload fields.
    model=ObjectiveModel(report["model_configuration"])
    run_manifest=dict(report["run_configuration"])
    run_manifest.pop("sha256",None)
    run=parse_run_config(run_manifest)
    c=contract.context
    expected=EnhancedRLContract.from_configs(run,model,planned_task_count=c["planned_task_count"],
        completion_penalty=c["completion_penalty"],deadline_penalty=c["deadline_penalty"])
    if expected.sha256!=contract.sha256:
        raise ValueError("report and reward contract contexts differ")
    m=report["metrics"]
    snapshot=RewardSnapshot(c["horizon_ms"],m["total_energy_j"],m["submitted_count"],m["completed_count"],
        m["rejected_count"],m["execution_failure_count"],m["deadline_missed_count"],m["unfinished_count"],
        m["deadline_eligible_count"],m["deadline_success_count"],m["successful_makespan_ms"] or 0.0)
    _validate_reward_snapshot(c,snapshot)
    return snapshot
