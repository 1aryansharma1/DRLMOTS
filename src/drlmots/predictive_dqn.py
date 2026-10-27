"""Versioned causal predictive inputs for the existing basic masked DQN.

The simulator still supplies contract-4 observations/rewards. A new, checkpoint-bound
feature schema augments those observations inside the network, without access to
physical state, task futures, outage traces or the native heuristic's estimates.
"""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

from drlmots.enhanced_dqn import (EnhancedDQN, EnhancedDQNConfig, EnhancedExperience,
    _observation, _pack, _tuples, _unpack, validate_experience)
from drlmots.enhanced_rl_contract import RewardDelta
from drlmots.normalization import _digest
from drlmots.objective_model import integer, number


PREDICTIONS = (
    ("upload_ms", "ms", "decision_time_ms"),
    ("queue_wait_ms", "ms", "decision_time_ms"),
    ("compute_ms", "ms", "decision_time_ms"),
    ("download_ms", "ms", "decision_time_ms"),
    ("latency_ms", "ms", "decision_time_ms"),
    ("finish_ms", "ms", "horizon_ms"),
    ("incremental_energy_j", "J", "decision_energy_j"),
    ("deadline_slack_ms", "ms", "horizon_ms"),
    ("deadline_lateness_ms", "ms", "horizon_ms"),
    ("deadline_met", "binary", None),
    ("failure_probability", "probability", None),
    ("estimate_available", "binary", None),
)


def feature_manifest(contract):
    c = contract.context
    features = [dict(name=f"prediction[{v}].{name}", unit=unit,
                     scale=c["horizon_ms"] if scale == "horizon_ms" else c["objective"]["scales"][scale] if scale else None)
                for v in contract.vm_ids for name, unit, scale in PREDICTIONS]
    payload = dict(schema_version=1, kind="observed_predictive_dqn_features",
        base_contract_sha256=contract.sha256, base_feature_count=len(c["features"]),
        network_feature_count=len(c["features"]) + len(features), features=features,
        normalization="nonnegative x/(x+public_scale); binary/probability unchanged",
        arithmetic="CPU float32; reject saturated normalized nonbinary inputs",
        estimates="sample-and-hold reports; upload waits for reported half-duplex bytes; FCFS queue tail; own output wire time; no future arrivals/outages",
        energy="reported other-VM host load plus candidate VM; dynamic host delta and own input/output bytes; shared idle excluded from prediction only",
        risk="configured conditional survival using observed uptime age; unknown age uses nominal zero age with availability flag false",
        reward="unchanged contract-4 actual system costs; no prediction enters rewards or masks")
    return {**payload, "sha256": _digest(payload)}


class PredictiveQNetwork(nn.Module):
    def __init__(self, contract, settings):
        super().__init__()
        self.context_sha256 = contract.sha256
        self.spec = feature_manifest(contract)
        c = contract.context
        self.index = {f["name"]: i for i, f in enumerate(c["features"])}
        self.vm_ids = contract.vm_ids
        self.vms = c["vms"]
        self.queue_slots = max(v["queue_capacity"] for v in self.vms.values())
        self.base_count = len(c["features"])
        self.horizon = c["horizon_ms"]
        # Persistent public constants are included in checkpoint/frozen fingerprints.
        def buffer(name, values):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))
        buffer("raw_scales", [f["scale"] if f["scale"] is not None else 1 for f in c["features"]])
        buffer("binary", [f["scale"] is None for f in c["features"]])
        buffer("mips", [c["vms"][v]["mips"] for v in self.vm_ids])
        buffer("capacity", [c["hosts"][self.vms[v]["host_id"]]["capacity_mips"] for v in self.vm_ids])
        buffer("bandwidth", [c["links"][self.vms[v]["link_id"]]["bandwidth_mbps"] for v in self.vm_ids])
        buffer("propagation", [c["links"][self.vms[v]["link_id"]]["propagation_ms"] for v in self.vm_ids])
        buffer("byte_energy", [c["links"][self.vms[v]["link_id"]]["joules_per_kb"] for v in self.vm_ids])
        curves = [c["hosts"][self.vms[v]["host_id"]]["power_curve"] for v in self.vm_ids]
        width = max(len(curve) - 1 for curve in curves)
        buffer("idle_power", [curve[0][1] for curve in curves])
        buffer("power_lower", [[curve[i][0] if i < len(curve)-1 else 0 for i in range(width)] for curve in curves])
        buffer("power_span", [[curve[i+1][0]-curve[i][0] if i < len(curve)-1 else 1 for i in range(width)] for curve in curves])
        buffer("power_delta", [[curve[i+1][1]-curve[i][1] if i < len(curve)-1 else 0 for i in range(width)] for curve in curves])
        failures = [c["failures"][self.vms[v]["node_id"]] for v in self.vm_ids]
        buffer("failure_scale", [f["scale_ms"] for f in failures])
        buffer("failure_shape", [f["shape"] if f["distribution"] == "weibull" else 1 for f in failures])
        buffer("failure_enabled", [f["distribution"] != "none" for f in failures])
        buffer("prediction_scales", [f["scale"] if f["scale"] is not None else 1 for f in self.spec["features"]])
        buffer("prediction_identity", [f["scale"] is None for f in self.spec["features"]])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(settings.seed)
            a, b = settings.hidden_widths
            self.layers = nn.Sequential(nn.Linear(self.spec["network_feature_count"], a, device="cpu", dtype=torch.float32), nn.ReLU(),
                                        nn.Linear(a, b, device="cpu", dtype=torch.float32), nn.ReLU(),
                                        nn.Linear(b, len(self.vm_ids), device="cpu", dtype=torch.float32))

    def _get(self, raw, names):
        return raw[..., [self.index[n] for n in names]]

    def _vm(self, raw, suffix):
        return self._get(raw, [f"vm[{v}].{suffix}" for v in self.vm_ids])

    def _domain(self, raw, domain, field, name):
        return self._get(raw, [f"{domain}[{self.vms[v][field]}].{name}" for v in self.vm_ids])

    def _power(self, utilization):
        fraction = ((utilization[..., None] - self.power_lower) / self.power_span).clamp(0, 1)
        return self.idle_power + (fraction * self.power_delta).sum(dim=-1)

    @torch.no_grad()
    def predictions(self, values):
        if values.shape[-1] != self.base_count or not torch.isfinite(values).all():
            raise ValueError("finite base contract-4 vector required")
        denominator = torch.where(self.binary.bool(), torch.ones_like(values), 1-values)
        if (denominator <= 0).any() or (values < 0).any() or (values > 1).any():
            raise ValueError("normalized telemetry saturates float32 or lies outside its range")
        raw = values * self.raw_scales / denominator
        task = lambda key: self._get(raw, [f"task.{key}"])
        remaining = self._get(raw, ["episode.remaining_ms"])
        now = self.horizon - remaining
        backlog = (self._domain(raw, "link", "link_id", "remaining_upload_kb")
                   + self._domain(raw, "link", "link_id", "remaining_download_kb"))
        drain = 8 * backlog / self.bandwidth
        upload = drain + 8 * task("input_kb") / self.bandwidth + self.propagation
        download = 8 * task("output_kb") / self.bandwidth + self.propagation
        compute = 1000 * task("compute_mi") / self.mips
        tail = self._vm(raw, "running_remaining_ms")
        for slot in range(self.queue_slots):
            def q(key):
                names = [f"vm[{v}].waiting[{slot}].{key}" if slot < self.vms[v]["queue_capacity"] else None for v in self.vm_ids]
                return torch.stack([raw[..., self.index[n]] if n else torch.zeros_like(raw[..., 0]) for n in names], dim=-1)
            ready = q("upload") * (drain + self.propagation) + q("propagation") * q("remaining_propagation_ms")
            tail = torch.where(q("present").bool(), torch.maximum(tail, ready) + 1000*q("compute_mi")/self.mips, tail)
        wait = (tail-upload).clamp_min(0)
        latency = torch.maximum(tail, upload) + compute + download
        finish = now + latency
        utilization = self._domain(raw, "host", "host_id", "observed_utilization")
        other_load = (utilization - self._vm(raw, "running") * self.mips / self.capacity).clamp(0, 1)
        new_load = (other_load + self.mips / self.capacity).clamp(0, 1)
        energy = (self._power(new_load)-self._power(other_load))*compute/1000 + (task("input_kb")+task("output_kb"))*self.byte_energy
        deadline_left = task("deadline_remaining_ms")
        has_deadline = task("has_deadline").bool()
        slack = torch.where(has_deadline, (deadline_left-latency).clamp_min(0), torch.zeros_like(latency))
        lateness = torch.where(has_deadline, (latency-deadline_left).clamp_min(0), torch.zeros_like(latency))
        deadline_met = (~has_deadline | ((latency <= deadline_left) & ~task("deadline_expired").bool())).float()
        age = self._domain(raw, "node", "node_id", "observed_up_age_ms")
        hazard = ((age+latency)/self.failure_scale).pow(self.failure_shape) - (age/self.failure_scale).pow(self.failure_shape)
        risk = -torch.expm1(-hazard.clamp_min(0)) * self.failure_enabled
        available = (self._domain(raw, "node", "node_id", "health_known").bool()
                     & self._domain(raw, "node", "node_id", "observed_up").bool()
                     & self._domain(raw, "node", "node_id", "age_known").bool()
                     & self._domain(raw, "host", "host_id", "utilization_known").bool()).float()
        predicted = torch.stack((upload, wait, compute, download, latency, finish, energy,
                                 slack, lateness, deadline_met, risk, available), dim=-1)
        if not torch.isfinite(predicted).all() or (predicted < 0).any():
            raise ValueError("nonfinite/negative predictive estimate")
        return predicted.flatten(start_dim=-2)

    def augmented(self, values):
        predicted = self.predictions(values)
        normalized = torch.where(self.prediction_identity.bool(), predicted,
                                 predicted/(predicted+self.prediction_scales))
        return torch.cat((values, normalized), dim=-1)

    def forward(self, values):
        return self.layers(self.augmented(values))


class PredictiveDQN(EnhancedDQN):
    """Same basic DQN/replay/controller/reward; new causal feature encoder/network."""
    def __init__(self, contract, settings=None, *, host_nodes=None):
        super().__init__(contract, settings, host_nodes=host_nodes)
        self.features = feature_manifest(contract)
        self.environment_sha256 = _digest(dict(base_environment_sha256=self.environment_sha256,
                                               predictive_features_sha256=self.features["sha256"]))
        self.online = PredictiveQNetwork(contract, self.settings)
        self.target = deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.settings.learning_rate)

    def artifact(self):
        data = super().artifact()
        data.pop("sha256")
        data.update(kind="predictive_masked_dqn_checkpoint", predictive_features=self.features)
        return {**data, "sha256": _digest(data)}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.artifact(), allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, expected_contract, *, host_nodes=None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        checksum = data.pop("sha256", None)
        keys = set("schema_version kind contract environment_sha256 host_nodes settings online target optimizer replay selector_rng replay_rng decisions updates episodes losses predictive_features".split())
        if set(data) != keys or data.get("kind") != "predictive_masked_dqn_checkpoint" or type(data.get("schema_version")) is not int or data["schema_version"] != 1 or _digest(data) != checksum:
            raise ValueError("invalid predictive checkpoint kind/version/checksum")
        if data["contract"] != expected_contract.manifest() or data["predictive_features"] != feature_manifest(expected_contract):
            raise ValueError("predictive checkpoint observation/feature context mismatch")
        obj = cls(expected_contract, EnhancedDQNConfig.parse(data["settings"]), host_nodes=host_nodes)
        if data["host_nodes"] != obj.host_nodes or data["environment_sha256"] != obj.environment_sha256:
            raise ValueError("predictive checkpoint environment mismatch")
        for name in ("online", "target"):
            state, expected = _unpack(data[name]), getattr(obj, name).state_dict()
            if set(state) != set(expected) or any(state[k].shape != expected[k].shape or not torch.isfinite(state[k]).all() for k in state):
                raise ValueError("invalid predictive network state")
            if any(not torch.equal(state[k], expected[k]) for k in state if not k.startswith("layers.")):
                raise ValueError("predictive public constants differ from context")
            getattr(obj, name).load_state_dict(state, strict=True)
        optimizer = _unpack(data["optimizer"])
        if set(optimizer) != {"state", "param_groups"} or _pack(optimizer["param_groups"]) != _pack(obj.optimizer.state_dict()["param_groups"]):
            raise ValueError("predictive optimizer settings mismatch")
        obj.optimizer.load_state_dict(optimizer)
        for name in ("decisions", "updates", "episodes"):
            integer(data[name], name)
            setattr(obj, name, data[name])
        ids = optimizer["param_groups"][0]["params"]
        if (not obj.updates and optimizer["state"]) or (obj.updates and set(optimizer["state"]) != set(ids)):
            raise ValueError("predictive optimizer state/counters disagree")
        for key, param in zip(ids, obj.online.parameters()):
            if not obj.updates:
                break
            state = optimizer["state"][key]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"} or state["step"].numel() != 1 or float(state["step"]) != obj.updates or any(state[k].shape != param.shape or not torch.isfinite(state[k]).all() for k in ("exp_avg", "exp_avg_sq")) or (state["exp_avg_sq"] < 0).any():
                raise ValueError("invalid predictive Adam state")
        if len(data["replay"]) > obj.settings.replay_capacity or len(data["losses"]) != obj.updates:
            raise ValueError("invalid predictive replay/loss counters")
        for loss in data["losses"]:
            number("predictive loss", loss)
        obj.losses = list(data["losses"])
        for raw in data["replay"]:
            item = EnhancedExperience(_observation(raw["state"]), raw["action"], RewardDelta(**raw["feedback"]),
                                      None if raw["next_state"] is None else _observation(raw["next_state"]))
            validate_experience(expected_contract, item)
            obj.replay.append(item)
        obj.selector_rng.setstate(_tuples(data["selector_rng"]))
        obj.replay_rng.setstate(_tuples(data["replay_rng"]))
        if obj.fingerprint != checksum:
            raise ValueError("predictive checkpoint round-trip differs")
        return obj
