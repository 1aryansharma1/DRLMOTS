"""Versioned live state, normalization context, legal masks and causal reward."""

from dataclasses import dataclass
import math

from drlmots.control import DecisionFrame
from drlmots.entities import ResourceTier, Task
from drlmots.normalization import NormalizationContext, _digest
from drlmots.reward import RewardConfig
from drlmots.objectives import MultiObjectiveConfig, frame_objectives
from drlmots.state import StateSchema


VARIANTS = ("standard", "fog_preferred")


@dataclass(frozen=True)
class RLStateSchema(StateSchema):
    queue_capacities: tuple[int, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        capacities = tuple(self.queue_capacities)
        if len(capacities) != len(self.vm_ids) or any(type(n) is not int or n <= 0 for n in capacities):
            raise ValueError("RL schema requires one positive queue capacity per VM")
        object.__setattr__(self, "queue_capacities", capacities)

    @classmethod
    def from_config(cls, config):
        base = StateSchema.from_topology(config.topology, config.resource_units)
        ordered = sorted(config.topology.vms, key=lambda vm: vm.vm_id)
        return cls(base.vm_ids, base.resource_units, tuple(vm.queue_capacity for vm in ordered))

    @property
    def features(self):
        fields = list(super().features)
        fields.append(("episode.arrivals_remaining", "tasks"))
        for index, capacity in enumerate(self.queue_capacities):
            prefix = f"vm[{index}]"
            fields.extend(((prefix + ".reachable", "binary"), (prefix + ".input_upload", "ms"),
                           (prefix + ".queue_tail_remaining", "ms")))
            for slot in range(capacity):
                head = f"{prefix}.waiting[{slot}]"
                fields.extend(((head + ".present", "binary"), (head + ".upload_remaining", "ms"),
                               (head + ".service", "ms")))
                fields.extend((head + ".demand." + name, unit) for name, unit in self.resource_units)
        return tuple(fields)

    @property
    def dimension(self):
        return len(self.features)

    @property
    def extra_binary_indices(self):
        binary, _ = self._extra_layout()
        return binary

    @property
    def extra_guards(self):
        _, guards = self._extra_layout()
        return guards

    def _extra_layout(self):
        index = super().dimension + 1
        binary, guards = set(), {}
        for capacity in self.queue_capacities:
            binary.add(index)
            guards[index + 1] = index  # unreachable link has a zero upload placeholder
            index += 3
            for _ in range(capacity):
                binary.add(index)
                for offset in range(1, 3 + len(self.resource_units)):
                    guards[index + offset] = index
                index += 3 + len(self.resource_units)
        return frozenset(binary), guards

    def manifest(self):
        payload = {"schema_version": 2, "kind": "live_scheduler_observation",
                   "vm_action_ids": list(self.vm_ids), "resource_units": dict(self.resource_units),
                   "queue_capacities": list(self.queue_capacities), "dimension": self.dimension,
                   "features": [{"name": name, "unit": unit} for name, unit in self.features],
                   "boundary": "after same-time completions/ready starts, before live task placement",
                   "queue_order": "FCFS; readiness/service/demands per occupied waiting slot",
                   "horizon": "known remaining task budget; future task values are hidden"}
        return {**payload, "sha256": _digest(payload)}


def learning_context(config, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown RL variant")
    base = NormalizationContext.from_topology(config.topology, config.resource_units)
    semantics = {"topology": base.manifest(), "contract_version": 2, "variant": variant,
                 "placement": config.placement.manifest() if config.placement else None,
                 "service_model": "length_bytes * 2 / 1000 ms", "terminal": "last arrival"}
    return NormalizationContext(RLStateSchema.from_config(config), _digest(semantics))


def encode_frame(schema, frame):
    if not isinstance(schema, RLStateSchema) or not isinstance(frame, DecisionFrame):
        raise TypeError("live encoding requires RLStateSchema and DecisionFrame")
    if tuple(vm.vm_id for vm in frame.vms) != schema.vm_ids or frame.resource_units != schema.resource_units:
        raise ValueError("live frame schema/order/units mismatch")
    if type(frame.arrivals_remaining) is not int or frame.arrivals_remaining < 0:
        raise ValueError("invalid known task budget")
    values = [frame.time_ms, frame.length_bytes, frame.data_size_kb, *frame.demands]
    for vm, capacity in zip(frame.vms, schema.queue_capacities):
        if vm.waiting_capacity != capacity or len(vm.queued) > capacity:
            raise ValueError("live queue shape/capacity mismatch")
        values.extend((int(vm.tier is ResourceTier.CLOUD), int(vm.running_remaining_ms > 0),
                       vm.running_remaining_ms, len(vm.queued), capacity,
                       math.fsum(item.service_ms for item in vm.queued)))
        for total, available in zip(vm.capacities, vm.available):
            values.extend((int(total is not None), total or 0, int(available is not None), available or 0))
    values.append(frame.arrivals_remaining)
    for vm, capacity in zip(frame.vms, schema.queue_capacities):
        tail = vm.running_remaining_ms
        for item in vm.queued:
            tail = math.fsum((max(tail, item.upload_remaining_ms), item.service_ms))
        values.extend((int(vm.reachable), vm.upload_ms, tail))
        for slot in range(capacity):
            if slot < len(vm.queued):
                item = vm.queued[slot]
                values.extend((1, item.upload_remaining_ms, item.service_ms, *item.demands))
            else:
                values.extend((0,) * (3 + len(schema.resource_units)))
    if len(values) != schema.dimension:
        raise ValueError("live feature dimensions mismatch")
    for value in values:
        Task._validate_number("live state value", value, allow_zero=True)
    return tuple(float(value) for value in values)


def variant_mask(frame, variant):
    if variant not in VARIANTS:
        raise ValueError("unknown RL variant")
    if variant == "standard":
        return frame.mask
    fog = [vm.estimate.predicted_finish_time_ms for vm in frame.vms
           if vm.estimate and vm.tier is ResourceTier.FOG]
    cloud = [vm.estimate.predicted_finish_time_ms for vm in frame.vms
             if vm.estimate and vm.tier is ResourceTier.CLOUD]
    if not fog or not cloud:
        return frame.mask
    prefer_fog = min(fog) <= min(cloud)
    return tuple(bool(vm.estimate) and (
        vm.tier is ResourceTier.FOG and vm.estimate.predicted_finish_time_ms <= min(cloud)
        if prefer_fog else vm.tier is ResourceTier.CLOUD and vm.estimate.predicted_finish_time_ms < min(fog)
    ) for vm in frame.vms)


@dataclass(frozen=True)
class RLRewardConfig(RewardConfig):
    multi_objective: MultiObjectiveConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.multi_objective is not None and not isinstance(self.multi_objective, MultiObjectiveConfig):
            raise TypeError("multi_objective must be MultiObjectiveConfig")

    def manifest(self):
        if self.multi_objective is not None:
            payload = {"schema_version": 3, "kind": "live_scheduler_reward",
                       "algorithm": "weighted_latency_energy_risk_v1",
                       "latency_scale_ms": self.latency_scale_ms, "rejection_penalty": self.rejection_penalty,
                       "multi_objective": self.multi_objective.manifest(),
                       "timing": "causal prediction at arrival; no drain bonus", "discount_unit": "arrivals",
                       "risk_scope": "analytical constant destination hazard over arrival-to-finish; no injected faults",
                       "energy_scope": "active compute plus input transmission; no idle energy"}
            return {**payload, "sha256": _digest(payload)}
        payload = {"schema_version": 2, "kind": "live_scheduler_reward",
                   "algorithm": "causal_end_to_end_v2", "latency_scale_ms": self.latency_scale_ms,
                   "rejection_penalty": self.rejection_penalty,
                   "timing": "immediate; no drain/completion bonus", "discount_unit": "arrivals"}
        return {**payload, "sha256": _digest(payload)}


def frame_reward(frame, action, config):
    if action is None:
        if any(frame.mask):
            raise ValueError("only forced rejection can use a null action")
        return -float(config.rejection_penalty)
    if type(action) is not int or not 0 <= action < len(frame.vms) or not frame.mask[action]:
        raise ValueError("cannot reward an illegal live action")
    if config.multi_objective is not None:
        return frame_objectives(frame, action, config.multi_objective, config.latency_scale_ms)["utility"]
    latency = frame.vms[action].estimate.completion_duration_ms
    Task._validate_number("causal completion duration", latency, allow_zero=False)
    scale = config.latency_scale_ms
    if latency >= scale:
        ratio = scale / latency
        return ratio / (1 + ratio)
    return 1 / (1 + latency / scale)
