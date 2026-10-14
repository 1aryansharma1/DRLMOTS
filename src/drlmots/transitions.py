"""Rewarded arrival transitions from baseline traces, without agent/replay control."""

from dataclasses import dataclass
import math

from drlmots.entities import Task
from drlmots.actions import action_mask
from drlmots.state import StateSchema, encode_state


@dataclass(frozen=True)
class ArrivalTransition:
    task_id: str
    time_ms: float
    state: tuple[float, ...]
    action_mask: tuple[bool, ...]
    action: int | None
    vm_id: str | None
    reward: float
    forced_rejection: bool
    next_state: tuple[float, ...] | None
    next_action_mask: tuple[bool, ...] | None
    next_time_ms: float | None
    done: bool


def _validate(decisions):
    ids = set()
    shape = None
    identity = None
    last_time = 0
    for record in decisions:
        if record.reward is None:
            raise ValueError("transition/reward export requires scored decisions")
        reward = record.reward
        if (reward.task_id != record.observation.task_id or reward.available_time_ms != record.observation.time_ms
                or reward.selected_action != record.selected_action or reward.selected_vm_id != record.selected_vm_id):
            raise ValueError("reward and decision identity/timing disagree")
        if isinstance(reward.value, bool) or not isinstance(reward.value, (int, float)):
            raise TypeError("reward value must be a number")
        Task._validate_number("reward magnitude", abs(reward.value), allow_zero=True)
        schema = StateSchema(tuple(vm.vm_id for vm in record.observation.vms), record.observation.resource_units)
        current_identity = (schema.vm_ids, schema.resource_units)
        if identity is not None and identity != current_identity:
            raise ValueError("transition VM/resource schema changed")
        identity = current_identity
        if record.state != encode_state(schema, record.observation) or record.action_mask != action_mask(record.observation):
            raise ValueError("transition state/mask disagrees with its observation")
        if reward.task_id in ids or record.observation.time_ms < last_time:
            raise ValueError("decisions must have unique tasks and chronological order")
        ids.add(reward.task_id)
        last_time = record.observation.time_ms
        current_shape = (len(record.state), len(record.action_mask))
        if shape is not None and shape != current_shape:
            raise ValueError("transition state/action dimensions changed")
        shape = current_shape
        if record.selected_action is None:
            if any(record.action_mask) or record.selected_vm_id is not None or reward.value >= 0:
                raise ValueError("forced rejection requires an all-false mask and negative reward")
        elif (type(record.selected_action) is not int or not 0 <= record.selected_action < len(record.action_mask)
              or not record.action_mask[record.selected_action] or record.selected_vm_id is None
              or record.selected_vm_id != record.observation.vms[record.selected_action].vm_id
              or not 0 <= reward.value <= 1):
            raise ValueError("transition action/reward is invalid")


def build_transitions(decisions):
    decisions = tuple(decisions)
    _validate(decisions)
    records = []
    for index, current in enumerate(decisions):
        following = decisions[index + 1] if index + 1 < len(decisions) else None
        records.append(ArrivalTransition(
            current.observation.task_id, current.observation.time_ms, current.state, current.action_mask,
            current.selected_action, current.selected_vm_id, current.reward.value, current.selected_action is None,
            following.state if following is not None else None,
            following.action_mask if following is not None else None,
            following.observation.time_ms if following is not None else None, following is None,
        ))
    return tuple(records)


def summarize_rewards(decisions):
    decisions = tuple(decisions)
    _validate(decisions)
    try:
        total = math.fsum(record.reward.value for record in decisions)
    except OverflowError as error:
        raise ValueError("reward return must be representable and finite") from error
    admitted = sum(record.selected_action is not None for record in decisions)
    return {"arrival_count": len(decisions), "admitted_count": admitted,
            "rejected_count": len(decisions) - admitted, "return": total,
            "average_reward": total / len(decisions) if decisions else None,
            "discounting": "none; sum over every arrival", "unit": "dimensionless"}


@dataclass(frozen=True)
class LearningTransition:
    task_id: str
    state: tuple[float, ...]
    action_mask: tuple[bool, ...]
    action: int
    reward: float
    discount_steps: int
    bootstrap_discount: float
    next_state: tuple[float, ...] | None
    next_action_mask: tuple[bool, ...] | None
    done: bool


def build_learning_transitions(decisions, *, gamma):
    """Absorb following forced costs; never bootstrap an all-false action set.

    Gamma discounts ARRIVALS, not elapsed milliseconds. Forced prefix costs remain
    in episode reporting; they have no preceding agent action to update.
    """
    Task._validate_number("gamma", gamma, allow_zero=True)
    if gamma > 1:
        raise ValueError("gamma must be in [0,1]")
    raw = build_transitions(decisions)
    records = []
    for index, current in enumerate(raw):
        if current.forced_rejection:
            continue
        weighted = [current.reward]
        steps, following = 1, index + 1
        while following < len(raw) and raw[following].forced_rejection:
            weighted.append((gamma ** steps) * raw[following].reward)
            steps += 1
            following += 1
        try:
            total = math.fsum(weighted)
        except OverflowError as error:
            raise ValueError("learning reward must be representable and finite") from error
        next_record = raw[following] if following < len(raw) else None
        records.append(LearningTransition(
            current.task_id, current.state, current.action_mask, current.action, total, steps,
            gamma ** steps if next_record is not None else 0.0,
            next_record.state if next_record is not None else None,
            next_record.action_mask if next_record is not None else None, next_record is None,
        ))
    return tuple(records)
