from dataclasses import FrozenInstanceError
import math

import pytest

from drlmots.actions import ActionMap, action_mask
from drlmots.entities import ResourceTier
from drlmots.state import DecisionObservation, StateSchema, VMObservation, encode_state


UNITS = (("cpu", "abstract units"), ("memory", "GB"))


def vm(**changes):
    fields = dict(vm_id="a", tier=ResourceTier.FOG, running=True, remaining_service_ms=2,
                  waiting_count=0, waiting_capacity=1, queued_service_ms=0,
                  capacities=(2, None), available=(0, None))
    return VMObservation(**(fields | changes))


def observation(vms=None, demands=(1, 0), **changes):
    fields = dict(task_id="task", time_ms=1, length_bytes=1000, data_size_kb=20,
                  resource_units=UNITS, demands=demands, vms=(vm(),) if vms is None else vms)
    return DecisionObservation(**(fields | changes))


def test_raw_vector_hand_calculation_distinguishes_unknown_and_exhausted():
    schema = StateSchema(("a",), UNITS)
    state = encode_state(schema, observation())
    assert state == (1, 1000, 20, 1, 0, 0, 1, 2, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 0)
    assert len(state) == schema.dimension == len(schema.features) == 19
    assert all(type(value) is float and math.isfinite(value) for value in state)
    features = dict(zip((name for name, _ in schema.features), state))
    assert features["vm[0].available.cpu"] == features["vm[0].available.memory"] == 0
    assert features["vm[0].available_known.cpu"] == 1
    assert features["vm[0].available_known.memory"] == 0


@pytest.mark.parametrize("demands,waiting,expected", [
    ((1, 0), 0, True), ((2, 0), 0, True), ((3, 0), 0, False),
    ((0, 1), 0, False), ((0, 0), 0, True), ((1, 0), 1, False),
])
def test_masks_use_total_capacity_and_waiting_slots(demands, waiting, expected):
    snapshot = vm(waiting_count=waiting, queued_service_ms=2 if waiting else 0)
    assert action_mask(observation((snapshot,), demands)) == (expected,)


def test_idle_and_unknown_zero_demand_actions_and_no_vms():
    idle = vm(running=False, remaining_service_ms=0, capacities=(None, None), available=(None, None))
    assert action_mask(observation((idle,), (0, 0))) == (True,)
    assert action_mask(observation((idle,), (1, 0))) == (False,)
    assert action_mask(observation(())) == ()


def test_schema_action_order_fingerprint_and_mapping_roundtrip():
    schema = StateSchema(("z", "a"), tuple(reversed(UNITS)))
    actions = ActionMap(("z", "a"))
    assert schema.vm_ids == actions.vm_ids == ("a", "z")
    assert schema.resource_units == UNITS
    assert [actions.index(actions.vm_id(i)) for i in range(2)] == [0, 1]
    assert schema.manifest() == StateSchema(("a", "z"), UNITS).manifest()
    assert schema.manifest()["sha256"] != StateSchema(("a", "z"), (("cpu", "cores"), ("memory", "GB"))).manifest()["sha256"]
    assert StateSchema((), ()).dimension == 3


@pytest.mark.parametrize("index", [-1, 1, True, 0.0, "0", None])
def test_invalid_action_index_cannot_alias_a_vm(index):
    with pytest.raises((TypeError, ValueError)):
        ActionMap(("a",)).vm_id(index)


def test_unknown_vm_and_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="unknown"):
        ActionMap(("a",)).index("b")
    with pytest.raises(ValueError, match="duplicate"):
        ActionMap(("a", "a"))


def test_snapshots_are_detached_and_immutable():
    capacities, available, demands, units = [2, None], [0, None], [1, 0], [list(item) for item in UNITS]
    snapshot = vm(capacities=capacities, available=available)
    obs = observation([snapshot], demands, resource_units=units)
    capacities[0], available[0], demands[0], units[0][1] = 100, 100, 100, "changed"
    assert obs.demands == (1, 0) and obs.vms[0].capacities == (2, None)
    assert obs.resource_units == UNITS
    with pytest.raises(FrozenInstanceError):
        obs.time_ms = 5


@pytest.mark.parametrize("changes", [
    {"remaining_service_ms": float("nan")}, {"remaining_service_ms": 0},
    {"waiting_count": True}, {"waiting_capacity": 0}, {"waiting_count": 2},
    {"waiting_count": 1}, {"running": False}, {"available": (-1, None)},
    {"available": (3, None)}, {"available": (None, None)},
    {"capacities": (0, None)}, {"available": (0,)}, {"queued_service_ms": float("inf")},
])
def test_invalid_runtime_snapshots_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        vm(**changes)


def test_encoder_rejects_wrong_schema_even_with_same_dimensions():
    with pytest.raises(ValueError, match="VM order"):
        encode_state(StateSchema(("b",), UNITS), observation())
    with pytest.raises(ValueError, match="resource columns"):
        encode_state(StateSchema(("a",), (("cpu", "cores"), ("memory", "GB"))), observation())
    with pytest.raises(ValueError, match="resource columns"):
        encode_state(StateSchema(("a",), ()), observation())


@pytest.mark.parametrize("changes", [{"time_ms": float("inf")}, {"demands": (-1, 0)},
                                      {"demands": (float("nan"), 0)}, {"length_bytes": 0}])
def test_nonfinite_or_negative_task_state_rejected(changes):
    with pytest.raises(ValueError):
        observation(**changes)
