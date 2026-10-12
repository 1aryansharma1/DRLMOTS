from dataclasses import replace

import pytest

from drlmots.entities import ResourceTier, Task, VM
from drlmots.resources import ResourceLedger, fits_resources, validate_resource_units


def vm(capacities=None):
    return VM("vm", ResourceTier.FOG, 200, queue_capacity=1, resource_capacities=capacities)


def task(task_id="task", **requirements):
    return Task(task_id, 0, 1000, 20, required_resources=requirements)


@pytest.mark.parametrize("amount", [0, -1, True, None, "2", float("nan"), float("inf"), 10**400])
def test_invalid_capacities_rejected(amount):
    with pytest.raises((TypeError, ValueError), match="resource_capacities"):
        vm({"cpu": amount})


@pytest.mark.parametrize("capacities", [[], "cpu", 2, True])
def test_capacities_require_a_mapping(capacities):
    with pytest.raises(TypeError, match="resource_capacities"):
        vm(capacities)


def test_capacity_mapping_copies_inputs_and_unknown_stays_unknown():
    amounts = {"cpu": 2}
    resource = vm(amounts)
    amounts["cpu"] = 10
    assert resource.resource_capacities == {"cpu": 2}
    assert vm().resource_capacities is None


@pytest.mark.parametrize("units", [None, [], {"cpu": None}, {"cpu": 2}, {"cpu": " "}, {"": "units"}])
def test_invalid_unit_catalogues(units):
    with pytest.raises((TypeError, ValueError), match="resource_units"):
        validate_resource_units(units)


def test_componentwise_fit_and_no_implicit_unit_conversion():
    resource = vm({"cpu": 2, "memory_gb": 4})
    assert fits_resources(task(cpu=2, memory_gb=4), resource)
    assert not fits_resources(task(cpu=3), resource)
    assert not fits_resources(task(memory_gb=5), resource)
    assert not fits_resources(task(memory_mb=1), resource)
    assert not fits_resources(task(cpu=1), vm())
    assert fits_resources(task(), vm())


def test_ledger_exact_reservation_and_release_restores_every_balance():
    ledger = ResourceLedger([vm({"cpu": 2, "memory_gb": 4})])
    work = task(cpu=2, memory_gb=4)
    ledger.reserve("vm", work, 0)
    assert ledger.available("vm") == {"cpu": 0, "memory_gb": 0}
    with pytest.raises(RuntimeError, match="unreleased"):
        ledger.final_balances()
    ledger.release("vm", work, 2)
    assert ledger.final_balances() == {"vm": {"cpu": 2, "memory_gb": 4}}
    assert [(event.action, event.time_ms) for event in ledger.events] == [("reserve", 0), ("release", 2)]


def test_ledger_rejects_overbooking_or_wrong_release_without_changing_balance():
    ledger = ResourceLedger([vm({"cpu": 2})])
    with pytest.raises(RuntimeError, match="exceed capacity"):
        ledger.reserve("vm", task(cpu=3), 0)
    work = task(cpu=1)
    ledger.reserve("vm", work, 0)
    with pytest.raises(RuntimeError, match="already has"):
        ledger.reserve("vm", task("other", cpu=1), 0)
    with pytest.raises(RuntimeError, match="does not match"):
        ledger.release("vm", replace(work, task_id="other"), 1)
    assert ledger.available("vm") == {"cpu": 1}
    ledger.release("vm", work, 2)
    assert ledger.final_balances() == {"vm": {"cpu": 2}}
