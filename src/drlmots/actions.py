"""Stable VM action indices and serial-resource queue masks (A-022)."""

from dataclasses import dataclass

from drlmots.state import DecisionObservation, _ids


@dataclass(frozen=True)
class ActionMap:
    vm_ids: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "vm_ids", _ids(self.vm_ids))

    def vm_id(self, index: int) -> str:
        if type(index) is not int:
            raise TypeError("action index must be an integer")
        if not 0 <= index < len(self.vm_ids):
            raise ValueError("action index out of range")
        return self.vm_ids[index]

    def index(self, vm_id: str) -> int:
        try:
            return self.vm_ids.index(vm_id)
        except ValueError:
            raise ValueError(f"unknown action VM {vm_id!r}") from None


def action_mask(observation: DecisionObservation) -> tuple[bool, ...]:
    # Eligibility uses TOTAL capacity. A queued task need not fit the live balance.
    return tuple(
        all(demand == 0 or (capacity is not None and demand <= capacity)
            for demand, capacity in zip(observation.demands, vm.capacities))
        and (not vm.running or vm.waiting_count < vm.waiting_capacity)
        for vm in observation.vms
    )
