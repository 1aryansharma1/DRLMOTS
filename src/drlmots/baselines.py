"""Simple software comparators; neither policy is a learned or paper baseline."""


POLICIES = ("least_queue_fcfs", "first_fit_fcfs")
POLICY_LABELS = {"least_queue_fcfs": "Least queue", "first_fit_fcfs": "First fit"}


def choose_vm(policy, eligible, loads):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose one of {POLICIES}")
    if not eligible:
        raise ValueError("cannot select a VM from an empty eligible set")
    if policy == "first_fit_fcfs":
        return min(eligible, key=lambda vm: vm.vm_id)
    return min(eligible, key=lambda vm: (loads[vm.vm_id], vm.vm_id))
