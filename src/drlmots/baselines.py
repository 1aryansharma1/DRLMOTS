"""Synthetic reference policies, separate from a learned or paper scheduler."""


BASELINE_POLICIES = ("least_queue_fcfs", "first_fit_fcfs")
FOG_FIRST_POLICY = "fog_first_completion"
POLICIES = (*BASELINE_POLICIES, FOG_FIRST_POLICY)
POLICY_LABELS = {"least_queue_fcfs": "Least queue", "first_fit_fcfs": "First fit",
                 FOG_FIRST_POLICY: "Fog-first completion"}


def choose_vm(policy, eligible, loads):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose one of {POLICIES}")
    if not eligible:
        raise ValueError("cannot select a VM from an empty eligible set")
    if policy == FOG_FIRST_POLICY:
        raise ValueError("fog-first completion requires network timing estimates")
    if policy == "first_fit_fcfs":
        return min(eligible, key=lambda vm: vm.vm_id)
    return min(eligible, key=lambda vm: (loads[vm.vm_id], vm.vm_id))
