"""Paired synthetic comparisons with fixed inputs and explicit rejection metrics."""

from drlmots.baselines import POLICIES
from drlmots.__main__ import configured_report
from drlmots.run_config import RunConfig


def compare_policies(config: RunConfig, policies=POLICIES) -> dict:
    if not policies or len(policies) != len(set(policies)):
        raise ValueError("comparison policies must be nonempty and unique")
    reports = [configured_report(config, policy=policy) for policy in policies]
    if len({report["workload"]["sha256"] for report in reports}) != 1:
        raise RuntimeError("comparison workloads do not match")
    if len({report["run_configuration"]["sha256"] for report in reports}) != 1:
        raise RuntimeError("comparison configurations do not match")
    for report in reports:
        count = len(report["workload"]["tasks"])
        if report["completed_count"] + report["failed_count"] != count:
            raise RuntimeError("comparison did not account for every input task")
        if any(report["available_resources"][vm.vm_id] != vm.resource_capacities for vm in config.topology.vms):
            raise RuntimeError("comparison ended with a resource leak")
        tasks = {task["task_id"]: task for task in report["workload"]["tasks"]}
        per_vm = []
        for vm in sorted(config.topology.vms, key=lambda item: item.vm_id):
            completed = [item for item in report["outcomes"] if item["vm_id"] == vm.vm_id]
            busy = sum(item["finish_time_ms"] - item["start_time_ms"] for item in completed)
            per_vm.append({
                "vm_id": vm.vm_id, "tier": vm.tier.value, "busy_ms": busy,
                "occupancy_pct": 100 * busy / report["makespan_ms"] if report["makespan_ms"] else 0,
                "completed_count": len(completed),
            })
        report["vm_usage"] = per_vm
        completed = [item for item in report["outcomes"] if item["finish_time_ms"] is not None]
        report["average_waiting_time_ms"] = (
            sum(item["start_time_ms"] - tasks[item["task_id"]]["arrival_time"] for item in completed) / len(completed)
            if completed else None
        )
    return {
        "kind": "paired_synthetic_baselines", "policies": list(policies),
        "workload_sha256": reports[0]["workload"]["sha256"],
        "configuration_sha256": reports[0]["run_configuration"]["sha256"],
        "timing_scope": "successful tasks only; compare completion/failure counts alongside timings",
        "reports": reports,
    }
