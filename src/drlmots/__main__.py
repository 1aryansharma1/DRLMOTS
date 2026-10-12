"""Run v0.1's deterministic, synthetic scheduling example."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from drlmots.entities import Host, ResourceTier, VM
from drlmots.environment import simulate
from drlmots.topology import AccessPoint, CloudDatacenter, EndNode, FogNode, Topology
from drlmots.workload import fixed_workload, workload_manifest


def smoke_report() -> dict:
    topology = Topology(
        hosts=(Host("host-fog", 3, 32, 1000), Host("host-cloud", 3, 32, 1000)),
        vms=(VM("vm-fog", ResourceTier.FOG, 200, queue_capacity=1),
             VM("vm-cloud", ResourceTier.CLOUD, 200, queue_capacity=1)),
        fog_nodes=(FogNode("fog", ("host-fog",), ("vm-fog",)),),
        cloud_datacenters=(CloudDatacenter("cloud", ("host-cloud",), ("vm-cloud",)),),
        access_points=(AccessPoint("ap", "end"),),
        end_nodes=(EndNode("end", ("fog",)),),
    )
    tasks = fixed_workload([
        {"task_id": "task-1", "arrival_time": 0, "length": 1000, "data_size_kb": 20},
        {"task_id": "task-2", "arrival_time": 0, "length": 2000, "data_size_kb": 20},
        {"task_id": "task-3", "arrival_time": 1, "length": 500, "data_size_kb": 20},
    ], length_unit="bytes")
    result = simulate(topology, tasks, task_length_unit="bytes")
    return {
        "version": "0.1.0",
        "implementation_scope": "deterministic software baseline; not paper reproduction",
        "policy": "least_queue_fcfs",
        "tie_breaking": "VM ID; simultaneous arrivals retain input order; completions first",
        "service_time_model": "explicit byte-valued fixture: length * 2 / 1000 milliseconds",
        "queue_capacity_semantics": "waiting tasks only; one additional running task per VM",
        "overflow_policy": "reject when all configured VM queues are full",
        "network_delay_ms": 0,
        "resource_feasibility": "not modelled; nonempty task resource requirements rejected",
        "workload": workload_manifest(tasks, length_unit="bytes"),
        "topology": asdict(topology),
        "completed_count": result.completed_count,
        "failed_count": result.failed_count,
        "makespan_ms": result.makespan_ms,
        "average_completion_time_ms": result.average_completion_time_ms,
        "outcomes": [asdict(item) for item in result.outcomes],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic DRLMOTS v0.1 example")
    parser.add_argument("--output", type=Path, help="Also save the exact inputs and results as JSON")
    args = parser.parse_args()
    report = smoke_report()
    content = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
