"""Run fixed or seeded synthetic workloads through the verified baseline."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from drlmots.entities import Host, ResourceTier, VM
from drlmots.environment import simulate
from drlmots.run_config import RunConfig, load_run_config
from drlmots.topology import AccessPoint, CloudDatacenter, EndNode, FogNode, Topology
from drlmots.workload import (
    SyntheticWorkloadSettings, fixed_workload, seeded_workload, workload_manifest,
)


def smoke_report(
    settings: SyntheticWorkloadSettings | None = None,
    *,
    topology: Topology | None = None,
) -> dict:
    if topology is None:
        topology = _smoke_topology()
    tasks = seeded_workload(settings) if settings is not None else fixed_workload([
        {"task_id": "task-1", "arrival_time": 0, "length": 1000, "data_size_kb": 20},
        {"task_id": "task-2", "arrival_time": 0, "length": 2000, "data_size_kb": 20},
        {"task_id": "task-3", "arrival_time": 1, "length": 500, "data_size_kb": 20},
    ], length_unit="bytes")
    result = simulate(topology, tasks, task_length_unit="bytes")
    return _report(topology, tasks, settings, result)


def _smoke_topology() -> Topology:
    return Topology(
        hosts=(Host("host-fog", 3, 32, 1000), Host("host-cloud", 3, 32, 1000)),
        vms=(VM("vm-fog", ResourceTier.FOG, 200, queue_capacity=1),
             VM("vm-cloud", ResourceTier.CLOUD, 200, queue_capacity=1)),
        fog_nodes=(FogNode("fog", ("host-fog",), ("vm-fog",)),),
        cloud_datacenters=(CloudDatacenter("cloud", ("host-cloud",), ("vm-cloud",)),),
        access_points=(AccessPoint("ap", "end"),),
        end_nodes=(EndNode("end", ("fog",)),),
    )


def _report(topology, tasks, settings, result) -> dict:
    return {
        "version": "0.1.0",
        "milestone": "v0.2 workload generation" if settings is not None else "v0.1 fixed smoke",
        "implementation_scope": "deterministic software baseline; not paper reproduction",
        "policy": "least_queue_fcfs",
        "tie_breaking": "VM ID; simultaneous arrivals retain input order; completions first",
        "service_time_model": "explicit byte-valued fixture: length * 2 / 1000 milliseconds",
        "queue_capacity_semantics": "waiting tasks only; one additional running task per VM",
        "overflow_policy": "reject when all configured VM queues are full",
        "network_delay_ms": 0,
        "resource_feasibility": "not modelled; nonempty task resource requirements rejected",
        "workload": workload_manifest(tasks, length_unit="bytes", generator_settings=settings),
        "topology": asdict(topology),
        "topology_counts": {name: len(getattr(topology, name)) for name in (
            "hosts", "vms", "fog_nodes", "cloud_datacenters", "access_points", "end_nodes"
        )},
        "completed_count": result.completed_count,
        "failed_count": result.failed_count,
        "makespan_ms": result.makespan_ms,
        "average_completion_time_ms": result.average_completion_time_ms,
        "outcomes": [asdict(item) for item in result.outcomes],
    }


def configured_report(config: RunConfig) -> dict:
    report = smoke_report(config.workload, topology=config.topology)
    report["milestone"] = "v0.2 configured topology"
    report["run_configuration"] = config.manifest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DRLMOTS software baseline with fixed or seeded inputs")
    parser.add_argument("--output", type=Path, help="Also save the exact inputs and results as JSON")
    parser.add_argument("--config", type=Path, help="Runnable synthetic YAML; cannot combine with workload flags")
    parser.add_argument("--task-count", type=int, help="Generate this many synthetic tasks; requires --seed")
    parser.add_argument("--seed", type=int, help="Explicit nonnegative synthetic-workload seed")
    parser.add_argument("--interarrival-ms", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Uniform interval bounds; demo default: 0 to 3 ms")
    parser.add_argument("--length-bytes", nargs=2, type=int, metavar=("MIN", "MAX"),
                        help="Inclusive integer length bounds; demo default: 500 to 2000 bytes")
    parser.add_argument("--data-size-kb", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Uniform data-size bounds; demo default: 20 to 150 kB")
    args = parser.parse_args()
    synthetic_requested = any(value is not None for value in (
        args.task_count, args.seed, args.interarrival_ms, args.length_bytes, args.data_size_kb
    ))
    settings = None
    if args.config is not None:
        if synthetic_requested:
            parser.error("--config cannot be combined with seed/count/range workload flags")
        try:
            report = configured_report(load_run_config(args.config))
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    elif synthetic_requested:
        if args.task_count is None or args.seed is None:
            parser.error("synthetic generation requires both --task-count and --seed")
        try:
            settings = SyntheticWorkloadSettings(
                task_count=args.task_count, seed=args.seed,
                interarrival_ms_range=args.interarrival_ms or (0, 3),
                length_bytes_range=args.length_bytes or (500, 2000),
                data_size_kb_range=args.data_size_kb or (20, 150),
            )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        report = smoke_report(settings)
    else:
        report = smoke_report()
    content = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
