"""Run fixed or seeded synthetic workloads through the verified baseline."""

import argparse
from dataclasses import asdict
import json
from importlib.metadata import version
from pathlib import Path

from drlmots.entities import Host, ResourceTier, VM
from drlmots.environment import simulate
from drlmots.baselines import POLICIES, FOG_FIRST_POLICY
from drlmots.run_config import RunConfig, load_run_config
from drlmots.reward import RewardConfig, load_reward_config
from drlmots.transitions import build_transitions, summarize_rewards
from drlmots.topology import AccessPoint, CloudDatacenter, EndNode, FogNode, Topology
from drlmots.workload import (
    SyntheticWorkloadSettings, fixed_workload, seeded_workload, workload_manifest,
)


def smoke_report(
    settings: SyntheticWorkloadSettings | None = None,
    *,
    topology: Topology | None = None,
    resource_units=None,
    policy: str = "least_queue_fcfs",
    capture_decisions: bool = False,
    reward_config: RewardConfig | None = None,
    placement_config=None,
    controller=None,
) -> dict:
    if topology is None:
        topology = _smoke_topology()
    tasks = seeded_workload(settings) if settings is not None else fixed_workload([
        {"task_id": "task-1", "arrival_time": 0, "length": 1000, "data_size_kb": 20},
        {"task_id": "task-2", "arrival_time": 0, "length": 2000, "data_size_kb": 20},
        {"task_id": "task-3", "arrival_time": 1, "length": 500, "data_size_kb": 20},
    ], length_unit="bytes")
    units = {} if resource_units is None else resource_units
    result = simulate(topology, tasks, task_length_unit="bytes", resource_units=units, policy=policy,
                      capture_decisions=capture_decisions, reward_config=reward_config,
                      placement_config=placement_config, controller=controller)
    return _report(topology, tasks, settings, result, units, policy)


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


def _report(topology, tasks, settings, result, resource_units, policy) -> dict:
    report = {
        "version": version("drlmots"),
        "milestone": "v0.2 workload generation" if settings is not None else "v0.1 fixed smoke",
        "implementation_scope": "deterministic software baseline; not paper reproduction",
        "policy": policy,
        "tie_breaking": "VM ID; simultaneous arrivals retain input order; completions first",
        "service_time_model": "explicit byte-valued fixture: length * 2 / 1000 milliseconds",
        "queue_capacity_semantics": "waiting tasks only; one additional running task per VM",
        "overflow_policy": "reject when all resource-feasible VM queues are full",
        "no_capacity_policy": "reject if no VM satisfies every task demand",
        "network_delay_ms": 0,
        "resource_feasibility": "componentwise per-VM fit; missing capacity is unknown; no host budget inferred",
        "resource_units": dict(resource_units),
        "resource_reservation_policy": "running tasks only; release before queued starts and same-time arrivals",
        "workload": workload_manifest(tasks, length_unit="bytes", generator_settings=settings,
                                      resource_units=resource_units),
        "topology": asdict(topology),
        "topology_counts": {name: len(getattr(topology, name)) for name in (
            "hosts", "vms", "fog_nodes", "cloud_datacenters", "access_points", "end_nodes"
        )},
        "completed_count": result.completed_count,
        "failed_count": result.failed_count,
        "makespan_ms": result.makespan_ms,
        "average_completion_time_ms": result.average_completion_time_ms,
        "outcomes": [asdict(item) for item in result.outcomes],
        "resource_events": [asdict(item) for item in result.resource_events],
        "available_resources": result.available_resources,
    }
    if result.decision_schema is not None:
        decisions = []
        for decision in result.decisions:
            item = asdict(decision)
            if decision.reward is None:
                item.pop("reward")
            decisions.append(item)
        report["decision_trace"] = {
            "schema": result.decision_schema,
            "decisions": decisions,
        }
    if result.placement_config is not None:
        report["network_delay_ms"] = None
        report["network_delay_model"] = "per-task upload plus propagation; see placement configuration and transfers"
        report["queue_capacity_semantics"] = "waiting/uploading tasks only; one additional running task per VM"
        report["no_capacity_policy"] = "reject if no reachable VM satisfies every task demand"
        report["overflow_policy"] = "reject when all reachable resource-feasible VM queues are full"
        if policy == FOG_FIRST_POLICY:
            report["tie_breaking"] = "earliest predicted finish; equal fog/cloud finish prefers fog; VM ID within tier; completions/ready starts precede arrivals"
        report["placement"] = {"configuration": result.placement_config,
                               "decisions": [asdict(item) for item in result.placement_decisions],
                               "transfers": list(result.transfers)}
    if result.reward_config is not None:
        report["reward"] = {
            "configuration": result.reward_config.manifest(),
            "scope": "synthetic projected-latency surrogate; not paper Eq. (25)",
            "timing": "immediate at each placement/rejection; no completion or terminal bonus",
            "summary": summarize_rewards(result.decisions),
            "transitions": [asdict(item) for item in build_transitions(result.decisions)],
        }
    return report


def configured_report(config: RunConfig, *, policy: str | None = None, capture_decisions: bool = False,
                      reward_config: RewardConfig | None = None, controller=None) -> dict:
    if controller is not None and policy is None:
        raise ValueError("controller requires an explicit policy label")
    selected_policy = policy if policy is not None else FOG_FIRST_POLICY if config.placement else "least_queue_fcfs"
    report = smoke_report(config.workload, topology=config.topology, resource_units=config.resource_units,
                          policy=selected_policy, capture_decisions=capture_decisions, reward_config=reward_config,
                          placement_config=config.placement, controller=controller)
    report["milestone"] = "v0.2 resource-aware baseline" if config.schema_version == 2 else "v0.2 configured topology"
    report["run_configuration"] = config.manifest()
    if config.placement is not None:
        report["milestone"] = "network-aware synthetic scheduling"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DRLMOTS software baseline with fixed or seeded inputs")
    parser.add_argument("--output", type=Path, help="Also save the exact inputs and results as JSON")
    parser.add_argument("--config", type=Path, help="Runnable synthetic YAML; cannot combine with workload flags")
    parser.add_argument("--policy", choices=POLICIES, help="Placement policy; schema 3 defaults to fog-first completion")
    parser.add_argument("--capture-decisions", action="store_true", help="Save provisional state vectors, VM action masks and decisions")
    parser.add_argument("--reward-config", type=Path, help="Synthetic reward YAML; implies decision capture")
    parser.add_argument("--task-count", type=int, help="Generate this many synthetic tasks; requires --seed")
    parser.add_argument("--seed", type=int, help="Explicit nonnegative synthetic-workload seed")
    parser.add_argument("--interarrival-ms", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Uniform interval bounds; demo default: 0 to 3 ms")
    parser.add_argument("--length-bytes", nargs=2, type=int, metavar=("MIN", "MAX"),
                        help="Inclusive integer length bounds; demo default: 500 to 2000 bytes")
    parser.add_argument("--data-size-kb", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Uniform data-size bounds; demo default: 20 to 150 kB")
    args = parser.parse_args()
    try:
        reward_config = load_reward_config(args.reward_config) if args.reward_config is not None else None
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    synthetic_requested = any(value is not None for value in (
        args.task_count, args.seed, args.interarrival_ms, args.length_bytes, args.data_size_kb
    ))
    settings = None
    if args.config is not None:
        if synthetic_requested:
            parser.error("--config cannot be combined with seed/count/range workload flags")
        try:
            report = configured_report(load_run_config(args.config), policy=args.policy,
                                       capture_decisions=args.capture_decisions, reward_config=reward_config)
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
        try:
            report = smoke_report(settings, policy=args.policy or "least_queue_fcfs", capture_decisions=args.capture_decisions,
                                  reward_config=reward_config)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    else:
        try:
            report = smoke_report(policy=args.policy or "least_queue_fcfs", capture_decisions=args.capture_decisions, reward_config=reward_config)
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    content = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
