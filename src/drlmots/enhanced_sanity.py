"""Fixed-budget enhanced DQN training sanity check (A-034), not a benchmark.

Validation never selects a checkpoint or extends the budget. Both the live frozen
trajectories and identical saved observation probes are compared to initialization.
"""
import argparse
from dataclasses import asdict, dataclass, fields, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import torch
import yaml

from drlmots.enhanced_dqn import EnhancedDQN, EnhancedDQNConfig, run_enhanced_episode
from drlmots.enhanced_rl_contract import EncodedObservation
from drlmots.normalization import _digest
from drlmots.objective_model import generate_work, integer, load_objective_model
from drlmots.run_config import _UniqueKeyLoader, load_run_config


@dataclass(frozen=True)
class SanityPlan:
    training_episodes: int = 16
    validation_every_episodes: int = 4
    training_workload_seed_start: int = 10000
    training_failure_seed_start: int = 20000
    evaluation_workload_seeds: tuple[int, ...] = (81000, 81001)
    evaluation_failure_seeds: tuple[int, ...] = (91000, 91001)
    recovery_probe_workload_seed: int = 81002

    def __post_init__(self):
        for name in ("training_episodes", "validation_every_episodes", "training_workload_seed_start",
                     "training_failure_seed_start", "recovery_probe_workload_seed"):
            integer(getattr(self, name), name)
        if not self.training_episodes or not self.validation_every_episodes:
            raise ValueError("positive episode/validation intervals required")
        for name in ("evaluation_workload_seeds", "evaluation_failure_seeds"):
            seeds = tuple(getattr(self, name))
            for seed in seeds:
                integer(seed, name)
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("nonempty distinct evaluation seeds required")
            object.__setattr__(self, name, seeds)
        if len(self.evaluation_workload_seeds) != len(self.evaluation_failure_seeds):
            raise ValueError("evaluation seeds must form matching workload/failure pairs")
        if self.recovery_probe_workload_seed in self.evaluation_workload_seeds:
            raise ValueError("recovery probe needs a separate workload seed")
        for start, held in ((self.training_workload_seed_start, (*self.evaluation_workload_seeds, self.recovery_probe_workload_seed)),
                            (self.training_failure_seed_start, self.evaluation_failure_seeds)):
            integer(start + self.training_episodes - 1, "last training seed")
            if any(start <= value < start + self.training_episodes for value in held):
                raise ValueError("training and evaluation seeds must be disjoint")

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)} | {"schema_version", "kind"}:
            raise ValueError("sanity plan requires exactly its documented fields")
        data = dict(data)
        version, kind = data.pop("schema_version"), data.pop("kind")
        if type(version) is not int or version != 1 or kind != "enhanced_dqn_training_sanity":
            raise ValueError("unsupported sanity plan kind/version")
        return cls(**data)

    def manifest(self):
        return dict(schema_version=1, kind="enhanced_dqn_training_sanity", **asdict(self))


def network_digest(network):
    return _digest({k: v.detach().tolist() for k, v in network.state_dict().items()})


def parameter_distance(network, original):
    return math.sqrt(math.fsum(float(((v.detach().double() - original[k].double()) ** 2).sum())
                              for k, v in network.state_dict().items()))


def _write_json(path, data, *, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=None if compact else 2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _observation(record):
    return EncodedObservation(record["context_sha256"], record["time_ms"], record["task_id"],
                              record["placement_kind"], tuple(record["values"]), tuple(record["legal_mask"]))


def recovery_probe(model, seed):
    """Explicit diagnostic trace; force one transfer interruption in any initial VM.

    This scenario is fixed before training and never inserted in replay. It is not
    sampled statistical evidence. The other tasks start after the common reboot.
    """
    tasks = generate_work(model, seed)
    if not tasks:
        raise ValueError("sanity experiment needs a nonempty workload")
    d, horizon = model.data, model.data["horizon_ms"]
    start, unit = min(5., horizon / 20), min(1., horizon / 40)
    input_kb = max(d["workload"]["input_kb"][1], (start + unit) * max(l["bandwidth_mbps"] for l in d["links"].values()) / 8)
    tasks[0] = replace(tasks[0], arrival_ms=0, input_kb=input_kb,
                       deadline_ms=min(horizon, 90.))
    for index in range(1, len(tasks)):
        arrival = horizon / 4 + (index - 1) * horizon / (4 * max(1, len(tasks) - 1))
        tasks[index] = replace(tasks[index], arrival_ms=arrival, deadline_ms=min(horizon, arrival + 90))
    outages = [dict(node_id=node, start_ms=start, detection_ms=start + 2 * unit,
                    repair_end_ms=start + 4 * unit, end_ms=start + 7 * unit)
               for node in sorted(d["failures"])]
    return tasks, outages


def diagnostic_scenarios(model, plan):
    result = [(f"generated-{i + 1}", dict(workload_seed=w, failure_seed=f))
              for i, (w, f) in enumerate(zip(plan.evaluation_workload_seeds, plan.evaluation_failure_seeds))]
    tasks, outages = recovery_probe(model, plan.recovery_probe_workload_seed)
    result.append(("recovery-probe", dict(tasks=tasks, outages=outages)))
    return result


def _metric_row(scenario, report):
    metrics = report["metrics"]
    return dict(scenario=scenario, return_value=report["rl"]["total_reward"],
        completed=metrics["completed_count"], rejected=metrics["rejected_count"],
        execution_failure=metrics["execution_failure_count"], deadline_missed=metrics["deadline_missed_count"],
        unfinished=metrics["unfinished_count"], total_energy_j=metrics["total_energy_j"],
        successful_makespan_ms=metrics["successful_makespan_ms"], recovered=metrics["recovered_task_count"],
        retry_attempts=metrics["retry_attempt_count"], feasible=report["objective"]["feasible"],
        tasks_sha256=report["workload"]["sha256"], outages_sha256=report["outage_trace"]["sha256"])


def _probes(agent, probes):
    rows = []
    for probe in probes:
        action, values = agent.select(_observation(probe["observation"]), epsilon=0)
        rows.append(dict(scenario=probe["scenario"], task_id=probe["observation"]["task_id"],
            placement_kind=probe["observation"]["placement_kind"], action=action, q_values=values,
            initial_action=probe["initial_action"], changed=action != probe["initial_action"],
            max_q_change=max(abs(a - b) for a, b in zip(values, probe["initial_q_values"]))))
    return rows


def _evaluate(agent, run, model, scenarios, output, episode, reference, probes):
    before = agent.fingerprint
    metrics, reports = [], {}
    for name, kwargs in scenarios:
        report = run_enhanced_episode(agent, run, model, learn=False, **kwargs)
        reports[name] = report
        row = _metric_row(name, report)
        row["episode"] = episode
        row["trace"] = f"validation/episode-{episode:03d}-{name}.json"
        _write_json(output / row["trace"], report, compact=True)
        metrics.append(row)
        if episode == 0:
            reference[name] = row
            for decision in report["rl"]["decisions"]:
                if decision["action"] is not None:
                    probes.append(dict(scenario=name, observation=decision["observation"],
                                       initial_action=decision["action"], initial_q_values=decision["q_values"]))
        elif any(row[k] != reference[name][k] for k in ("tasks_sha256", "outages_sha256")):
            raise RuntimeError("frozen diagnostic input traces changed between stages")
    fixed = _probes(agent, probes)
    if agent.fingerprint != before:
        raise RuntimeError("diagnostic evaluation mutated learner state")
    stage = dict(episode=episode, learner_fingerprint=before, online_sha256=network_digest(agent.online),
                 updates=agent.updates, epsilon=agent.epsilon, frozen_verified=True, metrics=metrics,
                 probe_changes={kind: sum(p["changed"] for p in fixed if p["placement_kind"] == kind)
                                for kind in ("initial", "recovery")}, fixed_observation_results=fixed)
    _write_json(output / f"validation/stage-{episode:03d}.json", stage)
    return stage, reports


def run_sanity(run, model, settings, plan, output_dir, *, render_figures=True):
    """Run a predeclared budget and retain the final learner, irrespective of scores."""
    if type(plan) is not SanityPlan or type(settings) is not EnhancedDQNConfig:
        raise ValueError("typed sanity plan/DQN settings required")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("choose an empty output directory; sanity runs never overwrite prior evidence")
    agent = EnhancedDQN.from_configs(run, model, settings)
    scenarios = diagnostic_scenarios(model, plan)
    output.mkdir(parents=True, exist_ok=True)
    source_names = ("enhanced_sanity.py", "enhanced_dqn.py", "enhanced_rl_contract.py", "enhanced_telemetry.py", "objective_sim.py", "objective_model.py")
    manifest = dict(kind="enhanced_dqn_training_sanity", status="running", plan=plan.manifest(),
        run_configuration=run.manifest(), model_configuration=model.manifest(), dqn_configuration=asdict(settings),
        contract_sha256=agent.contract.sha256, environment_sha256=agent.environment_sha256,
        runtime=dict(python=platform.python_version(), torch=torch.__version__, device="cpu", threads=1),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names},
        selection="final predeclared episode only; validation never changes the budget or selects a checkpoint",
        scope="training plumbing and behavior sanity; not a trained-policy superiority benchmark")
    _write_json(output / "experiment.json", manifest)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _execute(agent, run, model, plan, scenarios, output, manifest, render_figures)
    finally:
        torch.set_num_threads(original_threads)


def _execute(agent, run, model, plan, scenarios, output, manifest, render_figures):
    original = {k: v.detach().clone() for k, v in agent.online.state_dict().items()}
    reference, probes, episodes, stages = {}, [], [], []
    agent.save(output / "initialization.json")
    initial_fingerprint = agent.fingerprint
    stage, _ = _evaluate(agent, run, model, scenarios, output, 0, reference, probes)
    stages.append(stage)
    _write_json(output / "fixed-observation-probes.json", probes, compact=True)
    training_kinds = {"initial": 0, "recovery": 0}
    for episode in range(1, plan.training_episodes + 1):
        updates_before, epsilon_before = agent.updates, agent.epsilon
        workload_seed = plan.training_workload_seed_start + episode - 1
        failure_seed = plan.training_failure_seed_start + episode - 1
        report = run_enhanced_episode(agent, run, model, learn=True, workload_seed=workload_seed, failure_seed=failure_seed)
        losses = agent.losses[updates_before:]
        trace = f"training/episode-{episode:03d}.json"
        _write_json(output / trace, report, compact=True)
        for transition in report["rl"]["transitions"]:
            training_kinds[transition["state"]["placement_kind"]] += 1
        row = dict(episode=episode, workload_seed=workload_seed, failure_seed=failure_seed, trace=trace,
            **{k: v for k, v in _metric_row("training", report).items() if k != "scenario"},
            updates=agent.updates, episode_updates=agent.updates - updates_before,
            mean_loss=math.fsum(losses) / len(losses) if losses else None,
            epsilon_before=epsilon_before, epsilon_after=agent.epsilon,
            parameter_l2_change=parameter_distance(agent.online, original))
        episodes.append(row)
        print(f"Episode {episode}/{plan.training_episodes}: updates={agent.updates}, return={row['return_value']:.6f}, epsilon={agent.epsilon:.4f}", flush=True)
        if episode % plan.validation_every_episodes == 0 or episode == plan.training_episodes:
            stage, reports = _evaluate(agent, run, model, scenarios, output, episode, reference, probes)
            stages.append(stage)
    agent.save(output / "trained-final.json")
    restored = EnhancedDQN.load(output / "trained-final.json", agent.contract, host_nodes=agent.host_nodes)
    if restored.fingerprint != agent.fingerprint:
        raise RuntimeError("trained checkpoint failed exact round trip")
    # Re-run one full frozen scenario from the restored checkpoint, not just its weights.
    name, kwargs = scenarios[-1]
    replayed = run_enhanced_episode(restored, run, model, learn=False, **kwargs)
    if _digest(replayed) != _digest(reports[name]):
        raise RuntimeError("reloaded final checkpoint changed frozen behavior")
    final = stages[-1]
    distance = parameter_distance(agent.online, original)
    checks = dict(optimizer_updates=agent.updates > 0, weights_changed=distance > 0,
        finite_losses=bool(agent.losses) and all(math.isfinite(v) for v in agent.losses),
        frozen_validation=all(s["frozen_verified"] for s in stages), checkpoint_restored=True,
        fixed_state_q_changed=any(p["max_q_change"] > 0 for p in final["fixed_observation_results"]))
    result = dict(kind="enhanced_dqn_training_sanity_result", checks=checks,
        learning_updates_verified=all(checks.values()), training_episodes=agent.episodes,
        training_decisions=agent.decisions, optimizer_updates=agent.updates,
        target_syncs=agent.updates // agent.settings.target_sync_updates,
        replay_size=len(agent.replay), training_transition_kinds=training_kinds,
        parameter_count=sum(p.numel() for p in agent.online.parameters()), parameter_l2_change=distance,
        initial_online_sha256=stages[0]["online_sha256"], final_online_sha256=final["online_sha256"],
        initial_fingerprint=initial_fingerprint, final_fingerprint=agent.fingerprint,
        behavior_changed=any(p["changed"] for p in final["fixed_observation_results"]),
        fixed_probe_counts={kind: sum(p["observation"]["placement_kind"] == kind for p in probes) for kind in ("initial", "recovery")},
        fixed_probe_action_changes=final["probe_changes"], episodes=episodes, validation_stages=stages,
        losses=agent.losses, checkpoint_selection="last fixed-budget episode; no selection on validation",
        interpretation="Weight/Q/action changes verify learning activity; loss or efficiency improvement is not guaranteed. Four-heuristic held-out comparison remains milestone 5.")
    _write_json(output / "result.json", result)
    for filename, rows in (("training.csv", episodes), ("validation.csv", [r for s in stages for r in s["metrics"]]),
                           ("losses.csv", [dict(update=i + 1, huber_loss=v) for i, v in enumerate(agent.losses)])):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["update", "huber_loss"])
            writer.writeheader()
            writer.writerows(rows)
    if render_figures:
        _figures(result, output)
    _report(result, output)
    _write_json(output / "experiment.json", {**manifest, "status": "complete", "learning_updates_verified": result["learning_updates_verified"]})
    return result


def _figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    rows = result["episodes"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    axes[0].plot([r["episode"] for r in rows], [r["return_value"] for r in rows], marker="o")
    axes[0].set(xlabel="Training episode (different fixed seed pairs)", ylabel="Actual return", title="Training returns: exploration is enabled")
    axes[1].plot([r["episode"] for r in rows], [r["mean_loss"] if r["mean_loss"] is not None else math.nan for r in rows], marker="o")
    axes[1].set(xlabel="Training episode", ylabel="Mean Huber loss", title="Changing replay and target network: loss need not fall monotonically")
    for kind in ("initial", "recovery"):
        axes[2].plot([s["episode"] for s in result["validation_stages"]], [s["probe_changes"][kind] for s in result["validation_stages"]], marker="o", label=kind)
    axes[2].set(xlabel="Completed training episodes", ylabel="Changed legal argmax actions", title="Identical saved observations compared with initialization")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"learning.{extension}", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for name in [r["scenario"] for r in result["validation_stages"][0]["metrics"]]:
        points = [(s["episode"], next(r["return_value"] for r in s["metrics"] if r["scenario"] == name)) for s in result["validation_stages"]]
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
    ax.set(xlabel="Completed training episodes", ylabel="Frozen return (larger is better)", title="Fixed diagnostic traces; epsilon=0, no updates or checkpoint selection")
    ax.legend()
    ax.grid(alpha=.25)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"validation.{extension}", dpi=150)
    plt.close(fig)


def _report(result, output):
    lines = ["# Enhanced DQN: small training sanity experiment", "", result["interpretation"], "",
        f"Episodes: {result['training_episodes']}; updates: {result['optimizer_updates']}; parameter L2 change: {result['parameter_l2_change']:.8f}.",
        f"Fixed observation action changes: {result['fixed_probe_action_changes']} of {result['fixed_probe_counts']}.", "",
        "The final checkpoint is the last predeclared episode, even if a diagnostic result worsens.", "",
        "| Scenario | Stage | Done | Return | Energy J | Successful makespan ms | Recovered |", "|---|---|---:|---:|---:|---:|---:|"]
    for stage in (result["validation_stages"][0], result["validation_stages"][-1]):
        for row in stage["metrics"]:
            makespan = "undefined" if row["successful_makespan_ms"] is None else f"{row['successful_makespan_ms']:.6f}"
            lines.append(f"| {row['scenario']} | {stage['episode']} | {row['completed']} | {row['return_value']:.6f} | {row['total_energy_j']:.6f} | {makespan} | {row['recovered']} |")
    lines += ["", "Read completed/rejected/deadline/unfinished counts alongside timing and energy (validation.csv).",
        "The recovery probe is deliberately constructed; the generated scenarios use disjoint diagnostic seeds.",
        "No conclusion about superiority to the four rules follows from this small run.", ""]
    for name, title in (("learning", "Training, loss and identical-observation behavior"), ("validation", "Frozen diagnostic returns")):
        if (output / f"figures/{name}.png").exists():
            lines.extend([f"![{title}](figures/{name}.png)", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed-budget enhanced DQN training sanity experiment, not a policy benchmark.")
    parser.add_argument("--config", default="configs/rl_network.yaml")
    parser.add_argument("--model-config", default="configs/objective_model.yaml")
    parser.add_argument("--dqn-config", default="configs/enhanced_dqn.yaml")
    parser.add_argument("--plan-config", default="configs/enhanced_sanity.yaml")
    parser.add_argument("--output-dir", default="outputs/objective-model/dqn-sanity/run-1")
    args = parser.parse_args(argv)
    try:
        settings = EnhancedDQNConfig.parse(yaml.load(Path(args.dqn_config).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader))
        plan = SanityPlan.parse(yaml.load(Path(args.plan_config).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader))
        result = run_sanity(load_run_config(args.config), load_objective_model(args.model_config), settings, plan, args.output_dir)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"enhanced sanity: {error}\n")
    print(f"Saved {args.output_dir}; updates={result['optimizer_updates']}; fixed action changes={result['fixed_probe_action_changes']}")
    if not result["learning_updates_verified"]:
        parser.exit(2, "The fixed experiment did not establish all learning-update checks; see result.json.\n")


if __name__ == "__main__":
    main()
