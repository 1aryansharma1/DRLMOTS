"""Reproducible multi-objective experiments and honest efficiency reports.

Run: python -m drlmots.efficiency --help
"""

import argparse
from copy import deepcopy
from dataclasses import replace
from html import escape
import csv
import json
import math
from pathlib import Path
import statistics
import zipfile

import yaml

from drlmots.__main__ import configured_report
from drlmots.dqn import BaselineController
from drlmots.normalization import _digest
from drlmots.objectives import MultiObjectiveConfig, evaluate_objectives, frame_objectives
from drlmots.q_network import QNetwork, NetworkConfig
from drlmots.rl_state import RLRewardConfig, RLStateSchema, learning_context
from drlmots.run_config import _UniqueKeyLoader, load_run_config
from drlmots.train import (checkpoint_artifact, continue_training, cpu_execution, initialize,
                           load_checkpoint, load_training_config, run_episode, save_checkpoint,
                           seeded_config, validate_split_plan, write_json, _held_out)


SCENARIOS = {"light": (0, 4), "normal": (0, 2), "burst": (0, 0)}
POLICIES = ("least_queue_fcfs", "first_fit_fcfs", "fog_first_completion", "objective_greedy",
            "latency_dqn", "balanced_dqn", "fog_balanced_dqn")
METRICS = ("completion_rate", "makespan_ms", "successful_mean_latency_ms", "p95_latency_ms",
           "total_energy_j", "energy_per_completed_task_j", "energy_per_expected_success_j",
           "mean_failure_risk_admitted", "expected_success_rate_all_arrivals", "mean_all_arrival_utility",
           "occupancy", "cloud_fraction")
LOWER_BETTER = {"makespan_ms", "successful_mean_latency_ms", "p95_latency_ms", "total_energy_j",
                "energy_per_completed_task_j", "energy_per_expected_success_j", "mean_failure_risk_admitted"}


def load_objectives(path):
    return MultiObjectiveConfig.parse(yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader))


class ObjectiveGreedy:
    def __init__(self, objective, latency_scale_ms):
        self.objective, self.latency_scale_ms = objective, latency_scale_ms
        self.records = []

    def __call__(self, frame):
        legal = [i for i, vm in enumerate(frame.vms) if vm.estimate is not None]
        action = max(legal, key=lambda i: (frame_objectives(frame, i, self.objective,
                                                           self.latency_scale_ms)["utility"], -i)) if legal else None
        vm_id = frame.vms[action].vm_id if action is not None else None
        self.records.append({"task_id": frame.task_id, "vm_id": vm_id, "action": action, "legal_mask": frame.mask})
        return vm_id


def metrics_for(report, objective, settings):
    evidence = evaluate_objectives(report, objective, latency_scale_ms=settings.latency_scale_ms,
                                   rejection_penalty=settings.rejection_penalty)
    successful = [row for row in report["outcomes"] if row["status"] == "completed"]
    latencies = sorted(row["finish_time_ms"] - row["arrival_time_ms"] for row in successful)
    values = evidence["metrics"]
    # Nearest-rank p95, explicit even for a one-task run.
    values["p95_latency_ms"] = latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None
    busy = math.fsum(row["finish_time_ms"] - row["start_time_ms"] for row in successful)
    values["occupancy"] = busy / (len(objective.vms) * report["makespan_ms"]) if report["makespan_ms"] else 0.0
    cloud = {vm["vm_id"] for vm in report["run_configuration"]["topology"]["vms"] if vm["tier"] == "cloud"}
    values["cloud_fraction"] = sum(row["vm_id"] in cloud for row in successful) / len(successful) if successful else None
    report["multi_objective_evaluation"] = evidence
    return values


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def aggregate(rows):
    """Keep model-initialization and workload variation separate; no pseudoreplication."""
    result = []
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        for policy in POLICIES:
            group = [row for row in rows if row["scenario"] == scenario and row["policy"] == policy]
            if not group:
                continue
            seeds = sorted({row["model_seed"] for row in group})
            workloads = sorted({row["workload_seed"] for row in group})
            item = {"scenario": scenario, "policy": policy, "model_seeds": seeds if policy.endswith("dqn") else [],
                    "workload_seeds": workloads, "metrics": {}}
            for metric in METRICS:
                values = [row["metrics"][metric] for row in group]
                by_model = [_mean([row["metrics"][metric] for row in group if row["model_seed"] == seed]) for seed in seeds]
                by_workload = [_mean([row["metrics"][metric] for row in group if row["workload_seed"] == seed]) for seed in workloads]
                model_values = [v for v in by_model if v is not None]
                workload_values = [v for v in by_workload if v is not None]
                item["metrics"][metric] = {"mean": _mean(values),
                    "model_seed_std": statistics.stdev(model_values) if policy.endswith("dqn") and len(model_values) > 1 else None,
                    "workload_seed_std": statistics.stdev(workload_values) if len(workload_values) > 1 else None,
                    "defined_runs": sum(v is not None for v in values)}
            result.append(item)
    return result


def paired_changes(rows):
    indexed = {(r["scenario"], r["model_seed"], r["workload_seed"], r["policy"]): r for r in rows}
    changes = []
    for row in rows:
        baseline = indexed[(row["scenario"], row["model_seed"], row["workload_seed"], "least_queue_fcfs")]
        for metric in METRICS:
            if metric in ("occupancy", "cloud_fraction"):
                continue
            actual, reference = row["metrics"][metric], baseline["metrics"][metric]
            # Compute a paired relative change before averaging, not a ratio of means.
            improvement = None if actual is None or reference in (None, 0) else (
                (reference - actual) if metric in LOWER_BETTER else (actual - reference)) / abs(reference) * 100
            changes.append({"scenario": row["scenario"], "policy": row["policy"],
                            "model_seed": row["model_seed"], "workload_seed": row["workload_seed"],
                            "metric": metric, "improvement_pct": improvement})
    return changes


def pareto_policies(summaries, scenario):
    # Aggregate screening, not statistical dominance. Include rejection to avoid
    # labeling an energy-saving reject-all scheduler as efficient.
    dimensions = (("completion_rate", -1), ("successful_mean_latency_ms", 1),
                  ("energy_per_expected_success_j", 1), ("expected_success_rate_all_arrivals", -1))
    candidates = []
    for row in summaries:
        if row["scenario"] != scenario:
            continue
        values = [row["metrics"][name]["mean"] for name, _ in dimensions]
        if any(value is None for value in values):
            continue
        candidates.append((row["policy"], tuple(value * sign for value, (_, sign) in zip(values, dimensions))))
    return [name for name, vector in candidates if not any(
        all(a <= b for a, b in zip(other, vector)) and any(a < b for a, b in zip(other, vector))
        for other_name, other in candidates if other_name != name)]


def run_experiment(config, settings, objective, output_dir, *, model_seeds=(42, 43, 44), resume=False,
                   scenarios=("light", "normal", "burst")):
    import matplotlib  # Validate optional dependency before training.
    if config.placement is None or settings.multi_objective is not None:
        raise ValueError("efficiency experiment needs a network config and a legacy training config; use --objectives separately")
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("model seeds must be nonempty and unique")
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(name not in SCENARIOS for name in scenarios):
        raise ValueError("unknown or duplicate workload scenario")
    objective.validate_vm_ids(vm.vm_id for vm in config.topology.vms)
    validate_split_plan(config, settings)
    output = Path(output_dir)
    plan = {"schema_version": 1, "kind": "multi_objective_efficiency_experiment",
            "run_config": config.manifest(), "training_config": settings.manifest(),
            "objectives": objective.manifest(), "model_seeds": list(model_seeds),
            "scenarios": {name: list(SCENARIOS[name]) for name in scenarios},
            "evaluation": "paired workload seeds; frozen weights; no held-out selection or tuning"}
    plan_path = output / "experiment.json"
    checkpoints = list(output.glob("models/*/checkpoint.json"))
    if checkpoints and not resume:
        raise ValueError("output contains models; choose a fresh directory or --resume")
    if plan_path.exists() and (not resume or _digest(json.loads(plan_path.read_text(encoding="utf-8"))) != _digest(plan)):
        raise ValueError("output already has a different experiment; choose a fresh directory")
    write_json(plan_path, plan)
    agents, evidence = {}, {}
    specs = (("latency_dqn", "standard", None), ("balanced_dqn", "standard", objective),
             ("fog_balanced_dqn", "fog_preferred", objective))
    with cpu_execution():
        for seed in model_seeds:
            for label, variant, obj in specs:
                current = replace(settings, seed=seed, multi_objective=obj)
                folder = output / "models" / (label + "-" + str(seed))
                checkpoint = folder / "checkpoint.json"
                if checkpoint.exists():
                    agent = load_checkpoint(checkpoint, for_resume=True)
                    if (agent.config != current or agent.variant != variant or agent.run_config.manifest() != config.manifest()
                            or agent.normalizer.context != learning_context(config, variant)):
                        raise ValueError("saved model differs from experiment settings")
                else:
                    agent = initialize(config, current, variant)
                print("Training", label, "initialization", seed, flush=True)
                summary = continue_training(agent, output_dir=folder)
                save_checkpoint(agent, checkpoint)
                initial = QNetwork(agent.context, NetworkConfig(current.hidden_widths, current.seed))
                summary["initial_weights_sha256"] = _digest(initial.artifact()["weights"])
                if summary["initial_weights_sha256"] == summary["final_weights_sha256"]:
                    raise ValueError("training did not change initialized weights")
                if not agent.updates:
                    raise ValueError("model has no optimizer updates; increase the training budget")
                loaded = load_checkpoint(checkpoint, for_resume=True)
                if checkpoint_artifact(loaded)["sha256"] != checkpoint_artifact(agent)["sha256"]:
                    raise RuntimeError("checkpoint round trip changed training state")
                agents[(seed, label)] = loaded
                evidence[label + "-" + str(seed)] = {**summary, "checkpoint": checkpoint.relative_to(output).as_posix()}
        fingerprints = {key: checkpoint_artifact(agent)["sha256"] for key, agent in agents.items()}
        rows = []
        for scenario in scenarios:
            fixture = replace(config, workload=replace(config.workload, interarrival_ms_range=SCENARIOS[scenario]))
            for workload_seed in settings.evaluation_seeds:
                current = seeded_config(fixture, workload_seed)
                baseline_reports = []
                for policy in POLICIES[:4]:
                    if policy == "objective_greedy":
                        controller = ObjectiveGreedy(objective, settings.latency_scale_ms)
                    else:
                        controller = BaselineController(RLStateSchema.from_config(current),
                            RLRewardConfig(settings.latency_scale_ms, settings.rejection_penalty, objective), policy)
                    report = configured_report(current, policy=policy, controller=controller)
                    report["controller_decisions"] = controller.records
                    baseline_reports.append(report)
                for seed in model_seeds:
                    reports = deepcopy(baseline_reports)
                    for label, _, _ in specs:
                        agent = agents[(seed, label)]
                        _held_out(agent, current, workload_seed)
                        report = run_episode(agent, current)
                        report["policy"] = label
                        # Full states remain in training traces/checkpoints. Keep task,
                        # action, Q-values and objective components in evaluation traces.
                        for decision in report["rl"]["decisions"]:
                            decision.pop("raw_state", None)
                            decision.pop("normalized_state", None)
                        reports.append(report)
                    if len({r["workload"]["sha256"] for r in reports}) != 1 or len({r["run_configuration"]["sha256"] for r in reports}) != 1:
                        raise RuntimeError("paired evaluation inputs differ")
                    filename = "runs/" + scenario + "-model-" + str(seed) + "-workload-" + str(workload_seed) + ".json"
                    for report in reports:
                        metric = metrics_for(report, objective, settings)
                        rows.append({"scenario": scenario, "policy": report["policy"], "model_seed": seed,
                                     "workload_seed": workload_seed, "workload_sha256": report["workload"]["sha256"],
                                     "metrics": metric, "trace_file": filename})
                    write_json(output / filename, {"scenario": scenario, "model_seed": seed,
                               "workload_seed": workload_seed, "reports": reports})
                    print("Evaluated", scenario, "model", seed, "workload", workload_seed, flush=True)
        if fingerprints != {key: checkpoint_artifact(agent)["sha256"] for key, agent in agents.items()}:
            raise RuntimeError("evaluation mutated training state")
    result = {"experiment": plan, "models": evidence, "rows": rows, "summaries": aggregate(rows),
              "paired_changes": paired_changes(rows),
              "statistics": "model_seed_std: variation of workload-averaged model results; workload_seed_std: variation of model-averaged workloads. Rule repetitions are reused paired references, not independent samples. No significance claim.",
              "scope": "synthetic coefficients; analytical risk only; no injected failures, redundancy, idle energy or paper reproduction"}
    result["pareto_screen"] = {name: pareto_policies(result["summaries"], name) for name in scenarios}
    write_json(output / "evaluation.json", result)
    render_report(output, result, agents)
    return result


def _format(value):
    return "n/a" if value is None else f"{value:.4f}"


def render_report(output, result, agents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for seed in result["experiment"]["model_seeds"]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
        for ax, policy in zip(axes, POLICIES[-3:]):
            agent = agents[(seed, policy)]
            ax.plot([r["episode"] for r in agent.history], [r["return"] for r in agent.history], color="#087f8c")
            ax.set(title=policy.replace("_", " "), xlabel="Training episode", ylabel="Own training reward return")
            ax.grid(alpha=.2)
        fig.suptitle("Initialization " + str(seed) + ": reward contracts differ; compare final common evaluation metrics")
        for ext in ("png", "svg"):
            fig.savefig(figures / ("training-" + str(seed) + "." + ext), dpi=160)
        plt.close(fig)
    blocks, markdown = [], ["# Multi-objective efficiency experiment", "", result["scope"], "", result["statistics"], "",
        "## Method", "", "Initialization seeds: " + str(result["experiment"]["model_seeds"]),
        "Held-out workload seeds: " + str(result["experiment"]["training_config"]["evaluation_seeds"]),
        "Each model uses the fixed final training checkpoint. No validation tuning or held-out checkpoint selection.",
        "Light: interarrival 0-4 ms; normal: 0-2 ms; burst: simultaneous arrivals. Other workload settings and inputs are paired within each run.",
        "All policies are scored using one common objective model, even when trained with different rewards.",
        "Energy (J) = active power (W) * service time (ms) / 1000 + input data (decimal kB) * J/kB.",
        "Analytical failure probability = 1 - exp(-destination hazard per second * arrival-to-finish ms / 1000).",
        "Utility = 1 - weighted bounded latency, energy and failure costs; forced rejection = -P. This scalarized DQN is not a learned Pareto frontier.",
        "The latency reward is a surrogate; actual makespan is evaluated separately. No paper-equation equivalence is claimed.",
        "P95 is nearest-rank within each run. Reported latency and p95 averages are averages of run metrics, not pooled task quantiles.",
        "Rates and risks are fractions. Rejections count as zero expected successes. Energy excludes idle power, queue power, downloads and host sharing.",
        "Failure risk is expected analytical exposure, not observed failure/recovery performance; no redundancy is implemented.",
        "", "## Actual training", ""]
    for name, info in result["models"].items():
        markdown.append("- " + name + ": " + str(info["episodes_completed"]) + " episodes, " + str(info["updates"]) + " updates; " + info["checkpoint"])
    markdown.append("")
    display = (("completion_rate", "Completion rate"), ("makespan_ms", "Makespan ms"),
               ("successful_mean_latency_ms", "Mean latency ms"),
               ("p95_latency_ms", "p95 latency ms"), ("total_energy_j", "Active + upload J"),
               ("energy_per_expected_success_j", "J / expected success"),
               ("mean_failure_risk_admitted", "Analytical failure risk"),
               ("expected_success_rate_all_arrivals", "Expected success / arrivals"),
               ("mean_all_arrival_utility", "Common utility"))
    for scenario in result["experiment"]["scenarios"]:
        summaries = [r for r in result["summaries"] if r["scenario"] == scenario]
        labels = [r["policy"].replace("_", " ") for r in summaries]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        plot_specs = (("completion_rate", "Admitted and completed / submitted (%)", 100),
                      ("successful_mean_latency_ms", "Successful-task mean completion (ms)", 1),
                      ("energy_per_expected_success_j", "Active + input energy / expected success (J)", 1),
                      ("expected_success_rate_all_arrivals", "Analytical expected success / submitted (%)", 100))
        for ax, (metric, title, scale) in zip(axes.flat, plot_specs):
            values = [r["metrics"][metric]["mean"] for r in summaries]
            spread = [r["metrics"][metric]["model_seed_std"] for r in summaries]
            ax.barh(labels, [0 if value is None else value * scale for value in values],
                    xerr=[0 if value is None else value * scale for value in spread], color="#087f8c", capsize=3)
            ax.invert_yaxis()
            ax.set(title=title, xlabel="Mean; error bars = model-initialization SD (DQN only)")
            ax.grid(axis="x", alpha=.2)
        fig.suptitle(scenario.title() + " load: paired held-out workloads, synthetic energy/risk coefficients")
        for ext in ("png", "svg"):
            fig.savefig(figures / (scenario + "-metrics." + ext), dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        for row in summaries:
            x = row["metrics"]["energy_per_expected_success_j"]["mean"]
            y = row["metrics"]["successful_mean_latency_ms"]["mean"]
            if x is not None and y is not None:
                ax.scatter(x, y, s=65, label=row["policy"].replace("_", " "))
        ax.set(xlabel="J / expected successful task (lower is better)", ylabel="Successful mean latency ms (lower is better)",
               title=scenario.title() + ": energy-latency trade-off; read admission and risk alongside")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.margins(.15)
        ax.grid(alpha=.2)
        for ext in ("png", "svg"):
            fig.savefig(figures / (scenario + "-tradeoff." + ext), dpi=160)
        plt.close(fig)
        header = "<tr><th>Policy</th>" + "".join("<th>" + title + "</th>" for _, title in display) + "</tr>"
        table = []
        for row in summaries:
            table.append("<tr><th>" + escape(row["policy"]) + "</th>" + "".join(
                "<td>" + _format(row["metrics"][metric]["mean"]) + "</td>" for metric, _ in display) + "</tr>")
        details = []
        for row in summaries:
            changes = [r for r in result["paired_changes"] if r["scenario"] == scenario and r["policy"] == row["policy"]]
            details.append("<tr><th>" + escape(row["policy"]) + "</th>" + "".join("<td>" + _format(_mean([
                r["improvement_pct"] for r in changes if r["metric"] == metric])) + "%</td>" for metric, _ in display) + "</tr>")
        blocks.append('<section><h2>' + scenario.title() + ' workload</h2><img src="figures/' + scenario + '-metrics.svg" alt="Completion, latency, energy and analytical reliability comparison">'
                      + '<div class="scroll"><table>' + header + "".join(table) + '</table></div><p>Rates and risks in the table are fractions from 0 to 1. Latency and p95 exclude rejected tasks.</p>'
                      + '<img src="figures/' + scenario + '-tradeoff.svg" alt="Energy and latency trade-offs">'
                      + '<p>Non-dominated on aggregate admission, latency, energy per expected success and expected success rate: '
                      + escape(", ".join(result["pareto_screen"][scenario])) + '. This is descriptive screening, not statistical proof.</p>'
                      + '<details><summary>Paired changes versus least queue: positive is better, negative is worse</summary><div class="scroll"><table>'
                      + header + "".join(details) + '</table></div><p>Mean of per-pair relative improvements; zero/undefined baseline denominators are omitted.</p></details></section>')
        markdown.extend(["## " + scenario.title(), "", "| Policy | " + " | ".join(t for _, t in display) + " |",
                         "|---|" + "---:|" * len(display)])
        for row in summaries:
            markdown.append("| " + row["policy"] + " | " + " | ".join(_format(row["metrics"][m]["mean"]) for m, _ in display) + " |")
        markdown.append("")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "policy", "model_seed", "workload_seed", *result["rows"][0]["metrics"]])
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({name: row[name] for name in ("scenario", "policy", "model_seed", "workload_seed")} | row["metrics"])
    with (output / "paired-changes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["paired_changes"][0]))
        writer.writeheader()
        writer.writerows(result["paired_changes"])
    model_links = []
    for name, info in result["models"].items():
        model_links.append('<li>' + escape(name) + ': ' + str(info["episodes_completed"]) + ' episodes, ' + str(info["updates"]) + ' updates; <a href="'
                           + info["checkpoint"] + '" download>checkpoint</a> | <a href="'
                           + str(Path(info["checkpoint"]).parent / "history.json").replace("\\", "/") + '" download>training logs</a> | <a href="'
                           + str(Path(info["checkpoint"]).parent / "last-training-episode.json").replace("\\", "/") + '" download>last training trace</a></li>')
    traces = sorted({row["trace_file"] for row in result["rows"]})
    trace_links = ''.join('<li><a href="' + f + '" download>' + escape(f) + '</a></li>' for f in traces)
    model_figures = ''.join('<img src="figures/training-' + str(seed) + '.svg" alt="Actual model training returns">' for seed in result["experiment"]["model_seeds"])
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DRLMOTS | Multi-objective efficiency</title><style>
body{margin:0;background:#f2f6f6;color:#173143;font:16px/1.6 system-ui,sans-serif}main{max-width:1240px;margin:auto;padding:30px 18px}
h1{font-size:36px;line-height:1.2}h2{font-size:25px}section{background:white;border:1px solid #dae4e5;border-radius:14px;padding:24px;margin:24px 0}
.notice{background:#e2eeeb;border-left:4px solid #087f8c;padding:18px}img{width:100%;height:auto}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:10px;text-align:left;border-bottom:1px solid #dae4e5}a{color:#006c76}details{margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
</style></head><body><main><h1>Multi-objective cloud-fog efficiency</h1>
<p>Latency, active compute and transmission energy, and analytical failure risk. Standard latency-only DQN is retained as an ablation.</p>
<p class="notice">This report contains actual simulator runs and trained policies. Energy coefficients and failure hazards are illustrative synthetic inputs. No faults, backup recovery or real hardware energy measurements are simulated. Better results on one metric can cost performance on another.</p>
<p>Common score: 1 - [w_time L/(S_time+L) + w_energy E/(S_energy+E) + w_risk p_failure]. Forced rejections score -P. The time term is a per-task latency surrogate, not an exact makespan reward. Actual makespan is reported separately.</p>
'''
    html += '<p>' + escape(result["statistics"]) + '</p>' + ''.join(blocks)
    html += '<section><h2>Actual training and reproducibility</h2><ul>' + ''.join(model_links) + '</ul>' + model_figures
    html += '<p><a href="evaluation.json" download>Complete evaluation and statistics</a> | <a href="metrics.csv" download>Per-run metrics CSV</a> | <a href="paired-changes.csv" download>Paired changes CSV</a> | <a href="REPORT.md" download>Written report</a> | <a href="experiment.json" download>Experiment configuration</a></p>'
    html += '<details><summary>All paired execution traces</summary><ul>' + trace_links + '</ul></details><details><summary>Coefficients, weights and experiment scope</summary><pre>' + escape(json.dumps(result["experiment"], indent=2)) + '</pre></details></section></main></body></html>'
    (output / "efficiency-report.html").write_text(html, encoding="utf-8")
    (output / "REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    allowed = ["efficiency-report.html", "REPORT.md", "metrics.csv", "paired-changes.csv", "evaluation.json", "experiment.json", *traces]
    allowed += [info["checkpoint"] for info in result["models"].values()]
    for info in result["models"].values():
        folder = Path(info["checkpoint"]).parent
        allowed.extend((folder / filename).as_posix() for filename in ("history.json", "last-training-episode.json"))
    allowed += [p.relative_to(output).as_posix() for p in figures.glob("*") if p.suffix in (".svg", ".png")]
    # Whitelist outputs. Never archive arbitrary credentials or unrelated files.
    with zipfile.ZipFile(output / "efficiency-report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(allowed):
            archive.write(output / name, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--objectives", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run_experiment(load_run_config(args.config), load_training_config(args.training_config),
                       load_objectives(args.objectives), args.output_dir, model_seeds=tuple(args.model_seeds),
                       scenarios=tuple(args.scenarios), resume=args.resume)
    except (ValueError, TypeError, OSError, RuntimeError, OverflowError, yaml.YAMLError) as error:
        parser.error(str(error))
    print("Completed:", args.output_dir / "efficiency-report.html")


if __name__ == "__main__":
    main()
