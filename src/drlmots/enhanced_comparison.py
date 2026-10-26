"""Frozen A-034 checkpoints versus four preserved native rules (A-035).

Common workloads/outages/recovery settings; scheduler information access is reported
explicitly. No training, model selection or adaptation to this evaluation panel.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
from html import escape
import json
import math
from pathlib import Path
import platform
import zipfile

import torch
import yaml

from drlmots.enhanced_dqn import EnhancedDQN, run_enhanced_episode
from drlmots.enhanced_rl_contract import EnhancedRLContract, RewardSnapshot, reward_delta, terminal_reward_snapshot
from drlmots.enhanced_sanity import SanityPlan, network_digest, _write_json
from drlmots.normalization import _digest
from drlmots.objective_model import failure_trace, generate_work, integer, load_objective_model
from drlmots.objective_report import interval, mean
from drlmots.objective_sim import MODEL_POLICIES, simulate_objectives
from drlmots.run_config import _UniqueKeyLoader, load_run_config


POLICIES = ("untrained_dqn", "trained_dqn", *MODEL_POLICIES)
MODES = ("no_failures", "failures_with_recovery")
LABELS = dict(untrained_dqn="Untrained DQN", trained_dqn="Trained DQN (final)",
              least_queue_fcfs="Least queue", first_fit_fcfs="First fit",
              fog_first_completion="Fog-first completion", multi_objective="Multi-objective rule")
PAIRED_METRICS = ("completed_count", "total_energy_j", "successful_makespan_ms", "penalized_return")
NOTES = [
    "All six policies receive the same explicit tasks, outage trace, model, observation horizon and recovery-enabled setting. No-failure cases run once per workload, not once per failure seed.",
    "Both DQNs are frozen: epsilon=0, no optimizer/replay/counter/RNG changes. The saved A-034 initialization and final fixed-budget checkpoint are loaded, without retraining or checkpoint selection on this panel.",
    "Information access differs by implementation: DQN uses periodic delivered sensor reports/local admission cache; native rules retain their existing event-loop queue/traffic/risk estimates, which can reflect physical queue cleanup before delayed failure detection. Exogenous-input pairing does not equalize scheduler information. Interpret rule comparisons as comparisons of these implementations. Trained versus untrained DQN shares a telemetry protocol.",
    "System energy includes all hosts/links through the common horizon, partial attempts, restarts and coordinator overhead. Smaller energy or successful makespan can accompany more lost work: read all outcome counts and service feasibility together.",
    "Successful makespan excludes failed work; full-batch makespan exists only when all tasks complete. Undefined timing values are omitted from means and their contributing-run counts are recorded. No-completion surrogate belongs only to penalized return, never to reported makespan.",
    "Penalized return is recomputed for every policy with the same contract-4 costs and terminal service penalties. Native rules do not learn from it. Objective scores are valid only for feasible runs; a recommendation/Pareto candidate must satisfy both service floors in every run of its mode.",
    "Pareto screening uses the three declared objectives: energy, successful makespan and all-arrival unsuccessful fraction. It is limited to the six observed implementations on this fixed panel.",
    "Approximate 95% Student-t intervals use failure-seed means over the fixed workload panel in fault mode; no-failure intervals use workload seeds. Crossed workload/failure runs are not independent replications. Paired deltas use matched inputs. Missing timing pairs reduce the reported cluster count.",
    "This is one initialization and one fixed-budget training realization, not repeated model-seed evidence. The training budget is recorded in provenance. Intervals describe this evaluation panel, are exploratory across multiple comparisons, and do not establish general superiority or measured-hardware performance.",
    "All host/workload/failure coefficients are synthetic. Compiled HTML, local figures and data need no JavaScript, templates, external fonts or server. Share the ZIP file beside the report; model checkpoints are referenced by hashes, not bundled.",
]


@dataclass(frozen=True)
class ComparisonPlan:
    workload_seeds: tuple[int, ...] = (82000, 82001, 82002)
    failure_seeds: tuple[int, ...] = (92000, 92001, 92002, 92003, 92004)

    def __post_init__(self):
        for name in ("workload_seeds", "failure_seeds"):
            seeds = tuple(getattr(self, name))
            if not seeds or len(seeds) != len(set(seeds)):
                raise ValueError("comparison seeds must be nonempty and distinct")
            for seed in seeds:
                integer(seed, name)
            object.__setattr__(self, name, seeds)

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)} | {"schema_version", "kind"}:
            raise ValueError("comparison plan requires exactly its documented fields")
        data = dict(data)
        version, kind = data.pop("schema_version"), data.pop("kind")
        if type(version) is not int or version != 1 or kind != "frozen_enhanced_dqn_comparison":
            raise ValueError("unsupported comparison plan kind/version")
        return cls(**data)

    def manifest(self):
        return dict(schema_version=1, kind="frozen_enhanced_dqn_comparison", **asdict(self))

    def check_holdout(self, training):
        for seeds, start, diagnostics in (
            (self.workload_seeds, training.training_workload_seed_start,
             (*training.evaluation_workload_seeds, training.recovery_probe_workload_seed)),
            (self.failure_seeds, training.training_failure_seed_start, training.evaluation_failure_seeds)):
            if any(start <= value < start + training.training_episodes or value in diagnostics for value in seeds):
                raise ValueError("comparison panel must be disjoint from training and sanity diagnostic seeds")


def load_frozen_pair(run, model, training_dir, plan):
    directory = Path(training_dir)
    provenance = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if provenance.get("kind") != "enhanced_dqn_training_sanity" or provenance.get("status") != "complete" or not result.get("learning_updates_verified"):
        raise ValueError("completed verified A-034 training provenance required")
    training = SanityPlan.parse(provenance["plan"])
    plan.check_holdout(training)
    if provenance["run_configuration"] != json.loads(json.dumps(run.manifest())) or provenance["model_configuration"] != model.manifest():
        raise ValueError("evaluation topology/model differs from the trained environment")
    source = Path(__file__).parent
    required_sources = {"enhanced_sanity.py", "enhanced_dqn.py", "enhanced_rl_contract.py", "enhanced_telemetry.py", "objective_sim.py", "objective_model.py"}
    if set(provenance["source_sha256"]) != required_sources:
        raise ValueError("training provenance must cover all six implementation sources")
    for name, expected in provenance["source_sha256"].items():
        if name not in ("enhanced_sanity.py", "enhanced_dqn.py", "enhanced_rl_contract.py", "enhanced_telemetry.py", "objective_sim.py", "objective_model.py"):
            raise ValueError("unknown training source in provenance")
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("training implementation differs from the verified evaluation runtime")
    contract = EnhancedRLContract.from_configs(run, model)
    _, hosts = model.validate_topology(run)
    pair = {name: EnhancedDQN.load(directory / filename, contract, host_nodes=hosts)
            for name, filename in (("untrained_dqn", "initialization.json"), ("trained_dqn", "trained-final.json"))}
    initial, trained = pair.values()
    if initial.settings != trained.settings or not trained.settings.recovery_enabled:
        raise ValueError("matching recovery-enabled frozen DQN settings required")
    if (initial.decisions or initial.updates or initial.episodes or initial.replay or initial.losses
        or network_digest(initial.online) != network_digest(initial.target)
        or network_digest(initial.online) != network_digest(EnhancedDQN(contract, initial.settings, host_nodes=hosts).online)):
        raise ValueError("initialization checkpoint is not the untrained seeded network")
    if (trained.updates <= 0 or trained.episodes != training.training_episodes
        or trained.updates != result["optimizer_updates"]
        or initial.fingerprint != result["initial_fingerprint"] or trained.fingerprint != result["final_fingerprint"]):
        raise ValueError("checkpoints do not match the declared initialization/final training result")
    if asdict(trained.settings) != {**provenance["dqn_configuration"], "hidden_widths": tuple(provenance["dqn_configuration"]["hidden_widths"])}:
        raise ValueError("checkpoint settings differ from the training record")
    metadata = {name: dict(fingerprint=agent.fingerprint, online_sha256=network_digest(agent.online),
        updates=agent.updates, episodes=agent.episodes, decisions=agent.decisions,
        environment_sha256=agent.environment_sha256) for name, agent in pair.items()}
    return pair, contract, dict(training_plan=training.manifest(), checkpoints=metadata,
                               training_provenance_sha256=_digest(provenance), training_result_sha256=_digest(result))


def common_return(contract, report):
    measured = reward_delta(contract, RewardSnapshot(), terminal_reward_snapshot(contract, report), terminated=True).total
    if "rl" in report and not math.isclose(measured, report["rl"]["total_reward"], abs_tol=1e-9):
        raise RuntimeError("frozen DQN return disagrees with common objective accounting")
    return measured


def _clusters(rows, mode, key):
    grouped = {}
    for row in rows:
        value = row["metrics"][key]
        if value is not None:
            cluster = row["workload_seed"] if mode == "no_failures" else row["failure_seed"]
            grouped.setdefault(cluster, []).append(value)
    return [mean(values) for _, values in sorted(grouped.items())]


def summarize(rows):
    summaries = []
    for mode in MODES:
        for policy in POLICIES:
            group = [r for r in rows if r["mode"] == mode and r["policy"] == policy]
            if not group:
                raise ValueError("comparison missing a policy/mode group")
            keys = group[0]["metrics"]
            sums = {k: sum(r["metrics"][k] for r in group) for k in
                    ("submitted_count", "completed_count", "deadline_eligible_count", "deadline_success_count",
                     "interrupted_task_count", "recovered_task_count")}
            ratio = lambda a, b: sums[a] / sums[b] if sums[b] else None
            eligible = all(r["objective"]["feasible"] for r in group)
            summaries.append(dict(mode=mode, policy=policy, run_count=len(group),
                metrics={k: mean(r["metrics"][k] for r in group) for k in keys},
                metric_contributing_runs={k: sum(r["metrics"][k] is not None for r in group) for k in keys},
                uncertainty={k: interval(_clusters(group, mode, k)) for k in PAIRED_METRICS},
                pooled_rates=dict(completion=ratio("completed_count", "submitted_count"),
                    deadline_success=ratio("deadline_success_count", "deadline_eligible_count"),
                    recovery_success=ratio("recovered_task_count", "interrupted_task_count")),
                feasible_run_count=sum(r["objective"]["feasible"] for r in group),
                eligible_for_recommendation=eligible,
                objective_score_mean=mean(r["objective"]["score"] for r in group) if eligible else None,
                fully_completed_run_count=sum(r["metrics"]["batch_makespan_ms"] is not None for r in group)))
    for row in summaries:
        def vector(r):
            return [r["metrics"][k] for k in ("total_energy_j", "successful_makespan_ms", "observed_unsuccessful_fraction")]
        row["pareto_efficient"] = row["eligible_for_recommendation"] and not any(
            other["mode"] == row["mode"] and other["eligible_for_recommendation"]
            and all(a <= b for a, b in zip(vector(other), vector(row)))
            and any(a < b for a, b in zip(vector(other), vector(row))) for other in summaries)
    return summaries


def paired_effects(rows):
    effects = []
    lookup = {(r["mode"], r["workload_seed"], r["failure_seed"], r["policy"]): r for r in rows}
    for mode in MODES:
        trained = [r for r in rows if r["mode"] == mode and r["policy"] == "trained_dqn"]
        for baseline in (p for p in POLICIES if p != "trained_dqn"):
            pairs = []
            for row in trained:
                other = lookup[(mode, row["workload_seed"], row["failure_seed"], baseline)]
                if row["workload_sha256"] != other["workload_sha256"] or row["outage_sha256"] != other["outage_sha256"]:
                    raise RuntimeError("paired input hashes disagree")
                pairs.append(dict(workload_seed=row["workload_seed"], failure_seed=row["failure_seed"],
                    metrics={k: row["metrics"][k] - other["metrics"][k]
                             if row["metrics"][k] is not None and other["metrics"][k] is not None else None
                             for k in PAIRED_METRICS}))
            effects.append(dict(mode=mode, comparison="trained_dqn minus " + baseline, baseline=baseline,
                pairs=pairs, uncertainty={k: interval(_clusters(pairs, mode, k)) for k in PAIRED_METRICS}))
    return effects


def compare_frozen(run, model, plan, training_dir, output_dir, *, render_figures=True):
    if type(plan) is not ComparisonPlan:
        raise ValueError("typed comparison plan required")
    if render_figures:
        import matplotlib  # Fail before simulations if the report extra is absent.
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("choose an empty output directory; comparisons never overwrite prior evidence")
    agents, contract, provenance = load_frozen_pair(run, model, training_dir, plan)
    before = {p: a.fingerprint for p, a in agents.items()}
    sources = ("enhanced_comparison.py", "enhanced_dqn.py", "enhanced_rl_contract.py", "enhanced_telemetry.py", "objective_model.py", "objective_sim.py", "objective_report.py")
    manifest = dict(kind="frozen_enhanced_six_policy_study", status="running", plan=plan.manifest(),
        model_configuration=model.manifest(), run_configuration=run.manifest(), provenance=provenance,
        policies=POLICIES, modes=MODES, recovery_enabled=True, notes=NOTES,
        implementation_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
        runtime=dict(python=platform.python_version(), torch=torch.__version__, device="cpu", threads=1))
    _write_json(output / "experiment.json", manifest)
    rows = []
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for wseed in plan.workload_seeds:
            tasks = generate_work(model, wseed)
            specs = [("no_failures", None, [])] + [("failures_with_recovery", f, failure_trace(model, f)) for f in plan.failure_seeds]
            for mode, fseed, trace in specs:
                for policy in POLICIES:
                    if policy in agents:
                        report = run_enhanced_episode(agents[policy], run, model, tasks=tasks, outages=trace, failure_seed=fseed or 0)
                        if report["rl"]["learn"] or any(d["epsilon"] != 0 for d in report["rl"]["decisions"]):
                            raise RuntimeError("DQN evaluation is not frozen greedy inference")
                    else:
                        report = simulate_objectives(run, model, policy=policy, tasks=tasks, outages=trace,
                                                     failure_seed=fseed or 0, recovery_enabled=True)
                    value = common_return(contract, report)
                    native_digest = _digest(report)
                    if fseed is not None:
                        for availability in report["availability"]:
                            availability["theoretical_applies_to_trace"] = True
                    report.update(policy=policy, mode=mode, workload_seed=wseed, failure_seed=fseed,
                                  comparison_implementation_sha256=manifest["implementation_sha256"])
                    path = f"runs/work-{wseed}-fault-{fseed}-{mode}-{policy}.json"
                    _write_json(output / path, report, compact=True)
                    rows.append(dict(policy=policy, mode=mode, workload_seed=wseed, failure_seed=fseed,
                        trace_file=path, workload_sha256=report["workload"]["sha256"], outage_sha256=report["outage_trace"]["sha256"],
                        original_report_sha256=native_digest, metrics={**report["metrics"], "penalized_return": value},
                        objective=report["objective"], availability=report["availability"]))
            print(f"Evaluated workload {wseed}: {len(specs) * len(POLICIES)} simulations", flush=True)
        after = {p: a.fingerprint for p, a in agents.items()}
        if before != after:
            raise RuntimeError("comparison changed a frozen checkpoint")
        for wseed in plan.workload_seeds:
            group = [r for r in rows if r["workload_seed"] == wseed]
            if len({r["workload_sha256"] for r in group}) != 1:
                raise RuntimeError("workloads differ within the paired panel")
            for mode, fseed, _ in [("no_failures", None, [])] + [("failures_with_recovery", f, []) for f in plan.failure_seeds]:
                paired = [r for r in group if r["mode"] == mode and r["failure_seed"] == fseed]
                if len(paired) != len(POLICIES) or len({r["outage_sha256"] for r in paired}) != 1:
                    raise RuntimeError("failure scenario differs between policies")
        result = dict(kind="frozen_enhanced_six_policy_results", rows=rows, summaries=summarize(rows),
                      paired_comparisons=paired_effects(rows), frozen_audit=dict(before=before, after=after, unchanged=before == after))
        _write_json(output / "evaluation.json", result)
        _write_json(output / "frozen-audit.json", result["frozen_audit"])
        for filename, records in (("metrics.csv", [{**{k: r[k] for k in ("policy", "mode", "workload_seed", "failure_seed")},
                                                   **r["metrics"], "feasible": r["objective"]["feasible"], "score": r["objective"]["score"]} for r in rows]),
                                  ("paired-deltas.csv", [dict(mode=e["mode"], baseline=e["baseline"], workload_seed=p["workload_seed"],
                                       failure_seed=p["failure_seed"], **p["metrics"]) for e in result["paired_comparisons"] for p in e["pairs"]])):
            with (output / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        if render_figures:
            figures(output, result, model.data["workload"]["task_count"])
        render(output, result, manifest)
        _write_json(output / "experiment.json", {**manifest, "status": "complete", "frozen_verified": True})
        bundle(output, result)
        return result
    finally:
        torch.set_num_threads(original_threads)


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def _ci(summary):
    bounds = summary["ci95"]
    return _fmt(summary["mean"]) + (f" [{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else " [CI undefined]") + f"; n={summary['n']}"


def figures(output, result, task_count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    colors = ["#87939f", "#2465a1", "#297c68", "#986b29", "#8063a0", "#1b8d8d"]
    for mode in MODES:
        group = [r for r in result["summaries"] if r["mode"] == mode]
        labels = [LABELS[r["policy"]] for r in group]
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), layout="constrained")
        left = [0.] * len(group)
        for key, label, color in (("completed_count", "Delivered", "#247f70"), ("rejected_count", "Rejected", "#bd9041"),
            ("execution_failure_count", "Execution failure", "#b64d4d"), ("deadline_missed_count", "Deadline missed", "#8667ac"),
            ("unfinished_count", "Unfinished", "#9b9fa2")):
            values = [r["metrics"][key] for r in group]
            axes[0].barh(labels, values, left=left, color=color, label=label)
            left = [a + b for a, b in zip(left, values)]
        axes[0].set(title=f"All outcomes (of {task_count})", xlim=(0, task_count))
        for ax, key, title in ((axes[1], "total_energy_j", "System energy (J)"), (axes[2], "successful_makespan_ms", "Successful makespan (ms)")):
            for i, row in enumerate(group):
                value = row["metrics"][key]
                if value is None:
                    ax.text(0, i, "undefined", va="center")
                else:
                    ax.barh(i, value, color=colors[i])
                    bounds = row["uncertainty"][key]["ci95"]
                    if bounds:
                        ax.errorbar(value, i, xerr=[[max(0, value - bounds[0])], [max(0, bounds[1] - value)]], fmt="none", color="#263442", capsize=3)
            ax.set(title=title, yticks=range(len(group)), yticklabels=[])
        axes[3].barh(range(len(group)), [r["feasible_run_count"] / r["run_count"] for r in group], color=colors)
        axes[3].set(title="Service-feasible run fraction", xlim=(0, 1.08), yticks=range(len(group)), yticklabels=[])
        for i, row in enumerate(group):
            axes[3].text(.03, i, f"{row['feasible_run_count']}/{row['run_count']}", va="center", color="#142c3b")
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=5, fontsize=9)
        fig.suptitle(mode.replace("_", " ") + ": scenario means; native rule and DQN information access differ")
        for ext in ("png", "svg"):
            fig.savefig(directory / f"{mode}.{ext}", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    chosen = [e for e in result["paired_comparisons"] if e["baseline"] == "untrained_dqn"]
    for column, (ax, key, label) in enumerate(zip(axes, PAIRED_METRICS[:3], ("Completed tasks\nPositive favors trained", "Energy (J)\nNegative favors trained", "Successful makespan (ms)\nRead completion counts too"))):
        for i, effect in enumerate(chosen):
            u = effect["uncertainty"][key]
            if u["mean"] is not None:
                ax.plot(u["mean"], i, "o", color="#2465a1")
                if u["ci95"]:
                    ax.plot(u["ci95"], [i, i], color="#2465a1")
        ax.axvline(0, color="#88929a", linestyle="--")
        ax.set(title=label, yticks=range(len(chosen)), ylim=(-.4, len(chosen) - .6),
               yticklabels=[e["mode"].replace("_", " ") for e in chosen] if column == 0 else [])
        ax.grid(alpha=.2)
    fig.suptitle("Trained minus untrained: matching telemetry, paired approximate 95% intervals")
    for ext in ("png", "svg"):
        fig.savefig(directory / f"trained-minus-untrained.{ext}", dpi=150)
    plt.close(fig)


def render(output, result, manifest):
    checkpoints = manifest["provenance"]["checkpoints"]
    checkpoint_description = "; ".join(f"{LABELS[name]}: {data['episodes']} prior episodes, {data['updates']} prior updates"
                                       for name, data in checkpoints.items()) + ". Both accumulated zero evaluation updates."
    md = ["# Frozen untrained/trained DQN and four native heuristics (A-035)", "",
          f"Simulations: {len(result['rows'])}; six policies; checkpoints frozen throughout.", "",
          checkpoint_description, "", *["- " + n for n in NOTES], ""]
    blocks = []
    for mode in MODES:
        group = [r for r in result["summaries"] if r["mode"] == mode]
        headers = ["Policy", "Done", "Rejected", "Exec fail", "Deadline miss", "Unfinished", "Total J", "Successful M ms", "Mean latency ms", "Return", "Feasible runs", "Full batches", "Pareto member"]
        values = []
        md += ["## " + mode, "", "Numeric metrics are means over scenarios; final columns count runs or mark empirical Pareto membership.", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        for row in group:
            m = row["metrics"]
            cells = [LABELS[row["policy"]]] + [_fmt(m[k]) for k in ("completed_count", "rejected_count", "execution_failure_count", "deadline_missed_count", "unfinished_count", "total_energy_j", "successful_makespan_ms", "successful_mean_latency_ms", "penalized_return")] + [f"{row['feasible_run_count']}/{row['run_count']}", f"{row['fully_completed_run_count']}/{row['run_count']}", str(row["pareto_efficient"])]
            md.append("| " + " | ".join(cells) + " |")
            values.append(cells)
        image = f'<img src="figures/{mode}.svg" alt="Paired outcomes, energy, successful makespan and service feasibility">' if (output / f"figures/{mode}.svg").exists() else ""
        table = '<div class="scroll"><table><thead><tr>' + "".join("<th>" + escape(h) + "</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + escape(c) + "</td>" for c in cells) + "</tr>" for cells in values) + "</tbody></table></div>"
        timing_support = "; ".join(f"{LABELS[r['policy']]}: {r['metric_contributing_runs']['successful_makespan_ms']}/{r['run_count']} with successful makespan" for r in group)
        reliability = []
        for r in group:
            reliability.append("<tr><td>" + escape(LABELS[r["policy"]]) + "</td>" + "".join("<td>" + _fmt(r["pooled_rates"][k]) + "</td>" for k in ("completion", "deadline_success", "recovery_success")) + "</tr>")
        blocks.append('<section><h2>' + mode.replace("_", " ") + '</h2><p>Numeric metrics are means over scenarios; final columns count runs or mark empirical Pareto membership.</p>' + image + table + '<p>' + escape(timing_support) + '</p><div class="scroll"><table><tr><th>Policy</th><th>Pooled completion</th><th>Pooled deadline success</th><th>Pooled recovery success</th></tr>' + "".join(reliability) + '</table></div></section>')
        md += ["", "Timing support: " + timing_support, ""]
    paired_rows = []
    md += ["## Trained minus each baseline: paired deltas", "", "Each bracket is an approximate 95% interval; n counts independent clusters, not crossed runs.", ""]
    for effect in result["paired_comparisons"]:
        cells = [effect["mode"], LABELS[effect["baseline"]]] + [_ci(effect["uncertainty"][key]) for key in PAIRED_METRICS]
        paired_rows.append("<tr>" + "".join("<td>" + escape(c) + "</td>" for c in cells) + "</tr>")
        md.append("- " + " | ".join(cells))
    paired_image = '<img src="figures/trained-minus-untrained.svg" alt="Paired trained versus untrained differences with approximate intervals">' if (output / "figures/trained-minus-untrained.svg").exists() else ""
    blocks.append('<section><h2>Paired changes: trained minus baseline</h2><p>Positive completion/return changes favor trained; negative energy changes favor trained. Read makespan with completion counts. Intervals are exploratory, conditional on the declared workload panel in fault mode.</p>' + paired_image + '<div class="scroll"><table><tr><th>Mode</th><th>Baseline</th><th>Done delta</th><th>Energy delta J</th><th>Successful makespan delta ms</th><th>Return delta</th></tr>' + "".join(paired_rows) + '</table></div></section>')
    candidates = {mode: [LABELS[r["policy"]] for r in result["summaries"] if r["mode"] == mode and r["pareto_efficient"]] for mode in MODES}
    md += ["", "## Empirical Pareto candidates, after all-run service screening", "", *[mode + ": " + (", ".join(names) or "none") for mode, names in candidates.items()], "", "## Frozen audit", "", json.dumps(result["frozen_audit"], indent=2), ""]
    (output / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    candidate_text = " ".join(mode.replace("_", " ") + ": " + (", ".join(names) or "none") + "." for mode, names in candidates.items())
    links = "".join('<li><a href="' + r["trace_file"] + '">' + escape(r["trace_file"]) + '</a></li>' for r in result["rows"])
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DRLMOTS frozen six-policy comparison</title><style>body{font:16px/1.5 system-ui;background:#f1f5f5;color:#193040;max-width:1550px;margin:auto;padding:24px}h1{line-height:1.15}section{background:white;border:1px solid #cfdddd;border-radius:12px;padding:22px;margin:22px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:9px;border-bottom:1px solid #dce5e6;text-align:left;white-space:nowrap}img{max-width:100%;height:auto}a{color:#096e75}li{margin:9px 0}.lead{font-size:1.1rem}@media(max-width:700px){body{padding:12px}section{padding:14px}}</style></head><body><h1>Frozen DQN and four reference schedulers</h1><p class="lead">' + str(len(result["rows"])) + ' synthetic simulations. One untrained initialization, one fixed-budget trained checkpoint, six preserved policy paths.</p><p><a href="metrics.csv" download>Metrics CSV</a> · <a href="paired-deltas.csv" download>Paired deltas CSV</a> · <a href="evaluation.json">Results JSON</a> · <a href="REPORT.md">Markdown report</a> · <a href="experiment.json">Experiment/provenance</a> · <a href="frozen-audit.json">Frozen audit</a> · <a href="artifacts.json">Evidence hashes</a></p><p>Share comparison-report.zip, stored beside this page. Checkpoints are referenced by hashes and can be regenerated with the A-034 command.</p><section><h2>How to read this comparison</h2><ul>' + "".join('<li>' + escape(n) + '</li>' for n in NOTES) + '</ul></section>' + "".join(blocks) + '<section><h2>Eligible empirical tradeoffs</h2><p>' + escape(candidate_text) + '</p><p>These are observed tradeoffs for these implementations, conditional on the stated information access and panel. Policies failing any service-constrained run are excluded from recommendation.</p></section><section><details><summary>Every simulation trace</summary><ul>' + links + '</ul></details></section></body></html>'
    html = html.replace('<section><h2>How to read this comparison</h2>',
        '<section><h2>Frozen checkpoint history</h2><p>' + escape(checkpoint_description) +
        '</p><p>The final training episode was fixed before this panel. The full fingerprints and training record are in experiment.json and frozen-audit.json.</p></section><section><h2>How to read this comparison</h2>')
    (output / "comparison-report.html").write_text(html, encoding="utf-8")


def bundle(output, result):
    names = ["experiment.json", "evaluation.json", "frozen-audit.json", "metrics.csv", "paired-deltas.csv", "REPORT.md", "comparison-report.html"]
    names += [r["trace_file"] for r in result["rows"]]
    names += [str(p.relative_to(output)).replace("\\", "/") for p in sorted((output / "figures").glob("*"))] if (output / "figures").exists() else []
    _write_json(output / "artifacts.json", dict(kind="frozen_comparison_evidence", sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}))
    with zipfile.ZipFile(output / "comparison-report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*names, "artifacts.json"]:
            archive.write(output / name, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl_network.yaml")
    parser.add_argument("--model-config", default="configs/objective_model.yaml")
    parser.add_argument("--plan-config", default="configs/enhanced_comparison.yaml")
    parser.add_argument("--training-dir", default="outputs/objective-model/dqn-sanity/milestone4-20261027")
    parser.add_argument("--output-dir", default="outputs/objective-model/frozen-comparison/milestone5-20261027-release")
    args = parser.parse_args(argv)
    try:
        plan = ComparisonPlan.parse(yaml.load(Path(args.plan_config).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader))
        result = compare_frozen(load_run_config(args.config), load_objective_model(args.model_config), plan, args.training_dir, args.output_dir)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"frozen comparison: {error}\n")
    print(f"Saved {args.output_dir}; simulations={len(result['rows'])}; both checkpoint fingerprints unchanged")


if __name__ == "__main__":
    main()
