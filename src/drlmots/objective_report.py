"""Paired workload x independent failure-seed study for objective contract A-030."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import statistics
import zipfile

from drlmots.normalization import _digest
from drlmots.objective_model import failure_trace, generate_work, integer, load_objective_model
from drlmots.objective_sim import MODEL_POLICIES, simulate_objectives
from drlmots.run_config import load_run_config

MODES = ("no_failures", "failures_no_recovery", "failures_with_recovery")
DISPLAY = (("completed_count", "Completed"), ("rejected_count", "Rejected"),
           ("execution_failure_count", "Execution failure"), ("deadline_missed_count", "Deadline missed"),
           ("unfinished_count", "Unfinished"), ("successful_makespan_ms", "Successful makespan ms"),
           ("successful_mean_latency_ms", "Successful mean ms"),
           ("total_energy_j", "Total J"), ("energy_per_completed_task_j", "J/completed"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def interval(values):
    """Student t interval across independent failure traces, conditional on workload panel."""
    values = list(values)
    n = len(values)
    avg = mean(values)
    if n < 2:
        return dict(mean=avg, n=n, std=None, ci95=None)
    # Exact common small-df quantiles, conservative for larger df (df=30 or normal limit).
    quantiles = {1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262,
                 10:2.228, 15:2.131, 20:2.086, 30:2.043}
    df = n-1
    key = max(k for k in quantiles if k <= min(df, 30))
    std = statistics.stdev(values)
    radius = quantiles[key] * std / n**.5
    return dict(mean=avg, n=n, std=std, ci95=[avg-radius, avg+radius])


def summarize(rows):
    summaries = []
    for mode in MODES:
        for policy in MODEL_POLICIES:
            group = [r for r in rows if r["mode"] == mode and r["policy"] == policy]
            metrics = {k: mean(r["metrics"][k] for r in group) for k in group[0]["metrics"]}
            # Reusing one failure seed across workloads creates a cluster, not extra independent traces.
            clusters = {}
            for row in group:
                key = row["workload_seed"] if mode == "no_failures" else row["failure_seed"]
                clusters.setdefault(key, []).append(row)
            uncertainty = {k: interval(v for g in clusters.values() if (v := mean(r["metrics"][k] for r in g)) is not None)
                           for k in ("completed_count", "total_energy_j", "successful_mean_latency_ms")}
            summaries.append(dict(mode=mode, policy=policy, metrics=metrics, uncertainty=uncertainty,
                run_count=len(group), fully_completed_run_count=sum(r["metrics"]["batch_makespan_ms"] is not None for r in group),
                eligible_for_recommendation=all(r["objective"]["feasible"] for r in group),
                feasible_run_fraction=sum(r["objective"]["feasible"] for r in group)/len(group),
                mean_feasible_score=mean(r["objective"]["score"] for r in group),
                mean_observed_availability=mean(r["mean_observed_availability"] for r in group)))
    # Pareto filtering uses energy, successful mean latency and all-arrival unsuccessful fraction,
    # only where every run satisfies the predeclared service floors.
    for row in summaries:
        vector = lambda r: [r["metrics"][k] for k in ("total_energy_j", "successful_mean_latency_ms", "observed_unsuccessful_fraction")]
        row["pareto_efficient"] = row["eligible_for_recommendation"] and not any(
            other["mode"] == row["mode"] and other["eligible_for_recommendation"]
            and all(a <= b for a, b in zip(vector(other), vector(row)))
            and any(a < b for a, b in zip(vector(other), vector(row))) for other in summaries)
    effects = []
    for policy in MODEL_POLICIES:
        paired = []
        for on in rows:
            if on["policy"] != policy or on["mode"] != "failures_with_recovery":
                continue
            off = next(r for r in rows if r["policy"] == policy and r["mode"] == "failures_no_recovery" and r["workload_seed"] == on["workload_seed"] and r["failure_seed"] == on["failure_seed"])
            paired.append(dict(failure_seed=on["failure_seed"], workload_seed=on["workload_seed"],
                completed_delta=on["metrics"]["completed_count"]-off["metrics"]["completed_count"],
                energy_delta_j=on["metrics"]["total_energy_j"]-off["metrics"]["total_energy_j"]))
        effects.append(dict(policy=policy, pairs=paired,
            uncertainty={k: interval(mean(r[k] for r in paired if r["failure_seed"] == seed)
                for seed in sorted({r["failure_seed"] for r in paired})) for k in ("completed_delta", "energy_delta_j")}))
    return summaries, effects


def compare_objective_model(config, model, workload_seeds, failure_seeds, output_dir):
    import matplotlib  # Check plotting availability before writing the experiment.
    ws, fs = tuple(workload_seeds), tuple(failure_seeds)
    for name, seeds in (("workload", ws), ("failure", fs)):
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{name} seeds must be nonempty and distinct")
        for seed in seeds:
            integer(seed, name+" seed")
    model.validate_topology(config)
    source_dir = Path(__file__).parent
    implementation = {name: _digest((source_dir/name).read_text(encoding="utf-8"))
                      for name in ("objective_model.py", "objective_sim.py", "objective_report.py")}
    plan = dict(kind="paired_host_network_objective_study", implementation_sha256=implementation, model_configuration=model.manifest(),
                run_configuration=config.manifest(), workload_seeds=ws, failure_seeds=fs,
                policies=MODEL_POLICIES, modes=MODES,
                uncertainty="Fault modes: Student t interval of failure-seed means over the fixed workload panel; no-failure mode: workload seeds. Approximate small-sample intervals, no general superiority claim.")
    output = Path(output_dir)
    if (output/"experiment.json").exists() and _digest(json.loads((output/"experiment.json").read_text(encoding="utf-8"))) != _digest(plan):
        raise ValueError("output contains a different plan; choose a fresh directory")
    write_json(output/"experiment.json", plan)
    rows, sample = [], None
    for wseed in ws:
        tasks = generate_work(model, wseed)
        # No-failure runs happen once per workload, never duplicated across failure seeds.
        specs = [("no_failures", None, [], False)] + [(mode, fseed, failure_trace(model, fseed), mode == "failures_with_recovery")
                 for fseed in fs for mode in MODES[1:]]
        for mode, fseed, trace, enabled in specs:
            for policy in MODEL_POLICIES:
                report = simulate_objectives(config, model, policy=policy, tasks=tasks,
                    failure_seed=fseed or 0, outages=trace, recovery_enabled=enabled)
                report.update(mode=mode, workload_seed=wseed, failure_seed=fseed, implementation_sha256=implementation)
                # Trace was generated by the experiment, rather than the simulator, with the same configured process.
                if fseed is not None:
                    for availability in report["availability"]:
                        availability["theoretical_applies_to_trace"] = True
                path = f"runs/work-{wseed}-fault-{fseed}-{mode}-{policy}.json"
                write_json(output/path, report)
                rows.append(dict(workload_seed=wseed, failure_seed=fseed, mode=mode, policy=policy,
                    trace_file=path, workload_sha256=report["workload"]["sha256"], outage_sha256=report["outage_trace"]["sha256"],
                    metrics=report["metrics"], objective=report["objective"], availability=report["availability"],
                    mean_observed_availability=mean(a["observed_availability"] for a in report["availability"])))
                if sample is None and mode == "failures_with_recovery" and policy == "multi_objective":
                    sample = report
        print(f"Completed workload {wseed}: {len(specs)*len(MODEL_POLICIES)} simulations", flush=True)
    for wseed in ws:
        group = [r for r in rows if r["workload_seed"] == wseed]
        if len({r["workload_sha256"] for r in group}) != 1:
            raise RuntimeError("paired workloads differ")
        for fseed in fs:
            faulted = [r for r in group if r["failure_seed"] == fseed]
            if len({r["outage_sha256"] for r in faulted}) != 1:
                raise RuntimeError("paired outage traces differ")
    summaries, effects = summarize(rows)
    result = dict(experiment=plan, rows=rows, summaries=summaries, paired_recovery_effects=effects)
    write_json(output/"evaluation.json", result)
    render(output, result, sample)
    return result


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def render(output, result, sample):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_names = []
    for mode in MODES:
        group = [r for r in result["summaries"] if r["mode"] == mode]
        labels = [r["policy"].replace("_", " ") for r in group]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        left = [0.0]*len(group)
        for key, label, color in (("completed_count", "Delivered", "#167e75"), ("rejected_count", "Rejected", "#c49842"),
                 ("execution_failure_count", "Execution failure", "#b74d4d"), ("deadline_missed_count", "Deadline missed", "#8d6bb5"),
                 ("unfinished_count", "Unfinished", "#89959d")):
            values = [r["metrics"][key] for r in group]
            axes[0].barh(labels, values, left=left, label=label, color=color)
            left = [a+b for a, b in zip(left, values)]
        for ax, key, title in ((axes[1], "total_energy_j", "Fixed-window system energy (J)"),
                                (axes[2], "successful_mean_latency_ms", "Successful latency (ms)")):
            ax.barh(labels, [r["metrics"][key] or 0 for r in group], color="#167e75")
            ax.set(title=title)
            ax.tick_params(axis="y", labelleft=False)
        axes[0].set(title="Every submitted task has an outcome")
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.15)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=5, fontsize=9)
        fig.suptitle(mode.replace("_", " ") + " — synthetic model; means over paired runs")
        for ext in ("svg", "png"):
            fig.savefig(figures/(mode+"."+ext), dpi=150)
        plt.close(fig)
        plot_names.append(mode)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained", sharex=True)
    vm_ids = sorted(sample["model_configuration"]["vms"])
    for a in sample["attempts"]:
        y = vm_ids.index(a["vm_id"])
        axes[0].broken_barh([(a["dispatch_ms"], a["end_ms"]-a["dispatch_ms"])], (y-.35, .7),
                           facecolors="#167e75" if a["status"] == "completed" else "#b74d4d", alpha=.5)
        if a["start_ms"] is not None:
            end = a["compute_end_ms"] or a["end_ms"]
            axes[0].broken_barh([(a["start_ms"], end-a["start_ms"])], (y-.2, .4), facecolors="#132e43")
    axes[0].set(yticks=range(len(vm_ids)), yticklabels=vm_ids, title="One multi-objective run: attempts (green/red), computation (dark)")
    for host in sample["host_energy"]:
        segments = [s for s in sample["power_segments"] if s["host_id"] == host]
        xs = [s["start_ms"] for s in segments] + [segments[-1]["end_ms"]]
        ys = [s["power_w"] for s in segments] + [segments[-1]["power_w"]]
        axes[1].step(xs, ys, where="post", label=host)
    axes[1].set(title="Physical host power: idle, compute, repair and reboot", xlabel="Simulation time (ms)", ylabel="W")
    axes[1].legend()
    for ext in ("svg", "png"):
        fig.savefig(figures/("timeline-power."+ext), dpi=150)
    plt.close(fig)
    plot_names.append("timeline-power")
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        names = ["workload_seed", "failure_seed", "mode", "policy"] + list(result["rows"][0]["metrics"]) + ["feasible", "score"]
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for r in result["rows"]:
            writer.writerow({**{k:r[k] for k in names[:4]}, **r["metrics"], "feasible":r["objective"]["feasible"], "score":r["objective"]["score"]})
    notes = ["Explicit work in MI and VM speed in MIPS; input/output sizes are separate decimal kB.",
        "Host power is integrated once over a common cutoff, including idle, dynamic, down and reboot states. Network idle and actual bytes in both directions are included.",
        "Shared half-duplex links divide bandwidth equally among active transfers. Propagation consumes elapsed time without holding bandwidth.",
        "Node failures use independent renewal streams with exponential or Weibull uptime, deterministic repair/reboot and perfect age reset. All member VMs fail together.",
        "Detection and restart consume coordinator energy. A stale health view may dispatch into a down node. Future failure/repair timestamps are hidden from placement.",
        "A task completes only when its output arrives. Fully serialized output survives subsequent host failure while propagating; partial output is lost. No checkpointing: recovery repeats full input and compute on a different node; the durable source is assumed reliable.",
        "Unfinished at cutoff is censored, not automatically execution failure. Batch makespan is defined only when every task completes; successful makespan excludes losses.",
        "Deadline success denominator includes only tasks whose absolute deadline has matured by cutoff. Rejected and failed tasks in that set count against it.",
        "Weighted scores are linear fixed-scale ratios without clipping. A run below either service floor has no score. Only policies feasible on every run enter the Pareto recommendation.",
        "This A-030 report evaluates four native heuristics using causal predictions. Frozen enhanced DQN comparisons are available separately through enhanced_comparison; legacy DQN models remain separate.",
        "All coefficients are synthetic. Curves can accept measurements, but no measured calibration or real VM deployment is claimed.", result["experiment"]["uncertainty"]]
    md = ["# Host/network multi-objective study (A-030)", "", *["- "+n for n in notes], "",
          f"Simulations: {len(result['rows'])}; workload seeds: {list(result['experiment']['workload_seeds'])}; failure seeds: {list(result['experiment']['failure_seeds'])}.", ""]
    blocks = []
    for mode in MODES:
        table = ["| Policy | " + " | ".join(label for _, label in DISPLAY) + " | Feasible runs | Pareto eligible | Full batches |",
                 "|---|" + "---:|"*(len(DISPLAY)+3)]
        body = []
        for r in result["summaries"]:
            if r["mode"] != mode:
                continue
            values = [fmt(r["metrics"][k]) for k, _ in DISPLAY]
            table.append("| "+r["policy"]+" | "+" | ".join(values)+f" | {r['feasible_run_fraction']:.1%} | {r['pareto_efficient']} | {r['fully_completed_run_count']}/{r['run_count']} |")
            body.append("<tr><th>"+escape(r["policy"])+"</th>"+"".join("<td>"+v+"</td>" for v in values)+f"<td>{r['feasible_run_fraction']:.1%}</td><td>{r['pareto_efficient']}</td><td>{r['fully_completed_run_count']}/{r['run_count']}</td></tr>")
        md += ["## "+mode, "", *table, ""]
        headers = "<tr><th>Policy</th>"+"".join("<th>"+label+"</th>" for _, label in DISPLAY)+"<th>Feasible runs</th><th>Pareto eligible</th><th>Full batches</th></tr>"
        blocks.append(f'<section><h2>{mode.replace("_", " ")}</h2><img src="figures/{mode}.svg" alt="Outcomes, energy and latency"><div class="scroll"><table>'+headers+"".join(body)+"</table></div></section>")
    md += ["## Recovery on minus off, paired on exactly the same workload and outage trace", "",
           "Intervals use independent failure-seed means conditional on this workload panel; all individual pairs are in evaluation.json.", ""]
    effect_html = []
    for e in result["paired_recovery_effects"]:
        parts = []
        for key in ("completed_delta", "energy_delta_j"):
            u = e["uncertainty"][key]
            parts.append(f"{key}: {fmt(u['mean'])}; approximate 95% CI {u['ci95']}; n={u['n']} failure seeds")
        md.append("- "+e["policy"]+": "+"; ".join(parts))
        effect_html.append("<li>"+escape(e["policy"]+": "+"; ".join(parts))+"</li>")
    # Reliability results are shown explicitly, rather than buried in the run JSON.
    reliability_rows = []
    md += ["", "## Reliability with recovery", "", "| Policy | Deadline success | Recovery success | Recovery to delivery ms |", "|---|---:|---:|---:|"]
    for r in result["summaries"]:
        if r["mode"] == "failures_with_recovery":
            vals = [fmt(r["metrics"][k]) for k in ("deadline_success_rate", "recovery_success_rate", "mean_recovery_to_delivery_ms")]
            md.append("| "+r["policy"]+" | "+" | ".join(vals)+" |")
            reliability_rows.append("<tr><th>"+escape(r["policy"])+"</th>"+"".join("<td>"+v+"</td>" for v in vals)+"</tr>")
    # One policy/workload per seed suffices: outages do not depend on either.
    representatives = [r for r in result["rows"] if r["mode"] == "failures_with_recovery" and r["policy"] == "multi_objective" and r["workload_seed"] == result["experiment"]["workload_seeds"][0]]
    availability_rows = []
    md += ["", "## Node availability across the independent failure traces", "", "| Node | Mean observed availability | Configured steady-state availability |", "|---|---:|---:|"]
    for node in sorted(sample["model_configuration"]["failures"]):
        values = [next(a for a in r["availability"] if a["node_id"] == node) for r in representatives]
        obs, expected = mean(a["observed_availability"] for a in values), values[0]["configured_steady_state_availability"]
        md.append(f"| {node} | {fmt(obs)} | {fmt(expected)} |")
        availability_rows.append(f"<tr><th>{escape(node)}</th><td>{fmt(obs)}</td><td>{fmt(expected)}</td></tr>")
    blocks.append('<section><h2>Reliability and availability</h2><p>Rates are fractions, averaged across runs. Recovery delay is conditional on a successful recovery. Null denominators are omitted from means; individual denominators remain in each trace.</p><div class="scroll"><table><tr><th>Policy</th><th>Deadline success</th><th>Recovery success</th><th>Recovery to delivery ms</th></tr>'+"".join(reliability_rows)+'</table><table><tr><th>Node</th><th>Mean observed availability</th><th>Configured steady-state availability</th></tr>'+"".join(availability_rows)+'</table></div><p>Availability averages distinct failure traces, with no policy/workload duplication. Finite all-up-start windows need not equal the steady-state value.</p></section>')
    recovered = next((o for o in sample["outcomes"] if o["recovered"]), None)
    if recovered:
        example = [a for a in sample["attempts"] if a["task_id"] == recovered["task_id"]]
        example_rows = "".join("<tr>"+"".join("<td>"+escape(str(a[k]) if not isinstance(a[k], float) else fmt(a[k]))+"</td>" for k in ("attempt_id", "vm_id", "dispatch_ms", "start_ms", "end_ms", "status"))+"</tr>" for a in example)
        blocks.append('<section><h2>A real recovered task from this study: '+escape(recovered["task_id"])+f'</h2><p>Arrival {fmt(recovered["arrival_ms"])} ms; delivered {fmt(recovered["finish_ms"])} ms. Attempt time {fmt(recovered["attempt_elapsed_ms"])} ms + recovery blocking {fmt(recovered["recovery_blocked_ms"])} ms = original latency {fmt(recovered["finish_ms"]-recovered["arrival_ms"])} ms.</p><div class="scroll"><table><tr><th>Attempt</th><th>VM</th><th>Dispatch ms</th><th>Compute start ms</th><th>End ms</th><th>Status</th></tr>'+example_rows+'</table></div></section>')
        md += ["", "## Actual recovered example", "", f"Task {recovered['task_id']}: arrival {fmt(recovered['arrival_ms'])}, output delivered {fmt(recovered['finish_ms'])} ms; attempt elapsed {fmt(recovered['attempt_elapsed_ms'])} + recovery blocking {fmt(recovered['recovery_blocked_ms'])} ms."]

    # Evaluate alternate preferences offline; these do not imply the policy was retrained/reselected.
    sensitivity = []
    for r in result["rows"]:
        if not r["objective"]["feasible"]:
            continue
        for name, weights in (("time_focused", (.7,.15,.15)), ("energy_focused", (.15,.7,.15)), ("reliability_focused", (.15,.15,.7))):
            vector = r["objective"]["normalized"]
            sensitivity.append(dict(policy=r["policy"], mode=r["mode"], workload_seed=r["workload_seed"], failure_seed=r["failure_seed"],
                preference=name, score=sum(weight*vector[k] for weight,k in zip(weights, ("time", "energy", "failure")))))
    write_json(output/"weight-sensitivity.json", dict(scope="Offline rescoring of feasible existing runs, no retraining or new placement", rows=sensitivity))
    (output/"report.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    trace_links = "".join('<li><a href="'+r["trace_file"]+'">'+escape(r["trace_file"])+"</a></li>" for r in result["rows"])
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>DRLMOTS objective study</title><style>body{font:16px/1.5 system-ui;background:#f2f6f6;color:#193040;max-width:1450px;margin:auto;padding:28px}section{background:white;border:1px solid #cbdcdd;border-radius:12px;padding:24px;margin:22px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;min-width:1000px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce5e6}.scroll{overflow:auto}a{color:#086c69}li{margin:8px 0}h1{line-height:1.2}</style><h1>DRLMOTS: computation, energy and recovery</h1><p>Reproducible synthetic host/network study. '+str(len(result["rows"]))+' simulations.</p><p><a href="objective-report.zip">Download complete report ZIP</a> · <a href="metrics.csv">Metrics CSV</a> · <a href="evaluation.json">Evaluation JSON</a> · <a href="report.md">Report Markdown</a> · <a href="experiment.json">Experiment contract</a> · <a href="weight-sensitivity.json">Weight sensitivity</a></p><section><h2>How to read the results</h2><ul>'+"".join("<li>"+escape(n)+"</li>" for n in notes)+"</ul></section>"+"".join(blocks)+'<section><h2>Recovery effects and uncertainty</h2><ul>'+"".join(effect_html)+'</ul></section><section><h2>Trace and host power</h2><img src="figures/timeline-power.svg" alt="Attempt timeline and host power"></section><section><details><summary>Every run: full task, resource, energy and failure evidence</summary><ul>'+trace_links+"</ul></details></section></html>"
    (output/"objective-report.html").write_text(html, encoding="utf-8")
    paths = ["experiment.json", "evaluation.json", "metrics.csv", "report.md", "objective-report.html", "weight-sensitivity.json"] + [r["trace_file"] for r in result["rows"]] + [f"figures/{name}.{ext}" for name in plot_names for ext in ("png", "svg")]
    with zipfile.ZipFile(output/"objective-report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(output/path, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl_network.yaml")
    parser.add_argument("--model-config", default="configs/objective_model.yaml")
    parser.add_argument("--workload-seeds", nargs="+", type=int, default=[6200, 6201, 6202])
    parser.add_argument("--failure-seeds", nargs="+", type=int, default=[7100, 7101, 7102, 7103, 7104])
    parser.add_argument("--output-dir", default="outputs/objective-model/professor-final")
    args = parser.parse_args(argv)
    try:
        compare_objective_model(load_run_config(args.config), load_objective_model(args.model_config),
                                args.workload_seeds, args.failure_seeds, args.output_dir)
    except (ValueError, TypeError, OSError) as error:
        parser.exit(2, f"objective study: {error}\n")
    print(str(Path(args.output_dir)/"objective-report.html"))


if __name__ == "__main__":
    main()
