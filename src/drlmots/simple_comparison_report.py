"""Create a short three-method report from the saved frozen comparison."""
import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import zipfile


METHODS = {
    "trained_dqn": "Trained DQN",
    "untrained_dqn": "Untrained DQN",
    "multi_objective": "Without DQN (multi-objective rule)",
}


def create_report(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("choose a new output directory")
    evaluation = json.loads(source.read_text(encoding="utf-8"))
    experiment_path = source.with_name("experiment.json")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if evaluation["kind"] != "frozen_enhanced_six_policy_results" or not evaluation["frozen_audit"]["unchanged"]:
        raise ValueError("verified frozen comparison required")
    objective = experiment["model_configuration"]["objective"]
    weights, scales = objective["weights"], objective["scales"]
    rows = [r for r in evaluation["rows"] if r["mode"] == "failures_with_recovery" and r["policy"] in METHODS]
    cases = {}
    for row in rows:
        cases.setdefault((row["workload_seed"], row["failure_seed"]), []).append(row)
    if not cases or any(len(group) != 3 or {r["policy"] for r in group} != set(METHODS)
                        or len({r["workload_sha256"] for r in group}) != 1
                        or len({r["outage_sha256"] for r in group}) != 1 for group in cases.values()):
        raise ValueError("three methods must share every workload and outage trace")
    summaries = []
    for policy, label in METHODS.items():
        group = [r for r in rows if r["policy"] == policy]
        metrics = [r["metrics"] for r in group]
        if any(m["successful_makespan_ms"] is None for m in metrics):
            raise ValueError("short report requires a defined successful makespan in every run")
        mean = lambda key: math.fsum(m[key] for m in metrics) / len(metrics)
        costs = []
        for row in group:
            m = row["metrics"]
            cost = (weights["time"] * m["successful_makespan_ms"] / scales["time_ms"]
                    + weights["energy"] * m["total_energy_j"] / scales["energy_j"]
                    + weights["failure"] * (1 - m["completion_rate"]))
            if row["objective"]["feasible"] and not math.isclose(cost, row["objective"]["score"], abs_tol=1e-10):
                raise ValueError("central score disagrees with the original objective score")
            costs.append(cost)
        summaries.append(dict(policy=policy, label=label, makespan_ms=mean("successful_makespan_ms"),
            energy_j=mean("total_energy_j"), completion_percent=100 * mean("completion_rate"),
            completed_tasks=mean("completed_count"), runs=len(group),
            feasible=all(r["objective"]["feasible"] for r in group),
            cost=math.fsum(costs) / len(costs)))
    ranked = sorted((s for s in summaries if s["feasible"]), key=lambda s: s["cost"])
    if not ranked:
        raise ValueError("no method passes service floors on every run")
    trained, untrained = summaries[:2]
    gain = trained["completion_percent"] - untrained["completion_percent"]
    task_gain = trained["completed_tasks"] - untrained["completed_tasks"]
    energy_change = 100 * (trained["energy_j"] / untrained["energy_j"] - 1)
    time_change = 100 * (trained["makespan_ms"] / untrained["makespan_ms"] - 1)
    overall_change = 100 * (1 - trained["cost"] / untrained["cost"])
    if gain <= 0 or energy_change <= 0 or time_change <= 0 or not all(s["feasible"] for s in summaries):
        raise ValueError("this concise narrative applies to the recorded A-035 results only")
    work_count = len({w for w, f in cases})
    failure_count = len({f for w, f in cases})
    subtitle = f"{len(cases)} matched scenarios per method: {work_count} workloads × {failure_count} failure traces. Failures + recovery; 60 tasks per run."
    gain_line = f"Trained DQN: +{gain:.2f} percentage points in completion versus untrained ({untrained['completion_percent']:.2f}% → {trained['completion_percent']:.2f}%)."
    tradeoff = f"It completes {task_gain:.2f} more tasks per run on average, but uses {energy_change:.2f}% more energy and has {time_change:.2f}% longer successful makespan."
    winner_line = f"Best overall: {ranked[0]['label']}."
    near_tie = f"Trained vs untrained overall cost differs by only {overall_change:.2f}%: practically tied in this small trial."
    formula = (f"Cost = {weights['time']:g} × (makespan / {scales['time_ms']:g} ms) + "
               f"{weights['energy']:g} × (energy / {scales['energy_j']:g} J) + "
               f"{weights['failure']:g} × (1 − completion rate)")
    floors = f"Both service floors passed in every run: completion ≥ {100*objective['minimum_completion_rate']:g}%, deadline success ≥ {100*objective['minimum_deadline_success_rate']:g}%."
    note = "Observed averages from one small trained model. Completion-gain uncertainty includes zero; this is not a proven general improvement. Native rules and DQN use different information estimates."
    md = ["# Simple DQN comparison", "", winner_line, "", subtitle, "", "## Three methods, three metrics", "",
          "| Method | Successful makespan ms ↓ | Energy J ↓ | Completion rate % ↑ |", "|---|---:|---:|---:|"]
    for s in summaries:
        md.append(f"| {s['label']} | {s['makespan_ms']:.2f} | {s['energy_j']:.3f} | {s['completion_percent']:.2f} |")
    md += ["", "**Training benefit:** " + gain_line, "", tradeoff, "", "## One overall ranking", "",
           "Lower cost is better. Existing weights: time 40%, energy 40%, unsuccessful tasks 20%.", "",
           "| Rank | Method | Combined cost ↓ |", "|---|---|---:|"]
    for i, s in enumerate(ranked, 1):
        md.append(f"| {i} | {s['label']} | {s['cost']:.6f} |")
    md += ["", near_tie, "", formula, "", floors, "", "Makespan = last successful delivery from time zero; failed tasks are excluded. Completion rate includes all arrivals.", "", note, ""]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), layout="constrained")
    colors = ["#167347", "#9aa4ae", "#243c59"]
    for ax, key, title in zip(axes, ("makespan_ms", "energy_j", "completion_percent"),
                             ("Successful makespan (ms) · lower better", "Energy (J) · lower better", "Completion (%) · higher better")):
        values = [s[key] for s in summaries]
        bars = ax.bar(["Trained", "Untrained", "No DQN"], values, color=colors, width=.62)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4, fontsize=10)
        ax.set(title=title, ylim=(0, max(values) * 1.18))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    for ext in ("png", "svg"):
        fig.savefig(output / f"three-metrics.{ext}", dpi=160)
    plt.close(fig)
    svg = (output / "three-metrics.svg").read_text(encoding="utf-8")
    svg = svg[svg.index("<svg"):]
    metric_rows = "".join(f"<tr><th>{escape(s['label'])}</th><td>{s['makespan_ms']:.2f}</td><td>{s['energy_j']:.3f}</td><td>{s['completion_percent']:.2f}%</td></tr>" for s in summaries)
    ranking = "".join(f"<tr><td>{i}</td><th>{escape(s['label'])}</th><td>{s['cost']:.6f}</td></tr>" for i, s in enumerate(ranked, 1))
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Simple DQN comparison</title><style>body{font:16px/1.5 system-ui;color:#213044;background:#f3f5f8;margin:0}main{max-width:1100px;margin:auto;padding:28px}h1{margin:0;font-size:30px}h2{font-size:20px;margin:0 0 12px}p{margin:10px 0}.muted,footer{color:#536273;font-size:14px}section{background:white;border:1px solid #dbe2e8;border-radius:12px;padding:20px;margin:18px 0}.win{border-left:6px solid #243c59}.gain{color:#116438;background:#eef8f1;border-left:5px solid #167347;padding:14px 16px;border-radius:6px}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #e1e6eb}thead{background:#f2f5f8}.scroll{overflow-x:auto}svg{width:100%;height:auto}.formula{background:#f5f7fa;padding:12px;font-size:14px}a{color:#245f96}@media(max-width:650px){main{padding:14px}section{padding:14px}table{min-width:570px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid}.downloads{display:none}}</style></head><body><main><h1>Simple DQN comparison</h1><p class="muted">' + escape(subtitle) + '</p><section class="win"><h2>' + escape(winner_line) + '</h2><p>Lowest combined cost across the three objectives.</p></section><p class="gain"><strong>' + escape(gain_line) + '</strong></p><p>' + escape(tradeoff) + '</p><section><h2>Three methods · three metrics</h2><div class="scroll"><table><thead><tr><th>Method</th><th>Makespan ms ↓</th><th>Energy J ↓</th><th>Completion % ↑</th></tr></thead><tbody>' + metric_rows + '</tbody></table></div><p class="muted">Makespan = last successful delivery from time zero; failed tasks excluded. Completion rate includes all arrivals.</p>' + svg + '</section><section><h2>Overall ranking · lower cost is better</h2><div class="scroll"><table><thead><tr><th>Rank</th><th>Method</th><th>Combined cost ↓</th></tr></thead><tbody>' + ranking + '</tbody></table></div><p><strong>' + escape(near_tie) + '</strong></p><p class="formula">' + escape(formula) + '</p><p class="muted">Existing weights: 40% time, 40% energy, 20% unsuccessful tasks. ' + escape(floors) + '</p></section><footer><p>' + escape(note) + '</p><p class="downloads"><a href="REPORT.md">Written report</a> · <a href="three-metrics.png">Chart PNG</a> · <a href="data.json">Source summary</a></p></footer></main></body></html>'
    (output / "report.html").write_text(html, encoding="utf-8")
    data = dict(kind="simple_three_method_three_metric_report", mode="failures_with_recovery", summaries=summaries,
                ranking=[s["policy"] for s in ranked], objective=objective,
                completion_gain_percentage_points=gain, average_extra_completed_tasks=task_gain,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                experiment_sha256=hashlib.sha256(experiment_path.read_bytes()).hexdigest())
    (output / "data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "simple-report.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("report.html", "REPORT.md", "three-metrics.png", "three-metrics.svg", "data.json"):
            z.write(output / name, name)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/objective-model/frozen-comparison/milestone5-20261027-release/evaluation.json")
    parser.add_argument("--output-dir", default="outputs/objective-model/simple-comparison/20261027")
    args = parser.parse_args(argv)
    create_report(args.source, args.output_dir)
    print(f"Saved {args.output_dir}/report.html")


if __name__ == "__main__":
    main()
