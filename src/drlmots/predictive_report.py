"""Concise reports for the predeclared predictive-DQN training study."""
import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import zipfile

from drlmots.objective_report import interval


LABELS = dict(predictive_untrained="Untrained predictive DQN", without_dqn="Without DQN (multi-objective rule)",
              original_untrained_dqn="Original untrained DQN", original_trained_dqn="Original 16-episode DQN")


def mean(values):
    present = [v for v in values if v is not None]
    return math.fsum(present)/len(present) if present else None


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r["method"]), []).append(r)
    result = []
    for (mode, method), group in groups.items():
        eligible = all(r["feasible"] for r in group)
        result.append(dict(mode=mode, method=method, runs=len(group), feasible_runs=sum(r["feasible"] for r in group),
            eligible=eligible, metrics={k: mean(r["metrics"][k] for r in group) for k in group[0]["metrics"]},
            contributing_runs={k: sum(r["metrics"][k] is not None for r in group) for k in group[0]["metrics"]},
            mean_objective_cost=mean(r["objective_score"] for r in group) if eligible else None,
            mean_penalized_cost=-mean(r["metrics"]["penalized_return"] for r in group)))
    return result


def model_effects(rows, trained_method):
    result = []
    for mode in ("no_failures", "failures_with_recovery"):
        samples = []
        seeds = sorted({r["model_seed"] for r in rows if r["method"] == trained_method})
        for seed in seeds:
            pairs = []
            for r in rows:
                if r["method"] != trained_method or r["model_seed"] != seed or r["mode"] != mode:
                    continue
                other = next(o for o in rows if o["method"] == "predictive_untrained" and o["model_seed"] == seed
                    and o["mode"] == mode and o["workload_seed"] == r["workload_seed"] and o["failure_seed"] == r["failure_seed"])
                if (r["workload_sha256"], r["outage_sha256"]) != (other["workload_sha256"], other["outage_sha256"]):
                    raise ValueError("paired predictive inputs differ")
                pairs.append({k: r["metrics"][k]-other["metrics"][k] if r["metrics"][k] is not None and other["metrics"][k] is not None else None
                    for k in ("completed_count", "successful_makespan_ms", "total_energy_j", "penalized_return")})
            samples.append(dict(model_seed=seed, pairs=pairs,
                means={k: mean(p[k] for p in pairs) for k in pairs[0]}))
        result.append(dict(mode=mode, model_seed_means=samples,
            intervals={k: interval([s["means"][k] for s in samples if s["means"][k] is not None]) for k in samples[0]["means"]}))
    return result


def _fmt(value, digits=3):
    return "undefined" if value is None else f"{value:.{digits}f}"


def figures(output, study, summary, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = ("no_failures", "failures_with_recovery")
    for mode in modes:
        group = [next(r for r in summary if r["method"] == p and r["mode"] == mode) for p in primary]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for ax, key, title in zip(axes, ("successful_makespan_ms", "total_energy_j", "completion_rate"),
            ("Successful makespan (ms)\nLower better; read completion too", "System energy (J)\nLower better", "Completion (%)\nHigher better")):
            values = [r["metrics"][key]*(100 if key == "completion_rate" else 1) if r["metrics"][key] is not None else None for r in group]
            bars = ax.bar(["Trained", "Untrained", "No DQN"], [v or 0 for v in values], color=["#167347", "#98a3ae", "#263f5d"], width=.62)
            ax.bar_label(bars, labels=[_fmt(v, 2) for v in values], padding=4)
            ax.set(title=title, ylim=(0, max([1]+[v for v in values if v is not None])*1.2))
            if key == "completion_rate":
                ax.set(ylim=(0, 110), yticks=(0, 20, 40, 60, 80, 100))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
        fig.suptitle(mode.replace("_", " ") + ": frozen means on the same fresh scenarios")
        for ext in ("png", "svg"):
            fig.savefig(output/f"{mode}.{ext}", dpi=160)
        plt.close(fig)
    with (output/"training.csv").open(encoding="utf-8") as f:
        train = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for seed in sorted({s["model_seed"] for s in study["validation_stages"]}):
        rows = [r for r in train if int(r["model_seed"]) == seed]
        x = [int(r["episode"]) for r in rows]
        rewards = [float(r["penalized_return"]) for r in rows]
        smooth = [mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))]
        axes[0, 0].plot(x, smooth, label=f"seed {seed}")
        with (output/f"models/seed-{seed}/losses.csv").open() as f:
            losses = list(csv.DictReader(f))
        chunks = [losses[i:i+100] for i in range(0, len(losses), 100)]
        axes[0, 1].plot([int(c[-1]["update"]) for c in chunks], [mean(float(r["loss"]) for r in c) for c in chunks], label=f"seed {seed}")
        stages = [s for s in study["validation_stages"] if s["model_seed"] == seed]
        axes[1, 0].plot([s["episode"] for s in stages], [-s["selection_rank"][2] for s in stages], marker=".", label=f"seed {seed}")
        axes[1, 1].plot([s["episode"] for s in stages], [len(s["rows"])-s["selection_rank"][0] for s in stages], marker=".", label=f"seed {seed}")
    for ax, title, xlabel in zip(axes.flat,
        ("Training return: rolling 10 episodes", "Actual Huber loss: blocks of 100 updates", "Frozen validation return (higher better)", "Service-feasible validation cases"),
        ("Episode", "Optimizer update", "Episode", "Episode")):
        ax.set(title=title, xlabel=xlabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(output/f"learning.{ext}", dpi=160)
    plt.close(fig)


def render(directory):
    output = Path(directory)
    study = json.loads((output/"result.json").read_text(encoding="utf-8"))
    manifest = json.loads((output/"experiment.json").read_text(encoding="utf-8"))
    if study["kind"] != "predictive_dqn_training_results" or manifest["status"] != "complete":
        raise ValueError("completed predictive study required")
    budget = max(manifest["plan"]["training_budgets"])
    method = f"predictive_budget_{budget}"
    labels = {**LABELS, **{f"predictive_budget_{b}": f"Trained predictive DQN (budget {b})" for b in manifest["plan"]["training_budgets"]}}
    primary = (method, "predictive_untrained", "without_dqn")
    summary = summarize(study["test_rows"])
    effects = model_effects(study["test_rows"], method)
    figures(output, study, summary, primary)
    def table(group, simple=False):
        headers = ("Method", "Makespan ms ↓", "Energy J ↓", "Completion % ↑") if simple else ("Method", "Makespan ms", "Energy J", "Completion %", "Feasible runs", "Penalized cost ↓")
        rows = []
        for s in group:
            m = s["metrics"]
            cells = [labels[s["method"]], _fmt(m["successful_makespan_ms"], 2), _fmt(m["total_energy_j"]), _fmt(100*m["completion_rate"], 2)]
            if not simple:
                cells += [f"{s['feasible_runs']}/{s['runs']}", _fmt(s["mean_penalized_cost"], 6)]
            rows.append(cells)
        return headers, rows
    def html_table(headers, rows):
        return '<div class="scroll"><table><thead><tr>'+"".join("<th>"+escape(h)+"</th>" for h in headers)+'</tr></thead><tbody>'+"".join("<tr>"+"".join("<td>"+escape(c)+"</td>" for c in row)+"</tr>" for row in rows)+"</tbody></table></div>"
    def markdown_table(headers, rows):
        return ["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers), *["| "+" | ".join(row)+" |" for row in rows]]
    def svg(name):
        text = (output/name).read_text(encoding="utf-8")
        return text[text.index("<svg"):]
    md = ["# Predictive DQN: longer training results", "",
        f"{study['training_episode_count']} training episodes across {len(manifest['plan']['model_seeds'])} model seeds; {len(study['test_rows'])} frozen test simulations.", "",
        "Three metrics: successful makespan, full system energy and all-arrival completion rate. Primary trained model uses the predeclared largest budget; checkpoints are selected on validation only.", ""]
    blocks = []
    for mode in ("failures_with_recovery", "no_failures"):
        group = [next(s for s in summary if s["mode"] == mode and s["method"] == p) for p in primary]
        trained, untrained, rule = group
        gain = 100*(trained["metrics"]["completion_rate"]-untrained["metrics"]["completion_rate"])
        eligible = sorted((s for s in group if s["eligible"]), key=lambda s: s["mean_objective_cost"])
        winner = labels[eligible[0]["method"]] if eligible else "none: service floors failed"
        gain_text = f"Training completion change: {gain:+.2f} percentage points versus the same-feature untrained network."
        changes = []
        for key, name in (("successful_makespan_ms", "Makespan"), ("total_energy_j", "Energy")):
            a, b = trained["metrics"][key], untrained["metrics"][key]
            if a is not None and b:
                percent = 100*(a/b-1)
                changes.append((f"{name}: {abs(percent):.2f}% {'lower' if percent < 0 else 'higher'}", percent < 0))
        changes.append((f"Completion: {gain:+.2f} percentage points", gain > 0))
        change_text = "; ".join(text for text, good in changes) + " versus untrained."
        headers, cells = table(group, simple=True)
        rank_rows = [[str(i), labels[s["method"]], _fmt(s["mean_objective_cost"], 6)] for i, s in enumerate(eligible, 1)]
        excluded = "; ".join(labels[s["method"]]+f": {s['feasible_runs']}/{s['runs']} service-feasible runs" for s in group if not s["eligible"])
        md += ["## "+mode.replace("_", " "), "", "Best eligible overall: **"+winner+"**.", "", *markdown_table(headers, cells), "", gain_text, "", change_text, "",
               *markdown_table(("Rank", "Method", "Combined cost ↓"), rank_rows), "", excluded, ""]
        highlights = "".join('<p class="'+('gain' if good else 'caution')+'">'+escape(text)+'</p>' for text, good in changes)
        block = '<section><h2>'+escape(mode.replace("_", " "))+'</h2><p><strong>Best eligible overall: '+escape(winner)+'.</strong></p>'+html_table(headers, cells)+'<p>Trained versus the same-feature untrained network:</p>'+highlights+svg(mode+".svg")+'<h3>Overall ranking · lower cost is better</h3>'+html_table(("Rank", "Method", "Combined cost ↓"), rank_rows)+'<p>'+escape(excluded)+'</p></section>'
        blocks.append(block if mode == "failures_with_recovery" else '<details><summary>Without failures</summary>'+block+'</details>')
    o = manifest["model_configuration"]["objective"]
    formula = f"Cost = {o['weights']['time']:g} × makespan/{o['scales']['time_ms']:g} + {o['weights']['energy']:g} × energy/{o['scales']['energy_j']:g} + {o['weights']['failure']:g} × unsuccessful fraction."
    notes = [formula,
        "Only methods satisfying both service floors in every test run enter the overall ranking. Penalized costs and excluded runs remain visible below.",
        "The trained/untrained predictive comparison shares inputs, telemetry and initialization seed. Native-rule comparisons retain different information access; this is an implementation comparison.",
        "All model seeds train on the same declared training panel. Budget results are nested prefixes, not independent training replications. Test scenarios are generated only after all validation selections are locked.",
        "Prediction features and training budget changed together relative to the original 16-episode baseline; that comparison does not isolate the feature effect.",
        "Reported timing excludes unsuccessful tasks; energy includes all hosts/links, failed partial work and recovery through the same horizon. Read completion with timing/energy.",
        "The approximate paired intervals below use model-seed means on the fixed test panel. Three seeds give limited uncertainty estimates, not evidence over all future deployments.",
        "Checkpoint files, compressed full training traces and validation/test traces remain in the experiment directory. The compact ZIP contains report figures, summaries and CSVs; it does not bundle those large files."]
    md += ["## Selection and learning evidence", "", *["- "+n for n in notes], "", "| Seed | Budget | Selected episode | Updates | All validation cases feasible |", "|---|---|---|---|---|"]
    selection_rows = []
    for seed, budgets in study["selections"].items():
        for b, s in budgets.items():
            cells = [str(seed), str(b), str(s["episode"]), str(s["updates"]), str(s["eligible_for_promotion"])]
            selection_rows.append(cells)
            md.append("| "+" | ".join(cells)+" |")
    blocks.append('<details><summary>Training and validation selection</summary><section>'+html_table(("Seed", "Budget", "Selected episode", "Updates", "Validation feasible"), selection_rows)+svg("learning.svg")+'</section></details>')
    effect_rows = []
    md += ["", "## Paired trained minus untrained, model-seed intervals", ""]
    for effect in effects:
        for metric, stats in effect["intervals"].items():
            interval_text = str(stats["ci95"]) if stats["ci95"] is not None else "undefined"
            cells = [effect["mode"], metric, _fmt(stats["mean"], 6), interval_text, str(stats["n"])]
            effect_rows.append(cells)
            md.append("- "+" | ".join(cells))
    blocks.append('<details><summary>Paired uncertainty and preserved baselines</summary><section>'+html_table(("Mode", "Metric delta", "Mean", "Approx. 95% interval", "Model seeds"), effect_rows))
    for mode in ("failures_with_recovery", "no_failures"):
        group = [s for s in summary if s["mode"] == mode]
        headers, cells = table(group)
        blocks.append('<h3>'+escape(mode.replace("_", " "))+'</h3>'+html_table(headers, cells))
        md += ["", "## All preserved/new methods: "+mode, "", *markdown_table(headers, cells), ""]
    blocks.append('</section></details>')
    (output/"REPORT.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(dict(summaries=summary, paired_model_effects=effects), indent=2)+"\n", encoding="utf-8")
    validation_rows = [r for s in study["validation_stages"] for r in s["rows"]]
    with (output/"validation.csv").open("w", newline="", encoding="utf-8") as f:
        flat = [{**{k: v for k, v in r.items() if k != "metrics"}, **r["metrics"]} for r in validation_rows]
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Predictive DQN results</title><style>body{font:16px/1.5 system-ui;color:#213449;background:#f3f6f8;margin:0}main{max-width:1150px;margin:auto;padding:26px}h1{margin:0;font-size:30px}h2{font-size:21px}section{background:white;padding:20px;border:1px solid #dbe3e8;border-radius:12px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 8px;text-align:left;border-bottom:1px solid #dce4e9;white-space:nowrap}.scroll{overflow:auto}.gain{background:#ecf8ef;color:#116335;padding:12px}.caution{background:#fff4e7;padding:12px}svg{max-width:100%;height:auto}details{margin:18px 0}summary{cursor:pointer;font-weight:600}footer{font-size:13px;color:#536779}a{color:#1c6289}@media(max-width:650px){main{padding:12px}section{padding:12px}}</style></head><body><main><h1>Predictive DQN · longer training</h1><p>'+str(study["training_episode_count"])+' training episodes across '+str(len(manifest["plan"]["model_seeds"]))+' model seeds. '+str(len(study["test_rows"]))+' frozen simulations on fresh test inputs.</p><p>325 inputs = 253 observed features + 72 causal predictions. Same basic DQN, reward and legal-action rules.</p>'+"".join(blocks)+'<footer><p>'+escape(formula)+'</p><ul>'+"".join('<li>'+escape(n)+'</li>' for n in notes[1:])+'</ul><p><a href="REPORT.md">Written report</a> · <a href="test-metrics.csv">Test CSV</a> · <a href="training.csv">Training CSV</a> · <a href="validation.csv">Validation CSV</a> · <a href="summary.json">Summary JSON</a> · <a href="selection-locked.json">Locked selections</a> · <a href="experiment.json">Experiment</a> · <a href="evidence.json">Evidence hashes</a></p></footer></main></body></html>'
    (output/"report.html").write_text(html, encoding="utf-8")
    names = ["report.html", "REPORT.md", "summary.json", "experiment.json", "selection-locked.json", "test-metrics.csv", "training.csv", "validation.csv"]
    names += [f"{name}.{ext}" for name in ("no_failures", "failures_with_recovery", "learning") for ext in ("png", "svg")]
    evidence = dict(kind="predictive_training_report_evidence", renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256((output/"result.json").read_bytes()).hexdigest(),
        sha256={n: hashlib.sha256((output/n).read_bytes()).hexdigest() for n in names})
    (output/"evidence.json").write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
    with zipfile.ZipFile(output/"report.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in (*names, "evidence.json"):
            z.write(output/name, name)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="outputs/objective-model/predictive-training/study-20261027")
    args = parser.parse_args(argv)
    render(args.directory)
    print(f"Saved {args.directory}/report.html and report.zip")


if __name__ == "__main__":
    main()
