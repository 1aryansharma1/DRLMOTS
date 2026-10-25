"""Offline reports for observed node failures and bounded restart recovery."""

import argparse
import csv
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import statistics
import zipfile

import yaml

from drlmots.efficiency import load_objectives
from drlmots.normalization import _digest
from drlmots.recovery import RECOVERY_POLICIES, simulate_recovery
from drlmots.recovery_config import load_recovery_config
from drlmots.run_config import load_run_config
from drlmots.train import seeded_config, write_json


MODES = ("no_failures", "failures_no_recovery", "failures_with_recovery")
DISPLAY = (("completed_count", "Completed"), ("rejected_count", "Rejected"),
           ("permanent_failure_count", "Permanent failures"), ("recovered_task_count", "Recovered"),
           ("makespan_ms", "Makespan ms"), ("successful_mean_latency_ms", "Successful mean ms"),
           ("total_energy_j", "Total J"), ("energy_per_completed_task_j", "J / completed"))


def summary(rows):
    result = []
    for policy in RECOVERY_POLICIES:
        for mode in MODES:
            group = [r for r in rows if r["policy"] == policy and r["mode"] == mode]
            metrics = {}
            for key in group[0]["metrics"]:
                values = [r["metrics"][key] for r in group if r["metrics"][key] is not None]
                metrics[key] = {"mean": statistics.mean(values) if values else None,
                                "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}
            result.append({"policy": policy, "mode": mode, "metrics": metrics})
    return result


def compare_recovery(config, objectives, recovery, seeds, output_dir):
    import matplotlib  # Fail before simulations if optional figure dependency is absent.
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("workload seeds must be distinct nonnegative integers")
    if not recovery.outages or recovery.max_retries <= 0:
        raise ValueError("three-way report needs at least one outage and a positive recovery retry budget")
    output = Path(output_dir)
    plan = {"kind": "observed_failure_recovery_comparison", "run_configuration": config.manifest(),
            "recovery_configuration": recovery.manifest(), "objective_coefficients": objectives.manifest(),
            "workload_seeds": list(seeds), "policies": list(RECOVERY_POLICIES), "modes": list(MODES),
            "scope": "fixed synthetic node trace; variation is across workloads, not independently sampled failure traces; no learned fault-aware policy"}
    plan_path = output / "experiment.json"
    if plan_path.exists() and _digest(json.loads(plan_path.read_text(encoding="utf-8"))) != _digest(plan):
        raise ValueError("output contains a different experiment; choose a new directory")
    write_json(plan_path, plan)
    rows, reports = [], []
    specs = ((MODES[0], replace(recovery, outages=(), max_retries=0)),
             (MODES[1], replace(recovery, max_retries=0)), (MODES[2], recovery))
    for seed in seeds:
        current = seeded_config(config, seed)
        group = []
        for policy in RECOVERY_POLICIES:
            paired = []
            for mode, settings in specs:
                report = simulate_recovery(current, objectives, settings, policy=policy)
                report["mode"] = mode
                filename = f"runs/workload-{seed}-{policy}-{mode}.json"
                write_json(output / filename, report)
                rows.append({"seed": seed, "policy": policy, "mode": mode, "trace_file": filename,
                             "workload_sha256": report["workload"]["sha256"],
                             "outage_sha256": report["outage_trace"]["sha256"], "metrics": report["metrics"]})
                reports.append((filename, report))
                paired.append(report)
                group.append(report)
            if paired[1]["outage_trace"] != paired[2]["outage_trace"]:
                raise RuntimeError("recovery comparisons have different outage traces")
        if len({r["workload"]["sha256"] for r in group}) != 1 or len({r["run_configuration"]["sha256"] for r in group}) != 1:
            raise RuntimeError("comparison policies did not receive identical tasks/configuration")
        print("Evaluated workload", seed, "across", len(group), "policy/mode combinations", flush=True)
    result = {"experiment": plan, "rows": rows, "summaries": summary(rows),
              "interpretation": "Successful timing excludes rejected/failed tasks. Read counts alongside makespan/energy; a scheduler that loses work can appear faster or cheaper. Energy includes partial failed attempts, retransmissions, recovery overhead and idle uptime over a common horizon."}
    write_json(output / "evaluation.json", result)
    render(output, result, reports)
    return result


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def render(output, result, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    html_blocks, md = [], ["# Observed node failures and restart recovery", "", result["interpretation"], "",
        "## Experiment", "", result["experiment"]["scope"], "",
        "Workload seeds: " + str(result["experiment"]["workload_seeds"]),
        "Observation horizon ms: " + str(result["experiment"]["recovery_configuration"]["horizon_ms"]),
        "No failures and failure modes have different node uptime; recovery on/off shares exactly the same outage trace.",
        "Active power includes the running VM's full power. Idle power applies only while up and not running; down nodes draw zero.",
        "Input serialization precedes propagation. Partial transmission is billed by bytes sent, failed processing by actual executed time.",
        "Restart requires full reupload and full processing on a different node. Original input is retained by a durable source; detection is immediate.",
        "Only the configured delay consumes coordinator power; backup waiting beyond that delay does not.",
        "Legacy DQN state/reward/checkpoints are unchanged and are not evaluated as fault-aware agents. Training integration is the next versioned contract.",
        "Power coefficients and outage scenarios are synthetic. No host power sharing, link contention, output transfers or source/controller failures are modeled.", ""]
    for policy in RECOVERY_POLICIES:
        group = [r for r in result["summaries"] if r["policy"] == policy]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        labels = [r["mode"].replace("_", " ") for r in group]
        for key, color, label in (("completed_count", "#167e75", "Completed"), ("rejected_count", "#d7a34d", "Rejected"),
                                  ("permanent_failure_count", "#b94e4e", "Permanent failure")):
            left = [sum(r["metrics"][k]["mean"] for k in ("completed_count", "rejected_count", "permanent_failure_count")[:
                    ("completed_count", "rejected_count", "permanent_failure_count").index(key)]) for r in group]
            axes[0].barh(labels, [r["metrics"][key]["mean"] for r in group], left=left, color=color, label=label)
        axes[0].set(title="All submitted tasks accounted for", xlabel="Mean tasks")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncols=3, frameon=False)
        for ax, key, label in ((axes[1], "energy_per_completed_task_j", "Fixed-window J / completed task"),
                               (axes[2], "successful_mean_latency_ms", "Successful mean latency (ms)")):
            values = [r["metrics"][key]["mean"] for r in group]
            ax.barh(labels, [v if v is not None else 0 for v in values], color="#167e75")
            ax.set(title=label, xlabel="Mean over workload seeds")
            ax.tick_params(axis="y", labelleft=False)
            for index, value in enumerate(values):
                if value is None:
                    ax.text(0, index, "undefined: no completions", fontsize=8)
        for ax in axes:
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.15)
        fig.suptitle(policy.replace("_", " ") + ": recovery benefits and costs; synthetic fixed outage scenario")
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{policy}-comparison.{ext}", dpi=160)
        plt.close(fig)
        headers = "<tr><th>Mode</th>" + "".join("<th>" + label + "</th>" for _, label in DISPLAY) + "</tr>"
        rows = "".join("<tr><th>" + escape(r["mode"]) + "</th>" + "".join("<td>" + fmt(r["metrics"][key]["mean"]) + "</td>" for key, _ in DISPLAY) + "</tr>" for r in group)
        html_blocks.append('<section><h2>' + escape(policy.replace("_", " ")) + '</h2><img src="figures/' + policy + '-comparison.svg" alt="Observed outcomes, energy and latency">'
            + '<div class="scroll"><table>' + headers + rows + '</table></div></section>')
        md.extend(["## " + policy, "", "| Mode | " + " | ".join(label for _, label in DISPLAY) + " |",
                   "|---|" + "---:|" * len(DISPLAY)])
        for row in group:
            md.append("| " + row["mode"] + " | " + " | ".join(fmt(row["metrics"][key]["mean"]) for key, _ in DISPLAY) + " |")
        md.append("")
    # One trace diagram: multiple attempts per task; red segments are interrupted.
    selected = next(report for _, report in reports if report["mode"] == "failures_with_recovery" and report["policy"] == "fog_first_completion")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    vm_ids = sorted(u["vm_id"] for u in selected["vm_usage"])
    nodes = {u["vm_id"]: u["node_id"] for u in selected["vm_usage"]}
    for vm_index, vm in enumerate(vm_ids):
        for outage in selected["outage_trace"]["outages"]:
            if outage["node_id"] == nodes[vm]:
                ax.broken_barh([(outage["start_ms"], outage["end_ms"] - outage["start_ms"])], (vm_index-.4, .8), facecolors="#f6dedb")
        for attempt in selected["attempts"]:
            if attempt["vm_id"] != vm:
                continue
            finish = attempt["end_ms"]
            start = attempt["start_ms"]
            ax.broken_barh([(attempt["dispatch_ms"], attempt["upload_elapsed_ms"])], (vm_index-.25, .18), facecolors="#729bc7")
            if start is not None:
                ax.broken_barh([(start, finish-start)], (vm_index-.03, .3), facecolors="#167e75" if attempt["status"] == "completed" else "#b94e4e")
    ax.set(yticks=range(len(vm_ids)), yticklabels=vm_ids, xlabel="Simulation time ms",
           title="Example execution: blue upload, green completed service, red interrupted service; pale red node outage")
    ax.grid(axis="x", alpha=.2)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"recovery-timeline.{ext}", dpi=160)
    plt.close(fig)
    example = next((row for row in selected["outcomes"] if row["recovered"]), None)
    if example is not None:
        own = [attempt for attempt in selected["attempts"] if attempt["task_id"] == example["task_id"]]
        columns = ("attempt_id", "vm_id", "dispatch_ms", "start_ms", "end_ms", "status", "compute_energy_j", "transmission_energy_j")
        example_table = '<tr>' + ''.join('<th>' + escape(key) + '</th>' for key in columns) + '</tr>'
        for attempt in own:
            example_table += '<tr>' + ''.join('<td>' + escape(fmt(attempt[key]) if isinstance(attempt[key], (float, int)) else str(attempt[key])) + '</td>' for key in columns) + '</tr>'
        html_blocks.append('<section><h2>One recovered task: ' + escape(example["task_id"]) + '</h2>'
            + '<p>Final latency ' + fmt(example["latency_ms"]) + ' ms = ' + fmt(example["attempt_elapsed_ms"]) + ' ms across attempts + '
            + fmt(example["recovery_blocked_ms"]) + ' ms recovery delay/backup waiting. Task energy ' + fmt(example["attributed_energy_j"])
            + ' J includes all attempts and coordinator work; VM idle energy is accounted separately.</p><div class="scroll"><table>' + example_table + '</table></div></section>')

    html_blocks.append('<section><h2>Follow a task through failure and recovery</h2><img src="figures/recovery-timeline.svg" alt="Actual attempt and outage timeline">'
                       '<p>Each task has one final outcome and can have multiple attempts. Full JSON includes upload, queue, execution, retry blocking and energy for every attempt.</p></section>')
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "policy", "mode", *result["rows"][0]["metrics"]])
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({key: row[key] for key in ("seed", "policy", "mode")} | row["metrics"])
    links = ''.join('<li><a href="' + name + '" download>' + escape(name) + '</a></li>' for name, _ in reports)
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DRLMOTS | Observed failure and recovery</title><style>
body{margin:0;background:#f1f6f6;color:#193443;font:16px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:30px 18px}
h1{font-size:36px;line-height:1.2}section{background:white;border:1px solid #d8e3e5;border-radius:14px;padding:24px;margin:24px 0}img{width:100%;height:auto}
.notice{padding:18px;background:#e3eeea;border-left:4px solid #167e75}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;border-bottom:1px solid #d8e3e5}
a{color:#096e68}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}</style></head><body><main>
<h1>Observed node failure and restart recovery</h1><p class="notice">This simulator actually interrupts tasks and restarts eligible work on another node. These are observed simulation outcomes, not the older analytical failure-risk estimates. Synthetic coefficients and outage scenarios are explicit. No DQN has been trained for this new environment yet.</p>
<p>Compare no failures, failures without recovery, and identical failures with bounded recovery. Every comparison uses identical tasks and a common energy-accounting horizon. Makespan and successful latency exclude failed work; always read them alongside completed, rejected and permanently failed counts.</p>
'''
    html += ''.join(html_blocks)
    html += '<section><h2>Reproduce and inspect</h2><p><a href="evaluation.json" download>All results and workload variation</a> | <a href="metrics.csv" download>Metrics CSV</a> | <a href="REPORT.md" download>Written report</a> | <a href="experiment.json" download>Configuration</a></p>'
    html += '<details><summary>All actual execution traces</summary><ul>' + links + '</ul></details><details><summary>Configuration, units and scope</summary><pre>' + escape(json.dumps(result["experiment"], indent=2)) + '</pre></details></section></main></body></html>'
    (output / "recovery-report.html").write_text(html, encoding="utf-8")
    (output / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    allowed = ["recovery-report.html", "evaluation.json", "metrics.csv", "REPORT.md", "experiment.json"]
    allowed += [name for name, _ in reports]
    allowed += [p.relative_to(output).as_posix() for p in figures.glob("*") if p.suffix in (".png", ".svg")]
    with zipfile.ZipFile(output / "recovery-report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(allowed):
            archive.write(output / name, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--objectives", type=Path, required=True)
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5200, 5201, 5202, 5203, 5204])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        compare_recovery(load_run_config(args.config), load_objectives(args.objectives),
                         load_recovery_config(args.recovery_config), args.seeds, args.output_dir)
    except (ValueError, TypeError, OSError, RuntimeError, OverflowError, yaml.YAMLError) as error:
        parser.error(str(error))
    print("Completed:", args.output_dir / "recovery-report.html")


if __name__ == "__main__":
    main()
