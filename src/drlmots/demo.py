"""Generate an offline professor demo and export real synthetic comparison figures."""

import argparse
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from drlmots.baselines import POLICY_LABELS, FOG_FIRST_POLICY
from drlmots.evaluation import compare_policies
from drlmots.run_config import load_run_config


def demo_scenarios(config):
    presets = [
        ("worked", "Worked example", "Six simultaneous tasks: follow placement, waiting and resource release.",
         dict(task_count=6, seed=42, interarrival_ms_range=(0, 0), length_bytes_range=(1000, 1000), data_size_kb_range=(20, 20))),
        ("seeded", "Seeded workload", "100 tasks with variable arrivals and lengths, using seed 42.",
         dict(task_count=100, seed=42, interarrival_ms_range=(0, 3), length_bytes_range=(500, 2000), data_size_kb_range=(20, 150))),
        ("pressure", "Queue pressure", "25 simultaneous tasks expose waiting limits and explicit rejection.",
         dict(task_count=25, seed=42, interarrival_ms_range=(0, 0), length_bytes_range=(500, 2000), data_size_kb_range=(20, 150))),
    ]
    scenarios = []
    for key, label, description, settings in presets:
        scenario_config = replace(config, workload=replace(config.workload, **settings))
        scenarios.append({"id": key, "label": label, "description": description,
                          "comparison": compare_policies(scenario_config)})
    return scenarios


def _svg(fig, plt, salt):
    buffer = StringIO()
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": salt}):
        fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": "DRLMOTS demo"})
    content = buffer.getvalue()
    return content[content.index("<svg"):]


def _style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e5eaed", linewidth=0.7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5dc")
    ax.tick_params(colors="#425563", length=0, pad=7)


def _charts(scenario, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figures = {}
    reports = scenario["comparison"]["reports"]
    for report in reports:
        policy = report["policy"]
        stem = f"{scenario['id']}-{policy}"
        vm_ids = [item["vm_id"] for item in report["vm_usage"]]
        fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
        positions = {vm_id: index for index, vm_id in enumerate(vm_ids)}
        tooltips = {}
        for index, item in enumerate(report["outcomes"]):
            if item["finish_time_ms"] is None:
                continue
            start, finish = item["start_time_ms"], item["finish_time_ms"]
            bars = ax.barh(positions[item["vm_id"]], finish - start, left=start, height=0.5,
                           color="#198a83", edgecolor="white", linewidth=0.55)
            gid = f"{stem}-task-{index}"
            bars[0].set_gid(gid)
            tooltips[gid] = f"{item['task_id']} on {item['vm_id']}: {start:.3f}–{finish:.3f} ms"
        ax.set_yticks(range(len(vm_ids)), labels=vm_ids)
        ax.invert_yaxis()
        ax.set_xlim(0, max(report["makespan_ms"], 1))
        ax.set_xlabel("Simulation time (ms)")
        ax.set_title("Task execution timeline", loc="left", pad=14, color="#152835")
        _style_axis(ax)
        svg = _svg(fig, plt, stem + "-timeline")
        root = ET.fromstring(svg)
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        for element in root.iter():
            if element.get("id") in tooltips:
                ET.SubElement(element, "{http://www.w3.org/2000/svg}title").text = tooltips[element.get("id")]
        svg = ET.tostring(root, encoding="unicode")
        (output_dir / f"{stem}-timeline.svg").write_text(svg, encoding="utf-8")
        fig.savefig(output_dir / f"{stem}-timeline.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.8, 3.4), layout="constrained")
        occupancies = [item["occupancy_pct"] for item in report["vm_usage"]]
        ax.barh(vm_ids, occupancies, height=0.45, color="#697b95")
        ax.invert_yaxis()
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Busy time / run makespan (%)")
        ax.set_title("VM occupancy", loc="left", pad=14, color="#152835")
        for index, value in enumerate(occupancies):
            ax.text(value + 2, index, f"{value:.1f}%", va="center", color="#425563")
        _style_axis(ax)
        usage_svg = _svg(fig, plt, stem + "-usage")
        (output_dir / f"{stem}-usage.svg").write_text(usage_svg, encoding="utf-8")
        fig.savefig(output_dir / f"{stem}-usage.png", dpi=160)
        plt.close(fig)
        figures[policy] = {"timeline": svg, "usage": usage_svg}

    fig, ax = plt.subplots(figsize=(6.8, 2.9), layout="constrained")
    labels = [POLICY_LABELS[report["policy"]] for report in reports]
    complete = [report["completed_count"] for report in reports]
    failed = [report["failed_count"] for report in reports]
    ax.barh(labels, complete, height=0.45, color="#198a83", label="Completed")
    ax.barh(labels, failed, left=complete, height=0.45, color="#dba66a", label="Failed")
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, len(reports[0]["outcomes"])))
    ax.set_xlabel("Tasks (same inputs for all policies)")
    ax.set_title("Completion and rejection", loc="left", pad=14, color="#152835")
    ax.legend(loc="lower right", frameon=False, ncols=2, bbox_to_anchor=(1, 1.04))
    _style_axis(ax)
    comparison_svg = _svg(fig, plt, scenario["id"] + "-comparison")
    (output_dir / f"{scenario['id']}-comparison.svg").write_text(comparison_svg, encoding="utf-8")
    fig.savefig(output_dir / f"{scenario['id']}-comparison.png", dpi=160)
    plt.close(fig)
    return {"policies": figures, "comparison": comparison_svg}


def generate_demo(config_path, output_dir):
    config = load_run_config(config_path)
    scenarios = demo_scenarios(config)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        scenario["figures"] = _charts(scenario, figure_dir)
    bundle = {
        "title": "DRLMOTS · Scheduling prototype",
        "version": scenarios[0]["comparison"]["reports"][0]["version"],
        "scope": "Reference-rule comparisons only; trained DQN runs are in the separate training report. No paper reproduction claim.",
        "policy_labels": {policy: POLICY_LABELS[policy] for policy in scenarios[0]["comparison"]["policies"]},
        "default_policy": FOG_FIRST_POLICY if config.placement else "least_queue_fcfs",
        "scenarios": scenarios,
    }
    content = json.dumps(bundle, ensure_ascii=True, allow_nan=False)
    # A config identifier containing an HTML closing tag must stay inert in script data.
    embedded = content.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("templates").joinpath("demo.html").read_text(encoding="utf-8")
    html = template.replace("__DEMO_DATA__", embedded).replace(
        '<body data-demo="template">', '<body data-demo="report">', 1
    )
    output_dir.joinpath("drlmots-demo.html").write_text(html, encoding="utf-8")
    output_dir.joinpath("comparison.json").write_text(content + "\n", encoding="utf-8")
    return output_dir / "drlmots-demo.html"


def main():
    parser = argparse.ArgumentParser(description="Generate the offline professor demo with paired policy comparisons")
    parser.add_argument("--config", type=Path, default=Path("configs/resources.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demos"))
    args = parser.parse_args()
    try:
        output = generate_demo(args.config, args.output_dir)
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        parser.error("install the demo extra first: python -m pip install -e '.[demo]'")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Demo: {output.resolve()}")
    print("Paired input checks, JSON provenance, and SVG/PNG figures are included.")


if __name__ == "__main__":
    main()
