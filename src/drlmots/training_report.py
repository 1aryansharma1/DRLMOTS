"""Static, offline professor report from actual training and paired held-out runs."""

from html import escape
import json
from pathlib import Path
import zipfile

from dataclasses import replace

from drlmots.normalization import _digest
from drlmots.q_network import NetworkConfig, QNetwork
from drlmots.rl_state import learning_context
from drlmots.train import (checkpoint_artifact, continue_training, cpu_execution, evaluate,
                           initialize, load_checkpoint, save_checkpoint, validate_split_plan, write_json)


def benchmark_and_report(standard_config, network_config, settings, output_dir, *, resume=False):
    try:
        import matplotlib  # Check optional report dependency before any training starts.
    except ModuleNotFoundError as error:
        raise ValueError("benchmark report needs the demo extra: pip install -e '.[demo]'") from error
    if standard_config.placement is not None or network_config.placement is None:
        raise ValueError("benchmark requires a zero-network standard config and an explicit network config")
    output = Path(output_dir)
    model_names = ("standard-zero", "standard-network", "fog-preferred-network")
    if not resume and any((output / name / "checkpoint.json").exists() for name in model_names):
        raise ValueError("benchmark output already contains checkpoints; choose a fresh output directory")
    specs = [(model_names[0], standard_config, "standard"),
             (model_names[1], network_config, "standard"),
             (model_names[2], network_config, "fog_preferred")]
    agents, summaries = {}, {}
    with cpu_execution():
        for name, config, variant in specs:
            print(f"Training {name}", flush=True)
            existing = output / name / "checkpoint.json"
            if existing.exists():
                learner = load_checkpoint(existing, for_resume=True)
                if (learner.variant != variant or learner.normalizer.context != learning_context(config, variant)
                        or replace(learner.config, episodes=settings.episodes) != settings
                        or learner.episodes_completed > settings.episodes
                        or learner.run_config.workload != config.workload):
                    raise ValueError("saved benchmark differs from requested config/settings")
                learner.config = settings
                validate_split_plan(config, settings)
            else:
                learner = initialize(config, settings, variant)
            summary = continue_training(learner, output_dir=output / name)
            # Recreate the documented initialization, preserving global RNG, for change evidence.
            initial = QNetwork(learner.context, NetworkConfig(settings.hidden_widths, settings.seed))
            summary["initial_weights_sha256"] = _digest(initial.artifact()["weights"])
            save_checkpoint(learner, output / name / "checkpoint.json")
            if learner.updates == 0 or summary["initial_weights_sha256"] == summary["final_weights_sha256"]:
                raise ValueError("training produced no actual parameter updates")
            checkpoint = output / name / "checkpoint.json"
            restored = load_checkpoint(checkpoint, for_resume=True)
            if checkpoint_artifact(restored)["sha256"] != checkpoint_artifact(learner)["sha256"]:
                raise RuntimeError("full training checkpoint did not round-trip exactly")
            agents[name], summaries[name] = restored, summary
        zero = evaluate({"standard_dqn": agents[model_names[0]]}, standard_config, settings.evaluation_seeds)
        network = evaluate({"standard_dqn": agents[model_names[1]], "fog_preferred_dqn": agents[model_names[2]]},
                           network_config, settings.evaluation_seeds)
    write_json(output / "standard-evaluation.json", zero)
    write_json(output / "network-evaluation.json", network)
    manifest = {"kind": "actual_live_dqn_benchmark", "training_config": settings.manifest(),
                "models": summaries, "normalizer_scope": "training calibration only; frozen for training/evaluation",
                "comparison": "paired inputs within each environment; never compare zero/network timings as identical inputs",
                "fog_variant": "explicit preference constraint, with learned placement among allowed VMs",
                "scope": "synthetic CPU experiments; no paper reproduction or universal superiority claim"}
    write_json(output / "benchmark.json", manifest)
    render_report(output, agents, zero, network, manifest)
    return manifest


def render_report(output, agents, zero, network, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name, learner in agents.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
        axes[0].plot([row["episode"] for row in learner.history], [row["return"] for row in learner.history], color="#147c75")
        axes[0].set(xlabel="Training episode", ylabel="All-arrival undiscounted return", title="Actual exploratory training returns")
        axes[1].plot(range(1, len(learner.losses) + 1), learner.losses, color="#466385", linewidth=0.7)
        axes[1].set(xlabel="Optimizer update", ylabel="Smooth L1 loss", title="Replay learning loss (raw updates)")
        for ax in axes:
            ax.grid(alpha=0.2)
        for extension in ("png", "svg"):
            fig.savefig(figures / f"{name}-training.{extension}", dpi=150)
        plt.close(fig)
    evaluations = (("standard", "Standard zero-network environment", zero),
                   ("network", "Network-aware environment", network))
    sections = []
    for key, title, evaluation in evaluations:
        rows = evaluation["summaries"]
        fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
        labels = [row["policy"].replace("_", " ") for row in rows]
        completed = [row["completed_count"]["mean"] for row in rows]
        failed = [row["failed_count"]["mean"] for row in rows]
        ax.barh(labels, completed, color="#147c75", label="Completed")
        ax.barh(labels, failed, left=completed, color="#dca164", label="Rejected")
        ax.invert_yaxis()
        ax.set(xlabel=f"Mean tasks across {len(evaluation['evaluation_seeds'])} held-out workload seeds", title=title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncols=2, frameon=False)
        for extension in ("png", "svg"):
            fig.savefig(figures / f"{key}-comparison.{extension}", dpi=150)
        plt.close(fig)
        table = []
        for row in rows:
            mean = row["successful_mean_completion_ms"]["mean"]
            table.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
                row["policy"], f"{row['completed_count']['mean']:.2f}", f"{row['failed_count']['mean']:.2f}",
                "—" if mean is None else f"{mean:.3f}", f"{row['mean_all_arrival_return']:.3f}")) + "</tr>")
        sections.append(f'<section><h2>{escape(title)}</h2><img src="figures/{key}-comparison.svg" alt="Paired held-out completion comparison">'
                        '<div class="scroll"><table><tr><th>Policy</th><th>Mean done</th><th>Mean rejected</th><th>Successful mean ms</th><th>Mean return</th></tr>'
                        + "".join(table) + f'</table></div><p><a href="{key}-evaluation.json" download>Download all held-out runs and actual placement traces</a></p></section>')
    models = []
    for name, learner in agents.items():
        models.append(f'<section><h2>{escape(name)}</h2><p>{learner.episodes_completed} episodes · {learner.updates} actual optimizer updates · '
                      f'{len(learner.replay)} replay transitions · state dimension {learner.normalizer.context.schema.dimension}</p>'
                      f'<img src="figures/{name}-training.svg" alt="Training rewards and optimizer losses">'
                      f'<p><a href="{name}/checkpoint.json" download>Resumable checkpoint</a> · '
                      f'<a href="{name}/history.json" download>Full logs</a> · '
                      f'<a href="{name}/last-training-episode.json" download>Last live training episode</a></p></section>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DRLMOTS · Trained policy results</title><style>
body{margin:0;background:#f2f6f6;color:#183040;font:16px/1.55 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 18px}
h1{font-size:34px;line-height:1.15}h2{font-size:23px}section{background:white;border:1px solid #dbe5e7;border-radius:12px;margin:24px 0;padding:24px}
img{width:100%;height:auto}a{color:#096e68}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #dde5e8;padding:10px}
.notice{background:#e4efeb;border-left:4px solid #147c75;padding:16px}details{margin-top:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
</style><main><h1>Live task placement with trained DQN</h1>
<p>Standard mode is retained. Network-aware mode includes input upload, queue waiting and fog/cloud completion estimates.</p>
<p class="notice">These policies actually selected VMs inside the simulator during training and held-out evaluation. Evaluation uses greedy inference, frozen training-only scaling and no optimizer updates. The fog-preferred variant has an explicit completion-time constraint; that preference is not claimed as a learned discovery.</p>
<p>Compare policies within each environment below. Successful-task latency excludes rejections, so read completion counts and all-arrival return alongside it. These are synthetic experiments, not paper reproduction or evidence that DQN always outperforms rules.</p>
'''
    html += "".join(sections) + "".join(models)
    html += '<details><summary>Experiment manifest</summary><pre>' + escape(json.dumps(manifest, indent=2)) + '</pre></details>'
    html += '<p><a href="benchmark.json" download>Download benchmark manifest</a>. This saved report displays completed runs; opening it does not start or repeat training.</p></main></html>'
    (output / "training-demo.html").write_text(html, encoding="utf-8")
    files = [output / name for name in ("training-demo.html", "benchmark.json", "standard-evaluation.json", "network-evaluation.json")]
    for name in agents:
        files.extend(output / name / filename for filename in ("checkpoint.json", "history.json", "last-training-episode.json"))
        files.extend(figures / f"{name}-training.{extension}" for extension in ("png", "svg"))
    files.extend(figures / f"{mode}-comparison.{extension}" for mode in ("standard", "network") for extension in ("png", "svg"))
    with zipfile.ZipFile(output / "professor-demo.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(output)))
