"""Train/resume/evaluate live CPU DQN while retaining both environment modes."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import platform
import statistics
import tempfile
import time

import torch
import yaml

from drlmots.__main__ import configured_report
from drlmots.baselines import BASELINE_POLICIES, FOG_FIRST_POLICY
from drlmots.dqn import (BaselineController, EpisodeController, Experience, Learner, TrainingConfig,
                         restore_rng, reward_summary)
from drlmots.normalization import StateNormalizer, TraceBatch, _canonical, _digest, _read_json, fit_normalizer
from drlmots.q_network import QNetwork
from drlmots.rl_state import RLRewardConfig, RLStateSchema, VARIANTS, learning_context
from drlmots.run_config import _UniqueKeyLoader, load_run_config, parse_run_config
from drlmots.workload import seeded_workload


@contextmanager
def cpu_execution():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.set_num_threads(previous)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical(data) + "\n")
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError as error:
                # Windows scanners/readers can briefly deny replacing an existing file.
                # Keep the previous checkpoint intact; never delete it to force a save.
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 9:
                    raise
                time.sleep(min(0.025 * 2**attempt, 0.5))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def seeded_config(config, seed):
    return replace(config, workload=replace(config.workload, seed=seed))


def task_hash(config):
    return _digest([asdict(task) for task in seeded_workload(config.workload)])


def fit_training_states(config, settings, variant):
    context = learning_context(config, variant)
    reward = RLRewardConfig(settings.latency_scale_ms, settings.rejection_penalty)
    batches = []
    actionable = False
    for offset in range(settings.calibration_episodes):
        run = seeded_config(config, settings.training_seed_start + offset)
        controller = BaselineController(context.schema, reward)
        report = configured_report(run, policy="least_queue_fcfs", controller=controller)
        actionable = actionable or any(record["action"] is not None for record in controller.records)
        provenance = {"run_configuration": report["run_configuration"], "workload_sha256": report["workload"]["sha256"],
                      "tasks_sha256": _digest(report["workload"]["tasks"]), "policy": "least_queue_fcfs",
                      "software_version": report["version"], "purpose": "training-only calibration"}
        batches.append(TraceBatch(context, tuple(controller.states), "train", _digest(provenance), _canonical(provenance)))
    if not actionable:
        raise ValueError("training workload has no legal placement; check resources and routing")
    return fit_normalizer(batches)


def run_episode(learner, config, *, learn=False):
    if learning_context(config, learner.variant) != learner.normalizer.context:
        raise ValueError("episode topology/network/variant does not match training context")
    controller = EpisodeController(learner, learn=learn)
    label = "standard_dqn" if learner.variant == "standard" else "fog_preferred_dqn"
    report = configured_report(config, policy=label, controller=controller)
    controller.finish()
    report["implementation_scope"] = "synthetic live DQN-controlled simulator; not paper reproduction"
    report["milestone"] = "live CPU DQN"
    report["rl"] = {"contract": 2, "variant": learner.variant, "learn": learn,
                    "context_sha256": learner.context_sha, "reward": learner.reward.manifest(),
                    "summary": reward_summary(controller.records), "decisions": controller.records,
                    "transitions_added": controller.transitions_added,
                    "fog_preference": "explicit action constraint; DQN chooses inside that constraint"
                    if learner.variant == "fog_preferred" else "unconstrained among all resource/queue-legal VMs"}
    if len(controller.records) != len(report["outcomes"]) or any(record["vm_id"] != outcome["vm_id"] for record, outcome in zip(controller.records, report["outcomes"])):
        raise RuntimeError("agent choices were not executed by the simulator")
    return report


def validate_split_plan(config, settings):
    training_hashes = {task_hash(seeded_config(config, settings.training_seed_start + offset))
                       for offset in range(max(settings.episodes, settings.calibration_episodes))}
    if any(task_hash(seeded_config(config, seed)) in training_hashes for seed in settings.evaluation_seeds):
        raise ValueError("evaluation tasks overlap planned training/calibration even with another seed")


def initialize(config, settings, variant):
    validate_split_plan(config, settings)
    normalizer = fit_training_states(config, settings, variant)
    learner = Learner(normalizer, settings, variant)
    learner.run_config = config
    learner.training_task_hashes = {source["provenance"]["tasks_sha256"]
                                     for source in normalizer.manifest()["training_sources"]}
    return learner


def continue_training(learner, *, target_episodes=None, output_dir=None):
    goal = learner.config.episodes if target_episodes is None else target_episodes
    if type(goal) is not int or goal < learner.episodes_completed or goal > learner.config.episodes:
        raise ValueError("training target must be between completed and configured episodes")
    initial_weights = _digest(learner.online.artifact()["weights"])
    last = None
    with cpu_execution():
        for episode in range(learner.episodes_completed, goal):
            seed = learner.config.training_seed_start + episode
            config = seeded_config(learner.run_config, seed)
            learner.training_task_hashes.add(task_hash(config))
            before_updates = learner.updates
            last = run_episode(learner, config, learn=True)
            learner.episodes_completed += 1
            summary = {"episode": learner.episodes_completed, "seed": seed,
                       "completed_count": last["completed_count"], "failed_count": last["failed_count"],
                       "makespan_ms": last["makespan_ms"], "average_completion_time_ms": last["average_completion_time_ms"],
                       "return": last["rl"]["summary"]["undiscounted_return"], "updates": learner.updates,
                       "episode_updates": learner.updates - before_updates, "epsilon": learner.epsilon,
                       "workload_sha256": last["workload"]["sha256"]}
            learner.history.append(summary)
            if output_dir is not None:
                # Commit checkpoint last: a completed checkpoint always has matching logs.
                write_json(Path(output_dir) / "last-training-episode.json", last)
                write_json(Path(output_dir) / "history.json", {"episodes": learner.history, "losses": learner.losses})
                save_checkpoint(learner, Path(output_dir) / "checkpoint.json")
            print(f"{learner.variant}: episode {learner.episodes_completed}/{goal}, "
                  f"completed={last['completed_count']}, rejected={last['failed_count']}, updates={learner.updates}", flush=True)
    if output_dir is not None and last is not None:
        write_json(Path(output_dir) / "last-training-episode.json", last)
    return {"initial_weights_sha256": initial_weights,
            "final_weights_sha256": _digest(learner.online.artifact()["weights"]),
            "updates": learner.updates, "episodes_completed": learner.episodes_completed}


def _optimizer_artifact(learner):
    moments = {}
    for name, parameter in learner.online.named_parameters():
        state = learner.optimizer.state.get(parameter)
        if state:
            if float(state["step"]) != learner.updates:
                raise ValueError("optimizer step disagrees with update count")
            moments[name] = {key: state[key].detach().tolist() for key in ("exp_avg", "exp_avg_sq")}
    return {"kind": "adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
            "learning_rate": learner.config.learning_rate, "step": learner.updates, "moments": moments}


def checkpoint_artifact(learner):
    if learner.run_config is None:
        raise ValueError("checkpoint requires the training run configuration")
    payload = {"schema_version": 2, "kind": "live_dqn_training", "variant": learner.variant,
               "runtime": {"torch": str(torch.__version__), "python": platform.python_version()},
               "training_config": learner.config.manifest(), "run_config": learner.run_config.manifest(),
               "normalizer": learner.normalizer.manifest(), "context_sha256": learner.context_sha,
               "online": learner.online.artifact(), "target": learner.target.artifact(),
               "optimizer": _optimizer_artifact(learner), "updates": learner.updates,
               "decision_steps": learner.decision_steps, "episodes_completed": learner.episodes_completed,
               "selector_rng": learner.selector._rng.getstate(), "replay_rng": learner.replay.rng.getstate(),
               "replay": [asdict(item) for item in learner.replay.items],
               "training_task_hashes": sorted(learner.training_task_hashes),
               "history": learner.history, "losses": learner.losses}
    return {**payload, "sha256": _digest(payload)}


def save_checkpoint(learner, path):
    artifact = checkpoint_artifact(learner)
    write_json(path, artifact)
    return artifact["sha256"]


def load_checkpoint(path, *, for_resume=False):
    artifact = _read_json(Path(path).read_text(encoding="utf-8"))
    fields = {"schema_version", "kind", "variant", "runtime", "training_config", "run_config", "normalizer",
              "context_sha256", "online", "target", "optimizer", "updates", "decision_steps", "episodes_completed",
              "selector_rng", "replay_rng", "replay", "training_task_hashes", "history", "losses", "sha256"}
    if not isinstance(artifact, dict) or set(artifact) != fields:
        raise ValueError("invalid training checkpoint fields")
    if (type(artifact["schema_version"]) is not int or artifact["schema_version"] != 2
            or artifact["kind"] != "live_dqn_training" or artifact["variant"] not in VARIANTS):
        raise ValueError("unsupported training checkpoint version/kind/variant")
    if artifact["sha256"] != _digest({key: value for key, value in artifact.items() if key != "sha256"}):
        raise ValueError("training checkpoint checksum mismatch")
    runtime = {"torch": str(torch.__version__), "python": platform.python_version()}
    if (not isinstance(artifact["runtime"], dict) or set(artifact["runtime"]) != set(runtime)
            or any(not isinstance(value, str) for value in artifact["runtime"].values())):
        raise ValueError("invalid training runtime metadata")
    if for_resume and artifact["runtime"] != runtime:
        raise ValueError("exact resume requires the saved Python/Torch runtime")
    settings = TrainingConfig.parse(artifact["training_config"])
    config_data = dict(artifact["run_config"])
    config_data.pop("sha256", None)
    config = parse_run_config(config_data)
    if _canonical(config.manifest()) != _canonical(artifact["run_config"]):
        raise ValueError("training run configuration checksum mismatch")
    context = learning_context(config, artifact["variant"])
    scaler = StateNormalizer.from_manifest(artifact["normalizer"], context=context)
    learner = Learner(scaler, settings, artifact["variant"])
    learner.run_config = config
    if learner.context_sha != artifact["context_sha256"]:
        raise ValueError("training checkpoint context mismatch")
    for key in ("updates", "decision_steps", "episodes_completed"):
        if type(artifact[key]) is not int or artifact[key] < 0:
            raise ValueError("checkpoint counters must be nonnegative integers")
        setattr(learner, key, artifact[key])
    if learner.episodes_completed > settings.episodes:
        raise ValueError("completed episodes exceed configured budget")
    online = QNetwork.from_artifact(artifact["online"], context=learner.context)
    target = QNetwork.from_artifact(artifact["target"], context=learner.context)
    expected_architecture = learner.online.config
    if online.config != expected_architecture or target.config != expected_architecture:
        raise ValueError("checkpoint architecture disagrees with training config")
    learner.online.load_state_dict(online.state_dict())
    learner.target.load_state_dict(target.state_dict())
    optimizer = artifact["optimizer"]
    expected_optimizer = _optimizer_artifact(learner)
    if (not isinstance(optimizer, dict) or set(optimizer) != set(expected_optimizer)
            or any(optimizer[key] != expected_optimizer[key] for key in expected_optimizer if key != "moments")):
        raise ValueError("incompatible optimizer settings/step")
    parameters = dict(learner.online.named_parameters())
    expected_names = set(parameters) if learner.updates else set()
    if not isinstance(optimizer["moments"], dict) or set(optimizer["moments"]) != expected_names:
        raise ValueError("optimizer moment names mismatch")
    for name, values in optimizer["moments"].items():
        if not isinstance(values, dict) or set(values) != {"exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid optimizer moments")
        parameter = parameters[name]
        state = {"step": torch.tensor(float(learner.updates), dtype=torch.float32)}
        for key, raw in values.items():
            def numbers(value):
                if isinstance(value, list):
                    return all(numbers(item) for item in value)
                return type(value) in (int, float) and math.isfinite(value) and abs(value) <= torch.finfo(torch.float32).max
            if not numbers(raw):
                raise ValueError("optimizer moments must contain finite float32 numbers")
            tensor = torch.tensor(raw, dtype=torch.float32, device="cpu")
            if tensor.shape != parameter.shape or not torch.isfinite(tensor).all() or (key == "exp_avg_sq" and (tensor < 0).any()):
                raise ValueError("optimizer moment shape/values mismatch")
            state[key] = tensor
        learner.optimizer.state[parameter] = state
    restore_rng(learner.selector._rng, artifact["selector_rng"])
    restore_rng(learner.replay.rng, artifact["replay_rng"])
    if not isinstance(artifact["replay"], list) or len(artifact["replay"]) > settings.replay_capacity:
        raise ValueError("invalid replay capacity/content")
    for raw in artifact["replay"]:
        if not isinstance(raw, dict) or set(raw) != set(Experience.__dataclass_fields__):
            raise ValueError("invalid replay record fields")
        item = Experience(**raw)
        if item.bootstrap_discount != (0.0 if item.done else settings.gamma ** item.discount_steps):
            raise ValueError("replay discount disagrees with training gamma/forced steps")
        learner.replay.add(item, context=learner.context_sha)
    hashes = artifact["training_task_hashes"]
    if (not isinstance(hashes, list) or len(hashes) != len(set(hashes))
            or any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes)):
        raise ValueError("invalid training workload provenance")
    learner.training_task_hashes = set(hashes)
    calibration = {source["provenance"]["tasks_sha256"] for source in scaler.manifest()["training_sources"]}
    if not calibration <= learner.training_task_hashes:
        raise ValueError("training provenance omits calibration tasks")
    history, losses = artifact["history"], artifact["losses"]
    if not isinstance(history, list) or len(history) != learner.episodes_completed:
        raise ValueError("training history/counter mismatch")
    expected_fields = {"episode", "seed", "completed_count", "failed_count", "makespan_ms", "average_completion_time_ms",
                       "return", "updates", "episode_updates", "epsilon", "workload_sha256"}
    previous_updates, admitted = 0, 0
    for index, row in enumerate(history):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("invalid episode history fields")
        for key in ("episode", "seed", "completed_count", "failed_count", "updates", "episode_updates"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("invalid episode history counters")
        if (row["episode"] != index + 1 or row["seed"] != settings.training_seed_start + index
                or row["completed_count"] + row["failed_count"] != config.workload.task_count
                or row["updates"] - previous_updates != row["episode_updates"] or row["updates"] < previous_updates):
            raise ValueError("episode history identity/accounting mismatch")
        for key in ("makespan_ms", "return", "epsilon"):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError("episode metrics must be finite")
        if row["makespan_ms"] < 0 or not 0 <= row["epsilon"] <= 1:
            raise ValueError("episode metrics out of range")
        mean = row["average_completion_time_ms"]
        if (row["completed_count"] == 0 and mean is not None) or (row["completed_count"] > 0 and
                (type(mean) not in (int, float) or not math.isfinite(mean) or mean <= 0)):
            raise ValueError("successful latency/count mismatch")
        admitted += row["completed_count"]
        previous_updates = row["updates"]
    if (admitted != learner.decision_steps or previous_updates != learner.updates
            or learner.updates != max(0, learner.decision_steps - settings.batch_size + 1)
            or len(learner.replay) != min(learner.decision_steps, settings.replay_capacity)):
        raise ValueError("training counters/replay/history disagree")
    if (not isinstance(losses, list) or len(losses) != learner.updates
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in losses)):
        raise ValueError("invalid training loss history")
    learner.history, learner.losses = history, losses
    return learner


def _held_out(learner, config, seed):
    maximum = learner.config.training_seed_start + max(learner.config.episodes, learner.config.calibration_episodes)
    if learner.config.training_seed_start <= seed < maximum:
        raise ValueError("evaluation seed overlaps declared training/calibration range")
    if task_hash(config) in learner.training_task_hashes:
        raise ValueError("evaluation tasks overlap training/calibration even with another seed")


def evaluate(learners, config, seeds):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be nonempty and unique")
    if not learners:
        raise ValueError("at least one trained agent is required")
    rewards = {learner.reward.manifest()["sha256"] for learner in learners.values()}
    if len(rewards) != 1:
        raise ValueError("comparison agents must use the same reward contract")
    before = {label: checkpoint_artifact(learner)["sha256"] for label, learner in learners.items()}
    runs = []
    policies = (*BASELINE_POLICIES, FOG_FIRST_POLICY) if config.placement else BASELINE_POLICIES
    if any(not isinstance(label, str) or not label.strip() or label in policies for label in learners):
        raise ValueError("agent labels must be nonempty and distinct from reference policy names")
    with cpu_execution():
        for seed in seeds:
            current = seeded_config(config, seed)
            for learner in learners.values():
                _held_out(learner, current, seed)
                if learning_context(current, learner.variant) != learner.normalizer.context:
                    raise ValueError("evaluation environment differs from training context")
            reports = []
            for policy in policies:
                collector = BaselineController(RLStateSchema.from_config(current), next(iter(learners.values())).reward, policy)
                report = configured_report(current, policy=policy, controller=collector)
                report["rl"] = {"summary": reward_summary(collector.records), "decisions": collector.records,
                                "learn": False, "kind": "reference rule"}
                reports.append(report)
            for label, learner in learners.items():
                report = run_episode(learner, current)
                report["policy"] = label
                reports.append(report)
            if len({report["workload"]["sha256"] for report in reports}) != 1 or len({report["run_configuration"]["sha256"] for report in reports}) != 1:
                raise RuntimeError("evaluation comparison inputs mismatch")
            runs.append({"seed": seed, "reports": reports})
    if before != {label: checkpoint_artifact(learner)["sha256"] for label, learner in learners.items()}:
        raise RuntimeError("held-out evaluation mutated training state")
    summaries = []
    for policy in [report["policy"] for report in runs[0]["reports"]]:
        reports = [next(report for report in run["reports"] if report["policy"] == policy) for run in runs]
        def metric(name):
            values = [report[name] for report in reports if report[name] is not None]
            return {"mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}
        summaries.append({"policy": policy, "seed_count": len(seeds),
                          "completed_count": metric("completed_count"), "failed_count": metric("failed_count"),
                          "makespan_ms": metric("makespan_ms"),
                          "successful_mean_completion_ms": metric("average_completion_time_ms"),
                          "mean_all_arrival_return": statistics.mean(report["rl"]["summary"]["undiscounted_return"] for report in reports)})
    return {"kind": "held_out_live_dqn_comparison", "evaluation_seeds": seeds,
            "scope": "greedy inference, no updates/refitting; paired inputs within each environment/seed",
            "timing_scope": "mean of per-seed successful-task means, not pooled; compare failures/returns; std is sample std",
            "summaries": summaries, "runs": runs}


def load_training_config(path):
    try:
        data = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid training YAML: {error}") from error
    return TrainingConfig.parse(data)


def main():
    parser = argparse.ArgumentParser(description="Live CPU DQN training, exact resume and held-out evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--training-config", type=Path, required=True)
    training.add_argument("--variant", choices=VARIANTS, default="standard")
    training.add_argument("--output-dir", type=Path, required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--checkpoint", type=Path, required=True)
    resume.add_argument("--additional-episodes", type=int, required=True)
    resume.add_argument("--output-dir", type=Path, required=True)
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--checkpoint", type=Path, action="append", required=True)
    scoring.add_argument("--config", type=Path)
    scoring.add_argument("--seeds", type=int, nargs="+")
    scoring.add_argument("--output", type=Path, required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--standard-config", type=Path, required=True)
    benchmark.add_argument("--network-config", type=Path, required=True)
    benchmark.add_argument("--training-config", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--resume", action="store_true", help="Continue matching saved model runs without refitting")
    args = parser.parse_args()
    try:
        if args.command == "train":
            if (args.output_dir / "checkpoint.json").exists():
                raise ValueError("output already has a checkpoint; choose another directory or use resume")
            config, settings = load_run_config(args.config), load_training_config(args.training_config)
            with cpu_execution():
                learner = initialize(config, settings, args.variant)
                result = continue_training(learner, output_dir=args.output_dir)
            if learner.updates == 0:
                raise ValueError("no replay updates occurred; reduce batch size or resume for more episodes; checkpoint is saved")
            write_json(args.output_dir / "training-summary.json", result)
            write_json(args.output_dir / "evaluation.json", evaluate({args.variant + "_dqn": learner}, config, settings.evaluation_seeds))
        elif args.command == "resume":
            if args.additional_episodes <= 0:
                raise ValueError("additional episodes must be positive")
            learner = load_checkpoint(args.checkpoint, for_resume=True)
            destination = args.output_dir / "checkpoint.json"
            if destination.exists() and destination.resolve() != args.checkpoint.resolve():
                raise ValueError("resume output already contains another checkpoint")
            learner.config = replace(learner.config, episodes=learner.episodes_completed + args.additional_episodes)
            validate_split_plan(learner.run_config, learner.config)
            continue_training(learner, output_dir=args.output_dir)
            write_json(args.output_dir / "evaluation.json", evaluate({learner.variant + "_dqn": learner}, learner.run_config,
                                                                    learner.config.evaluation_seeds))
        elif args.command == "evaluate":
            if args.output.resolve() in {path.resolve() for path in args.checkpoint} or (args.config and args.output.resolve() == args.config.resolve()):
                raise ValueError("evaluation output cannot overwrite an input")
            agents = {}
            for path in args.checkpoint:
                learner = load_checkpoint(path)
                label = learner.variant + "_dqn"
                if label in agents:
                    raise ValueError("comparison checkpoints must have distinct variants")
                agents[label] = learner
            config = load_run_config(args.config) if args.config else learner.run_config
            write_json(args.output, evaluate(agents, config, args.seeds or learner.config.evaluation_seeds))
        else:
            from drlmots.training_report import benchmark_and_report
            benchmark_and_report(load_run_config(args.standard_config), load_run_config(args.network_config),
                                 load_training_config(args.training_config), args.output_dir, resume=args.resume)
    except (ValueError, TypeError, OSError, RuntimeError, OverflowError) as error:
        parser.error(str(error))
    print("Completed:", args.output if args.command == "evaluate" else args.output_dir)


if __name__ == "__main__":
    main()
