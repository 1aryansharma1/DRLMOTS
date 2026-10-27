"""Fixed-budget predictive basic-DQN training, validation selection, fresh testing.

Budgets are nested prefixes of the same trajectory per model seed, not independent
training replications. Test scenarios are generated only after every model finishes.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

import torch
import yaml

from drlmots.enhanced_comparison import ComparisonPlan, common_return, load_frozen_pair
from drlmots.enhanced_dqn import EnhancedDQNConfig
from drlmots.enhanced_rl_contract import EnhancedRLContract
from drlmots.enhanced_sanity import _write_json, network_digest
from drlmots.objective_model import failure_trace, generate_work, integer, load_objective_model
from drlmots.objective_sim import simulate_objectives
from drlmots.predictive_dqn import PredictiveDQN, feature_manifest
from drlmots.run_config import _UniqueKeyLoader, load_run_config


BASE_SOURCES = ("enhanced_dqn.py", "enhanced_rl_contract.py", "enhanced_telemetry.py", "objective_sim.py", "objective_model.py", "enhanced_sanity.py")
SELECTION = "trained checkpoints only; minimize (violating validation cases, mean service-floor shortfall, mean penalized cost, episode); all-case feasibility required for promotion"


@dataclass(frozen=True)
class TrainingPlan:
    model_seeds: tuple[int, ...] = (42, 43, 44)
    training_budgets: tuple[int, ...] = (100, 300)
    validation_every_episodes: int = 20
    training_workload_seed_start: int = 30000
    training_failure_seed_start: int = 40000
    validation_workload_seeds: tuple[int, ...] = (84000, 84001, 84002)
    validation_failure_seeds: tuple[int, ...] = (94000, 94001, 94002)
    test_workload_seeds: tuple[int, ...] = (85000, 85001, 85002)
    test_failure_seeds: tuple[int, ...] = (95000, 95001, 95002, 95003, 95004)

    def __post_init__(self):
        for name in ("model_seeds", "training_budgets", "validation_workload_seeds", "validation_failure_seeds", "test_workload_seeds", "test_failure_seeds"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ValueError("plan sequences must be nonempty and distinct")
            for value in values:
                integer(value, name)
            object.__setattr__(self, name, values)
        for name in ("validation_every_episodes", "training_workload_seed_start", "training_failure_seed_start"):
            integer(getattr(self, name), name)
        if not self.validation_every_episodes or self.training_budgets != tuple(sorted(self.training_budgets)) or any(not b or b % self.validation_every_episodes for b in self.training_budgets):
            raise ValueError("positive ordered budgets must align with validation cadence")
        if len(self.validation_workload_seeds) != len(self.validation_failure_seeds):
            raise ValueError("validation workload/failure seeds must form pairs")
        # Previous diagnostic/test panels are now seen evidence, never new training/selection/test inputs.
        seen_work = set(range(10000, 10016)) | {6200, 6201, 6202, 81000, 81001, 81002, 82000, 82001, 82002}
        seen_fault = set(range(20000, 20016)) | set(range(7100, 7105)) | {91000, 91001} | set(range(92000, 92005))
        for start, validation, test, seen in (
            (self.training_workload_seed_start, self.validation_workload_seeds, self.test_workload_seeds, seen_work),
            (self.training_failure_seed_start, self.validation_failure_seeds, self.test_failure_seeds, seen_fault)):
            integer(start + max(self.training_budgets) - 1, "last training seed")
            train = set(range(start, start+max(self.training_budgets)))
            if train & set(validation) or train & set(test) or set(validation) & set(test) or (train | set(validation) | set(test)) & seen:
                raise ValueError("training, validation, fresh test and previously seen panels must be disjoint")

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)} | {"kind", "schema_version"}:
            raise ValueError("training plan requires exactly the documented fields")
        data = dict(data)
        version, kind = data.pop("schema_version"), data.pop("kind")
        if type(version) is not int or version != 1 or kind != "predictive_dqn_training_study":
            raise ValueError("unsupported predictive training plan")
        return cls(**data)

    def manifest(self):
        return dict(schema_version=1, kind="predictive_dqn_training_study", **asdict(self))


def panel(model, work_seeds, failure_seeds, *, crossed):
    scenarios = []
    for i, w in enumerate(work_seeds):
        tasks = generate_work(model, w)
        choices = failure_seeds if crossed else (failure_seeds[i],)
        for f in (None, *choices):
            scenarios.append(dict(mode="no_failures" if f is None else "failures_with_recovery",
                workload_seed=w, failure_seed=f, tasks=tasks,
                outages=[] if f is None else failure_trace(model, f)))
    return scenarios


def episode(agent, run, model, *, learn=False, **kwargs):
    """Freeze auditing is performed once around the complete evaluation panel."""
    report = simulate_objectives(run, model, policy="masked_dqn", controller=agent.controller(learn=learn),
                                 recovery_enabled=agent.settings.recovery_enabled, **kwargs)
    report["rl"].update(learner_updates=agent.updates, learner_decisions=agent.decisions)
    if isinstance(agent, PredictiveDQN):
        report["rl"].update(predictive_features=agent.features, network_feature_count=agent.features["network_feature_count"])
    if not learn and (report["rl"]["learn"] or any(d["epsilon"] != 0 for d in report["rl"]["decisions"])):
        raise RuntimeError("evaluation is not frozen greedy inference")
    return report


def record(report, contract, **metadata):
    value = common_return(contract, report)
    return dict(**metadata, metrics={**report["metrics"], "penalized_return": value},
                feasible=report["objective"]["feasible"], objective_score=report["objective"]["score"],
                workload_sha256=report["workload"]["sha256"], outage_sha256=report["outage_trace"]["sha256"])


def validation_rank(rows, episode_number, model):
    o = model.data["objective"]
    shortfalls = []
    for row in rows:
        m = row["metrics"]
        shortfall = max(0, o["minimum_completion_rate"]-m["completion_rate"])
        if m["deadline_success_rate"] is not None:
            shortfall += max(0, o["minimum_deadline_success_rate"]-m["deadline_success_rate"])
        shortfalls.append(shortfall)
    return (sum(not r["feasible"] for r in rows), math.fsum(shortfalls)/len(rows),
            -math.fsum(r["metrics"]["penalized_return"] for r in rows)/len(rows), episode_number)


def evaluate(agent, run, model, scenarios, output, *, label, model_seed, stage=None):
    before = agent.fingerprint
    rows = []
    for s in scenarios:
        report = episode(agent, run, model, tasks=s["tasks"], outages=s["outages"], failure_seed=s["failure_seed"] or 0)
        path = f"{label}/seed-{model_seed}/" + (f"episode-{stage:03d}-" if stage is not None else "") + f"work-{s['workload_seed']}-fault-{s['failure_seed']}.json"
        _write_json(output/path, report, compact=True)
        rows.append(record(report, agent.contract, method=label, model_seed=model_seed, episode=stage,
            mode=s["mode"], workload_seed=s["workload_seed"], failure_seed=s["failure_seed"], trace_file=path))
    if agent.fingerprint != before:
        raise RuntimeError("frozen panel changed network/optimizer/replay/RNG/counters")
    return rows, before


def _csv(path, rows):
    if not rows:
        return
    flattened = [{**{k: v for k, v in r.items() if k != "metrics"}, **r.get("metrics", {})} for r in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def run_study(run, model, settings, plan, output_dir, *, baseline_dir=None):
    if type(settings) is not EnhancedDQNConfig or type(plan) is not TrainingPlan or not settings.recovery_enabled:
        raise ValueError("typed recovery-enabled DQN settings and study plan required")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("choose an empty output directory; previous evidence is never overwritten")
    contract = EnhancedRLContract.from_configs(run, model)
    _, host_nodes = model.validate_topology(run)
    baseline, baseline_metadata = {}, None
    if baseline_dir is not None:
        baseline, _, baseline_metadata = load_frozen_pair(run, model, baseline_dir,
            ComparisonPlan(plan.test_workload_seeds, plan.test_failure_seeds))
    sources = (*BASE_SOURCES, "predictive_dqn.py", "predictive_training.py")
    source_hashes = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources}
    manifest = dict(kind="predictive_dqn_training_study", status="running", plan=plan.manifest(),
        run_configuration=run.manifest(), model_configuration=model.manifest(), learner_configuration=asdict(settings),
        features=feature_manifest(contract), selection_rule=SELECTION, baseline=baseline_metadata,
        sources_sha256=source_hashes, runtime=dict(python=platform.python_version(), torch=torch.__version__, threads=1, device="cpu"),
        scope="predictive inputs and longer basic DQN only; no Double DQN, demonstrations, prioritized replay or reward/mask changes")
    _write_json(output/"experiment.json", manifest)
    validations = panel(model, plan.validation_workload_seeds, plan.validation_failure_seeds, crossed=False)
    training_rows, stages, selections, audits = [], [], {}, []
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        for seed in plan.model_seeds:
            agent = PredictiveDQN.from_configs(run, model, replace(settings, seed=seed))
            root = output/f"models/seed-{seed}"
            root.mkdir(parents=True, exist_ok=True)
            agent.save(root/"initialization.json")
            initial_online = network_digest(agent.online)
            selected = {}
            rows, frozen = evaluate(agent, run, model, validations, output, label="validation", model_seed=seed, stage=0)
            stages.append(dict(model_seed=seed, episode=0, updates=0, epsilon=agent.epsilon,
                frozen_fingerprint=frozen, rows=rows, selection_rank=validation_rank(rows, 0, model)))
            _write_json(root/"validation-000.json", stages[-1])
            with gzip.open(root/"training-traces.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as traces, (root/"losses.csv").open("w", newline="", encoding="utf-8") as losses:
                loss_writer = csv.writer(losses)
                loss_writer.writerow(("update", "loss"))
                for ep in range(1, max(plan.training_budgets)+1):
                    previous_updates = agent.updates
                    w, f = plan.training_workload_seed_start+ep-1, plan.training_failure_seed_start+ep-1
                    report = episode(agent, run, model, learn=True, workload_seed=w, failure_seed=f)
                    report.update(model_seed=seed, training_episode=ep)
                    traces.write(json.dumps(report, allow_nan=False)+"\n")
                    row = record(report, contract, model_seed=seed, episode=ep, workload_seed=w, failure_seed=f,
                        optimizer_updates=agent.updates, epsilon=agent.epsilon,
                        initial_transitions=sum(x["state"]["placement_kind"] == "initial" for x in report["rl"]["transitions"]),
                        recovery_transitions=sum(x["state"]["placement_kind"] == "recovery" for x in report["rl"]["transitions"]))
                    training_rows.append(row)
                    for i in range(previous_updates, agent.updates):
                        loss_writer.writerow((i+1, agent.losses[i]))
                    if ep % plan.validation_every_episodes == 0:
                        rows, frozen = evaluate(agent, run, model, validations, output, label="validation", model_seed=seed, stage=ep)
                        rank = validation_rank(rows, ep, model)
                        improved = [b for b in plan.training_budgets if ep <= b and (b not in selected or rank < tuple(selected[b]["rank"]))]
                        if improved:
                            candidate = root/"candidate.json"
                            agent.save(candidate)
                            for budget in improved:
                                filename = f"models/seed-{seed}/selected-budget-{budget}.json"
                                shutil.copyfile(candidate, output/filename)
                                selected[budget] = dict(episode=ep, updates=agent.updates, rank=rank,
                                    eligible_for_promotion=rank[0] == 0, fingerprint=frozen, file=filename)
                        stage = dict(model_seed=seed, episode=ep, updates=agent.updates, epsilon=agent.epsilon,
                            frozen_fingerprint=frozen, rows=rows, selection_rank=rank, selected_for_budgets=improved)
                        stages.append(stage)
                        _write_json(root/f"validation-{ep:03d}.json", stage)
                        _write_json(root/"selection.json", selected)
                        _csv(output/"training.csv", training_rows)
                        print(f"seed={seed} episode={ep}/{max(plan.training_budgets)} updates={agent.updates} epsilon={agent.epsilon:.4f} validation_violations={rank[0]} selected=" + str({b: d["episode"] for b, d in selected.items()}), flush=True)
                        _write_json(output/"progress.json", dict(model_seed=seed, episode=ep, completed_model_seeds=list(selections), elapsed_seconds=time.monotonic()-started))
            agent.save(root/"final.json")
            if agent.updates == 0 or network_digest(agent.online) == initial_online:
                raise RuntimeError("longer training produced no verified learning change")
            selections[str(seed)] = selected
            audits.append(dict(model_seed=seed, episodes=agent.episodes, updates=agent.updates, decisions=agent.decisions,
                final_epsilon=agent.epsilon, initial_online_sha256=initial_online, final_online_sha256=network_digest(agent.online)))
            del agent
        # Selection is complete for EVERY seed before any new held-out input is generated.
        _write_json(output/"selection-locked.json", dict(selections=selections, all_training_complete=True,
            test_inputs_generated=False, selection_rule=SELECTION))
        print("All models complete; selections locked. Generating the fresh held-out panel.", flush=True)
        tests = panel(model, plan.test_workload_seeds, plan.test_failure_seeds, crossed=True)
        test_rows, frozen_audits = [], []
        for seed in plan.model_seeds:
            variants = [("predictive_untrained", f"models/seed-{seed}/initialization.json")]
            variants += [(f"predictive_budget_{b}", selections[str(seed)][b]["file"]) for b in plan.training_budgets]
            for label, filename in variants:
                agent = PredictiveDQN.load(output/filename, contract, host_nodes=host_nodes)
                rows, frozen = evaluate(agent, run, model, tests, output, label=label, model_seed=seed)
                test_rows.extend(rows)
                frozen_audits.append(dict(method=label, model_seed=seed, fingerprint=frozen, unchanged=True,
                    selected_episode=agent.episodes, optimizer_updates=agent.updates))
                print(f"Held-out {label} seed={seed}: {len(rows)} cases; frozen verified", flush=True)
                del agent
        for label, agent in baseline.items():
            rows, frozen = evaluate(agent, run, model, tests, output, label="original_"+label, model_seed=agent.settings.seed)
            test_rows.extend(rows)
            frozen_audits.append(dict(method="original_"+label, model_seed=agent.settings.seed,
                fingerprint=frozen, unchanged=True, selected_episode=agent.episodes, optimizer_updates=agent.updates))
        for s in tests:
            report = simulate_objectives(run, model, policy="multi_objective", tasks=s["tasks"], outages=s["outages"],
                                         failure_seed=s["failure_seed"] or 0, recovery_enabled=True)
            filename = f"without_dqn/work-{s['workload_seed']}-fault-{s['failure_seed']}.json"
            _write_json(output/filename, report, compact=True)
            test_rows.append(record(report, contract, method="without_dqn", model_seed=None, episode=None,
                mode=s["mode"], workload_seed=s["workload_seed"], failure_seed=s["failure_seed"], trace_file=filename))
        for s in tests:
            group = [r for r in test_rows if r["mode"] == s["mode"] and r["workload_seed"] == s["workload_seed"] and r["failure_seed"] == s["failure_seed"]]
            if len({r["workload_sha256"] for r in group}) != 1 or len({r["outage_sha256"] for r in group}) != 1:
                raise RuntimeError("fresh held-out inputs differ between methods")
        current_sources = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources}
        if current_sources != source_hashes:
            raise RuntimeError("training implementation changed during the experiment")
        result = dict(kind="predictive_dqn_training_results", selections=selections, training_audits=audits,
            validation_stages=stages, test_rows=test_rows, frozen_audits=frozen_audits,
            training_episode_count=len(training_rows), elapsed_seconds=time.monotonic()-started,
            baseline_preserved=True, test_generated_after_selection=True)
        _write_json(output/"result.json", result)
        _csv(output/"training.csv", training_rows)
        _csv(output/"test-metrics.csv", test_rows)
        _write_json(output/"experiment.json", {**manifest, "status": "complete"})
        return result
    except BaseException as error:
        _write_json(output/"experiment.json", {**manifest, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error": str(error)})
        raise
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl_network.yaml")
    parser.add_argument("--model-config", default="configs/objective_model.yaml")
    parser.add_argument("--dqn-config", default="configs/enhanced_dqn.yaml")
    parser.add_argument("--plan-config", default="configs/predictive_training.yaml")
    parser.add_argument("--baseline-dir", default="outputs/objective-model/dqn-sanity/milestone4-20261027")
    parser.add_argument("--output-dir", default="outputs/objective-model/predictive-training/study-20261027")
    args = parser.parse_args(argv)
    read_yaml = lambda p: yaml.load(Path(p).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    run_study(load_run_config(args.config), load_objective_model(args.model_config),
        EnhancedDQNConfig.parse(read_yaml(args.dqn_config)), TrainingPlan.parse(read_yaml(args.plan_config)),
        args.output_dir, baseline_dir=args.baseline_dir)
    print(f"Saved {args.output_dir}/result.json; training/validation/test complete")


if __name__ == "__main__":
    main()
