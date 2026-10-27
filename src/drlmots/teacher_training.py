"""Predeclared BC-then-DQN study with matched RL-only control and fresh test panel."""
import argparse
from dataclasses import asdict, dataclass, fields, replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import yaml

from drlmots.enhanced_comparison import ComparisonPlan, load_frozen_pair
from drlmots.enhanced_dqn import EnhancedDQNConfig
from drlmots.enhanced_rl_contract import EnhancedRLContract
from drlmots.enhanced_sanity import _write_json, network_digest
from drlmots.normalization import _digest
from drlmots.objective_model import integer, load_objective_model, number
from drlmots.objective_sim import simulate_objectives
from drlmots.predictive_dqn import PredictiveDQN
from drlmots.predictive_training import (BASE_SOURCES, SELECTION, TrainingPlan, _csv,
    episode, evaluate, panel, record, validation_rank)
from drlmots.run_config import _UniqueKeyLoader, load_run_config
from drlmots.teacher_dqn import TEACHER, ObservedTeacher, behavior_clone, dataset_from_reports


@dataclass(frozen=True)
class TeacherPlan(TrainingPlan):
    training_budgets: tuple[int, ...] = (300,)
    training_workload_seed_start: int = 51000
    training_failure_seed_start: int = 61000
    validation_workload_seeds: tuple[int, ...] = (86000, 86001, 86002)
    validation_failure_seeds: tuple[int, ...] = (96000, 96001, 96002)
    test_workload_seeds: tuple[int, ...] = (87000, 87001, 87002)
    test_failure_seeds: tuple[int, ...] = (97000, 97001, 97002, 97003, 97004)
    demonstration_episodes: int = 40
    demonstration_workload_seed_start: int = 50000
    demonstration_failure_seed_start: int = 60000
    cloning_epochs: int = 20
    cloning_batch_size: int = 64
    cloning_learning_rate: float = .001

    def __post_init__(self):
        super().__post_init__()
        if len(self.training_budgets) != 1:
            raise ValueError("one predeclared RL budget required")
        for name in ("demonstration_episodes", "cloning_epochs", "cloning_batch_size"):
            integer(getattr(self, name), name)
            if not getattr(self, name):
                raise ValueError("positive teacher counts required")
        number("cloning learning rate", self.cloning_learning_rate, positive=True)
        for key, start, old, validation, test, seen in (
            ("workload", self.demonstration_workload_seed_start, self.training_workload_seed_start,
             self.validation_workload_seeds, self.test_workload_seeds,
             set(range(30000, 30300)) | set(range(84000, 84003)) | set(range(85000, 85003)) | set(range(10000, 10016)) | {6200,6201,6202,81000,81001,81002,82000,82001,82002}),
            ("failure", self.demonstration_failure_seed_start, self.training_failure_seed_start,
             self.validation_failure_seeds, self.test_failure_seeds,
             set(range(40000, 40300)) | set(range(94000, 94003)) | set(range(95000, 95005)) | set(range(20000,20016)) | set(range(7100,7105)) | {91000,91001} | set(range(92000,92005)))):
            integer(start, key+" demonstration seed")
            integer(start+self.demonstration_episodes-1, "last demonstration seed")
            demo = set(range(start, start+self.demonstration_episodes))
            other = set(range(old, old+max(self.training_budgets))) | set(validation) | set(test)
            if demo & other or (demo | other) & seen:
                raise ValueError("demonstrations, RL, validation, test and prior panels must be disjoint")

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)} | {"kind", "schema_version"}:
            raise ValueError("teacher plan requires exactly documented fields")
        data = dict(data)
        kind, version = data.pop("kind"), data.pop("schema_version")
        if kind != "teacher_dqn_study" or type(version) is not int or version != 1:
            raise ValueError("unsupported teacher study plan")
        return cls(**data)

    def manifest(self):
        return dict(schema_version=1, kind="teacher_dqn_study", **asdict(self))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_predictive(run, model, directory):
    """Validate previously locked baseline selection and source provenance."""
    root = Path(directory)
    manifest = json.loads((root/"experiment.json").read_text())
    result = json.loads((root/"result.json").read_text())
    lock = json.loads((root/"selection-locked.json").read_text())
    if manifest["status"] != "complete" or manifest["kind"] != "predictive_dqn_training_study" or not lock["all_training_complete"] or lock["selections"] != result["selections"]:
        raise ValueError("prior predictive baseline must be complete with locked selection")
    if manifest["run_configuration"] != run.manifest() or manifest["model_configuration"] != model.manifest():
        raise ValueError("prior predictive baseline configuration mismatch")
    if any(sha(Path(__file__).with_name(n)) != h for n, h in manifest["sources_sha256"].items()):
        raise ValueError("preserved predictive baseline source mismatch")
    contract = EnhancedRLContract.from_configs(run, model)
    _, hosts = model.validate_topology(run)
    agents, provenance = [], []
    budget = str(max(manifest["plan"]["training_budgets"]))
    for seed, selections in result["selections"].items():
        selection = selections[budget]
        p = root/selection["file"]
        agent = PredictiveDQN.load(p, contract, host_nodes=hosts)
        if agent.fingerprint != selection["fingerprint"] or agent.episodes != selection["episode"]:
            raise ValueError("preserved predictive baseline selection mismatch")
        agents.append(("previous_predictive_dqn", agent))
        provenance.append(dict(path=str(p.resolve()), sha256=sha(p), fingerprint=agent.fingerprint))
    return agents, provenance


def run_study(run, model, settings, plan, output_dir, *, baseline_dir=None, predictive_dir=None):
    if type(plan) is not TeacherPlan or type(settings) is not EnhancedDQNConfig or not settings.recovery_enabled:
        raise ValueError("typed teacher plan and recovery-enabled DQN settings required")
    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("choose an empty output directory; never overwrite evidence")
    contract = EnhancedRLContract.from_configs(run, model)
    _, hosts = model.validate_topology(run)
    originals, prior_meta = {}, []
    if baseline_dir is not None:
        originals, _, original_meta = load_frozen_pair(run, model, baseline_dir,
            ComparisonPlan(plan.test_workload_seeds, plan.test_failure_seeds))
        for filename in ("initialization.json", "trained-final.json"):
            p = Path(baseline_dir)/filename
            prior_meta.append(dict(path=str(p.resolve()), sha256=sha(p)))
    else:
        original_meta = None
    previous, preserved = prior_predictive(run, model, predictive_dir) if predictive_dir else ([], [])
    prior_meta.extend(preserved)
    sources = (*BASE_SOURCES, "predictive_dqn.py", "predictive_training.py", "teacher_dqn.py", "teacher_training.py")
    hashes = {n: sha(Path(__file__).with_name(n)) for n in sources}
    manifest = dict(kind="teacher_dqn_study", status="running", plan=plan.manifest(),
        teacher=TEACHER, selection_rule=SELECTION, run_configuration=run.manifest(),
        model_configuration=model.manifest(), learner_configuration=asdict(settings),
        sources_sha256=hashes, prior_baselines=prior_meta, original_baseline=original_meta,
        runtime=dict(python=platform.python_version(), torch=torch.__version__, device="cpu", threads=1),
        scope="fixed masked behavior-cloning warm start followed by ordinary masked DQN; paired RL-only control; no teacher at DQN inference; no DQfD/PER/Double DQN/reward changes")
    _write_json(output/"experiment.json", manifest)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        teacher = ObservedTeacher.from_configs(run, model, settings)
        before = teacher.fingerprint
        examples, demo_rows = [], []
        with gzip.open(output/"demonstrations.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as traces:
            for ep in range(plan.demonstration_episodes):
                w, f = plan.demonstration_workload_seed_start+ep, plan.demonstration_failure_seed_start+ep
                report = episode(teacher, run, model, workload_seed=w, failure_seed=f)
                traces.write(json.dumps(report, allow_nan=False)+"\n")
                examples.extend(dataset_from_reports([report]))
                demo_rows.append(record(report, contract, episode=ep+1, workload_seed=w, failure_seed=f))
        if teacher.fingerprint != before or not examples:
            raise RuntimeError("teacher collection mutated weights or produced no examples")
        dataset = dict(kind="training_only_teacher_dataset", teacher=TEACHER, episodes=len(demo_rows),
            examples=len(examples), initial_count=sum(s.placement_kind == "initial" for s,a in examples),
            recovery_count=sum(s.placement_kind == "recovery" for s,a in examples),
            sha256=_digest([dict(state=asdict(s), action=a) for s,a in examples]),
            trace_sha256=sha(output/"demonstrations.jsonl.gz"))
        _write_json(output/"dataset.json", dataset)
        _csv(output/"demonstrations.csv", demo_rows)
        validations = panel(model, plan.validation_workload_seeds, plan.validation_failure_seeds, crossed=False)
        selections, stages, training_rows, bc_audits, training_audits = {}, [], [], [], []
        budget = max(plan.training_budgets)
        for seed in plan.model_seeds:
            for method in ("rl_only", "teacher_rl"):
                agent = PredictiveDQN.from_configs(run, model, replace(settings, seed=seed))
                root = output/f"models/{method}/seed-{seed}"
                root.mkdir(parents=True, exist_ok=True)
                initial = network_digest(agent.online)
                if method == "teacher_rl":
                    agent.save(root/"initialization.json")
                    bc = behavior_clone(agent, examples, epochs=plan.cloning_epochs,
                        batch_size=plan.cloning_batch_size, learning_rate=plan.cloning_learning_rate)
                    bc.update(model_seed=seed, dataset_sha256=dataset["sha256"])
                    bc_audits.append(bc)
                    _write_json(root/"cloning.json", bc)
                    agent.save(root/"cloned.json")
                    print(f"BC seed={seed}: {len(examples)} examples; training agreement {bc['training_agreement_before']['all']['accuracy']:.3f} -> {bc['training_agreement_after']['all']['accuracy']:.3f}", flush=True)
                rows, frozen = evaluate(agent, run, model, validations, output, label="validation_"+method, model_seed=seed, stage=0)
                stages.append(dict(method=method, model_seed=seed, episode=0, updates=0,
                    rows=rows, selection_rank=validation_rank(rows, 0, model), frozen_fingerprint=frozen))
                selected = None
                with gzip.open(root/"training-traces.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as traces, (root/"losses.csv").open("w", encoding="utf-8") as losses:
                    losses.write("update,loss\n")
                    for ep in range(1, budget+1):
                        previous_updates = agent.updates
                        w, f = plan.training_workload_seed_start+ep-1, plan.training_failure_seed_start+ep-1
                        report = episode(agent, run, model, learn=True, workload_seed=w, failure_seed=f)
                        report.update(method=method, model_seed=seed, training_episode=ep)
                        traces.write(json.dumps(report, allow_nan=False)+"\n")
                        training_rows.append(record(report, contract, method=method, model_seed=seed, episode=ep,
                            workload_seed=w, failure_seed=f, optimizer_updates=agent.updates, epsilon=agent.epsilon))
                        for i in range(previous_updates, agent.updates):
                            losses.write(f"{i+1},{agent.losses[i]}\n")
                        if ep % plan.validation_every_episodes == 0:
                            rows, frozen = evaluate(agent, run, model, validations, output, label="validation_"+method, model_seed=seed, stage=ep)
                            rank = validation_rank(rows, ep, model)
                            if selected is None or rank < tuple(selected["rank"]):
                                filename = f"models/{method}/seed-{seed}/selected.json"
                                agent.save(output/filename)
                                selected = dict(episode=ep, updates=agent.updates, rank=rank,
                                    eligible_for_promotion=rank[0] == 0, fingerprint=frozen, file=filename)
                            stages.append(dict(method=method, model_seed=seed, episode=ep, updates=agent.updates,
                                rows=rows, selection_rank=rank, frozen_fingerprint=frozen))
                            _write_json(root/"selection.json", selected)
                            _write_json(root/f"validation-{ep:03d}.json", stages[-1])
                            _csv(output/"training.csv", training_rows)
                            print(f"{method} seed={seed} episode={ep}/{budget} updates={agent.updates} validation_violations={rank[0]} selected={selected['episode']}", flush=True)
                agent.save(root/"final.json")
                if not agent.updates or network_digest(agent.online) == initial:
                    raise RuntimeError("RL did not update weights")
                selections[f"{method}/{seed}"] = selected
                training_audits.append(dict(method=method, model_seed=seed, episodes=agent.episodes,
                    updates=agent.updates, decisions=agent.decisions, initial_online_sha256=initial,
                    final_online_sha256=network_digest(agent.online), final_epsilon=agent.epsilon))
                del agent
        _write_json(output/"selection-locked.json", dict(selections=selections, all_training_complete=True,
            test_inputs_generated=False, selection_rule=SELECTION))
        print("All BC/RL/control selections locked; now generating fresh test scenarios.", flush=True)
        tests = panel(model, plan.test_workload_seeds, plan.test_failure_seeds, crossed=True)
        test_rows, frozen_audits = [], []
        def evaluate_one(label, agent):
            rows, fingerprint = evaluate(agent, run, model, tests, output, label=label, model_seed=agent.settings.seed)
            test_rows.extend(rows)
            frozen_audits.append(dict(method=label, model_seed=agent.settings.seed, fingerprint=fingerprint,
                unchanged=True, optimizer_updates=agent.updates, episodes=agent.episodes))
            print(f"Frozen {label} seed={agent.settings.seed}: {len(rows)} cases", flush=True)
        for seed in plan.model_seeds:
            root = output/f"models/teacher_rl/seed-{seed}"
            for label, filename in (("predictive_untrained", root/"initialization.json"), ("cloned_only", root/"cloned.json"),
                ("teacher_rl", output/selections[f"teacher_rl/{seed}"]["file"]),
                ("rl_only", output/selections[f"rl_only/{seed}"]["file"])):
                evaluate_one(label, PredictiveDQN.load(filename, contract, host_nodes=hosts))
        for label, agent in originals.items():
            evaluate_one("original_"+label, agent)
        for label, agent in previous:
            evaluate_one(label, agent)
        evaluate_one("observed_teacher", teacher)
        for s in tests:
            report = simulate_objectives(run, model, policy="multi_objective", tasks=s["tasks"], outages=s["outages"],
                failure_seed=s["failure_seed"] or 0, recovery_enabled=True)
            filename = f"without_dqn/work-{s['workload_seed']}-fault-{s['failure_seed']}.json"
            _write_json(output/filename, report, compact=True)
            test_rows.append(record(report, contract, method="without_dqn", model_seed=None, episode=None,
                mode=s["mode"], workload_seed=s["workload_seed"], failure_seed=s["failure_seed"], trace_file=filename))
        for s in tests:
            group = [r for r in test_rows if (r["mode"],r["workload_seed"],r["failure_seed"]) == (s["mode"],s["workload_seed"],s["failure_seed"])]
            if len({r["workload_sha256"] for r in group}) != 1 or len({r["outage_sha256"] for r in group}) != 1:
                raise RuntimeError("test inputs differ between methods")
        if hashes != {n: sha(Path(__file__).with_name(n)) for n in sources} or any(sha(p["path"]) != p["sha256"] for p in prior_meta):
            raise RuntimeError("implementation or preserved baseline changed during experiment")
        result = dict(kind="teacher_dqn_results", selections=selections, training_audits=training_audits,
            cloning_audits=bc_audits, dataset=dataset, validation_stages=stages, test_rows=test_rows,
            frozen_audits=frozen_audits, training_episode_count=len(training_rows),
            baseline_preserved=True, test_generated_after_selection=True, elapsed_seconds=time.monotonic()-started)
        _write_json(output/"result.json", result)
        _csv(output/"training.csv", training_rows)
        _csv(output/"test-metrics.csv", test_rows)
        # Rows already carry their validation method labels and stage/seed metadata.
        _csv(output/"validation.csv", [r for s in stages for r in s["rows"]])
        _write_json(output/"experiment.json", {**manifest, "status":"complete"})
        return result
    except BaseException as error:
        _write_json(output/"experiment.json", {**manifest, "status":"interrupted" if isinstance(error,KeyboardInterrupt) else "failed", "error":str(error)})
        raise
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl_network.yaml")
    parser.add_argument("--model-config", default="configs/objective_model.yaml")
    parser.add_argument("--dqn-config", default="configs/enhanced_dqn.yaml")
    parser.add_argument("--plan-config", default="configs/teacher_training.yaml")
    parser.add_argument("--baseline-dir", default="outputs/objective-model/dqn-sanity/milestone4-20261027")
    parser.add_argument("--predictive-dir", default="outputs/objective-model/predictive-training/study-20261027")
    parser.add_argument("--output-dir", default="outputs/objective-model/teacher-training/study-20261027")
    args = parser.parse_args(argv)
    read = lambda p: yaml.load(Path(p).read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    run_study(load_run_config(args.config), load_objective_model(args.model_config),
        EnhancedDQNConfig.parse(read(args.dqn_config)), TeacherPlan.parse(read(args.plan_config)), args.output_dir,
        baseline_dir=args.baseline_dir, predictive_dir=args.predictive_dir)
    print(f"Saved {args.output_dir}/result.json")


if __name__ == "__main__":
    main()
