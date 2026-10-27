from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from drlmots.enhanced_dqn import EnhancedDQNConfig
from drlmots.normalization import _digest
from drlmots.objective_model import load_objective_model
from drlmots.predictive_dqn import PredictiveDQN
from drlmots.run_config import load_run_config
from drlmots.teacher_training import TeacherPlan, run_study


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    run = load_run_config("configs/rl_network.yaml")
    model = load_objective_model("configs/objective_model.yaml")
    model.data["horizon_ms"] = 60
    model.data["workload"].update(task_count=4, interarrival_ms=[1,2])
    settings = EnhancedDQNConfig(batch_size=2,warmup_transitions=2,replay_capacity=32,
        epsilon_decay_decisions=8,target_sync_updates=2)
    plan = TeacherPlan(model_seeds=(42,43),training_budgets=(4,),validation_every_episodes=2,
        demonstration_episodes=4,cloning_epochs=2,cloning_batch_size=4,
        validation_workload_seeds=(86000,),validation_failure_seeds=(96000,),
        test_workload_seeds=(87000,),test_failure_seeds=(97000,97001))
    output = tmp_path_factory.mktemp("teacher-study")/"run"
    import drlmots.teacher_training as module
    original_panel = module.panel
    def checked_panel(*args,**kwargs):
        if kwargs["crossed"]:
            lock = json.loads((output/"selection-locked.json").read_text())
            assert lock["all_training_complete"] and not lock["test_inputs_generated"]
            for method in ("rl_only","teacher_rl"):
                for seed in plan.model_seeds:
                    root=output/f"models/{method}/seed-{seed}"
                    assert json.loads((root/"final.json").read_text())["episodes"] == 4
                    with gzip.open(root/"training-traces.jsonl.gz","rt",encoding="utf-8") as f:
                        assert sum(1 for line in f)==4
        return original_panel(*args,**kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module,"panel",checked_panel)
        result=run_study(run,model,settings,plan,output)
    return run,model,settings,plan,output,result


def test_teacher_pipeline_runs_actual_bc_then_rl_with_matched_control(study):
    run,model,settings,plan,output,result=study
    assert result["training_episode_count"]==16
    assert len(result["cloning_audits"])==2
    assert len(result["test_rows"])==30
    assert all(a["episodes"]==4 and a["updates"]>0 for a in result["training_audits"])
    for seed in plan.model_seeds:
        teacher = next(a for a in result["training_audits"] if a["method"]=="teacher_rl" and a["model_seed"]==seed)
        control = next(a for a in result["training_audits"] if a["method"]=="rl_only" and a["model_seed"]==seed)
        assert teacher["initial_online_sha256"]==control["initial_online_sha256"]
        root=output/f"models/teacher_rl/seed-{seed}"
        cloned=PredictiveDQN.load(root/"cloned.json", PredictiveDQN.from_configs(run,model,settings).contract)
        assert cloned.updates==cloned.decisions==cloned.episodes==0 and not cloned.replay
        bc=next(a for a in result["cloning_audits"] if a["model_seed"]==seed)
        assert bc["initial_online_sha256"]!=bc["final_online_sha256"]


def test_training_only_demonstrations_and_reward_reconciliation(study):
    _,_,_,plan,output,result=study
    with gzip.open(output/"demonstrations.jsonl.gz","rt",encoding="utf-8") as f:
        demonstrations=[json.loads(line) for line in f]
    assert len(demonstrations)==plan.demonstration_episodes
    assert result["dataset"]["examples"]==sum(len(r["rl"]["transitions"]) for r in demonstrations)
    assert all(not r["rl"]["learn"] and r["rl"]["learner_updates"]==0 for r in demonstrations)
    for r in demonstrations:
        assert r["rl"]["total_reward"]==pytest.approx(r["rl"]["expected_reward"])
    for method in ("teacher_rl","rl_only"):
        for seed in plan.model_seeds:
            root=output/f"models/{method}/seed-{seed}"
            with gzip.open(root/"training-traces.jsonl.gz","rt",encoding="utf-8") as f:
                traces=[json.loads(line) for line in f]
            allowed={_digest(t) for r in traces for t in r["rl"]["transitions"]}
            final=json.loads((root/"final.json").read_text())
            assert {_digest(t) for t in final["replay"]} <= allowed
            assert all(r["rl"]["learn"] and r["rl"]["total_reward"]==pytest.approx(r["rl"]["expected_reward"]) for r in traces)
            assert [r["workload"]["sha256"] for r in traces] == [json.loads(line)["workload"]["sha256"] for line in gzip.open(output/f"models/rl_only/seed-{seed}/training-traces.jsonl.gz","rt",encoding="utf-8")]


def test_validation_selection_and_fresh_frozen_pairing(study):
    result=study[-1]
    for name,selection in result["selections"].items():
        method,seed=name.split("/")
        candidates=[s for s in result["validation_stages"] if s["method"]==method and s["model_seed"]==int(seed) and s["episode"]>0]
        best=min(candidates,key=lambda s:s["selection_rank"])
        assert selection["rank"]==best["selection_rank"] and selection["fingerprint"]==best["frozen_fingerprint"]
    assert all(r["unchanged"] for r in result["frozen_audits"])
    for mode in ("no_failures","failures_with_recovery"):
        group=[r for r in result["test_rows"] if r["mode"]==mode]
        for failure in {r["failure_seed"] for r in group}:
            cases=[r for r in group if r["failure_seed"]==failure]
            assert len(cases)==10
            assert len({r["workload_sha256"] for r in cases})==len({r["outage_sha256"] for r in cases})==1
    assert result["test_generated_after_selection"] and result["baseline_preserved"]


@pytest.mark.parametrize("kwargs",[
    dict(test_workload_seeds=(85000,)),dict(training_workload_seed_start=30000),
    dict(demonstration_workload_seed_start=84000),dict(demonstration_failure_seed_start=95000),
    dict(demonstration_workload_seed_start=51000),dict(cloning_epochs=0),
    dict(cloning_learning_rate=-1),dict(training_budgets=(100,300)),
    dict(demonstration_episodes=0),dict(demonstration_failure_seed_start=96000),
])
def test_teacher_plan_rejects_seen_overlapping_or_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        TeacherPlan(**kwargs)


def test_plan_roundtrip_and_no_overwrite(study):
    run,model,settings,plan,output,_=study
    assert TeacherPlan.parse(json.loads(json.dumps(plan.manifest())))==plan
    with pytest.raises(ValueError):
        TeacherPlan.parse({**plan.manifest(),"surprise":1})
    with pytest.raises(ValueError,match="empty"):
        run_study(run,model,settings,plan,output)


def test_report_means_local_assets_zip_hashes_and_paired_effects(study):
    import re
    import zipfile
    from drlmots.teacher_report import render
    output,result=study[-2:]
    before=hashlib.sha256((output/"result.json").read_bytes()).hexdigest()
    summaries=render(output)
    assert hashlib.sha256((output/"result.json").read_bytes()).hexdigest()==before
    assert len(summaries)==12
    html=(output/"report.html").read_text(encoding="utf-8")
    for href in re.findall(r'href="([^"]+)"',html):
        assert href.startswith("#") or (output/href).is_file()
    summary=json.loads((output/"summary.json").read_text())
    assert len(summary["paired_effects"])==6
    assert all(e["intervals"]["total_energy_j"]["n"]==2 for e in summary["paired_effects"])
    evidence=json.loads((output/"evidence.json").read_text())
    for name,digest in evidence["sha256"].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest
    with zipfile.ZipFile(output/"report.zip") as z:
        assert z.testzip() is None
        assert all(z.read(n)==(output/n).read_bytes() for n in z.namelist())
    for s in summaries:
        rows=[r for r in result["test_rows"] if (r["mode"],r["method"])==(s["mode"],s["method"])]
        assert s["mean_penalized_cost"]==pytest.approx(-sum(r["metrics"]["penalized_return"] for r in rows)/len(rows))


def test_paired_report_rejects_missing_or_mismatched_control(study):
    from copy import deepcopy
    from drlmots.teacher_report import paired_effects
    rows=deepcopy(study[-1]["test_rows"])
    with pytest.raises(ValueError,match="exactly one"):
        paired_effects([r for r in rows if r["method"]!="rl_only"],"rl_only")
    for r in rows:
        if r["method"]=="rl_only":
            r["workload_sha256"]="tampered"
    with pytest.raises(ValueError,match="inputs differ"):
        paired_effects(rows,"rl_only")


def test_cloned_and_selected_checkpoint_run_through_existing_inference_cli(study):
    from drlmots.predictive_inference import main
    run,model,_,_,output,result=study
    # The run configuration is unchanged; save the small-study physical model.
    import yaml
    (output/"model.yaml").write_text(yaml.safe_dump(model.manifest()),encoding="utf-8")
    for label,checkpoint in (("cloned",output/"models/teacher_rl/seed-42/cloned.json"),
        ("selected",output/result["selections"]["teacher_rl/42"]["file"])):
        target=output/f"inference-{label}.json"
        main(["--checkpoint",str(checkpoint),"--model-config",str(output/"model.yaml"),
              "--workload-seed","86000","--failure-seed","96000","--output",str(target)])
        report=json.loads(target.read_text())
        assert report["rl"]["network_feature_count"]==325 and not report["rl"]["learn"]
        assert all(d["epsilon"]==0 for d in report["rl"]["decisions"])
        with pytest.raises(SystemExit):
            main(["--checkpoint",str(checkpoint),"--output",str(target)])


def test_independent_audit_recomputes_complete_physics_and_detects_bad_energy(study):
    from drlmots.teacher_report import render
    from drlmots.teacher_audit import audit_study
    output,result=study[-2:]
    render(output)
    audit=audit_study(output)
    assert audit["status"]=="passed"
    assert audit["training_episodes"]==16 and audit["frozen_test_cases"]==30
    assert audit["totals"]["reports"]==74  # 4 demo +16 RL +24 validation +30 test.
    target=output/result["test_rows"][0]["trace_file"]
    original=target.read_bytes()
    bad=json.loads(original)
    bad["metrics"]["total_energy_j"]+=1
    try:
        target.write_text(json.dumps(bad),encoding="utf-8")
        with pytest.raises(AssertionError):
            audit_study(output)
    finally:
        target.write_bytes(original)


def test_recovery_preserves_complete_models_and_exactly_replays_partial_checkpoint(study,tmp_path):
    import shutil
    import torch
    from drlmots.teacher_dqn import behavior_clone,dataset_from_reports
    from drlmots.teacher_resume import recover_reports,resume_study
    from drlmots.predictive_training import episode
    run,model,settings,plan,output,expected=study
    source=tmp_path/'interrupted'
    source.mkdir()
    for name in ('experiment.json','dataset.json','demonstrations.jsonl.gz','demonstrations.csv'):
        shutil.copyfile(output/name,source/name)
    shutil.copytree(output/'models',source/'models')
    for p in output.glob('validation_*'):
        if p.is_dir():
            shutil.copytree(p,source/p.name)
    manifest=json.loads((source/'experiment.json').read_text())
    manifest['status']='running'
    (source/'experiment.json').write_text(json.dumps(manifest))
    partial=source/'models/teacher_rl/seed-43'
    (partial/'final.json').unlink()
    before_source=hashlib.sha256((source/'experiment.json').read_bytes()).hexdigest()
    examples=dataset_from_reports(recover_reports(source/'demonstrations.jsonl.gz'))
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent=PredictiveDQN.from_configs(run,model,replace(settings,seed=43))
        behavior_clone(agent,examples,epochs=plan.cloning_epochs,batch_size=plan.cloning_batch_size,learning_rate=plan.cloning_learning_rate)
        for ep in (1,2):
            episode(agent,run,model,learn=True,workload_seed=plan.training_workload_seed_start+ep-1,failure_seed=plan.training_failure_seed_start+ep-1)
        stage=next(s for s in expected['validation_stages'] if (s['method'],s['model_seed'],s['episode'])==('teacher_rl',43,2))
        assert agent.fingerprint==stage['frozen_fingerprint']
        agent.save(partial/'latest.json')
        agent.save(partial/'selected.json')
        selected=dict(episode=2,updates=agent.updates,rank=stage['selection_rank'],eligible_for_promotion=stage['selection_rank'][0]==0,fingerprint=agent.fingerprint,file='models/teacher_rl/seed-43/selected.json')
        (partial/'selection.json').write_text(json.dumps(selected))
        recovered=tmp_path/'recovered'
        actual=resume_study(source,recovered)
    finally:
        torch.set_num_threads(old_threads)
    assert actual['recovery']['exact_replayed_episodes']==2
    assert actual['training_episode_count']==16
    assert actual['test_rows']==expected['test_rows']
    for audit in actual['training_audits']:
        original=next(a for a in expected['training_audits'] if (a['method'],a['model_seed'])==(audit['method'],audit['model_seed']))
        assert audit==original
    assert hashlib.sha256((source/'experiment.json').read_bytes()).hexdigest()==before_source
    assert not (source/'selection-locked.json').exists()
    with pytest.raises(ValueError,match='empty'):
        resume_study(source,recovered)
    with pytest.raises(ValueError,match='pre-test'):
        resume_study(output,tmp_path/'invalid')


def test_recover_complete_lines_from_gzip_missing_footer(tmp_path):
    from drlmots.teacher_resume import recover_reports
    path=tmp_path/'partial.gz'
    path.write_bytes(gzip.compress(b'{"episode":1}\n{"episode":2}\n')[:-8])
    assert recover_reports(path)==[dict(episode=1),dict(episode=2)]