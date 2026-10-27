"""Recover an interrupted teacher study at complete checkpoint boundaries.

Copies evidence to a fresh folder; never changes the source experiment. Settings,
training seeds and validation selection stay fixed. Complete replayed episodes
must exactly match the interrupted trace before any new training proceeds.
"""
import argparse
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path
import shutil
import time
import zlib

import torch

from drlmots.enhanced_comparison import ComparisonPlan, load_frozen_pair
from drlmots.enhanced_dqn import EnhancedDQNConfig
from drlmots.enhanced_rl_contract import EnhancedRLContract
from drlmots.enhanced_sanity import _write_json, network_digest
from drlmots.normalization import _digest
from drlmots.objective_model import ObjectiveModel
from drlmots.objective_sim import simulate_objectives
from drlmots.predictive_dqn import PredictiveDQN
from drlmots.predictive_training import SELECTION, _csv, episode, evaluate, panel, record, validation_rank
from drlmots.run_config import load_run_config
from drlmots.teacher_dqn import ObservedTeacher, behavior_clone, dataset_from_reports
from drlmots.teacher_training import TeacherPlan, prior_predictive, sha


def recover_reports(path):
    """Read only complete newline-terminated reports, including truncated gzip."""
    data=zlib.decompressobj(31).decompress(Path(path).read_bytes())
    return [json.loads(line) for line in data.split(b'\n')[:-1]]


def resume_study(source, output_dir, *, run_config='configs/rl_network.yaml', baseline_dir=None, predictive_dir=None):
    source,output=Path(source),Path(output_dir)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    original=load(source/'experiment.json')
    if original['kind']!='teacher_dqn_study' or original['status'] not in ('running','interrupted') or (source/'selection-locked.json').exists() or (source/'result.json').exists():
        raise ValueError('only an interrupted pre-test teacher study may be recovered')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('choose an empty recovery output; source evidence is never overwritten')
    run=load_run_config(run_config)
    if run.manifest()!=original['run_configuration']:
        raise ValueError('recovery run configuration differs')
    model=ObjectiveModel(original['model_configuration'])
    settings=EnhancedDQNConfig.parse(original['learner_configuration'])
    plan=TeacherPlan.parse(original['plan'])
    for n,h in original['sources_sha256'].items():
        if sha(Path(__file__).with_name(n))!=h:
            raise ValueError('interrupted implementation source changed')
    for p in original['prior_baselines']:
        if sha(p['path'])!=p['sha256']:
            raise ValueError('preserved baseline changed')
    contract=EnhancedRLContract.from_configs(run,model)
    _,hosts=model.validate_topology(run)
    originals={}
    if baseline_dir:
        originals,_,_=load_frozen_pair(run,model,baseline_dir,ComparisonPlan(plan.test_workload_seeds,plan.test_failure_seeds))
    previous,_=prior_predictive(run,model,predictive_dir) if predictive_dir else ([],[])
    if len(originals)+(len(previous)) != len(original['prior_baselines']):
        raise ValueError('supply the same prior baseline families for recovery')
    output.mkdir(parents=True,exist_ok=True)
    recovery=dict(source_directory=str(source.resolve()),source_manifest_sha256=sha(source/'experiment.json'),
        source_status=original['status'],strategy='complete models reused; partial model resumes complete latest/selected checkpoint; exact prefix replay required',
        checkpoints=[],exact_replayed_episodes=0)
    hashes={**original['sources_sha256'],'teacher_resume.py':sha(__file__)}
    manifest={**original,'status':'running','sources_sha256':hashes,'recovery':recovery}
    for name in ('demonstrations.jsonl.gz','demonstrations.csv','dataset.json','dataset-analysis.json'):
        if (source/name).exists():
            shutil.copyfile(source/name,output/name)
    dataset=load(output/'dataset.json')
    examples=[]
    for report in recover_reports(output/'demonstrations.jsonl.gz'):
        examples.extend(dataset_from_reports([report]))
    if _digest([dict(state=asdict(s),action=a) for s,a in examples])!=dataset['sha256']:
        raise ValueError('recovered teacher dataset differs')
    _write_json(output/'experiment.json',manifest)
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    started=time.monotonic()
    selections,stages,training_rows,bc_audits,training_audits={ },[],[],[],[]
    teacher=ObservedTeacher.from_configs(run,model,settings)
    validations=panel(model,plan.validation_workload_seeds,plan.validation_failure_seeds,crossed=False)
    budget=max(plan.training_budgets)
    try:
        for seed in plan.model_seeds:
            for method in ('rl_only','teacher_rl'):
                relative=Path(f'models/{method}/seed-{seed}')
                oldroot,root=source/relative,output/relative
                root.mkdir(parents=True,exist_ok=True)
                fresh=PredictiveDQN.from_configs(run,model,replace(settings,seed=seed))
                initial=network_digest(fresh.online)
                if method=='teacher_rl':
                    for name in ('initialization.json','cloned.json','cloning.json'):
                        if (oldroot/name).exists():
                            shutil.copyfile(oldroot/name,root/name)
                    if not (root/'cloned.json').exists():
                        fresh.save(root/'initialization.json')
                        bc=behavior_clone(fresh,examples,epochs=plan.cloning_epochs,batch_size=plan.cloning_batch_size,learning_rate=plan.cloning_learning_rate)
                        bc.update(model_seed=seed,dataset_sha256=dataset['sha256'])
                        _write_json(root/'cloning.json',bc)
                        fresh.save(root/'cloned.json')
                    bc_audits.append(load(root/'cloning.json'))
                    initial_agent=PredictiveDQN.load(root/'cloned.json',contract,host_nodes=hosts)
                else:
                    initial_agent=fresh
                rows,frozen=evaluate(initial_agent,run,model,validations,output,label='validation_'+method,model_seed=seed,stage=0)
                stages.append(dict(method=method,model_seed=seed,episode=0,updates=0,rows=rows,selection_rank=validation_rank(rows,0,model),frozen_fingerprint=frozen))
                selected=load(oldroot/'selection.json') if (oldroot/'selection.json').exists() else None
                if selected:
                    shutil.copyfile(source/selected['file'],output/selected['file'])
                    _write_json(root/'selection.json',selected)
                checkpoint=next((oldroot/n for n in ('final.json','latest.json','selected.json') if (oldroot/n).exists()),None)
                agent=PredictiveDQN.load(checkpoint,contract,host_nodes=hosts) if checkpoint else initial_agent
                start=agent.episodes
                if start>budget or (start and start%plan.validation_every_episodes):
                    raise ValueError('recovery checkpoint is not a complete validation boundary')
                recovered=recover_reports(oldroot/'training-traces.jsonl.gz') if (oldroot/'training-traces.jsonl.gz').exists() else []
                if len(recovered)<start:
                    raise ValueError('complete checkpoint lacks matching complete traces')
                old_reports={r['training_episode']:r for r in recovered}
                for n in range(1,start+1):
                    r=old_reports[n]
                    if r['method']!=method or r['model_seed']!=seed:
                        raise ValueError('recovered model trace metadata differs')
                    decisions=r['rl']['learner_decisions']
                    eps=settings.epsilon_start+min(1,decisions/settings.epsilon_decay_decisions)*(settings.epsilon_end-settings.epsilon_start)
                    training_rows.append(record(r,contract,method=method,model_seed=seed,episode=n,
                        workload_seed=plan.training_workload_seed_start+n-1,failure_seed=plan.training_failure_seed_start+n-1,
                        optimizer_updates=r['rl']['learner_updates'],epsilon=eps))
                for p in sorted(oldroot.glob('validation-*.json')):
                    stage=load(p)
                    if 0<stage['episode']<=start:
                        stages.append(stage)
                        shutil.copyfile(p,root/p.name)
                        for row in stage['rows']:
                            target=output/row['trace_file']
                            target.parent.mkdir(parents=True,exist_ok=True)
                            shutil.copyfile(source/row['trace_file'],target)
                if selected and tuple(selected['rank'])!=min(tuple(s['selection_rank']) for s in stages if s['method']==method and s['model_seed']==seed and s['episode']>0):
                    raise ValueError('recovered validation selection differs')
                recovery['checkpoints'].append(dict(method=method,model_seed=seed,resumed_episode=start,
                    checkpoint_sha256=sha(checkpoint) if checkpoint else None))
                print(f'Recovery {method} seed={seed}: {start}/{budget} episodes restored',flush=True)
                with gzip.open(root/'training-traces.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as traces, (root/'losses.csv').open('w',encoding='utf-8') as losses:
                    losses.write('update,loss\n')
                    for i,loss in enumerate(agent.losses):
                        losses.write(f'{i+1},{loss}\n')
                    for n in range(1,start+1):
                        traces.write(json.dumps(old_reports[n],allow_nan=False)+'\n')
                    for ep in range(start+1,budget+1):
                        previous_updates=agent.updates
                        w,f=plan.training_workload_seed_start+ep-1,plan.training_failure_seed_start+ep-1
                        report=episode(agent,run,model,learn=True,workload_seed=w,failure_seed=f)
                        report.update(method=method,model_seed=seed,training_episode=ep)
                        if ep in old_reports:
                            if _digest(report)!=_digest(old_reports[ep]):
                                raise RuntimeError('resumed episode differs from interrupted complete trace')
                            recovery['exact_replayed_episodes']+=1
                        traces.write(json.dumps(report,allow_nan=False)+'\n')
                        training_rows.append(record(report,contract,method=method,model_seed=seed,episode=ep,workload_seed=w,failure_seed=f,optimizer_updates=agent.updates,epsilon=agent.epsilon))
                        for i in range(previous_updates,agent.updates):
                            losses.write(f'{i+1},{agent.losses[i]}\n')
                        if ep%plan.validation_every_episodes==0:
                            rows,frozen=evaluate(agent,run,model,validations,output,label='validation_'+method,model_seed=seed,stage=ep)
                            rank=validation_rank(rows,ep,model)
                            if selected is None or rank<tuple(selected['rank']):
                                filename=(relative/'selected.json').as_posix()
                                agent.save(output/filename)
                                selected=dict(episode=ep,updates=agent.updates,rank=rank,eligible_for_promotion=rank[0]==0,fingerprint=frozen,file=filename)
                            stage=dict(method=method,model_seed=seed,episode=ep,updates=agent.updates,rows=rows,selection_rank=rank,frozen_fingerprint=frozen)
                            stages.append(stage)
                            _write_json(root/f'validation-{ep:03d}.json',stage)
                            _write_json(root/'selection.json',selected)
                            agent.save(root/'latest.json')
                            traces.flush()
                            losses.flush()
                            _csv(output/'training.csv',training_rows)
                            _write_json(output/'experiment.json',manifest)
                            print(f'{method} seed={seed} episode={ep}/{budget} updates={agent.updates} violations={rank[0]} selected={selected["episode"]}',flush=True)
                agent.save(root/'final.json')
                selections[f'{method}/{seed}']=selected
                training_audits.append(dict(method=method,model_seed=seed,episodes=agent.episodes,updates=agent.updates,decisions=agent.decisions,initial_online_sha256=initial,final_online_sha256=network_digest(agent.online),final_epsilon=agent.epsilon))
        _write_json(output/'selection-locked.json',dict(selections=selections,all_training_complete=True,test_inputs_generated=False,selection_rule=SELECTION))
        print('All recovered/new training complete; selections locked before fresh test generation.',flush=True)
        tests=panel(model,plan.test_workload_seeds,plan.test_failure_seeds,crossed=True)
        test_rows,frozen_audits=[],[]
        def evaluate_one(label,agent):
            rows,fingerprint=evaluate(agent,run,model,tests,output,label=label,model_seed=agent.settings.seed)
            test_rows.extend(rows)
            frozen_audits.append(dict(method=label,model_seed=agent.settings.seed,fingerprint=fingerprint,unchanged=True,optimizer_updates=agent.updates,episodes=agent.episodes))
            print(f'Frozen {label} seed={agent.settings.seed}: {len(rows)} cases',flush=True)
        for seed in plan.model_seeds:
            root=output/f'models/teacher_rl/seed-{seed}'
            for label,filename in (('predictive_untrained',root/'initialization.json'),('cloned_only',root/'cloned.json'),('teacher_rl',output/selections[f'teacher_rl/{seed}']['file']),('rl_only',output/selections[f'rl_only/{seed}']['file'])):
                evaluate_one(label,PredictiveDQN.load(filename,contract,host_nodes=hosts))
        for label,agent in originals.items():
            evaluate_one('original_'+label,agent)
        for label,agent in previous:
            evaluate_one(label,agent)
        evaluate_one('observed_teacher',teacher)
        for s in tests:
            report=simulate_objectives(run,model,policy='multi_objective',tasks=s['tasks'],outages=s['outages'],failure_seed=s['failure_seed'] or 0,recovery_enabled=True)
            filename=f"without_dqn/work-{s['workload_seed']}-fault-{s['failure_seed']}.json"
            _write_json(output/filename,report,compact=True)
            test_rows.append(record(report,contract,method='without_dqn',model_seed=None,episode=None,mode=s['mode'],workload_seed=s['workload_seed'],failure_seed=s['failure_seed'],trace_file=filename))
        for s in tests:
            group=[r for r in test_rows if (r['mode'],r['workload_seed'],r['failure_seed'])==(s['mode'],s['workload_seed'],s['failure_seed'])]
            if len({r['workload_sha256'] for r in group})!=1 or len({r['outage_sha256'] for r in group})!=1:
                raise RuntimeError('recovered comparison test inputs differ')
        if any(sha(Path(__file__).with_name(n))!=h for n,h in hashes.items()) or any(sha(p['path'])!=p['sha256'] for p in original['prior_baselines']):
            raise RuntimeError('recovery implementation/baseline provenance changed')
        result=dict(kind='teacher_dqn_results',selections=selections,training_audits=training_audits,cloning_audits=bc_audits,dataset=dataset,validation_stages=stages,test_rows=test_rows,frozen_audits=frozen_audits,training_episode_count=len(training_rows),baseline_preserved=True,test_generated_after_selection=True,elapsed_seconds=time.monotonic()-started,recovery=recovery)
        _write_json(output/'result.json',result)
        _csv(output/'training.csv',training_rows)
        _csv(output/'test-metrics.csv',test_rows)
        _csv(output/'validation.csv',[r for s in stages for r in s['rows']])
        _write_json(output/'experiment.json',{**manifest,'status':'complete'})
        return result
    except BaseException as error:
        _write_json(output/'experiment.json',{**manifest,'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed','error':str(error)})
        raise
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--config',default='configs/rl_network.yaml')
    parser.add_argument('--baseline-dir',default='outputs/objective-model/dqn-sanity/milestone4-20261027')
    parser.add_argument('--predictive-dir',default='outputs/objective-model/predictive-training/study-20261027')
    args=parser.parse_args(argv)
    resume_study(args.source,args.output_dir,run_config=args.config,baseline_dir=args.baseline_dir,predictive_dir=args.predictive_dir)


if __name__=='__main__':
    main()
