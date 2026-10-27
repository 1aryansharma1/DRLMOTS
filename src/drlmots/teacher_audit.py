"""Independently audit saved teacher-study labels, accounting, selection and reports."""
import argparse
from collections import Counter, defaultdict, deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from drlmots.normalization import _digest
from drlmots.enhanced_dqn import _observation, EnhancedDQNConfig
from drlmots.teacher_dqn import ObservedTeacher, teacher_choice
from drlmots.run_config import load_run_config
from drlmots.objective_model import ObjectiveModel


def audit_study(directory, run_config='configs/rl_network.yaml'):
    ROOT=Path(directory)
    REPO=Path(__file__).resolve().parents[2]
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    close=lambda a,b:math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)
    study=load(ROOT/'result.json')
    experiment=load(ROOT/'experiment.json')
    assert experiment['status']=='complete' and study['kind']=='teacher_dqn_results'
    plan=experiment['plan']
    for name,digest in experiment['sources_sha256'].items():
        assert sha(REPO/'src/drlmots'/name)==digest
    totals=Counter()
    def audit(report):
        m, config = report["metrics"], report["model_configuration"]
        o, n = config["objective"], m["submitted_count"]
        assert n > 0 and len(report["outcomes"]) == n
        counts = Counter(t["status"] for t in report["outcomes"])
        for status, key in (("completed", "completed_count"), ("rejected", "rejected_count"),
            ("execution_failure", "execution_failure_count"), ("deadline_missed", "deadline_missed_count"), ("unfinished", "unfinished_count")):
            assert counts[status] == m[key]
        assert sum(counts.values()) == n
        assert report["available_resources"] == {v["vm_id"]: v["resource_capacities"] for v in report["run_configuration"]["topology"]["vms"]}
        completed = [t for t in report["outcomes"] if t["status"] == "completed"]
        if completed:
            assert close(max(t["finish_ms"] for t in completed), m["successful_makespan_ms"])
            assert close(math.fsum(t["finish_ms"]-t["arrival_ms"] for t in completed)/len(completed), m["successful_mean_latency_ms"])
        else:
            assert m["successful_makespan_ms"] is None
        assert (m["batch_makespan_ms"] is not None) == (len(completed) == n)
        host = math.fsum(h[k] for h in report["host_energy"].values() for k in ("idle_j", "dynamic_j", "down_j", "reboot_j"))
        assert close(host, math.fsum(s["energy_j"] for s in report["power_segments"]))
        assert close(host+m["network_traffic_energy_j"]+m["network_idle_energy_j"]+m["recovery_energy_j"], m["total_energy_j"])
        for segment in report["link_segments"]:
            assert -1e-8 <= segment["transmitted_kb"] <= segment["capacity_kb"]+1e-8
        for a in report["attempts"]:
            assert close(math.fsum(a["phase_ms"].values()), a["end_ms"]-a["dispatch_ms"])
            speed = config["vms"][a["vm_id"]]["mips"]
            assert close(a["executed_mi"], a["phase_ms"]["compute"]*speed/1000)
        for t in report["outcomes"]:
            assert close(t["attempt_elapsed_ms"]+t["recovery_blocked_ms"], t["terminal_ms"]-t["arrival_ms"])
        penalty = 10*max(0, o["minimum_completion_rate"]-m["completed_count"]/n)
        if m["deadline_eligible_count"]:
            penalty += 10*max(0, o["minimum_deadline_success_rate"]-m["deadline_success_count"]/m["deadline_eligible_count"])
        time = m["successful_makespan_ms"] if completed else config["horizon_ms"]
        cost = o["weights"]["time"]*time/o["scales"]["time_ms"]+o["weights"]["energy"]*m["total_energy_j"]/o["scales"]["energy_j"]+o["weights"]["failure"]*(n-len(completed))/n+penalty
        if "rl" in report:
            rl = report["rl"]
            assert close(rl["total_reward"], -cost)
            for d in rl["decisions"]:
                if d["action"] is not None:
                    assert d["observation"]["legal_mask"][d["action"]]
                else:
                    assert not any(d["observation"]["legal_mask"])
            totals["initial_transitions"] += sum(t["state"]["placement_kind"] == "initial" for t in rl["transitions"])
            totals["recovery_transitions"] += sum(t["state"]["placement_kind"] == "recovery" for t in rl["transitions"])
        totals["reports"] += 1
        totals["outcomes"] += n
        totals["attempts"] += len(report["attempts"])
        return cost
    for p in experiment['prior_baselines']:
        assert sha(Path(p['path'])) == p['sha256']
    run=load_run_config(run_config)
    assert run.manifest()==experiment['run_configuration']
    model=ObjectiveModel(experiment['model_configuration'])
    teacher=ObservedTeacher.from_configs(run,model,EnhancedDQNConfig.parse(experiment['learner_configuration']))
    demo=[]
    with gzip.open(ROOT/'demonstrations.jsonl.gz','rt',encoding='utf-8') as f:
        reports=0
        for line in f:
            r=json.loads(line)
            audit(r)
            reports+=1
            assert not r['rl']['learn'] and r['rl']['learner_updates']==0
            for t in r['rl']['transitions']:
                state=_observation(t['state'])
                assert teacher_choice(teacher,state)[0] == t['action']
                demo.append(dict(state=t['state'],action=t['action']))
    assert reports==plan['demonstration_episodes']
    assert study['dataset']['sha256']==_digest(demo)
    assert len(demo)==study['dataset']['examples']
    assert sum(d['state']['placement_kind']=='recovery' for d in demo)==study['dataset']['recovery_count']
    train_inputs=defaultdict(set)
    for method in ('teacher_rl','rl_only'):
        for seed in plan['model_seeds']:
            root=ROOT/f'models/{method}/seed-{seed}'
            final=load(root/'final.json')
            transitions=0
            allowed=deque(maxlen=final['settings']['replay_capacity'])
            with gzip.open(root/'training-traces.jsonl.gz','rt',encoding='utf-8') as f:
                count=0
                for line in f:
                    r=json.loads(line)
                    count+=1
                    audit(r)
                    assert r['training_episode']==count and r['model_seed']==seed and r['method']==method
                    assert r['rl']['learn'] and r['rl']['network_feature_count']==325
                    transitions+=len(r['rl']['transitions'])
                    allowed.extend(r['rl']['transitions'])
                    train_inputs[count].add((r['workload']['sha256'],r['outage_trace']['sha256']))
            assert count==final['episodes']==max(plan['training_budgets'])
            assert final['decisions']==transitions
            assert final['updates']==transitions-final['settings']['warmup_transitions']+1
            assert final['replay']==list(allowed)
            stages=[s for s in study['validation_stages'] if s['method']==method and s['model_seed']==seed]
            for stage in stages:
                costs,shortfalls,violations=[],[],0
                for row in stage['rows']:
                    r=load(ROOT/row['trace_file'])
                    costs.append(audit(r))
                    assert not r['rl']['learn'] and all(d['epsilon']==0 for d in r['rl']['decisions'])
                    m,o=r['metrics'],r['model_configuration']['objective']
                    shortfalls.append(max(0,o['minimum_completion_rate']-m['completion_rate'])+max(0,o['minimum_deadline_success_rate']-(m['deadline_success_rate'] if m['deadline_success_rate'] is not None else 1)))
                    violations+=not r['objective']['feasible']
                rank=stage['selection_rank']
                assert rank[0]==violations and close(rank[1],math.fsum(shortfalls)/len(shortfalls)) and close(rank[2],math.fsum(costs)/len(costs)) and rank[3]==stage['episode']
            best=min((s for s in stages if s['episode']>0),key=lambda s:s['selection_rank'])
            chosen=study['selections'][f'{method}/{seed}']
            checkpoint=load(ROOT/chosen['file'])
            assert chosen['fingerprint']==checkpoint['sha256']==best['frozen_fingerprint']
            assert checkpoint['episodes']==chosen['episode']==best['episode']
            if method=='teacher_rl':
                cloned=load(root/'cloned.json')
                assert cloned['episodes']==cloned['updates']==cloned['decisions']==0 and cloned['replay']==[]
                assert cloned['optimizer']['state']=={}
                bc=load(root/'cloning.json')
                assert bc['dataset_sha256']==study['dataset']['sha256']
                assert bc['initial_online_sha256']!=bc['final_online_sha256']
    assert len(train_inputs)==max(plan['training_budgets']) and all(len(v)==1 for v in train_inputs.values())
    locked=ROOT/'selection-locked.json'
    assert load(locked)['all_training_complete'] and not load(locked)['test_inputs_generated']
    assert all((ROOT/f'models/{m}/seed-{s}/final.json').stat().st_mtime_ns<=locked.stat().st_mtime_ns for m in ('teacher_rl','rl_only') for s in plan['model_seeds'])
    cases=defaultdict(list)
    for row in study['test_rows']:
        p=ROOT/row['trace_file']
        assert p.stat().st_mtime_ns>=locked.stat().st_mtime_ns
        r=load(p)
        assert close(audit(r),-row['metrics']['penalized_return'])
        cases[(row['mode'],row['workload_seed'],row['failure_seed'])].append(r)
        if 'rl' in r:
            assert not r['rl']['learn'] and all(d['epsilon']==0 for d in r['rl']['decisions'])
    expected_cases=len(plan['test_workload_seeds'])*(1+len(plan['test_failure_seeds']))
    expected_methods=len(study['test_rows'])//expected_cases
    assert len(cases)==expected_cases and all(len(v)==expected_methods for v in cases.values())
    assert all(all(r['workload']==group[0]['workload'] and r['outage_trace']==group[0]['outage_trace'] for r in group) for group in cases.values())
    assert all(a['unchanged'] for a in study['frozen_audits'])
    train_count=2*len(plan['model_seeds'])*max(plan['training_budgets'])
    validation_count=sum(len(s['rows']) for s in study['validation_stages'])
    for name,count in (('training.csv',train_count),('test-metrics.csv',len(study['test_rows'])),('validation.csv',validation_count),('demonstrations.csv',plan['demonstration_episodes'])):
        with (ROOT/name).open() as f:
            assert len(list(csv.DictReader(f)))==count

    evidence=load(ROOT/'evidence.json')
    assert all(sha(ROOT/n)==h for n,h in evidence['sha256'].items())
    assert sha(ROOT/'result.json')==evidence['result_sha256']
    html=(ROOT/'report.html').read_text(encoding='utf-8')
    links=[p for p in re.findall(r'(?:href|src)="([^"]+)"',html) if not p.startswith('#')]
    with zipfile.ZipFile(ROOT/'report.zip') as z:
        assert z.testzip() is None
        assert all(n in z.namelist() for n in links)
        assert all(z.read(n)==(ROOT/n).read_bytes() for n in z.namelist())
        entries=len(z.namelist())
    result=dict(status='passed',totals=dict(totals),teacher_episodes=plan['demonstration_episodes'],training_episodes=train_count,validation_cases=validation_count,frozen_test_cases=len(study['test_rows']),shared_test_scenarios=len(cases),zip_entries=entries,local_links=len(links),teacher_labels_observation_only_verified=True,training_only_replay_verified=True,baseline_sources_and_checkpoints_unchanged=True,selection_and_lock_verified=True,zip_sha256=sha(ROOT/'report.zip'))
    (ROOT/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default='outputs/objective-model/teacher-training/study-20261027')
    parser.add_argument('--config',default='configs/rl_network.yaml')
    args=parser.parse_args(argv)
    print(json.dumps(audit_study(args.directory,args.config),indent=2))


if __name__=='__main__':
    main()
