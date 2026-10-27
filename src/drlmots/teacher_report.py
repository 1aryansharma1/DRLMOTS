"""Simple teacher-assisted DQN report; matched controls and uncertainty in details."""
import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import zipfile

from drlmots.objective_report import interval
from drlmots.predictive_report import mean, summarize


LABELS = dict(teacher_rl="Teacher + RL DQN", predictive_untrained="Untrained DQN",
    without_dqn="Without DQN (native rule)", rl_only="Matched RL-only DQN",
    cloned_only="Teacher imitation only", observed_teacher="Observed multi-objective teacher",
    previous_predictive_dqn="Preserved predictive DQN", original_untrained_dqn="Original untrained DQN",
    original_trained_dqn="Original 16-episode DQN")


def paired_effects(rows, comparison):
    output = []
    for mode in ("failures_with_recovery", "no_failures"):
        samples = []
        for seed in sorted({r["model_seed"] for r in rows if r["method"] == "teacher_rl"}):
            deltas = []
            for r in rows:
                if (r["method"],r["model_seed"],r["mode"]) != ("teacher_rl",seed,mode):
                    continue
                matches = [o for o in rows if (o["method"],o["model_seed"],o["mode"],o["workload_seed"],o["failure_seed"]) ==
                    (comparison,seed,mode,r["workload_seed"],r["failure_seed"])]
                if len(matches) != 1:
                    raise ValueError("exactly one matched model/scenario control required")
                other = matches[0]
                if (r["workload_sha256"],r["outage_sha256"]) != (other["workload_sha256"],other["outage_sha256"]):
                    raise ValueError("paired test inputs differ")
                deltas.append({k:r["metrics"][k]-other["metrics"][k] if r["metrics"][k] is not None and other["metrics"][k] is not None else None
                    for k in ("successful_makespan_ms","total_energy_j","completion_rate","penalized_return")})
            samples.append(dict(model_seed=seed, means={k:mean(d[k] for d in deltas) for k in deltas[0]}))
        output.append(dict(mode=mode, comparison=comparison, model_seed_means=samples,
            intervals={k:interval([s["means"][k] for s in samples if s["means"][k] is not None]) for k in samples[0]["means"]}))
    return output


def fmt(x, digits=3):
    return "undefined" if x is None else f"{x:.{digits}f}"


def table_rows(group):
    return [[LABELS[r["method"]],fmt(r["metrics"]["successful_makespan_ms"],2),fmt(r["metrics"]["total_energy_j"]),
        fmt(100*r["metrics"]["completion_rate"],2),fmt(r["mean_penalized_cost"],6),f"{r['feasible_runs']}/{r['runs']}"] for r in group]


HEADERS = ("Method","Makespan ms ↓","Energy J ↓","Completion % ↑","Combined cost ↓","Feasible runs")


def markdown_table(headers, rows):
    return ["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),*["| "+" | ".join(row)+" |" for row in rows]]


def html_table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def figures(output, study, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mode in ("no_failures","failures_with_recovery"):
        group = [next(r for r in summaries if (r["mode"],r["method"]) == (mode,m)) for m in ("teacher_rl","predictive_untrained","without_dqn")]
        fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
        for ax,key,title in zip(axes,("successful_makespan_ms","total_energy_j","completion_rate"),("Successful makespan (ms) ↓","System energy (J) ↓","Completion (%) ↑")):
            values=[r["metrics"][key]*(100 if key=="completion_rate" else 1) if r["metrics"][key] is not None else None for r in group]
            bars=ax.bar(["Teacher + RL","Untrained","No DQN"],[v or 0 for v in values],color=["#197354","#9ba7b4","#264766"],width=.6)
            ax.bar_label(bars,labels=[fmt(v,2) for v in values],padding=4)
            ax.set(title=title,ylim=(0,max([1]+[v for v in values if v is not None])*1.2))
            if key=="completion_rate":
                ax.set(ylim=(0,110),yticks=(0,20,40,60,80,100))
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.15)
            ax.set_axisbelow(True)
        fig.suptitle(mode.replace("_"," ")+" · frozen policies, matched fresh scenarios")
        for ext in ("png","svg"):
            fig.savefig(output/f"{mode}.{ext}",dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    for bc in study["cloning_audits"]:
        axes[0].plot([x["epoch"] for x in bc["history"]],[x["cross_entropy"] for x in bc["history"]],label=f"seed {bc['model_seed']}")
    for method in ("teacher_rl","rl_only"):
        for seed in sorted({s["model_seed"] for s in study["validation_stages"]}):
            stages=[s for s in study["validation_stages"] if (s["method"],s["model_seed"])==(method,seed)]
            x=[s["episode"] for s in stages]
            label=f"{'Teacher + RL' if method=='teacher_rl' else 'RL only'} {seed}"
            style="-" if method=="teacher_rl" else "--"
            axes[1].plot(x,[-s["selection_rank"][2] for s in stages],style,label=label)
            axes[2].plot(x,[len(s["rows"])-s["selection_rank"][0] for s in stages],style,label=label)
    for ax,title,xlabel in zip(axes,("Cloning loss (training examples)","Frozen validation return ↑","Service-feasible validation cases"),("BC epoch","RL episode","RL episode")):
        ax.set(title=title,xlabel=xlabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    for ext in ("png","svg"):
        fig.savefig(output/f"learning.{ext}",dpi=160)
    plt.close(fig)


def render(directory):
    output=Path(directory)
    study=json.loads((output/"result.json").read_text(encoding="utf-8"))
    manifest=json.loads((output/"experiment.json").read_text(encoding="utf-8"))
    if study["kind"] != "teacher_dqn_results" or manifest["status"] != "complete":
        raise ValueError("complete teacher study required")
    summaries=summarize(study["test_rows"])
    for row in summaries:
        if row["method"] in ("teacher_rl","rl_only"):
            selected=[s for name,s in study["selections"].items() if name.startswith(row["method"]+"/")]
            row["validation_eligible"]=bool(selected) and all(s["eligible_for_promotion"] for s in selected)
            if not row["validation_eligible"]:
                row["eligible"]=False
                row["mean_objective_cost"]=None
    effects=[e for method in ("rl_only","cloned_only","predictive_untrained") for e in paired_effects(study["test_rows"],method)]
    figures(output,study,summaries)
    def svg(name):
        content=(output/(name+".svg")).read_text(encoding="utf-8")
        return content[content.index("<svg"):]
    plan=manifest["plan"]
    description=f"{plan['demonstration_episodes']} teacher rollout episodes; {plan['cloning_epochs']} fixed imitation epochs per model; {study['training_episode_count']} actual RL episodes across both training branches and {len(plan['model_seeds'])} seeds; {len(study['test_rows'])} frozen test simulations."
    md=["# Teacher-assisted DQN results","",description,"",
        "Teacher imitation → ordinary masked DQN training → validation-selected checkpoint → frozen fresh testing. Baselines are preserved.",""]
    blocks=[]
    for mode in ("failures_with_recovery","no_failures"):
        primary=[next(s for s in summaries if (s["mode"],s["method"])==(mode,m)) for m in ("teacher_rl","predictive_untrained","without_dqn")]
        all_group=[s for s in summaries if s["mode"]==mode]
        eligible=sorted((s for s in primary if s["eligible"]),key=lambda s:s["mean_objective_cost"])
        winner=LABELS[eligible[0]["method"]] if eligible else "None: service floors failed"
        changes=[]
        for key,label in (("successful_makespan_ms","Makespan"),("total_energy_j","Energy"),("completion_rate","Completion")):
            a,b=primary[0]["metrics"][key],primary[1]["metrics"][key]
            if a is None or b is None:
                changes.append((label+": undefined",False))
            elif key=="completion_rate":
                changes.append((f"Completion {100*(a-b):+.2f} percentage points",a>b))
            elif b:
                delta=100*(a/b-1)
                changes.append((f"{label} {abs(delta):.2f}% {'lower' if delta<0 else 'higher'}",delta<0))
        headers=HEADERS
        rows=table_rows(primary)
        title=mode.replace("_"," ")
        control=next(s for s in all_group if s["method"]=="rl_only")
        imitation=next(s for s in all_group if s["method"]=="cloned_only")
        trained_cost=primary[0]["mean_penalized_cost"]
        control_change=100*(trained_cost/control["mean_penalized_cost"]-1)
        rl_change=100*(trained_cost/imitation["mean_penalized_cost"]-1)
        control_text=f"Teacher warm start versus matched RL-only: combined penalized cost {abs(control_change):.2f}% {'lower' if control_change<0 else 'higher'}."
        rl_text=f"Continuing RL versus imitation alone: combined penalized cost {abs(rl_change):.2f}% {'lower' if rl_change<0 else 'higher'}."
        ci=next(e for e in effects if e["comparison"]=="rl_only" and e["mode"]==mode)["intervals"]["penalized_return"]["ci95"]
        uncertainty="The paired model-seed interval includes zero; a repeatable teacher benefit is not established." if ci is not None and ci[0]<=0<=ci[1] else "See paired model-seed intervals below; this fixed panel does not establish general superiority."
        md += ["## "+title,"","Best eligible among the three primary methods: **"+winner+"**.","",*markdown_table(headers,rows),"",
               "Versus untrained: "+"; ".join(t for t,good in changes)+".","",control_text,rl_text,uncertainty,""]
        highlight="".join('<p class="'+('gain' if good else 'tradeoff')+'">'+escape(t)+'</p>' for t,good in changes)
        content='<section><h2>'+escape(title)+'</h2><p><strong>Best eligible among the three primary methods: '+escape(winner)+'</strong></p>'+html_table(headers,rows)+'<p>Teacher + RL versus untrained:</p>'+highlight+'<p>'+escape(control_text)+'</p><p>'+escape(rl_text)+'</p><p>'+escape(uncertainty)+'</p>'+svg(mode)+'</section>'
        blocks.append(content if mode=="failures_with_recovery" else '<details><summary>Without failures</summary>'+content+'</details>')
        md += ["### Matched controls and preserved baselines","",*markdown_table(headers,table_rows(all_group)),""]
        blocks.append('<details><summary>'+escape(title)+': controls and preserved baselines</summary><section>'+html_table(headers,table_rows(all_group))+'</section></details>')
    selections=[[name,str(s["episode"]),str(s["updates"]),str(s["eligible_for_promotion"])] for name,s in study["selections"].items()]
    bc_rows=[[str(b["model_seed"]),str(b["examples"]),fmt(100*b["training_agreement_before"]["all"]["accuracy"],2),fmt(100*b["training_agreement_after"]["all"]["accuracy"],2),str(b["optimizer_updates"])] for b in study["cloning_audits"]]
    md += ["## Learning and validation","",*markdown_table(("Seed","Teacher examples","Agreement before %","Agreement after %","BC updates"),bc_rows),"",
        *markdown_table(("Branch/seed","Selected RL episode","RL updates","Validation feasible"),selections),""]
    blocks.append('<details><summary>Cloning and validation learning curves</summary><section>'+html_table(("Seed","Examples","Before %","After %","BC updates"),bc_rows)+html_table(("Branch/seed","Selected episode","RL updates","Validation feasible"),selections)+svg("learning")+'</section></details>')
    effect_rows=[]
    for effect in effects:
        for key,stats in effect["intervals"].items():
            effect_rows.append([effect["mode"],LABELS[effect["comparison"]],key,fmt(stats["mean"],6),str(stats["ci95"]),str(stats["n"])])
    md += ["## Paired teacher + RL minus control","",*markdown_table(("Mode","Control","Metric delta","Mean","Approx. 95% interval","Model seeds"),effect_rows),""]
    blocks.append('<details><summary>Paired effects and uncertainty</summary><section>'+html_table(("Mode","Control","Metric delta","Mean","Approx. 95% interval","Seeds"),effect_rows)+'</section></details>')
    objective=manifest["model_configuration"]["objective"]
    w,s=objective["weights"],objective["scales"]
    formula=f"Combined cost = {w['time']:g} × makespan/{s['time_ms']:g} + {w['energy']:g} × energy/{s['energy_j']:g} + {w['failure']:g} × unsuccessful fraction. Lower is better."
    notes=[formula,
        "Service-floor penalties are included in the displayed penalized cost. New RL candidates must have service-feasible validation selections for every model seed and be feasible in every test run to enter the overall recommendation. Read completion alongside makespan, which counts successful tasks only.",
        "The observation-only teacher uses the multi-objective rule's deadline filter and time/energy/risk scoring on the existing delayed predictions. It cannot see physical hidden queues, future failures, future tasks or actual outcomes. The native rule remains unchanged and has different information access.",
        "Energy and makespan describe simulated task execution. Workstation training energy/time and neural decision-computation overhead are outside these simulated objectives; extra teacher/BC compute is disclosed separately.",
        "BC and RL-only models share initialization seeds, network, reward, legal masks, RL workload/failure panel, 300-episode budget (or configured test budget), exploration and validation selection rule. BC has extra demonstration and supervised-compute cost, disclosed above. The BC-only ablation checks whether RL preserves imitation benefits.",
        "Teacher labels are used only in fixed offline cloning. RL replay contains only the agent's own transitions. Inference uses the neural network alone, epsilon zero, and has no teacher fallback. This is BC then DQN, not full DQfD.",
        "Demonstration, RL, validation and test seeds are disjoint and exclude all previous registered study panels. Test inputs are generated only after all selections are locked. There is no test-based checkpoint or hyperparameter selection.",
        "The three-seed intervals use paired model-seed mean effects on a fixed scenario panel, not all simulation rows as independent replicates. They do not establish general superiority. Models are averaged for reporting, not deployed as an ensemble.",
        "Baseline metrics here are re-evaluated on this fresh panel; compare within this report, not with previous panels. The original default scheduler is unchanged. Full traces and model checkpoints remain in the experiment directory; the compact ZIP includes only reports and summary evidence."]
    md += ["## How to read this report","",*["- "+n for n in notes],""]
    (output/"REPORT.md").write_text("\n".join(md),encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(dict(summaries=summaries,paired_effects=effects),indent=2)+"\n",encoding="utf-8")
    links=("REPORT.md","test-metrics.csv","training.csv","validation.csv","demonstrations.csv","dataset.json","summary.json","selection-locked.json","experiment.json","evidence.json")
    html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Teacher-assisted DQN results</title><style>body{font:16px/1.5 system-ui;background:#f3f6f8;color:#20364b;margin:0}main{max-width:1200px;margin:auto;padding:24px}section{background:white;border:1px solid #dce5eb;border-radius:12px;padding:20px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;white-space:nowrap;padding:10px;border-bottom:1px solid #dce5eb}.scroll{overflow:auto}svg{max-width:100%;height:auto}.gain{background:#e7f5ec;color:#145f3c;padding:12px}.tradeoff{background:#fff2e3;padding:12px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}footer{font-size:13px}a{color:#195f8b}@media(max-width:650px){main,section{padding:12px}}</style></head><body><main><h1>Teacher-assisted DQN</h1><p>'+escape(description)+'</p><p>Imitate the multi-objective teacher, then learn from actual rewards. All evaluation is frozen.</p>'+''.join(blocks)+'<footer><ul>'+''.join('<li>'+escape(n)+'</li>' for n in notes)+'</ul><p>'+" · ".join('<a href="'+n+'">'+n+'</a>' for n in links)+'</p></footer></main></body></html>'
    (output/"report.html").write_text(html,encoding="utf-8")
    names=["report.html",*links[:-1],*[f"{n}.{ext}" for n in ("failures_with_recovery","no_failures","learning") for ext in ("png","svg")]]
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    evidence=dict(kind="teacher_dqn_report_evidence",renderer_sha256=digest(__file__),result_sha256=digest(output/"result.json"),
        sha256={n:digest(output/n) for n in names})
    (output/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    with zipfile.ZipFile(output/"report.zip","w",zipfile.ZIP_DEFLATED) as z:
        for n in (*names,"evidence.json"):
            z.write(output/n,n)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",default="outputs/objective-model/teacher-training/study-20261027")
    args=parser.parse_args(argv)
    render(args.directory)
    print(f"Saved {args.directory}/report.html and report.zip")


if __name__=="__main__":
    main()
