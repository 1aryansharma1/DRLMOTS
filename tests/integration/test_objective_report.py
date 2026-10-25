import json
from pathlib import Path
import re
import zipfile

import pytest

from drlmots.objective_model import load_objective_model
from drlmots.objective_report import compare_objective_model, interval, summarize
from drlmots.run_config import load_run_config


def test_paired_report_has_independent_failure_clusters_and_complete_bundle(tmp_path):
    c=load_run_config("configs/rl_network.yaml")
    m=load_objective_model("configs/objective_model.yaml")
    r=compare_objective_model(c,m,[6200,6201],[7100,7101],tmp_path)
    assert len(r["rows"]) == 40 # 2 workloads x 4 rules x (1 no-fault + 2x2 fault modes)
    assert len(r["summaries"]) == 12
    for wseed in (6200,6201):
        group=[row for row in r["rows"] if row["workload_seed"] == wseed]
        assert len({row["workload_sha256"] for row in group}) == 1
        for fseed in (7100,7101):
            paired=[row for row in group if row["failure_seed"] == fseed]
            assert len(paired) == 8 and len({row["outage_sha256"] for row in paired}) == 1
    for s in r["summaries"]:
        assert s["uncertainty"]["completed_count"]["n"] == 2 # not 4 crossed runs
        if not s["eligible_for_recommendation"]:
            assert not s["pareto_efficient"]
    for effect in r["paired_recovery_effects"]:
        assert len(effect["pairs"]) == 4
        assert effect["uncertainty"]["completed_delta"]["n"] == 2
    html=(tmp_path/"objective-report.html").read_text(encoding="utf-8")
    for target in re.findall(r'(?:href|src)="([^"]+)"',html):
        assert (tmp_path/target).is_file(), target
    with zipfile.ZipFile(tmp_path/"objective-report.zip") as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith("runs/")]) == 40
        for n in z.namelist():
            assert not Path(n).is_absolute() and ".." not in Path(n).parts
        assert "weight-sensitivity.json" in z.namelist()
    for row in r["rows"]:
        trace=json.loads((tmp_path/row["trace_file"]).read_text(encoding="utf-8"))
        assert trace["metrics"] == row["metrics"]
        assert len(trace["outcomes"]) == 60
    with pytest.raises(ValueError,match="different plan"):
        compare_objective_model(c,m,[6202],[7100],tmp_path)


def test_uncertainty_is_undefined_with_one_independent_trace():
    assert interval([3])["ci95"] is None
    u=interval([1,2,3,4,5])
    assert u["mean"] == 3 and u["n"] == 5
    assert u["ci95"][0] < 3 < u["ci95"][1]


@pytest.mark.parametrize("ws,fs", [([], [1]),([1],[]),([1,1],[2]),([1],[-2]),([True],[1])])
def test_bad_experiment_seeds_are_rejected(ws,fs,tmp_path):
    with pytest.raises((ValueError,TypeError)):
        compare_objective_model(load_run_config("configs/rl_network.yaml"),load_objective_model("configs/objective_model.yaml"),ws,fs,tmp_path)
