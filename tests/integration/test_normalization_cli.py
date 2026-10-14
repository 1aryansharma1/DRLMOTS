"""Actual normalization runs, split isolation, compatibility and CLI artifacts."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from drlmots.__main__ import configured_report
from drlmots.normalization import NormalizationContext, StateNormalizer, capture_batch, fit_normalizer
from drlmots.run_config import load_run_config


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "configs" / "normalization_train.yaml"
EVAL = ROOT / "configs" / "normalization_evaluation.yaml"


def cli(*args):
    return subprocess.run([sys.executable, "-m", "drlmots.normalization", *map(str, args)],
                          capture_output=True, text=True)


def test_real_training_evaluation_isolation_and_baseline_preservation():
    train, evaluation = load_run_config(TRAIN), load_run_config(EVAL)
    original = deepcopy((train, evaluation))
    training, report = capture_batch(train, split="train")
    scaler = fit_normalizer([training])
    before = scaler.manifest()
    test, held_out = capture_batch(evaluation, split="test")
    transformed = scaler.transform_batch(test)
    assert len(transformed) == 100
    assert all(len(row) == 47 and all(math.isfinite(v) and 0 <= v <= 1 for v in row) for row in transformed)
    assert scaler.manifest() == before
    assert report == configured_report(train, capture_decisions=True)
    assert held_out == configured_report(evaluation, capture_decisions=True)
    assert (train, evaluation) == original
    assert (report["completed_count"], report["failed_count"]) == (96, 4)
    assert before["training_sources"][0]["provenance"]["generator"]["settings"]["seed"] == 42
    assert before["training_sources"][0]["provenance"]["workload_sha256"] != held_out["workload"]["sha256"]


@pytest.mark.parametrize("change", ["queue", "capacity", "tier", "unit", "host", "link", "membership"])
def test_topology_or_catalogue_changes_rejected_even_at_same_dimension(change):
    config = load_run_config(TRAIN)
    train, _ = capture_batch(config, split="train")
    scaler = fit_normalizer([train])
    altered = deepcopy(config)
    if change == "queue":
        altered.topology.vms[0].queue_capacity += 1
    elif change == "capacity":
        altered.topology.vms[0].resource_capacities["cpu_units"] += 1
    elif change == "unit":
        altered.resource_units["cpu_units"] = "different units"
    elif change == "host":
        altered.topology.hosts[0].memory_gb += 1
    elif change == "link":
        end = altered.topology.end_nodes[0]
        altered = replace(altered, topology=replace(altered.topology, end_nodes=(replace(end, fog_node_ids=("fog-a",)),)))
    elif change == "tier":
        vm = altered.topology.vms[0]
        cloud_vm = altered.topology.vms[-1]
        cloud_vm.tier = vm.tier
        vm.tier = type(vm.tier).CLOUD
        a, b = altered.topology.fog_nodes
        cloud = altered.topology.cloud_datacenters[0]
        altered = replace(altered, topology=replace(altered.topology,
            fog_nodes=(replace(a, vm_ids=(cloud_vm.vm_id,)), b),
            cloud_datacenters=(replace(cloud, vm_ids=(vm.vm_id,)),)))
    else:
        # Move the two fog VM IDs between fog memberships: same dimensions, different topology.
        a, b = altered.topology.fog_nodes
        altered = replace(altered, topology=replace(altered.topology, fog_nodes=(
            replace(a, vm_ids=b.vm_ids), replace(b, vm_ids=a.vm_ids))))
    ctx = NormalizationContext.from_topology(altered.topology, altered.resource_units)
    assert ctx.schema.dimension == train.context.schema.dimension
    with pytest.raises(ValueError, match="incompatible"):
        scaler.transform(ctx, train.states[0])


def test_topology_list_permutations_are_compatible():
    config = load_run_config(TRAIN)
    topology = config.topology
    reordered = replace(topology, vms=tuple(reversed(topology.vms)), hosts=tuple(reversed(topology.hosts)),
                        fog_nodes=tuple(reversed(topology.fog_nodes)))
    assert NormalizationContext.from_topology(topology, config.resource_units) == NormalizationContext.from_topology(reordered, config.resource_units)


def test_same_tasks_under_another_policy_are_not_a_held_out_run():
    config = load_run_config(TRAIN)
    training, _ = capture_batch(config, split="train")
    test, _ = capture_batch(config, split="test", policy="first_fit_fcfs")
    with pytest.raises(ValueError, match="overlaps"):
        fit_normalizer([training]).transform_batch(test)


def test_fit_apply_actual_cli_roundtrip_and_unchanged_artifact(tmp_path):
    artifact, output = tmp_path / "model.json", tmp_path / "evaluation.json"
    fitted = cli("fit", "--train-config", TRAIN, "--output", artifact)
    assert fitted.returncode == 0, fitted.stderr
    original = artifact.read_bytes()
    applied = cli("apply", "--config", EVAL, "--normalizer", artifact, "--output", output)
    assert applied.returncode == 0, applied.stderr
    result = json.loads(output.read_text())
    assert result["normalization"]["split"] == "test"
    assert result["normalization"]["artifact_sha256"] == json.loads(original)["sha256"]
    for decision in result["decision_trace"]["decisions"]:
        assert len(decision.pop("normalized_state")) == 47
    result.pop("normalization")
    expected = configured_report(load_run_config(EVAL), capture_decisions=True)
    assert result == json.loads(json.dumps(expected))
    assert artifact.read_bytes() == original
    ctx = NormalizationContext.from_topology(load_run_config(TRAIN).topology, load_run_config(TRAIN).resource_units)
    assert StateNormalizer.load(artifact, context=ctx).manifest()["training_vector_count"] == 100


def test_cli_invalid_apply_has_no_output_or_artifact_mutation(tmp_path):
    artifact, output = tmp_path / "model.json", tmp_path / "bad.json"
    assert cli("fit", "--train-config", TRAIN, "--output", artifact).returncode == 0
    original = artifact.read_bytes()
    rejected = cli("apply", "--config", TRAIN, "--normalizer", artifact, "--output", output)
    assert rejected.returncode != 0 and "overlaps" in rejected.stderr
    assert not output.exists() and artifact.read_bytes() == original
    overwrite = cli("apply", "--config", EVAL, "--normalizer", artifact, "--output", artifact)
    assert overwrite.returncode != 0 and "overwrite" in overwrite.stderr
    assert artifact.read_bytes() == original


def test_cli_fit_cannot_overwrite_its_training_config(tmp_path):
    config = tmp_path / "training.yaml"
    config.write_bytes(TRAIN.read_bytes())
    original = config.read_bytes()
    result = cli("fit", "--train-config", config, "--output", config)
    assert result.returncode != 0 and "overwrite" in result.stderr
    assert config.read_bytes() == original
