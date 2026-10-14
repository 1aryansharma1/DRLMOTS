"""Real normalized observations and CLI inference/checkpoint smoke checks."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from drlmots.__main__ import configured_report
from drlmots.actions import action_mask
from drlmots.environment import simulate
from drlmots.normalization import capture_batch, fit_normalizer
from drlmots.q_network import MaskedSelector, NetworkContext, QNetwork
from drlmots.reward import load_reward_config
from drlmots.run_config import load_run_config
from drlmots.workload import seeded_workload


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "configs/normalization_train.yaml"
EVAL = ROOT / "configs/normalization_evaluation.yaml"
REWARD = ROOT / "configs/reward.yaml"


def test_real_inference_preserves_baseline_scaler_and_reward_reports():
    training, _ = capture_batch(load_run_config(TRAIN), split="train")
    normalizer = fit_normalizer([training])
    reward = load_reward_config(REWARD)
    context = NetworkContext.from_artifacts(normalizer, reward)
    net = QNetwork(context)
    config = load_run_config(EVAL)
    before = deepcopy((normalizer.manifest(), config))
    plain, scored = configured_report(config), configured_report(config, reward_config=reward)
    tasks = seeded_workload(config.workload)
    result = simulate(config.topology, tasks, task_length_unit="bytes",
                      resource_units=config.resource_units, reward_config=reward)
    selector = MaskedSelector(42)
    values = []
    for decision in result.decisions:
        q = net.infer(decision.observation, normalizer=normalizer, reward=reward)
        assert q.shape == (3,) and not q.requires_grad and torch.isfinite(q).all()
        mask = action_mask(decision.observation)
        action = selector.select(q, mask, epsilon=0.2)
        assert action is None if not any(mask) else mask[action]
        values.append(q)
    held_out, _ = capture_batch(config, split="test")
    states = torch.tensor(normalizer.transform_batch(held_out), dtype=torch.float32)
    assert torch.allclose(torch.stack(values), net(states, context=context), rtol=1e-5, atol=1e-7)
    assert (normalizer.manifest(), config) == before
    assert configured_report(config) == plain
    assert configured_report(config, reward_config=reward) == scored
    assert (plain["completed_count"], plain["failed_count"]) == (92, 8)
    with pytest.raises(ValueError, match="incompatible"):
        net.infer(result.decisions[0].observation, normalizer=normalizer,
                  reward=replace(reward, rejection_penalty=2))
    different_scaler = fit_normalizer([replace(training, source_id="different-training-source")])
    with pytest.raises(ValueError, match="incompatible"):
        net.infer(result.decisions[0].observation, normalizer=different_scaler, reward=reward)


def cli(*args):
    return subprocess.run([sys.executable, "-m", "drlmots.q_network", *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture
def paths(tmp_path):
    training, _ = capture_batch(load_run_config(TRAIN), split="train")
    normalizer = tmp_path / "normalizer.json"
    fit_normalizer([training]).save(normalizer)
    return normalizer, tmp_path / "network.json", tmp_path / "decisions.json"


def test_cli_held_out_inference_reloads_checkpoint_and_is_reproducible(paths):
    normalizer, checkpoint, output = paths
    args = ("--config", EVAL, "--normalizer", normalizer, "--reward-config", REWARD,
            "--checkpoint", checkpoint, "--output", output, "--epsilon", "0.3")
    original = normalizer.read_bytes()
    for _ in range(2):
        process = cli(*args)
        assert process.returncode == 0, process.stderr
        current = output.read_bytes()
        if _ == 0:
            first, weights = current, checkpoint.read_bytes()
        else:
            assert current == first and checkpoint.read_bytes() == weights
    report = json.loads(output.read_text())
    assert "not executed" in report["scope"]
    assert report["checkpoint_reload_equal"] is True
    assert len(report["decisions"]) == 100
    assert sum(d["selected_action"] is None for d in report["decisions"]) == 8
    for decision in report["decisions"]:
        action = decision["selected_action"]
        if action is not None:
            assert decision["action_mask"][action]
            assert decision["selected_vm_id"] == report["context"]["action_vm_ids"][action]
    assert normalizer.read_bytes() == original


@pytest.mark.parametrize("case", ["epsilon", "seed", "width", "overlap", "overwrite"])
def test_cli_invalid_inputs_do_not_create_artifacts(paths, case):
    normalizer, checkpoint, output = paths
    inputs = ["--config", TRAIN if case == "overlap" else EVAL,
              "--normalizer", normalizer, "--reward-config", REWARD,
              "--checkpoint", checkpoint, "--output", output]
    extra = {"epsilon": ["--epsilon", "nan"], "seed": ["--seed", "-1"],
             "width": ["--hidden-widths", "0", "50"], "overlap": [],
             "overwrite": ["--checkpoint", normalizer]}[case]
    original = normalizer.read_bytes()
    process = cli(*inputs, *extra)
    assert process.returncode != 0 and "error:" in process.stderr
    assert not output.exists() and not checkpoint.exists()
    assert original == normalizer.read_bytes()
