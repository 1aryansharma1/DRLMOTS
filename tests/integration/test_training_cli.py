from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

from drlmots.dqn import TrainingConfig
from drlmots.run_config import load_run_config
from drlmots.train import checkpoint_artifact, load_checkpoint, write_json


ROOT = Path(__file__).resolve().parents[2]


def cli(*args):
    return subprocess.run([sys.executable, "-m", "drlmots.train", *map(str, args)], capture_output=True, text=True)


def inputs(tmp_path, *, network=False, episodes=2):
    config = load_run_config(ROOT / ("configs/rl_network.yaml" if network else "configs/normalization_train.yaml"))
    config = replace(config, workload=replace(config.workload, task_count=10))
    data = config.manifest()
    data.pop("sha256")
    source = tmp_path / ("network.yaml" if network else "standard.yaml")
    source.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = TrainingConfig(episodes=episodes, calibration_episodes=1, batch_size=4, replay_capacity=32,
                              hidden_widths=(8, 8), target_sync_updates=3, evaluation_seeds=(3000, 3001))
    hyper = tmp_path / "training.yaml"
    hyper.write_text(yaml.safe_dump(settings.manifest()), encoding="utf-8")
    return source, hyper


def test_actual_train_resume_and_evaluate_commands(tmp_path):
    config, settings = inputs(tmp_path)
    output = tmp_path / "trained"
    process = cli("train", "--config", config, "--training-config", settings, "--output-dir", output)
    assert process.returncode == 0, process.stderr
    checkpoint = output / "checkpoint.json"
    learner = load_checkpoint(checkpoint, for_resume=True)
    assert learner.episodes_completed == 2 and learner.updates > 0
    assert json.loads((output / "last-training-episode.json").read_text())["rl"]["learn"] is True
    process = cli("resume", "--checkpoint", checkpoint, "--additional-episodes", "1", "--output-dir", output)
    assert process.returncode == 0, process.stderr
    resumed = load_checkpoint(checkpoint, for_resume=True)
    assert resumed.episodes_completed == 3 and resumed.updates > learner.updates
    before = checkpoint.read_bytes()
    evaluation = tmp_path / "evaluation.json"
    process = cli("evaluate", "--checkpoint", checkpoint, "--output", evaluation, "--seeds", "3100", "3101")
    assert process.returncode == 0, process.stderr
    assert checkpoint.read_bytes() == before
    result = json.loads(evaluation.read_text())
    assert result["evaluation_seeds"] == [3100, 3101] and len(result["summaries"]) == 3
    overlap = cli("evaluate", "--checkpoint", checkpoint, "--output", tmp_path / "bad.json", "--seeds", "2000")
    assert overlap.returncode != 0 and "overlaps" in overlap.stderr
    assert not (tmp_path / "bad.json").exists()
    overwrite = cli("evaluate", "--checkpoint", checkpoint, "--output", checkpoint)
    assert overwrite.returncode != 0 and "overwrite" in overwrite.stderr
    assert checkpoint.read_bytes() == before
    duplicate = cli("train", "--config", config, "--training-config", settings, "--output-dir", output)
    assert duplicate.returncode != 0 and "already has" in duplicate.stderr


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.paths = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img":
            self.paths.append(attrs["src"])
        if tag == "a":
            self.paths.append(attrs["href"])


def test_actual_benchmark_builds_all_modes_report_and_resumable_checkpoints(tmp_path):
    pytest.importorskip("matplotlib")
    standard, settings = inputs(tmp_path, episodes=2)
    network, _ = inputs(tmp_path, network=True, episodes=2)
    output = tmp_path / "professor"
    args = ("benchmark", "--standard-config", standard, "--network-config", network,
            "--training-config", settings, "--output-dir", output)
    process = cli(*args)
    assert process.returncode == 0, process.stderr
    names = ("standard-zero", "standard-network", "fog-preferred-network")
    before = {}
    for name in names:
        agent = load_checkpoint(output / name / "checkpoint.json", for_resume=True)
        assert agent.episodes_completed == 2 and agent.updates > 0
        before[name] = checkpoint_artifact(agent)["sha256"]
    html = (output / "training-demo.html").read_text(encoding="utf-8")
    assert "actually selected VMs" in html and "explicit completion-time constraint" in html
    assert "<script" not in html
    for relative in Links(html).paths:
        assert "://" not in relative and (output / relative).is_file()
    assert len(list((output / "figures").glob("*.png"))) == 5
    assert len(list((output / "figures").glob("*.svg"))) == 5
    assert len(json.loads((output / "standard-evaluation.json").read_text())["summaries"]) == 3
    assert len(json.loads((output / "network-evaluation.json").read_text())["summaries"]) == 5
    with zipfile.ZipFile(output / "professor-demo.zip") as archive:
        assert "training-demo.html" in archive.namelist()
        assert all(".." not in name for name in archive.namelist())
        assert archive.testzip() is None
    process = cli(*args, "--resume")
    assert process.returncode == 0, process.stderr
    for name in names:
        assert checkpoint_artifact(load_checkpoint(output / name / "checkpoint.json"))["sha256"] == before[name]


def test_cli_bad_training_configuration_fails_before_outputs(tmp_path):
    config, settings = inputs(tmp_path)
    raw = yaml.safe_load(settings.read_text())
    raw["evaluation_seeds"] = [2000]
    settings.write_text(yaml.safe_dump(raw))
    output = tmp_path / "bad-run"
    result = cli("train", "--config", config, "--training-config", settings, "--output-dir", output)
    assert result.returncode != 0 and "disjoint" in result.stderr
    assert not output.exists()


def test_atomic_checkpoint_save_retries_windows_sharing_lock_without_deleting_old_file(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint.json"
    destination.write_text('{"previous":true}')
    original = Path.replace
    attempts = []
    def transient_replace(path, target):
        attempts.append(path)
        if len(attempts) <= 2:
            assert json.loads(destination.read_text()) == {"previous": True}
            error = PermissionError("temporarily locked")
            error.winerror = 32
            raise error
        return original(path, target)
    monkeypatch.setattr(Path, "replace", transient_replace)
    monkeypatch.setattr("drlmots.train.time.sleep", lambda seconds: None)
    write_json(destination, {"new": True})
    assert len(attempts) == 3
    assert json.loads(destination.read_text()) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_cannot_claim_training_success_without_an_optimizer_update(tmp_path):
    config, settings = inputs(tmp_path, episodes=1)
    raw = yaml.safe_load(config.read_text())
    raw["workload"]["task_count"] = 1
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path / "warmup-only"
    result = cli("train", "--config", config, "--training-config", settings, "--output-dir", output)
    assert result.returncode != 0 and "no replay updates" in result.stderr
    assert (output / "checkpoint.json").exists()
    assert not (output / "evaluation.json").exists()
