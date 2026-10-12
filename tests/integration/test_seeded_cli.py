import json
import subprocess
import sys

import pytest

from drlmots.__main__ import smoke_report
from drlmots.workload import SyntheticWorkloadSettings


def test_seeded_tasks_have_repeatable_simulation_and_exact_accounting():
    settings = SyntheticWorkloadSettings(100, 42, (0, 3), (500, 2000), (20, 150))
    report = smoke_report(settings)
    assert report == smoke_report(settings)
    assert report["completed_count"] + report["failed_count"] == 100
    assert len({item["task_id"] for item in report["outcomes"]}) == 100
    assert report["workload"]["data_source"] == "synthetic_seeded"
    assert report["workload"]["seed"] == 42


def test_cli_runs_and_saves_seeded_workload_with_explicit_ranges(tmp_path):
    output = tmp_path / "report.json"
    result = subprocess.run([
        sys.executable, "-m", "drlmots", "--task-count", "4", "--seed", "0",
        "--interarrival-ms", "2", "2", "--length-bytes", "1000", "1000",
        "--data-size-kb", "20", "20", "--output", str(output),
    ], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["completed_count"] == 4
    assert report["failed_count"] == 0
    assert report["makespan_ms"] == 8
    assert report["workload"]["seed"] == 0
    assert [task["arrival_time"] for task in report["workload"]["tasks"]] == [0, 2, 4, 6]


def test_cli_empty_generated_workload():
    result = subprocess.run([sys.executable, "-m", "drlmots", "--task-count", "0", "--seed", "0"],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["completed_count"] == 0
    assert report["failed_count"] == 0
    assert report["average_completion_time_ms"] is None
    assert report["workload"]["data_source"] == "synthetic_seeded"


@pytest.mark.parametrize("arguments", [
    ["--task-count", "5"], ["--seed", "42"], ["--length-bytes", "500", "1000"],
    ["--task-count", "-1", "--seed", "42"], ["--task-count", "1.5", "--seed", "42"],
    ["--task-count", "5", "--seed", "42", "--interarrival-ms", "3", "2"],
    ["--task-count", "5", "--seed", "42", "--interarrival-ms", "0", "nan"],
])
def test_invalid_cli_inputs_do_not_write_results(arguments, tmp_path):
    output = tmp_path / "invalid.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", *arguments, "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert not output.exists()
