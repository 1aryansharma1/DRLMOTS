import json
from pathlib import Path
import subprocess
import sys
from importlib.metadata import version


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "resources.yaml"


def test_actual_cli_resource_run_has_balances_events_and_provenance(tmp_path):
    output = tmp_path / "resources.json"
    result = subprocess.run([sys.executable, "-m", "drlmots", "--config", str(CONFIG), "--output", str(output)],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["version"] == version("drlmots")
    assert report["run_configuration"]["schema_version"] == 2
    assert report["resource_units"]["memory_gb"] == "GB"
    assert (report["completed_count"], report["failed_count"]) == (4, 2)
    assert report["makespan_ms"] == 4
    assert len(report["resource_events"]) == 8
    assert report["available_resources"]["vm-fog-a"] == {"cpu_units": 2, "memory_gb": 4}
    assert report["workload"]["generator"]["settings"]["required_resources"] == {"cpu_units": 2, "memory_gb": 4}
