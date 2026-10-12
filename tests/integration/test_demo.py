import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from drlmots.demo import demo_scenarios
from drlmots.run_config import load_run_config


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "resources.yaml"


def test_presets_keep_resource_inputs_and_pair_workloads():
    scenarios = demo_scenarios(load_run_config(CONFIG))
    assert [s["id"] for s in scenarios] == ["worked", "seeded", "pressure"]
    assert [len(s["comparison"]["reports"][0]["outcomes"]) for s in scenarios] == [6, 100, 25]
    seeded = scenarios[1]["comparison"]["reports"]
    assert [(r["completed_count"], r["failed_count"]) for r in seeded] == [(96, 4), (91, 9)]
    for scenario in scenarios:
        a, b = scenario["comparison"]["reports"]
        assert a["workload"] == b["workload"]
        assert a["topology"] == b["topology"]
        assert a["resource_units"] == b["resource_units"]


def test_actual_demo_cli_exports_safe_embedded_data_and_valid_figures(tmp_path):
    pytest.importorskip("matplotlib")
    # Legal identifiers containing HTML must stay inert inside the JSON script.
    malicious = "</script><script>alert(1)</script>"
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG.read_text(encoding="utf-8").replace("vm-cloud", malicious), encoding="utf-8")
    output_dir = tmp_path / "demo"
    result = subprocess.run([sys.executable, "-m", "drlmots.demo", "--config", str(config),
                             "--output-dir", str(output_dir)], capture_output=True, text=True, check=True)
    html = (output_dir / "drlmots-demo.html").read_text(encoding="utf-8")
    embedded = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "</script>" not in embedded
    bundle = json.loads(embedded)
    assert bundle == json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    assert malicious in {vm["vm_id"] for vm in bundle["scenarios"][0]["comparison"]["reports"][0]["topology"]["vms"]}
    assert "__DEMO_DATA__" not in html
    assert re.search(r'<(?:script|link|img)\b[^>]*(?:src|href)=', html) is None
    assert "Demo:" in result.stdout
    assert len(list((output_dir / "figures").glob("*.png"))) == 15
    assert len(list((output_dir / "figures").glob("*.svg"))) == 15
    for scenario in bundle["scenarios"]:
        for report in scenario["comparison"]["reports"]:
            svg = scenario["figures"]["policies"][report["policy"]]["timeline"]
            root = ET.fromstring(svg)
            assert svg.startswith("<svg")
            assert root.tag == "{http://www.w3.org/2000/svg}svg"
            assert len(root.findall(".//{http://www.w3.org/2000/svg}title")) == report["completed_count"]
