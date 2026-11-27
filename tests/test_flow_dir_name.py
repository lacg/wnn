"""IDS-81: the worker's checkpoint-folder rule and the dashboard's twin
(dashboard/src/api/checkpoint_dir_name.rs) agree on the shared cases.

Run: PYTHONPATH=src python -m pytest tests/test_flow_dir_name.py -q
"""
import json
from pathlib import Path

from wnn.ram.experiments.flow_dir_name import flow_dir_name

CASES = Path(__file__).resolve().parents[1] / "src/wnn/ram/experiments/flow_dir_name_cases.json"


def test_worker_rule_matches_the_shared_cases():
	cases = json.loads(CASES.read_text())["cases"]
	checked = 0
	for name, want in cases:
		if want is None:  # dashboard refuses these; the worker value is irrelevant
			continue
		assert flow_dir_name(name) == want, name
		checked += 1
	assert checked >= 4


def test_dotted_rate_keeps_its_dot():
	assert flow_dir_name("X-rate0.25-abi30") == "x-rate0.25-abi30"
