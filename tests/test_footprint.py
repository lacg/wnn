"""Footprint plumbing (IDS-37 stage 3): the accelerator tuple -> Footprint ->
dashboard payload -> cached validation, without the wheel. The numbers
themselves are computed in Rust (GenomeExport::footprint, cargo tests)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.ram.experiments.dashboard_client import DashboardClient
from wnn.ram.experiments.validation_cache_scope import ValidationCacheScope
from wnn.ram.footprint import Footprint

FP = Footprint(logical_neurons=500, physical_neurons=137, logical_cells=4_410_000, physical_cells=1_210_000)


def test_from_tuple_follows_the_abi17_order():
	assert Footprint.from_tuple((500, 137, 4_410_000, 1_210_000)) == FP
	assert Footprint.from_tuple(None) is None


def test_render_always_reports_both():
	assert FP.render([34] * 500) == "500n × 34b · 4.41 Mcells (logical) | 137n · 1.21 Mcells (physical)"
	plain = Footprint(5, 5, 37, 37)
	assert plain.render([4, 14, 14, 4, 14]) == "5n × 4-14b · 37 cells (logical) | 5n · 37 cells (physical)"


def test_payload_round_trip_and_incomplete_rows():
	assert Footprint.from_payload(FP.to_payload()) == FP
	assert Footprint.from_payload(None) is None
	assert Footprint.from_payload(dict(FP.to_payload(), physical_cells=None)) is None


def _client(responses: list):
	client = DashboardClient.__new__(DashboardClient)
	client._logger = lambda msg: None
	client.sent = []

	def request(method, path, json_data=None, params=None):
		client.sent.append((method, path, json_data))
		return responses.pop(0)

	client._request = request
	return client


def test_summary_posts_the_four_columns_flat():
	client = _client([{"id": 1}])
	client.create_validation_summary(7, "final", "best_f1", "h", 0.1, 0.9, footprint=FP)
	_, path, data = client.sent[0]
	assert path == "/api/experiments/7/summaries"
	assert {k: data[k] for k in FP.to_payload()} == FP.to_payload()


def test_summary_without_footprint_sends_no_columns():
	client = _client([{"id": 1}])
	client.create_validation_summary(7, "final", "best_f1", "h", 0.1, 0.9)
	assert not set(FP.to_payload()) & set(client.sent[0][2])


def test_cached_validation_carries_the_footprint():
	hit = {"found": True, "ce": 0.1, "accuracy": 0.9, "f1_macro": 0.8, "fpr": 0.05,
	       "threshold_metadata": None, "footprint": FP.to_payload()}
	client = _client([hit, dict(hit, footprint=None)])
	scope = ValidationCacheScope("k", 17)
	assert client.check_cached_validation("h", scope)[5] == FP
	assert client.check_cached_validation("h", scope)[5] is None
