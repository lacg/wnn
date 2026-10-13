"""FPGA export fidelity on REAL cohort genomes (IDS-37 stage 7).

For a rate-1.0 (SR1) and a rate-0.25 (GAF) genome: the exporter re-trains the
genome exactly as final validation did and must reproduce the recorded DB row
(every threshold mode, bit-exact) — else it refuses to write. Then the emitted
per-unit tables, driven through the RTL ADDER CONTRACT (sum over units of
count_u x quarters(cell_u)), must equal the Rust score x 4N on held-out vectors
— the same check run_sim.sh makes in xsim. At rate 1.0 duplicates collapse
(physical < logical); at 0.25 they cannot (physical == logical).

Reads the dashboard DB read-only; skipped where it is absent. Needs the ABI-17
wheel (IDSCacheWrapper.export_trained).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

DB = Path("/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db")
SCRIPTS = Path(__file__).resolve().parents[1] / "fpga" / "scripts"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="dashboard DB not mounted")
sys.path.insert(0, str(SCRIPTS))


def _adder_sums(trained, rows: np.ndarray) -> np.ndarray:
	"""What the generated RTL computes: per vector, sum_u count_u x quarters(cell_u(addr_u))."""
	from export_test_vectors import rows_as_bits
	from export_metadata import input_bits
	layer = trained.layer
	bits = rows_as_bits(trained.dataset.X_test, rows, input_bits(trained.dataset.X_test)).astype(np.uint64)
	weights = np.asarray(layer.cell_weight_quarters, dtype=np.int64)
	total = np.zeros(len(rows), dtype=np.int64)
	for unit, count in zip(layer.units, layer.counts):
		addr = np.zeros(len(rows), dtype=np.uint64)
		for c in unit.connections:
			addr = (addr << np.uint64(1)) | bits[:, c]
		pos = np.searchsorted(unit.keys, addr)
		hit = (pos < len(unit.keys)) & (unit.keys[np.minimum(pos, len(unit.keys) - 1)] == addr)
		cells = np.where(hit, unit.cells[np.minimum(pos, len(unit.keys) - 1)], layer.miss_cell)
		total += count * weights[cells]
	return total


@pytest.fixture(scope="module", params=[(6945, 1.0), (6928, 0.25)], ids=["SR1-rate1.0", "GAF-rate0.25"])
def trained(request):
	from flow_genome_source import FlowGenomeSource
	from flow_layer_trainer import FlowLayerTrainer
	flow_id, rate = request.param
	source = FlowGenomeSource.read(str(DB), flow_id, "best_f1", "ga_neurons")
	assert source.params["neuron_sample_rate"] == rate
	return FlowLayerTrainer(log=lambda m: None).train(source)


def test_export_reproduces_the_recorded_row(trained):
	assert trained.modes_verified >= 6, "train_cal, fixed_05, val_cal + the calibrations"
	fp = trained.layer.footprint
	assert fp.logical_neurons == len(trained.source.bits_per_neuron)
	if trained.settings.neuron_sample_rate >= 1.0:
		assert fp.physical_neurons < fp.logical_neurons, "rate 1.0 duplicates collapse"
	else:
		assert fp.physical_neurons == fp.logical_neurons, "rate < 1.0: copies may differ, plain layer"
	assert len(trained.layer.units) == fp.physical_neurons == len(trained.layer.counts)
	assert sum(trained.layer.counts) == fp.logical_neurons


def test_rtl_adder_equals_rust_scores(trained):
	from export_test_vectors import select_vectors
	rows = select_vectors(trained.labels, 400)
	expected = np.rint(np.asarray(trained.layer.eval_scores)[rows] * 4 * trained.layer.footprint.logical_neurons)
	assert np.array_equal(_adder_sums(trained, rows), expected.astype(np.int64))


def test_fidelity_gate_refuses_a_tampered_row(trained):
	from fidelity_gate import FidelityGate, FidelityMismatch
	tampered = {k: dict(v) for k, v in trained.source.threshold_metadata.items() if isinstance(v, dict)}
	tampered["fixed_05"]["f1"] += 1.0 / 41166  # one flipped prediction's worth
	gate = FidelityGate(trained.labels, 0)
	with pytest.raises(FidelityMismatch, match="fixed_05.f1"):
		gate.require(trained.layer.eval_scores, tampered)


def test_files_and_rtl_are_per_unit(trained, tmp_path):
	from export_genome import write_sparse_units
	from export_test_vectors import write_test_vectors
	from generate_rtl import generate_classifier_impl, load_export, write_mem_files
	meta = write_sparse_units(tmp_path, trained)
	write_test_vectors(tmp_path / "testbench", trained, meta, 50)
	meta, connections, sparse = load_export(tmp_path)
	assert len(connections) == len(sparse) == meta["physical_neurons"]
	write_mem_files(tmp_path, sparse)
	sv = generate_classifier_impl(tmp_path, meta, connections, sparse).read_text()
	assert sv.count("wnn_neuron #(") == meta["physical_neurons"]
	assert f"localparam int NUM_UNITS  = {meta['physical_neurons']};" in sv
	assert all(f"ACC_BITS'({c}) * ACC_BITS'(cell_quarters(neuron_latched[{u}]))" in sv for u, c in enumerate(meta["counts"]))
	assert len(list((tmp_path / "mem").glob("unit_*_keys.mem"))) == meta["physical_neurons"]
