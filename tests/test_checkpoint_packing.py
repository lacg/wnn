"""Tests for the shared int-column packing primitive + its use in the
controller genome codec (13/06/2026 checkpoint-bloat fix).

Run: PYTHONPATH=src python tests/test_checkpoint_packing.py
"""
import sys

from wnn.ram.strategies.phased.packing import (
	pack_int_columns, unpack_int_columns, pack_int_array, unpack_int_array, is_packed,
)

PASS = "[PASS]"


def test_primitive_roundtrip():
	rows = [(0, 17, 3), (5, 2**40, 1), (123, 0, 0), (1, 2**52 - 1, 2)]
	packed = pack_int_columns(rows, 3)
	assert is_packed(packed), "should be tagged packed"
	assert packed["n"] == len(rows) and packed["cols"] == 3
	out = unpack_int_columns(packed)
	assert out == rows, f"roundtrip mismatch: {out} != {rows}"
	print(f"  {PASS} primitive round-trip ({len(rows)} triples, incl. addr=2^52-1)")


def test_primitive_empty():
	packed = pack_int_columns([], 3)
	assert packed["n"] == 0 and unpack_int_columns(packed) == []
	print(f"  {PASS} empty round-trip")


def test_primitive_overflow_signals():
	try:
		pack_int_columns([(0, 2**63, 0)], 3)  # > int64 max
	except OverflowError:
		print(f"  {PASS} value > int64 raises OverflowError (caller falls back)")
		return
	raise AssertionError("expected OverflowError for value > int64")


def test_flat_array_roundtrip():
	vals = [0, 1, 47, 2**40, 2**52 - 1, 9, 9, 0]
	packed = pack_int_array(vals)
	assert is_packed(packed) and packed["cols"] == 1 and packed["n"] == len(vals)
	out = unpack_int_array(packed)
	assert out == vals, f"flat roundtrip mismatch: {out} != {vals}"
	assert isinstance(out, list) and not isinstance(out[0], tuple), "flat → plain ints"
	assert unpack_int_array(pack_int_array([])) == []
	print(f"  {PASS} flat int-array round-trip (returns flat list, not 1-tuples)")


def test_flat_array_overflow_signals():
	try:
		pack_int_array([0, 2**63, 1])  # > int64 max
	except OverflowError:
		print(f"  {PASS} pack_int_array value > int64 raises OverflowError")
		return
	raise AssertionError("expected OverflowError for value > int64")


def test_cluster_codec_packs_connections():
	"""ClusterGenomeCodec compacts the bulk `connections` array at the checkpoint
	seam while round-tripping the genome unchanged."""
	from wnn.ram.genome import ClusterGenome
	from wnn.ram.strategies.phased import ClusterGenomeCodec
	g = ClusterGenome(bits_per_neuron=[3, 3, 4], neurons_per_cluster=[2, 1],
	                  connections=[0, 5, 11, 2, 7, 9, 1, 3, 6, 8])
	codec = ClusterGenomeCodec()
	enc = codec.encode(g)
	assert is_packed(enc["connections"]), "connections should be packed in the checkpoint payload"
	g2 = codec.decode(enc)
	assert g2.connections == g.connections, "connections changed across codec round-trip"
	assert g2.bits_per_neuron == g.bits_per_neuron and g2.neurons_per_cluster == g.neurons_per_cluster
	print(f"  {PASS} ClusterGenomeCodec packs connections + round-trips genome")


def test_cluster_codec_legacy_plain_connections():
	"""A legacy checkpoint with a plain `connections` list must still decode (the
	worker may resume from pre-packing checkpoints)."""
	from wnn.ram.strategies.phased import ClusterGenomeCodec
	codec = ClusterGenomeCodec()
	legacy = {"bits_per_neuron": [3, 3], "neurons_per_cluster": [2],
	          "connections": [0, 5, 11, 2, 7, 9]}  # verbose plain list
	g = codec.decode(legacy)
	assert g.connections == legacy["connections"]
	print(f"  {PASS} legacy plain-list connections still decodes (backward-compat)")


def test_cluster_codec_no_connections():
	"""Arch-only genome (connections not initialized) round-trips with no packing."""
	from wnn.ram.strategies.phased import ClusterGenomeCodec
	codec = ClusterGenomeCodec()
	from wnn.ram.genome import ClusterGenome
	g = ClusterGenome(bits_per_neuron=[3, 3], neurons_per_cluster=[2])
	enc = codec.encode(g)
	assert "connections" not in enc or enc.get("connections") is None
	g2 = codec.decode(enc)
	assert g2.connections is None
	print(f"  {PASS} connections=None genome round-trips (no packing)")


def test_is_packed_negatives():
	assert not is_packed([[1, 2, 3]])
	assert not is_packed({"foo": "bar"})
	assert not is_packed(None)
	print(f"  {PASS} is_packed rejects legacy lists / plain dicts / None")


def _make_genome_with_cells():
	from wnn.control.recurrent_genome import RecurrentArchGenome, RecurrentArchShape, MemoryPayload
	shape = RecurrentArchShape(prefix_factor=1, state_input_space=24,
	                           output_input_space=24, output_quantum=16)
	# small synthetic cells: (neuron, addr, value)
	state = [(0, 5, 3), (0, 9, 1), (1, 2, 2), (2, 100, 3)]
	output = [(0, 0, 1), (3, 4095, 2), (7, 1, 0)]
	cells = MemoryPayload.from_triples(state, output)
	g = RecurrentArchGenome(
		shape=shape, state_neurons=3, output_neurons=8,
		state_sampled=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
		output_sampled=[[0], [1], [2], [3], [4], [5], [6], [7]],
		cells=cells,
	)
	return g


def test_controller_genome_roundtrip_packed():
	g = _make_genome_with_cells()
	d = g.serialize()
	# New format: cells is a dict with packed state/output
	assert isinstance(d["cells"], dict), f"expected packed dict, got {type(d['cells'])}"
	assert is_packed(d["cells"]["state"]) and is_packed(d["cells"]["output"])
	from wnn.control.recurrent_genome import RecurrentArchGenome
	g2 = RecurrentArchGenome.deserialize(d)
	assert g2.cells is not None
	assert g.cells.to_triples() == g2.cells.to_triples(), "cells changed across round-trip"
	assert g.fingerprint() == g2.fingerprint(), "fingerprint changed across round-trip"
	print(f"  {PASS} controller genome serialize→deserialize preserves cells (packed)")


def test_controller_genome_legacy_load():
	"""A checkpoint written in the OLD verbose shape ([state, output] lists) must
	still deserialize identically — backward compatibility for existing files."""
	g = _make_genome_with_cells()
	d = g.serialize()
	st, ot = g.cells.to_triples()
	d_legacy = dict(d)
	d_legacy["cells"] = [list(st), list(ot)]  # legacy verbose shape
	from wnn.control.recurrent_genome import RecurrentArchGenome
	g_legacy = RecurrentArchGenome.deserialize(d_legacy)
	assert g_legacy.cells.to_triples() == g.cells.to_triples()
	print(f"  {PASS} legacy verbose-cells checkpoint still loads (backward-compat)")


def test_controller_genome_no_cells():
	from wnn.control.recurrent_genome import RecurrentArchGenome, RecurrentArchShape
	g = RecurrentArchGenome(
		shape=RecurrentArchShape(1, 24, 24, 16), state_neurons=1, output_neurons=1,
		state_sampled=[[0]], output_sampled=[[0]], cells=None)
	d = g.serialize()
	assert d["cells"] is None
	g2 = RecurrentArchGenome.deserialize(d)
	assert g2.cells is None
	print(f"  {PASS} cells=None (paradigm-A arch genome) round-trips as None")


if __name__ == "__main__":
	tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	print(f"Running {len(tests)} packing/codec tests...")
	for t in tests:
		t()
	print("ALL PASS")
