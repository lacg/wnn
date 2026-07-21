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


def test_primitive_overflow_uses_i128():
	# >int64 values (e.g. sb=65 cell addresses — the 12/07/2026 pid@31337004 OOM)
	# transparently switch the WHOLE table to the 16-byte "i128" format.
	rows = [(0, 5, 1), (2, 2**63, 3), (7, 2**65 + 11, 0), (1, -2**64, 2)]
	packed = pack_int_columns(rows, 3)
	assert is_packed(packed) and packed.get("fmt") == "i128"
	out = unpack_int_columns(packed)
	assert out == [tuple(r) for r in rows], f"i128 roundtrip mismatch: {out}"
	# Overflow mid-row must not duplicate the partial row (extend raises mid-append).
	rows2 = [(1, 2, 3), (4, 2**70, 6)]
	out2 = unpack_int_columns(pack_int_columns(rows2, 3))
	assert out2 == [tuple(r) for r in rows2], f"partial-row dedup failed: {out2}"
	print(f"  {PASS} >int64 columns switch to i128 format and round-trip exactly")


def test_flat_array_roundtrip():
	vals = [0, 1, 47, 2**40, 2**52 - 1, 9, 9, 0]
	packed = pack_int_array(vals)
	assert is_packed(packed) and packed["cols"] == 1 and packed["n"] == len(vals)
	out = unpack_int_array(packed)
	assert out == vals, f"flat roundtrip mismatch: {out} != {vals}"
	assert isinstance(out, list) and not isinstance(out[0], tuple), "flat → plain ints"
	assert unpack_int_array(pack_int_array([])) == []
	print(f"  {PASS} flat int-array round-trip (returns flat list, not 1-tuples)")


def test_flat_array_overflow_uses_i128():
	vals = [0, 2**63, 1, 2**66 + 5, -7]
	packed = pack_int_array(vals)
	assert is_packed(packed) and packed.get("fmt") == "i128" and packed["n"] == len(vals)
	assert unpack_int_array(packed) == vals
	print(f"  {PASS} pack_int_array >int64 switches to i128 and round-trips")


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


def test_controller_genome_wide_addresses_roundtrip():
	# sb>63 genome (the pid@31337004 killer): addresses beyond int64 must pack
	# via i128 — serialize() has NO verbose-list fallback anymore.
	#
	# CONTRACT UPDATE (u64-keyed cells): the controller's cell memory is keyed
	# by u64 (compute_address_sparse → u64), and since the buffer-backed
	# MemoryPayload (8c87e273) the payload cannot hold an address ≥ 2^64 either
	# — from_triples raises OverflowError, loudly. (This test previously fed it
	# 2^64+3, which that commit had ALREADY broken; the assert below pins the
	# new behaviour.) In-range addresses > int64::MAX still need the i128 pack.
	import pytest
	from wnn.control.recurrent_genome import RecurrentArchGenome, RecurrentArchShape, MemoryPayload
	shape = RecurrentArchShape(prefix_factor=1, state_input_space=24,
	                           output_input_space=24, output_quantum=16)
	with pytest.raises(OverflowError):
		MemoryPayload.from_triples([(0, 2**64 + 3, 3)], [])
	state = [(0, 2**64 - 1, 3), (1, 2**63, 1)]
	output = [(0, 5, 2)]
	g = RecurrentArchGenome(
		shape=shape, state_neurons=2, output_neurons=1,
		state_sampled=[[0, 1], [2, 3]], output_sampled=[[0]],
		cells=MemoryPayload.from_triples(state, output),
	)
	d = g.serialize()
	assert is_packed(d["cells"]["state"]) and d["cells"]["state"].get("fmt") == "i128"
	g2 = RecurrentArchGenome.deserialize(d)
	assert g.cells.to_triples() == g2.cells.to_triples(), ">int64 cells changed across round-trip"
	print(f"  {PASS} sb>63 genome (2^64 addresses) round-trips via i128 — no verbose fallback")


def test_chunked_population_write_loads_back():
	# The chunked per-genome writer must produce a file the normal loader (and
	# extract_checkpoint_head, which needs final_population LAST) still reads.
	import tempfile
	from pathlib import Path
	from wnn.ram.strategies.phased import (
		PhaseCheckpoint, save_checkpoint, load_checkpoint, ControllerGenomeCodec)
	from wnn.ram.strategies.phased.checkpoint import extract_checkpoint_head
	codec = ControllerGenomeCodec()
	pop = [_make_genome_with_cells() for _ in range(3)]
	ckpt = PhaseCheckpoint(
		phase_key="4", phase_name="memory", strategy_type="GA",
		best_genome=pop[0], final_population=pop, iterations_run=6, patience=2,
		extra={"spec": {"state_neurons": 3}},
	)
	with tempfile.TemporaryDirectory() as td:
		p = save_checkpoint(Path(td) / "w.yaml.gz", ckpt, codec)
		back = load_checkpoint(p, codec)
		assert back is not None and len(back.final_population) == 3
		for a, b in zip(pop, back.final_population):
			assert a.fingerprint() == b.fingerprint(), "population changed across chunked write"
		assert back.iterations_run == 6 and back.patience == 2
		# skip_population + head extraction both rely on the same layout
		slim = load_checkpoint(p, codec, skip_population=True)
		assert slim.final_population is None and slim.best_genome is not None
		head = extract_checkpoint_head(p, Path(td) / "head.yaml.gz")
		slim2 = load_checkpoint(head, codec)
		assert slim2.best_genome is not None and not slim2.final_population
	print(f"  {PASS} chunked population write loads back (full, skip_population, head-extract)")


if __name__ == "__main__":
	tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	print(f"Running {len(tests)} packing/codec tests...")
	for t in tests:
		t()
	print("ALL PASS")
