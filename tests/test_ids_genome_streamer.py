"""Tests for IDSGenomeStreamer (Option F item 2).

The critical correctness test: streaming train + score across N chunks
must produce the same final metrics as the in-memory IDSCache one-shot
path on the same data.

NOTE: bit-exact equivalence is NOT guaranteed because the in-memory path
uses parallel training over the whole dataset (rayon par_iter), while the
streaming path is chunk-sequential. With "last writer wins" non-atomic
memory cell updates, parallel-write collisions resolve differently
depending on iteration order — same root cause as the Phase 2 smoke
showing small consistent metric shifts. So we compare with a numerical
tolerance, not strict equality.

What we CAN test bit-exact: streaming-vs-streaming with different chunk
sizes — the in-chunk parallel training is the same code path, but the
order of chunks doesn't matter for correctness (each chunk is trained
independently, accumulating into the same memory).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))


class _SkipException(Exception):
	pass


def _importorskip(name: str):
	try:
		__import__(name)
	except ImportError as e:
		raise _SkipException(f"{name} not installed: {e}")


def _make_dataset(n_train=400, n_eval=100, total_features=16, seed=7):
	"""Deterministic dataset where label = bit 0 (highly learnable)."""
	rng = np.random.default_rng(seed)
	train_bool = rng.integers(0, 2, size=(n_train, total_features), dtype=np.uint8).astype(bool)
	eval_bool = rng.integers(0, 2, size=(n_eval, total_features), dtype=np.uint8).astype(bool)
	train_labels = train_bool[:, 0].astype(np.int64)
	eval_labels = eval_bool[:, 0].astype(np.int64)
	return train_bool, train_labels, eval_bool, eval_labels


def _pack(bool_arr):
	return np.ascontiguousarray(np.packbits(bool_arr.astype(np.uint8), axis=1, bitorder="little"))


def _eval_via_in_memory(train_bool, train_labels, eval_bool, eval_labels, total_features):
	"""Use IDSCacheWrapper.new_from_numpy + evaluate_genomes_full_hybrid (current path)."""
	import ram_accelerator

	train_packed = _pack(train_bool).ravel()
	eval_packed = _pack(eval_bool).ravel()
	cache = ram_accelerator.IDSCacheWrapper.new_from_numpy(
		train_features=train_packed,
		train_labels=train_labels.tolist(),
		eval_features=eval_packed,
		eval_labels=eval_labels.tolist(),
		num_classes=2,
		total_features=total_features,
		num_parts=5,
		num_negatives=1,
		seed=42,
		single_cluster=True,
	)
	bits_flat = [4, 4, 4, 4]
	neurons_flat = [4]
	connections = [0, 1, 2, 3, 0, 5, 7, 9, 0, 2, 4, 8, 1, 3, 5, 11]
	return cache.evaluate_genomes_full_hybrid(
		genomes_bits_flat=bits_flat,
		genomes_neurons_flat=neurons_flat,
		genomes_connections_flat=connections,
		num_genomes=1,
		empty_value=0.5,
		neuron_sample_rate=1.0,
		rng_seed=42,
	)[0]


def _eval_via_streaming(train_bool, train_labels, eval_bool, eval_labels, total_features, n_chunks):
	"""Use IDSGenomeStreamer chunked path."""
	import ram_accelerator

	bits_flat = [4, 4, 4, 4]
	neurons_flat = [4]
	connections = [0, 1, 2, 3, 0, 5, 7, 9, 0, 2, 4, 8, 1, 3, 5, 11]

	streamer = ram_accelerator.IDSGenomeStreamerWrapper(
		bits_flat=bits_flat,
		neurons_flat=neurons_flat,
		connections=connections,
		num_classes=2,
		num_negatives=1,
		single_cluster=True,
		total_features=total_features,
		empty_value=0.5,
		neuron_sample_rate=1.0,
		rng_seed=42,
		normal_class=0,
	)

	# Train phase: split into n_chunks
	for chunk_bool, chunk_lbl in zip(
		np.array_split(train_bool, n_chunks, axis=0),
		np.array_split(train_labels, n_chunks),
	):
		chunk_packed = _pack(chunk_bool).ravel()
		streamer.train_chunk(chunk_packed, chunk_lbl.tolist(), total_features)

	streamer.seal_for_scoring()

	# Score phase
	for chunk_bool, chunk_lbl in zip(
		np.array_split(eval_bool, n_chunks, axis=0),
		np.array_split(eval_labels, n_chunks),
	):
		chunk_packed = _pack(chunk_bool).ravel()
		streamer.score_chunk(chunk_packed, chunk_lbl.tolist(), total_features)

	return streamer.finalize_metrics()


def test_streaming_single_chunk_close_to_in_memory():
	"""Streaming with one chunk should be very close to in-memory path on same data.

	Not bit-exact because the in-memory path uses different parallel ordering
	(full par_iter over all examples vs chunk-by-chunk). Tolerance reflects
	"last writer wins" non-determinism observed in the Phase 2 smoke.
	"""
	_importorskip("ram_accelerator")

	train_bool, train_labels, eval_bool, eval_labels = _make_dataset()

	in_mem = _eval_via_in_memory(train_bool, train_labels, eval_bool, eval_labels, 16)
	streaming = _eval_via_streaming(train_bool, train_labels, eval_bool, eval_labels, 16, n_chunks=1)

	# CE and accuracy should be in the same ballpark
	assert abs(in_mem[0] - streaming[0]) < 0.1, f"CE diverged: {in_mem[0]} vs {streaming[0]}"
	assert abs(in_mem[1] - streaming[1]) < 0.05, f"Acc diverged: {in_mem[1]} vs {streaming[1]}"

	# Both should achieve high accuracy on the label=bit_0 task (genome's first
	# neuron sees bit 0)
	assert streaming[1] > 0.9, f"streaming acc too low: {streaming[1]}"
	assert in_mem[1] > 0.9, f"in-memory acc too low: {in_mem[1]}"


def test_streaming_chunk_size_invariance():
	"""Streaming with different chunk sizes produces equivalent (close) metrics.

	The key correctness property: chunk size shouldn't significantly affect
	the final metrics. Tiny differences come from "last writer wins" race
	resolution within each chunk's parallel training pass.
	"""
	_importorskip("ram_accelerator")

	train_bool, train_labels, eval_bool, eval_labels = _make_dataset(n_train=400, n_eval=100)

	results = {}
	for n_chunks in (1, 4, 8, 20):
		results[n_chunks] = _eval_via_streaming(train_bool, train_labels, eval_bool, eval_labels, 16, n_chunks)

	# All chunk sizes should produce similar accuracy on this learnable task
	accs = [r[1] for r in results.values()]
	acc_spread = max(accs) - min(accs)
	assert acc_spread < 0.05, f"chunk_size accuracy spread too large: {accs}"

	# All should classify the label=bit_0 task well
	for n_chunks, r in results.items():
		assert r[1] > 0.85, f"n_chunks={n_chunks} acc too low: {r[1]}"


def test_streaming_lifecycle_train_after_seal_raises():
	"""train_chunk after seal_for_scoring panics in Rust (PyO3 surfaces as PanicException or BaseException)."""
	_importorskip("ram_accelerator")
	import ram_accelerator

	train_bool, train_labels, eval_bool, eval_labels = _make_dataset(n_train=50, n_eval=10)

	bits_flat = [4, 4, 4, 4]
	neurons_flat = [4]
	connections = [0, 1, 2, 3, 0, 5, 7, 9, 0, 2, 4, 8, 1, 3, 5, 11]

	s = ram_accelerator.IDSGenomeStreamerWrapper(
		bits_flat=bits_flat, neurons_flat=neurons_flat, connections=connections,
		num_classes=2, num_negatives=1, single_cluster=True, total_features=16,
	)
	s.train_chunk(_pack(train_bool).ravel(), train_labels.tolist(), 16)
	s.seal_for_scoring()

	# Now train_chunk should panic
	try:
		s.train_chunk(_pack(train_bool).ravel(), train_labels.tolist(), 16)
	except BaseException:
		return  # expected — Rust panic surfaces as some exception
	raise AssertionError("train_chunk after seal_for_scoring should panic/raise")


def test_streaming_lifecycle_score_before_seal_raises():
	"""score_chunk before seal_for_scoring panics."""
	_importorskip("ram_accelerator")
	import ram_accelerator

	train_bool, _, eval_bool, eval_labels = _make_dataset(n_train=50, n_eval=10)

	bits_flat = [4, 4, 4, 4]
	neurons_flat = [4]
	connections = [0, 1, 2, 3, 0, 5, 7, 9, 0, 2, 4, 8, 1, 3, 5, 11]

	s = ram_accelerator.IDSGenomeStreamerWrapper(
		bits_flat=bits_flat, neurons_flat=neurons_flat, connections=connections,
		num_classes=2, num_negatives=1, single_cluster=True, total_features=16,
	)
	# No train + no seal; score_chunk should panic
	try:
		s.score_chunk(_pack(eval_bool).ravel(), eval_labels.tolist(), 16)
	except BaseException:
		return  # expected
	raise AssertionError("score_chunk before seal_for_scoring should panic/raise")


def test_streaming_finalize_one_shot():
	"""finalize_metrics consumes the state; second call raises."""
	_importorskip("ram_accelerator")
	import ram_accelerator

	train_bool, train_labels, eval_bool, eval_labels = _make_dataset(n_train=50, n_eval=10)

	bits_flat = [4, 4, 4, 4]
	neurons_flat = [4]
	connections = [0, 1, 2, 3, 0, 5, 7, 9, 0, 2, 4, 8, 1, 3, 5, 11]

	s = ram_accelerator.IDSGenomeStreamerWrapper(
		bits_flat=bits_flat, neurons_flat=neurons_flat, connections=connections,
		num_classes=2, num_negatives=1, single_cluster=True, total_features=16,
	)
	s.train_chunk(_pack(train_bool).ravel(), train_labels.tolist(), 16)
	s.seal_for_scoring()
	s.score_chunk(_pack(eval_bool).ravel(), eval_labels.tolist(), 16)

	s.finalize_metrics()
	try:
		s.finalize_metrics()
	except RuntimeError:
		return
	raise AssertionError("second finalize_metrics() should raise")


def test_streaming_train_seen_and_eval_scored_counts():
	"""train_seen() and eval_scored() reflect accumulated row counts."""
	_importorskip("ram_accelerator")
	import ram_accelerator

	train_bool, train_labels, eval_bool, eval_labels = _make_dataset(n_train=200, n_eval=80)

	bits_flat = [4, 4, 4, 4]
	neurons_flat = [4]
	connections = [0, 1, 2, 3, 0, 5, 7, 9, 0, 2, 4, 8, 1, 3, 5, 11]

	s = ram_accelerator.IDSGenomeStreamerWrapper(
		bits_flat=bits_flat, neurons_flat=neurons_flat, connections=connections,
		num_classes=2, num_negatives=1, single_cluster=True, total_features=16,
	)
	assert s.train_seen() == 0
	assert s.eval_scored() == 0

	# Train in 4 chunks of 50
	for chunk_bool, chunk_lbl in zip(
		np.array_split(train_bool, 4, axis=0),
		np.array_split(train_labels, 4),
	):
		s.train_chunk(_pack(chunk_bool).ravel(), chunk_lbl.tolist(), 16)

	assert s.train_seen() == 200, f"expected 200, got {s.train_seen()}"
	assert s.eval_scored() == 0

	s.seal_for_scoring()

	# Score in 2 chunks of 40
	for chunk_bool, chunk_lbl in zip(
		np.array_split(eval_bool, 2, axis=0),
		np.array_split(eval_labels, 2),
	):
		s.score_chunk(_pack(chunk_bool).ravel(), chunk_lbl.tolist(), 16)

	assert s.eval_scored() == 80, f"expected 80, got {s.eval_scored()}"


if __name__ == "__main__":
	import traceback

	tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
	passed, failed, skipped = 0, 0, 0
	for t in tests:
		try:
			t()
			print(f"  ✓ {t.__name__}")
			passed += 1
		except _SkipException as e:
			print(f"  ⊘ {t.__name__}: {e}")
			skipped += 1
		except Exception as e:
			print(f"  ✗ {t.__name__}: {e}")
			traceback.print_exc()
			failed += 1
	print(f"\n{passed} passed, {failed} failed, {skipped} skipped of {len(tests)}")
	sys.exit(0 if failed == 0 else 1)
