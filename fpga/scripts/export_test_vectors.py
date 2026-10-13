#!/usr/bin/env python3
"""HW-vs-Rust test vectors: held-out inputs + the RUST score each must produce.

Expected values come from the production scorer (the exported model's
eval_scores, already proven against the DB row by the FidelityGate): the RTL
adder sums count_u x quarters(cell_u) over units, which equals
score x (4 x logical_neurons) exactly for the deterministic modes. No Python
re-implementation of the model. Written by export_genome(.py|_dense.py).
"""

import json
from pathlib import Path

import numpy as np

VECTOR_SEED = 42


def select_vectors(labels: list[int], num_vectors: int) -> np.ndarray:
	"""Stratified, seeded: half normal, half attack."""
	y = np.asarray(labels)
	rng = np.random.RandomState(VECTOR_SEED)
	half = num_vectors // 2
	picks = [rng.choice(np.where(y == c)[0], min(half, int((y == c).sum())), replace=False) for c in (0, 1)]
	selected = np.concatenate(picks)
	rng.shuffle(selected)
	return selected[:num_vectors]


def rows_as_bits(matrix, rows: np.ndarray, total_bits: int) -> np.ndarray:
	if hasattr(matrix, "as_packed_uint8"):
		packed = matrix.as_packed_uint8()[rows]
		return np.unpackbits(packed, axis=1, bitorder="little")[:, :total_bits].astype(np.uint8)
	return np.asarray(matrix[rows], dtype=np.uint8)


def hex_msb_first(bits: np.ndarray) -> str:
	padded = np.zeros(((len(bits) + 3) // 4) * 4, dtype=np.uint8)
	padded[:len(bits)] = bits
	return "".join(f"{int(''.join(map(str, padded[j:j + 4])), 2):x}" for j in range(0, len(padded), 4))


def write_test_vectors(output_dir: Path, trained, meta: dict, num_vectors: int) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	rows = select_vectors(trained.labels, min(num_vectors, len(trained.labels)))
	scores = np.asarray(trained.layer.eval_scores, dtype=np.float64)[rows]
	expected = np.rint(scores * meta["score_denominator_quarters"]).astype(np.int64)
	bits = rows_as_bits(trained.dataset.X_test, rows, meta["input_bits"])
	(output_dir / "test_inputs.mem").write_text("".join(hex_msb_first(b) + "\n" for b in bits))
	(output_dir / "test_expected.mem").write_text("".join(f"{int(v):x}\n" for v in expected))
	(output_dir / "test_labels.mem").write_text("".join(f"{trained.labels[r]}\n" for r in rows))
	(output_dir / "tb_metadata.json").write_text(json.dumps({
		"num_vectors": int(len(rows)), "rows": rows.tolist(), "rust_scores": scores.tolist(),
		"expected_sum_quarters": expected.tolist(), "input_bits": meta["input_bits"],
		"logical_neurons": meta["logical_neurons"], "physical_neurons": meta["physical_neurons"],
		"flow_id": meta["flow_id"],
	}, indent=2))
	print(f"  Test vectors: {len(rows)} -> {output_dir} (expected = Rust score x {meta['score_denominator_quarters']})")
