"""Cascade routing proof for the hierarchical IDS arm.

The property under test is the one measured 8/8 across RF/XGB on three datasets:
a cascade's benign-FPR EQUALS its binary gate's FPR, because routing is by S0's
prediction and every benign row S0 admits is relabelled as some attack class.

Before 26/08/2026 the combined path returned S0's BINARY metrics under the
combined keys, so the hierarchical arm reported ~90% "macro-F1" next to the flat
multi arm's ~40% — a 2-class number beside a 10-class one.
"""
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, "src/wnn")

from wnn.ram.experiments.flow import Flow


def _flow(s0_preds, s1_preds, y_true_multi, y_true_binary):
	"""A Flow shell carrying only what the routing path reads."""
	return SimpleNamespace(
		_s0_full_evaluator=SimpleNamespace(
			predict_classes=lambda g: list(s0_preds),
			y_test=list(y_true_binary),
			y_test_multi=list(y_true_multi),
		),
		_s1_route_evaluator=SimpleNamespace(predict_classes=lambda g: list(s1_preds)),
	)


def test_routing_remaps_and_gates():
	# 6 rows: 3 benign (0), 3 attacks of classes 1,2,3
	y_multi = [0, 0, 0, 1, 2, 3]
	y_bin = [0, 0, 0, 1, 1, 1]
	# S0 admits one benign (row 1) and misses one attack (row 5)
	s0 = [0, 1, 0, 1, 1, 0]
	# S1 would call every row class index 4 (-> 10-class label 5)
	s1 = [4, 4, 4, 0, 1, 4]

	f = _flow(s0, s1, y_multi, y_bin)
	combined, routed, s0_out = Flow._hierarchical_routed_predictions(f, None, None)

	assert list(combined) == [0, 5, 0, 1, 2, 0], list(combined)
	assert list(routed) == [False, True, False, True, True, False]

	y = np.asarray(y_multi)
	benign = y == 0
	cascade_benign_fpr = float((combined[benign] != 0).mean())
	gate_fpr = float((np.asarray(s0)[benign] == 1).mean())
	assert cascade_benign_fpr == gate_fpr == 1 / 3, (cascade_benign_fpr, gate_fpr)


def test_gate_fpr_identity_holds_on_random_data():
	"""The identity must hold for ANY gate and ANY S1, not just a chosen example."""
	rng = np.random.default_rng(20260826)
	for _ in range(200):
		n = int(rng.integers(20, 200))
		y = rng.integers(0, 10, size=n)
		s0 = rng.integers(0, 2, size=n)
		s1 = rng.integers(0, 9, size=n)
		f = _flow(s0, s1, y.tolist(), (y > 0).astype(int).tolist())
		combined, _, _ = Flow._hierarchical_routed_predictions(f, None, None)
		benign = y == 0
		if not benign.any():
			continue
		assert float((combined[benign] != 0).mean()) == float(s0[benign].mean())


def test_row_mismatch_raises_rather_than_joining_unrelated_rows():
	f = _flow([0, 1, 0], [4, 4], [0, 1, 2], [0, 1, 1])
	try:
		Flow._hierarchical_routed_predictions(f, None, None)
	except ValueError as e:
		assert "row mismatch" in str(e)
	else:
		raise AssertionError("a row-count mismatch must raise, not silently truncate")


if __name__ == "__main__":
	test_routing_remaps_and_gates()
	test_gate_fpr_identity_holds_on_random_data()
	test_row_mismatch_raises_rather_than_joining_unrelated_rows()
	print("all cascade routing tests PASSED")
