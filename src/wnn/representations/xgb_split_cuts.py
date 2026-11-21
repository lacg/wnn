"""Every split threshold of a TRAIN-fit XGBoost model, per feature (IDS-70, cut method XGB_SPLITS).

Ported from the IDS-63 pre-check (logs/ids/encoding_B/enc_b.py xgb_cuts + cuts.py xgb_split_cuts),
whose cuts_B.json this reproduces. Hyper-parameters are the measured baseline's (seed_baselines.py:
100 trees, depth 6, lr 0.1). XGBoost's default model has no row/column subsampling, so the seed does
not move the splits; the thread count is pinned so the histogram reductions are fixed too.
XGBoost sends x < t left, so the thermometer bit x >= t is exactly the split's right branch.
"""

import numpy as np

XGB_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss", verbosity=0)
XGB_THREADS = 6


def xgb_split_cuts(X: np.ndarray, y_binary: np.ndarray, seed: int) -> list[np.ndarray]:
	"""Sorted distinct split thresholds per column of X (non-finite values = missing)."""
	from xgboost import XGBClassifier
	X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
	model = XGBClassifier(n_jobs=XGB_THREADS, random_state=seed, **XGB_PARAMS)
	model.fit(X, np.asarray(y_binary))
	nodes = model.get_booster().trees_to_dataframe()
	nodes = nodes[nodes.Feature != "Leaf"]
	return [np.unique(nodes[nodes.Feature == f"f{i}"].Split.values.astype(np.float64)) for i in range(X.shape[1])]
