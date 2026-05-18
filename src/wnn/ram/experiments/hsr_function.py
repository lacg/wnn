"""HSR (WNN_HYBRID_SPEED_RATIO) prediction function.

Predicts the optimal hybrid-speed-ratio for a given flow based on workload
proxy: (neurons, bits, thermo_width, n_train_samples). Initial fit from
CIC-IoT-2023 OI-v2 cohort data (18/05/2026, n=15 flows).

The cohort data shows shallow optima within {1, 5, 7, 8, 10} (~5% spread
on most shapes) and catastrophic penalties for HSR=2/3 (40-70% slower).
Function buckets workload into three regimes:

  Tiny workload    (<10M bit-ops/fold):    HSR=1   ← pure-path
  Mid-range        (10M to 5B):            HSR=8   ← shallow-optimum default
  Extreme workload (≥5B bit-ops):          HSR=10  ← clipped extrapolation

Cross-dataset Stage 1 + Stage 2 sweeps will refine the bucket thresholds.

See docs/paper_updates_pending.md for the methodology rationale.
"""

# Dataset → train-set size lookup (used to estimate per-fold samples when
# the worker hasn't loaded the dataset yet). Keys match the `ids_dataset`
# config param. Sizes from paper Table 1 / dataset loaders.
_DATASET_TRAIN_SIZE = {
	"ciciot2023_neto_subsample": 914_000,    # 1.14M total
	"ciciot2023_neto_full":      37_300_000, # 46M total
	"cicids2017":                2_300_000,
	"unsw-nb15":                 1_270_000,  # random split (deduplicated)
}

# UNSW temporal split (much smaller — separate key)
_DATASET_TRAIN_SIZE_BY_SPLIT = {
	("unsw-nb15", "temporal"): 175_000,
}


def estimate_samples_per_fold(params: dict) -> int:
	"""Estimate examples-per-fold from flow params (dataset name, split, K)."""
	dataset = params.get("ids_dataset", "ciciot2023_neto_subsample")
	split = params.get("ids_split", "random")
	k_folds = params.get("ids_k_folds", 5)

	# Try (dataset, split) first, then fall back to dataset alone
	train_size = _DATASET_TRAIN_SIZE_BY_SPLIT.get((dataset, split))
	if train_size is None:
		train_size = _DATASET_TRAIN_SIZE.get(dataset)
	if train_size is None:
		# Unknown dataset — use a conservative middle-of-the-road estimate
		train_size = 1_000_000

	return max(1, train_size // k_folds)


def compute_workload(neurons: int, bits: int, thermo_width: int, n_train_samples: int) -> float:
	"""Workload proxy in bit-operations per fold.

	Two terms: input bandwidth (samples × thermo × features) + neuron compute
	(samples × neurons × bits). Assumes 20-feature top-20 selection.
	"""
	bandwidth = n_train_samples * thermo_width * 20
	compute = n_train_samples * neurons * bits
	return bandwidth + compute


def predict_hsr(neurons: int, bits: int, thermo_width: int, n_train_samples: int) -> int:
	"""Predict optimal WNN_HYBRID_SPEED_RATIO for the given flow.

	Inputs are flow-level (use max_neurons, max_bits — the GA's architectural
	ceiling). The Rust dispatcher adapts per-batch via measured speed_ratio at
	runtime; this function just sets the gate threshold for the whole flow.

	Returns: int in {1, 8, 10}. (HSR=2/3 excluded as dominated; HSR=5/7 wins
	only by noise margins over HSR=8 in current data.)
	"""
	workload = compute_workload(neurons, bits, thermo_width, n_train_samples)

	if workload < 10_000_000:    # ~10M bit-ops/fold — dispatch overhead dominates
		return 1
	if workload >= 5_000_000_000:  # ~5B bit-ops/fold — extreme, max hybrid
		return 10
	return 8                       # mid-range — shallow-optimum default


def predict_hsr_from_params(params: dict) -> int:
	"""Convenience wrapper: extract architectural params + estimate samples,
	then call predict_hsr(). Logs the chosen HSR for traceability."""
	neurons = params.get("max_neurons", 100)
	bits = params.get("max_bits", 32)
	thermo_width = params.get("ids_n_bits", 96)
	n_samples = estimate_samples_per_fold(params)
	return predict_hsr(neurons, bits, thermo_width, n_samples)
