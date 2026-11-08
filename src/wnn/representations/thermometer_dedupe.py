"""
Train-informative thermometer bits (IDS-52, 08/11/2026; flow param
ids_encoding_dedupe="on").

The distributive encoder places n_bits thresholds at evenly spaced train
quantiles and "dedupes" equal quantiles by +1e-10. On integer / heavy-tailed
data that leaves zero-width bins: their bit columns are exact copies of a
neighbour (or constant), so a 96-bit feature can carry a handful of distinct
bits (audit: 48-52% of unswr/cicids/ciciot input bits were constant or
duplicate). With dedupe ON the encoder keeps only bits that are informative ON
THE TRAIN PARTITION (Protocol v2 — val/test never shape the encoding):

  - a threshold whose train bit column is constant (all 0 / all 1) is dropped;
  - of thresholds whose train columns are identical, one is kept;
  - a threshold whose train column equals a bit already kept for an EARLIER
    feature is dropped (cross-feature: e.g. CICIDS `Avg Bwd Segment Size`
    differs from `Bwd Packet Length Mean` only by 1e-6 rounding in 48 of 2.26M
    train rows, yet every one of its thermometer bits is a copy);
  - a feature left with no informative value bit (and no informative
    is_invalid flag) is dropped; the flag bit is kept only for features that
    have invalid rows in train.

Thermometer columns are NESTED (rows >= t2 are a subset of rows >= t1 for
t1 < t2), so two columns are identical iff they have the same number of ones —
the check is an exact count, no bit matrix is materialised.
"""

import hashlib

import numpy as np


def informative_thresholds(raw: np.ndarray, thresholds: np.ndarray, flagged: bool) -> np.ndarray:
	"""Thresholds whose TRAIN bit column (exactly as `transform` computes it:
	invalid -> 0.0, or cleared when the is_invalid flag is used) is
	non-constant and distinct from every kept column."""
	invalid = ~np.isfinite(raw)
	encodable = raw[~invalid] if flagged else np.where(invalid, 0.0, raw)
	ones = len(encodable) - np.searchsorted(np.sort(encodable), thresholds, side="left")
	keep, seen = [], set()
	for t, k in zip(thresholds, ones):
		if 0 < k < len(raw) and k not in seen:
			seen.add(int(k))
			keep.append(t)
	return np.asarray(keep, dtype=np.float64)


def has_invalid(raw: np.ndarray) -> bool:
	"""True iff the train column holds NaN / +-Inf (the flag bit varies)."""
	return bool((~np.isfinite(raw)).any())


def is_constant(series) -> bool:
	"""A binary / categorical train column with at most one distinct value
	(NaN counted as a value)."""
	return series.nunique(dropna=False) <= 1


def column_digests(bits: np.ndarray) -> list:
	"""128-bit digest of every column of a (rows x k) bool matrix."""
	packed = np.packbits(bits, axis=0)
	return [hashlib.blake2b(packed[:, j].tobytes(), digest_size=16).digest() for j in range(bits.shape[1])]
