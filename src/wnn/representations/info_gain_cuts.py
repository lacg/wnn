"""Best-first information-gain cut points under one global bit budget (IDS-70, cut method
INFO_GAIN_MC). Ported verbatim from the IDS-63 offline pre-check (logs/ids/encoding_B/cuts.py),
whose cuts_E.json this reproduces. Pure functions over (sorted unique value, class count) tables.
"""

import heapq

import numpy as np


def class_table(values: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
	"""(sorted unique values, per-unique class counts [U, C])."""
	u, inv = np.unique(values, return_inverse=True)
	counts = np.zeros((len(u), n_classes), dtype=np.int64)
	np.add.at(counts, (inv, y), 1)
	return u, counts


def entropy_rows(c: np.ndarray) -> np.ndarray:
	"""Shannon entropy (bits) of each row of a count matrix."""
	n = c.sum(axis=-1, keepdims=True).astype(np.float64)
	p = np.where(n > 0, c / np.maximum(n, 1), 0.0)
	return -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0.0)).sum(axis=-1)


def best_split(counts: np.ndarray, lo: int, hi: int) -> tuple[float, int]:
	"""Best boundary k in (lo, hi) splitting rows [lo,k) | [k,hi): (weighted gain, k) or (0, -1)."""
	if hi - lo < 2:
		return 0.0, -1
	seg = counts[lo:hi]
	left = np.cumsum(seg, axis=0)[:-1]
	total = seg.sum(axis=0)
	right = total - left
	child = left.sum(1) * entropy_rows(left) + right.sum(1) * entropy_rows(right)
	parent = total.sum() * entropy_rows(total[None])[0]
	i = int(np.argmin(child))
	return float(parent - child[i]), lo + i + 1


def cut_value(u: np.ndarray, k: int) -> float:
	"""Midpoint between the unique values either side of boundary k (x >= cut <=> right side)."""
	return 0.5 * (u[k - 1] + u[k])


def budget_cuts(tables: dict, budget: int) -> dict[str, np.ndarray]:
	"""Best-first information-gain cuts across features under one global budget.
	tables: {name: (u, counts)}. Returns {name: sorted cut array}."""
	heap, cuts = [], {name: [] for name in tables}
	for name, (u, c) in tables.items():
		_push(heap, name, c, 0, len(u))
	placed = 0
	while heap and placed < budget:
		neg_gain, name, lo, k, hi = heapq.heappop(heap)
		if -neg_gain <= 0:
			break
		u, c = tables[name]
		cuts[name].append(cut_value(u, k))
		placed += 1
		_push(heap, name, c, lo, k)
		_push(heap, name, c, k, hi)
	return {name: np.sort(np.array(v, dtype=np.float64)) for name, v in cuts.items()}


def _push(heap: list, name: str, counts: np.ndarray, lo: int, hi: int) -> None:
	gain, k = best_split(counts, lo, hi)
	if k > 0 and gain > 0:
		heapq.heappush(heap, (-gain, name, lo, k, hi))
