"""Best-first information-gain cut points (IDS-70 / IDS-72). Pure functions over (sorted unique
value, class count) tables.

INFO_GAIN_MC  -- budget_cuts: one global bit budget. Ported verbatim from the IDS-63 offline
                 pre-check (logs/ids/encoding_B/cuts.py), whose cuts_E.json this reproduces.
INFO_GAIN_MDL -- mdl_cuts: no budget; a split is placed only if it passes the Fayyad-Irani (1993)
                 MDLP test on its own segment, so the data decides each feature's bit count.
                 Ported from the IDS-72 offline pre-check (logs/ids/encoding_mdl/mdl.py).
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
	return _best_first(tables, budget, accept=None)


def mdl_cuts(tables: dict) -> dict[str, np.ndarray]:
	"""Best-first information-gain cuts with no budget: a split is placed only if it passes
	mdlp_accept on its segment; a rejected segment is closed. Order-independent, so it equals
	recursive Fayyad-Irani MDLP per feature. tables: {name: (u, counts)}."""
	return _best_first(tables, None, accept=mdlp_accept)


def mdlp_accept(counts: np.ndarray, lo: int, k: int, hi: int, gain: float) -> bool:
	"""Fayyad-Irani MDLP: accept splitting rows [lo,hi) at k iff
	gain/N > (log2(N-1) + Delta)/N, Delta = log2(3^c - 2) - [c Ent(S) - c1 Ent(S1) - c2 Ent(S2)],
	c = classes present. `gain` is best_split's ROW-WEIGHTED gain (N x per-row gain), so gain/N is
	the per-row information gain the criterion is stated in."""
	s, s1, s2 = counts[lo:hi].sum(0), counts[lo:k].sum(0), counts[k:hi].sum(0)
	n = int(s.sum())
	if n < 2:
		return False
	c, c1, c2 = (int((x > 0).sum()) for x in (s, s1, s2))
	h, h1, h2 = (entropy_rows(x[None])[0] for x in (s, s1, s2))
	delta = np.log2(3.0 ** c - 2) - (c * h - c1 * h1 - c2 * h2)
	return gain / n > (np.log2(n - 1) + delta) / n


def _best_first(tables: dict, budget: int | None, accept) -> dict[str, np.ndarray]:
	"""Pop the highest-gain open segment; place its cut (if `accept` allows) and re-open both
	halves, until the budget (None = unbounded) is spent or no segment has positive gain."""
	heap, cuts = [], {name: [] for name in tables}
	for name, (u, c) in tables.items():
		_push(heap, name, c, 0, len(u))
	placed = 0
	while heap and (budget is None or placed < budget):
		neg_gain, name, lo, k, hi = heapq.heappop(heap)
		if -neg_gain <= 0:
			break
		u, c = tables[name]
		if accept is not None and not accept(c, lo, k, hi, -neg_gain):
			continue
		cuts[name].append(cut_value(u, k))
		placed += 1
		_push(heap, name, c, lo, k)
		_push(heap, name, c, k, hi)
	return {name: np.sort(np.array(v, dtype=np.float64)) for name, v in cuts.items()}


def _push(heap: list, name: str, counts: np.ndarray, lo: int, hi: int) -> None:
	gain, k = best_split(counts, lo, hi)
	if k > 0 and gain > 0:
		heapq.heappush(heap, (-gain, name, lo, k, hi))
