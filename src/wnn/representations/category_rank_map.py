"""Category -> rank of its TRAIN attack rate (IDS-70, cut method INFO_GAIN_MC).

Turns a categorical feature into an ordered numeric one so a thermometer can cut it:
categories are ranked by their attack rate on the TRAIN rows (ties broken by name).
A category never seen in train (or NaN) maps just below the rank the global train
attack rate would take. Ported from logs/ids/encoding_B/cuts.py (target_rank_map).
"""

import numpy as np
import pandas as pd


class CategoryRankMap:

	def __init__(self, ranks: dict, unseen: float):
		self.ranks = ranks
		self.unseen = unseen

	@classmethod
	def fit(cls, series: pd.Series, y_binary: np.ndarray) -> "CategoryRankMap":
		"""Rank every TRAIN category by its attack rate (binary label)."""
		rates = pd.Series(np.asarray(y_binary)).groupby(np.asarray(series)).mean()
		order = sorted(rates.index, key=lambda c: (rates[c], str(c)))
		ranks = {c: float(i) for i, c in enumerate(order)}
		global_rank = float(np.searchsorted(np.sort(rates.values), float(np.mean(y_binary))))
		return cls(ranks, global_rank - 0.5)

	def transform(self, series: pd.Series) -> np.ndarray:
		"""float64 rank per row; unseen / NaN -> self.unseen."""
		mapped = pd.Series(np.asarray(series, dtype=object)).map(self.ranks)
		return mapped.astype(np.float64).fillna(self.unseen).to_numpy()
