"""SupervisedCutFitter: label-aware thermometer cut points, fitted on the TRAIN partition only.

ThermometerEncoder.fit calls this with the train DataFrame and the train labels (it never sees
val/test rows). Methods (cut_method.py): INFO_GAIN_MC = IDS-63 pre-check variant E, XGB_SPLITS =
variant B. Binary features are never touched; with INFO_GAIN_MC the categoricals become a
thermometer over their train attack-rate rank and share the numeric features' bit budget.
"""

import numpy as np
import pandas as pd

from wnn.representations.category_rank_map import CategoryRankMap
from wnn.representations.cut_method import CutMethod
from wnn.representations.info_gain_cuts import budget_cuts, class_table
from wnn.representations.supervised_cuts import SupervisedCuts
from wnn.representations.xgb_split_cuts import xgb_split_cuts


class SupervisedCutFitter:

	def __init__(self, method: CutMethod, seed: int):
		if method == CutMethod.QUANTILE:
			raise ValueError("SupervisedCutFitter: QUANTILE is the unsupervised encoder default, not a fitter")
		self.method = method
		self.seed = seed

	def fit(self, df: pd.DataFrame, feature_types: dict[str, str], quantile_bits: dict[str, int],
			y_binary: np.ndarray, y_multi: np.ndarray) -> SupervisedCuts:
		"""df/labels = the TRAIN rows. quantile_bits = each feature's width under QUANTILE (the budget)."""
		numeric = [f for f, t in feature_types.items() if t == "numeric"]
		categorical = [f for f, t in feature_types.items() if t == "categorical"]
		if self.method == CutMethod.XGB_SPLITS:
			return self._xgb(df, numeric, y_binary)
		budget = sum(quantile_bits[f] for f in numeric + categorical)
		return self._info_gain(df, numeric, categorical, budget, y_binary, y_multi)

	def _xgb(self, df: pd.DataFrame, numeric: list[str], y_binary: np.ndarray) -> SupervisedCuts:
		cuts = xgb_split_cuts(df[numeric].values.astype(np.float64), y_binary, self.seed)
		return SupervisedCuts(thresholds=dict(zip(numeric, cuts)))

	def _info_gain(self, df: pd.DataFrame, numeric: list[str], categorical: list[str], budget: int,
			y_binary: np.ndarray, y_multi: np.ndarray) -> SupervisedCuts:
		rank_maps = {f: CategoryRankMap.fit(df[f], y_binary) for f in categorical}
		columns = {f: df[f].values.astype(np.float64) for f in numeric}
		columns.update({f: rank_maps[f].transform(df[f]) for f in categorical})
		y_multi = np.asarray(y_multi, dtype=np.int64)
		n_classes = int(y_multi.max()) + 1
		tables = {f: self._table(v, y_multi, n_classes) for f, v in columns.items()}
		return SupervisedCuts(thresholds=budget_cuts(tables, budget), rank_maps=rank_maps)

	@staticmethod
	def _table(values: np.ndarray, y: np.ndarray, n_classes: int):
		"""Class table over the finite values (NaN/±Inf rows are the is_invalid flag's business)."""
		finite = np.isfinite(values)
		return class_table(values[finite], y[finite], n_classes)
