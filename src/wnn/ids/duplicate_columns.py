"""
Exact-duplicate feature columns (IDS-52, 08/11/2026; ids_encoding_dedupe="on").

Some IDS feature pairs are the same column under two names — CICIDS2017
`Bwd Packet Length Mean` == `Avg Bwd Segment Size`, CIC-IoT-2023 `Srate` ==
`Rate`. Encoding both spends twice the input bits on one signal. Detection runs
on the TRAIN partition ONLY (Protocol v2: val/test never shape the encoding);
the first column of each identical group (in selection order) is kept.
"""

import hashlib

import numpy as np
import pandas as pd


def distinct_columns(df_train: pd.DataFrame, columns: list) -> tuple:
	"""(kept columns in order, {dropped: the kept column it duplicates})."""
	kept, dropped, by_signature = [], {}, {}
	for col in columns:
		group = by_signature.setdefault(_signature(df_train[col]), [])
		twin = _find_twin(df_train, col, group)
		if twin is None:
			kept.append(col)
			group.append(col)
		else:
			dropped[col] = twin
	if dropped:
		print(f"  Encoder dedupe: dropped {len(dropped)} exact-duplicate columns (train) {dropped}")
	return kept, dropped


def _find_twin(df: pd.DataFrame, col: str, candidates: list):
	return next((c for c in candidates if _equal(df[c], df[col])), None)


def _numeric(series: pd.Series) -> bool:
	return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _as_float(series: pd.Series) -> np.ndarray:
	# + 0.0 folds -0.0 into 0.0 so equal values hash equal.
	return series.to_numpy(dtype=np.float64, na_value=np.nan) + 0.0


def _signature(series: pd.Series) -> str:
	"""Cheap hash that equal columns share (int vs float dtype included)."""
	if _numeric(series):
		data = np.nan_to_num(_as_float(series), nan=1.5e308, posinf=1.6e308, neginf=-1.6e308).tobytes()
	else:
		data = pd.util.hash_pandas_object(series.astype(str), index=False).to_numpy().tobytes()
	return hashlib.blake2b(data, digest_size=16).hexdigest()


def _equal(a: pd.Series, b: pd.Series) -> bool:
	if _numeric(a) and _numeric(b):
		return bool(np.array_equal(_as_float(a), _as_float(b), equal_nan=True))
	return a.astype(str).reset_index(drop=True).equals(b.astype(str).reset_index(drop=True))
