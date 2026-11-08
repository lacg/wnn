"""
ONE precedence rule for an experiment's run settings (IDS-51, 08/11/2026).

An experiment row carries creation-time SNAPSHOTS of flow params
(fitness_calculator, fitness_weight_ce / _acc, population_size, max_iterations).
worker.py used to read them as `exp_data.get(k) or params...`, so:
  - the stale row snapshot BEAT the live flow param (PUB50-ciciot, flows 711-846,
    ran ce/acc = 0.2/0.1 under params 0.1/0.2; also flows 1079-1089, 1123-1150,
    1156), while population_size preferred params — two rules in one function;
  - `or` treated an explicit 0.0 weight as unset.

The rule now, for every key:
  1. an explicit PER-EXPERIMENT override (experiments.params_json — e.g.
     `generations` / `iterations`, the dashboard's own source for the row's
     max_iterations) is the experiment's intent, not a snapshot;
  2. else the LIVE flow param, whenever it is set (`is not None`, so 0.0 counts);
  3. else the row snapshot;
  4. else the default.
A row snapshot that DISAGREES with the value that runs is logged loudly at
ingestion — the run never silently differs from what its row records.

Verified behaviour-neutral (08/11/2026, read-only): the 9,682 experiment rows of
flows > 1156 have zero row/params disagreements on every key resolved here.
"""

from typing import Callable, Optional


class ExperimentSettingResolver:
	"""Resolves one experiment's settings from (row, live flow params)."""

	def __init__(self, params: dict, exp_data: dict, log: Callable[[str], None]):
		self._params = params
		self._row = exp_data
		self._overrides = exp_data.get("params") if isinstance(exp_data.get("params"), dict) else {}
		self._log = log
		self._name = exp_data.get("name", "Unnamed")

	def resolve(self, row_key: str, flow_keys: tuple = (), default=None):
		"""Per-experiment override > live flow param (first of `flow_keys` that is
		set, default `(row_key,)`) > row snapshot > default; warns when the row
		snapshot disagrees with the value that runs."""
		row = self._row_snapshot(row_key)
		value = self._first_set((row_key,), self._overrides)
		if value is None:
			value = self._first_set(flow_keys or (row_key,))
		if value is None:
			value = row if row is not None else default
		self._warn_if_disagrees(row_key, row, value)
		return value

	def _row_snapshot(self, key: str):
		"""The row's own column — not a params_json key merged into exp_data."""
		if key in self._overrides and self._row.get(key) == self._overrides.get(key):
			return None
		return self._row.get(key)

	def max_iterations(self, flow_key: str, default: int) -> int:
		"""Per-experiment generations/iterations > live `flow_key` > row > default."""
		explicit = self._first_set(("generations", "iterations"), self._overrides)
		if explicit is not None:
			self._warn_if_disagrees("max_iterations", self._row_snapshot("max_iterations"), explicit)
			return explicit
		return self.resolve("max_iterations", (flow_key,), default)

	def _first_set(self, keys: tuple, source: Optional[dict] = None):
		source = self._params if source is None else source
		return next((source[k] for k in keys if source.get(k) is not None), None)

	def _warn_if_disagrees(self, key: str, row, value) -> None:
		if row is None or value is None or _same(row, value):
			return
		self._log(
			f"[PARAMS] ⚠️  experiment '{self._name}': row {key}={row!r} DISAGREES with the "
			f"live flow param — running {key}={value!r} (live params win, IDS-51)"
		)


def _same(a, b) -> bool:
	"""Numeric-tolerant equality (row floats come back from SQLite)."""
	if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
		return abs(float(a) - float(b)) <= 1e-12
	return str(a) == str(b)
