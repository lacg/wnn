"""CTRL-58: the staged-refinement fitness calculator (see refine_config.py for the why).

ZRank over steady / reward / near-fail tail / pos (CTRL-58 weights 0.50 / 0.25 / 0.15 /
0.10), behind a TIGHTENED four-way gate: stable >= S, err <= E, steady <= D, pos <= P.
stable is in the gate only — never a weighted column.

THE GATE IS THE WHEEL'S, NATIVE (ABI 59): gated_fitness_combine(gate_steady=,
gate_steady_max=, gate_pos=, gate_pos_max=) — controller crate gate4.rs on top of
ram_core's combine — computes feasible = stable >= S AND err <= E AND steady <= D AND
pos <= P and orders infeasibles by the summed normalised violation (Deb's rules).
Python only marshals the four gate inputs. (Before ABI 59 the steady / pos excess was
folded into the err slot as err' = E·(1+V); the native gate replaces that composite.)
"""

import math

from wnn.ram.fitness import FitnessCalculatorControllerHarmonic

from .refine_config import RefineConfig


class FitnessCalculatorControllerRefine(FitnessCalculatorControllerHarmonic):
	"""The CTRL-58 refinement fitness: ZRank + tightened four-way gate (+ tail column)."""

	def __init__(self, cfg: RefineConfig, zrank_clamp: float, zrank_mad_floor: bool,
	             zrank_pos_floor: float, pos_metric: str, with_tail: bool):
		super().__init__(
			weight_err_sq=cfg.w_reward, weight_stable=0.0, weight_steady=cfg.w_steady,
			weight_pos=cfg.w_pos, aggregation="zscore", zrank_clamp=zrank_clamp,
			zrank_mad_floor=zrank_mad_floor, zrank_pos_floor=zrank_pos_floor,
			pos_metric=pos_metric, gate_stable_min=cfg.gate_stable, gate_err_max=cfg.gate_err)
		self.refine = cfg
		# Stage-select passes with_tail=False: there the tail enters as rank(p99) via
		# the existing --select-tail rank sum, never twice.
		self.weight_tail = cfg.w_tail if with_tail else 0.0

	def _columns(self, metrics_list) -> tuple:
		cols = super()._columns(metrics_list)
		if self.weight_tail > 0.0:
			self._add_column(cols, self._tail_column(metrics_list), self.weight_tail,
			                 False, "weight_tail")
		return cols

	def _tail_column(self, metrics_list) -> list:
		"""ControllerMetrics.episode_err_p99_deg, refused (never dropped) when absent or
		non-finite: the wheel refuses non-finite values, and a silently dropped weight
		is the failure class this calculator must not re-create."""
		vals = [getattr(m, "episode_err_p99_deg", None) for m in metrics_list]
		if any(v is None or not math.isfinite(float(v)) for v in vals):
			raise ValueError(
				"refinement tail weight > 0 but a candidate has no finite "
				"episode_err_p99_deg — no scorer reports the train-pool tail yet "
				"(set --refine-w-tail 0).")
		return [float(v) for v in vals]

	def _combine(self, metrics_list, flat, weights, higher, floors) -> list:
		"""The base combine with the NATIVE four-input gate (stable, err, steady, pos)."""
		n = len(metrics_list)
		gate_st, gate_er = self._gate_inputs(metrics_list)
		c = self.refine
		kw = dict(gate_steady=self._required(metrics_list, "mean_steady_error_deg"),
		          gate_steady_max=c.gate_steady, gate_pos=self._gate_pos(metrics_list),
		          gate_pos_max=c.gate_pos)
		if any(f > 0.0 for f in floors):
			kw["scale_floors"] = floors
		from wnn.control._accel import gated_fitness_combine
		return list(gated_fitness_combine(
			flat, n, weights, higher, self.aggregation, self.zrank_clamp,
			gate_st, gate_er, self.gate_stable_min, self.gate_err_max, **kw))

	def _gate_pos(self, metrics_list) -> list:
		"""The gate reads the column the fitness ranks (--fit-pos-metric). Under `hold` a
		candidate with no hold value takes the column's worst-plus-penalty (as it does
		in the ranking), so it violates the bound instead of reaching the wheel as NaN."""
		if self.pos_metric == "hold":
			vals = self._hold_column(metrics_list)
			if vals is None:
				raise ValueError("refinement gate armed on pos_hold but no metric carries it.")
			return vals
		return self._required(metrics_list, "mean_position_error_m")

	@staticmethod
	def _required(metrics_list, attr: str) -> list:
		vals = [getattr(m, attr, None) for m in metrics_list]
		if any(v is None for v in vals):
			raise ValueError(f"refinement gate armed but Metrics.{attr} is None — the scorer "
			                 "does not report it; an armed gate must never silently not gate.")
		return [float(v) for v in vals]

	@property
	def name(self) -> str:
		c = self.refine
		gate = (f"st>={c.gate_stable:g},err<={c.gate_err:g}°,steady<={c.gate_steady:g}°,"
		        f"pos<={c.gate_pos:g}m")
		tail = f", tail_p99={self.weight_tail:g}" if self.weight_tail > 0.0 else ""
		return f"Refine[gate=({gate})]{super().name}{tail}"


def build_refine_calculator(cfg: RefineConfig, args, with_tail: bool) -> FitnessCalculatorControllerRefine:
	"""The refinement calculator under the run's own ZRank knobs (clamp, MAD floor, pos
	floor, pos column) — one fitness family with the base calculator it replaces."""
	from .pos_floor import fit_pos_metric_from_args, zrank_pos_floor_from_args
	return FitnessCalculatorControllerRefine(
		cfg, zrank_clamp=float(getattr(args, "zrank_clamp", 3.0)),
		zrank_mad_floor=bool(getattr(args, "zrank_mad_floor", False)),
		zrank_pos_floor=zrank_pos_floor_from_args(args),
		pos_metric=fit_pos_metric_from_args(args), with_tail=with_tail)
