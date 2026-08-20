"""
ControllerGridSearch — the controller's adapter onto the shared GenericGridSearch.

Replaces the hand-rolled `stage0_grid` loop. Two correctness fixes ride here:
  1. **Top-K seeding** (not single-winner): the base returns a full top-K seed
     population that seeds Stage 1, instead of discarding all but the best shape.
  2. **Rank by the fitness calculator, not raw CE**: `_fitness` uses the SAME
     `FitnessCalculatorType.CONTROLLER`/`CONTROLLER_HARMONIC` the GA stages use
     (via `default_controller_ga_config(...).create_fitness_calculator()`), so the
     grid optimizes the same objective as the stages it seeds. With default
     weights (err_sq=1) this reproduces the old CE ordering exactly.

It also evaluates every grid point through ONE shared mixed-shape
`ControllerEvaluator` (a single batched call), rather than a throwaway evaluator
per point — the evaluator derives per-genome specs, so a mixed-shape population
scores in one pass (only num_motors/bits_per_feature/input_window_k must match,
which the grid holds constant).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from wnn.ram.strategies.connectivity.generic_grid_search import GenericGridSearch
from wnn.control.evaluator import (
	ControllerEvaluator,
	arch_shape_from_spec,
	fit_thresholds_from_pid_rollouts,
	calib_episode_config as _calib_ec,
)
from wnn.control.recurrent_genome import RecurrentArchGenome, RecurrentArchConfig
from wnn.control.ga_strategy import default_controller_ga_config, search_aggregation


def _steady_str(m) -> str:
	"""Steady-state error in degrees, or an em dash when the path did not produce one.

	Never prints 0.00 for a missing measurement — a missing steady must not be able to
	masquerade as a perfect one. CE used to occupy this column; it was removed because it
	plays no part in the controller's fitness ranking (ControllerHarmonic weighs
	err/stable/jerk/mono, no CE term) and reporting it invited exactly the confusion it
	caused before (Luiz, 05/08/2026).
	"""
	v = getattr(m, "mean_steady_error_deg", None)
	return "—" if v is None else f"{v:.2f}°"


def _alt_str(m) -> str:
	"""Mean |altitude error| in METRES, or an em dash if the path did not produce one.

	The grid stage ranks on altitude like every other stage (its calculator carries
	weight_alt since 38ef82e6), but its two log lines printed only steady/err/stable —
	so the one place the grid's altitude could be read was the held-out block at the very
	end of the run. That is too late to be useful and it forced the status tick to report
	a dash for a quantity that was measured all along (Luiz, 18/08/2026: showing the real
	altitude is a REQUIREMENT, not a preference).

	Same em-dash discipline as steady: never 0.000 for a missing measurement, because a
	zero here reads as a vehicle holding altitude perfectly — the exact opposite of "not
	measured". Metres carry their unit because every neighbouring number is degrees.
	"""
	v = getattr(m, "mean_altitude_error_m", None)
	return "—" if v is None else f"{v:.3f}m"



@dataclass


class _GridPoint:
	"""One (state_neurons, state_bits, output_bits, output_neurons) grid cell, with
	its spec/shape precomputed so `_make_genome`/`_make_variant` and the winner-spec
	readout are cheap. Tightly coupled to ControllerGridSearch — kept in this file.

	`on` is the output-neuron count = num_motors·levels_per_motor, i.e. the PWM
	decode resolution. It is a real grid axis only when --grid-output-neurons is
	given; otherwise every point carries the same num_motors·--levels."""
	sn: int
	sb: int
	ob: int
	on: int
	spec: Any
	shape: Any
	state_suffix: int
	output_suffix: int


class ControllerGridSearch(GenericGridSearch):
	"""Grid over (state_neurons × bits) for the drone controller."""

	def __init__(self, args, ec, seed: int, thresholds_override=None):
		self._args = args
		self._ec = ec
		self._seed = seed
		self._rng_master = np.random.default_rng(seed)
		self._t0 = time.time()
		self.elapsed = 0.0

		valid_pairs, probe_spec = self._compute_valid_pairs()
		self._valid_pairs = valid_pairs
		self._probe_spec = probe_spec
		# Thresholds come from PID rollouts (arch-independent) — fit once on the
		# probe spec and reuse for the shared evaluator AND the later stages.
		# thresholds_override: the REGRID pass of the student-state refit hands in a
		# ladder already fitted on the student's own visited states, so this must NOT
		# re-derive the teacher-only one. Everything downstream (the shared
		# evaluator, every genome's address function) then uses the refitted ladder.
		self.thresholds = thresholds_override if thresholds_override is not None else \
			fit_thresholds_from_pid_rollouts(
				probe_spec, num_episodes=10, seed=seed,
				geometry=getattr(ec, "geometry", None),
				alloc=getattr(ec, "alloc_residual", None),
				episode_config=_calib_ec(args, ec),
				outer_quantile=getattr(args, "threshold_outer_quantile", None))
		# ONE shared mixed-shape evaluator for every grid point (kills the old
		# throwaway-evaluator-per-point).
		self._evaluator = ControllerEvaluator(
			probe_spec, num_eval_episodes=args.eval_episodes, seed=seed,
			episode_config=ec, thresholds=self.thresholds,
			rg_config=self._rg_config(), max_train_workers=args.train_workers,
			num_eval_folds=args.num_eval_folds)
		self._calc = default_controller_ga_config(
			population_size=args.pop,
			weight_err_sq=args.fit_weight_err_sq,
			weight_stable=args.fit_weight_stable,
			weight_jerk=args.fit_weight_jerk,
			weight_mono=args.fit_weight_mono,
			weight_steady=args.fit_weight_steady,
			weight_effort=getattr(args, "fit_weight_effort", 0.0),
			# The grid ranks its points with this calculator, so leaving alt out
			# made the GRID stage blind to altitude too. It also fed the
			# "GRID WINNER (by ControllerHarmonic(...))" line, which printed no
			# alt= and was therefore telling the truth — that line was diagnostic,
			# not cosmetic, and was misread as a display gap on 18/08.
			weight_alt=getattr(args, "fit_weight_alt", 0.0),
			weight_pos=getattr(args, "fit_weight_pos", 0.0),
			# The grid is IN-SEARCH: unset --fit-aggregation keeps it on the
			# legacy harmonic, set applies the one coherent mode end-to-end.
			aggregation=search_aggregation(args),
			zrank_clamp=getattr(args, "zrank_clamp", 3.0),
		).create_fitness_calculator()
		super().__init__(top_k=args.grid_top_k, population_size=args.pop, log=print)

	# ---- setup helpers ----------------------------------------------------
	def _spec(self, sn: int, b: int, ob: int, on: "int | None" = None):
		"""Build a ControllerSpec for a grid point. Lazy-imports `_make_spec`
		from phased_ga (module-level import would be circular).

		`on` (output_neurons) selects the point's PWM resolution: levels_per_motor
		= on // num_motors, matching evaluator.spec_from_arch. None → --levels."""
		from wnn.control.phased_ga import _make_spec
		args = self._args
		num_motors = getattr(args, '_geometry_num_motors', 4)
		levels = args.levels if on is None else int(on) // num_motors
		return _make_spec(
			sn, levels, b, args.delta_control, args.delta_leak,
			getattr(args, "delta_max", 0.1),
			getattr(args, "delta_gamma", 1.0),
			obs_tilt_p=args.obs_tilt_p, obs_tilt_i=args.obs_tilt_i,
			obs_peraxis_p=args.obs_peraxis_p, obs_peraxis_i=args.obs_peraxis_i,
			obs_peraxis_yaw=args.obs_peraxis_yaw, obs_pwm=args.obs_pwm,
			obs_yaw_err=args.obs_yaw_err, obs_yaw_err_i=args.obs_yaw_err_i,
			dhat_b=getattr(args, "_dhat_b", None),
			dhat_l_gain=getattr(args, "dhat_l_gain", 0.05),
			dhat_ff=getattr(args, "dhat_ff", False),
			dhat_ff_clamp=getattr(args, "dhat_ff_clamp", 0.30),
			obs_collective_cmd=getattr(args, "obs_collective_cmd", False),
			obs_pos_err_xy=getattr(args, "obs_pos_err_xy", False),
			obs_vel_xy=getattr(args, "obs_vel_xy", False),
			obs_alt_err=getattr(args, "obs_alt_err", False),
			obs_vz=getattr(args, "obs_vz", False),
			integral_leak=args.integral_leak, integral_scale=args.integral_scale,
			decouple_outputs=args.decouple_outputs,
			bits_per_feature=args.bits_per_feature,
			feature_balance_ratio=args.feature_balance_ratio,
			conn_policy=getattr(args, "_conn_policy", "spread"),
			conn_policy_min=getattr(args, "_conn_policy_min", 2),
			conn_mutation_scope=getattr(args, "conn_mutation_scope", "free"),
			output_full_window=getattr(args, "output_full_window", False),
			frame_stride=int(getattr(args, "frame_stride", 1)),
			target_levels=int(getattr(args, "target_levels", 0)),
			threshold_gamma=args.threshold_gamma, action_repeat=args.action_repeat,
			output_bits=ob, num_motors=num_motors,
			input_window_k=getattr(args, "input_window_k", 4),
			memory_mode=args.memory_mode,
			output_decode=getattr(args, "output_decode", None))

	def _rg_config(self):
		from wnn.control.phased_ga import _rg_config
		return _rg_config(self._args, self._ec, self._seed)

	def _compute_valid_pairs(self):
		"""Enumerate + validity-filter the (sn, sb, ob, on) grid; return
		(valid_pairs, probe_spec). Mirrors the old stage0_grid pre-filter.

		`on_axis` is a single point unless --grid-output-neurons was given, so the
		default cardinality is unchanged. The suffix geometry does NOT depend on it:
		arch_shape_from_spec derives state/output_input_space from input_window_k,
		num_features and bits_per_feature only — never from levels_per_motor — so
		one probe still serves the whole coverage branch."""
		from wnn.control.phased_ga import grid_output_neuron_axis
		args = self._args
		min_suffix = args.grid_min_suffix
		cov = getattr(args, "suffix_coverage", 0.0)
		on_axis = grid_output_neuron_axis(args)
		if cov > 0.0:
			probe = self._spec(args.grid_state_neurons[0], args.grid_state_neurons[0] + min_suffix,
			                   args.grid_state_neurons[0] + min_suffix, on_axis[0])
			sh = arch_shape_from_spec(probe); pf = sh.prefix_factor
			osuf = min(max(min_suffix, round(cov * sh.output_input_space)), sh.output_input_space)
			ssuf = min(max(min_suffix, round(cov * sh.state_input_space)), args.suffix_cap, sh.state_input_space)
			valid_pairs = [(sn, pf * sn + ssuf, pf * sn + osuf, on)
			               for sn in args.grid_state_neurons for on in on_axis]
			all_pairs = valid_pairs
			print(f"  [grid] per-layer coverage={cov}: state_suffix={ssuf} (of {sh.state_input_space}), "
			      f"output_suffix={osuf} (of {sh.output_input_space}), cap={args.suffix_cap}")
		else:
			all_pairs = [(sn, b, b, on) for sn in args.grid_state_neurons
			             for b in args.grid_bits for on in on_axis]
			valid_pairs = [(sn, sb, ob, on) for (sn, sb, ob, on) in all_pairs if (sb - sn) >= min_suffix]
		n_skipped = len(all_pairs) - len(valid_pairs)
		num_motors = getattr(args, '_geometry_num_motors', 4)
		lv_desc = (f"levels={args.levels}" if len(on_axis) == 1
		           else f"levels={[on // num_motors for on in on_axis]}")
		print(f"\n{'='*72}\n  STAGE 0: GRID SEARCH "
		      f"({len(valid_pairs)} valid pts of {len(all_pairs)} requested, "
		      f"{lv_desc}, min_suffix={min_suffix})\n{'='*72}")
		if n_skipped:
			print(f"  [grid] {n_skipped} pts skipped (bits − 2·state_neurons < {min_suffix}; "
			      f"need ≥{min_suffix} suffix bits for meaningful input sampling)")
		if not valid_pairs:
			raise RuntimeError(
				f"Grid search has zero valid points — every (sn, b) pair produces fewer than "
				f"{min_suffix} suffix bits. Increase --grid-bits, reduce --grid-state-neurons, "
				f"or lower --grid-min-suffix (currently {min_suffix}). "
				f"Requested sn={list(args.grid_state_neurons)}, bits={list(args.grid_bits)}.")
		probe_spec = self._spec(*valid_pairs[0])
		return valid_pairs, probe_spec

	# ---- GenericGridSearch hooks -----------------------------------------
	def _enumerate_points(self) -> list:
		points = []
		for sn, b, ob, on in self._valid_pairs:
			spec = self._spec(sn, b, ob, on)
			shape = arch_shape_from_spec(spec)
			points.append(_GridPoint(
				sn=sn, sb=b, ob=ob, on=on, spec=spec, shape=shape,
				state_suffix=b - shape.prefix_factor * sn,
				output_suffix=ob - shape.prefix_factor * sn))
		return points

	def _random_genome(self, point: _GridPoint) -> RecurrentArchGenome:
		args = self._args
		rng = np.random.default_rng(int(self._rng_master.integers(0, 2**32 - 1)))
		genome = RecurrentArchGenome.random(
			point.shape, state_neurons=point.sn,
			output_neurons=point.on,
			state_suffix=point.state_suffix, output_suffix=point.output_suffix, rng=rng,
			config=RecurrentArchConfig(conn_policy=getattr(args, "_conn_policy", "spread"),
			                           conn_policy_min=getattr(args, "_conn_policy_min", 2),
			                           feature_balance_ratio=args.feature_balance_ratio,
			                           bits_per_feature=args.bits_per_feature,
			                           memory_mode=args.memory_mode))
		genome.cells = None   # evaluate_batch trains via reward-gated adaptation
		return genome

	def _make_genome(self, point: _GridPoint) -> RecurrentArchGenome:
		return self._random_genome(point)

	def _make_variant(self, point: _GridPoint) -> RecurrentArchGenome:
		return self._random_genome(point)

	def _evaluate(self, genomes: list, is_expansion: bool) -> list:
		"""Score the whole batch through the one shared mixed-shape evaluator."""
		if getattr(self._ec, "geometry", None) is not None:
			metrics = self._evaluator.score_genomes(genomes)
		else:
			metrics = self._evaluator.evaluate_batch(genomes)
		tag = "expand" if is_expansion else "grid"
		for i, (g, m) in enumerate(zip(genomes, metrics)):
			print(f"  [{tag} {i+1:>2}/{len(genomes):>2}] sn={g.state_neurons:>2} "
			      f"on={g.output_neurons:>3} "
			      f"steady={_steady_str(m):>7}  "
			      f"err={m.mean_attitude_error_deg:>6.2f}°  "
			      f"stable={m.acc*100:>5.1f}%  "
			      f"alt={_alt_str(m):>8}")
		return metrics

	def _fitness(self, metrics: list) -> list:
		return self._calc.fitness(metrics)

	def _on_final(self, outcome) -> None:
		self.elapsed = time.time() - self._t0
		w = outcome.best_point
		wm = outcome.best_metrics
		print(f"\n  GRID WINNER (by {self._calc.name}): sn={w.spec.state_neurons} "
		      f"b={w.spec.state_bits_per_neuron} levels={w.spec.levels_per_motor}  "
		      f"steady={_steady_str(wm)}  err={wm.mean_attitude_error_deg:.2f}°  "
		      f"stable={wm.acc*100:.1f}%  alt={_alt_str(wm)}  "
		      f"seed_pop={len(outcome.seed_population)}  "
		      f"({self.elapsed:.0f}s)")
