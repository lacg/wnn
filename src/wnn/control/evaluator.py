"""ControllerEvaluator — mirror of IDSEvaluator for the controller pipeline.

Same role: given a "genome" (architecture + connectivity + learned cells),
evaluate it on a held-out set of test episodes and return metrics. The
worker dispatches `architecture_type='controller'` flows here instead of
IDSEvaluator.

A "genome" for the controller is:
  - (state_neurons, state_bits_per_neuron, output_bits_per_neuron):
    shape parameters that come from the grid_search / ga_neurons phases.
  - state_connections, output_connections: which input bits each neuron
    addresses. Same role as IDS connections — evolved by the GA.
  - thresholds: per-feature thermometer thresholds (NUM_FEATURES *
    bits_per_feature floats). Either fitted from sim rollouts at
    evaluator init (preferred) or supplied externally.
  - state_cells, output_cells: trained cell values. The trainer
    (BPTT/EDRA) writes these. For an untrained genome the cells are
    empty (default WEAK_FALSE) and the controller emits its default
    PWM=0.75 from Strategy 5.

The evaluator is reproducible via a seed: every episode uses an RNG
derived deterministically from (seed, episode_idx).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from wnn.control._accel import AttitudeSim, WnnController

from .pid import AttitudePID, AttitudePIDConfig
from .training import (
	EpisodeConfig,
	fitness_function,
	make_pid_action_fn,
	make_wnn_action_fn,
)


# Layout constants matching controller.rs NUM_FEATURES = 9
NUM_FEATURES = 9

# DAGGER teacher name → RewardGatedConfigPacked integer id (dagger_train.rs).
_TEACHER_IDS = {"pid": 0, "lqr": 1, "mpc": 2, "lqi": 3, "mpcof": 4}


def _dist_packed_fields(rg) -> tuple:
	"""W2: the 12 disturbance args for RewardGatedConfigPacked, read from
	rg.episode_config.disturbance (a training.DisturbanceConfig). None →
	disabled defaults (the packed config's own defaults = pre-W2 behavior).

	Returns (enabled, tau_bias, gust_sigma, gust_tau_c, motor_asym,
	gyro_sigma, gyro_bias_walk, accel_sigma, dropout_prob, dropout_len_steps,
	obs_delay_steps, torque_scale_jitter) — the last 4 are the W2.4 D5/D6/D7
	levers (0 = exactly-off). Note: the batched Rust trainer uses the FIXED
	motor_asym multipliers — motor_asym_mag's per-episode δ draw is a
	run_episode-only convenience (motor wear is per-airframe)."""
	d = getattr(getattr(rg, "episode_config", None), "disturbance", None)
	if d is None:
		return (False, [0.0, 0.0, 0.0], 0.0, 0.1, [1.0, 1.0, 1.0, 1.0], 0.0, 0.0, 0.0,
		        0.0, 0, 0, 0.0)
	return (
		True,
		[float(x) for x in d.tau_bias],
		float(d.gust_sigma),
		float(d.gust_tau_c),
		[float(x) for x in d.motor_asym],
		float(d.gyro_sigma),
		float(d.gyro_bias_walk),
		float(d.accel_sigma),
		float(d.dropout_prob),
		int(d.dropout_len_steps),
		int(d.obs_delay_steps),
		float(d.torque_scale_jitter),
	)


def _rust_dagger_enabled() -> bool:
	"""Native Rust reward-gated DAGGER training is the DEFAULT (opt-OUT).

	It runs the whole rollout+train loop in Rust (Rayon-parallel across genomes,
	~3× faster with cores) AND computes the motor-jerk + monotonicity-violation
	fitness metrics that the Python reference path drops. The Python path
	(reward_gated_train) is a slow, single-genome reference/parity fallback — it
	is NOT what you want by default. Set WNN_RUST_DAGGER=0/off/false/no to force
	it for parity tests or debugging.

	Flipped from opt-in → opt-out 01/06/2026: the path was gated for validation
	when added (29/05), parity was confirmed (test_controller_gpu_parity /
	_solver_parity), but the default was never flipped — so every default run was
	slow and silently ignored weight_jerk/weight_mono."""
	import os
	return os.environ.get("WNN_RUST_DAGGER", "1").strip().lower() not in ("0", "false", "off", "no")


@dataclass
class ControllerSpec:
	"""Shape/architecture of a controller — the equivalent of IDS's
	(neurons, bits) genome but extended for the controller-specific
	layers + window."""
	num_motors: int = 4
	levels_per_motor: int = 256
	bits_per_feature: int = 8
	input_window_k: int = 4

	# State layer
	state_neurons: int = 200
	state_bits_per_neuron: int = 18

	# Output layer (one neuron per motor × level)
	output_bits_per_neuron: int = 18

	# Delta-control: output decodes to a per-step PWM delta (accumulated),
	# not an absolute throttle. Untrained → hold (stable bootstrap). See
	# project_controller_state. delta_max is the per-step clamp.
	delta_control: bool = False
	delta_max: float = 0.1
	# Leaky integrator (delta mode): accumulator deviation from hover decays by
	# delta_leak each step. 1.0 = pure integrator (can run away); <1.0 bounds the
	# steady-state offset to delta/(1-leak).
	delta_leak: float = 1.0
	# Non-uniform delta alphabet (09/08/2026): |t|^gamma before scaling. Same range,
	# neutral, level count and footprint — resolution CONCENTRATED near zero where
	# the hold/steady window lives, coarser near full authority where the transient
	# dominates and precision is worthless. 1.0 = the original piecewise-linear map
	# (bit-identical; Rust/Metal short-circuit gamma==1). The cheap alternative to
	# raising `levels`, which the alphabet probe showed costs 3x cells for an
	# unreliable gain.
	delta_gamma: float = 1.0

	# H2 observation features (18/06/2026): append error/integral features to the
	# 9 raw sensors. num_features = 9 + tilt_p + tilt_i + 3·peraxis_p + 3·peraxis_i.
	# All-off (default) ⇒ 9 features ⇒ identical to pre-H2. "_p" = the error itself
	# (proportional perception), "_i" = its leaky integral (the steady-state killer).
	obs_tilt_p: bool = False      # tilt-to-vertical error (gravity ref, accel-only)
	obs_tilt_i: bool = False      # leaky integral of the tilt error
	obs_peraxis_p: bool = False   # roll/pitch/yaw error (3 features, or 2 if yaw dropped)
	obs_peraxis_i: bool = False   # leaky integrals of the 3-axis error (3 features, or 2 if yaw dropped)
	# When False, per-axis features push only roll+pitch (gravity-observable from accel)
	# and DROP yaw, whose dead-reckoned (gyro-z) estimate drifts and poisons control.
	# Default True = original 3-axis behaviour (parity anchor).
	obs_peraxis_yaw: bool = True
	# obs_pwm: expose the RAW throttle accumulator (current pwm, num_motors feats) —
	# the DIRECT fix for delta-mode's hidden state (∫error via obs_tilt_i is only a
	# proxy that decorrelates in the untrained regime; confirmed insufficient 18/06).
	obs_pwm: bool = False
	# Yaw-anchor (Phase A, 26/06): a CLEAN scalar yaw-error channel — NOT via obs_peraxis
	# (which degenerates on roll/pitch atan2 at large tilt). obs_yaw_err = proportional
	# (target_yaw − anchored heading); obs_yaw_err_i its leaky integral. When either is on,
	# the controller is yaw-anchored: yaw_heading seeds to the episode's true initial yaw
	# (from q0) and integrates with dt → an absolute yaw reference for the 4–6° SOFT band.
	obs_yaw_err: bool = False
	obs_yaw_err_i: bool = False
	# L1 (06/08): the mpcof teacher's disturbance observer, moved into the student.
	# dhat_b = the plant's control effectiveness [b_roll,b_pitch,b_yaw] from
	# ram_controller.calibrate_control_gains (NEVER re-derived in Python). None = OFF,
	# which is the parity anchor for every pre-L1 run. When set, +3 features carrying
	# the estimated external angular acceleration — the term the D2 decomposition
	# showed the student cannot otherwise observe (docs/hold_floor_levers_spec.md).
	dhat_b: "tuple[float, float, float] | None" = None
	dhat_l_gain: float = 0.05
	# OUTPUT-SIDE disturbance observer (10/08/2026). L1 fed d̂ to the student as an
	# INPUT and lost 4/4 — a quantized LUT cannot learn to subtract a continuous
	# bias. mpcof does not learn it either: it computes u = policy − clamp(d̂/b) in
	# f64 downstream of the policy, which is why it posts 0.00±0.00 steady. This
	# moves that line into the student's actuator path. Requires dhat_b (the plant
	# gains) to be set; the LUT itself is unchanged. ~6 flops/axis/step.
	dhat_ff: bool = False
	dhat_ff_clamp: float = 0.30     # observer gain (teacher default)
	# SCOPE C STAGE 1 (13/08): the vertical channel — canonical order LAST, so every
	# pre-stage-1 layout is unchanged when these are off (docs/scope_c_full_controller_spec.md).
	# obs_collective_cmd is what makes the controller COMPOSABLE: any outer loop
	# (including pybullet's DSLPID) can hand it a collective. Mass and gravity are
	# never features — they are randomized PLANT parameters (Luiz, 12/08).
	obs_collective_cmd: bool = False
	obs_alt_err: bool = False
	obs_vz: bool = False
	# SCOPE C STAGE 2 (14/08): the horizontal channel. Each flag carries BOTH
	# axes — x and y are the same physics rotated 90° on a symmetric quad, so a
	# one-axis controller would be an artifact (decision 5, chunk-B/C doc).
	obs_pos_err_xy: bool = False
	obs_vel_xy: bool = False
	integral_leak: float = 0.99   # leaky-integral decay for "_i" (≠ delta_leak)
	integral_scale: float = 1.0   # pre-threshold scale for integral features
	dt: float = 0.001             # physics step (s); MUST match episode/plant dt (yaw integ.)
	# H3 (18/06): the 4 output banks are CONTROLS [T, τ_roll, τ_pitch, τ_yaw] mixed to
	# motors (orthogonal action space; per-control delta accumulator). Requires num_motors=4.
	decouple_outputs: bool = False
	# Feature-balance cap (26/06/2026): no input feature may capture more than this ratio ×
	# the least-wired feature's connection count (targets obs_yaw_err's 2.14x over-wiring →
	# coupling). 0/≤1 disables. Threaded → RecurrentArchConfig.feature_balance_ratio.
	feature_balance_ratio: float = 0.0
	# Connection-creation policy (14/08/2026 specialist programme): how fresh
	# OUTPUT-layer maps are drawn. "spread" = legacy uniform (bit-identical);
	# "min_per_cluster" = every touched feature gets >= conn_policy_min bits,
	# unaffordable features dropped, remainder donated. Threaded ->
	# RecurrentArchConfig like feature_balance_ratio.
	conn_policy: str = "spread"
	conn_policy_min: int = 2
	# ARM D (14/08/2026): output layer samples the FULL K-frame window instead of
	# frame t-0 only. Requires state_neurons == 0 (Rust refuses otherwise).
	output_full_window: bool = False
	# FRAME STRIDE (15/08/2026): the K-window shifts once every N pushes, so it
	# spans N*K steps. At dt=1ms, k=4/stride=1 is a 4 ms lookback where t-1 is
	# nearly a copy of t-0; stride=10 gives 40 ms. 1 = legacy every-step window.
	frame_stride: int = 1
	# E3 threshold-density warp (01/07/2026, plan controller_break_90_v2): warp the
	# thermometer quantile positions toward the MEDIAN of each feature's PID-rollout
	# distribution. gamma=1.0 = uniform quantiles (parity anchor); gamma>1 densifies
	# thresholds near hover (where the ki=0 re-anchor located the precision gap —
	# soft-fail fixed points at ~5.6° = coarse decode near zero error).
	threshold_gamma: float = 1.0
	# Action-repeat (arm R, 02/07/2026 — Sajus frame-skip adapted): decide every
	# Nth physical step, HOLD the PWM in between. 1 = today's behavior
	# (bit-identical). The 4-frame window then spans 4N physical steps; jerk drops;
	# each decision's consequence is larger. Propagated by spec_from_arch (like
	# threshold_gamma) so it can't silently revert after the grid stage.
	action_repeat: int = 1
	# Memory mode of both layers' cells (ABI 12 granularity ablation, Luiz
	# 12/07/2026): "QUAD_WEIGHTED" (default, bit-identical to pre-12) /
	# "TERNARY" (FALSE/TRUE/EMPTY, empty decodes 0.5 — PLN convention) /
	# "BINARY" (classical WiSARD 1-bit; output decodes via antagonist-pair
	# E/I halves so the effective neutral is 0.5). split_train[_loop]
	# (WNN_STATE_SPLIT) is mode-aware since 12/07/2026 (plant_cell: QUAD
	# strong-on/soft-off, TERNARY/BINARY hard TRUE/FALSE).
	memory_mode: str = "QUAD_WEIGHTED"

	# Output decode TOPOLOGY, orthogonal to memory_mode (03/08/2026). None => the
	# mode's historical default, so every cohort measured before this reproduces:
	# BINARY -> antagonist, everything else -> cumulative.
	#
	# It is a separate axis because "antagonist E/I" was only ever welded to BINARY
	# out of necessity — a 1-bit cell reads 0 untrained, so one thermometer bank can
	# only push up from the floor. QUAD does not NEED it, but QUAD's neutral is
	# QUAD_WEIGHTS[EMPTY]=0.75, which is not the middle: a cell travels 0.75 down but
	# only 0.25 up, a 3:1 asymmetry around hover. Under the antagonist decode an
	# untrained QUAD bank cancels to exactly 0.5 with symmetric authority.
	output_decode: "str | None" = None

	OUTPUT_DECODES = {"CUMULATIVE": 0, "ANTAGONIST": 1}

	# Canonical name → Rust neuron_memory constant (single mapping site).
	# QSR (4) = stochastic QUAD read (per-timestep coin, E[fire]=QUAD weight);
	# PLN (5) = stochastic TERNARY read (shares TERNARY's 3-state cells). Both
	# decode stochastically in Rust (Part 5, controller ABI 12); the deterministic
	# weight IS the fire probability.
	MEMORY_MODES = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2, "BINARY": 3,
	                "QSR": 4, "PLN": 5}

	def memory_mode_int(self) -> int:
		"""The Rust memory-mode constant for this spec (loud on typos)."""
		key = self.memory_mode.upper()
		if key not in self.MEMORY_MODES:
			raise ValueError(
				f"unknown memory_mode {self.memory_mode!r} — one of {sorted(self.MEMORY_MODES)}")
		return self.MEMORY_MODES[key]

	def output_decode_int(self) -> "int | None":
		"""The Rust topology constant, or None to let Rust pick the mode default.

		Returning None rather than resolving the default here keeps ONE source of
		truth for "what does this mode normally use" — cell_mode::default_output_decode
		— instead of a Python copy that could drift from it."""
		if self.output_decode is None:
			return None
		key = self.output_decode.upper()
		if key not in self.OUTPUT_DECODES:
			raise ValueError(
				f"unknown output_decode {self.output_decode!r} — one of {sorted(self.OUTPUT_DECODES)}")
		return self.OUTPUT_DECODES[key]

	def resolved_output_decode(self) -> str:
		"""The decode this spec WILL run under, with the mode default applied.

		output_decode_int() deliberately returns None for "let Rust pick", which is
		right for the forward path — but the ARCHITECTURE SEARCH has to know the
		answer in Python, because the decode constrains what shapes are legal
		(antagonist needs an even levels_per_motor). This is the one Python site
		allowed to mirror cell_mode::default_output_decode; every other caller must
		go through here rather than re-deriving it, so there is a single place to fix
		if the Rust default ever changes.

		Mirrors cell_mode.rs:83 default_output_decode — BINARY→antagonist (a 1-bit
		cell reads 0 untrained, so one bank could only push up from the floor),
		everything else→cumulative."""
		if self.output_decode is not None:
			return self.output_decode.lower()
		return "antagonist" if self.memory_mode.upper() == "BINARY" else "cumulative"

	def num_features(self) -> int:
		"""9 base sensors + enabled extras (H2 error/integral + raw accumulator)."""
		peraxis_n = 3 if self.obs_peraxis_yaw else 2  # drop yaw → roll+pitch only
		return 9 + int(self.obs_tilt_p) + int(self.obs_tilt_i) \
			+ peraxis_n * int(self.obs_peraxis_p) + peraxis_n * int(self.obs_peraxis_i) \
			+ self.num_motors * int(self.obs_pwm) \
			+ int(self.obs_yaw_err) + int(self.obs_yaw_err_i) \
			+ 3 * int(self.dhat_b is not None) \
			+ int(self.obs_collective_cmd) + int(self.obs_alt_err) + int(self.obs_vz) \
			+ 2 * int(self.obs_pos_err_xy) + 2 * int(self.obs_vel_xy)


@dataclass
class AdaptationStats:
	"""Per-genome stats surfaced by evaluate_for_adaptation to guide genesis.
	cell counts = distinct addresses written per neuron (the controller-side
	'fill' signal); reward/stable_rate are the closed-loop fitness components."""
	reward: float
	stable_rate: float
	state_cell_counts: list   # len = state_neurons
	output_cell_counts: list  # len = num_motors * levels_per_motor


@dataclass
class ControllerGenome:
	"""A specific instantiation of a controller — connectivity + thresholds
	+ trained cells. Built by the GA + trainer; consumed by the evaluator."""
	spec: ControllerSpec
	# Per-feature thermometer thresholds, flat: NUM_FEATURES * bits_per_feature.
	thresholds: list[float]
	# State layer connectivity, flat: state_neurons * state_bits_per_neuron.
	state_connections: list[int]
	# Output layer connectivity, flat: (num_motors * levels_per_motor) *
	# output_bits_per_neuron.
	output_connections: list[int]
	# Trained cell values per layer. Each is a list of (neuron_idx, address,
	# value) tuples. Defaults empty → all-EMPTY controller.
	state_cells: list[tuple[int, int, int]] = field(default_factory=list)
	output_cells: list[tuple[int, int, int]] = field(default_factory=list)
	# Stage B (ABI 20): the genome's cells as a Rust GenomeCells handle. When
	# set, build_controller bulk-loads it (load_cells_handle) instead of the
	# per-cell write loop — no triple materialisation. The triple lists above
	# remain for explicit-cell callers and stay authoritative when non-empty.
	cells_handle: object | None = None


def fold_pool_seed(seed: int, fold_index: int) -> int:
	"""Episode-pool seed for one K-fold index — the seed the scorer ACTUALLY samples
	initial conditions from when K>1.

	Promoted to module level 29/07/2026 so anything that must reproduce a scoring run
	derives the pool seed from here instead of re-deriving the constants. Any second
	copy silently scores a DIFFERENT episode set: compute_baselines used the raw report
	seed and so flew the classical baselines on episodes no WNN cell ever saw, worth
	11pp of PID stability on the canonical seed.

	Note the asymmetry with K=1, where _advance_fold keeps the RAW seed instead
	(legacy single-pool behavior) — so a caller reproducing a run needs the fold COUNT
	as well as the index, not just this function.
	"""
	return (seed * 0x9E3779B97F4A7C15 + fold_index * 0xBF58476D1CE4E5B9) & 0xFFFFFFFF


def disturbance_stream(dist, score_seed: int) -> tuple:
	"""(stream_seed, resolved_motor_asym) for one scoring pass.

	Promoted alongside fold_pool_seed for the same reason: the asymmetry must be the
	RESOLVED per-airframe draw, and it must be resolved from the XOR'd stream seed.
	compute_baselines passed the raw dist.motor_asym — the fixed (1,1,1,1) multiplier —
	and so flew the classical baselines on a perfectly symmetric quadrotor while every
	WNN cell carried an ~8% weak motor. Worth 8pp of PID stability.

	score_seed is the ACTIVE score seed (the fold pool seed when K>1), not the report
	seed; see fold_pool_seed.
	"""
	dseed = (int(dist.seed) ^ int(score_seed)) & 0xFFFFFFFFFFFFFFFF
	return dseed, dist.resolved_motor_asym(np.random.default_rng(dseed))


def apply_motor_fault(dist, fault: str):
	"""Inject a fixed single-motor effectiveness fault: 'idx:factor' multiplies
	dist.motor_asym[idx] (the FIXED per-airframe multiplier) by factor.

	One source for BOTH phased_ga (training + scoring) and compute_baselines —
	the 29/07 lesson is that any condition applied to one side only silently
	unmatches the comparison. Multiplying the fixed multiplier composes with the
	per-airframe draw in resolved_motor_asym (multiplication commutes), and flows
	through _dist_packed_fields to the batched Rust trainer, so the student
	TRAINS on the faulted plant too — which is the point of the experiment.
	"""
	if not fault:
		return dist
	idx_s, fac_s = fault.split(":")
	ma = list(dist.motor_asym)
	ma[int(idx_s)] = float(ma[int(idx_s)]) * float(fac_s)
	dist.motor_asym = tuple(ma)
	return dist


def calib_episode_config(args, ec):
	"""The EpisodeConfig the THERMOMETER is calibrated on.

	Defaults to `ec` — calibrate on the regime we fly (the 09/08/2026 fix; the old
	hardcoded 30 deg wasted ~83% of the ladder on states a 5 deg episode never
	visits). `--threshold-calib-tilt` decouples the two: calibrating NARROWER than
	the flown tilt buys finer near-zero bins (where the hold metric lives) at the
	cost of saturating the transient, which is a cheap trade because a large error
	only needs "big, this sign", not its magnitude. Measured saturation against the
	flown 5 deg distribution: cal 30 deg 11.3% outside the ladder, 5 deg 31.0%,
	2.5 deg 39.6%, 1 deg 59.1% — 1 deg is over the cliff, 2.5 deg is the live
	candidate. Stability is the metric that catches this going wrong."""
	tilt = getattr(args, "threshold_calib_tilt", None)
	if tilt is None:
		return ec
	import dataclasses, math as _m
	return dataclasses.replace(ec, max_initial_tilt_rad=_m.radians(float(tilt)))


def fit_thresholds_from_pid_rollouts(
	spec: ControllerSpec,
	num_episodes: int = 20,
	seed: int = 0,
	method: str = "quantile",
	geometry=None,        # Optional[GeometryConfig] — N-rotor TRUE table (sim side)
	alloc=None,           # Optional[AllocResidualConfig] — baseline driver gains
	episode_config=None,  # Optional[EpisodeConfig] — the OPERATING regime (see below)
	outer_quantile=None,  # Optional[float] — coverage margin, see below
	extra_samples=None,   # Optional[list[list[float]]] — per-feature student states
) -> list[float]:
	"""Fit per-feature thermometer thresholds by running reference-driven
	rollouts and collecting the empirical sensor distributions.

	⚠️ CALIBRATE ON THE REGIME YOU FLY (09/08/2026). The rollout config used to be
	hardcoded at 30° initial tilt while every production recipe flies `--tilt 5.0`,
	so the thermometer was quantile-fitted on a state distribution ~6× wider than
	the controller ever visits. Quantile spacing puts bins where the data is, so a
	30° transient spends resolution on states that never occur in a 5° episode and
	coarsens the near-zero region — exactly where the hold floor lives (the steady
	metric is the mean error over the settled window). Pass the run's real
	`episode_config` and the thermometer calibrates on the operating regime.

	`None` keeps the legacy 30° config, so the ~60 scripts/tests that call this
	without an episode config are unchanged — but every production path in
	phased_ga / controller_grid_search passes `ec`.

	Quad (default): PID drives sim.step. Overactuated (geometry set): the
	allocator-LQR baseline drives step_n on the TRUE table, so the
	thermometer calibrates on the residual policy's actual operating region.

	Args:
		spec:         ControllerSpec for the controller architecture.
		num_episodes: reference rollouts used to gather sensor distribution data.
		seed:         RNG seed for reproducibility.
		method:       'quantile' (uniformly spaced quantiles → distributive
		              thermometer) or 'linear' (min/max linear spacing).

	Returns:
		thresholds: flat list of length NUM_FEATURES * bits_per_feature.
	"""
	rng = np.random.default_rng(seed)
	# SCOPE C STAGE 1 (13/08/2026): when the run flies TRANSLATION, the ladder
	# must be fit on that plant. Otherwise z/vz/collective are identically 0
	# during calibration and their thermometer thresholds all come out 0.0 —
	# measured: span 0.0000 on all three, so COLLECTIVE became a constant and
	# ALT_ERR/VZ carried one sign bit each instead of eight, while still costing
	# 24 address bits. Strictly worse than not having the features.
	#
	# Scoped deliberately to translation runs: switching the ATTITUDE-only
	# calibration to the airframe would shift every banked result, which is a
	# separate decision (see the note returned to Luiz 13/08).
	_ec_af = getattr(episode_config, "airframe", None) if episode_config is not None else None
	_ec_translation = bool(getattr(episode_config, "translation", False)) if episode_config is not None else False
	# CALIBRATION-PLANT A/B (13/08/2026, task #11): --calib-airframe fits the
	# ladder on the airframe we actually fly even for ATTITUDE-ONLY runs. Default
	# False = the historical synthetic-plant fit, so every banked run is
	# bit-identical until the A/B says otherwise. Adoption is a lineage break
	# (~85% of 30-bit addresses move), hence a flag and a paired experiment
	# rather than a silent flip.
	_ec_calib_af = bool(getattr(episode_config, "calib_airframe", False)) if episode_config is not None else False
	_stage1_cal = (_ec_translation or _ec_calib_af) and _ec_af is not None
	if _stage1_cal:
		sim = AttitudeSim(dt=float(getattr(episode_config, "dt", 0.001)),
		                  arm_length=float(_ec_af.arm_length), k_thrust=float(_ec_af.k_thrust),
		                  k_drag=float(_ec_af.k_drag),
		                  inertia=[float(x) for x in _ec_af.inertia],
		                  gravity=float(_ec_af.gravity))
		# Vertical dynamics only when the RUN has them; the calib-airframe arm on
		# an attitude-only run must stay attitude-only or it is not an A/B of the
		# ladder, it is an A/B of the plant.
		if _ec_translation:
			sim.set_translation(float(_ec_af.mass))
			_hover_pwm = sim.hover_pwm()
		else:
			_hover_pwm = 0.5
		_target_alt = float(getattr(episode_config, "target_altitude", 0.0))
	else:
		sim = AttitudeSim()
		_hover_pwm, _target_alt = 0.5, 0.0
	if geometry is not None:
		from wnn.control._accel import AllocLqrRs
		sim.set_geometry([list(r) for r in geometry.rows])
		if geometry.rotor_asym is not None:
			sim.set_rotor_asym([float(x) for x in geometry.rotor_asym])
		nominal = (alloc.nominal_rows if alloc is not None and alloc.nominal_rows is not None
		           else geometry.rows)
		driver = AllocLqrRs(
			[list(r) for r in nominal],
			q_att=(alloc.q_att if alloc else 12.0), q_rate=(alloc.q_rate if alloc else 1.0),
			r_ctrl=(alloc.r_ctrl if alloc else 1.0), tau_max=(alloc.tau_max if alloc else 0.144),
			f_hover=(alloc.f_hover if alloc else None),
			pinv_lambda=(alloc.pinv_lambda if alloc else 1e-6))
	elif _stage1_cal:
		# The firmware cascade commands mass*g/4, so it actually hovers — the
		# legacy PID sits at 0.5 and would fall, corrupting the accel ladder.
		from wnn.control.pid_firmware import AttitudePidFirmware
		pid = AttitudePidFirmware(_ec_af, _ec_af.gains())
	else:
		pid = AttitudePID(AttitudePIDConfig())
	cfg = episode_config or EpisodeConfig(
		dt=0.001, steps_per_episode=2000,
		max_initial_tilt_rad=math.radians(30.0),
		max_initial_yaw_rad=math.radians(30.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)

	# Collect per-feature samples across rollouts. The base 9 (gyro/accel/target)
	# are read directly; H2 extras (tilt/per-axis/integrals) come from the Rust
	# getter on a feature-extraction controller, so the thermometer calibrates on
	# the SAME values step() produces (single source of truth — no Python re-derive).
	nf = spec.num_features()
	needs_extras = nf > NUM_FEATURES
	samples_per_feature: list[list[float]] = [[] for _ in range(nf)]
	feat_ctl = None
	if needs_extras:
		# Dummy thresholds/connections: compute_features reads neither, so this
		# controller exists ONLY to evolve the integral state + expose features.
		# NOTE: feat_ctl deliberately keeps the default action_repeat=1 — the
		# threshold calibration must sample features at EVERY physical step
		# (the accumulators tick per step regardless of the deploy-time N).
		dummy_th = [0.0] * (nf * spec.bits_per_feature)
		s_conns = [0] * (spec.state_neurons * spec.state_bits_per_neuron)
		o_conns = [0] * (spec.num_motors * spec.levels_per_motor * spec.output_bits_per_neuron)
		feat_ctl = WnnController(
			num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
			bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
			state_neurons=spec.state_neurons, state_bits_per_neuron=spec.state_bits_per_neuron,
			output_bits_per_neuron=spec.output_bits_per_neuron, thresholds=dummy_th,
			state_connections=s_conns, output_connections=o_conns,
			delta_control=spec.delta_control, delta_max=spec.delta_max, delta_leak=spec.delta_leak, delta_gamma=getattr(spec, 'delta_gamma', 1.0),
			obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i,
			obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i,
			obs_collective_cmd=spec.obs_collective_cmd, obs_alt_err=spec.obs_alt_err, obs_vz=spec.obs_vz,
			obs_pos_err_xy=getattr(spec, 'obs_pos_err_xy', False), obs_vel_xy=getattr(spec, 'obs_vel_xy', False),
		output_full_window=getattr(spec, 'output_full_window', False),
		frame_stride=int(getattr(spec, 'frame_stride', 1)),
			dhat_b=(list(spec.dhat_b) if spec.dhat_b is not None else None), dhat_l_gain=spec.dhat_l_gain, dhat_ff=getattr(spec, 'dhat_ff', False), dhat_ff_clamp=getattr(spec, 'dhat_ff_clamp', 0.30), dt=spec.dt,
			integral_leak=spec.integral_leak, integral_scale=spec.integral_scale)

	for ep_idx in range(num_episodes):
		ep_seed = int(rng.integers(0, 2**32 - 1))
		ep_rng = np.random.default_rng(ep_seed)
		# Run one PID episode while recording sensor values at every step.
		# Re-do the inner loop directly (rather than via run_episode) so we
		# can capture every sample.
		from .training import (_sample_initial_state, _euler_to_quat_xyz,  # type: ignore
		                       apply_disturbance)
		init_q, init_omega = _sample_initial_state(
			ep_rng,
			cfg.max_initial_tilt_rad,
			cfg.max_initial_yaw_rad,
			cfg.max_initial_body_rate,
			cfg.max_initial_yaw_rate,
		)
		sim.reset(q=list(init_q), omega=list(init_omega))
		# CALIBRATE ON THE PLANT WE FLY (10/08/2026). The fitter used a CLEAN sim
		# while every run flies a disturbance, so the ladder never saw the sustained
		# bias the controller spends its whole settled window fighting — the very
		# regime the `steady` metric measures. Mirrors training.py's per-episode
		# arming (apply_disturbance after the IC draw, clear otherwise) so the
		# calibration distribution comes from the SAME weather as the rollouts.
		_dist = getattr(cfg, "disturbance", None)
		if _dist is not None:
			apply_disturbance(sim, _dist, ep_rng)
		else:
			sim.clear_disturbance()
		if geometry is None:
			pid.reset()
		_ep_coll = _hover_pwm
		if _stage1_cal and _ec_translation:
			sim.set_vertical_state(
				float(ep_rng.uniform(-1.0, 1.0) * getattr(cfg, "max_initial_alt_offset_m", 0.0)),
				float(ep_rng.uniform(-1.0, 1.0) * getattr(cfg, "max_initial_vz", 0.0)))
			# The commanded collective VARIES per episode in a real run, so it must
			# vary here too — a constant would leave its ladder degenerate (span 0)
			# and the feature would carry no information at all.
			_jit = float(getattr(cfg, "collective_cmd_jitter", 0.0))
			_ep_coll = float(_hover_pwm * (1.0 + ep_rng.uniform(-_jit, _jit))) if _jit > 0 else _hover_pwm
		if feat_ctl is not None:
			feat_ctl.reset()   # zero the integral accumulators per episode
			if _stage1_cal and _ec_translation:
				feat_ctl.set_collective_anchor(_ep_coll)
		target = (0.0, 0.0, 0.0)
		for _ in range(cfg.steps_per_episode):
			gyro, accel = sim.read_imu()
			q = sim.quaternion
			pwm = (driver.step(list(q), list(gyro), list(target)) if geometry is not None
			       else pid.step(q, gyro, target))
			# Record base-9 sensor samples (unchanged).
			samples_per_feature[0].append(float(gyro[0]))
			samples_per_feature[1].append(float(gyro[1]))
			samples_per_feature[2].append(float(gyro[2]))
			samples_per_feature[3].append(float(accel[0]))
			samples_per_feature[4].append(float(accel[1]))
			samples_per_feature[5].append(float(accel[2]))
			samples_per_feature[6].append(float(target[0]))
			samples_per_feature[7].append(float(target[1]))
			samples_per_feature[8].append(float(target[2]))
			# H2 extras: drive the feature controller (same gyro/accel/target, in
			# order, with per-episode reset) so its integral state matches a real
			# rollout, then read indices [9, nf) from the getter.
			if feat_ctl is not None:
				if _stage1_cal and _ec_translation:
					feat_ctl.set_vertical_obs(_ep_coll,
					                          _target_alt - sim.altitude,
					                          sim.vertical_velocity)
				feat_ctl.step(list(gyro), list(accel), list(target))
				feats = feat_ctl.get_last_feature_vector()
				for k in range(NUM_FEATURES, nf):
					samples_per_feature[k].append(float(feats[k]))
			if geometry is not None:
				sim.step_n(list(pwm))
			else:
				sim.step(list(pwm))
			if sim.is_unstable():
				break

	# Now derive thresholds per feature (base 9 + enabled H2 extras).
	bpf = spec.bits_per_feature
	thresholds = []
	for f in range(nf):
		# STUDENT STATES (option A, 10/08/2026). The fitter rolls out PID — a BETTER
		# controller than the student — so the ladder is fitted on a distribution the
		# student never visits: DAgger covariate shift, but in the INPUT
		# REPRESENTATION, where no amount of training can repair it. `extra_samples`
		# carries per-feature values collected from a real student rollout; they are
		# CONCATENATED with the teacher's rather than replacing them, so the ladder
		# covers both the recovery the teacher demonstrates and the excursions the
		# student actually makes.
		_samples = samples_per_feature[f]
		if extra_samples is not None and f < len(extra_samples) and extra_samples[f]:
			_samples = list(_samples) + list(extra_samples[f])
		arr = np.array(_samples, dtype=float)
		if arr.size == 0:
			# Feature never observed (constant target?). Fall back to [-1, 1] linear.
			arr = np.array([-1.0, 1.0])
		if method == "quantile":
			# Uniform percentiles 1/(bpf+1)..bpf/(bpf+1)
			# COVERAGE MARGIN (option C, 10/08/2026). The default outer quantiles are
			# 1/(b+1) and b/(b+1) — with b=8 that is 0.111/0.889, so ~22% of the
			# operating distribution falls OUTSIDE the ladder by construction and
			# saturates to an all-0/all-1 code. That is survivable for the settled
			# window and expensive for the transient, where a saturated encoder is
			# blind exactly when the controller is furthest from target (measured:
			# calib=5deg lost stable as well as steady, 2/2 seeds). outer_quantile
			# reaches further into the tails: 0.02 spans [0.02, 0.98]. None keeps the
			# legacy positions, so every flown number stays reproducible.
			if outer_quantile is not None:
				lo_q = float(outer_quantile)
				qs = np.linspace(lo_q, 1.0 - lo_q, bpf)
			else:
				qs = np.linspace(1.0 / (bpf + 1), bpf / (bpf + 1), bpf)
			# E3 gamma warp: pull quantile POSITIONS toward 0.5 (the median) with
			# |2q-1|^gamma, gamma>1 → threshold VALUES cluster near the feature's
			# hover region → finer decode where the controller actually settles.
			# gamma=1.0 is the exact identity (parity anchor).
			gamma = getattr(spec, "threshold_gamma", 1.0)
			if gamma and gamma != 1.0:
				qs = 0.5 + np.sign(qs - 0.5) * 0.5 * np.abs(2.0 * qs - 1.0) ** gamma
			ts = np.quantile(arr, qs)
		elif method == "linear":
			lo, hi = float(arr.min()), float(arr.max())
			# If lo == hi (constant feature), spread by ±1 around it
			if hi - lo < 1e-9:
				lo, hi = lo - 1.0, hi + 1.0
			ts = np.linspace(lo, hi, bpf, endpoint=False)
		else:
			raise ValueError(f"unknown method: {method!r}")
		thresholds.extend(float(t) for t in ts)
	return thresholds


def random_connectivity(spec: ControllerSpec, seed: int = 0) -> tuple[list[int], list[int]]:
	"""Structured connectivity for a coherent recurrent FSM.

	For the network to behave as one automaton (not N disjoint mini-automata),
	every state neuron and every output neuron must observe the FULL state
	(see DFA argument: the next-state/output of any neuron depends on which
	GLOBAL state we are in). So we FORCE all state bits into each neuron's
	connections, and only the INPUT connections are sampled (the legitimate
	feature-selection / generalization knob the GA later optimizes).

	State layer input space:  [sensor window (K*F*b) | prev_state (2*n_state)].
	    Each neuron: all 2*n_state state bits + (state_bits_per_neuron - 2*n_state)
	    sampled sensor-window bits.
	Output layer input space (Mealy): [current frame (F*b) | new_state (2*n_state)].
	    Each neuron: all 2*n_state state bits + (output_bits_per_neuron - 2*n_state)
	    sampled current-frame bits.
	"""
	rng = np.random.default_rng(seed)
	n_state = spec.state_neurons
	state_bits = n_state  # forced prefix = 1 bit (MSB)/state neuron (08/06/2026; was 2·n_state)
	# 27/06 frame-misalignment fix: ACTUAL feature count (base 9 + obs extras), so the
	# forced state-prefix offsets match the Rust controller's sensor_total/frame_bits.
	nf = spec.num_features()
	sensor_window = spec.input_window_k * nf * spec.bits_per_feature
	sensor_frame = nf * spec.bits_per_feature

	n_state_sampled = spec.state_bits_per_neuron - state_bits
	n_out_sampled = spec.output_bits_per_neuron - state_bits
	if n_state_sampled < 0 or n_out_sampled < 0:
		raise ValueError(
			f"bits_per_neuron must be >= state_neurons ({state_bits}) for full-state "
			f"connectivity: state={spec.state_bits_per_neuron}, output={spec.output_bits_per_neuron}"
		)

	# State layer: state bits live at [sensor_window, sensor_window+state_bits).
	state_state_idx = list(range(sensor_window, sensor_window + state_bits))
	state_conn: list[int] = []
	for _ in range(n_state):
		sampled = (rng.choice(sensor_window, size=min(n_state_sampled, sensor_window), replace=False).tolist()
		           if n_state_sampled > 0 else [])
		state_conn.extend(state_state_idx + [int(x) for x in sampled])

	# Output layer (Mealy): state bits live at [sensor_frame, sensor_frame+state_bits).
	out_state_idx = list(range(sensor_frame, sensor_frame + state_bits))
	num_output_neurons = spec.num_motors * spec.levels_per_motor
	output_conn: list[int] = []
	for _ in range(num_output_neurons):
		sampled = (rng.choice(sensor_frame, size=min(n_out_sampled, sensor_frame), replace=False).tolist()
		           if n_out_sampled > 0 else [])
		output_conn.extend(out_state_idx + [int(x) for x in sampled])

	return [int(x) for x in state_conn], [int(x) for x in output_conn]


def collect_student_feature_samples(genome, episode_config, num_episodes: int,
                                    seed: int) -> list[list[float]]:
	"""Roll out a TRAINED STUDENT and harvest the feature values it actually visits.

	The point of option A (10/08/2026). `fit_thresholds_from_pid_rollouts` rolls out
	PID — a better controller than the student — so the ladder is fitted on states
	the student never occupies and saturates on the excursions it does make. This
	returns per-feature sample lists suitable for that function's `extra_samples`,
	closing the loop on the STUDENT's own distribution instead of the teacher's.

	SIZING MATTERS. A quantile fit over ~10 x steps teacher samples ignores a
	handful of outliers — measured, 2 extreme values per feature moved the total
	ladder span 1.00x. Call this with enough episodes that the student's samples are
	the same ORDER as the teacher's, or A is a placebo that looks implemented
	(pinned by tests/test_threshold_student_coverage.py).

	THE CALLER OWES A RETRAIN. Refitting shifts the thermometer, hence the ADDRESS
	function, hence every learned cell — the paper-critical THRESHOLD MISALIGNMENT
	finding. A refit is only valid if the memory is retrained under the new ladder;
	--threshold-refit-from-student re-runs the GRID stage for exactly that reason.

	Uses the SAME plant the student flies (episode_config carries the disturbance),
	and the controller's OWN feature extractor via get_last_feature_vector(), so the
	values are byte-for-byte the ones step() thermometer-encodes — no Python
	re-derivation that could drift from the Rust definition.
	"""
	import numpy as np
	from .training import _sample_initial_state, apply_disturbance

	ctl = build_controller(genome)
	sim = AttitudeSim()
	rng = np.random.default_rng(seed)
	nf = genome.spec.num_features()
	samples: list[list[float]] = [[] for _ in range(nf)]
	target = [0.0, 0.0, 0.0]

	for _ in range(num_episodes):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		init_q, init_omega = _sample_initial_state(
			ep_rng,
			episode_config.max_initial_tilt_rad,
			episode_config.max_initial_yaw_rad,
			episode_config.max_initial_body_rate,
			episode_config.max_initial_yaw_rate,
		)
		sim.reset(q=list(init_q), omega=list(init_omega))
		dist = getattr(episode_config, "disturbance", None)
		if dist is not None:
			apply_disturbance(sim, dist, ep_rng)
		else:
			sim.clear_disturbance()
		ctl.reset()
		for _step in range(episode_config.steps_per_episode):
			gyro, accel = sim.read_imu()
			pwm = ctl.step(list(gyro), list(accel), target)
			feats = ctl.get_last_feature_vector()
			for f in range(min(nf, len(feats))):
				samples[f].append(float(feats[f]))
			sim.step(list(pwm))
	return samples


def _stage1_train_kwargs(ec) -> dict:
	"""SCOPE C STAGE 1: the vertical-channel fields RewardGatedConfigPacked needs
	so the TRAINING rollout flies the same plant the scorer does.

	Empty when translation is off, which is what keeps every pre-stage-1 run
	bit-identical (the Rust defaults are all inert). af_mass comes from the
	airframe: it is a PLANT parameter, randomized per episode by mass_jitter,
	and never a feature."""
	if not getattr(ec, "translation", False):
		return {}
	af = getattr(ec, "airframe", None)
	if af is None:
		raise ValueError("EpisodeConfig.translation requires an airframe — mass is a "
		                 "plant parameter and the synthetic default has none.")
	return dict(
		translation=True,
		af_mass=float(af.mass),
		mass_jitter=float(getattr(ec, "mass_jitter", 0.0)),
		alt_offset=float(getattr(ec, "max_initial_alt_offset_m", 0.0)),
		init_vz=float(getattr(ec, "max_initial_vz", 0.0)),
		collective_jitter=float(getattr(ec, "collective_cmd_jitter", 0.0)),
		target_altitude=float(getattr(ec, "target_altitude", 0.0)),
		# STAGE 2 (14/08): xy_offset doubles as the enable — 0.0 draws NOTHING
		# in the trainer, keeping every stage-1 run's rng sequence intact.
		xy_offset=float(getattr(ec, "max_initial_xy_offset_m", 0.0)),
		lambda_pos=float(getattr(ec, "lambda_pos", 0.0)),
	)


def build_controller(genome: ControllerGenome) -> WnnController:
	"""Instantiate a Rust WnnController from a ControllerGenome and apply
	all learned cells. The Rust controller takes connectivity at
	construction time and cell writes via the per-layer write methods."""
	spec = genome.spec
	c = WnnController(
		num_motors=spec.num_motors,
		levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature,
		input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons,
		state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron,
		thresholds=genome.thresholds,
		state_connections=genome.state_connections,
		output_connections=genome.output_connections,
		delta_control=spec.delta_control,
		delta_max=spec.delta_max,
		delta_leak=spec.delta_leak, delta_gamma=getattr(spec, 'delta_gamma', 1.0),
		obs_tilt_p=spec.obs_tilt_p,
		obs_tilt_i=spec.obs_tilt_i,
		obs_peraxis_p=spec.obs_peraxis_p,
		obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i,
		obs_collective_cmd=spec.obs_collective_cmd, obs_alt_err=spec.obs_alt_err, obs_vz=spec.obs_vz,
			obs_pos_err_xy=getattr(spec, 'obs_pos_err_xy', False), obs_vel_xy=getattr(spec, 'obs_vel_xy', False),
		output_full_window=getattr(spec, 'output_full_window', False),
		frame_stride=int(getattr(spec, 'frame_stride', 1)),
			dhat_b=(list(spec.dhat_b) if spec.dhat_b is not None else None), dhat_l_gain=spec.dhat_l_gain, dhat_ff=getattr(spec, 'dhat_ff', False), dhat_ff_clamp=getattr(spec, 'dhat_ff_clamp', 0.30), dt=spec.dt,
		integral_leak=spec.integral_leak,
		integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
		action_repeat=spec.action_repeat,
		memory_mode=spec.memory_mode_int(),
		output_decode=spec.output_decode_int(),
	)
	if genome.cells_handle is not None:
		# Bulk Rust ingress: same canonicalising write path as the loops below,
		# zero per-cell Python objects.
		c.load_cells_handle(genome.cells_handle)
	for (n, addr, v) in genome.state_cells:
		c.write_state_cell(n, addr, v)
	for (n, addr, v) in genome.output_cells:
		c.write_output_cell(n, addr, v)
	return c


# ---------------------------------------------------------------------------
# Drone adapter: the ONLY place where drone vocabulary (motors, levels, sensor
# window) meets the domain-free RecurrentArchGenome. Keeps the generic genome
# reusable by any two-layer recurrent RAM arch (see recurrent_genome.py).
# ---------------------------------------------------------------------------

def arch_shape_from_spec(spec: ControllerSpec) -> "RecurrentArchShape":
	"""Project the drone ControllerSpec onto the genome's fixed structural
	constants: motors/levels → output count granularity, K·F·b → input spaces."""
	from .recurrent_genome import RecurrentArchShape
	# CRITICAL (27/06/2026 frame-misalignment fix): the input spaces MUST use the
	# controller's ACTUAL feature count (spec.num_features() = 9 base + enabled H2
	# obs extras), NOT the base-9 NUM_FEATURES. The Rust controller places the
	# recurrent state at input index sensor_total = input_window_k·num_features·bpf
	# (state layer) and frame_bits = num_features·bpf (output layer). to_connections()
	# builds the forced full-state prefix as range(input_space, input_space+prefix),
	# so input_space must EQUAL those offsets or the prefix lands inside the sensor
	# region — the controller never observes its own memory ⇒ memoryless ⇒ brittle.
	# With NUM_FEATURES=9 hardcoded, every obs-feature run (yaw-anchor, tilt-i, …) was
	# mis-wired (e.g. 10 feat ⇒ state at 320 but prefix targeted 288). obs-OFF (9 feat)
	# happened to align, which is why S16 was robust. See project_controller_frame_misalignment.
	nf = spec.num_features()
	# Output-neurogenesis granularity: normally num_motors (one PWM level = num_motors
	# output neurons). BINARY decodes each motor's levels via antagonist E/I halves
	# (levels 0..L/2 excitatory | L/2..L inhibitory, decoded 0.5+(ΣE−ΣI)/L — cell_mode.rs),
	# which needs an EVEN levels_per_motor for a symmetric split (odd L drifts the neutral
	# off 0.5). Double the quantum under BINARY so neurogenesis only ever steps by whole
	# EVEN level counts → the Rust WnnController's even-levels invariant always holds.
	# 04/08/2026: keyed on the DECODE, not the memory mode. The even-levels
	# invariant belongs to the antagonist E/I split, and until 03/08 antagonist was
	# welded to BINARY so "mode == BINARY" was an accurate proxy. Making
	# --output-decode orthogonal (ABI 21) broke that proxy: QUAD+antagonist got q=4,
	# so neurogenesis stepped by 4, produced on=92 → levels=23 (odd), and the Rust
	# controller rejected the shape → every affected genome silently fell back to the
	# ~3× slower Python reference path mid-run (caught on P4a, 04/08).
	# BINARY→antagonist and QUAD-without-flag→cumulative both resolve exactly as
	# before, so every prior cohort reproduces.
	q = spec.num_motors * 2 if spec.resolved_output_decode() == "antagonist" else spec.num_motors
	return RecurrentArchShape(
		# 08/06/2026: recurrent state output is now 1 bit/neuron (the QSR MSB =
		# fired/not), NOT 2 (the LSB was training-confidence, semantically wrong to
		# feed back). Halves the forced prefix (sn, not 2·sn). Must match the Rust
		# controller's state_bits_in = state_neurons.
		prefix_factor=1,  # state output = 1 bit (MSB) per state neuron
		state_input_space=spec.input_window_k * nf * spec.bits_per_feature,
		# ARM D: with the flag on the output layer samples the SAME K-frame window
		# the state layer reads; legacy = one frame (the Mealy current-observation
		# design). This is the ONLY line that widens the genome's legal space —
		# the Rust controller and shader read whatever indices the genome carries.
		output_input_space=(spec.input_window_k if getattr(spec, "output_full_window", False) else 1)
		                   * nf * spec.bits_per_feature,
		output_quantum=q,
	)


def spec_from_arch(genome: "RecurrentArchGenome", base: ControllerSpec) -> ControllerSpec:
	"""Rebuild a concrete ControllerSpec from a genome's evolved shape, inheriting
	the fixed environment params (motors, sensor encoding, delta config) from
	`base`. `levels_per_motor` is DERIVED from output_neurons / num_motors."""
	return ControllerSpec(
		num_motors=base.num_motors,
		levels_per_motor=genome.output_neurons // base.num_motors,
		bits_per_feature=base.bits_per_feature,
		input_window_k=base.input_window_k,
		state_neurons=genome.state_neurons,
		state_bits_per_neuron=genome.state_bits_per_neuron,
		output_bits_per_neuron=genome.output_bits_per_neuron,
		delta_control=base.delta_control,
		delta_max=base.delta_max,
		delta_leak=base.delta_leak, delta_gamma=getattr(base, 'delta_gamma', 1.0),
		obs_tilt_p=base.obs_tilt_p,
		obs_tilt_i=base.obs_tilt_i,
		obs_peraxis_p=base.obs_peraxis_p,
		obs_peraxis_i=base.obs_peraxis_i, obs_peraxis_yaw=base.obs_peraxis_yaw, obs_pwm=base.obs_pwm, obs_yaw_err=base.obs_yaw_err, obs_yaw_err_i=base.obs_yaw_err_i,
		obs_collective_cmd=base.obs_collective_cmd, obs_alt_err=base.obs_alt_err, obs_vz=base.obs_vz,
		obs_pos_err_xy=getattr(base, 'obs_pos_err_xy', False), obs_vel_xy=getattr(base, 'obs_vel_xy', False),
		output_full_window=getattr(base, 'output_full_window', False),
		frame_stride=int(getattr(base, 'frame_stride', 1)),
		dhat_b=base.dhat_b, dhat_l_gain=base.dhat_l_gain,
		dhat_ff=base.dhat_ff, dhat_ff_clamp=base.dhat_ff_clamp, dt=base.dt,
		integral_leak=base.integral_leak,
		integral_scale=base.integral_scale, decouple_outputs=base.decouple_outputs,
		threshold_gamma=base.threshold_gamma,
		action_repeat=base.action_repeat,
		memory_mode=base.memory_mode,
		output_decode=base.output_decode,
	)


def controller_genome_from_arch(
	genome: "RecurrentArchGenome", base: ControllerSpec, thresholds: list[float],
	state_cells: list | None = None, output_cells: list | None = None,
) -> ControllerGenome:
	"""Materialize a generic arch genome into a concrete, buildable ControllerGenome
	(connectivity + thresholds + optional cells). Explicit cell args win; otherwise
	the genome's own MemoryPayload (if any) is used — so a unified genome carrying
	evolved/Lamarckian cells builds directly."""
	sc, oc = genome.to_connections()
	cells_handle = None
	if state_cells is None and output_cells is None and genome.cells is not None:
		# Stage B: hand the Rust handle through — build_controller bulk-loads it.
		# The old path materialised to_triples() here, one 3-int tuple per cell
		# per genome per score call (HOT in the MEMORY stage, where phased_ga
		# routes every generation through score_genomes).
		cells_handle = genome.cells
	return ControllerGenome(
		spec=spec_from_arch(genome, base),
		thresholds=thresholds,
		state_connections=sc,
		output_connections=oc,
		state_cells=state_cells or [],
		output_cells=output_cells or [],
		cells_handle=cells_handle,
	)


class ControllerEvaluator:
	"""Evaluate a controller genome over a held-out episode set.

	Mirrors the IDSEvaluator interface used by the GA/grid-search phases:
	  - `__init__(spec, num_episodes, seed)`: prepare the evaluator with
	    the architecture spec and the held-out episode plan.
	  - `evaluate(genome)`: returns (fitness_scalar, metrics_dict).
	  - `validate(genome)`: same as evaluate but for the final validation
	    checkpoint — uses a larger episode count for tighter statistics.

	The fitness scalar is `mean_reward`, which is negative (since reward =
	-attitude_error² typically). The GA should maximize fitness, so the
	convention matches: higher = better.
	"""

	def __init__(
		self,
		spec: ControllerSpec,
		num_eval_episodes: int = 30,
		num_validate_episodes: int = 100,
		seed: int = 0,
		episode_config: Optional[EpisodeConfig] = None,
		thresholds: Optional[list[float]] = None,
		rg_config=None,
		max_train_workers: int = 1,
		max_eval_workers_gpu: bool = True,
		fitness_seeds: int = 1,
		num_eval_folds: int = 1,
	):
		self.spec = spec
		self.num_eval = num_eval_episodes
		self.num_validate = num_validate_episodes
		self.seed = seed
		# K-fold scoring (added 30/05/2026 post-Plan-A-v1-overfit diagnosis).
		# num_eval_folds=1 reproduces legacy single-pool behavior. With K>1, the
		# evaluator pre-generates K deterministic episode-pool seeds (from `seed`)
		# and rotates through them per evaluate_batch call. _active_score_seed is
		# what the GPU/CPU scoring paths read for episode IC sampling — defaults
		# to self.seed, gets overwritten to fold_seeds[k] when a fold is active.
		self.num_eval_folds = max(1, num_eval_folds)
		self._fold_seeds = [fold_pool_seed(seed, k) for k in range(self.num_eval_folds)]
		self._fold_counter = 0
		self._active_score_seed = seed
		self.episode_config = episode_config or EpisodeConfig(
			dt=0.001, steps_per_episode=2000,
			max_initial_tilt_rad=math.radians(30.0),
			max_initial_yaw_rad=math.radians(30.0),
			max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
		)
		# Each evaluator owns its own AttitudeSim (cheap to construct;
		# stateless across episodes after reset).
		self._sim = AttitudeSim()
		# H4 axis curriculum: when set to a generation count, evaluate_batch ramps
		# the active axes roll → roll+pitch → all over those gens (training IC +
		# in-search eval IC). None ⇒ full 3-axis always. The HELD-OUT report builds
		# its OWN evaluator (curriculum None) so it always measures the full problem.
		self.axis_curriculum_gens: Optional[int] = None
		# H4 (7-phase): a FIXED active-axis mask for this evaluator, used by the
		# combinatorial curriculum (singles→pairs→triple). When set it overrides
		# the per-gen ramp entirely — every generation in the sub-phase trains +
		# scores on exactly these axes. None ⇒ ramp (legacy) or full 3-axis.
		self.fixed_axes: Optional[tuple] = None
		self._cur_axes: tuple = (True, True, True)
		# Generation counter for the curriculum — advanced once per GA batch-eval
		# (the Lamarckian path drops the framework's generation arg, so we count).
		self._generation: int = -1
		# GA-path config: shared (PID-fit) thresholds held across all genomes,
		# and the inner-loop trainer config (carries target_source = C1 "pid" /
		# C2 "student"). Lazily filled if not supplied.
		self.thresholds = thresholds
		self.rg_config = rg_config
		# CPU across-genome parallelism for the inner training step. Each genome
		# has its own WnnController (independent), so threads parallelise the
		# GIL-releasing Rust solver. Capped to coexist with the IDS worker.
		self.max_train_workers = max_train_workers
		# Use the GPU-batched Metal kernel for the closed-loop SCORING step
		# (training stays CPU). Falls back to CPU if Metal is unavailable.
		# TOGGLE (09/07/2026): WNN_CONTROLLER_GPU_EVAL=0 forces CPU scoring — needed
		# when the IDS worker owns the GPU (its non-preemptible 46M-row kernels starve
		# the controller's command buffer for tens of minutes). env=0 overrides the
		# constructor arg; default (unset/1) keeps GPU scoring.
		# 3-way: 0/false → CPU (rayon), 1/unset → GPU (default), 2/hybrid → run BOTH
		# concurrently over shape-groups (GPU worker + CPU rayon pull from a shared
		# queue; if the GPU stalls on its last group the CPU steals it). Hybrid uses
		# the GPU in the gaps between the worker's kernels and CPU the rest.
		_gpu_eval_env = os.environ.get("WNN_CONTROLLER_GPU_EVAL", "1").strip().lower()
		if _gpu_eval_env in ("2", "hybrid", "both"):
			self.eval_mode = "hybrid"
		elif _gpu_eval_env in ("0", "false", "off", "no"):
			self.eval_mode = "cpu"
		else:
			self.eval_mode = "gpu"
		# Back-compat flag (score_population reads it): True for gpu OR hybrid.
		self.max_eval_workers_gpu = max_eval_workers_gpu and self.eval_mode != "cpu"
		# Multi-seed genome fitness (A): the inner loop is chaotic, so the SAME
		# connectivity yields different controllers per training seed. Averaging
		# the closed-loop score over K independent train+score seeds gives the GA
		# a stable estimate to climb (variance ÷√K) instead of selecting noise.
		self.fitness_seeds = fitness_seeds

	def advance_generation(self) -> None:
		"""Bump the curriculum generation (call once per GA generation's batch eval)."""
		self._generation += 1

	def _active_axes(self, generation) -> tuple:
		"""Per-gen axis mask for the curriculum: 1st third roll, 2nd +pitch, last all."""
		# 7-phase combinatorial curriculum: a fixed mask for the whole sub-phase
		# takes precedence over the per-gen ramp (and over full-3-axis).
		if self.fixed_axes is not None:
			return self.fixed_axes
		g = self.axis_curriculum_gens
		if not g or generation is None:
			return (True, True, True)
		third = max(1, g // 3)
		if generation < third:
			return (True, False, False)
		if generation < 2 * third:
			return (True, True, False)
		return (True, True, True)

	def evaluate(self, genome: ControllerGenome) -> tuple[float, dict]:
		"""Returns (fitness, metrics) over num_eval episodes."""
		controller = build_controller(genome)
		action_fn = make_wnn_action_fn(controller)
		mean_reward, metrics = fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_eval, seed=self.seed,
		)
		return mean_reward, metrics

	def validate(self, genome: ControllerGenome) -> tuple[float, dict]:
		"""Higher-episode-count validation pass for the final checkpoint."""
		controller = build_controller(genome)
		action_fn = make_wnn_action_fn(controller)
		mean_reward, metrics = fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_validate, seed=self.seed + 1_000_000,
		)
		return mean_reward, metrics

	def train_and_evaluate(
		self,
		thresholds: list[float],
		state_connections: list[int],
		output_connections: list[int],
		dagger_config=None,
	) -> tuple[float, dict]:
		"""DAGGER-train a controller with the given connectivity, then score it.

		The controller analog of training+evaluating an IDS genome: the GA
		supplies connectivity (+ PID-fit thresholds); DAGGER fills the QSR
		cells by rolling out the student and labelling with the PID teacher;
		we return the closed-loop fitness. This is what
		`architecture_type='controller'` grid_search/ga flows call per genome.

		The final score is recomputed with the evaluator's own seed so it is
		directly comparable to `evaluate()` / `evaluate_pid_baseline()` (same
		held-out episode set), independent of DAGGER's internal eval seed.
		"""
		from .dagger import DaggerConfig, train_dagger

		cfg = dagger_config or DaggerConfig(
			seed=self.seed,
			eval_episodes=self.num_eval,
			episode_config=self.episode_config,
		)
		controller, dagger_stats = train_dagger(
			self.spec, thresholds, state_connections, output_connections, cfg,
		)
		# Score the trained controller on the evaluator's episode set (fresh
		# recurrent state, matching build_controller's fresh-state convention).
		controller.reset()
		action_fn = make_wnn_action_fn(controller)
		fitness, metrics = fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_eval, seed=self.seed,
		)
		metrics["dagger"] = {
			k: dagger_stats[k] for k in (
				"iter_fitness", "iter_mean_err_deg", "iter_stable_rate",
				"iter_beta", "best_iter", "best_fitness", "final_fitness",
				"train_steps",
			)
		}
		return fitness, metrics

	def evaluate_pid_baseline(self) -> tuple[float, dict]:
		"""Run the PID baseline over the same episode set for direct comparison.
		Returns (mean_reward, metrics) — same shape as evaluate()."""
		pid = AttitudePID(AttitudePIDConfig())
		action_fn = make_pid_action_fn(pid)
		return fitness_function(
			action_fn, self._sim, self.episode_config,
			num_episodes=self.num_eval, seed=self.seed,
		)

	# ------------------------------------------------------------------
	# GA-facing batch interface (consumed by ControllerGAStrategy.optimize via
	# evaluate_fn / batch_evaluate_fn). A "genome" here is duck-typed: anything
	# with .state_connections / .output_connections (a FiniteStateGenome). The
	# shared self.thresholds + self.rg_config make the genome carry ONLY the
	# evolvable connectivity — cells are produced per-genome by the inner loop.
	# ------------------------------------------------------------------

	# ------------------------------------------------------------------
	# Compatibility shims so the shared Experiment orchestration (built around
	# the IDS ClusterGenome model) can construct + run a controller flow as a
	# first-class peer. These are only read by the orchestration's bookkeeping;
	# the controller's actual sizing comes from self.spec. vocab_size = the
	# output-neuron count (a benign positive int); there are no clusters/parts.
	# ------------------------------------------------------------------

	@property
	def vocab_size(self) -> int:
		return self.spec.num_motors * self.spec.levels_per_motor

	@property
	def num_clusters(self) -> int:
		return self.vocab_size

	@property
	def num_parts(self) -> int:
		return 1

	@property
	def total_input_bits(self) -> int:
		# State-layer input size: sensor window + recurrent-state bits. Benign for
		# the orchestration's bookkeeping; the controller's real wiring is in the genome.
		# Uses ACTUAL feature count (base 9 + obs extras) — 27/06 frame-misalignment fix.
		return (self.spec.input_window_k * self.spec.num_features() * self.spec.bits_per_feature
		        + self.spec.state_neurons)  # 1 bit (MSB)/state neuron (was 2·)

	def _ensure_ga_ready(self):
		from .reward_gated import RewardGatedConfig
		if self.thresholds is None:
			self.thresholds = fit_thresholds_from_pid_rollouts(self.spec, num_episodes=10, seed=self.seed)
		if self.rg_config is None:
			self.rg_config = RewardGatedConfig(seed=self.seed, episode_config=self.episode_config)

	def _materialize(self, genome) -> tuple:
		"""(spec, state_connections, output_connections) for either genome type.

		Fixed-shape FiniteStateGenome carries .state_connections + shares self.spec.
		Variable-shape RecurrentArchGenome rebuilds its own spec (via spec_from_arch)
		and connectivity (via to_connections) — so one evaluator drives both."""
		if hasattr(genome, "to_connections") and hasattr(genome, "shape"):
			sc, oc = genome.to_connections()
			return spec_from_arch(genome, self.spec), sc, oc
		return self.spec, genome.state_connections, genome.output_connections

	def _shape_key(self, genome) -> tuple:
		"""GPU-batch grouping key — the dims score_controllers_metal applies
		uniformly. Fixed-shape genomes all collapse to one key (= self.spec)."""
		spec, _sc, _oc = self._materialize(genome)
		return (spec.num_motors, spec.levels_per_motor, spec.state_neurons,
		        spec.state_bits_per_neuron, spec.output_bits_per_neuron)

	def _train_genome(self, genome, seed: int):
		"""Inner-train one genome's cells (C1 or C2 per rg_config.target_source)
		with the given training seed. Returns (WnnController, stats).

		Lamarckian warm-start: if the genome carries inherited cells (genome.cells),
		training starts FROM them (write_back stamps them; 4a remap keeps them valid
		across arch genesis). GA/TS genomes carry no cells → empty start, unchanged.

		Rust fast-path (29/05/2026): if `WNN_RUST_DAGGER=1` and the accelerator
		exposes `dagger_train_inplace`, run the full reward-gated training loop
		natively in Rust (one Python↔Rust crossing per genome, vs ~288K per-step
		crossings in the Python path). 30-100× speedup on the M4 Max for the GA
		pop-build hot path. Algorithmically equivalent to the Python reference;
		RNG values differ bit-for-bit from numpy (Rust uses SmallRng) but produce
		statistically equivalent training distributions. Falls through to Python
		on any error to preserve correctness."""
		spec, sc, oc = self._materialize(genome)
		init_s = init_o = None
		cells = getattr(genome, "cells", None)
		if cells is not None:
			init_s, init_o = cells.to_triples()
		return self._train_core(spec, sc, oc, init_s, init_o, seed)

	def _train_core(self, spec, sc, oc, init_s, init_o, seed):
		"""Train ONE controller from explicit init cells (the shared core of
		_train_genome and the K-fold accumulate loop). init_s/init_o are cell
		triples to warm-start from (None = empty memory). Rust fast-path with the
		Python reward_gated_train reference as fallback."""
		if _rust_dagger_enabled():
			try:
				return self._train_genome_rust(spec, sc, oc, init_s, init_o, seed)
			except Exception as e:
				# Loud (once-per-process): the Python fallback is ~3× slower AND
				# silently drops jerk/mono unless those are plumbed — so a silent
				# degrade here is exactly the trap we just fixed. Make it visible.
				if not getattr(self, "_rust_dagger_warned", False):
					import sys
					print(f"[ControllerEvaluator] ⚠️ Rust DAGGER FELL BACK to the slow Python "
					      f"reference path: {e}. (Set WNN_RUST_DAGGER=0 to silence if intentional.)",
					      file=sys.stderr, flush=True)
					self._rust_dagger_warned = True

		from .reward_gated import reward_gated_train
		import copy
		rg = copy.copy(self.rg_config)
		rg.seed = seed
		rg.progress = False
		return reward_gated_train(spec, self.thresholds, sc, oc, rg,
		                          init_state_cells=init_s, init_output_cells=init_o)

	def _train_genomes_rust_batched(self, genomes, tasks, init_override=None):
		"""B.5-var batched fast-path. ONE Rust call for the whole task list;
		per-genome shape vectors let Rayon parallelize across genomes regardless
		of whether their shapes match.

		Run-level dims (num_motors / levels_per_motor / bits_per_feature /
		input_window_k) are shared scalars — they don't change across genomes
		in any GA dimension. Per-genome shape (state_neurons,
		state_bits_per_neuron, output_bits_per_neuron) flows in as Vec<usize>;
		each Rayon task constructs its own WnnController with its own shape.

		Pre-29/05/2026 this method grouped by shape and called Rust once per
		group, which for variable-shape GAs (Neurons, Bits) produced many
		small calls (~30-50 groups of 4-7 candidates each), defeating the
		batching win. Now a single call handles the entire task list.

		Returns list[(controller, stats_dict)] in task-order.
		"""
		from wnn.control import _accel as ra

		# Materialize per-task: (spec, sc, oc, init_s, init_o, seed).
		mats = []
		for ti, (gi, seed_list) in enumerate(tasks):
			spec, sc, oc = self._materialize(genomes[gi])
			if init_override is not None:
				init_h = init_override[ti]
			else:
				cells = getattr(genomes[gi], "cells", None)
				init_h = cells if cells is not None else ra.GenomeCells()
			mats.append((spec, sc, oc, init_h, [int(s) for s in seed_list]))

		# Sanity: TRULY run-level dims (num_motors / bits_per_feature /
		# input_window_k) must agree. levels_per_motor IS per-genome because
		# spec_from_arch derives it from output_neurons // num_motors, and the
		# Neurons GA mutates output_neurons (29/05/2026 fix).
		first_spec = mats[0][0]
		for (s, *_) in mats[1:]:
			assert (s.num_motors, s.bits_per_feature, s.input_window_k) == (
				first_spec.num_motors, first_spec.bits_per_feature,
				first_spec.input_window_k), (
				"num_motors / bits_per_feature / input_window_k must be uniform "
				"across the batch; levels_per_motor / state_neurons / state_bits "
				"/ output_bits may vary (and are passed per-genome).")

		# Build packed config ONCE.
		rg = self.rg_config
		# W2 disturbances → the in-search training rollouts (train-under-weather).
		(dist_en, dist_tb, dist_gs, dist_gtc, dist_ma,
		 dist_gys, dist_gbw, dist_acs,
		 dist_dp, dist_dls, dist_ods, dist_tsj) = _dist_packed_fields(rg)
		cfg = ra.RewardGatedConfigPacked(
			num_rounds=rg.num_rounds, episodes_per_round=rg.episodes_per_round,
			steps_per_episode=rg.steps_per_episode, bptt_window=rg.bptt_window,
			topk_per_neuron=rg.topk_per_neuron, protect_learned=rg.protect_learned,
			gate_mode=0 if rg.gate_mode == "improvement" else 1,
			gate_use_best=rg.gate_use_best, gate_window=rg.gate_window,
			gate_quantile=rg.gate_quantile, gate_running=rg.gate_running,
			target_source=0 if rg.target_source == "pid" else 1,
			teacher=_TEACHER_IDS[getattr(rg, "teacher", "pid")],
			teacher_schedule=[_TEACHER_IDS[t] for t in getattr(rg, "teacher_schedule", [])],
			teacher_blend=[_TEACHER_IDS[t] for t in getattr(rg, "teacher_blend", [])],
			keep_best_checkpoint=rg.keep_best_checkpoint,
			explore_eps=rg.explore_eps, explore_scale=rg.explore_scale,
			curriculum=rg.curriculum, easy_tilt_deg=rg.easy_tilt_deg,
			full_tilt_deg=rg.full_tilt_deg,
			dt=rg.episode_config.dt,
			max_initial_yaw_rad=rg.episode_config.max_initial_yaw_rad,
			max_initial_body_rate=rg.episode_config.max_initial_body_rate,
			max_initial_yaw_rate=rg.episode_config.max_initial_yaw_rate,
			eval_episodes=rg.eval_episodes,
			split_tau=rg.split_tau, split_clean_gain=rg.split_clean_gain,
			split_accum_corr=rg.split_accum_corr, split_max_rounds=rg.split_max_rounds,
			split_k_start=rg.split_k_start, split_coarse_target=rg.split_coarse_target,
			split_selective_output=rg.split_selective_output,
			active_roll=self._cur_axes[0], active_pitch=self._cur_axes[1], active_yaw=self._cur_axes[2],
			dist_enabled=dist_en, dist_tau_bias=dist_tb,
			dist_gust_sigma=dist_gs, dist_gust_tau_c=dist_gtc,
			dist_motor_asym=dist_ma, dist_gyro_sigma=dist_gys,
			dist_gyro_bias_walk=dist_gbw, dist_accel_sigma=dist_acs,
			dist_dropout_prob=dist_dp, dist_dropout_len_steps=dist_dls,
			dist_obs_delay_steps=dist_ods, dist_torque_scale_jitter=dist_tsj,
			expert_drives=getattr(rg, "expert_drives", False),
			write_priority_err=getattr(rg, "write_priority_err", False),
			write_err_floor_deg=getattr(rg, "write_err_floor_deg", 0.0),
			# SCOPE C STAGE 1: the TRAINING rollout must fly the same plant the
			# scorer does, or the vertical features are zeros here and real there
			# (the DOB divergence — the Rust side asserts on the mismatch).
			**_stage1_train_kwargs(rg.episode_config),
		)
		target_rpy = list(rg.target_rpy) if rg.target_rpy is not None else [0.0, 0.0, 0.0]

		# ONE call — Rayon par_iter across the whole batch inside Rust.
		results = ra.dagger_train_batch_inplace(
			num_motors=first_spec.num_motors,
			bits_per_feature=first_spec.bits_per_feature,
			input_window_k=first_spec.input_window_k,
			levels_per_motor_per_genome=       [m[0].levels_per_motor for m in mats],
			state_neurons_per_genome=          [m[0].state_neurons for m in mats],
			state_bits_per_neuron_per_genome=  [m[0].state_bits_per_neuron for m in mats],
			output_bits_per_neuron_per_genome= [m[0].output_bits_per_neuron for m in mats],
			thresholds=self.thresholds,
			delta_control=first_spec.delta_control,
			delta_max=first_spec.delta_max,
			delta_leak=first_spec.delta_leak, delta_gamma=getattr(first_spec, 'delta_gamma', 1.0),
			obs_tilt_p=first_spec.obs_tilt_p,
			obs_tilt_i=first_spec.obs_tilt_i,
			obs_peraxis_p=first_spec.obs_peraxis_p,
			obs_peraxis_i=first_spec.obs_peraxis_i, obs_peraxis_yaw=first_spec.obs_peraxis_yaw, obs_pwm=first_spec.obs_pwm, obs_yaw_err=first_spec.obs_yaw_err, obs_yaw_err_i=first_spec.obs_yaw_err_i,
			dhat_b=(list(first_spec.dhat_b) if first_spec.dhat_b is not None else None), dhat_l_gain=first_spec.dhat_l_gain,
			dhat_ff=first_spec.dhat_ff, dhat_ff_clamp=first_spec.dhat_ff_clamp, dt=first_spec.dt,
			integral_leak=first_spec.integral_leak,
			integral_scale=first_spec.integral_scale, decouple_outputs=first_spec.decouple_outputs,
			state_connections_per_genome= [m[1] for m in mats],
			output_connections_per_genome=[m[2] for m in mats],
			# Stage B: warm-start cells as GenomeCells handles. The old form built
			# one 3-int tuple per cell per genome per generation AND re-filtered
			# the u64 range in Python — the handle's addresses are u64 by
			# construction, and Rust memcpys the columns under the GIL.
			init_cells_per_genome=[m[3] for m in mats],
			cfg=cfg, target_rpy=target_rpy,
			fold_seeds=[m[4] for m in mats],
			action_repeat=first_spec.action_repeat,
			memory_mode=first_spec.memory_mode_int(),
			output_decode=first_spec.output_decode_int(),
			# SCOPE C STAGE 1: the trainer builds its own WnnController, so the
			# vertical toggles must reach it too — without them it constructs a
			# 15-feature controller for an 18-feature ladder and the batched path
			# falls back to Python (caught by the 13/08 boundary smoke).
			obs_collective_cmd=first_spec.obs_collective_cmd,
			obs_alt_err=first_spec.obs_alt_err,
			obs_vz=first_spec.obs_vz,
			obs_pos_err_xy=getattr(first_spec, 'obs_pos_err_xy', False),
			obs_vel_xy=getattr(first_spec, 'obs_vel_xy', False),
			output_full_window=getattr(first_spec, 'output_full_window', False),
			frame_stride=int(getattr(first_spec, 'frame_stride', 1)),
		)
		trained = []
		for (controller, ts) in results:
			stats = {
				"iter_fitness":             list(ts.iter_fitness),
				"iter_mean_err_deg":        list(ts.iter_mean_err_deg),
				"iter_stable_rate":         list(ts.iter_stable_rate),
				"iter_tilt_deg":            list(ts.iter_tilt_deg),
				"iter_n_trained":           list(ts.iter_n_trained),
				"iter_cells_written":       list(ts.iter_cells_written),
				"iter_mean_episode_reward": list(ts.iter_mean_episode_reward),
				"iter_motor_jerk_mean":      list(ts.iter_motor_jerk_mean),
				"iter_mono_violations":      list(ts.iter_mono_violations),
				"train_steps":              int(ts.train_steps),
				"split_saturation":         int(ts.split_saturation),
				"split_wish_bits":          list(ts.split_wish_bits),
			}
			trained.append((controller, stats))
		return trained

	def _train_genome_rust(self, spec, state_conns, output_conns, init_s, init_o, seed):
		"""Rust dagger_train_inplace fast-path. Returns (WnnController, stats_dict)
		matching the Python reward_gated_train return shape."""
		from wnn.control import _accel as ra
		# Map the Python RewardGatedConfig → RewardGatedConfigPacked. String
		# enums become integers (0=improvement/pid, 1=quantile/student).
		rg = self.rg_config
		# W2 disturbances → the in-search training rollouts (train-under-weather).
		(dist_en, dist_tb, dist_gs, dist_gtc, dist_ma,
		 dist_gys, dist_gbw, dist_acs,
		 dist_dp, dist_dls, dist_ods, dist_tsj) = _dist_packed_fields(rg)
		cfg = ra.RewardGatedConfigPacked(
			num_rounds=rg.num_rounds,
			episodes_per_round=rg.episodes_per_round,
			steps_per_episode=rg.steps_per_episode,
			bptt_window=rg.bptt_window,
			topk_per_neuron=rg.topk_per_neuron,
			protect_learned=rg.protect_learned,
			gate_mode=0 if rg.gate_mode == "improvement" else 1,
			gate_use_best=rg.gate_use_best,
			gate_window=rg.gate_window,
			gate_quantile=rg.gate_quantile,
			gate_running=rg.gate_running,
			target_source=0 if rg.target_source == "pid" else 1,
			teacher=_TEACHER_IDS[getattr(rg, "teacher", "pid")],
			teacher_schedule=[_TEACHER_IDS[t] for t in getattr(rg, "teacher_schedule", [])],
			teacher_blend=[_TEACHER_IDS[t] for t in getattr(rg, "teacher_blend", [])],
			keep_best_checkpoint=rg.keep_best_checkpoint,
			explore_eps=rg.explore_eps,
			explore_scale=rg.explore_scale,
			curriculum=rg.curriculum,
			easy_tilt_deg=rg.easy_tilt_deg,
			full_tilt_deg=rg.full_tilt_deg,
			dt=rg.episode_config.dt,
			max_initial_yaw_rad=rg.episode_config.max_initial_yaw_rad,
			max_initial_body_rate=rg.episode_config.max_initial_body_rate,
			max_initial_yaw_rate=rg.episode_config.max_initial_yaw_rate,
			eval_episodes=rg.eval_episodes,
			split_tau=rg.split_tau, split_clean_gain=rg.split_clean_gain,
			split_accum_corr=rg.split_accum_corr, split_max_rounds=rg.split_max_rounds,
			split_k_start=rg.split_k_start, split_coarse_target=rg.split_coarse_target,
			split_selective_output=rg.split_selective_output,
			active_roll=self._cur_axes[0], active_pitch=self._cur_axes[1], active_yaw=self._cur_axes[2],
			dist_enabled=dist_en, dist_tau_bias=dist_tb,
			dist_gust_sigma=dist_gs, dist_gust_tau_c=dist_gtc,
			dist_motor_asym=dist_ma, dist_gyro_sigma=dist_gys,
			dist_gyro_bias_walk=dist_gbw, dist_accel_sigma=dist_acs,
			dist_dropout_prob=dist_dp, dist_dropout_len_steps=dist_dls,
			dist_obs_delay_steps=dist_ods, dist_torque_scale_jitter=dist_tsj,
			expert_drives=getattr(rg, "expert_drives", False),
			write_priority_err=getattr(rg, "write_priority_err", False),
			write_err_floor_deg=getattr(rg, "write_err_floor_deg", 0.0),
			# SCOPE C STAGE 1: the TRAINING rollout must fly the same plant the
			# scorer does, or the vertical features are zeros here and real there
			# (the DOB divergence — the Rust side asserts on the mismatch).
			**_stage1_train_kwargs(rg.episode_config),
		)
		controller = ra.WnnController(
			num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
			bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
			state_neurons=spec.state_neurons,
			state_bits_per_neuron=spec.state_bits_per_neuron,
			output_bits_per_neuron=spec.output_bits_per_neuron,
			thresholds=self.thresholds,
			state_connections=state_conns, output_connections=output_conns,
			delta_control=spec.delta_control, delta_max=spec.delta_max,
			delta_leak=spec.delta_leak, delta_gamma=getattr(spec, 'delta_gamma', 1.0),
			obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i,
			obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i, obs_peraxis_yaw=spec.obs_peraxis_yaw, obs_pwm=spec.obs_pwm, obs_yaw_err=spec.obs_yaw_err, obs_yaw_err_i=spec.obs_yaw_err_i,
			obs_collective_cmd=spec.obs_collective_cmd, obs_alt_err=spec.obs_alt_err, obs_vz=spec.obs_vz,
			obs_pos_err_xy=getattr(spec, 'obs_pos_err_xy', False), obs_vel_xy=getattr(spec, 'obs_vel_xy', False),
		output_full_window=getattr(spec, 'output_full_window', False),
		frame_stride=int(getattr(spec, 'frame_stride', 1)),
			dhat_b=(list(spec.dhat_b) if spec.dhat_b is not None else None), dhat_l_gain=spec.dhat_l_gain, dhat_ff=getattr(spec, 'dhat_ff', False), dhat_ff_clamp=getattr(spec, 'dhat_ff_clamp', 0.30), dt=spec.dt,
			integral_leak=spec.integral_leak, integral_scale=spec.integral_scale, decouple_outputs=spec.decouple_outputs,
			action_repeat=spec.action_repeat,
			memory_mode=spec.memory_mode_int(),
			output_decode=spec.output_decode_int(),
		)
		# ONE FFI call for the whole warm-start (was one per cell, ~500k/genome).
		# load_cells reproduces write_*_cell semantics exactly — canonicalising
		# write, 2-bit mask, bounds check — unlike restore_cells, which uses the
		# raw import and would store default-valued cells the write path drops.
		controller.load_cells(init_s or [], init_o or [])
		target_rpy = list(rg.target_rpy) if rg.target_rpy is not None else [0.0, 0.0, 0.0]
		ts = ra.dagger_train_inplace(controller, cfg, target_rpy, int(seed))
		# Re-pack stats to match Python reward_gated_train's dict shape (the
		# fields ControllerEvaluator + downstream actually read).
		stats = {
			"iter_fitness":             list(ts.iter_fitness),
			"iter_mean_err_deg":        list(ts.iter_mean_err_deg),
			"iter_stable_rate":         list(ts.iter_stable_rate),
			"iter_tilt_deg":            list(ts.iter_tilt_deg),
			"iter_n_trained":           list(ts.iter_n_trained),
			"iter_cells_written":       list(ts.iter_cells_written),
			"iter_mean_episode_reward": list(ts.iter_mean_episode_reward),
				"iter_motor_jerk_mean":      list(ts.iter_motor_jerk_mean),
				"iter_mono_violations":      list(ts.iter_mono_violations),
			"train_steps":              int(ts.train_steps),
			"split_saturation":         int(ts.split_saturation),
			"split_wish_bits":          list(ts.split_wish_bits),
		}
		return controller, stats

	def score_population(self, controllers: list) -> list[tuple[float, dict]]:
		"""Closed-loop score each trained controller on the evaluator's fixed
		episode set (fresh recurrent state per episode, comparable to PID).

		Every controller has IDENTICAL shape (same state_neurons/bits/levels —
		they differ only in connectivity + cells), so the GPU path steps all
		(controllers × episodes) rollouts in ONE uniform Metal kernel. The
		closed-loop eval (forward rollout, no solver) is GPU-friendly; the inner
		training (branchy QSR beam-search) stays on CPU. GPU↔CPU parity is
		verified (tests/test_controller_gpu_parity.py) — the GA fitness is the
		same whichever path runs. Falls back to CPU if Metal is unavailable.
		"""
		if controllers:
			# Pick the Rust scorer: GPU (score_controllers_metal) when enabled, else the
			# fast rayon-CPU batch (score_controllers_cpu) — NOT the slow serial Python
			# per-step loop below. WNN_CONTROLLER_GPU_EVAL=0 selects CPU (worker owns GPU).
			try:
				from wnn.control._accel import score_controllers_metal, score_controllers_cpu
				_scorer = score_controllers_metal if self.max_eval_workers_gpu else score_controllers_cpu
			except Exception:
				_scorer = None
			res = self._score_population_rust(controllers, _scorer) if _scorer is not None else None
			if res is not None:
				return res
			# The Rust scorer failing → the slow Python per-step loop (10-50x). Warn once.
			if not getattr(self, "_gpu_score_fallback_warned", False):
				import sys
				print("[ControllerEvaluator] ⚠️ Metal scorer unavailable/failed — falling back to "
				      "the per-step CPU eval loop (10-50x slower). Investigate if unexpected.",
				      file=sys.stderr, flush=True)
				self._gpu_score_fallback_warned = True
		from .dagger import eval_closed_loop_reset
		out = []
		for c in controllers:
			c.reset()
			fit, m = eval_closed_loop_reset(
				make_wnn_action_fn(c), c.reset,
				self.episode_config, self.num_eval, self._active_score_seed,
			)
			# CPU-path parity with the GPU scorer: surface mono from the last
			# emitted output thermometer (jerk already in m as mean_pwm_jerk).
			try:
				from wnn.control._accel import monotonicity_violations
				m["mono_violations"] = float(monotonicity_violations(
					c.get_last_output_cells(), self.spec.levels_per_motor, self.spec.num_motors))
			except Exception:
				m["mono_violations"] = 0.0
			out.append((fit, m))
		return out

	def _score_population_rust(self, controllers: list, scorer):
		"""Batched closed-loop scoring via a Rust scorer (`scorer` = score_controllers_metal
		for GPU or score_controllers_cpu for rayon-CPU — same signature + 12-metric row
		contract). Samples the SAME per-episode ICs as the CPU eval_closed_loop_reset plan,
		so results are interchangeable. Returns list[(mean_reward, metrics)] or None on
		failure. The CPU scorer is a rayon batch (5.6× faster than the GPU under contention,
		no waitUntilCompleted); it fills the 5 GA-fitness metrics (reward/err/stable/jerk/mono
		— jerk formula matched to the GPU kernel), leaving the transient/display metrics 0
		(the held-out report uses the GPU scorer for those).
		"""
		if scorer is None:
			return None
		ec = self.episode_config
		from .training import sample_ics_flat
		q0, omega0 = sample_ics_flat(self._active_score_seed, self.num_eval, ec, active_axes=self._cur_axes)
		# SCOPE C STAGE 1: the vertical draws come from the SAME active score seed
		# as the attitude ICs, via the canonical sampler — so a fold's episodes are
		# one coherent set. Empty/False when translation is off ⇒ both scorers take
		# their bit-identical attitude-only path.
		from .training import sample_vertical_ics_flat
		s1_on = bool(getattr(ec, "translation", False))
		if s1_on:
			z0, vz0, coll, mass = sample_vertical_ics_flat(
				self._active_score_seed, self.num_eval, ec)
			af_mass = float(ec.airframe.mass) if ec.airframe is not None else 1.0
			# STAGE 2: horizontal starts from their OWN salted stream (all-zero
			# when max_initial_xy_offset_m is 0, which the scorers treat as
			# origin starts — the stage-1 behaviour exactly).
			from .training import sample_horizontal_ics_flat
			x0, y0 = sample_horizontal_ics_flat(
				self._active_score_seed, self.num_eval, ec)
			stage1_kwargs = dict(
				translation=True,
				target_altitude=float(getattr(ec, "target_altitude", 0.0)),
				lambda_alt=float(getattr(ec, "lambda_alt", 0.0)),
				init_z=[float(v) for v in z0],
				init_vz=[float(v) for v in vz0],
				# mass_scale × the airframe's nominal mass: the PLANT draw.
				ep_mass=[af_mass * float(m) for m in mass],
				ep_collective_frac=[float(c) for c in coll],
				lambda_pos=float(getattr(ec, "lambda_pos", 0.0)),
				init_x=[float(v) for v in x0],
				init_y=[float(v) for v in y0],
			)
		else:
			stage1_kwargs = {}
		dist = getattr(ec, "disturbance", None)
		# Overactuated Phase 1: N-rotor geometry passthrough (None = legacy quad).
		# Both Rust scorers take the same geometry=/rotor_asym= kwargs and refuse
		# a rows-vs-num_motors mismatch loudly.
		geo = getattr(ec, "geometry", None)
		geo_rows = None if geo is None else [[float(x) for x in row] for row in geo.rows]
		geo_asym = (None if geo is None or geo.rotor_asym is None
		            else [float(x) for x in geo.rotor_asym])
		# Phase 2: allocator-LQR residual baseline. nominal_rows=None ⇒ reuse the
		# sim geometry rows (no allocator-model mismatch). Both scorers take the
		# same alloc_* kwargs; the Metal one additionally needs residual_enabled
		# (its scale/clamp ride the E5 residual fields, pid gains unused).
		ar = getattr(ec, "alloc_residual", None)
		if ar is not None and geo_rows is None:
			raise ValueError("alloc_residual requires EpisodeConfig.geometry")
		alloc_kwargs = {}
		if ar is not None:
			nominal = ar.nominal_rows if ar.nominal_rows is not None else geo.rows
			alloc_kwargs = dict(
				alloc_rows=[[float(x) for x in row] for row in nominal],
				alloc_q_att=float(ar.q_att), alloc_q_rate=float(ar.q_rate),
				alloc_r_ctrl=float(ar.r_ctrl), alloc_tau_max=float(ar.tau_max),
				alloc_f_hover=None if ar.f_hover is None else float(ar.f_hover),
				alloc_lambda=float(ar.pinv_lambda),
			)
			from wnn.control._accel import score_controllers_metal as _metal
			if scorer is _metal:
				alloc_kwargs.update(residual_enabled=True,
				                    residual_scale=float(ar.scale),
				                    residual_clamp=float(ar.clamp))
			else:
				alloc_kwargs.update(residual_scale=float(ar.scale),
				                    residual_clamp=float(ar.clamp))
		try:
			if dist is None:
				agg = scorer(
					controllers, q0, omega0, self.num_eval, ec.steps_per_episode,
					geometry=geo_rows, rotor_asym=geo_asym,
					**ec.sim_kwargs(), **alloc_kwargs, **stage1_kwargs)
			else:
				# W2: weather-on scoring. Base seed = dist.seed XOR the active
				# fold seed, so each K-fold episode pool gets its own weather
				# stream (per-episode variation comes from the in-kernel
				# channel-15 derivation on the episode index). Motor asym: one
				# per-call resolve of the ±mag draw, seeded on the same pair —
				# per-airframe wear, deterministic per fold.
				dseed, asym = disturbance_stream(dist, self._active_score_seed)
				agg = scorer(
					controllers, q0, omega0, self.num_eval, ec.steps_per_episode,
					dist_enabled=True,
					dist_tau_bias=[float(x) for x in dist.tau_bias],
					dist_gust_sigma=float(dist.gust_sigma),
					dist_gust_tau_c=float(dist.gust_tau_c),
					dist_motor_asym=[float(x) for x in asym],
					dist_gyro_sigma=float(dist.gyro_sigma),
					dist_gyro_bias_walk=float(dist.gyro_bias_walk),
					dist_accel_sigma=float(dist.accel_sigma),
					dist_seed=dseed,
					dist_dropout_prob=float(dist.dropout_prob),
					dist_dropout_len_steps=int(dist.dropout_len_steps),
					dist_obs_delay_steps=int(dist.obs_delay_steps),
					dist_torque_scale_jitter=float(dist.torque_scale_jitter),
					geometry=geo_rows, rotor_asym=geo_asym,
					**ec.sim_kwargs(), **alloc_kwargs, **stage1_kwargs)
		except Exception:
			return None
		out = []
		# Each row is 14 metrics (Vec<Vec<f64>> from score_controllers_metal):
		# [reward, err_rad, stable, jerk, mono, steady_rad, rise_s, settle_abs_s,
		#  settle_rel_s, itae, iae, ise, effort, pos_err_m]. Transient-speed metrics
		# (rise/settle/ITAE) are computed in the SAME Rust rollout — see
		# controller_rollout.metal. `effort` = mean per-step Σ pwm² (Σu², Phase 3).
		for row in agg:
			(mean_reward, mean_err_rad, stable_rate, jerk, mono, steady_rad,
			 rise_s, settle_abs_s, settle_rel_s, itae, iae, ise) = row[:12]
			# 13th metric (effort, ABI 9) — tolerate an older 12-row wheel so a
			# mid-cohort process mix can't crash the unpack.
			effort = row[12] if len(row) > 12 else None
			# 14th metric (14/08/2026): mean 3D Euclidean position error in
			# METRES — |alt err| on a vertical-only stage-1 run, 0.0 with
			# translation off. Tolerate a 13-row wheel the same way as effort.
			pos_err_m = row[13] if len(row) > 13 else None
			out.append((float(mean_reward), {
				"mean_reward": float(mean_reward),
				"mean_attitude_error_rad": float(mean_err_rad),
				"mean_attitude_error_deg": math.degrees(mean_err_rad),
				"stable_rate": float(stable_rate),
				# Jerk + mono are now produced by the Rust scorer (single source for
				# ALL stages — see controller_rollout.metal), so every stage that
				# scores via score_population ranks on them identically.
				"mean_pwm_jerk": float(jerk),
				"mono_violations": float(mono),
				# Steady-state-window err (last 20% of steps) — the I-pressure metric.
				"mean_steady_error_deg": math.degrees(steady_rad),
				# Transient-speed metrics (seconds / natural units) — how FAST it corrects.
				"mean_rise_time_s": float(rise_s),
				"mean_settle_time_abs2deg_s": float(settle_abs_s),
				"mean_settle_time_rel5pct_s": float(settle_rel_s),
				"mean_itae": float(itae),
				"mean_iae": float(iae),
				# STAGE 1+2: metres ALONGSIDE the degrees triple, never replacing
				# it (decision 7 — a controller that buys position accuracy by
				# thrashing attitude must stay visible).
				"mean_position_error_m": (float(pos_err_m) if pos_err_m is not None else None),
				"mean_ise": float(ise),
				# Allocation-effort proxy (Σu², Phase 3): the Σu² fitness input.
				"mean_effort": (float(effort) if effort is not None else None),
			}))
		return out

	def _advance_fold(self) -> int:
		"""Rotate to the next fold's episode-pool seed for K-fold scoring.

		Called at the start of every evaluate_batch / score_genomes invocation.
		With K=1 (default) this no-ops: _active_score_seed stays at self.seed,
		legacy behavior preserved. With K>1, _active_score_seed cycles through
		_fold_seeds — all genomes in this batch share the SAME pool (fair
		within-gen ranking); next batch rotates to the next pool (prevents
		single-pool overfit across gens).

		Returns the active fold index (0..K-1) for logging.
		"""
		if self.num_eval_folds <= 1:
			self._active_score_seed = self.seed
			return 0
		fold_idx = self._fold_counter % self.num_eval_folds
		self._active_score_seed = self._fold_seeds[fold_idx]
		self._fold_counter += 1
		return fold_idx

	def _split_cell_floor(self, genomes: list) -> int:
		"""Cell-count floor for the state-splitting trainer (WNN_STATE_SPLIT=1).

		`measured` only sees cells a PRIOR generation wrote back, so at pop-build
		(gen 0, cells=None) the estimator falls to the mode floor — 200k for
		BINARY — and hands back "whole population in one batch". Under split that
		is ~15x too optimistic: split_retrain_output commits an output cell per
		RECORD per neuron, so a genome accumulates ~(episodes_per_round·steps)
		addresses per neuron, not 200k in total. That blind spot is what let the
		20/07 phase-2 arms clone 50 genomes at once and thrash swap.

		Returns the address ceiling scaled by SPLIT_FILL. 0 when split is off or no
		genome has a state layer, so the non-split paths keep full-population
		parallelism."""
		import os
		if os.environ.get("WNN_STATE_SPLIT") != "1":
			return 0
		if not any(getattr(g, "state_neurons", self.spec.state_neurons) > 0 for g in genomes):
			return 0
		rg = self.rg_config
		records = rg.episodes_per_round * rg.steps_per_episode
		per_neuron = min(records, 1 << min(self.spec.output_bits_per_neuron, 62))
		out_neurons = max(
			(getattr(g, "output_neurons", 0) or 0) for g in genomes
		) or (self.spec.num_motors * self.spec.levels_per_motor)
		return int(per_neuron * out_neurons * self.SPLIT_FILL)

	# Fraction of the (records x neurons) ADDRESS ceiling a split-trained genome
	# actually writes. Trajectories revisit attitude regions, so distinct addresses
	# are far below the ceiling: measured 23,985 cells against a 153,600 ceiling
	# (6 eps x 400 steps, 64 output neurons, K=5, BINARY) = 15.6%. Rounded up to
	# 0.2 for headroom. Using the raw ceiling collapsed the chunk to 1 genome and
	# serialised the rayon fan-out; using it unscaled is the safe-but-useless end.
	# Overridable end-to-end by WNN_CTRL_EVAL_BATCH if a run disagrees.
	SPLIT_FILL = 0.2

	def _eval_batch_size(self, genomes: list) -> int:
		"""Genomes per train+score sub-batch (fix 3), so peak memory stays near a budget
		instead of scaling with the whole population. Light modes (QUAD/QSR/BINARY) take
		the whole population in one batch = full parallelism; heavy modes (TERNARY/PLN
		accumulate ~30x QUAD's cells during DAGGER) get small batches. Sized from the
		per-genome cell count (measured from warm-start cells if present, else a mode
		floor). Override with WNN_CTRL_EVAL_BATCH."""
		import os
		N = len(genomes)
		ov = os.environ.get("WNN_CTRL_EVAL_BATCH")
		if ov:
			try:
				return max(1, min(N, int(ov)))
			except ValueError:
				pass
		mode = self.spec.memory_mode_int()
		heavy = mode in (0, 5)  # TERNARY, PLN — hard cells that never consolidate
		measured = 0
		for g in genomes:
			c = getattr(g, "cells", None)
			if c is not None:
				try:
					# O(1) count off the Rust handle — no materialisation at all.
					# (History: this site once called to_triples() to read two
					# lengths, ~1 GB of tuples per genome, BEFORE the sub-batching
					# it feeds; then len() on on-demand numpy views, which still
					# copied both value buffers per genome.)
					measured = max(measured, c.cell_count())
				except Exception:
					pass
		floor = 7_000_000 if heavy else 200_000
		per_genome = max(measured, floor, self._split_cell_floor(genomes))
		# 6GB, not 10: the mem-watchdog SIGTERMs a controller on sustained swap
		# thrash, and every 20/07 phase-2 kill fired with the controller at
		# 9.5-10.6GB RSS. A budget set AT the kill threshold is not a budget.
		budget_bytes = 6 * 1024 * 1024 * 1024
		# PEAK bytes per cell for one ADDITIONAL live genome — retained cells PLUS
		# that genome's share of the training working set. NOT the same quantity as
		# cpu_score.rs's BYTES_PER_CELL (160), which is clone-only: the DashMap
		# entries a read-only rollout copy costs. Two different questions; neither
		# used to say which, which is how they came to disagree 4x.
		# Measured 20/07/2026, marginal batch=1 -> batch=8 under WNN_STATE_SPLIT=1:
		#   1269 B/cell  original
		#    985 B/cell  + pre-sized split_record buffers
		#    758 B/cell  + state_ins_flat bit-packed to the Metal word layout
		# 800 tracks the current build with ~5% headroom. It sat at 1000 while the
		# bit-packed wheel was built-but-not-installed, so the figure stayed valid
		# for whichever wheel a run picked up; that skew window is closed.
		bytes_per_cell = 800
		return max(1, min(N, budget_bytes // (per_genome * bytes_per_cell)))

	def _evaluate_core(self, genomes: list, *, write_back: bool = False,
	                   return_stats: bool = False, seed_offset: int = 0,
	                   generation=None, _skip_advance: bool = False) -> list:
		"""Unified train+score core behind BOTH controller eval entry points.

		Each genome trains by ACCUMULATING across K=num_eval_folds folds into ONE
		canonical controller — fold k+1 warm-starts from fold k's exported cells, so
		writes compound (RAM evidence accumulation, no weight-averaging problem) — then
		the final controller is closed-loop scored once. The "two arms" differ only by
		two booleans; the train/score/cancel-guard machinery is shared (no duplicate
		path):
		  * evaluate_batch          → write_back=False, return_stats=False
		  * evaluate_for_adaptation → write_back=?,     return_stats=True

		Per-fold training uses the Rust-batched trainer (ONE call across all genomes,
		~5x over per-genome), so the Lamarckian path gets the same speedup as the GA
		path; falls back to the per-genome Python core on Rust error. K=num_eval_folds
		is the project-wide "kfold always 5" for controllers (folds are random
		episode-pool seeds → accumulating is "more rollouts", NOT a CV leak;
		generalization is judged by the held-out --report-seed). Subsumes the old
		`fitness_seeds` averaging knob.

		Returns list[Metrics] (return_stats=False) or list[(Metrics, AdaptationStats)]
		(return_stats=True). ONE cancel-guard: PROPER cancel → sentinels and SKIP
		write-back (stamping untrained cells overflows the next warm-start); SPURIOUS →
		reset + retry up to _CANCEL_RETRIES, then raise loudly."""
		from wnn.ram.metrics import ControllerMetrics as Metrics
		from .recurrent_genome import MemoryPayload
		from wnn.control import cancel_state
		# H4: resolve this generation's active-axis mask (full 3-axis unless this
		# evaluator has axis_curriculum_gens set — only the NEURONS evaluator does).
		# Use the explicit generation if the framework passes one, else our counter.
		gen = generation if generation is not None else self._generation
		self._cur_axes = self._active_axes(gen)
		self._ensure_ga_ready()
		if not _skip_advance:
			self._advance_fold()  # advance the fold counter ONCE per eval, not per sub-batch
		from wnn.accel import accel_or_none
		# Loud by default: with ram_accelerator None the cancel-flag check
		# degrades to "never cancelled" (the F1=0.49 bug class).
		ram_accelerator = accel_or_none()

		N = len(genomes)
		K = self.num_eval_folds
		shape_keys = [self._shape_key(g) for g in genomes]
		base_seeds = [self.seed * 100 + seed_offset + gi * K for gi in range(N)]

		# Fix 3 (15/07): bound peak memory instead of letting it scale with the whole
		# population. The train+score below holds EVERY genome's controller cells at once
		# (pop=50 TERNARY ≈ 150GB — it accumulates ~30x QUAD's cells). So process the
		# population in memory-sized sub-batches: each RE-ENTERS this same core with fewer
		# genomes, so K-fold accumulate, per-genome seeds (base_seeds use the GLOBAL index
		# via seed_offset), the cancel-guard and write-back are all bit-identical to the
		# unbatched path. `_skip_advance` keeps the fold counter advancing exactly ONCE.
		_batch = self._eval_batch_size(genomes)
		if _batch < N:
			out = []
			for _bs in range(0, N, _batch):
				out.extend(self._evaluate_core(
					genomes[_bs:_bs + _batch], write_back=write_back,
					return_stats=return_stats, seed_offset=seed_offset + _bs * K,
					generation=generation, _skip_advance=True))
			return out

		_CANCEL_RETRIES = 3
		_cancel_attempt = 0
		while True:
			# Fold 0 inits from genome.cells (Lamarckian warm-start) or empty.
			from wnn.control import _accel as _ra
			cur_inits = []
			for g in genomes:
				cells = getattr(g, "cells", None)
				# Handles, not triples: an empty GenomeCells == "no warm-start".
				cur_inits.append(cells if cells is not None else _ra.GenomeCells())
			controllers = None
			last_stats = [None] * N
			trained = None
			if _rust_dagger_enabled() and N >= 1:
				# ONE call for the WHOLE fold chain: Rust accumulates fold k+1 onto
				# fold k's memory in place, so the cells never cross the FFI boundary
				# between folds. The old loop exported them to Python triples each
				# fold (~95 B/cell × N ≈ 2.4 GB at pop=50) purely to feed them back.
				all_fold_tasks = [(gi, [base_seeds[gi] + k for k in range(K)]) for gi in range(N)]
				try:
					trained = self._train_genomes_rust_batched(
						genomes, all_fold_tasks, init_override=cur_inits)
				except Exception as e:
					if not getattr(self, "_rust_dagger_batch_warned", False):
						import sys
						print(f"[ControllerEvaluator] ⚠️ batched Rust DAGGER FELL BACK to the "
						      f"per-genome Python path (accumulate): {e}",
						      file=sys.stderr, flush=True)
						self._rust_dagger_batch_warned = True
					trained = None
			if trained is None:
				# Fallback keeps the per-fold Python chain (cells DO round-trip here,
				# but this path only runs when the Rust batch trainer is unavailable).
				# cur_inits are handles now — materialise to triples on this legacy
				# path only (an empty handle yields ([], []) == no warm-start).
				fold_inits = [h.to_triples() for h in cur_inits]
				for k in range(K):
					trained = [
						self._train_core(*self._materialize(genomes[gi]),
						                 *fold_inits[gi], base_seeds[gi] + k)
						for gi in range(N)
					]
					if k < K - 1:
						fold_inits = [c.export_cells() for (c, _s) in trained]
			controllers = [c for (c, _s) in trained]
			last_stats = [s for (_c, s) in trained]

			scored = self._score_grouped(controllers, shape_keys)

			try:
				_cancelled = bool(ram_accelerator.is_cancelled()) if ram_accelerator else False
			except Exception:
				_cancelled = False
			if not _cancelled:
				break
			proper = cancel_state.sigterm_received()
			gp = f"gen {generation}" if generation is not None else "gen ?"
			print(f"[ControllerEvaluator] {gp} _evaluate_core [CANCEL-GUARD] cancel flag SET "
			      f"during eval of {N} genomes — "
			      f"{'PROPER (signum=' + str(cancel_state.last_signum()) + ')' if proper else 'SPURIOUS'}. "
			      f"write_back={write_back} → skipping write-back; "
			      f"{'returning sentinels' if proper else 'reset+retry'}.", flush=True)
			if proper:
				sentinel = Metrics(reward=float('-inf'), stable_rate=0.0,
				                   fitness=float('-inf'), mean_attitude_error_deg=180.0)
				if return_stats:
					return [(sentinel, AdaptationStats(reward=float('-inf'), stable_rate=0.0,
					         state_cell_counts=[], output_cell_counts=[])) for _ in genomes]
				return [sentinel for _ in genomes]
			_cancel_attempt += 1
			try:
				if ram_accelerator:
					ram_accelerator.reset_cancel_flag()
			except Exception as _e:
				print(f"[ControllerEvaluator] [CANCEL-GUARD] reset_cancel_flag failed: {_e}", flush=True)
			if _cancel_attempt >= _CANCEL_RETRIES:
				raise RuntimeError(
					f"[ControllerEvaluator] _evaluate_core cancelled {_CANCEL_RETRIES}x consecutively "
					f"(SPURIOUS — flag re-set each time) — refusing to return UNTRAINED controllers.")

		# Build per-genome results from the FINAL accumulated controller.
		out = []
		for gi, g in enumerate(genomes):
			reward, m = scored[gi]
			stable = float(m.get("stable_rate", 0.0))
			err = float(m.get("mean_attitude_error_deg", 0.0))
			st = last_stats[gi]
			# Phase 5c: stamp the splitting trainer's GA-handshake pressure onto the
			# genome so its offspring's mutation consumes it (grow state on
			# saturation, route connections to wished bits). Eval metadata; only the
			# arch genomes carry a `pressure` field.
			if isinstance(st, dict) and hasattr(g, "pressure"):
				g.pressure = (int(st.get("split_saturation", 0)),
				              tuple(st.get("split_wish_bits", ()) or ()))
			# Jerk + mono come from the SCORING dict `m` (the Rust scorer's single
			# source for ALL stages), NOT the training stats — so the fitness is
			# orthogonal to which stage produced the controller.
			_jerk = m.get("mean_pwm_jerk")
			_mono = m.get("mono_violations")
			_steady = m.get("mean_steady_error_deg")
			_effort = m.get("mean_effort")
			_pem = m.get("mean_position_error_m")
			metrics = Metrics(
				reward=float(reward), stable_rate=stable, fitness=float(reward),
				mean_attitude_error_deg=err,
				motor_jerk_mean=(float(_jerk) if _jerk is not None else None),
				mono_violations_total=(float(_mono) if _mono is not None else None),
				mean_steady_error_deg=(float(_steady) if _steady is not None else None),
				mean_effort=(float(_effort) if _effort is not None else None),
				mean_position_error_m=(float(_pem) if _pem is not None else None),
			)
			if write_back or return_stats:
				# Fill counts straight from Rust; the old _cell_stats ALSO called
				# export_cells() unconditionally, materialising every cell as a
				# Python triple even when write_back was False.
				s_counts, o_counts = controllers[gi].cell_fill_counts()
				if write_back and hasattr(g, "cells"):
					# Stage B write-back: cells go controller → GenomeCells handle
					# → MemoryPayload wrapper, never through Python triples. This
					# was the single biggest allocation site in the evaluator.
					g.cells = controllers[gi].export_cells_handle()
				if return_stats:
					out.append((metrics, AdaptationStats(
						reward=float(reward), stable_rate=stable,
						state_cell_counts=s_counts, output_cell_counts=o_counts)))
					continue
			out.append(metrics)
		return out

	def evaluate_batch(self, genomes: list, *, generation: Optional[int] = None,
	                   total_generations: Optional[int] = None,
	                   min_accuracy: Optional[float] = None) -> list:
		"""Train + closed-loop-score a batch → list[Metrics]. Thin wrapper over
		_evaluate_core (write_back=False, return_stats=False): each genome trains by
		accumulating K=num_eval_folds folds into one controller, then is scored.
		(total_generations / min_accuracy are accepted for interface compatibility —
		the GA viability check applies min_accuracy itself.)"""
		return self._evaluate_core(genomes, write_back=False, return_stats=False,
		                           generation=generation)

	def score_genomes(self, genomes: list, **_kw) -> list:
		"""Paradigm-B / MEMORY-dimension scorer: build each controller directly
		from the genome's OWN cells (NO training) and closed-loop score it. The
		cells ARE the genome here, so there's nothing to train — this is the
		batch_evaluate_fn a MEMORY-dimension strategy passes to optimize().

		Memory stage K-fold: same fold rotation as evaluate_batch. Genomes are
		cell-values overfit-prone in their own right (no arch search space to
		distract the GA), so K>1 here matters too."""
		from wnn.ram.metrics import ControllerMetrics as Metrics
		from wnn.control import cancel_state
		import os
		self._ensure_ga_ready()
		self._advance_fold()
		from wnn.accel import accel_or_none
		# Loud by default: with ram_accelerator None the cancel-flag check
		# degrades to "never cancelled" (the F1=0.49 bug class).
		ram_accelerator = accel_or_none()
		# CANCEL GUARD (01/06/2026 — same proper-vs-spurious logic as
		# evaluate_batch). No training here (cells ARE the genome), but the GPU
		# score still polls the cancel flag, so a cancel mid-score yields a
		# degenerate batch. Proper cancel → sentinels (GA unwinds + dump/resume);
		# spurious → reset + retry up to 3×, then raise loudly.
		_CANCEL_RETRIES = 3
		_cancel_attempt = 0
		while True:
			controllers = [build_controller(controller_genome_from_arch(g, self.spec, self.thresholds))
			               for g in genomes]
			scored = self._score_grouped(controllers, [self._shape_key(g) for g in genomes])
			try:
				_cancelled = bool(ram_accelerator.is_cancelled()) if ram_accelerator else False
			except Exception:
				_cancelled = False
			if not _cancelled:
				break
			proper = cancel_state.sigterm_received()
			print(
				f"[ControllerEvaluator] score_genomes [CANCEL-GUARD] cancel flag SET during "
				f"score of {len(genomes)} genomes — "
				f"{'PROPER (signum=' + str(cancel_state.last_signum()) + ')' if proper else 'SPURIOUS'}.",
				flush=True,
			)
			if proper:
				return [
					Metrics(reward=float("-inf"), stable_rate=0.0, fitness=float("-inf"),
					        mean_attitude_error_deg=180.0)
					for _ in genomes
				]
			_cancel_attempt += 1
			try:
				if ram_accelerator:
					ram_accelerator.reset_cancel_flag()
			except Exception as _e:
				print(f"[ControllerEvaluator] [CANCEL-GUARD] reset_cancel_flag failed: {_e}", flush=True)
			if _cancel_attempt >= _CANCEL_RETRIES:
				raise RuntimeError(
					f"[ControllerEvaluator] score_genomes cancelled {_CANCEL_RETRIES}x consecutively "
					f"(SPURIOUS — no SIGTERM to this process) — refusing to return a degenerate batch."
				)
		# jerk + mono from the SAME scoring dict the arch stages use → the MEMORY
		# stage now ranks on them identically (the orthogonality fix).
		return [Metrics(reward=float(r), stable_rate=float(m.get("stable_rate", 0.0)), fitness=float(r),
		                mean_attitude_error_deg=float(m.get("mean_attitude_error_deg", 0.0)),
		                motor_jerk_mean=(float(m["mean_pwm_jerk"]) if m.get("mean_pwm_jerk") is not None else None),
		                mono_violations_total=(float(m["mono_violations"]) if m.get("mono_violations") is not None else None),
		                mean_steady_error_deg=(float(m["mean_steady_error_deg"]) if m.get("mean_steady_error_deg") is not None else None),
		                mean_effort=(float(m["mean_effort"]) if m.get("mean_effort") is not None else None),
		                mean_position_error_m=(float(m["mean_position_error_m"])
		                                       if m.get("mean_position_error_m") is not None else None))
		        for (r, m) in scored]

	def _score_grouped(self, controllers: list, shape_keys: list) -> list:
		"""Score controllers in shape-uniform groups, reassembled in input order.
		Each group goes through score_population (GPU-batched, uniform shape).
		In hybrid mode (WNN_CONTROLLER_GPU_EVAL=2) groups are dispatched concurrently
		across a GPU worker + CPU rayon with straggler-stealing (_score_grouped_hybrid)."""
		from collections import defaultdict
		groups: dict = defaultdict(list)
		for i, key in enumerate(shape_keys):
			groups[key].append(i)
		group_list = list(groups.items())
		if getattr(self, "eval_mode", "gpu") == "hybrid" and len(group_list) > 1:
			return self._score_grouped_hybrid(controllers, group_list)
		scored: list = [None] * len(controllers)
		for key, idxs in group_list:
			sub = self.score_population([controllers[i] for i in idxs])
			for j, i in enumerate(idxs):
				scored[i] = sub[j]
		return scored

	def _score_grouped_hybrid(self, controllers: list, group_list: list) -> list:
		"""Concurrent GPU+CPU group dispatch with straggler-stealing (Luiz 09/07):
		a GPU worker thread and the CPU (rayon, main thread) both pull shape-groups
		from a shared queue — whoever's free grabs the next. If the GPU is stalled on
		its last group (starved behind the IDS worker's kernels), the CPU steals it
		(re-scores on CPU); whichever finishes first wins, the loser's result is dropped.
		So the GPU is used only in the gaps it's actually available, CPU covers the rest,
		and no group blocks on a starved GPU. The abandoned GPU thread is a daemon and
		finishes on its own (its late result is discarded via the `done` guard)."""
		import threading
		from queue import Queue, Empty
		try:
			from wnn.control._accel import score_controllers_metal, score_controllers_cpu
		except Exception:
			# No accel → fall back to the serial GPU/CPU path.
			scored = [None] * len(controllers)
			for key, idxs in group_list:
				sub = self.score_population([controllers[i] for i in idxs])
				for j, i in enumerate(idxs):
					scored[i] = sub[j]
			return scored
		scored: list = [None] * len(controllers)
		done = [False] * len(group_list)
		lock = threading.Lock()
		pending: Queue = Queue()
		for gi in range(len(group_list)):
			pending.put(gi)

		def _store(gi: int, sub: list) -> None:
			_, idxs = group_list[gi]
			for j, i in enumerate(idxs):
				scored[i] = sub[j]
			done[gi] = True

		def _score(gi: int, scorer):
			_, idxs = group_list[gi]
			return self._score_population_rust([controllers[i] for i in idxs], scorer)

		def _gpu_worker():
			while True:
				try:
					gi = pending.get_nowait()
				except Empty:
					break
				with lock:
					if done[gi]:
						continue
				sub = _score(gi, score_controllers_metal)  # blocks on GPU (may be starved)
				with lock:
					if sub is None:
						pending.put(gi)      # GPU failed → hand back to the CPU
					elif not done[gi]:
						_store(gi, sub)

		gpu_t = threading.Thread(target=_gpu_worker, daemon=True)
		gpu_t.start()
		# CPU (main thread): drain the queue with the rayon scorer.
		while True:
			try:
				gi = pending.get_nowait()
			except Empty:
				break
			with lock:
				if done[gi]:
					continue
			sub = _score(gi, score_controllers_cpu)
			with lock:
				if sub is not None and not done[gi]:
					_store(gi, sub)
		# Straggler steal: any group still in-flight on a stalled GPU → CPU re-scores.
		gpu_t.join(timeout=0.2)
		for gi in range(len(group_list)):
			with lock:
				if done[gi]:
					continue
			sub = _score(gi, score_controllers_cpu)
			with lock:
				if not done[gi]:
					_store(gi, sub)
		return scored

	def evaluate_single(self, genome) -> float:
		"""Single-genome lower-is-better scalar (-reward). Fallback path."""
		return -self.evaluate_batch([genome])[0].reward

	# ------------------------------------------------------------------
	# Lamarckian / adaptive-eval interface (Phase B step 4b). Trains + scores
	# AND surfaces per-neuron fill stats from the trained controller (export_cells)
	# so ControllerAdaptationStrategy can drive stats-guided genesis. Optionally
	# stamps the trained cells back into genome.cells (Lamarckian write-back) —
	# the acquired memory is then inherited (and remapped by 4a under arch genesis).
	# ------------------------------------------------------------------

	def evaluate_for_adaptation(self, genomes: list, *, write_back: bool = False,
	                            seed_offset: int = 0) -> list:
		"""Train (Lamarckian warm-start) + score + optionally write the accumulated
		cells back, returning per genome (Metrics, AdaptationStats). Thin wrapper over
		_evaluate_core (return_stats=True). See _evaluate_core for the K-fold accumulate
		semantics; with write_back the final accumulated state is stamped into
		genome.cells so offspring inherit it."""
		return self._evaluate_core(genomes, write_back=write_back, return_stats=True,
		                           seed_offset=seed_offset)

__all__ = [
	"ControllerSpec",
	"ControllerGenome",
	"AdaptationStats",
	"ControllerEvaluator",
	"fit_thresholds_from_pid_rollouts",
	"collect_student_feature_samples",
	"calib_episode_config",
	"random_connectivity",
	"build_controller",
	"NUM_FEATURES",
]
