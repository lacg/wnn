"""Episode runner + GA orchestration for WNN attitude controllers.

Stays in Python (per `project_drone_controller_paper1.md`): outer-loop
orchestration is cheap and benefits from Python's ecosystem (numpy,
matplotlib for diagnostics, pytest for unit tests). The per-step hot
path (sim physics, controller forward, decoder, reward) is in Rust —
see `wnn.control.sim`, `.controller`, `.decoders`, and `compute_reward`
re-exported below.

The GA itself reuses the existing pipeline in
`src/wnn/ram/experiments/` (same one IDS flows use). Controller-specific
glue is the episode runner + fitness function that calls into Rust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle
	from wnn.control.airframe import Airframe

from wnn.control._accel import (
	AttitudeSim,
	WnnController,
	compute_reward,
	monotonicity_violations,
	strategy_5_qsr_weighted,
)


# Reward defaults: attitude-error² weight is implicit (compute_reward returns
# -attitude_error²). The other lambdas are zero by default and turned on by
# the training script that wants smoothness or thermometer regularization.
_DEFAULT_LAMBDA_SMOOTH = 0.0
_DEFAULT_LAMBDA_MONO = 0.0


def _euler_to_quat(roll: float, pitch: float, pitch_unused: float = 0.0) -> tuple[float, float, float, float]:
	"""Roll-pitch-yaw (XYZ extrinsic) → unit quaternion (w, x, y, z).

	Matches the convention in controller.rs / pid.py.
	"""
	cr = math.cos(roll * 0.5)
	sr = math.sin(roll * 0.5)
	cp = math.cos(pitch_unused * 0.5)  # actually pitch
	sp = math.sin(pitch_unused * 0.5)
	# Yaw passed via pitch_unused for clarity issue — fixed below
	# Use the explicit version with 3 args
	raise RuntimeError("use _euler_to_quat_xyz instead")


def _euler_to_quat_xyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
	"""(roll, pitch, yaw) radians → unit quaternion (w, x, y, z).

	Z-Y-X (intrinsic / Tait-Bryan) convention matching pid._quat_to_euler.
	"""
	cr = math.cos(roll * 0.5)
	sr = math.sin(roll * 0.5)
	cp = math.cos(pitch * 0.5)
	sp = math.sin(pitch * 0.5)
	cy = math.cos(yaw * 0.5)
	sy = math.sin(yaw * 0.5)
	w = cr * cp * cy + sr * sp * sy
	x = sr * cp * cy - cr * sp * sy
	y = cr * sp * cy + sr * cp * sy
	z = cr * cp * sy - sr * sp * cy
	# normalize defensively
	n = math.sqrt(w * w + x * x + y * y + z * z)
	if n > 0:
		return (w / n, x / n, y / n, z / n)
	return (1.0, 0.0, 0.0, 0.0)


@dataclass
class GeometryConfig:
	"""Overactuated N-rotor geometry for the sim (Phase 1; None = legacy quad).

	Threads through the Rust batch scorers (`score_controllers_metal` /
	`score_controllers_cpu` geometry=/rotor_asym= kwargs) — the sim then runs
	the generic r×F + spin-drag torque (AttitudeSim.step_n) instead of the
	quad mixer. Rows follow the AttitudeSim.set_geometry contract:
	[px,py,pz, ax,ay,az, spin, k_thrust, k_drag] per rotor (axis need not be
	pre-normalized). Pass the PERTURBED (true-vehicle) table to model
	tilt/position error vs the nominal allocator — build it with
	AttitudeSim.perturb_geometry or RotorGeometry::perturbed (Rust).
	"""

	# N rows of 9 floats (see contract above). len(rows) must equal the
	# controllers' num_motors — the Rust scorers refuse a mismatch loudly.
	rows: list = field(default_factory=list)
	# Per-rotor thrust multipliers (N-rotor D3 twin; baked into effective
	# k_thrust at upload). None = clean motors.
	rotor_asym: Optional[list] = None


@dataclass
class AllocResidualConfig:
	"""Overactuated Phase 2: allocator-LQR residual baseline for the Rust
	batch scorers. When set (with EpisodeConfig.geometry), the WNN output is
	scored as a SIGNED residual on the in-rollout allocator-LQR baseline:
	pwm = clamp(base + clamp((wnn−0.5)·scale, ±clamp)). `nominal_rows` is the
	ALLOCATOR's geometry model (None ⇒ reuse GeometryConfig.rows — i.e. no
	model mismatch; pass the unperturbed table when the sim side is perturbed).
	Gains/limits mirror AllocLqrRs so teacher ≡ baseline by construction.
	"""

	nominal_rows: Optional[list] = None
	q_att: float = 12.0
	q_rate: float = 1.0
	r_ctrl: float = 1.0
	tau_max: float = 0.144       # quad-equivalent physical authority (N·m)
	f_hover: Optional[float] = None  # None ⇒ Σ kᵢ·0.25 (hover PWM 0.5/rotor)
	pinv_lambda: float = 1e-6
	scale: float = 1.0           # residual gain on (wnn − 0.5)
	clamp: float = 0.15          # |Δu| bound — the safety argument


@dataclass
class DisturbanceConfig:
	"""W2 disturbance parameters (D1-D4) for the attitude sim.

	Mirrors the Rust `Disturbance` struct (controller.rs) field-for-field; the
	extra Python-side conveniences are `motor_asym_mag` (per-episode δ draw)
	and `episode_seed` (verbatim seed override for parity tests).

	Explicit fields ALWAYS win — the `preset()` levels are just initial-guess
	constructors over these fields (to be calibrated by W2.0).
	"""

	# D1: constant body-frame torque bias (N·m).
	tau_bias: tuple = (0.0, 0.0, 0.0)
	# D2: Ornstein-Uhlenbeck gust torque per axis (σ in N·m/√s, τ_c in s).
	gust_sigma: float = 0.0
	gust_tau_c: float = 0.1
	# D3: per-motor k_thrust multipliers (1.0 = clean motor). Fixed part.
	motor_asym: tuple = (1.0, 1.0, 1.0, 1.0)
	# D3 per-episode draw: each episode multiplies motor_asym[i] by
	# (1 + U(-mag, +mag)) drawn from the episode rng. Applied by run_episode
	# (and apply_disturbance) only — the batched Rust/Metal paths use the
	# fixed `motor_asym` (motor wear is per-airframe, not per-flight).
	motor_asym_mag: float = 0.0
	# D4: sensor noise — gyro white σ (rad/s) + slow bias walk (rad/s/√s);
	# accel white σ (m/s²).
	gyro_sigma: float = 0.0
	gyro_bias_walk: float = 0.0
	accel_sigma: float = 0.0
	# D5: sensor dropout/freeze — per-step probability a freeze episode STARTS
	# + freeze duration (steps). While frozen read_imu returns the LAST
	# pre-freeze cached reading (frozen sensors, not zeros). 0 = off.
	dropout_prob: float = 0.0
	dropout_len_steps: int = 0
	# D6: observation latency — read_imu returns the NOISY reading from this
	# many steps ago (ring-buffered in the sim, max 8). 0 = off.
	obs_delay_steps: int = 0
	# D7: dynamics randomization — per-EPISODE per-axis torque scale drawn
	# uniform in [1-j, 1+j]; multiplies the TOTAL torque (base+bias+gust) —
	# emulates inertia/mass mismatch. 0 = off.
	torque_scale_jitter: float = 0.0
	# Base seed for the counter RNG. Per-episode seeds derive from it (batched
	# paths: ram_controller.disturbance_episode_seed; run_episode: episode rng).
	seed: int = 0
	# Parity-test hook: when set, run_episode uses THIS seed verbatim instead
	# of drawing one from the episode rng (single-episode use).
	episode_seed: Optional[int] = None
	# Verbatim motor-asymmetry override, same idiom as `episode_seed` above.
	# When set, apply_disturbance uses THIS vector for every episode instead of
	# redrawing the ±mag δ per episode.
	#
	# WHY (07/08/2026). The batched Rust/Metal scorers resolve ONE asym per
	# scoring call — "per-airframe wear, deterministic per fold"
	# (evaluator.disturbance_stream) — because a real airframe has FIXED wear
	# and is flown many times. The Python sim redrew it EVERY episode, so a
	# Python scoring pass averaged err over a distribution of airframes while
	# the kernel evaluated it at a single one. Since err is convex in
	# disturbance magnitude, that put the Python column systematically ABOVE
	# the Rust one (Jensen, not noise) — it is why the L2 table's two columns
	# could not be compared even after the seed was threaded.
	#
	# SCORING passes now bind this once per pass (see
	# dagger.eval_closed_loop_reset) so Python and the kernel fly the same
	# airframe. TRAINING paths deliberately keep the per-episode redraw: there
	# the variety IS the point (domain randomization over airframe wear).
	resolved_asym: Optional[tuple] = None

	# Intensity-ladder presets (plan w2_disturbances.md). Magnitudes are % of
	# max control torque L·k_thrust (default sim: 0.075 m × 2.4 N = 0.18 N·m),
	# bias on the ROLL axis.
	# W2.0 calibration v1 (06/07, logs/controller/W2Calibrate_20260706): the
	# original guesses ({2,5,10}% bias) moved steady-state error linearly
	# (PD 0.62/1.50/2.95° @2000) but destabilized NOTHING — offsets stayed
	# under the stability threshold; PID and PD both held 100% everywhere.
	# v2 scales ~3× so L2 pushes PD toward the threshold while a working
	# integrator can still trim it, and L3 threatens PID itself.
	# ------------------------------------------------------------------
	# L4 — the LITERATURE-GROUNDED ladder (05/08/2026). SI everywhere.
	#
	# Every value below cites a paper or a datasheet; see
	# docs/disturbance_param_sources.md for the full provenance table and for
	# what was DROPPED and why. L1/L2/L3/L2D/L3D are retained ONLY so old
	# markers remain re-readable — see _LEVELS' deprecation note.
	#
	# SENSOR NOISE IS NOT AN AXIS. It is fixed at the ADIS16448 datasheet
	# (RotorS `component_snippets.xacro`, the file Furrer/Molchanov delegate to)
	# because the two datasheets we read sit within ~2x of each other AND in the
	# opposite order to intuition: the hobby MPU-9250 is QUIETER on paper
	# (1.745e-4 rad/s/sqrt(Hz)) than the "research-grade" ADIS16448
	# (3.394e-4). There is no sourced sensor-quality span to build an axis from.
	#
	# Density -> per-sample sigma uses RotorS' OWN convention, read from
	# gazebo_imu_plugin.cpp so we match the source rather than guess:
	#     sigma_g_d = 1 / sqrt(dt) * gyroscope_noise_density
	# At dt = 1e-3 s (our 1 kHz sim): sqrt(1/dt) = 31.6228.
	#     gyro : 3.394e-4 * 31.6228 = 1.0733e-2 rad/s
	#     accel: 4.0e-3   * 31.6228 = 1.2649e-1 m/s^2
	# For scale, the OLD ladder ran gyro_sigma 0.030 (L2D) / 0.080 (L3D) — 2.8x
	# and 7.5x this, and 17x/46x the MPU-9250's filtered 0.1 deg/s-rms figure.
	#
	# `gyro_bias_walk` is S2's `gyroscope_random_walk` verbatim (our sim applies
	# walk * sqrt_dt, which IS the standard random-walk form, so no conversion).
	#
	# THE AXIS IS PLANT UNCERTAINTY, per Molchanov et al. 2019 (arXiv:1903.04628)
	# Table I + Table IV: randomization "works best if fairly small (20% in our
	# case)", and their 30% row MEASURABLY DEGRADES transfer. So 0.20 is a
	# sourced CEILING, not a midpoint — L4c sits ON it, nothing goes past it.
	#
	# Tuple: (torque_scale_jitter, motor_asym_mag)
	_L4_LEVELS = {
		"L4A": (0.00, 0.00),   # clean plant — the isolation/ablation baseline
		"L4B": (0.10, 0.10),   # Molchanov's measured-helpful band
		"L4C": (0.20, 0.20),   # Molchanov's measured ceiling; 0.30 is known-harmful
	}
	# ADIS16448 @ 1 kHz, RotorS convention. Shared by every L4 rung.
	_L4_GYRO_SIGMA = 3.394e-4 * 31.6228        # 1.0733e-2 rad/s
	_L4_ACCEL_SIGMA = 4.0e-3 * 31.6228         # 1.2649e-1 m/s^2
	_L4_GYRO_BIAS_WALK = 3.8785e-5             # rad/s/s/sqrt(Hz), S2 verbatim

	# DEPRECATED (05/08/2026). Kept ONLY so pre-L4 markers/winners stay loadable.
	# Do NOT start new work on these: the magnitudes are unsourced initial
	# guesses, and L2D/L3D additionally carry D5/D6/D7 fields that NO surveyed
	# simulator or DR paper models (sensor dropout, observation delay) plus a
	# constant tau_bias with no literature counterpart. Nothing measured on them
	# is submission-grade. Use L4A/L4B/L4C.
	_LEVELS = {
		"L1": (0.05, 0.05, 0.010),
		"L2": (0.15, 0.10, 0.030),
		"L3": (0.30, 0.15, 0.080),
	}

	# W2.4 D5-D7 extension presets: L2D/L3D = the L2/L3 ladder values PLUS
	# sensor dropout/freeze + observation latency + dynamics randomization.
	# Base levels keep the new fields at 0 (bit-identical to pre-W2.4).
	# Tuples: (base level, dropout_prob, dropout_len_steps, obs_delay_steps,
	# torque_scale_jitter).
	_D_LEVELS = {
		"L2D": ("L2", 0.002, 20, 2, 0.15),
		"L3D": ("L3", 0.005, 40, 4, 0.25),
	}

	@classmethod
	def preset(cls, level: str, seed: int = 0) -> Optional["DisturbanceConfig"]:
		"""Level → config. OFF → None. L1/L2/L3 → initial-guess ladder:
		τ_bias {2,5,10}% of L·k_thrust; OU σ matched so the stationary gust
		std equals τ_bias (σ = mag/√(τ_c/2)); δ_i ±{3,6,10}%;
		σ_g {0.005,0.02,0.05} rad/s; bias walk σ_g/10; σ_a = 10·σ_g m/s²
		(accel guess — the plan leaves it open)."""
		lv = (level or "OFF").strip().upper()
		if lv in ("OFF", "", "NONE"):
			return None
		if lv in cls._L4_LEVELS:
			# L4: sourced ladder. Sensor noise FIXED at ADIS16448; the rung
			# varies ONLY plant uncertainty (Molchanov's axis).
			#
			# tau_bias and gust_sigma are deliberately ZERO here. tau_bias had
			# no literature counterpart at all, and gust_sigma was slaved to it
			# (sigma = bias/sqrt(tau_c/2)) — that coupling is what made
			# "quality airframe in serious wind" inexpressible. The WEATHER axis
			# is Dryden MIL-F-8785C (W20 = 7.72 / 15.43 / 23.15 m/s for
			# light/moderate/severe, converted from 15/30/45 kt at 6.096 m) and
			# is NOT wired yet: Dryden yields a wind VELOCITY field, our sim
			# takes a gust TORQUE, and the velocity->torque coupling needs the
			# airframe's drag area / centre-of-pressure offset. Deriving that
			# without the airframe numbers would be exactly the invention this
			# ladder exists to remove, so weather lands as L4*-W* once the
			# coupling is written and reviewed.
			jitter, asym = cls._L4_LEVELS[lv]
			return cls(
				tau_bias=(0.0, 0.0, 0.0),
				gust_sigma=0.0,
				gust_tau_c=0.1,
				motor_asym=(1.0, 1.0, 1.0, 1.0),
				motor_asym_mag=asym,
				gyro_sigma=cls._L4_GYRO_SIGMA,
				gyro_bias_walk=cls._L4_GYRO_BIAS_WALK,
				accel_sigma=cls._L4_ACCEL_SIGMA,
				# D5/D6 stay OFF: sensor dropout and observation latency appear
				# in NO surveyed simulator. Molchanov models MOTOR LAG instead
				# (T = 0.15 s settling) and measured its impact as "very
				# small"; our sim lacks that field, so adding it is the sourced
				# follow-up, not a reason to keep the unsourced ones.
				dropout_prob=0.0,
				dropout_len_steps=0,
				obs_delay_steps=0,
				torque_scale_jitter=jitter,
				seed=seed,
			)
		if lv in cls._D_LEVELS:
			# W2.4: D-extended level = the base ladder level + D5/D6/D7 fields.
			base_lv, dp, dl, od, tj = cls._D_LEVELS[lv]
			cfg = cls.preset(base_lv, seed=seed)
			cfg.dropout_prob = dp
			cfg.dropout_len_steps = dl
			cfg.obs_delay_steps = od
			cfg.torque_scale_jitter = tj
			return cfg
		if lv not in cls._LEVELS:
			raise ValueError(
				f"unknown disturbance level {level!r}; known: OFF, "
				f"L4A, L4B, L4C (sourced — use these), "
				f"L1, L2, L3, L2D, L3D (DEPRECATED, pre-L4 compatibility only)")
		pct, asym_mag, gyro_sigma = cls._LEVELS[lv]
		max_torque = 0.075 * 2.4   # default-sim L · k_thrust (N·m)
		tau_c = 0.1
		bias = pct * max_torque
		return cls(
			tau_bias=(bias, 0.0, 0.0),
			gust_sigma=bias / math.sqrt(tau_c / 2.0),
			gust_tau_c=tau_c,
			motor_asym=(1.0, 1.0, 1.0, 1.0),
			motor_asym_mag=asym_mag,
			gyro_sigma=gyro_sigma,
			gyro_bias_walk=gyro_sigma / 10.0,
			accel_sigma=gyro_sigma * 10.0,
			seed=seed,
		)

	def resolved_motor_asym(self, rng: np.random.Generator) -> tuple:
		"""Fixed multipliers × the per-draw (1 + U(-mag, +mag)) factor."""
		if self.motor_asym_mag <= 0.0:
			return tuple(float(x) for x in self.motor_asym)
		return tuple(
			float(x) * (1.0 + float(rng.uniform(-self.motor_asym_mag, self.motor_asym_mag)))
			for x in self.motor_asym
		)


def apply_disturbance(sim: "AttitudeSim", dist: DisturbanceConfig, rng: np.random.Generator) -> None:
	"""Arm one episode's disturbance on the sim.

	Per-episode seed derivation (documented contract): `dist.episode_seed` if
	set (parity tests), else ONE uint32 drawn from the episode rng — AFTER the
	IC draw in run_episode, so the IC sequence of disturbance-off runs is
	unchanged. The per-episode motor-asym δ (if motor_asym_mag > 0) draws next
	(4 uniforms)."""
	if dist.episode_seed is not None:
		ep_seed = int(dist.episode_seed)
	else:
		ep_seed = int(rng.integers(0, 2**32 - 1))
	# The draw happens EITHER WAY, even when it is discarded, so that binding
	# `resolved_asym` changes ONLY the asymmetry and leaves the per-episode
	# ep_seed stream (and therefore the weather and the IC sequence) untouched.
	# Without this, switching to a fixed asym would silently shift every
	# subsequent episode's seed and confound the two changes.
	drawn = dist.resolved_motor_asym(rng)
	asym = tuple(dist.resolved_asym) if dist.resolved_asym is not None else drawn
	sim.set_disturbance(
		tau_bias=[float(x) for x in dist.tau_bias],
		gust_sigma=float(dist.gust_sigma),
		gust_tau_c=float(dist.gust_tau_c),
		motor_asym=[float(x) for x in asym],
		gyro_sigma=float(dist.gyro_sigma),
		gyro_bias_walk=float(dist.gyro_bias_walk),
		accel_sigma=float(dist.accel_sigma),
		seed=ep_seed,
		dropout_prob=float(dist.dropout_prob),
		dropout_len_steps=int(dist.dropout_len_steps),
		obs_delay_steps=int(dist.obs_delay_steps),
		torque_scale_jitter=float(dist.torque_scale_jitter),
	)


@dataclass
class EpisodeConfig:
	"""Configuration for a single training/eval episode."""

	# Sim
	dt: float = 0.001                       # 1 ms = 1 kHz update
	steps_per_episode: int = 2000           # 2 s of simulated flight
	max_initial_tilt_rad: float = math.radians(60.0)
	max_initial_yaw_rad: float = math.radians(45.0)  # bound on initial yaw (not full -π to π)
	max_initial_yaw_rate: float = 0.5       # rad/s; bound on initial omega_z
	max_initial_body_rate: float = 1.0      # rad/s; bound on initial omega_x/y

	# H4 axis curriculum: which attitude axes are perturbed in the IC
	# (roll, pitch, yaw). Inactive ⇒ that axis' tilt + body rate zeroed.
	active_axes: tuple = (True, True, True)

	# Reward weights
	lambda_smooth: float = _DEFAULT_LAMBDA_SMOOTH
	lambda_mono: float = _DEFAULT_LAMBDA_MONO

	# Early termination: stop episode if attitude error exceeds this (rad).
	# Default disabled. Useful during training to skip episodes that have
	# already diverged so we don't waste compute on hopeless rollouts.
	abort_attitude_error_rad: Optional[float] = None

	# Steady-state window: fraction of (full) steps at the END of the episode over
	# which the steady-state-offset metric is averaged. Must match the Metal kernel
	# constant (0.20 = last 20%). The I-pressure fitness term ranks on this.
	steady_window_frac: float = 0.20

	# W2 disturbances (None = clean sim, pre-W2 behavior). Threads through
	# run_episode (CPU), score_controllers_metal (GPU), and the packed
	# reward-gated training config (W2.3 train-under-weather).
	disturbance: Optional[DisturbanceConfig] = None

	# Overactuated N-rotor geometry (Phase 1; None = legacy quad, bit-identical).
	# Threads through the Rust batch scorers ONLY (score_controllers_metal /
	# score_controllers_cpu). The serial Python run_episode path is quad-only —
	# an N≠4 controller fails there loudly (sim.step takes 4 PWMs). Training
	# (DAGGER teachers / allocator) is Phase 2.
	geometry: Optional[GeometryConfig] = None

	# Overactuated Phase 2: allocator-LQR residual baseline (requires geometry;
	# None = pure-WNN scoring). See AllocResidualConfig.
	alloc_residual: Optional[AllocResidualConfig] = None

	# AIRFRAME (05/08/2026). None = the pre-airframe synthetic plant, so every
	# existing caller and parity anchor stays bit-identical. Set it to an entry
	# from wnn.control.airframe (which carries the citation) and BOTH the sim
	# and the model-based teachers read the same numbers — LQR/LQI/MPC/MPCOF
	# derive their gains from the plant and adapt automatically. PID does NOT:
	# its gains are fixed, so switching airframe requires the matching
	# Airframe.gains() to be applied too (see that module's unit_note).
	airframe: Optional["Airframe"] = None

	# --- SCOPE C STAGE 1 (13/08/2026): the vertical episode axes. All zero/None ⇒
	#     translation is inert and every attitude-only episode is unchanged.
	#     docs/scope_c_full_controller_spec.md §"Stage 1 — the change" items 1 & 4.
	# Integrate vertical dynamics at all. Mass comes from the airframe; it is a
	# randomized PLANT parameter, never a feature (Luiz, 12/08).
	translation: bool = False
	# Initial altitude offset bound (m): z0 ~ U(-x, +x). A controller that always
	# starts at its target has never been asked to CORRECT altitude.
	max_initial_alt_offset_m: float = 0.0
	# Initial vertical velocity bound (m/s): vz0 ~ U(-x, +x).
	max_initial_vz: float = 0.0
	# Commanded-collective variation as a FRACTION of hover thrust: the episode's
	# commanded collective is hover·(1 + U(-x, +x)). 0.0 ⇒ always hover, which is
	# the regime a fixed-collective controller has effectively been flying — and
	# "a controller that only ever sees hover has not learned to work at other
	# throttles" (the spec's own words).
	collective_cmd_jitter: float = 0.0
	# Per-episode mass randomization as a fraction: m = m_nominal·(1 + U(-x, +x)).
	# Molchanov randomizes thrust-to-weight U(1.8, 2.5) and never inputs it; this
	# is the same idea expressed against our airframe's nominal mass.
	mass_jitter: float = 0.0
	# The altitude every episode holds (m) and the reward weight on altitude
	# error. lambda_alt = 0.0 ⇒ the attitude-only reward, bit-identically; it
	# also carries the metres↔radians conversion, so it comes from a SWEEP.
	target_altitude: float = 0.0
	lambda_alt: float = 0.0

	def airframe_kwargs(self) -> dict:
		"""The af_* kwargs the Rust scorers/trainer take. Empty when no airframe
		is set, which leaves the Rust-side defaults (the synthetic plant) in
		force — that is what keeps untouched callers bit-identical."""
		af = self.airframe
		if af is None:
			return {}
		out = dict(
			af_arm_length=float(af.arm_length), af_k_thrust=float(af.k_thrust),
			af_k_drag=float(af.k_drag),
			af_inertia=[float(x) for x in af.inertia],
			af_gravity=float(af.gravity), af_dt=float(self.dt),
		)
		out.update(self._pid_cascade_kwargs(af))
		return out

	@staticmethod
	def _pid_cascade_kwargs(af: "Airframe") -> dict:
		"""The firmware-sourced PID cascade, converted to SI HERE and nowhere else.

		This is the boundary: `_SiGains.from_firmware` is the only code in the stack that
		sees degrees or actuator counts, and Rust receives rad / rad-per-second / newtons.
		Returns {} when the airframe registers no cascade gains, which makes the Rust side
		keep the legacy hand-tuned single loop — the fallback that holds the
		synthetic-plant parity anchors bit-identical.
		"""
		from wnn.control.pid_firmware import RATE_LPF_CUTOFF_HZ, _SiGains
		try:
			gains = af.gains()
		except KeyError:
			return {}
		if gains.rate is None:
			return {}
		si = _SiGains.from_firmware(gains, af.k_thrust)
		flat = lambda axes: [
			float(v) for a in axes for v in (a.kp, a.ki, a.kd, a.i_limit)
		]
		return dict(
			af_pid_att=flat(si.attitude), af_pid_rate=flat(si.rate),
			af_pid_out_limit_n=float(si.rate_output_limit_n),
			af_pid_hover_n=float(af.mass * af.gravity / 4.0),
			af_pid_attitude_hz=500.0,   # stabilizer_types.h ATTITUDE_RATE
			af_pid_lpf_hz=float(RATE_LPF_CUTOFF_HZ),
		)

	def cascade_kwargs(self) -> dict:
		"""Just the af_pid_* cascade subset of airframe_kwargs, under the names the
		RESIDUAL scorers take (score_controllers_metal's af_pid_* params, L2 06/08/2026).

		Exists so a residual caller can hand the GPU the SAME cascade the CPU
		AttitudePidFirmware runs — the two-copies-of-one-number risk that the old
		dagger.py::_refuse_cascade_on_residual guard existed to prevent. Empty when the
		airframe registers no cascade, which leaves the kernel on the legacy single-loop
		pid_step (bit-identical to every pre-L2 residual run).
		"""
		af = self.airframe
		if af is None:
			return {}
		return {k: v for k, v in self._pid_cascade_kwargs(af).items()
		        if k.startswith("af_pid_")}

	def sim_kwargs(self) -> dict:
		"""Same plant, under the names the BATCH SCORERS take
		(score_controllers_cpu/metal use bare `arm_length`/`k_thrust`/... while
		the trainer and baseline take `af_*`). Two spellings of one source —
		which is exactly why the airframe lives in a struct instead of being
		typed out at each call site."""
		af = self.airframe
		if af is None:
			return {}
		return dict(
			arm_length=float(af.arm_length), k_thrust=float(af.k_thrust),
			k_drag=float(af.k_drag),
			inertia=[float(x) for x in af.inertia],
			gravity=float(af.gravity),
		)


@dataclass
class EpisodeResult:
	"""Per-episode telemetry returned by run_episode for analysis."""
	cumulative_reward: float
	mean_attitude_error_rad: float
	max_attitude_error_rad: float
	max_omega_norm: float          # peak |omega| during the episode (rad/s)
	steps_completed: int           # < steps_per_episode if early aborted
	diverged: bool                 # True iff sim.is_unstable() fired
	mean_pwm_jerk: float           # mean |pwm[t] - pwm[t-1]| over the episode
	mean_steady_error_rad: float = 0.0  # mean attitude err over the last steady_window_frac of steps
	# --- Transient-speed metrics (how FAST it corrects, not just how well). All
	# times in seconds; diverged episodes get full-duration sentinels (worst case). ---
	rise_time_s: float = 0.0            # time for |err| to first fall to 10% of initial (90% correction)
	settle_time_abs2deg_s: float = 0.0 # first t after which |err| stays < 2° for the rest of the episode
	settle_time_rel5pct_s: float = 0.0 # same, band = 5% of the initial |err|
	itae: float = 0.0                  # Σ t·|err|·dt  (time-weighted abs error; primary transient metric)
	iae: float = 0.0                   # Σ |err|·dt   (integral of abs error)
	ise: float = 0.0                   # Σ err²·dt    (integral of squared error)


def sample_vertical_ics_flat(seed, num_eval: int, ec) -> tuple[list[float], list[float], list[float], list[float]]:
	"""SCOPE C STAGE 1 companion to sample_ics_flat: per-episode vertical initial
	conditions and plant draw — (z0, vz0, collective_cmd_frac, mass_scale).

	SINGLE source of truth for the vertical draw order, for the same reason
	sample_ics_flat is one for attitude: the CPU and Metal paths are only
	interchangeable if both draw in exactly this order.

	Uses its OWN rng stream (seed XOR a fixed salt) rather than extending
	sample_ics_flat's, so enabling the vertical axes cannot shift a single
	attitude initial condition — every banked attitude result stays reproducible
	under a stage-1 build.

	collective_cmd_frac is a FRACTION: the episode's commanded collective is
	hover·(1 + frac). mass_scale multiplies the airframe's nominal mass. Both are
	1.0-neutral at zero jitter. Returns all-neutral lists when translation is off.
	"""
	import numpy as _np
	z0: list[float] = []
	vz0: list[float] = []
	cmd: list[float] = []
	mass: list[float] = []
	if not getattr(ec, "translation", False):
		return ([0.0] * num_eval, [0.0] * num_eval, [0.0] * num_eval, [1.0] * num_eval)
	alt = float(getattr(ec, "max_initial_alt_offset_m", 0.0))
	vzb = float(getattr(ec, "max_initial_vz", 0.0))
	cjit = float(getattr(ec, "collective_cmd_jitter", 0.0))
	mjit = float(getattr(ec, "mass_jitter", 0.0))
	# 0x5AC0... = "vertical" salt; any fixed constant works, it just has to differ
	# from the attitude stream so the two never correlate.
	rng = _np.random.default_rng(int(seed) ^ 0x5AC01D)
	for _ in range(num_eval):
		ep_rng = _np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		z0.append(float(ep_rng.uniform(-alt, alt)) if alt > 0.0 else 0.0)
		vz0.append(float(ep_rng.uniform(-vzb, vzb)) if vzb > 0.0 else 0.0)
		cmd.append(float(ep_rng.uniform(-cjit, cjit)) if cjit > 0.0 else 0.0)
		mass.append(1.0 + float(ep_rng.uniform(-mjit, mjit)) if mjit > 0.0 else 1.0)
	return (z0, vz0, cmd, mass)


def sample_ics_flat(seed, num_eval: int, ec, active_axes=None) -> tuple[list[float], list[float]]:
	"""Sample num_eval initial conditions as FLAT (q0, omega0) lists for the
	GPU rollout kernel. SINGLE source of truth for the RNG draw order — the
	CPU path (_sample_initial_state per episode) and the Metal path
	(score_controllers_metal) are only interchangeable if every caller draws
	ICs in exactly this order. Was duplicated in control/evaluator.py and
	control/ga_memory.py (parity by convention only).

	active_axes (H4): override the per-episode axis mask (else ec.active_axes,
	else full 3-axis). The in-search curriculum eval passes the per-gen mask here."""
	import numpy as _np
	aa = active_axes if active_axes is not None else getattr(ec, "active_axes", (True, True, True))
	rng = _np.random.default_rng(seed)
	q0: list[float] = []
	omega0: list[float] = []
	for _ in range(num_eval):
		ep_rng = _np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		q, om = _sample_initial_state(
			ep_rng, ec.max_initial_tilt_rad, ec.max_initial_yaw_rad,
			ec.max_initial_body_rate, ec.max_initial_yaw_rate, aa,
		)
		q0 += [float(x) for x in q]
		omega0 += [float(x) for x in om]
	return q0, omega0


def _sample_initial_state(
	rng: np.random.Generator,
	max_tilt: float,
	max_yaw: float,
	max_body_rate: float,
	max_yaw_rate: float,
	active_axes: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
	"""Sample (initial_q, initial_omega) within the per-config bounds.

	H4 axis curriculum: an inactive axis has its initial tilt AND body rate zeroed.
	We DRAW always (then zero) so all-axes-active stays RNG-identical to pre-H4."""
	# Tilt: sample uniform in roll/pitch up to max_tilt, and yaw in a
	# bounded range (NOT -π to π) because target attitude is also a small
	# range and we don't want yaw error to dominate the rollout.
	r = float(rng.uniform(-max_tilt, max_tilt))
	p = float(rng.uniform(-max_tilt, max_tilt))
	y = float(rng.uniform(-max_yaw, max_yaw))
	roll = r if active_axes[0] else 0.0
	pitch = p if active_axes[1] else 0.0
	yaw = y if active_axes[2] else 0.0
	q = _euler_to_quat_xyz(roll, pitch, yaw)
	ox = float(rng.uniform(-max_body_rate, max_body_rate))
	oy = float(rng.uniform(-max_body_rate, max_body_rate))
	oz = float(rng.uniform(-max_yaw_rate, max_yaw_rate))
	omega = (
		ox if active_axes[0] else 0.0,
		oy if active_axes[1] else 0.0,
		oz if active_axes[2] else 0.0,
	)
	return q, omega


# Type alias: action_fn takes (gyro, accel, target_rpy, q) → 4 motor PWMs.
# WnnController.step takes (gyro, accel, target_attitude) but doesn't see q.
# PID needs q for attitude error. We pass both to the action_fn so the runner
# is agnostic to which controller is being evaluated.
ActionFn = Callable[
	[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]],
	tuple[float, float, float, float],
]


def run_episode(
	action_fn: ActionFn,
	sim: AttitudeSim,
	config: EpisodeConfig,
	target_attitude_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
	rng: np.random.Generator | None = None,
) -> EpisodeResult:
	"""Run one episode and return per-step + cumulative metrics.

	Args:
		action_fn:        callable that computes per-step motor PWMs. Signature:
		                  (gyro, accel, target_rpy, q) → (pwm0, pwm1, pwm2, pwm3).
		                  PID + WNN controllers wrap differently; the runner
		                  stays agnostic.
		sim:              AttitudeSim instance (will be reset).
		config:           EpisodeConfig.
		target_attitude_rpy: target (roll, pitch, yaw) in radians. Default is level.
		rng:              optional numpy RNG for reproducible initial conditions.

	Returns:
		EpisodeResult with cumulative_reward + per-episode telemetry.
	"""
	if rng is None:
		rng = np.random.default_rng()

	# Target quaternion for attitude_error comparisons.
	target_q = _euler_to_quat_xyz(*target_attitude_rpy)

	# Sample + apply initial conditions.
	init_q, init_omega = _sample_initial_state(
		rng,
		config.max_initial_tilt_rad,
		config.max_initial_yaw_rad,
		config.max_initial_body_rate,
		config.max_initial_yaw_rate,
		getattr(config, "active_axes", (True, True, True)),
	)
	sim.reset(q=list(init_q), omega=list(init_omega))

	# W2: arm this episode's weather (per-episode seed drawn from the episode
	# rng AFTER the IC draw — see apply_disturbance). Clear when the config has
	# none so a sim shared across configs can't carry stale weather.
	dist = getattr(config, "disturbance", None)
	if dist is not None:
		apply_disturbance(sim, dist, rng)
	else:
		sim.clear_disturbance()

	cumulative = 0.0
	sum_err = 0.0
	max_err = 0.0
	max_omega = 0.0
	prev_pwm: Optional[tuple[float, float, float, float]] = None
	sum_jerk = 0.0
	jerk_count = 0
	diverged = False
	steps_done = 0
	# Steady-state window: last steady_window_frac of FULL steps (matches the Metal
	# kernel's `t >= ceil(steps * 0.80)`). Isolates the residual offset (I metric).
	tail_start = math.ceil(config.steps_per_episode * (1.0 - config.steady_window_frac))
	tail_sum_err = 0.0
	tail_cnt = 0

	# Transient-speed tracking. initial_err/band_rel are set on step 0; sentinels
	# default to the FULL intended duration so "never rose/settled" scores worst.
	full_duration_s = config.steps_per_episode * config.dt
	initial_err: Optional[float] = None
	band_rel = 0.0
	band_abs = math.radians(2.0)
	rise_time_s = full_duration_s
	rise_done = False
	last_exc_abs = 0.0                  # time+dt of last step with |err| ≥ 2° band
	last_exc_rel = 0.0                  # time+dt of last step with |err| ≥ 5%-of-initial band
	itae = 0.0
	iae = 0.0
	ise = 0.0

	for step_idx in range(config.steps_per_episode):
		if sim.is_unstable():
			diverged = True
			break

		gyro, accel = sim.read_imu()
		q = sim.quaternion

		pwm = action_fn(gyro, accel, target_attitude_rpy, q)

		# Apply action to sim.
		sim.step(list(pwm))

		# Compute reward components.
		attitude_err = sim.attitude_error(target_q)
		jerk = 0.0
		if prev_pwm is not None:
			dx = pwm[0] - prev_pwm[0]
			dy = pwm[1] - prev_pwm[1]
			dz = pwm[2] - prev_pwm[2]
			dw = pwm[3] - prev_pwm[3]
			jerk = dx * dx + dy * dy + dz * dz + dw * dw
			sum_jerk += math.sqrt(jerk)
			jerk_count += 1
		prev_pwm = pwm

		reward = compute_reward(
			attitude_err,
			motor_command_jerk=jerk,
			mono_violations=0,
			lambda_smooth=config.lambda_smooth,
			lambda_mono=config.lambda_mono,
		)
		cumulative += reward
		sum_err += attitude_err
		if step_idx >= tail_start:
			tail_sum_err += attitude_err
			tail_cnt += 1

		# --- Transient-speed metrics (single pass) ---
		t_s = step_idx * config.dt
		if initial_err is None:
			initial_err = attitude_err
			band_rel = 0.05 * initial_err
		# Integral-of-error family (rectangle rule); ITAE weights late error by time.
		iae += attitude_err * config.dt
		ise += attitude_err * attitude_err * config.dt
		itae += t_s * attitude_err * config.dt
		# Rise time: first moment the error is knocked down to 10% of its initial value.
		if (not rise_done) and attitude_err <= 0.10 * initial_err:
			rise_time_s = t_s
			rise_done = True
		# Settling: remember the last time we were OUTSIDE each band → settle = last excursion + dt.
		if attitude_err >= band_abs:
			last_exc_abs = t_s + config.dt
		if attitude_err >= band_rel:
			last_exc_rel = t_s + config.dt
		if attitude_err > max_err:
			max_err = attitude_err
		omega = sim.angular_velocity
		om_norm = math.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
		if om_norm > max_omega:
			max_omega = om_norm
		steps_done = step_idx + 1

		if config.abort_attitude_error_rad is not None and attitude_err > config.abort_attitude_error_rad:
			break

	mean_err = sum_err / max(steps_done, 1)
	mean_jerk = sum_jerk / max(jerk_count, 1) if jerk_count > 0 else 0.0
	# Diverged before the tail window → no settled samples; fall back to the
	# whole-episode mean (mirrors the kernel's fallback).
	mean_steady = (tail_sum_err / tail_cnt) if tail_cnt > 0 else mean_err
	# Transient times: a diverged episode never settles → force worst-case
	# sentinels so it can't score "faster" than a controller that simply never
	# tightened (its error blows up LATE, which would fool last-excursion).
	if diverged:
		rise_time_s = full_duration_s
		settle_abs = full_duration_s
		settle_rel = full_duration_s
	else:
		settle_abs = min(last_exc_abs, full_duration_s)
		settle_rel = min(last_exc_rel, full_duration_s)
	return EpisodeResult(
		cumulative_reward=cumulative,
		mean_attitude_error_rad=mean_err,
		max_attitude_error_rad=max_err,
		max_omega_norm=max_omega,
		steps_completed=steps_done,
		diverged=diverged,
		mean_pwm_jerk=mean_jerk,
		mean_steady_error_rad=mean_steady,
		rise_time_s=rise_time_s,
		settle_time_abs2deg_s=settle_abs,
		settle_time_rel5pct_s=settle_rel,
		itae=itae,
		iae=iae,
		ise=ise,
	)


def make_pid_action_fn(pid) -> ActionFn:
	"""Adapter from AttitudePID to ActionFn signature.

	AttitudePID.step takes (q, gyro, target_rpy). The runner provides q in the
	last positional arg.
	"""
	def fn(gyro, accel, target_rpy, q):
		return pid.step(q, gyro, target_rpy)
	return fn


def make_wnn_action_fn(controller: WnnController) -> ActionFn:
	"""Adapter from WnnController to ActionFn signature."""
	def fn(gyro, accel, target_rpy, q):
		return tuple(controller.step(list(gyro), list(accel), list(target_rpy)))
	return fn


def _clip01(x: float) -> float:
	return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _clamps_tuple(clamp_per_motor: "tuple[float, ...] | float", num_motors: int) -> tuple[float, ...]:
	return (tuple(clamp_per_motor) if isinstance(clamp_per_motor, (tuple, list))
	        else (float(clamp_per_motor),) * num_motors)


def _neutral_decode(memory_mode: "str | None" = None) -> float:
	"""Untrained-cell decode anchor from the wheel, derived from the cell
	semantics — QUAD empty→0.75, TERNARY→0.5 (PLN convention), BINARY→0.5
	(antagonist-pair effective neutral). None → the QUAD default (back-compat;
	callers with a ControllerSpec should pass spec.memory_mode). ABI 12."""
	if memory_mode is None:
		from wnn.control._accel import NEUTRAL_DECODE
		return float(NEUTRAL_DECODE)
	from wnn.control._accel import neutral_decode_for_mode
	from wnn.control.evaluator import ControllerSpec
	mode_int = ControllerSpec.MEMORY_MODES[memory_mode.upper()]
	return float(neutral_decode_for_mode(mode_int))


def compose_residual(
	base_pwm, wnn_out, residual_scale: float,
	clamp_per_motor: "tuple[float, ...] | float", num_motors: int = 4,
	neutral: "float | None" = None,
) -> tuple[float, ...]:
	"""E5 residual hybrid composition (the SINGLE source of truth, shared by the
	deployed action_fn AND residual-DAGGER so they can't diverge):

	`pwm[m] = clip01( base_pwm[m] + clamp( (wnn_out[m] − neutral)·scale ) )`.

	`neutral` defaults to the wheel's NEUTRAL_DECODE — what an UNTRAINED
	(EMPTY) cell actually decodes to (0.75 under QUAD; the pre-ABI-11
	hardcoded 0.5 was WRONG and composed a hidden +clamp offset), so the
	residual is exactly 0 before training — an untrained hybrid IS the
	analytic baseline."""
	n = _neutral_decode() if neutral is None else neutral
	clamps = _clamps_tuple(clamp_per_motor, num_motors)
	out = []
	for m in range(num_motors):
		r = (wnn_out[m] - n) * residual_scale
		c = clamps[m]
		r = c if r > c else (-c if r < -c else r)
		out.append(_clip01(base_pwm[m] + r))
	return tuple(out)


def residual_train_target(
	expert_pwm, base_pwm, residual_scale: float,
	clamp_per_motor: "tuple[float, ...] | float", num_motors: int = 4,
	neutral: "float | None" = None,
) -> list[float]:
	"""Inverse of `compose_residual`: the WNN-output-space target that makes the
	learned residual reproduce `clamp(expert_pwm − base_pwm)`. Train the WNN cells
	toward `neutral + r/scale` so that `(out − neutral)·scale == r` at deployment. The
	clamp is applied to the TARGET too, so the WNN never chases authority it can't
	express — the residual-DAGGER teacher for closing 84→99.8 (expert = PID+).
	`neutral` defaults to the QUAD wheel constant; mode-aware callers pass the
	controller's own `.neutral_decode` (ABI 12)."""
	clamps = _clamps_tuple(clamp_per_motor, num_motors)
	n = _neutral_decode() if neutral is None else neutral
	tgt = []
	for m in range(num_motors):
		r = expert_pwm[m] - base_pwm[m]
		c = clamps[m]
		r = c if r > c else (-c if r < -c else r)
		tgt.append(_clip01(n + r / residual_scale))
	return tgt


def make_residual_action_fn(
	baseline_fn: ActionFn,
	residual_controller: WnnController,
	residual_scale: float = 1.0,
	clamp_per_motor: "tuple[float, ...] | float" = 0.2,
	num_motors: int = 4,
) -> ActionFn:
	"""E5 residual hybrid deployed action (see .claude/plans/e5_residual_hybrid.md):
	`action = clip01(baseline(err) + clamp(scale·(wnn − 0.5)))` via
	`compose_residual`. The analytic `baseline_fn` (PD / stock-PID) carries the
	bulk action; the learned WNN adds only the clamped residual (the integral
	action PD lacks). `clamp_per_motor` = the learn-the-clamp authority knob."""
	# Mode-derived anchor from the controller itself (ABI 12; = the wheel's
	# NEUTRAL_DECODE under QUAD, 0.5 under TERNARY/BINARY).
	neutral = float(residual_controller.neutral_decode)
	def fn(gyro, accel, target_rpy, q):
		base = baseline_fn(gyro, accel, target_rpy, q)
		res_raw = residual_controller.step(list(gyro), list(accel), list(target_rpy))
		return compose_residual(base, res_raw, residual_scale, clamp_per_motor, num_motors,
		                        neutral=neutral)
	return fn


def fitness_function(
	action_fn: ActionFn,
	sim: AttitudeSim,
	config: EpisodeConfig,
	num_episodes: int = 30,
	seed: int = 0,
) -> tuple[float, dict]:
	"""Average cumulative reward across `num_episodes` random initial conditions.

	Used as the GA's fitness signal. Reuses the existing GA pipeline in
	`src/wnn/ram/experiments/` — this function plugs into wherever the
	IDS pipeline currently computes per-genome fitness from a dataset.

	Returns:
		(mean_reward, metrics_dict): metrics_dict has aggregate stats matching
		the controller-specific JSON schema we'll write into
		`validation_summaries.threshold_metadata`.
	"""
	rng = np.random.default_rng(seed)
	results: list[EpisodeResult] = []
	for ep_idx in range(num_episodes):
		# Sub-RNG per episode so each is reproducible from (seed, ep_idx).
		ep_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
		res = run_episode(action_fn, sim, config, rng=ep_rng)
		results.append(res)

	rewards = np.array([r.cumulative_reward for r in results])
	mean_errs = np.array([r.mean_attitude_error_rad for r in results])
	steady_errs = np.array([r.mean_steady_error_rad for r in results])
	max_errs = np.array([r.max_attitude_error_rad for r in results])
	steps_done = np.array([r.steps_completed for r in results])
	diverged_count = sum(1 for r in results if r.diverged)
	steady_threshold_rad = math.radians(5.0)
	stable_count = sum(
		1 for r in results
		if (not r.diverged) and r.mean_attitude_error_rad <= steady_threshold_rad
	)

	metrics = {
		"num_episodes": num_episodes,
		"mean_reward": float(rewards.mean()),
		"std_reward": float(rewards.std()),
		"min_reward": float(rewards.min()),
		"max_reward": float(rewards.max()),
		"mean_attitude_error_rad": float(mean_errs.mean()),
		"mean_attitude_error_deg": float(math.degrees(mean_errs.mean())),
		"mean_steady_error_deg": float(math.degrees(steady_errs.mean())),
		"mean_max_attitude_error_rad": float(max_errs.mean()),
		"mean_steps_completed": float(steps_done.mean()),
		"diverged_rate": diverged_count / num_episodes,
		"stable_rate": stable_count / num_episodes,
	}
	return float(rewards.mean()), metrics


__all__ = [
	"DisturbanceConfig",
	"EpisodeConfig",
	"EpisodeResult",
	"ActionFn",
	"apply_disturbance",
	"run_episode",
	"make_pid_action_fn",
	"make_wnn_action_fn",
	"fitness_function",
]
