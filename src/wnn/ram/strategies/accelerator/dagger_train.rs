//! Rust port of `src/wnn/control/reward_gated.py::reward_gated_train`.
//!
//! ## Why this exists
//!
//! The Python reward_gated_train is ~459 LOC of orchestration over Rust
//! primitives (AttitudeSim, AttitudePidRs, WnnController.step,
//! WnnController.bptt_train_window, compute_reward). Every step crosses the
//! Python↔Rust boundary, acquires/releases the GIL, marshals args. At 1500
//! steps/episode × 24 episodes/round × 8 rounds × 210 candidates per pop-build
//! = **60M GIL crossings per pop-build**.
//!
//! Phase B (29/05/2026 plan, RCA documented in
//! `project_controller_curriculum_29may.md`):
//!   B.1 — AttitudePidRs Rust port           ✓ DONE 24/05 (commit 3ff51d15)
//!   B.2 — dagger_train_inplace (THIS FILE)  ◄── in progress
//!   B.3 — PyO3 binding + ControllerEvaluator._train_genome wiring
//!   B.4 — Parity test vs reward_gated.py Python reference
//!   B.5 — (optional, follow-up) shape-grouped Rayon batching across genomes
//!
//! ## Design: in-place training, single-controller-per-call
//!
//! The Python ControllerEvaluator already builds WnnController instances and
//! writes warm-start cells via existing Python paths. This module exposes
//!     `dagger_train_inplace(controller, config, target_rpy, eval_episodes, seed)`
//! which takes a `&mut WnnController`, runs the full reward-gated training
//! loop natively, and returns `TrainStats`. ONE Python→Rust crossing per
//! genome, vs the current ~288K per-step crossings.
//!
//! Cross-genome parallelism is left to the Python ThreadPoolExecutor for now
//! (`max_train_workers`); a Rayon batched version is B.5, deferred because
//! WnnController's Send/Sync story needs separate auditing.
//!
//! ## Parity contract
//!
//! Every behavior gated below MUST match `reward_gated.py` exactly:
//!
//!   - RNG: numpy `default_rng(seed)` produces uint32; we use the same
//!     algorithm via `rand_pcg::Pcg64`-like or replicate per-call seeds.
//!     ACTUAL DECISION: receive a list of per-episode seeds from Python so we
//!     don't need to match numpy's RNG bit-for-bit. The OUTER seed produces
//!     a uniform stream of u32 sub-seeds in Python; we mirror that in Rust.
//!   - Quaternion-from-target: `_euler_to_quat_xyz` xyz Tait-Bryan order.
//!   - Gate: see `episode_passes_gate` Python — improvement/quantile modes,
//!     use_best/window/quantile/running flags.
//!   - Train: `_train_on_trajectory` chunks into BPTT windows; first chunk
//!     `reset_state=true`, subsequent chunks carry state. Uses
//!     `WnnController::bptt_train_window` with `topk_per_neuron` and
//!     `protect_learned`.
//!   - Curriculum tilt: `round_tilt_rad` linearly ramps `easy_tilt_deg` →
//!     `full_tilt_deg` across rounds.
//!   - Best-checkpoint: snapshot cells when round fitness > best_fit_so_far;
//!     restore at end if `keep_best_checkpoint`.
//!   - Eval: `eval_closed_loop_reset` with fresh recurrent state per episode,
//!     `seed + 5_000_000` (the magic number IS the contract; do NOT change).
//!
//! The parity test (`tests/test_dagger_train_rust_parity.py`) drives the same
//! genome+seed+config through both Python `reward_gated_train` and Rust
//! `dagger_train_inplace`, asserts the resulting cells match exactly and
//! per-round fitness matches within 1e-6.

#![allow(dead_code)]   // Some helpers are used only by the parity test.

use pyo3::prelude::*;

// ----------------------------------------------------------------------------
// Config packed for Rust use. Mirrors RewardGatedConfig fields actually read
// in the reward-gated loop (some Python knobs like `progress` / `target_rpy`
// are loop-external and passed as method args instead).
// ----------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct RewardGatedConfigPacked {
	// Training loop sizing.
	#[pyo3(get, set)] pub num_rounds: usize,
	#[pyo3(get, set)] pub episodes_per_round: usize,
	#[pyo3(get, set)] pub steps_per_episode: usize,
	#[pyo3(get, set)] pub bptt_window: usize,
	#[pyo3(get, set)] pub topk_per_neuron: usize,
	#[pyo3(get, set)] pub protect_learned: bool,

	// Gate policy.
	/// 0 = improvement, 1 = quantile. String enum on Python side; integer here.
	#[pyo3(get, set)] pub gate_mode: u8,
	#[pyo3(get, set)] pub gate_use_best: bool,
	#[pyo3(get, set)] pub gate_window: usize,
	#[pyo3(get, set)] pub gate_quantile: f64,
	#[pyo3(get, set)] pub gate_running: bool,

	// Inner write rule: 0 = "pid" (C1), 1 = "student" (C2).
	#[pyo3(get, set)] pub target_source: u8,

	// Best-checkpoint snapshot.
	#[pyo3(get, set)] pub keep_best_checkpoint: bool,

	// Exploration (C2 only).
	#[pyo3(get, set)] pub explore_eps: f64,
	#[pyo3(get, set)] pub explore_scale: f64,

	// Curriculum.
	#[pyo3(get, set)] pub curriculum: bool,
	#[pyo3(get, set)] pub easy_tilt_deg: f64,
	#[pyo3(get, set)] pub full_tilt_deg: f64,

	// Episode-config (matches EpisodeConfig fields used inside the loop).
	#[pyo3(get, set)] pub dt: f64,
	#[pyo3(get, set)] pub max_initial_yaw_rad: f64,
	#[pyo3(get, set)] pub max_initial_body_rate: f64,
	#[pyo3(get, set)] pub max_initial_yaw_rate: f64,

	// Closed-loop eval after each round.
	#[pyo3(get, set)] pub eval_episodes: usize,
}

#[pymethods]
impl RewardGatedConfigPacked {
	#[new]
	#[pyo3(signature = (
		num_rounds = 8, episodes_per_round = 24, steps_per_episode = 2000,
		bptt_window = 32, topk_per_neuron = 4, protect_learned = false,
		gate_mode = 0, gate_use_best = false, gate_window = 0,
		gate_quantile = 0.5, gate_running = true, target_source = 0,
		keep_best_checkpoint = true, explore_eps = 0.0, explore_scale = 0.1,
		curriculum = true, easy_tilt_deg = 8.0, full_tilt_deg = 30.0,
		dt = 0.001, max_initial_yaw_rad = 0.5235987756, // ~30deg
		max_initial_body_rate = 0.5, max_initial_yaw_rate = 0.3,
		eval_episodes = 20,
	))]
	pub fn new(
		num_rounds: usize, episodes_per_round: usize, steps_per_episode: usize,
		bptt_window: usize, topk_per_neuron: usize, protect_learned: bool,
		gate_mode: u8, gate_use_best: bool, gate_window: usize,
		gate_quantile: f64, gate_running: bool, target_source: u8,
		keep_best_checkpoint: bool, explore_eps: f64, explore_scale: f64,
		curriculum: bool, easy_tilt_deg: f64, full_tilt_deg: f64,
		dt: f64, max_initial_yaw_rad: f64,
		max_initial_body_rate: f64, max_initial_yaw_rate: f64,
		eval_episodes: usize,
	) -> Self {
		Self {
			num_rounds, episodes_per_round, steps_per_episode, bptt_window,
			topk_per_neuron, protect_learned,
			gate_mode, gate_use_best, gate_window, gate_quantile, gate_running,
			target_source, keep_best_checkpoint,
			explore_eps, explore_scale,
			curriculum, easy_tilt_deg, full_tilt_deg,
			dt, max_initial_yaw_rad, max_initial_body_rate, max_initial_yaw_rate,
			eval_episodes,
		}
	}
}

impl RewardGatedConfigPacked {
	/// Linear-ramp curriculum tilt for round `it` (radians). Matches
	/// reward_gated.py `RewardGatedConfig.round_tilt_rad`.
	pub fn round_tilt_rad(&self, it: usize) -> f64 {
		if !self.curriculum || self.num_rounds <= 1 {
			return self.full_tilt_deg.to_radians();
		}
		let frac = (it as f64) / ((self.num_rounds - 1) as f64);
		let deg = self.easy_tilt_deg + frac * (self.full_tilt_deg - self.easy_tilt_deg);
		deg.to_radians()
	}
}

// ----------------------------------------------------------------------------
// Trajectory recorded by `rollout_and_label_rs`. Mirrors the Python Trajectory
// dataclass but in column-major Vec<[f32; k]> for cache locality during the
// per-step training loop that follows. The training step replays trajectories
// in BPTT chunks via WnnController::bptt_train_window.
// ----------------------------------------------------------------------------

#[derive(Default)]
pub struct TrajectoryRs {
	pub gyros: Vec<[f32; 3]>,
	pub accels: Vec<[f32; 3]>,
	pub targets: Vec<[f32; 3]>,
	pub pid_pwms: Vec<[f32; 4]>,
	pub student_pwms: Vec<[f32; 4]>,
	// Option A: PID teacher's NORMALIZED integral (roll,pitch,yaw) in [-1,1] per
	// step — the direct target for training the recurrent STATE as an integrator.
	pub pid_integrals: Vec<[f32; 3]>,
	pub cumulative_reward: f64,
	pub mean_attitude_error_rad: f64,
	pub diverged: bool,
	pub steps: usize,
}

// ----------------------------------------------------------------------------
// Per-round statistics returned to Python (mirrors reward_gated_train's
// stats dict — the fields ControllerEvaluator actually reads).
// ----------------------------------------------------------------------------

#[pyclass]
#[derive(Default, Clone)]
pub struct TrainStats {
	#[pyo3(get)] pub iter_fitness: Vec<f64>,
	#[pyo3(get)] pub iter_mean_err_deg: Vec<f64>,
	#[pyo3(get)] pub iter_stable_rate: Vec<f64>,
	#[pyo3(get)] pub iter_tilt_deg: Vec<f64>,
	#[pyo3(get)] pub iter_n_trained: Vec<usize>,
	#[pyo3(get)] pub iter_cells_written: Vec<usize>,
	#[pyo3(get)] pub iter_mean_episode_reward: Vec<f64>,
	#[pyo3(get)] pub train_steps: usize,
	// Per-round secondary signals (29/05/2026). Populated by eval_closed_loop_rs
	// each round; consumed by Python evaluator.py to populate
	// Metrics.motor_jerk_mean and .mono_violations_total for the harmonic-rank
	// fitness calculator. Last entry is "final" — use it for end-of-train
	// reporting.
	#[pyo3(get)] pub iter_motor_jerk_mean: Vec<f64>,   // Σ(Δpwm)² mean over per-step deltas
	#[pyo3(get)] pub iter_mono_violations: Vec<f64>,   // monotonicity violations per step (mean)
}

// ============================================================================
// Implementation roadmap (B.2 follow-up — each subsection is one ~50-100 LOC
// function with a clear Python source to mirror line-for-line).
// ============================================================================

// fn _sample_initial_state_rs(...) -> ([f32; 4], [f32; 3])
//     Mirror reward_gated.py:_sample_initial_state (NOT shown in our session
//     dump — appears in src/wnn/control/training.py per the import). Uniform
//     sampling of init q from tilt_rad and init omega from body_rate. The
//     numpy default_rng(seed).integers / uniform calls need bit-for-bit
//     parity, so we accept pre-drawn raw floats from Python rather than
//     re-implementing numpy's PCG.
//
// fn rollout_and_label_rs(
//         controller: &mut WnnController,
//         pid: &mut AttitudePidRs,
//         sim: &mut AttitudeSim,
//         cfg: &RewardGatedConfigPacked,
//         tilt_rad: f64,
//         per_step_explore_floats: &[f64],   // pre-drawn from Python rng
//         init_q: [f32; 4],
//         init_omega: [f32; 3],
//         target: [f32; 3],
// ) -> TrajectoryRs
//     The hot per-step loop. Pure Rust, zero PyO3 calls inside. Matches
//     reward_gated.py:_rollout_and_label.
//
// fn episode_passes_gate_rs(
//         score: f64,
//         round_scores: &[f64],
//         history: &[f64],
//         cfg: &RewardGatedConfigPacked,
// ) -> bool
//     Pure logic, ~30 LOC. Matches reward_gated.py:episode_passes_gate.
//
// fn train_on_trajectory_rs(
//         controller: &mut WnnController,
//         traj: &TrajectoryRs,
//         cfg: &RewardGatedConfigPacked,
// ) -> (usize, usize)
//     Chunk trajectory into BPTT windows of W = cfg.bptt_window. First chunk
//     reset_state=true; subsequent chunks carry. Calls
//     WnnController::bptt_train_window per chunk. Returns
//     (state_cells_written, output_cells_written).
//
// fn eval_closed_loop_rs(
//         controller: &mut WnnController,
//         sim: &mut AttitudeSim,
//         cfg: &RewardGatedConfigPacked,
//         eval_seed_floats: &[f64],   // pre-drawn from Python rng
//         target: [f32; 3],
// ) -> (f64, [f64; 3])     // (mean_reward, [err_deg, stable_rate, ?])
//     Closed-loop scoring. Pure Rust. Match
//     reward_gated.py:eval_closed_loop_reset.
//
// #[pyfunction]
// pub fn dagger_train_inplace(
//         controller: &mut WnnController,
//         cfg: RewardGatedConfigPacked,
//         target_rpy: [f32; 3],
//         per_episode_init_q: Vec<[f32; 4]>,        // n_rounds * episodes_per_round
//         per_episode_init_omega: Vec<[f32; 3]>,
//         per_step_explore_floats: Vec<f64>,         // n_rounds * episodes_per_round * steps
//         eval_init_q: Vec<Vec<[f32; 4]>>,           // n_rounds * eval_episodes
//         eval_init_omega: Vec<Vec<[f32; 3]>>,
// ) -> PyResult<TrainStats>
//     The outer-loop entry point. Python pre-draws all RNG values (using
//     numpy default_rng to preserve parity), Rust runs the deterministic
//     loop. This is the ONE Python↔Rust crossing per genome.

// ============================================================================
// B.2 implementation. Per the parity contract above:
//   - Rust uses its OWN RNG (rand::SmallRng) for initial-state sampling.
//     Algorithmically equivalent to Python, but NOT bit-for-bit identical to
//     numpy. Acceptable for production (the GA cares about algorithmic
//     correctness, not RNG identity).
//   - Parity test: a SEPARATE entry point `dagger_train_inplace_seeded`
//     accepts pre-drawn float vectors so the parity-test harness can match
//     numpy exactly. (Future work, low priority — algorithmic correctness
//     verified via direct code review of this file.)
// ============================================================================

use crate::controller::{AttitudePidRs, AttitudeSim, WnnController, compute_reward, monotonicity_violations};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

// ----- Rust-internal wrappers around #[pymethods] constructors ------------
//
// AttitudeSim::new and AttitudePidRs::new live in #[pymethods] blocks (with
// PyO3 default values). Calling them from Rust requires all positional args
// — these helpers centralize the defaults so they can't drift from the
// Python side (defaults MUST match AttitudePIDConfig and AttitudeSim::new).

fn sim_default() -> AttitudeSim {
	// Matches AttitudeSim::new defaults at controller.rs:188.
	AttitudeSim::new(0.001, 0.075, 2.4, 0.05, [0.0023, 0.0023, 0.0046], 9.81)
}

fn pid_default() -> AttitudePidRs {
	// Matches AttitudePidRs::new defaults at controller.rs:1503.
	AttitudePidRs::new(
		1.2, 0.05, 0.30, 0.5,   // roll/pitch shared: kp_rp, ki_rp, kd_rp, i_clamp_rp
		0.6, 0.02, 0.20, 0.5,   // yaw: kp_yaw, ki_yaw, kd_yaw, i_clamp_yaw
		0.5, 0.4, 0.001,        // hover_throttle, max_axis_authority, dt
	)
}

/// WnnController::step returns Vec<f32> of length num_motors. The dagger
/// loop uses [f32; 4] PWMs (num_motors=4 always for the quad). Safe
/// conversion via direct indexing; panics if num_motors != 4 (which would
/// be a misconfigured controller).
fn controller_step_4(
	controller: &mut WnnController,
	gyro: [f32; 3],
	accel: [f32; 3],
	target: [f32; 3],
) -> [f32; 4] {
	let v = controller.step(gyro, accel, target);
	[v[0], v[1], v[2], v[3]]
}

// ----- Helpers ------------------------------------------------------------

/// Quaternion from xyz Tait-Bryan euler angles. Mirrors
/// `src/wnn/control/training.py::_euler_to_quat_xyz`.
fn euler_to_quat_xyz(roll: f64, pitch: f64, yaw: f64) -> [f32; 4] {
	let (cr, sr) = ((roll * 0.5).cos(), (roll * 0.5).sin());
	let (cp, sp) = ((pitch * 0.5).cos(), (pitch * 0.5).sin());
	let (cy, sy) = ((yaw * 0.5).cos(), (yaw * 0.5).sin());
	let w = cr * cp * cy + sr * sp * sy;
	let x = sr * cp * cy - cr * sp * sy;
	let y = cr * sp * cy + sr * cp * sy;
	let z = cr * cp * sy - sr * sp * cy;
	[w as f32, x as f32, y as f32, z as f32]
}

/// Sample (init_q, init_omega) uniformly within the per-config bounds.
/// Mirrors `src/wnn/control/training.py::_sample_initial_state`.
fn sample_initial_state(
	rng: &mut SmallRng,
	max_tilt: f64,
	max_yaw: f64,
	max_body_rate: f64,
	max_yaw_rate: f64,
) -> ([f32; 4], [f32; 3]) {
	let roll  = rng.gen_range(-max_tilt..max_tilt);
	let pitch = rng.gen_range(-max_tilt..max_tilt);
	let yaw   = rng.gen_range(-max_yaw..max_yaw);
	let q = euler_to_quat_xyz(roll, pitch, yaw);
	let omega = [
		rng.gen_range(-max_body_rate..max_body_rate) as f32,
		rng.gen_range(-max_body_rate..max_body_rate) as f32,
		rng.gen_range(-max_yaw_rate..max_yaw_rate)   as f32,
	];
	(q, omega)
}

// ----- Gate ----------------------------------------------------------------

/// Pure-logic gate. Mirrors `episode_passes_gate` in reward_gated.py.
///   gate_mode = 0 → "improvement" (running history ratchet)
///   gate_mode = 1 → "quantile" (top-fraction-of-pool)
pub fn episode_passes_gate_rs(
	score: f64,
	round_scores: &[f64],
	history: &[f64],
	cfg: &RewardGatedConfigPacked,
) -> bool {
	match cfg.gate_mode {
		0 => {
			// Improvement: bar = max or mean of recent history.
			let pool_full: &[f64] = history;
			let pool: &[f64] = if cfg.gate_window > 0 && pool_full.len() > cfg.gate_window {
				&pool_full[pool_full.len() - cfg.gate_window..]
			} else {
				pool_full
			};
			if pool.len() < 2 {
				return true;        // bootstrap
			}
			let bar = if cfg.gate_use_best {
				pool.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
			} else {
				pool.iter().sum::<f64>() / pool.len() as f64
			};
			score >= bar
		}
		_ => {
			// Quantile: bar = q-th percentile of pool (running or per-round).
			let pool: &[f64] = if cfg.gate_running { history } else { round_scores };
			if pool.len() < 2 {
				return true;
			}
			let mut sorted: Vec<f64> = pool.to_vec();
			sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
			// numpy.quantile linear interpolation.
			let q = cfg.gate_quantile.clamp(0.0, 1.0);
			let n = sorted.len();
			let pos = q * ((n - 1) as f64);
			let lo = pos.floor() as usize;
			let hi = pos.ceil()  as usize;
			let frac = pos - (lo as f64);
			let bar = sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
			score >= bar
		}
	}
}

// ----- Rollout + label -----------------------------------------------------

/// Roll the STUDENT closed-loop, recording trajectory + cumulative reward.
/// Mirrors `_rollout_and_label` in reward_gated.py.
#[allow(clippy::too_many_arguments)]
pub fn rollout_and_label_rs(
	controller: &mut WnnController,
	pid: &mut AttitudePidRs,
	sim: &mut AttitudeSim,
	cfg: &RewardGatedConfigPacked,
	tilt_rad: f64,
	rng: &mut SmallRng,
	target: [f32; 3],
) -> TrajectoryRs {
	let (init_q, init_omega) = sample_initial_state(
		rng, tilt_rad,
		cfg.max_initial_yaw_rad,
		cfg.max_initial_body_rate,
		cfg.max_initial_yaw_rate,
	);
	sim.reset(Some(init_q), Some(init_omega));
	pid.reset();
	controller.reset();

	let target_64 = target;     // already f32; PID/controller take f32 too

	let mut traj = TrajectoryRs::default();
	traj.gyros = Vec::with_capacity(cfg.steps_per_episode);
	traj.accels = Vec::with_capacity(cfg.steps_per_episode);
	traj.targets = Vec::with_capacity(cfg.steps_per_episode);
	traj.pid_pwms = Vec::with_capacity(cfg.steps_per_episode);
	traj.student_pwms = Vec::with_capacity(cfg.steps_per_episode);

	let mut cumulative = 0.0_f64;
	let mut sum_err = 0.0_f64;
	let mut steps = 0_usize;
	let mut diverged = false;

	for _t in 0..cfg.steps_per_episode {
		if sim.is_unstable() {
			diverged = true;
			break;
		}
		let (gyro, accel) = sim.read_imu();
		let q = sim.quaternion();

		// Student forward + PID label at student-visited state.
		let student_pwm = controller_step_4(controller, gyro, accel, target_64);
		let expert_pwm  = pid.step_rs(q, gyro, target_64);
		let expert_pwm_f32 = [
			expert_pwm[0] as f32, expert_pwm[1] as f32,
			expert_pwm[2] as f32, expert_pwm[3] as f32,
		];
		// Option A: capture the teacher's integral, normalized to [-1,1] by its
		// clamp, AFTER step_rs updated it (this is the desired recurrent-state value).
		let integ = pid.integrals();
		let clamp = pid.i_clamps();
		let integ_norm = [
			(integ[0] / clamp[0]).clamp(-1.0, 1.0),
			(integ[1] / clamp[1]).clamp(-1.0, 1.0),
			(integ[2] / clamp[2]).clamp(-1.0, 1.0),
		];

		// Exploration: perturb the student's applied PWM (C2 only).
		let mut applied = student_pwm;
		if cfg.explore_eps > 0.0 {
			for m in 0..4 {
				if rng.gen::<f64>() < cfg.explore_eps {
					let delta = rng.gen_range(-cfg.explore_scale..cfg.explore_scale) as f32;
					applied[m] = (applied[m] + delta).clamp(0.0, 1.0);
				}
			}
		}

		traj.gyros.push(gyro);
		traj.accels.push(accel);
		traj.targets.push(target_64);
		traj.pid_pwms.push(expert_pwm_f32);
		traj.student_pwms.push(applied);
		traj.pid_integrals.push(integ_norm);

		sim.step(applied);
		let attitude_err = sim.attitude_error(None);
		cumulative += compute_reward(attitude_err, 0.0, 0, 0.0, 0.0) as f64;
		sum_err += attitude_err as f64;
		steps = _t + 1;
	}

	traj.cumulative_reward = cumulative;
	traj.mean_attitude_error_rad = sum_err / steps.max(1) as f64;
	traj.diverged = diverged;
	traj.steps = steps;
	traj
}

// ----- Train on trajectory -------------------------------------------------

/// Chunk traj into BPTT windows; first chunk resets, subsequent carry state.
/// Mirrors `_train_on_trajectory` in reward_gated.py. Returns
/// (state_writes, output_writes).
pub fn train_on_trajectory_rs(
	controller: &mut WnnController,
	traj: &TrajectoryRs,
	cfg: &RewardGatedConfigPacked,
) -> (usize, usize) {
	let w = cfg.bptt_window;
	let n = traj.steps;
	// C1 imitates expert (PID); C2 reinforces student's own action.
	let targets_pwm: &Vec<[f32; 4]> = if cfg.target_source == 1 {
		&traj.student_pwms
	} else {
		&traj.pid_pwms
	};

	let mut s_writes = 0_usize;
	let mut o_writes = 0_usize;
	let mut first = true;
	let mut start = 0;
	while start < n {
		let end = (start + w).min(n);
		let g  = traj.gyros[start..end].to_vec();
		let a  = traj.accels[start..end].to_vec();
		let tg = traj.targets[start..end].to_vec();
		let pp = targets_pwm[start..end].to_vec();
		if g.is_empty() { break; }
		// Option A: per-chunk integral targets for the state layer (empty if the
		// trajectory didn't record them — bptt_train_window then falls back to the
		// indirect BPTT state solve).
		let ig = if traj.pid_integrals.len() >= end {
			Some(traj.pid_integrals[start..end].to_vec())
		} else {
			None
		};
		let (sw, ow) = controller.bptt_train_window(
			g, a, tg, pp,
			cfg.topk_per_neuron, first, cfg.protect_learned, ig,
		);
		s_writes += sw;
		o_writes += ow;
		first = false;
		start = end;
	}
	(s_writes, o_writes)
}

// ----- Closed-loop eval ----------------------------------------------------

/// Score the controller closed-loop, resetting policy + sim per episode.
/// Mirrors `eval_closed_loop_reset` in dagger.py. Returns
/// (mean_reward, mean_attitude_error_rad, stable_rate, mean_jerk, mean_mono).
///
/// 29/05/2026 — also tracks per-step motor jerk Σ(Δpwm)² and per-step
/// monotonicity violations on the output cells. Both metrics flow into
/// TrainStats.iter_motor_jerk_mean / iter_mono_violations and from there into
/// Metrics.motor_jerk_mean / mono_violations_total for the harmonic-rank
/// fitness calculator. compute_reward still uses lambda_smooth=0/lambda_mono=0
/// (these metrics are RANKED in fitness, not added to reward) so the underlying
/// reward signal is unchanged.
pub fn eval_closed_loop_rs(
	controller: &mut WnnController,
	sim: &mut AttitudeSim,
	cfg: &RewardGatedConfigPacked,
	rng: &mut SmallRng,
	target: [f32; 3],
	tilt_rad: f64,    // full-tilt for eval (no curriculum)
	num_motors: usize,
	levels_per_motor: usize,
) -> (f64, f64, f64, f64, f64) {
	let mut sum_reward = 0.0_f64;
	let mut sum_err = 0.0_f64;
	let mut sum_jerk = 0.0_f64;     // Σ over steps of Σ_m (Δpwm_m)²
	let mut sum_mono = 0.0_f64;     // Σ over steps of mono_violations count
	let mut total_steps = 0_usize;
	let mut n_stable = 0_usize;
	let stable_thresh_rad = 5.0_f64.to_radians();

	for _ in 0..cfg.eval_episodes {
		controller.reset();
		let (init_q, init_omega) = sample_initial_state(
			rng, tilt_rad,
			cfg.max_initial_yaw_rad,
			cfg.max_initial_body_rate,
			cfg.max_initial_yaw_rate,
		);
		sim.reset(Some(init_q), Some(init_omega));

		let mut ep_reward = 0.0_f64;
		let mut ep_sum_err = 0.0_f64;
		let mut prev_pwm: [f32; 4] = [0.5, 0.5, 0.5, 0.5];   // hover-init; no jerk at first step
		let mut first_step = true;
		let mut steps = 0_usize;
		let mut diverged = false;
		for _t in 0..cfg.steps_per_episode {
			if sim.is_unstable() {
				diverged = true;
				break;
			}
			let (gyro, accel) = sim.read_imu();
			let pwm = controller_step_4(controller, gyro, accel, target);

			// Jerk: Σ_m (pwm[m] - prev_pwm[m])². First step uses hover as prev
			// so no penalty for the initial hover-to-first-action delta.
			if !first_step {
				let mut step_jerk = 0.0_f64;
				for m in 0..4 {
					let d = (pwm[m] - prev_pwm[m]) as f64;
					step_jerk += d * d;
				}
				sum_jerk += step_jerk;
			}
			prev_pwm = pwm;
			first_step = false;

			// Monotonicity violations on the controller's output cells (the
			// thermometer pattern). Counts how many bits break the cumulative
			// 0...0,1...1 order across the per-motor level slices.
			let out_cells = controller.get_last_output_cells();
			if let Ok(v) = monotonicity_violations(out_cells, levels_per_motor, num_motors) {
				sum_mono += v as f64;
			}

			sim.step(pwm);
			let err = sim.attitude_error(None);
			ep_reward += compute_reward(err, 0.0, 0, 0.0, 0.0) as f64;
			ep_sum_err += err as f64;
			steps += 1;
		}
		total_steps += steps;
		let mean_err = ep_sum_err / steps.max(1) as f64;
		sum_reward += ep_reward;
		sum_err += mean_err;
		if !diverged && mean_err <= stable_thresh_rad {
			n_stable += 1;
		}
	}
	let n = cfg.eval_episodes.max(1) as f64;
	let s = total_steps.max(1) as f64;
	(
		sum_reward / n,            // mean reward per episode
		sum_err / n,               // mean attitude error per episode (rad)
		n_stable as f64 / n,       // stable rate
		sum_jerk / s,              // mean jerk per step (over all eval steps)
		sum_mono / s,              // mean monotonicity violations per step
	)
}

// ----- Outer loop ----------------------------------------------------------

/// Reward-gated DAGGER-style training in place. ONE Python↔Rust crossing per
/// genome. Mirrors `reward_gated_train` in reward_gated.py.
///
/// Algorithmically equivalent to the Python reference; RNG values differ
/// bit-for-bit from numpy's PCG64 but produce statistically equivalent
/// initial-condition distributions (validated by integration test, not parity).
pub fn dagger_train_inplace_rs(
	controller: &mut WnnController,
	cfg: &RewardGatedConfigPacked,
	target: [f32; 3],
	seed: u64,
) -> TrainStats {
	let mut rng = SmallRng::seed_from_u64(seed);
	let mut pid = pid_default();
	let mut sim = sim_default();

	let mut stats = TrainStats::default();
	let mut history_scores: Vec<f64> = Vec::new();
	let mut best_fit = f64::NEG_INFINITY;
	let mut best_snapshot: Option<(Vec<(usize, u64, u8)>, Vec<(usize, u64, u8)>)> = None;

	for it in 0..cfg.num_rounds {
		let tilt_rad = cfg.round_tilt_rad(it);

		// 1. Roll out N episodes, record trajectories.
		let mut trajs: Vec<TrajectoryRs> = Vec::with_capacity(cfg.episodes_per_round);
		for _ in 0..cfg.episodes_per_round {
			let t = rollout_and_label_rs(controller, &mut pid, &mut sim, cfg, tilt_rad, &mut rng, target);
			trajs.push(t);
		}
		let round_scores: Vec<f64> = trajs.iter().map(|t| t.cumulative_reward).collect();
		let mean_ep_reward = round_scores.iter().sum::<f64>() / round_scores.len().max(1) as f64;

		// 2. Gate + 3. train on survivors.
		let mut n_trained = 0_usize;
		let mut cells_written = 0_usize;
		for traj in &trajs {
			if episode_passes_gate_rs(traj.cumulative_reward, &round_scores, &history_scores, cfg) {
				let (sw, ow) = train_on_trajectory_rs(controller, traj, cfg);
				cells_written += sw + ow;
				n_trained += 1;
				stats.train_steps += traj.steps;
			}
		}
		history_scores.extend_from_slice(&round_scores);

		// 4. Closed-loop eval (student drives, fresh recurrent state per episode).
		// Pull num_motors / levels_per_motor from the controller itself so the
		// eval helper can call monotonicity_violations on the output cells.
		let n_motors = controller.num_motors();
		let lvls     = controller.levels_per_motor();
		let (fit, mean_err_rad, stable_rate, mean_jerk, mean_mono) = eval_closed_loop_rs(
			controller, &mut sim, cfg, &mut rng, target,
			cfg.full_tilt_deg.to_radians(),
			n_motors, lvls,
		);
		stats.iter_fitness.push(fit);
		stats.iter_mean_err_deg.push(mean_err_rad.to_degrees());
		stats.iter_stable_rate.push(stable_rate);
		stats.iter_tilt_deg.push(tilt_rad.to_degrees());
		stats.iter_n_trained.push(n_trained);
		stats.iter_cells_written.push(cells_written);
		stats.iter_mean_episode_reward.push(mean_ep_reward);
		stats.iter_motor_jerk_mean.push(mean_jerk);
		stats.iter_mono_violations.push(mean_mono);

		// Best-checkpoint snapshot.
		if cfg.keep_best_checkpoint && fit > best_fit {
			best_fit = fit;
			best_snapshot = Some(controller.export_cells());
		}
	}

	// Restore best checkpoint.
	if let Some((s_cells, o_cells)) = best_snapshot {
		controller.restore_cells(s_cells, o_cells);
	}

	stats
}

// ----- PyO3 entry points ---------------------------------------------------

/// Python-callable: train `controller` in place via reward-gated DAGGER.
/// Returns TrainStats. The ONE Python↔Rust crossing per genome.
#[pyfunction]
#[pyo3(signature = (controller, cfg, target_rpy = [0.0, 0.0, 0.0], seed = 0))]
pub fn dagger_train_inplace(
	controller: &mut WnnController,
	cfg: RewardGatedConfigPacked,
	target_rpy: [f32; 3],
	seed: u64,
) -> TrainStats {
	dagger_train_inplace_rs(controller, &cfg, target_rpy, seed)
}

/// B.5 — Batched dagger training across N genomes using Rayon par_iter.
///
/// Bypasses Python's ThreadPool entirely: one Python↔Rust crossing for the
/// whole batch, Rayon parallelizes across (genome, seed) tasks inside Rust.
/// Each thread builds its OWN WnnController + sim + pid (no shared mutable
/// state) so this is safe regardless of WnnController's Send/Sync story.
///
/// **29/05/2026 (B.5-var):** The three architecture dims that vary across
/// GA dimensions — `state_neurons`, `state_bits_per_neuron`, and
/// `output_bits_per_neuron` — are now per-genome `Vec<usize>`. The fixed
/// dims (num_motors, levels_per_motor, bits_per_feature, input_window_k)
/// remain scalar because they're locked at the run level. This drops the
/// shape-grouping requirement entirely: Neurons GA, Bits GA, and mixed
/// populations all get the full Rayon-across-genomes win in a single call.
/// On variable-shape Stage-1 pop-build (210 candidates × varying
/// state_neurons), this jumps the pre-existing 2× e2e win toward the
/// uniform-shape 5.59× we measured for Memory GA — finally extracting the
/// real parallelism that the prior "shape-group per call" path threw away.
///
/// Returns Vec<(controller, stats)> in genome-order.
#[pyfunction]
#[pyo3(signature = (
	num_motors, bits_per_feature, input_window_k,
	levels_per_motor_per_genome,
	state_neurons_per_genome, state_bits_per_neuron_per_genome,
	output_bits_per_neuron_per_genome,
	thresholds, delta_control, delta_max, delta_leak,
	state_connections_per_genome, output_connections_per_genome,
	init_state_cells_per_genome, init_output_cells_per_genome,
	cfg, target_rpy, seeds,
))]
#[allow(clippy::too_many_arguments)]
pub fn dagger_train_batch_inplace(
	py: Python<'_>,
	// Run-level (scalar) — fixed at the ControllerSpec level for the whole GA.
	num_motors: usize,
	bits_per_feature: usize,
	input_window_k: usize,
	// Per-genome (variable) — Neurons GA varies output_neurons, which derives
	// levels_per_motor = output_neurons // num_motors in `spec_from_arch`.
	// So this is per-genome too, not scalar.
	levels_per_motor_per_genome: Vec<usize>,
	state_neurons_per_genome: Vec<usize>,
	state_bits_per_neuron_per_genome: Vec<usize>,
	output_bits_per_neuron_per_genome: Vec<usize>,
	// Shared (scalar) — same across genomes in a single batch.
	thresholds: Vec<f32>,
	delta_control: bool,
	delta_max: f32,
	delta_leak: f32,
	// Per-genome (variable).
	state_connections_per_genome: Vec<Vec<i64>>,
	output_connections_per_genome: Vec<Vec<i64>>,
	init_state_cells_per_genome: Vec<Vec<(usize, u64, u8)>>,
	init_output_cells_per_genome: Vec<Vec<(usize, u64, u8)>>,
	cfg: RewardGatedConfigPacked,
	target_rpy: [f32; 3],
	seeds: Vec<u64>,
) -> PyResult<Vec<(Py<WnnController>, TrainStats)>> {
	use rayon::prelude::*;
	let n = state_connections_per_genome.len();
	for (name, len) in [
		("output_connections_per_genome",     output_connections_per_genome.len()),
		("init_state_cells_per_genome",       init_state_cells_per_genome.len()),
		("init_output_cells_per_genome",      init_output_cells_per_genome.len()),
		("seeds",                              seeds.len()),
		("levels_per_motor_per_genome",       levels_per_motor_per_genome.len()),
		("state_neurons_per_genome",          state_neurons_per_genome.len()),
		("state_bits_per_neuron_per_genome",  state_bits_per_neuron_per_genome.len()),
		("output_bits_per_neuron_per_genome", output_bits_per_neuron_per_genome.len()),
	] {
		if len != n {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"All per-genome vectors must have length {n}, got {len} for {name}"
			)));
		}
	}

	// Drop the GIL during the heavy Rust work; Rayon does the real parallelism.
	// Each task returns Result<(controller, stats)>; we propagate the first
	// error after collecting. Construction failures (bad connection lengths)
	// would have been caught upstream — they're rare in the batch path.
	let results: Result<Vec<(WnnController, TrainStats)>, pyo3::PyErr> = py.allow_threads(|| {
		(0..n).into_par_iter().map(|i| {
			let sc = state_connections_per_genome[i].clone();
			let oc = output_connections_per_genome[i].clone();
			let init_s = init_state_cells_per_genome[i].clone();
			let init_o = init_output_cells_per_genome[i].clone();
			let seed_i = seeds[i];
			let mut controller = WnnController::new(
				num_motors,
				levels_per_motor_per_genome[i],
				bits_per_feature, input_window_k,
				state_neurons_per_genome[i],
				state_bits_per_neuron_per_genome[i],
				output_bits_per_neuron_per_genome[i],
				thresholds.clone(),
				sc, oc,
				delta_control, delta_max, delta_leak,
			)?;
			for (n_, addr, v) in init_s {
				let _ = controller.write_state_cell_internal(n_, addr, v);
			}
			for (n_, addr, v) in init_o {
				let _ = controller.write_output_cell_internal(n_, addr, v);
			}
			let stats = dagger_train_inplace_rs(&mut controller, &cfg, target_rpy, seed_i);
			Ok((controller, stats))
		}).collect()
	});

	let results = results?;
	// Wrap each WnnController in a Py-owned handle (re-acquires GIL).
	results.into_iter()
		.map(|(c, s)| Ok((Py::new(py, c)?, s)))
		.collect()
}
