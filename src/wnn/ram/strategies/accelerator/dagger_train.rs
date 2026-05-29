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

#![allow(dead_code)]   // Scaffold — function bodies land in follow-up.

use pyo3::prelude::*;

// NOTE: imports for the Rust primitives (AttitudeSim, AttitudePidRs,
// WnnController, compute_reward) will be added when B.2-impl lands function
// bodies. Leaving them out keeps the scaffold warning-clean.

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

// TODO(b2-impl): land the function bodies above in commit B.2-impl. Tests
// live at tests/test_dagger_train_rust_parity.py (also to write).
