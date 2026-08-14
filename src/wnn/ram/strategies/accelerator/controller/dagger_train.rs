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

	// DAGGER teacher (the expert whose action the WNN imitates): 0=PID, 1=LQR,
	// 2=MPC, 3=LQI. LQR/MPC are optimal-control teachers (controller/optimal.rs);
	// they are memoryless so their Option-A integral target is zero. LQI is the
	// STATEFUL optimal teacher (integral-augmented LQR) — integral target active.
	#[pyo3(get, set)] pub teacher: u8,
	// Hybrid teachers (both empty = OFF → the scalar `teacher` above, the
	// bit-exact legacy path). `teacher_schedule` = per-ROUND curriculum
	// (indexed min(round, last) — the last entry extends); `teacher_blend` =
	// per-episode round-robin WITHIN every round (episode % len) and overrides
	// schedule+teacher when non-empty. Selection is deterministic — it never
	// draws from the loop RNG, so enabling it cannot perturb episode ICs.
	#[pyo3(get, set)] pub teacher_schedule: Vec<u8>,
	#[pyo3(get, set)] pub teacher_blend: Vec<u8>,

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

	// State-splitting trainer (Phase 6 Rust port). Active when env WNN_STATE_SPLIT=1;
	// then split_train_loop REPLACES the per-traj BPTT step on the gated batch.
	#[pyo3(get, set)] pub split_tau: f32,
	#[pyo3(get, set)] pub split_clean_gain: f32,
	#[pyo3(get, set)] pub split_accum_corr: f32,
	#[pyo3(get, set)] pub split_max_rounds: usize,
	#[pyo3(get, set)] pub split_k_start: usize,
	#[pyo3(get, set)] pub split_coarse_target: usize,
	#[pyo3(get, set)] pub split_selective_output: bool,
	// H4 axis curriculum: active attitude axes in the episode IC (inactive =>
	// that axis' initial tilt + body rate zeroed). All-true = full 3-axis (anchor).
	#[pyo3(get, set)] pub active_roll: bool,
	#[pyo3(get, set)] pub active_pitch: bool,
	#[pyo3(get, set)] pub active_yaw: bool,

	// W2 disturbances for the in-search training rollouts + per-round eval
	// (W2.3 train-under-weather). Disabled by default = pre-W2 behavior
	// (including the SmallRng draw sequence — the per-episode disturbance
	// seed is only drawn when enabled). Fields mirror controller::Disturbance;
	// dist_seed here is UNUSED by the training loop (per-episode seeds come
	// from the loop's SmallRng so every episode gets fresh weather).
	#[pyo3(get, set)] pub dist_enabled: bool,
	#[pyo3(get, set)] pub dist_tau_bias: [f32; 3],
	#[pyo3(get, set)] pub dist_gust_sigma: f32,
	#[pyo3(get, set)] pub dist_gust_tau_c: f32,
	#[pyo3(get, set)] pub dist_motor_asym: [f32; 4],
	#[pyo3(get, set)] pub dist_gyro_sigma: f32,
	#[pyo3(get, set)] pub dist_gyro_bias_walk: f32,
	#[pyo3(get, set)] pub dist_accel_sigma: f32,
	// W2.4 D5 sensor dropout/freeze + D6 observation latency + D7 dynamics
	// randomization (torque-scale jitter). 0-defaults = exactly-off =
	// bit-identical pre-W2.4 rollouts.
	#[pyo3(get, set)] pub dist_dropout_prob: f32,
	#[pyo3(get, set)] pub dist_dropout_len_steps: u32,
	#[pyo3(get, set)] pub dist_obs_delay_steps: u32,
	#[pyo3(get, set)] pub dist_torque_scale_jitter: f32,

	// Pure behavior cloning (19/07/2026, single-layer promotion): when true the
	// TEACHER's pwm drives sim.step (labels unchanged — C1 teacher targets), so
	// the student only ever trains on expert-visited states. Combined with
	// num_rounds=1 + gate-off this is classic one-pass BC — the fastest trainer
	// and the covariate-shift baseline. false = DAGGER (student drives), the
	// bit-identical legacy path.
	#[pyo3(get, set)] pub expert_drives: bool,

	// AIRFRAME (05/08/2026). The plant used to be hardcoded HERE and in
	// AttitudeSim::new's defaults — two copies, free to drift apart, and a
	// teacher whose linear model does not match the sim it controls produces
	// plausible wrong numbers rather than a crash. Now it arrives from
	// Python's Airframe registry (wnn/control/airframe.py), which carries the
	// citation. Defaults reproduce the pre-airframe synthetic plant EXACTLY so
	// every existing caller and parity anchor stays bit-identical until it
	// opts in.
	#[pyo3(get, set)] pub af_arm_length: f32,
	#[pyo3(get, set)] pub af_k_thrust: f32,
	#[pyo3(get, set)] pub af_k_drag: f32,
	#[pyo3(get, set)] pub af_inertia: [f32; 3],
	#[pyo3(get, set)] pub af_gravity: f32,
	// FIRMWARE-SOURCED PID CASCADE (05/08/2026). Already in SI — Python's
	// `_SiGains.from_firmware` is the single place degrees and actuator counts are
	// converted, so nothing here or downstream sees a foreign unit. Layout is
	// [roll, pitch, yaw] x [kp, ki, kd, i_limit].
	//
	// `af_pid_attitude_hz == 0.0` means "no cascade gains supplied" and the PID teacher
	// falls back to the legacy hand-tuned single loop — which is what keeps the
	// synthetic-plant parity anchors bit-identical. Any airframe run supplies them.
	#[pyo3(get, set)] pub af_pid_att: [f64; 12],
	#[pyo3(get, set)] pub af_pid_rate: [f64; 12],
	#[pyo3(get, set)] pub af_pid_out_limit_n: f64,
	#[pyo3(get, set)] pub af_pid_hover_n: f64,
	#[pyo3(get, set)] pub af_pid_attitude_hz: f64,
	#[pyo3(get, set)] pub af_pid_lpf_hz: f64,

	// L4 (magnitude-priority output writes, 07/08/2026). BINARY is last-writer-
	// wins, so WHICH record writes a contested cell last decides what it stores;
	// the legacy backward walk hands that to the window's EARLIEST record —
	// arbitrary w.r.t. error magnitude. Both default OFF = bit-identical legacy.
	// Single-layer (sn=0) only: sn>0 records are order-dependent (d's commits
	// feed d-1's solve) and the trainer ignores these flags there.
	/// Arm A: commit section-(d) writes in ascending-|err| order so the
	/// HIGHEST-error record writes last and owns contested cells.
	#[pyo3(get, set)] pub write_priority_err: bool,
	/// Arm B: skip output commits for records with |attitude err| below this
	/// floor (degrees) — near-hover mass cannot overwrite corrections. 0 = off.
	#[pyo3(get, set)] pub write_err_floor_deg: f32,

	// --- SCOPE C STAGE 1 (13/08/2026): the vertical channel in TRAINING. ---
	// The scorer already flies with translation; without these the trainer would
	// roll out with z frozen and the vertical features constant zero, i.e. the
	// student would learn addresses deploy never visits (the DOB divergence).
	// translation=false ⇒ every pre-stage-1 run is bit-identical.
	#[pyo3(get, set)] pub translation: bool,
	/// Vehicle mass (kg) and its per-episode randomization fraction. A PLANT
	/// parameter, never a feature (Luiz, 12/08).
	#[pyo3(get, set)] pub af_mass: f32,
	#[pyo3(get, set)] pub mass_jitter: f32,
	/// Episode axes: initial altitude offset (m), initial vertical velocity
	/// (m/s), commanded-collective jitter (fraction of hover).
	#[pyo3(get, set)] pub alt_offset: f32,
	#[pyo3(get, set)] pub init_vz: f32,
	#[pyo3(get, set)] pub collective_jitter: f32,
	/// The altitude every episode holds (m).
	#[pyo3(get, set)] pub target_altitude: f32,
	/// Outer altitude-PD shaping: closed-loop bandwidth (rad/s), damping, and
	/// the bound on the collective correction. Gains are DERIVED from the plant
	/// inside AltitudePd — these only shape the loop.
	#[pyo3(get, set)] pub alt_pd_omega: f32,
	#[pyo3(get, set)] pub alt_pd_zeta: f32,
	#[pyo3(get, set)] pub alt_pd_max_delta: f32,
}

#[pymethods]
impl RewardGatedConfigPacked {
	#[new]
	#[pyo3(signature = (
		num_rounds = 8, episodes_per_round = 24, steps_per_episode = 2000,
		bptt_window = 32, topk_per_neuron = 4, protect_learned = false,
		gate_mode = 0, gate_use_best = false, gate_window = 0,
		gate_quantile = 0.5, gate_running = true, target_source = 0,
		teacher = 0, teacher_schedule = vec![], teacher_blend = vec![],
		keep_best_checkpoint = true, explore_eps = 0.0, explore_scale = 0.1,
		curriculum = true, easy_tilt_deg = 8.0, full_tilt_deg = 30.0,
		dt = 0.001, max_initial_yaw_rad = 0.5235987756, // ~30deg
		max_initial_body_rate = 0.5, max_initial_yaw_rate = 0.3,
		eval_episodes = 20,
		split_tau = 0.1, split_clean_gain = 0.999, split_accum_corr = 0.9,
		split_max_rounds = 5, split_k_start = 1, split_coarse_target = 32,
		split_selective_output = true,
		active_roll = true, active_pitch = true, active_yaw = true,
		dist_enabled = false, dist_tau_bias = [0.0, 0.0, 0.0],
		dist_gust_sigma = 0.0, dist_gust_tau_c = 0.1,
		dist_motor_asym = [1.0, 1.0, 1.0, 1.0],
		dist_gyro_sigma = 0.0, dist_gyro_bias_walk = 0.0, dist_accel_sigma = 0.0,
		dist_dropout_prob = 0.0, dist_dropout_len_steps = 0,
		dist_obs_delay_steps = 0, dist_torque_scale_jitter = 0.0,
		expert_drives = false,
		af_arm_length = 0.075, af_k_thrust = 2.4, af_k_drag = 0.05,
		af_inertia = [0.0023, 0.0023, 0.0046], af_gravity = 9.81,
		af_pid_att = [0.0; 12], af_pid_rate = [0.0; 12],
		af_pid_out_limit_n = 0.0, af_pid_hover_n = 0.0,
		af_pid_attitude_hz = 0.0, af_pid_lpf_hz = 0.0,
		write_priority_err = false, write_err_floor_deg = 0.0,
		translation = false, af_mass = 0.0, mass_jitter = 0.0,
		alt_offset = 0.0, init_vz = 0.0, collective_jitter = 0.0,
		target_altitude = 0.0,
		alt_pd_omega = 2.0, alt_pd_zeta = 1.0, alt_pd_max_delta = 0.25,
	))]
	#[allow(clippy::too_many_arguments)]
	pub fn new(
		num_rounds: usize, episodes_per_round: usize, steps_per_episode: usize,
		bptt_window: usize, topk_per_neuron: usize, protect_learned: bool,
		gate_mode: u8, gate_use_best: bool, gate_window: usize,
		gate_quantile: f64, gate_running: bool, target_source: u8,
		teacher: u8, teacher_schedule: Vec<u8>, teacher_blend: Vec<u8>,
		keep_best_checkpoint: bool, explore_eps: f64, explore_scale: f64,
		curriculum: bool, easy_tilt_deg: f64, full_tilt_deg: f64,
		dt: f64, max_initial_yaw_rad: f64,
		max_initial_body_rate: f64, max_initial_yaw_rate: f64,
		eval_episodes: usize,
		split_tau: f32, split_clean_gain: f32, split_accum_corr: f32,
		split_max_rounds: usize, split_k_start: usize, split_coarse_target: usize,
		split_selective_output: bool,
		active_roll: bool, active_pitch: bool, active_yaw: bool,
		dist_enabled: bool, dist_tau_bias: [f32; 3],
		dist_gust_sigma: f32, dist_gust_tau_c: f32,
		dist_motor_asym: [f32; 4],
		dist_gyro_sigma: f32, dist_gyro_bias_walk: f32, dist_accel_sigma: f32,
		dist_dropout_prob: f32, dist_dropout_len_steps: u32,
		dist_obs_delay_steps: u32, dist_torque_scale_jitter: f32,
		expert_drives: bool,
		af_arm_length: f32, af_k_thrust: f32, af_k_drag: f32,
		af_inertia: [f32; 3], af_gravity: f32,
		af_pid_att: [f64; 12], af_pid_rate: [f64; 12],
		af_pid_out_limit_n: f64, af_pid_hover_n: f64,
		af_pid_attitude_hz: f64, af_pid_lpf_hz: f64,
		write_priority_err: bool, write_err_floor_deg: f32,
		translation: bool, af_mass: f32, mass_jitter: f32,
		alt_offset: f32, init_vz: f32, collective_jitter: f32,
		target_altitude: f32,
		alt_pd_omega: f32, alt_pd_zeta: f32, alt_pd_max_delta: f32,
	) -> Self {
		Self {
			num_rounds, episodes_per_round, steps_per_episode, bptt_window,
			topk_per_neuron, protect_learned,
			gate_mode, gate_use_best, gate_window, gate_quantile, gate_running,
			target_source, teacher, teacher_schedule, teacher_blend,
			keep_best_checkpoint,
			explore_eps, explore_scale,
			curriculum, easy_tilt_deg, full_tilt_deg,
			dt, max_initial_yaw_rad, max_initial_body_rate, max_initial_yaw_rate,
			eval_episodes,
			split_tau, split_clean_gain, split_accum_corr,
			split_max_rounds, split_k_start, split_coarse_target, split_selective_output,
			active_roll, active_pitch, active_yaw,
			dist_enabled, dist_tau_bias, dist_gust_sigma, dist_gust_tau_c,
			dist_motor_asym, dist_gyro_sigma, dist_gyro_bias_walk, dist_accel_sigma,
			dist_dropout_prob, dist_dropout_len_steps,
			dist_obs_delay_steps, dist_torque_scale_jitter,
			expert_drives,
			af_arm_length, af_k_thrust, af_k_drag, af_inertia, af_gravity,
			af_pid_att, af_pid_rate, af_pid_out_limit_n, af_pid_hover_n,
			af_pid_attitude_hz, af_pid_lpf_hz,
			translation, af_mass, mass_jitter,
			alt_offset, init_vz, collective_jitter, target_altitude,
			alt_pd_omega, alt_pd_zeta, alt_pd_max_delta,
			write_priority_err, write_err_floor_deg,
		}
	}
}

impl RewardGatedConfigPacked {
	/// Hybrid-teacher selector — the roadmap's `teacher_for(round, episode)`.
	/// Precedence: blend (per-episode round-robin) > schedule (per-round, last
	/// entry extends) > scalar `teacher`. Deterministic: never touches the loop
	/// RNG, so `schedule=[X]*N` is bit-exact vs `teacher=X`.
	pub fn teacher_id_for(&self, round: usize, episode: usize) -> u8 {
		if !self.teacher_blend.is_empty() {
			return self.teacher_blend[episode % self.teacher_blend.len()];
		}
		if !self.teacher_schedule.is_empty() {
			return self.teacher_schedule[round.min(self.teacher_schedule.len() - 1)];
		}
		self.teacher
	}

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
	// L4 (magnitude-priority writes): |attitude error| (rad) at each DECISION
	// step — the sim's ground-truth error at the state the controller observed
	// (recorded BEFORE sim.step, unlike the post-step error the reward uses).
	// Empty when the rollout predates the field; consumers must length-check.
	pub att_errs: Vec<f32>,
	pub cumulative_reward: f64,
	pub mean_attitude_error_rad: f64,
	pub diverged: bool,
	pub steps: usize,
	// Yaw-anchor: this episode's true initial yaw (rad), from yaw_from_quat(init_q).
	// Seeds yaw_heading when the trajectory is later replayed for training.
	pub init_yaw: f32,
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
	// State-splitting GA-handshake pressure (Phase 6 Rust port → consumed by 5c
	// mutation). Accumulated across rounds when WNN_STATE_SPLIT=1.
	#[pyo3(get)] pub split_saturation: usize,
	#[pyo3(get)] pub split_wish_bits: Vec<usize>,
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

use crate::controller::{AttitudeSim, WnnController, compute_reward, monotonicity_violations, yaw_from_quat_rs};
use crate::optimal::Teacher;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

/// Periodic progress heartbeat for long opaque batch calls.
///
/// Why this exists: `dagger_train_batch_inplace` trains the WHOLE batch in one
/// call that can run for hours, and the Python caller only prints its
/// `[expand i/N]` / `[grid i/N]` lines AFTER it returns. Those lines look like
/// streaming progress but are retrospective, so a watcher sees nothing at all
/// during the work and can only infer health from CPU% — which distinguishes
/// "spinning" from "stopped" but NOT "making progress" from "wedged in a loop".
/// This emits positive evidence of forward progress instead.
///
/// Writes to stderr (the driver merges 2>&1 into the cell log). Interval comes
/// from WNN_PROGRESS_SECS; 0 disables. Cheap: one relaxed atomic per completed
/// genome plus one sleeping thread per batch.
struct BatchProgress {
	done: Arc<AtomicUsize>,
	stop: Arc<AtomicBool>,
	/// Number of periodic lines actually emitted. Exists so a test can assert the
	/// heartbeat FIRED rather than merely that it shut down cleanly — a test that
	/// only checks the join passes whether or not anything was ever reported.
	emits: Arc<AtomicUsize>,
	handle: Option<std::thread::JoinHandle<()>>,
}

impl BatchProgress {
	fn start(label: &str, total: usize) -> Self {
		// The env var is read HERE ONLY, then passed down as a parameter. Tests call
		// start_with_interval directly and never touch the environment: two tests that
		// each set WNN_PROGRESS_SECS to a different value used to race, because cargo
		// runs tests as parallel threads of ONE process and the env is process-global.
		// Whichever lost the race failed, ~2 of 3 runs. Threading it as a parameter
		// removes the shared mutable state instead of serialising access to it.
		let secs: u64 = std::env::var("WNN_PROGRESS_SECS")
			.ok().and_then(|s| s.parse().ok()).unwrap_or(300);
		Self::start_with_interval(label, total, secs)
	}

	/// `secs == 0` disables entirely (no thread; `finish` is a safe no-op).
	fn start_with_interval(label: &str, total: usize, secs: u64) -> Self {
		let done = Arc::new(AtomicUsize::new(0));
		let stop = Arc::new(AtomicBool::new(false));
		let emits = Arc::new(AtomicUsize::new(0));
		if secs == 0 || total == 0 {
			return Self { done, stop, emits, handle: None };
		}
		let (d, s, e, lbl) = (done.clone(), stop.clone(), emits.clone(), label.to_string());
		let handle = std::thread::spawn(move || {
			let t0 = Instant::now();
			// Poll in short slices so a finished batch joins promptly instead of
			// blocking for the remainder of a long interval.
			let mut waited = 0u64;
			loop {
				std::thread::sleep(std::time::Duration::from_millis(500));
				if s.load(Ordering::Relaxed) { return; }
				waited += 500;
				if waited < secs * 1000 { continue; }
				waited = 0;
				let k = d.load(Ordering::Relaxed);
				let el = t0.elapsed().as_secs_f64();
				let pct = 100.0 * k as f64 / total as f64;
				// ETA only once something has finished — extrapolating from zero
				// completions would print a fabricated number.
				let eta = if k > 0 {
					format!("~{:.0}s left", el / k as f64 * (total - k) as f64)
				} else {
					"eta unknown (0 done)".to_string()
				};
				eprintln!("[progress] {lbl}: {k}/{total} ({pct:.0}%) {el:.0}s elapsed, {eta}");
				e.fetch_add(1, Ordering::Relaxed);
			}
		});
		Self { done, stop, emits, handle: Some(handle) }
	}

	#[inline]
	fn tick(&self) { self.done.fetch_add(1, Ordering::Relaxed); }

	#[cfg(test)]
	fn emit_count(&self) -> usize { self.emits.load(Ordering::Relaxed) }

	fn finish(mut self, label: &str, total: usize) {
		self.stop.store(true, Ordering::Relaxed);
		if let Some(h) = self.handle.take() {
			let _ = h.join();
			eprintln!("[progress] {label}: {total}/{total} (100%) done");
		}
	}
}

// ----- Rust-internal wrappers around #[pymethods] constructors ------------
//
// AttitudeSim::new and AttitudePidRs::new live in #[pymethods] blocks (with
// PyO3 default values). Calling them from Rust requires all positional args
// — these helpers centralize the defaults so they can't drift from the
// Python side (defaults MUST match AttitudePIDConfig and AttitudeSim::new).

/// The plant, in one place. Mirrors `wnn/control/airframe.py::Airframe` field
/// for field so the Python registry (which carries the citation) is the single
/// source of truth and this is only its carrier. `DEFAULT` reproduces the
/// pre-airframe synthetic plant exactly, keeping every existing caller and
/// parity anchor bit-identical until it opts in.
#[derive(Clone, Copy)]
pub struct AirframeRs {
	pub dt: f32,
	pub arm_length: f32,
	pub k_thrust: f32,
	pub k_drag: f32,
	pub inertia: [f32; 3],
	pub gravity: f32,
	/// Firmware-sourced PID cascade gains, already SI. None = none supplied, so the PID
	/// teacher stays the legacy hand-tuned single loop (keeps the parity anchors).
	pub pid_fw: Option<crate::pid_firmware::AttitudePidFirmwareRs>,
}

impl AirframeRs {
	pub const DEFAULT: AirframeRs = AirframeRs {
		dt: 0.001, arm_length: 0.075, k_thrust: 2.4, k_drag: 0.05,
		inertia: [0.0023, 0.0023, 0.0046], gravity: 9.81,
		pid_fw: None,
	};
	fn from_cfg(cfg: &RewardGatedConfigPacked) -> Self {
		AirframeRs {
			dt: cfg.dt as f32,
			arm_length: cfg.af_arm_length,
			k_thrust: cfg.af_k_thrust,
			k_drag: cfg.af_k_drag,
			inertia: cfg.af_inertia,
			gravity: cfg.af_gravity,
			pid_fw: crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(
				cfg.af_pid_att, cfg.af_pid_rate, cfg.af_pid_out_limit_n,
				cfg.af_pid_hover_n, cfg.af_k_thrust as f64,
				(1.0 / cfg.dt.max(1e-9)).round() as u32,
				cfg.af_pid_attitude_hz, cfg.af_pid_lpf_hz,
			),
		}
	}
	pub(crate) fn sim(&self) -> AttitudeSim {
		AttitudeSim::new(self.dt, self.arm_length, self.k_thrust, self.k_drag,
			self.inertia, self.gravity)
	}
	pub(crate) fn teacher(&self, id: u8) -> Teacher {
		// PID (id 0) now derives from the airframe like every other teacher, WHEN the
		// airframe supplied cascade gains. Without them it stays the legacy hand-tuned
		// single loop — that fallback is what keeps the synthetic-plant parity anchors
		// bit-identical, and it is the only remaining path to the retired gains.
		if id == 0 {
			if let Some(p) = self.pid_fw.clone() {
				return Teacher::PidFw(p);
			}
		}
		Teacher::from_id(id, self.dt, self.arm_length, self.k_thrust,
			self.k_drag, self.inertia, self.gravity)
	}
}

fn sim_default(cfg: &RewardGatedConfigPacked) -> AttitudeSim {
	// The sim the training loop controls. Airframe comes from the config (see
	// the af_* fields) so it is defined ONCE, on the Python side, with its
	// citation attached. Clean sim; W2 disturbances are applied PER EPISODE via
	// apply_cfg_disturbance (each episode needs its own weather seed), not at
	// construction. Config defaults reproduce the old hardcoded plant exactly,
	// so untouched callers stay bit-identical.
	AirframeRs::from_cfg(cfg).sim()
}

/// W2: arm this episode's disturbance on the sim from the packed config.
/// Disabled ⇒ strict no-op — the SmallRng sequence is untouched, so
/// disturbance-off runs stay bit-identical to pre-W2 (the parity anchor).
/// Enabled ⇒ one extra u64 draw per episode = that episode's weather seed.
fn apply_cfg_disturbance(sim: &mut AttitudeSim, cfg: &RewardGatedConfigPacked, rng: &mut SmallRng) {
	if !cfg.dist_enabled {
		return;
	}
	let ep_seed: u64 = rng.gen();
	sim.set_disturbance(
		cfg.dist_tau_bias, cfg.dist_gust_sigma, cfg.dist_gust_tau_c,
		cfg.dist_motor_asym, cfg.dist_gyro_sigma, cfg.dist_gyro_bias_walk,
		cfg.dist_accel_sigma, ep_seed,
		cfg.dist_dropout_prob, cfg.dist_dropout_len_steps,
		cfg.dist_obs_delay_steps, cfg.dist_torque_scale_jitter,
	);
}

fn teacher_default(id: u8, cfg: &RewardGatedConfigPacked) -> Teacher {
	// The DAGGER teacher (0=PID, 1=LQR, 2=MPC). Sim params MUST match
	// sim_default() so the LQR/MPC linear plant model matches the sim the loop
	// controls — they now read the SAME af_* fields, so the two cannot drift
	// apart the way two hardcoded copies could. LQR/LQI/MPC/MPCOF derive their
	// gains from these numbers and adapt automatically; PID does NOT (its gains
	// are fixed in AttitudePidRs::new) and must be re-sourced per airframe —
	// see wnn/control/airframe.py PidGains.
	AirframeRs::from_cfg(cfg).teacher(id)
}

/// Lazily-built bank of the (≤3) distinct teachers a hybrid schedule can
/// reference. One instance per id, built on first use and reused after — a
/// constant schedule therefore runs the SAME single instance (reset per
/// episode) as the legacy scalar path, and MPC's QP setup cost is only paid
/// when an MPC round/episode actually occurs.
/// One slot per teacher id Teacher::from_id can build: 0=PID 1=LQR 2=MPC
/// 3=LQI 4=MPCOF.
///
/// THIS ARRAY WAS [_; 3] UNTIL 05/08/2026 AND SILENTLY CLAMPED ids > 2 TO PID.
/// LQI and MPCOF were added to Teacher::from_id but the bank was never widened,
/// so every DAGGER run asking for them trained against PID instead — with no
/// error, and with the CLASSICAL BASELINES still correct (they call
/// Teacher::from_id directly, bypassing the bank), which is what made the two
/// disagree in a way nobody could see. Caught only because an mpcof cell and an
/// lqi cell produced bit-identical output.
///
/// The length is now tied to TEACHER_IDS so widening from_id without widening
/// the bank is a compile error rather than a silent demotion to PID.
const TEACHER_IDS: usize = 5;
struct TeacherBank([Option<Teacher>; TEACHER_IDS], AirframeRs);

impl TeacherBank {
	fn new(af: AirframeRs) -> Self {
		TeacherBank([None, None, None, None, None], af)
	}
	fn get_mut(&mut self, id: u8) -> &mut Teacher {
		// Only ids Teacher::from_id cannot build collapse to PID — that is its
		// `_` arm, and the bound MUST track TEACHER_IDS. Getting this wrong does
		// not fail loudly: it trains the wrong teacher and reports success.
		let id = if (id as usize) >= TEACHER_IDS { 0 } else { id };
		let af = self.1;
		self.0[id as usize].get_or_insert_with(|| af.teacher(id))
	}
}

/// WnnController::step returns Vec<f32> of length num_motors. The dagger
/// loop uses [f32; 4] PWMs (num_motors=4 always for the quad). Safe
/// H3 inverse of WnnController::mix_controls_to_motors: 4 motor PWMs →
/// controls [T, τ_roll, τ_pitch, τ_yaw]. Exact inverse (mix signs:
/// p0=T-τp+τy, p1=T-τr-τy, p2=T+τp+τy, p3=T+τr-τy).
#[inline]
fn unmix_motors_to_controls(p: [f32; 4]) -> [f32; 4] {
	[
		(p[0] + p[1] + p[2] + p[3]) * 0.25,  // T  = mean
		(p[3] - p[1]) * 0.5,                 // τ_roll  = (right − left)/2
		(p[2] - p[0]) * 0.5,                 // τ_pitch = (back − front)/2
		(p[0] - p[1] + p[2] - p[3]) * 0.25,  // τ_yaw
	]
}

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
	active_axes: [bool; 3],   // H4 [roll,pitch,yaw]: zero the inactive axes (tilt + matching rate)
) -> ([f32; 4], [f32; 3]) {
	// Draw ALWAYS (then zero if inactive) so all-axes-active is RNG-identical to
	// the pre-H4 sequence (the curriculum parity anchor).
	let r = rng.gen_range(-max_tilt..max_tilt);
	let p = rng.gen_range(-max_tilt..max_tilt);
	let y = rng.gen_range(-max_yaw..max_yaw);
	let roll  = if active_axes[0] { r } else { 0.0 };
	let pitch = if active_axes[1] { p } else { 0.0 };
	let yaw   = if active_axes[2] { y } else { 0.0 };
	let q = euler_to_quat_xyz(roll, pitch, yaw);
	let ox = rng.gen_range(-max_body_rate..max_body_rate) as f32;
	let oy = rng.gen_range(-max_body_rate..max_body_rate) as f32;
	let oz = rng.gen_range(-max_yaw_rate..max_yaw_rate)   as f32;
	let omega = [
		if active_axes[0] { ox } else { 0.0 },
		if active_axes[1] { oy } else { 0.0 },
		if active_axes[2] { oz } else { 0.0 },
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

/// SCOPE C STAGE 1: a symmetric jitter draw, U(-mag, +mag). mag == 0 draws
/// NOTHING from the rng — that is what keeps a translation-off run's random
/// sequence bit-identical to every pre-stage-1 result.
#[inline]
fn jitter_sym(rng: &mut SmallRng, mag: f32) -> f32 {
	if mag == 0.0 { return 0.0; }
	rng.gen_range(-mag..mag)
}

/// SCOPE C STAGE 1: the outer altitude PD the teacher rides on, derived from the
/// config's own plant (never guessed). None when translation is off.
fn alt_pd_for(cfg: &RewardGatedConfigPacked) -> Option<crate::altitude_pd::AltitudePd> {
	if !cfg.translation { return None; }
	crate::altitude_pd::AltitudePd::from_plant(
		cfg.af_mass as f64, cfg.af_gravity as f64, cfg.af_k_thrust as f64,
		cfg.alt_pd_omega as f64, cfg.alt_pd_zeta as f64, cfg.alt_pd_max_delta as f64,
	).ok()
}

// ----- Rollout + label -----------------------------------------------------

/// Roll the STUDENT closed-loop, recording trajectory + cumulative reward.
/// Mirrors `_rollout_and_label` in reward_gated.py.
#[allow(clippy::too_many_arguments)]
pub fn rollout_and_label_rs(
	controller: &mut WnnController,
	teacher: &mut Teacher,
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
		[cfg.active_roll, cfg.active_pitch, cfg.active_yaw],
	);
	sim.reset(Some(init_q), Some(init_omega));
	// SCOPE C STAGE 1: per-episode PLANT draw + vertical ICs, drawn from the SAME
	// loop rng as the attitude IC so a stage-1 run is reproducible from its seed.
	// Gated: translation=false draws NOTHING, keeping every pre-stage-1 run
	// bit-identical to the banked sequence (the disturbance-off parity anchor).
	// STAGE 1: this episode's plant draw, vertical ICs, and the commanded
	// collective in ABSOLUTE pwm — the operating point the student rides on.
	let ep_collective_pwm: f32 = if cfg.translation {
		let m = cfg.af_mass * (1.0 + jitter_sym(rng, cfg.mass_jitter));
		sim.set_translation_core(m).expect("stage-1 mass must be positive");
		sim.set_vertical_state(jitter_sym(rng, cfg.alt_offset),
		                       jitter_sym(rng, cfg.init_vz));
		let hover = (m * cfg.af_gravity / (4.0 * cfg.af_k_thrust)).sqrt();
		(hover * (1.0 + jitter_sym(rng, cfg.collective_jitter))).clamp(0.0, 1.0)
	} else {
		0.0
	};
	// W2: per-episode weather (no-op when cfg.dist_enabled is false).
	apply_cfg_disturbance(sim, cfg, rng);
	teacher.reset();
	// Yaw-anchor: seed the controller's heading to this episode's true initial yaw so
	// the GENERATING rollout's obs_yaw_err matches what training/scoring will see.
	let init_yaw = yaw_from_quat_rs(init_q);
	controller.reset(init_yaw);
	// STAGE 1: anchor the delta accumulator at the commanded collective AFTER
	// reset (reset seeds from whatever anchor is current), so the episode opens
	// at hover rather than free-falling from the legacy 0.5 neutral.
	if cfg.translation {
		controller.set_collective_anchor(ep_collective_pwm);
	}
	// QSR/PLN decode coin: a fresh per-episode seed from the SAME RNG stream (this
	// is a CPU-only collection rollout — no GPU twin to match; it just needs
	// reproducible per-episode stochasticity). GATED on is_stochastic so
	// deterministic modes draw NOTHING from `rng` and stay bit-identical to the
	// pre-Part-5 sequence (the disturbance-off parity anchor, dagger:405).
	if crate::cell_mode::is_stochastic(controller.memory_mode_u8()) {
		controller.set_decode_seed(rng.gen());
	}

	// STAGE 1: the teacher's outer altitude loop (None ⇒ attitude-only teacher,
	// the bit-identical legacy path).
	let alt_pd = alt_pd_for(cfg);
	let target_64 = target;     // already f32; PID/controller take f32 too

	let mut traj = TrajectoryRs::default();
	traj.init_yaw = init_yaw;   // yaw-anchor: remember for the training replay
	traj.gyros = Vec::with_capacity(cfg.steps_per_episode);
	traj.accels = Vec::with_capacity(cfg.steps_per_episode);
	traj.targets = Vec::with_capacity(cfg.steps_per_episode);
	traj.pid_pwms = Vec::with_capacity(cfg.steps_per_episode);
	traj.student_pwms = Vec::with_capacity(cfg.steps_per_episode);
	traj.att_errs = Vec::with_capacity(cfg.steps_per_episode);

	// Offset-free MPC observer state: the action ACTUALLY applied to the sim last
	// step (the student's, under DAGGER). Hover default so step-0's observe() —
	// which the teacher ignores anyway (no prior gyro) — is harmless. No-op for
	// every teacher except MpcOf.
	let mut last_applied = [0.5f32; 4];
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
		// L4: ground-truth |attitude err| at the state the controller is about to
		// act on. A SEPARATE read from the post-step attitude_err below (which
		// feeds the reward and stays untouched for parity).
		let att_err_now = sim.attitude_error(None);

		// Offset-free MPC observer: feed it (current gyro, action applied last
		// step) so it estimates the input disturbance from the model residual
		// BEFORE it plans this step. No-op for pid/lqr/mpc/lqi.
		teacher.observe(gyro, [
			last_applied[0] as f64, last_applied[1] as f64,
			last_applied[2] as f64, last_applied[3] as f64,
		]);

		// SCOPE C STAGE 1: the vertical observation for THIS step, read at the
		// same start-of-step snapshot the IMU is — the exact twin of what the
		// scorer's rollout does. Without this the features would be constant
		// zeros here and real at scoring (the DOB train/deploy divergence).
		if cfg.translation {
			controller.set_vertical_obs(ep_collective_pwm,
			                            cfg.target_altitude - sim.altitude_rs(),
			                            sim.vertical_velocity_rs());
		}
		// Student forward + teacher label at student-visited state.
		let student_pwm = controller_step_4(controller, gyro, accel, target_64);
		// STAGE 1: the teacher gains a collective channel — an outer altitude PD
		// on top of the attitude law, the DISCLOSED CASCADE the classical rivals
		// already are. Without it the student is asked to learn altitude from a
		// teacher that never commands it.
		let expert_pwm = match alt_pd.as_ref() {
			Some(pd) => teacher.step_with_collective(
				q, gyro, target_64, pd,
				(cfg.target_altitude - sim.altitude_rs()) as f64,
				sim.vertical_velocity_rs() as f64),
			None => teacher.step_rs(q, gyro, target_64),
		};
		let expert_pwm_f32 = [
			expert_pwm[0] as f32, expert_pwm[1] as f32,
			expert_pwm[2] as f32, expert_pwm[3] as f32,
		];
		// Option A: capture the teacher's integral, normalized to [-1,1] by its
		// clamp, AFTER step_rs updated it (this is the desired recurrent-state value).
		// LQR/MPC are memoryless → integrals()=0 (Option-A only bites for PID).
		let integ = teacher.integrals();
		let clamp = teacher.i_clamps();
		let integ_norm = [
			(integ[0] / clamp[0]).clamp(-1.0, 1.0),
			(integ[1] / clamp[1]).clamp(-1.0, 1.0),
			(integ[2] / clamp[2]).clamp(-1.0, 1.0),
		];

		// Pure BC (expert_drives): the TEACHER's action is what the sim executes,
		// so trajectories follow the expert's state distribution exactly. The
		// student forward still runs above (its pwm is recorded for C2/metrics).
		// Exploration: perturb the applied PWM (C2 only).
		let mut applied = if cfg.expert_drives { expert_pwm_f32 } else { student_pwm };
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
		// H3: when outputs are decoupled, the output banks are CONTROLS, so the
		// training TARGETS (teacher + student MOTOR pwms) must be un-mixed into
		// control space [T,τr,τp,τy]. Single point ⇒ all downstream paths (split /
		// non-split, C1 teacher / C2 student) train toward the right controls.
		if controller.decouple_outputs_flag() {
			traj.pid_pwms.push(unmix_motors_to_controls(expert_pwm_f32));
			traj.student_pwms.push(unmix_motors_to_controls(applied));
		} else {
			traj.pid_pwms.push(expert_pwm_f32);
			traj.student_pwms.push(applied);
		}
		traj.pid_integrals.push(integ_norm);
		traj.att_errs.push(att_err_now);

		// DOB Fix A: tell the STUDENT's observer what actually flew, exactly as
		// teacher.observe() is told above. step() stored the student's own
		// return as a default, but expert_drives / explore_eps may have replaced
		// or perturbed it — the observer must see the plant's real input.
		controller.observe_applied(applied);
		sim.step(applied);
		last_applied = applied;   // offset-free MPC observer: what the sim saw
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
		// L4: per-record |attitude err| for the magnitude-priority commit. Only
		// sliced when the trajectory recorded it (length-aligned with gyros).
		let ae = if traj.att_errs.len() >= end {
			Some(traj.att_errs[start..end].to_vec())
		} else {
			None
		};
		// DOB Fix A: the applied-pwm stream for the replay's observer, sliced
		// exactly like gyros. student_pwms records what the sim received
		// (expert_drives / exploration included), so replay == deploy.
		let sp = if traj.student_pwms.len() >= end {
			Some(traj.student_pwms[start..end].to_vec())
		} else {
			None
		};
		let (sw, ow) = controller.bptt_train_window(
			g, a, tg, pp,
			cfg.topk_per_neuron, first, cfg.protect_learned, ig,
			traj.init_yaw,   // yaw-anchor: re-seed heading on the reset window
			ae, cfg.write_priority_err, cfg.write_err_floor_deg,
			sp,
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
	// DOB Fix A discriminator: WNN_DHAT_TRACE=<path> appends one CSV row per
	// eval step (episode,t,d̂_roll,d̂_pitch,d̂_yaw). A policy-correlated
	// oscillation in the trace = model-error re-injection (mechanism 2); a
	// clean, converged d̂ that still hurts = the hidden-state cost (mechanism 3).
	// Env read once per call; no-op unless the observer is on.
	let mut dhat_trace = std::env::var("WNN_DHAT_TRACE").ok()
		.filter(|_| controller.dhat_params().is_some())
		.and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok())
		.map(std::io::BufWriter::new);
	let stable_thresh_rad = 5.0_f64.to_radians();

	for ep_idx in 0..cfg.eval_episodes {
		let _ = ep_idx; // read by the WNN_DHAT_TRACE writer below
		// Cooperative SIGTERM cancel on the closed-loop eval path (held-out / dagger),
		// same rationale as cpu_score::rollout_one: bail fast so a paused run dumps at
		// the next boundary instead of finishing all eval_episodes. Partial aggregate
		// is discarded on the unwinding resume.
		if ram_core::cancel::check_cancel() {
			break;
		}
		let (init_q, init_omega) = sample_initial_state(
			rng, tilt_rad,
			cfg.max_initial_yaw_rad,
			cfg.max_initial_body_rate,
			cfg.max_initial_yaw_rate,
			[cfg.active_roll, cfg.active_pitch, cfg.active_yaw],
		);
		// Yaw-anchor: seed the eval rollout's heading from this episode's true initial yaw.
		controller.reset(yaw_from_quat_rs(init_q));
		sim.reset(Some(init_q), Some(init_omega));
		// W2: per-episode weather in the per-round eval too (train-under-weather
		// must be SCORED under weather or the gate/checkpoint ranks on the wrong
		// regime). No-op when disabled.
		apply_cfg_disturbance(sim, cfg, rng);

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
			if let Ok(v) = monotonicity_violations(out_cells, levels_per_motor, num_motors,
			                                        controller.memory_mode_u8(), Some(controller.output_decode_u8())) {
				sum_mono += v as f64;
			}

			sim.step(pwm);
			// DOB Fix A discriminator: one CSV row per step when armed.
			if let Some(w) = dhat_trace.as_mut() {
				use std::io::Write;
				let d = controller.dhat_estimate();
				let _ = writeln!(w, "{ep_idx},{_t},{},{},{}", d[0], d[1], d[2]);
			}
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

// ----- GPU state-split offload (Task 3) ------------------------------------

/// Flat, GPU-ready form of a gated trajectory batch (single genome). Mirrors the
/// `TrainBatch` layout: ep_base/ep_count group episodes per-genome; step_base/
/// step_count index the flat sensor arrays (gyros/accels/targets are *3 per step,
/// pid_pwms *4); init_q is the per-episode yaw-only quaternion (w,x,y,z).
struct GatedFlat {
	ep_base: Vec<u32>,
	ep_count: Vec<u32>,
	step_base: Vec<u32>,
	step_count: Vec<u32>,
	gyros: Vec<f32>,
	accels: Vec<f32>,
	targets: Vec<f32>,
	pids: Vec<f32>,
	init_q: Vec<f32>,
}

/// Flatten the gated (episode-major) trajectories into `GatedFlat`. All episodes
/// belong to the ONE genome being trained ⇒ ep_base=[0], ep_count=[N]. The CPU
/// split path re-seeds only yaw, so init_q is the yaw-only quaternion
/// (cos θ/2, 0, 0, sin θ/2) that `yaw_from_quat` inverts back to `init_yaw`.
fn flatten_gated(gated: &[&TrajectoryRs]) -> GatedFlat {
	let ne = gated.len();
	let total_steps: usize = gated.iter().map(|t| t.gyros.len()).sum();
	let mut f = GatedFlat {
		ep_base: vec![0u32],
		ep_count: vec![ne as u32],
		step_base: Vec::with_capacity(ne),
		step_count: Vec::with_capacity(ne),
		gyros: Vec::with_capacity(total_steps * 3),
		accels: Vec::with_capacity(total_steps * 3),
		targets: Vec::with_capacity(total_steps * 3),
		pids: Vec::with_capacity(total_steps * 4),
		init_q: Vec::with_capacity(ne * 4),
	};
	let mut sbase = 0u32;
	for t in gated {
		let n = t.gyros.len();
		f.step_base.push(sbase);
		f.step_count.push(n as u32);
		sbase += n as u32;
		for s in 0..n {
			f.gyros.extend_from_slice(&t.gyros[s]);
			f.accels.extend_from_slice(&t.accels[s]);
			f.targets.extend_from_slice(&t.targets[s]);
			f.pids.extend_from_slice(&t.pid_pwms[s]);
		}
		let (sn, cs) = (0.5 * t.init_yaw).sin_cos();
		f.init_q.extend_from_slice(&[cs, 0.0, 0.0, sn]);
	}
	f
}

thread_local! {
	// One Metal trainer per worker thread, built lazily on first GPU-split use and
	// reused across genomes on that thread (shader compile is expensive; per-genome
	// rebuild would swamp the offload). None ⇒ no Metal device / compile failed on
	// this thread → the caller falls back to the CPU split. Not a mutable global:
	// per-thread, and ControllerTrainer methods take &self.
	static GPU_SPLIT_TRAINER: std::cell::OnceCell<Option<crate::metal_controller::ControllerTrainer>>
		= std::cell::OnceCell::new();
}

/// Warn once (per process) that the GPU split path skips the CPU wish-analysis,
/// so GA saturation-grow gets no `saturation`/`wish_bits` signal from GPU-trained
/// genomes. Results are still cell-correct (parity-proven); only the grow hint is absent.
fn warn_gpu_split_no_pressure() {
	static ONCE: std::sync::Once = std::sync::Once::new();
	ONCE.call_once(|| {
		eprintln!("[GPU-TRAIN] WNN_CONTROLLER_GPU_TRAIN=1: split trained on GPU — the \
		           CPU wish-analysis (saturation / wish_bits for GA saturation-grow) is \
		           NOT computed on this path (cells are parity-identical to CPU).");
	});
}

/// Try to run the whole state-split loop on the GPU for `gated`. Returns
/// `Some(planted)` on success (controller mutated in place via interior
/// mutability), or `None` if there is no usable Metal trainer or the dispatch
/// errored — the caller then runs the CPU split so training never silently no-ops.
fn try_gpu_split(
	controller: &WnnController,
	gated: &[&TrajectoryRs],
	cfg: &RewardGatedConfigPacked,
	target: [f32; 3],
) -> Option<usize> {
	let fb = flatten_gated(gated);
	let batch = crate::metal_controller::TrainBatch {
		ep_base: &fb.ep_base,
		ep_count: &fb.ep_count,
		step_base: &fb.step_base,
		step_count: &fb.step_count,
		gyros: &fb.gyros,
		accels: &fb.accels,
		targets: &fb.targets,
		pid_pwms: &fb.pids,
		init_q: &fb.init_q,
		selective: cfg.split_selective_output,
		target_rpy: target,
	};
	GPU_SPLIT_TRAINER.with(|cell| {
		let trainer = cell.get_or_init(|| match crate::metal_controller::ControllerTrainer::new() {
			Ok(t) => Some(t),
			Err(e) => {
				eprintln!("[GPU-TRAIN] ControllerTrainer::new failed on this thread ({e}); CPU-split fallback");
				None
			}
		});
		let trainer = trainer.as_ref()?;
		match trainer.split_train_loop_gpu(
			controller, &batch, cfg.split_tau, cfg.split_clean_gain, cfg.split_accum_corr,
			cfg.split_max_rounds, cfg.split_k_start, cfg.split_coarse_target,
		) {
			Ok((_rounds, _conflicts, planted, _per_round)) => {
				warn_gpu_split_no_pressure();
				Some(planted)
			}
			Err(e) => {
				eprintln!("[GPU-TRAIN] split_train_loop_gpu error ({e}); CPU-split fallback");
				None
			}
		}
	})
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
	// SCOPE C STAGE 1 (13/08/2026) — the trainer now DOES roll out with vertical
	// dynamics and a collective teacher, so the features are live here. What must
	// still be refused is the MISMATCH: vertical features with cfg.translation
	// off would train on constant zeros while the scorer feeds real values (the
	// DOB train/deploy divergence). Keep it an assert, not a silent skip.
	{
		let (cc, ae, vz) = controller.vert_params();
		assert!(!((cc || ae || vz) && !cfg.translation),
			"dagger_train: the controller has stage-1 vertical features enabled \
			 (collective_cmd={cc}, alt_err={ae}, vz={vz}) but cfg.translation is OFF — \
			 they would be constant zeros in training and real at scoring. Set \
			 translation (and af_mass) on the training config.");
		assert!(!(cfg.translation && !(cfg.af_mass > 0.0)),
			"dagger_train: cfg.translation is ON but af_mass = {} — mass is a PLANT \
			 parameter and the vertical dynamics divide by it.", cfg.af_mass);
	}
	let mut rng = SmallRng::seed_from_u64(seed);
	let af = AirframeRs::from_cfg(cfg);
	let mut teachers = TeacherBank::new(af);
	let mut sim = sim_default(cfg);

	let mut stats = TrainStats::default();
	let mut history_scores: Vec<f64> = Vec::new();
	let mut best_fit = f64::NEG_INFINITY;
	let mut best_snapshot: Option<(Vec<(usize, u64, u8)>, Vec<(usize, u64, u8)>)> = None;
	// State-splitting trainer (Phase 6 Rust port). When ON, the per-traj BPTT step
	// is replaced by split_train_loop on the gated batch; matches reward_gated.py.
	// Single-layer fast path (sn=0, 19/07/2026): nothing to split into AND
	// split_train_loop owns the output retrain — forcing the non-split path here
	// keeps gated episodes trained (direct output writes) instead of no-oping.
	let use_split = std::env::var("WNN_STATE_SPLIT").map(|s| s == "1").unwrap_or(false)
		&& controller.gpu_dims().2 > 0;
	// Task 3: offload the split loop to Metal (only meaningful WITH state-split).
	// Contention-negative while the IDS worker owns the GPU — opt-in, run when free.
	// DOB Fix A: never with the observer on — the GPU twin's replay still feeds
	// d̂ the frozen accumulator (see split_train_loop's assert; the CPU split
	// refuses too, but try_gpu_split would run FIRST and silently diverge).
	// SCOPE C STAGE 1 (13/08/2026): same refusal for the vertical channel, same
	// reason. The train/record kernels REPLAY recorded trajectories and carry no
	// z/vz state, so they would append the vertical features as ZEROS while the
	// CPU/score path appends the real values — a train/deploy address divergence,
	// which is precisely the DOB frozen-accumulator bug in a new costume. Until
	// the recorder carries the vertical observation, stage-1 trains on the CPU.
	let vert_on = {
		let (cc, ae, vz) = controller.vert_params();
		cc || ae || vz
	};
	let use_gpu_split = use_split
		&& std::env::var("WNN_CONTROLLER_GPU_TRAIN").map(|s| s == "1").unwrap_or(false)
		&& controller.dhat_params().is_none()
		&& !vert_on;

	for it in 0..cfg.num_rounds {
		let tilt_rad = cfg.round_tilt_rad(it);

		// 1. Roll out N episodes, record trajectories.
		let mut trajs: Vec<TrajectoryRs> = Vec::with_capacity(cfg.episodes_per_round);
		for ep in 0..cfg.episodes_per_round {
			let teacher = teachers.get_mut(cfg.teacher_id_for(it, ep));
			let t = rollout_and_label_rs(controller, teacher, &mut sim, cfg, tilt_rad, &mut rng, target);
			trajs.push(t);
		}
		let round_scores: Vec<f64> = trajs.iter().map(|t| t.cumulative_reward).collect();
		let mean_ep_reward = round_scores.iter().sum::<f64>() / round_scores.len().max(1) as f64;

		// 2. Gate + 3. train on survivors.
		let mut n_trained = 0_usize;
		let mut cells_written = 0_usize;
		if use_split {
			// State-splitting trainer: hand the WHOLE gated batch to split_train_loop
			// (conflicts must be found ACROSS episodes), which builds state +
			// retrains output in place and reports GA-handshake pressure.
			let gated: Vec<&TrajectoryRs> = trajs.iter()
				.filter(|t| episode_passes_gate_rs(t.cumulative_reward, &round_scores, &history_scores, cfg))
				.collect();
			if !gated.is_empty() {
				// GPU offload (Task 3): run the whole split loop on Metal when
				// WNN_CONTROLLER_GPU_TRAIN=1. On any Metal error try_gpu_split returns
				// None and we fall through to the CPU split — training never silently
				// no-ops. The GPU path skips the CPU wish-analysis, so saturation /
				// wish_bits stay 0 / empty (cells are parity-identical; warned once).
				let mut trained_on_gpu = false;
				if use_gpu_split {
					if let Some(planted) = try_gpu_split(controller, &gated, cfg, target) {
						cells_written = planted;
						n_trained = gated.len();
						stats.train_steps += gated.iter().map(|t| t.steps).sum::<usize>();
						trained_on_gpu = true;
					}
				}
				if !trained_on_gpu {
					// NOTE: these four clones (~2.5 MB total) stay. split_train_loop is a
					// #[pyo3] method taking owned Vecs, so borrowing here would need either
					// a generic over AsRef or a parallel _rs entry point — new surface for
					// ~25 MB across the fan-out, against the ~3.3 GB the fold-chain and
					// batch-size fixes already removed. Revisit if the pymethod ever loses
					// its Python callers.
					let g: Vec<Vec<[f32; 3]>> = gated.iter().map(|t| t.gyros.clone()).collect();
					let a: Vec<Vec<[f32; 3]>> = gated.iter().map(|t| t.accels.clone()).collect();
					let tg: Vec<Vec<[f32; 3]>> = gated.iter().map(|t| t.targets.clone()).collect();
					let pp: Vec<Vec<[f32; 4]>> = gated.iter().map(|t| t.pid_pwms.clone()).collect();
					// Yaw-anchor: per-episode initial yaw parallel to the gated batch, so
					// split_record/split_retrain_output re-seed yaw to match score-time.
					let iy: Vec<f32> = gated.iter().map(|t| t.init_yaw).collect();
					let (_r, _cf, planted, _pr, saturation, wishes) = controller.split_train_loop(
						g, a, tg, pp, cfg.split_tau, cfg.split_clean_gain, cfg.split_accum_corr,
						cfg.split_max_rounds, cfg.split_k_start, cfg.split_coarse_target,
						cfg.split_selective_output, iy,
					);
					cells_written = planted;
					n_trained = gated.len();
					stats.train_steps += gated.iter().map(|t| t.steps).sum::<usize>();
					stats.split_saturation += saturation;
					for w in wishes {
						if !stats.split_wish_bits.contains(&w) {
							stats.split_wish_bits.push(w);
						}
					}
				}
			}
		} else {
			for traj in &trajs {
				if episode_passes_gate_rs(traj.cumulative_reward, &round_scores, &history_scores, cfg) {
					let (sw, ow) = train_on_trajectory_rs(controller, traj, cfg);
					cells_written += sw + ow;
					n_trained += 1;
					stats.train_steps += traj.steps;
				}
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
	thresholds, delta_control, delta_max, delta_leak, delta_gamma,
	obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i, obs_peraxis_yaw, obs_pwm,
	obs_yaw_err, obs_yaw_err_i, integral_leak, integral_scale, dt,
	decouple_outputs,
	state_connections_per_genome, output_connections_per_genome,
	init_cells_per_genome,
	cfg, target_rpy, fold_seeds,
	action_repeat = 1,
	memory_mode = 2,
	output_decode = None,
	dhat_b = None,
	dhat_l_gain = 0.05,
	dhat_ff = false,
	dhat_ff_clamp = 0.30,
	// SCOPE C STAGE 1 (13/08/2026): the vertical FEATURE toggles. They must
	// reach the trainer's own WnnController construction or it builds a
	// 15-feature controller for an 18-feature threshold ladder and the batched
	// path silently falls back to Python — which the 13/08 boundary smoke caught.
	obs_collective_cmd = false,
	obs_alt_err = false,
	obs_vz = false,
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
	delta_gamma: f32,
	// H2 observation-feature config (run-level scalar — fixed for the GA).
	obs_tilt_p: bool,
	obs_tilt_i: bool,
	obs_peraxis_p: bool,
	obs_peraxis_i: bool,
	obs_peraxis_yaw: bool,
	obs_pwm: bool,
	obs_yaw_err: bool,
	obs_yaw_err_i: bool,
	integral_leak: f32,
	integral_scale: f32,
	dt: f32,
	decouple_outputs: bool,
	// Per-genome (variable).
	state_connections_per_genome: Vec<Vec<i64>>,
	output_connections_per_genome: Vec<Vec<i64>>,
	// Stage B (ABI 20): warm-start cells arrive as GenomeCells HANDLES, not
	// Vec<Vec<triple>>. The old form made Python build one 3-int tuple per cell
	// per genome per generation (5-13.6M cells/genome measured 21/07/2026);
	// a handle is borrowed and its columns memcpy'd while the GIL is held.
	// A genome without cells passes an EMPTY GenomeCells (== no init writes).
	init_cells_per_genome: Vec<Py<crate::genome_cells::GenomeCells>>,
	cfg: RewardGatedConfigPacked,
	target_rpy: [f32; 3],
	// K-fold seeds PER GENOME, in fold order. The controller-side folds are random
	// episode-pool seeds over an effectively infinite IID stream, so they ACCUMULATE:
	// fold k+1 continues the SAME memory fold k left behind (CLAUDE.md "K-fold: always
	// 5, accumulate for controllers"). Keeping the whole chain here means the cells
	// never leave Rust — the old caller exported them to Python triples between folds
	// (~95 B/cell × N genomes ≈ 2.4 GB at pop=50), rebuilt a controller, and re-wrote
	// them. `reset()` clears every runtime field per episode, so continuing in place is
	// bit-identical to that export→rebuild round-trip. A single-element inner vec
	// reproduces the pre-ABI-15 one-fold-per-call behaviour exactly.
	fold_seeds: Vec<Vec<u64>>,
	// Action-repeat N (arm R): decide every Nth physical step, hold in between.
	// Run-level scalar like the obs config; 1 = today's behavior.
	action_repeat: usize,
	// Memory mode (ABI 12; run-level scalar): TERNARY=0 / QUAD=1,2 / BINARY=3.
	memory_mode: u8,
	output_decode: Option<u8>,
	// L1: d̂ observer plant constant + gain, forwarded to every trained controller.
	dhat_b: Option<[f64; 3]>,
	dhat_l_gain: f32,
	dhat_ff: bool,
	dhat_ff_clamp: f32,
	obs_collective_cmd: bool,
	obs_alt_err: bool,
	obs_vz: bool,
) -> PyResult<Vec<(Py<WnnController>, TrainStats)>> {
	use rayon::prelude::*;
	let n = state_connections_per_genome.len();
	for (name, len) in [
		("output_connections_per_genome",     output_connections_per_genome.len()),
		("init_cells_per_genome",             init_cells_per_genome.len()),
		("fold_seeds",                         fold_seeds.len()),
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
	if let Some(i) = fold_seeds.iter().position(|f| f.is_empty()) {
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"fold_seeds[{i}] is empty — every genome needs at least one fold seed"
		)));
	}

	// Snapshot the handle columns while the GIL is held (pure memcpy); the
	// rayon loop below reads them GIL-free.
	let inits: Vec<crate::genome_cells::GenomeCells> =
		init_cells_per_genome.iter().map(|h| h.borrow(py).clone()).collect();

	// Drop the GIL during the heavy Rust work; Rayon does the real parallelism.
	// Each task returns Result<(controller, stats)>; we propagate the first
	// error after collecting. Construction failures (bad connection lengths)
	// would have been caught upstream — they're rare in the batch path.
	let progress = BatchProgress::start("dagger-batch", n);
	let results: Result<Vec<(WnnController, TrainStats)>, pyo3::PyErr> = py.allow_threads(|| {
		(0..n).into_par_iter().map(|i| {
			let sc = state_connections_per_genome[i].clone();
			let oc = output_connections_per_genome[i].clone();
			let mut controller = WnnController::new(
				num_motors,
				levels_per_motor_per_genome[i],
				bits_per_feature, input_window_k,
				state_neurons_per_genome[i],
				state_bits_per_neuron_per_genome[i],
				output_bits_per_neuron_per_genome[i],
				thresholds.clone(),
				sc, oc,
				delta_control, delta_max, delta_leak, delta_gamma,
				obs_tilt_p, obs_tilt_i, obs_peraxis_p, obs_peraxis_i, obs_peraxis_yaw, obs_pwm,
				obs_yaw_err, obs_yaw_err_i,
				integral_leak, integral_scale, dt, decouple_outputs,
				action_repeat,
				memory_mode, output_decode, dhat_b, dhat_l_gain, dhat_ff, dhat_ff_clamp,
				obs_collective_cmd, obs_alt_err, obs_vz,
			)?;
			let ic = &inits[i];
			for j in 0..ic.sn.len() {
				let _ = controller.write_state_cell_internal(ic.sn.get(j) as usize, ic.sa.get(j), ic.sv[j]);
			}
			for j in 0..ic.on_.len() {
				let _ = controller.write_output_cell_internal(ic.on_.get(j) as usize, ic.oa.get(j), ic.ov[j]);
			}
			// ACCUMULATE across folds into ONE controller: each fold trains the same
			// memory further, so writes compound (QUAD nudging settles same-address
			// disagreement by vote tally). The caller keeps the LAST fold's stats,
			// which is what the pre-ABI-15 Python fold loop reported.
			let mut stats = TrainStats::default();
			for &seed_k in &fold_seeds[i] {
				stats = dagger_train_inplace_rs(&mut controller, &cfg, target_rpy, seed_k);
			}
			progress.tick();
			Ok((controller, stats))
		}).collect()
	});
	progress.finish("dagger-batch", n);

	let results = results?;
	// Wrap each WnnController in a Py-owned handle (re-acquires GIL).
	results.into_iter()
		.map(|(c, s)| Ok((Py::new(py, c)?, s)))
		.collect()
}

// ============================================================================
// E4 committee scoring (02/07/2026) — K controllers vote (mean or median PWM
// per motor) in a closed loop. The per-step hot path lives HERE per the
// rust-first rule (the Python harness loop in scripts/e4_best_of_k.py was
// ~24M interpreter-dispatched steps at 10k-step probes). ICs are pre-drawn in
// PYTHON with the numpy rng chain of dagger.eval_closed_loop_reset, so the
// fresh-seed protocol numbers reproduce exactly (numpy PCG64 is not
// reimplemented in Rust — parity by injection, the dagger_train convention).
// Loop shape mirrors eval_closed_loop_rs above (the certified single-
// controller match of run_episode): is_unstable at top, read_imu, step,
// sim.step, post-step attitude_error; stable = !diverged && mean_err <= 5 deg.
// reset(init_yaw) is passed the episode's true yaw — unanchored members
// ignore it (legacy 0.0 seed), anchored members get deploy semantics.
// ============================================================================

/// Returns (stable_rate, mean_err_deg, steady_err_deg). steady = mean error
/// over the last 20% of PLANNED steps (training.py steady_window_frac),
/// averaged over episodes that reached the tail window.
#[pyfunction]
#[pyo3(signature = (controllers, init_qs, init_omegas, steps, median = false, stable_deg = 5.0,
	dist_enabled = false, dist_tau_bias = [0.0, 0.0, 0.0],
	dist_gust_sigma = 0.0, dist_gust_tau_c = 0.1,
	dist_motor_asym = [1.0, 1.0, 1.0, 1.0],
	dist_gyro_sigma = 0.0, dist_gyro_bias_walk = 0.0, dist_accel_sigma = 0.0,
	dist_seed = 0,
	dist_dropout_prob = 0.0, dist_dropout_len_steps = 0,
	dist_obs_delay_steps = 0, dist_torque_scale_jitter = 0.0,
	af_arm_length = 0.075, af_k_thrust = 2.4, af_k_drag = 0.05,
	af_inertia = [0.0023, 0.0023, 0.0046], af_gravity = 9.81, af_dt = 0.001))]
#[allow(clippy::too_many_arguments)]
pub fn eval_ensemble_closed_loop(
	py: Python<'_>,
	controllers: Vec<Py<WnnController>>,
	init_qs: Vec<f32>,      // 4 floats per episode (w, x, y, z)
	init_omegas: Vec<f32>,  // 3 floats per episode
	steps: usize,
	median: bool,
	stable_deg: f64,
	// W2 disturbances — defaults = disabled = pre-W2 behavior. Per-episode
	// seeds derive from dist_seed via disturbance_episode_seed (the SAME
	// channel-15 hash the Metal rollout kernel uses), keyed on episode index.
	dist_enabled: bool,
	dist_tau_bias: [f32; 3],
	dist_gust_sigma: f32,
	dist_gust_tau_c: f32,
	dist_motor_asym: [f32; 4],
	dist_gyro_sigma: f32,
	dist_gyro_bias_walk: f32,
	dist_accel_sigma: f32,
	dist_seed: u64,
	// W2.4 D5/D6/D7 — 0-defaults = exactly-off (bit-identical legacy eval).
	dist_dropout_prob: f32,
	dist_dropout_len_steps: u32,
	dist_obs_delay_steps: u32,
	dist_torque_scale_jitter: f32,
	af_arm_length: f32, af_k_thrust: f32, af_k_drag: f32,
	af_inertia: [f32; 3], af_gravity: f32, af_dt: f32,
) -> PyResult<(f64, f64, f64)> {
	if controllers.is_empty() {
		return Err(pyo3::exceptions::PyValueError::new_err("eval_ensemble_closed_loop: no controllers"));
	}
	if init_qs.len() % 4 != 0 || init_omegas.len() % 3 != 0 || init_qs.len() / 4 != init_omegas.len() / 3 {
		return Err(pyo3::exceptions::PyValueError::new_err(
			"eval_ensemble_closed_loop: init_qs (4/ep) and init_omegas (3/ep) episode counts differ"));
	}
	let num_episodes = init_qs.len() / 4;
	let k = controllers.len();
	let stable_thresh_rad = stable_deg.to_radians();
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	// pid_fw: None — this scorer evaluates WNN ensembles and never builds a teacher.
	let af = AirframeRs { dt: af_dt, arm_length: af_arm_length, k_thrust: af_k_thrust,
		k_drag: af_k_drag, inertia: af_inertia, gravity: af_gravity, pid_fw: None };
	let mut sim = af.sim();
	let target = [0.0_f32, 0.0, 0.0];

	let mut n_stable = 0_usize;
	let mut sum_mean_err = 0.0_f64;
	let mut sum_steady = 0.0_f64;
	let mut steady_eps = 0_usize;

	for ep in 0..num_episodes {
		let init_q = [init_qs[ep * 4], init_qs[ep * 4 + 1], init_qs[ep * 4 + 2], init_qs[ep * 4 + 3]];
		let init_omega = [init_omegas[ep * 3], init_omegas[ep * 3 + 1], init_omegas[ep * 3 + 2]];
		let iy = yaw_from_quat_rs(init_q);
		// QSR/PLN decode coin: same per-episode seed for every committee member
		// (common random numbers across members — fair, variance-reduced). Pure fn
		// of the seed; deterministic modes ignore it.
		let coin_seed = crate::controller::disturbance_episode_seed(dist_seed, ep as u64);
		for c in &controllers {
			let mut cb = c.borrow_mut(py);
			cb.reset(iy);
			cb.set_decode_seed(coin_seed);
		}
		sim.reset(Some(init_q), Some(init_omega));
		// W2: per-episode weather (deterministic in (dist_seed, ep) — matches
		// the Metal kernel's derivation, so committee scores reproduce).
		if dist_enabled {
			let ep_seed = crate::controller::disturbance_episode_seed(dist_seed, ep as u64);
			sim.set_disturbance(
				dist_tau_bias, dist_gust_sigma, dist_gust_tau_c, dist_motor_asym,
				dist_gyro_sigma, dist_gyro_bias_walk, dist_accel_sigma, ep_seed,
				dist_dropout_prob, dist_dropout_len_steps,
				dist_obs_delay_steps, dist_torque_scale_jitter,
			);
		}

		let mut ep_sum_err = 0.0_f64;
		let mut tail_sum = 0.0_f64;
		let mut tail_cnt = 0_usize;
		let mut steps_done = 0_usize;
		let mut diverged = false;
		for t in 0..steps {
			if sim.is_unstable() {
				diverged = true;
				break;
			}
			let (gyro, accel) = sim.read_imu();
			// Collect each member's 4-motor command, aggregate per motor.
			let mut pwm = [0.0_f32; 4];
			if median {
				let mut per_motor: Vec<Vec<f32>> = vec![Vec::with_capacity(k); 4];
				for c in &controllers {
					let v = c.borrow_mut(py).step(gyro, accel, target);
					for m in 0..4 { per_motor[m].push(v[m]); }
				}
				for m in 0..4 {
					per_motor[m].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
					let mid = k / 2;
					pwm[m] = if k % 2 == 1 { per_motor[m][mid] }
					         else { 0.5 * (per_motor[m][mid - 1] + per_motor[m][mid]) };
				}
			} else {
				for c in &controllers {
					let v = c.borrow_mut(py).step(gyro, accel, target);
					for m in 0..4 { pwm[m] += v[m]; }
				}
				let kn = k as f32;
				for m in 0..4 { pwm[m] /= kn; }
			}
			sim.step(pwm);
			let err = sim.attitude_error(None) as f64;
			ep_sum_err += err;
			if t >= tail_start {
				tail_sum += err;
				tail_cnt += 1;
			}
			steps_done += 1;
		}
		let mean_err = ep_sum_err / steps_done.max(1) as f64;
		sum_mean_err += mean_err;
		if !diverged && mean_err <= stable_thresh_rad {
			n_stable += 1;
		}
		if tail_cnt > 0 {
			sum_steady += tail_sum / tail_cnt as f64;
			steady_eps += 1;
		}
	}
	let n = num_episodes.max(1) as f64;
	Ok((
		n_stable as f64 / n,
		(sum_mean_err / n).to_degrees(),
		if steady_eps > 0 { (sum_steady / steady_eps as f64).to_degrees() } else { f64::NAN },
	))
}

/// Held-out score for ONE classical controller (PID/LQR/MPC/LQI/MPCOF) on an
/// episode set, under the SAME sim + W2/W2.4 disturbance the WNN scorer uses.
///
/// This is the classical-baseline twin of eval_ensemble_closed_loop: identical
/// per-episode reset, disturbance derivation (disturbance_episode_seed), the
/// 80%-tail steady window, and the (stable_rate, mean_err_deg, steady_deg)
/// accounting — the ONLY difference is that a `Teacher` (built by the same
/// teacher_default() the training path trusts, so its linear model matches the
/// sim) drives the loop instead of a WnnController. Both the WNN and its five
/// classical rivals therefore come from ONE physics engine — no Python/Rust
/// cross-engine confound in a published table.
///
/// teacher_id: 0=PID 1=LQR 2=MPC 3=LQI 4=MPCOF (Teacher::from_id).
#[pyfunction]
#[pyo3(signature = (teacher_id, init_qs, init_omegas, steps, stable_deg,
	dist_enabled = false, dist_tau_bias = [0.0, 0.0, 0.0], dist_gust_sigma = 0.0,
	dist_gust_tau_c = 0.1, dist_motor_asym = [1.0, 1.0, 1.0, 1.0],
	dist_gyro_sigma = 0.0, dist_gyro_bias_walk = 0.0, dist_accel_sigma = 0.0,
	dist_seed = 0, dist_dropout_prob = 0.0, dist_dropout_len_steps = 0,
	dist_obs_delay_steps = 0, dist_torque_scale_jitter = 0.0,
	af_arm_length = 0.075, af_k_thrust = 2.4, af_k_drag = 0.05,
	af_inertia = [0.0023, 0.0023, 0.0046], af_gravity = 9.81, af_dt = 0.001,
	af_pid_att = [0.0; 12], af_pid_rate = [0.0; 12], af_pid_out_limit_n = 0.0,
	af_pid_hover_n = 0.0, af_pid_attitude_hz = 0.0, af_pid_lpf_hz = 0.0,
	use_estimator = false, est_kp = 2.0, est_ki = 0.1))]
#[allow(clippy::too_many_arguments)]
pub fn score_classical_baseline(
	teacher_id: u8,
	init_qs: Vec<f32>,      // 4 floats per episode (w, x, y, z)
	init_omegas: Vec<f32>,  // 3 floats per episode
	steps: usize,
	stable_deg: f64,
	dist_enabled: bool,
	dist_tau_bias: [f32; 3],
	dist_gust_sigma: f32,
	dist_gust_tau_c: f32,
	dist_motor_asym: [f32; 4],
	dist_gyro_sigma: f32,
	dist_gyro_bias_walk: f32,
	dist_accel_sigma: f32,
	dist_seed: u64,
	dist_dropout_prob: f32,
	dist_dropout_len_steps: u32,
	dist_obs_delay_steps: u32,
	dist_torque_scale_jitter: f32,
	af_arm_length: f32, af_k_thrust: f32, af_k_drag: f32,
	af_inertia: [f32; 3], af_gravity: f32, af_dt: f32,
	af_pid_att: [f64; 12], af_pid_rate: [f64; 12], af_pid_out_limit_n: f64,
	af_pid_hover_n: f64, af_pid_attitude_hz: f64, af_pid_lpf_hz: f64,
	use_estimator: bool, est_kp: f64, est_ki: f64,
) -> PyResult<(f64, f64, f64)> {
	if init_qs.len() % 4 != 0 || init_omegas.len() % 3 != 0
		|| init_qs.len() / 4 != init_omegas.len() / 3 {
		return Err(pyo3::exceptions::PyValueError::new_err(
			"score_classical_baseline: init_qs (4/ep) and init_omegas (3/ep) episode counts differ"));
	}
	let num_episodes = init_qs.len() / 4;
	let stable_thresh_rad = stable_deg.to_radians();
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	let af = AirframeRs { dt: af_dt, arm_length: af_arm_length, k_thrust: af_k_thrust,
		k_drag: af_k_drag, inertia: af_inertia, gravity: af_gravity,
		pid_fw: crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(
			af_pid_att, af_pid_rate, af_pid_out_limit_n, af_pid_hover_n,
			af_k_thrust as f64, (1.0 / af_dt.max(1e-9)).round() as u32,
			af_pid_attitude_hz, af_pid_lpf_hz) };
	let mut sim = af.sim();
	let mut teacher = af.teacher(teacher_id);
	let target = [0.0_f32, 0.0, 0.0];
	// Estimator-fed teachers (13/08 rule): comparison rows fly on a Mahony
	// estimate of the same noisy IMU the WNN reads — the true quaternion never
	// enters the rival's chain. false ⇒ the byte-identical oracle path.
	let mut est = if use_estimator {
		Some(crate::estimator::MahonyFilter::new(af_dt as f64, est_kp, est_ki))
	} else {
		None
	};

	let mut n_stable = 0_usize;
	let mut sum_mean_err = 0.0_f64;
	let mut sum_steady = 0.0_f64;
	let mut steady_eps = 0_usize;

	for ep in 0..num_episodes {
		let init_q = [init_qs[ep * 4], init_qs[ep * 4 + 1], init_qs[ep * 4 + 2], init_qs[ep * 4 + 3]];
		let init_omega = [init_omegas[ep * 3], init_omegas[ep * 3 + 1], init_omegas[ep * 3 + 2]];
		teacher.reset();
		sim.reset(Some(init_q), Some(init_omega));
		// Warm-start from the episode's initial attitude (converged-filter
		// assumption; the same yaw anchor the WNN gets at reset).
		if let Some(e) = est.as_mut() {
			e.reset(Some(init_q));
		}
		if dist_enabled {
			let ep_seed = crate::controller::disturbance_episode_seed(dist_seed, ep as u64);
			sim.set_disturbance(
				dist_tau_bias, dist_gust_sigma, dist_gust_tau_c, dist_motor_asym,
				dist_gyro_sigma, dist_gyro_bias_walk, dist_accel_sigma, ep_seed,
				dist_dropout_prob, dist_dropout_len_steps,
				dist_obs_delay_steps, dist_torque_scale_jitter,
			);
		}

		// MPCOF observer needs the action applied LAST step; 0.5 hover on step 0
		// (matches the training loop's last_applied init).
		let mut last_applied = [0.5f32; 4];
		let mut ep_sum_err = 0.0_f64;
		let mut tail_sum = 0.0_f64;
		let mut tail_cnt = 0_usize;
		let mut steps_done = 0_usize;
		let mut diverged = false;
		for t in 0..steps {
			if sim.is_unstable() {
				diverged = true;
				break;
			}
			let (gyro, accel) = sim.read_imu();
			let q = match est.as_mut() {
				Some(e) => e.update(gyro, accel),
				None => sim.quaternion(),
			};
			// Offset-free MPC observer (no-op for pid/lqr/mpc/lqi) — same call the
			// training loop makes before the teacher plans this step.
			teacher.observe(gyro, [
				last_applied[0] as f64, last_applied[1] as f64,
				last_applied[2] as f64, last_applied[3] as f64,
			]);
			let cmd = teacher.step_rs(q, gyro, target);
			let pwm = [cmd[0] as f32, cmd[1] as f32, cmd[2] as f32, cmd[3] as f32];
			sim.step(pwm);
			last_applied = pwm;
			let err = sim.attitude_error(None) as f64;
			ep_sum_err += err;
			if t >= tail_start {
				tail_sum += err;
				tail_cnt += 1;
			}
			steps_done += 1;
		}
		let mean_err = ep_sum_err / steps_done.max(1) as f64;
		sum_mean_err += mean_err;
		if !diverged && mean_err <= stable_thresh_rad {
			n_stable += 1;
		}
		if tail_cnt > 0 {
			sum_steady += tail_sum / tail_cnt as f64;
			steady_eps += 1;
		}
	}
	let n = num_episodes.max(1) as f64;
	Ok((
		n_stable as f64 / n,
		(sum_mean_err / n).to_degrees(),
		if steady_eps > 0 { (sum_steady / steady_eps as f64).to_degrees() } else { f64::NAN },
	))
}

/// D1 diagnostic (06/08/2026): TRACE TWIN of score_classical_baseline — identical
/// episode loop, plus a per-step attitude-error trace (radians, one Vec per episode,
/// diverged episodes shorter). Returns (stable_frac, err_deg, steady_deg, traces) so
/// callers can assert the aggregates match score_classical_baseline on the same
/// inputs (transcription-drift self-check). Deliberately a sibling, NOT a refactor:
/// the scoring path must stay byte-identical while a live chain imports this wheel.
/// Additive-only export — no ABI bump.
#[pyfunction]
#[pyo3(signature = (teacher_id, init_qs, init_omegas, steps, stable_deg,
	dist_enabled = false, dist_tau_bias = [0.0, 0.0, 0.0], dist_gust_sigma = 0.0,
	dist_gust_tau_c = 0.1, dist_motor_asym = [1.0, 1.0, 1.0, 1.0],
	dist_gyro_sigma = 0.0, dist_gyro_bias_walk = 0.0, dist_accel_sigma = 0.0,
	dist_seed = 0, dist_dropout_prob = 0.0, dist_dropout_len_steps = 0,
	dist_obs_delay_steps = 0, dist_torque_scale_jitter = 0.0,
	af_arm_length = 0.075, af_k_thrust = 2.4, af_k_drag = 0.05,
	af_inertia = [0.0023, 0.0023, 0.0046], af_gravity = 9.81, af_dt = 0.001,
	af_pid_att = [0.0; 12], af_pid_rate = [0.0; 12], af_pid_out_limit_n = 0.0,
	af_pid_hover_n = 0.0, af_pid_attitude_hz = 0.0, af_pid_lpf_hz = 0.0,
	use_estimator = false, est_kp = 2.0, est_ki = 0.1))]
#[allow(clippy::too_many_arguments)]
pub fn trace_classical_baseline(
	teacher_id: u8,
	init_qs: Vec<f32>,      // 4 floats per episode (w, x, y, z)
	init_omegas: Vec<f32>,  // 3 floats per episode
	steps: usize,
	stable_deg: f64,
	dist_enabled: bool,
	dist_tau_bias: [f32; 3],
	dist_gust_sigma: f32,
	dist_gust_tau_c: f32,
	dist_motor_asym: [f32; 4],
	dist_gyro_sigma: f32,
	dist_gyro_bias_walk: f32,
	dist_accel_sigma: f32,
	dist_seed: u64,
	dist_dropout_prob: f32,
	dist_dropout_len_steps: u32,
	dist_obs_delay_steps: u32,
	dist_torque_scale_jitter: f32,
	af_arm_length: f32, af_k_thrust: f32, af_k_drag: f32,
	af_inertia: [f32; 3], af_gravity: f32, af_dt: f32,
	af_pid_att: [f64; 12], af_pid_rate: [f64; 12], af_pid_out_limit_n: f64,
	af_pid_hover_n: f64, af_pid_attitude_hz: f64, af_pid_lpf_hz: f64,
	use_estimator: bool, est_kp: f64, est_ki: f64,
) -> PyResult<(f64, f64, f64, Vec<Vec<f64>>)> {
	if init_qs.len() % 4 != 0 || init_omegas.len() % 3 != 0
		|| init_qs.len() / 4 != init_omegas.len() / 3 {
		return Err(pyo3::exceptions::PyValueError::new_err(
			"trace_classical_baseline: init_qs (4/ep) and init_omegas (3/ep) episode counts differ"));
	}
	let num_episodes = init_qs.len() / 4;
	let stable_thresh_rad = stable_deg.to_radians();
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	let af = AirframeRs { dt: af_dt, arm_length: af_arm_length, k_thrust: af_k_thrust,
		k_drag: af_k_drag, inertia: af_inertia, gravity: af_gravity,
		pid_fw: crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(
			af_pid_att, af_pid_rate, af_pid_out_limit_n, af_pid_hover_n,
			af_k_thrust as f64, (1.0 / af_dt.max(1e-9)).round() as u32,
			af_pid_attitude_hz, af_pid_lpf_hz) };
	let mut sim = af.sim();
	let mut teacher = af.teacher(teacher_id);
	let target = [0.0_f32, 0.0, 0.0];
	// Estimator-fed teachers (13/08 rule): comparison rows fly on a Mahony
	// estimate of the same noisy IMU the WNN reads — the true quaternion never
	// enters the rival's chain. false ⇒ the byte-identical oracle path.
	let mut est = if use_estimator {
		Some(crate::estimator::MahonyFilter::new(af_dt as f64, est_kp, est_ki))
	} else {
		None
	};

	let mut n_stable = 0_usize;
	let mut sum_mean_err = 0.0_f64;
	let mut sum_steady = 0.0_f64;
	let mut steady_eps = 0_usize;
	let mut traces: Vec<Vec<f64>> = Vec::with_capacity(num_episodes);

	for ep in 0..num_episodes {
		let init_q = [init_qs[ep * 4], init_qs[ep * 4 + 1], init_qs[ep * 4 + 2], init_qs[ep * 4 + 3]];
		let init_omega = [init_omegas[ep * 3], init_omegas[ep * 3 + 1], init_omegas[ep * 3 + 2]];
		teacher.reset();
		sim.reset(Some(init_q), Some(init_omega));
		// Warm-start from the episode's initial attitude (converged-filter
		// assumption; the same yaw anchor the WNN gets at reset).
		if let Some(e) = est.as_mut() {
			e.reset(Some(init_q));
		}
		if dist_enabled {
			let ep_seed = crate::controller::disturbance_episode_seed(dist_seed, ep as u64);
			sim.set_disturbance(
				dist_tau_bias, dist_gust_sigma, dist_gust_tau_c, dist_motor_asym,
				dist_gyro_sigma, dist_gyro_bias_walk, dist_accel_sigma, ep_seed,
				dist_dropout_prob, dist_dropout_len_steps,
				dist_obs_delay_steps, dist_torque_scale_jitter,
			);
		}

		let mut last_applied = [0.5f32; 4];
		let mut ep_sum_err = 0.0_f64;
		let mut tail_sum = 0.0_f64;
		let mut tail_cnt = 0_usize;
		let mut steps_done = 0_usize;
		let mut diverged = false;
		let mut ep_trace: Vec<f64> = Vec::with_capacity(steps);
		for t in 0..steps {
			if sim.is_unstable() {
				diverged = true;
				break;
			}
			let (gyro, accel) = sim.read_imu();
			let q = match est.as_mut() {
				Some(e) => e.update(gyro, accel),
				None => sim.quaternion(),
			};
			teacher.observe(gyro, [
				last_applied[0] as f64, last_applied[1] as f64,
				last_applied[2] as f64, last_applied[3] as f64,
			]);
			let cmd = teacher.step_rs(q, gyro, target);
			let pwm = [cmd[0] as f32, cmd[1] as f32, cmd[2] as f32, cmd[3] as f32];
			sim.step(pwm);
			last_applied = pwm;
			let err = sim.attitude_error(None) as f64;
			ep_trace.push(err);
			ep_sum_err += err;
			if t >= tail_start {
				tail_sum += err;
				tail_cnt += 1;
			}
			steps_done += 1;
		}
		let mean_err = ep_sum_err / steps_done.max(1) as f64;
		sum_mean_err += mean_err;
		if !diverged && mean_err <= stable_thresh_rad {
			n_stable += 1;
		}
		if tail_cnt > 0 {
			sum_steady += tail_sum / tail_cnt as f64;
			steady_eps += 1;
		}
		traces.push(ep_trace);
	}
	let n = num_episodes.max(1) as f64;
	Ok((
		n_stable as f64 / n,
		(sum_mean_err / n).to_degrees(),
		if steady_eps > 0 { (sum_steady / steady_eps as f64).to_degrees() } else { f64::NAN },
		traces,
	))
}

#[cfg(test)]
mod teacher_bank_tests {
	use super::*;

	/// Every id Teacher::from_id can build must reach a DISTINCT teacher through
	/// the bank. The pre-05/08/2026 bank clamped >2 to PID, so lqi and mpcof both
	/// silently trained against PID and produced bit-identical runs. Asserting on
	/// the discriminant means widening from_id without widening the bank fails
	/// here instead of in six hours of wasted GPU time.
	#[test]
	fn bank_maps_every_id_to_its_own_teacher() {
		let af = AirframeRs::DEFAULT;
		let mut bank = TeacherBank::new(af);
		let mut kinds: Vec<String> = Vec::new();
		for id in 0u8..(TEACHER_IDS as u8) {
			let t = bank.get_mut(id);
			kinds.push(match t {
				Teacher::Pid(_) => "pid",
				// DEFAULT carries no cascade gains, so id 0 must still resolve to the
				// legacy single loop here. If this ever reports "pidfw" the fallback
				// that protects the synthetic-plant parity anchors has broken.
				Teacher::PidFw(_) => "pidfw",
				Teacher::Lqr(_) => "lqr",
				Teacher::Mpc(_) => "mpc",
				Teacher::Lqi(_) => "lqi",
				Teacher::MpcOf(_) => "mpcof",
			}.to_string());
		}
		assert_eq!(kinds, vec!["pid", "lqr", "mpc", "lqi", "mpcof"],
			"TeacherBank remapped an id — ids > 2 used to collapse to PID");
		// And an id from_id genuinely cannot build still falls back to PID.
		assert!(matches!(bank.get_mut(TEACHER_IDS as u8), Teacher::Pid(_)));
	}
}

#[cfg(test)]
mod batch_progress_tests {
	use super::BatchProgress;

	/// The heartbeat must not deadlock, must join promptly when the batch ends
	/// (it polls in 500ms slices rather than sleeping the whole interval), and
	/// must count every completion. A hung join here would stall EVERY batch.
	#[test]
	fn heartbeat_counts_and_joins_promptly() {
		let p = BatchProgress::start_with_interval("test-batch", 4, 1);
		for _ in 0..4 { p.tick(); }
		// Generous margin: the poll slices are 500ms and the interval is 1s, so
		// 2.6s guarantees at least two emissions even with spawn jitter.
		std::thread::sleep(std::time::Duration::from_millis(2600));
		let emitted = p.emit_count();
		let t0 = std::time::Instant::now();
		p.finish("test-batch", 4);
		assert!(emitted >= 2, "heartbeat emitted {emitted} lines in 2.6s at a 1s interval — it is not firing");
		assert!(t0.elapsed().as_millis() < 900, "finish() took {}ms — join is not prompt", t0.elapsed().as_millis());
	}

	/// WNN_PROGRESS_SECS=0 disables it entirely: no thread, and finish() is a
	/// no-op that must still be safe to call.
	#[test]
	fn disabled_by_zero_interval() {
		let p = BatchProgress::start_with_interval("off", 10, 0);
		p.tick();
		std::thread::sleep(std::time::Duration::from_millis(1200));
		assert_eq!(p.emit_count(), 0, "disabled heartbeat still emitted");
		p.finish("off", 10);
	}
}
