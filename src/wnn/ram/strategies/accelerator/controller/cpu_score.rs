//! CPU (rayon) batch controller scorer — the CPU twin of score_controllers_metal.
//!
//! Used when WNN_CONTROLLER_GPU_EVAL=0 (the IDS worker owns the GPU: its non-preemptible
//! kernels starve the controller's Metal command buffer for tens of minutes). Rolls out
//! each controller on its OWN thread-local clone (WnnController: Clone, 09/07/2026),
//! rayon-parallel across the population — so the GA scoring uses all allotted cores
//! instead of the old serial Python per-step loop.
//!
//! Returns 12-metric rows to match the GPU scorer's contract. It computes the 5 metrics
//! the GA fitness (ControllerHarmonic: err²+stable+jerk+mono) actually ranks — reward,
//! err_rad, stable, jerk, mono. The 7 transient/display metrics (steady, rise,
//! settle×2, itae, iae, ise) are left 0.0 here; the held-out REPORT (once per stage)
//! still uses the GPU scorer for those.
//!
//! SEMANTICS UNIFIED 12/07/2026 (Luiz order): mono = the LAST decision step's
//! thermometer violations per episode, jerk = per-episode mean of |Δpwm| —
//! both averaged over episodes, exactly the GPU kernel's aggregation (and the
//! old serial Python fallback's "last emitted thermometer" for mono). The
//! original rayon port accidentally accumulated mono per-step / jerk globally;
//! runs searched before this date rank fitness under the old semantics.

use crate::controller::{
	compute_reward, disturbance_episode_seed, monotonicity_violations_core, yaw_from_quat_rs,
	AttitudeSim, WnnController,
};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Roll out ONE controller over `num_eps` episodes with EXPLICIT initial conditions
/// (q0/omega0, same contract as score_controllers_metal) and return a 12-metric row.
/// Mirrors eval_closed_loop_rs's inner loop; the caller supplies a clone it may mutate.
/// pub(crate): the GPU parity suite (metal_controller.rs tests) scores through THIS
/// to pin CPU-scorer ↔ kernel agreement on all 5 fitness metrics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rollout_one(
	c: &mut WnnController,
	q0: &[f32],
	omega0: &[f32],
	num_eps: usize,
	steps: usize,
	dt: f32,
	arm: f32,
	k_thrust: f32,
	k_drag: f32,
	inertia: [f32; 3],
	gravity: f32,
	target: [f32; 3],
	dist_enabled: bool,
	dist_tau_bias: [f32; 3],
	dist_gust_sigma: f32,
	dist_gust_tau_c: f32,
	dist_motor_asym: [f32; 4],
	dist_gyro_sigma: f32,
	dist_gyro_bias_walk: f32,
	dist_accel_sigma: f32,
	dist_seed: u64,
	// W2.4 D5/D6/D7 — 0-defaults = exactly-off (bit-identical legacy rollout).
	dist_dropout_prob: f32,
	dist_dropout_len_steps: u32,
	dist_obs_delay_steps: u32,
	dist_torque_scale_jitter: f32,
	levels_per_motor: usize,
	num_motors: usize,
	// Overactuated Phase 1: None = the legacy quad sim (bit-identical
	// pre-geometry path via step()). Some(rows) ⇒ step_n on the N-rotor
	// geometry (rows already validated + rotor_asym applied by the caller).
	geometry: Option<&[[f32; 9]]>,
	rotor_asym: Option<&[f32]>,
	// Overactuated Phase 2: allocator-LQR residual baseline (twin of the
	// kernel's alloc_step path). When set, the WNN output is a signed
	// residual: pwm = clamp(base + clamp((wnn−0.5)·scale, ±rclamp), 0..1).
	// Caller guarantees num_rotors == num_motors and action_repeat == 1.
	alloc: Option<&crate::optimal::AllocBaseline>,
	residual_scale: f32,
	residual_clamp: f32,
) -> [f64; 13] {
	let mut sim = AttitudeSim::new(dt, arm, k_thrust, k_drag, inertia, gravity);
	if let Some(rows) = geometry {
		// Validated in score_controllers_cpu before the rayon fan-out; these
		// cannot fail here (non-empty rows, asym len == N).
		sim.set_geometry_core(rows.to_vec()).expect("validated geometry");
		sim.set_rotor_asym_core(rotor_asym.map(|a| a.to_vec())).expect("validated rotor_asym");
	}
	// Excess-effort precompute (kernel twin): B columns of the TRUE table
	// (per-unit-thrust wrench contributions) + effective k (asym baked), and
	// the nominal pinv rows from the alloc baseline. Only on alloc runs.
	let excess_geo: Option<(Vec<[f32; 6]>, Vec<f32>)> = match (geometry, alloc) {
		(Some(rows), Some(_)) => {
			let geo = crate::overactuated::RotorGeometry::from_rows(rows).expect("validated geometry");
			let b = geo.allocation_matrix(); // 6×N, b[row][rotor]
			let n_r = geo.num_rotors();
			let bcols: Vec<[f32; 6]> = (0..n_r)
				.map(|i| [b[0][i], b[1][i], b[2][i], b[3][i], b[4][i], b[5][i]])
				.collect();
			let keff: Vec<f32> = geo.rotors.iter().enumerate()
				.map(|(i, r)| r.k_thrust * rotor_asym.map_or(1.0, |a| a[i]))
				.collect();
			Some((bcols, keff))
		}
		_ => None,
	};
	let stable_thresh_rad = 5.0_f64.to_radians();
	let mut sum_reward = 0.0f64;
	let mut sum_err = 0.0f64;
	// Per-episode means summed here, averaged over episodes at the end — the
	// GPU host's aggregation (sum_jerk_per_g / sum_mono_per_g ÷ episodes).
	let mut sum_jerk = 0.0f64;
	let mut sum_mono = 0.0f64;
	// Allocation-effort (Phase 3): per-episode mean Σ_m pwm² of the APPLIED
	// command — the kernel's out_effort twin.
	let mut sum_effort = 0.0f64;
	let mut n_stable = 0usize;
	// Transient/display metrics (20/07/2026): previously hardcoded 0.0 on this
	// path, so any run with WNN_CONTROLLER_GPU_EVAL=0 reported steady=0.00° and
	// friends. Definitions MIRROR controller_rollout.metal's kernel verbatim
	// (tail-20% steady window, 10%-of-initial rise, ±2° absolute / ±5%-of-initial
	// relative settle bands via LAST excursion, dt-weighted ITAE/IAE/ISE, and the
	// diverged→full-duration sentinels) so CPU and GPU rows stay comparable.
	let mut sum_steady = 0.0f64;
	let mut sum_rise = 0.0f64;
	let mut sum_settleab = 0.0f64;
	let mut sum_settlere = 0.0f64;
	let mut sum_itae = 0.0f64;
	let mut sum_iae = 0.0f64;
	let mut sum_ise = 0.0f64;
	// Metal: tail_start = ceil(steps * 0.80); band_abs = radians(2.0).
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	let dt_f = dt as f64;   // dt is f32 on this path; transient math is f64
	let full_dur = steps as f64 * dt_f;
	const BAND_ABS: f64 = 0.034_906_585_0; // radians(2.0), the kernel's literal

	for ep in 0..num_eps {
		// 14/07/2026: cooperative SIGTERM cancellation on the SCORE path (the GPU
		// scorer already polls; the CPU scorer did not, so a paused re-eval waited
		// out the whole num_eps×steps batch — ~80s — before the dump could land).
		// Poll per episode: a ~1 ns Relaxed atomic load, negligible vs one 2000-step
		// episode. On cancel, bail with the aggregate over episodes done so far; the
		// caller is unwinding to dump+resume, and resume re-evaluates anyway, so the
		// partial value is discarded (never selected on).
		if ram_core::cancel::check_cancel() {
			break;
		}
		let q = [q0[ep * 4], q0[ep * 4 + 1], q0[ep * 4 + 2], q0[ep * 4 + 3]];
		let om = [omega0[ep * 3], omega0[ep * 3 + 1], omega0[ep * 3 + 2]];
		c.reset(yaw_from_quat_rs(q)); // yaw-anchor: seed heading from the episode's true yaw
		// QSR/PLN decode coin: seed per-episode with the SAME channel-15 derivation
		// the Metal scorer uses, so the stochastic decode is reproducible AND bit-
		// identical to score_controllers_metal. Unconditional (pure fn of the seed);
		// deterministic modes ignore decode_run_seed.
		c.set_decode_seed(disturbance_episode_seed(dist_seed, ep as u64));
		sim.reset(Some(q), Some(om));
		if dist_enabled {
			let eps_seed = disturbance_episode_seed(dist_seed, ep as u64);
			sim.set_disturbance(
				dist_tau_bias, dist_gust_sigma, dist_gust_tau_c, dist_motor_asym,
				dist_gyro_sigma, dist_gyro_bias_walk, dist_accel_sigma, eps_seed,
				dist_dropout_prob, dist_dropout_len_steps, dist_obs_delay_steps,
				dist_torque_scale_jitter,
			);
		}

		let mut ep_sum_err = 0.0f64;
		let mut ep_jerk = 0.0f64;
		let mut ep_jerk_count = 0usize;
		let mut ep_effort = 0.0f64;
		let mut mono_last = 0.0f64;
		let mut prev_pwm = vec![0.5f32; num_motors];
		let mut first_step = true;
		let mut ep_steps = 0usize;
		let mut diverged = false;
		// Per-episode transient state (kernel twins).
		let mut tail_sum_err = 0.0f64;
		let mut tail_cnt = 0usize;
		let mut init_err = -1.0f64;      // <0 ⇒ not yet captured (kernel sentinel)
		let mut band_rel = 0.0f64;
		let mut rise_s = full_dur;
		let mut rise_done = false;
		let mut last_exc_abs = 0.0f64;
		let mut last_exc_rel = 0.0f64;
		let (mut ep_itae, mut ep_iae, mut ep_ise) = (0.0f64, 0.0f64, 0.0f64);
		for _t in 0..steps {
			if sim.is_unstable() {
				diverged = true;
				break;
			}
			let (gyro, accel) = sim.read_imu();
			let mut pwm = c.step(gyro, accel, target);
			if let Some(ab) = alloc {
				// Same q/gyro the kernel's alloc_step sees (true attitude,
				// noisy gyro). mono_last below stays the RAW WNN thermometer
				// (composition never touches output cells) — kernel-identical.
				let mut base = vec![0.0f32; num_motors];
				ab.pwm(sim.quaternion(), gyro, target, &mut base);
				// Mode-derived residual anchor (ABI 12; = NEUTRAL_DECODE under QUAD).
				let neutral = c.neutral_f32();
				for m in 0..num_motors {
					let r = ((pwm[m] - neutral) * residual_scale)
						.clamp(-residual_clamp, residual_clamp);
					pwm[m] = (base[m] + r).clamp(0.0, 1.0);
				}
			}

			// Motor jerk: sqrt(Σ_m (Δpwm_m)²) per step (the L2 norm of the
			// per-step motor-delta vector) — EXACTLY the GPU kernel's formula
			// (controller_rollout.metal: sum_jerk += sqrt(dj); jerk_count++),
			// normalized PER EPISODE like out_jerk. First step has no prev.
			if !first_step {
				let mut step_jerk = 0.0f64;
				for m in 0..num_motors {
					let d = (pwm[m] - prev_pwm[m]) as f64;
					step_jerk += d * d;
				}
				ep_jerk += step_jerk.sqrt();
				ep_jerk_count += 1;
			}
			prev_pwm.copy_from_slice(&pwm);
			first_step = false;
			// Allocation-effort: EXCESS vs the pinv optimum for the same
			// realized wrench on alloc runs (kernel twin — see the Metal
			// comment for why raw Σu² is gameable); raw Σ pwm² otherwise.
			if let (Some((bcols, keff)), Some(ab)) = (&excess_geo, alloc) {
				let mut w6 = [0.0f32; 6];
				let mut sum_t2 = 0.0f32;
				for i in 0..num_motors {
					let pcl = pwm[i].clamp(0.0, 1.0);
					let t = keff[i] * pcl * pcl;
					sum_t2 += t * t;
					for j in 0..6 {
						w6[j] += bcols[i][j] * t;
					}
				}
				let mut sum_topt2 = 0.0f32;
				for i in 0..num_motors {
					let row = &ab.rows[i];
					let mut topt = 0.0f32;
					for j in 0..6 {
						topt += row[j] * w6[j];
					}
					let topt = topt.max(0.0);
					sum_topt2 += topt * topt;
				}
				ep_effort += (sum_t2 - sum_topt2).max(0.0) as f64;
			} else {
				let mut se = 0.0f64;
				for m in 0..num_motors {
					se += (pwm[m] as f64) * (pwm[m] as f64);
				}
				ep_effort += se;
			}

			// Mono: keep the LAST decision step's violation count (the GPU's
			// mono_last / the serial fallback's "last emitted thermometer").
			// On action-repeat hold steps get_last_output_cells still holds the
			// decision step's cells, so this stays the decision value.
			if let Ok(mv) = monotonicity_violations_core(&c.get_last_output_cells(), levels_per_motor, num_motors, c.memory_mode_u8(), c.output_decode_u8()) {
				mono_last = mv as f64;
			}

			if geometry.is_some() {
				// Validated N == num_motors == pwm.len() → cannot fail here.
				sim.step_n_core(&pwm).expect("validated step_n");
			} else {
				// Legacy quad path — the EXACT pre-geometry call (bit-identical).
				sim.step([pwm[0], pwm[1], pwm[2], pwm[3]]);
			}
			let err = sim.attitude_error(None);
			sum_reward += compute_reward(err, 0.0, 0, 0.0, 0.0) as f64;
			ep_sum_err += err as f64;
			// --- Transient-speed metrics (single pass; kernel-order twin) ---
			// `_t` is the pre-increment step index, matching the kernel's `t`.
			let errd = err as f64;
			if _t >= tail_start {
				tail_sum_err += errd;
				tail_cnt += 1;
			}
			let t_s = _t as f64 * dt_f;
			if init_err < 0.0 {
				init_err = errd;
				band_rel = 0.05 * init_err;
			}
			ep_iae += errd * dt_f;
			ep_ise += errd * errd * dt_f;
			ep_itae += t_s * errd * dt_f;
			if !rise_done && errd <= 0.10 * init_err {
				rise_s = t_s;
				rise_done = true;
			}
			if errd >= BAND_ABS {
				last_exc_abs = t_s + dt_f;
			}
			if errd >= band_rel {
				last_exc_rel = t_s + dt_f;
			}
			ep_steps += 1;
		}
		let mean_err = ep_sum_err / ep_steps.max(1) as f64;
		sum_err += mean_err;
		sum_jerk += if ep_jerk_count > 0 { ep_jerk / ep_jerk_count as f64 } else { 0.0 };
		sum_effort += if ep_steps > 0 { ep_effort / ep_steps as f64 } else { 0.0 };
		sum_mono += mono_last;
		// Steady: mean err over the tail-20% window. Diverged before reaching it
		// ⇒ no samples ⇒ fall back to the whole-episode mean (kernel's else-branch).
		sum_steady += if tail_cnt > 0 {
			tail_sum_err / tail_cnt as f64
		} else if ep_steps > 0 {
			ep_sum_err / ep_steps as f64
		} else {
			0.0
		};
		// Transient times: a diverged episode never settles → worst-case sentinels,
		// so its late error blow-up can't fool last-excursion into a "fast" settle.
		if diverged {
			sum_rise += full_dur;
			sum_settleab += full_dur;
			sum_settlere += full_dur;
		} else {
			sum_rise += rise_s;
			sum_settleab += last_exc_abs.min(full_dur);
			sum_settlere += last_exc_rel.min(full_dur);
		}
		sum_itae += ep_itae;
		sum_iae += ep_iae;
		sum_ise += ep_ise;
		if !diverged && mean_err <= stable_thresh_rad {
			n_stable += 1;
		}
	}

	let n = num_eps.max(1) as f64;
	// Row order matches metal_controller.rs: [reward, err_rad, stable, jerk, mono,
	// steady, rise, settle_abs, settle_rel, itae, iae, ise, effort]. ALL 13 are
	// computed here since 20/07/2026 (the 7 transient/display metrics used to be
	// hardcoded 0.0, so CPU-scored runs reported steady=0.00° — see the kernel
	// twin in controller_rollout.metal for the shared definitions).
	[
		sum_reward / n,
		sum_err / n,
		n_stable as f64 / n,
		sum_jerk / n,
		sum_mono / n,
		sum_steady / n,
		sum_rise / n,
		sum_settleab / n,
		sum_settlere / n,
		sum_itae / n,
		sum_iae / n,
		sum_ise / n,
		sum_effort / n,
	]
}

/// PyO3 entry: CPU-score a population (uniform shape) rayon-parallel. Same signature +
/// return contract as score_controllers_metal (per-genome 12-metric rows, input order).
#[pyfunction]
#[pyo3(signature = (
	controllers, q0, omega0, num_episodes, steps,
	dt = 0.001, arm_length = 0.075, k_thrust = 2.4, k_drag = 0.05,
	inertia = [0.0023, 0.0023, 0.0046], gravity = 9.81,
	target = [0.0, 0.0, 0.0],
	dist_enabled = false,
	dist_tau_bias = [0.0, 0.0, 0.0],
	dist_gust_sigma = 0.0,
	dist_gust_tau_c = 0.1,
	dist_motor_asym = [1.0, 1.0, 1.0, 1.0],
	dist_gyro_sigma = 0.0,
	dist_gyro_bias_walk = 0.0,
	dist_accel_sigma = 0.0,
	dist_seed = 0,
	// W2.4 D5 dropout/freeze + D6 latency + D7 torque-scale jitter —
	// 0-defaults = exactly-off (bit-identical legacy rollout).
	dist_dropout_prob = 0.0,
	dist_dropout_len_steps = 0,
	dist_obs_delay_steps = 0,
	dist_torque_scale_jitter = 0.0,
	// Overactuated Phase 1 — None = legacy quad sim. Same contract as
	// score_controllers_metal: rows [px,py,pz, ax,ay,az, spin, k_thrust,
	// k_drag] (pass the PERTURBED table for tilt/pos error); rotor_asym =
	// per-rotor thrust multipliers (N-rotor D3 twin).
	geometry = None,
	rotor_asym = None,
	// Overactuated Phase 2 — allocator-LQR residual baseline (NOMINAL rows;
	// same contract + params as score_controllers_metal's alloc_* kwargs).
	alloc_rows = None,
	alloc_q_att = 12.0,
	alloc_q_rate = 1.0,
	alloc_r_ctrl = 1.0,
	alloc_tau_max = 0.144,
	alloc_f_hover = None,
	alloc_lambda = 1e-6,
	residual_scale = 1.0,
	residual_clamp = 0.4,
))]
#[allow(clippy::too_many_arguments)]
pub fn score_controllers_cpu(
	py: Python<'_>,
	controllers: Vec<PyRef<WnnController>>,
	q0: Vec<f32>,
	omega0: Vec<f32>,
	num_episodes: usize,
	steps: usize,
	dt: f32,
	arm_length: f32,
	k_thrust: f32,
	k_drag: f32,
	inertia: [f32; 3],
	gravity: f32,
	target: [f32; 3],
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
	geometry: Option<Vec<[f32; 9]>>,
	rotor_asym: Option<Vec<f32>>,
	alloc_rows: Option<Vec<[f32; 9]>>,
	alloc_q_att: f64,
	alloc_q_rate: f64,
	alloc_r_ctrl: f64,
	alloc_tau_max: f64,
	alloc_f_hover: Option<f64>,
	alloc_lambda: f32,
	residual_scale: f32,
	residual_clamp: f32,
) -> PyResult<Vec<Vec<f64>>> {
	if controllers.is_empty() {
		return Ok(vec![]);
	}
	let (num_motors, levels, ..) = controllers[0].gpu_dims();
	// Validate the geometry ONCE before the rayon fan-out (mirrors the Metal
	// scorer's guards) so rollout_one can unwrap unconditionally.
	if let Some(rows) = &geometry {
		if rows.is_empty() || rows.len() != num_motors {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"geometry has {} rotors but controllers emit num_motors={} PWMs — they must match.",
				rows.len(), num_motors)));
		}
		if let Some(a) = &rotor_asym {
			if a.len() != rows.len() {
				return Err(pyo3::exceptions::PyValueError::new_err(format!(
					"rotor_asym len {} != num_rotors {}", a.len(), rows.len())));
			}
		}
	} else if rotor_asym.is_some() {
		return Err(pyo3::exceptions::PyValueError::new_err(
			"rotor_asym requires geometry (the quad path models motor asymmetry \
			 via dist_motor_asym instead)".to_string()));
	}
	let alloc = match &alloc_rows {
		Some(rows) => {
			let ab = crate::optimal::AllocBaseline::build(
				rows, inertia, alloc_q_att, alloc_q_rate, alloc_r_ctrl,
				alloc_tau_max, alloc_f_hover, alloc_lambda,
			).map_err(pyo3::exceptions::PyValueError::new_err)?;
			if ab.num_rotors() != num_motors {
				return Err(pyo3::exceptions::PyValueError::new_err(format!(
					"alloc baseline has {} rotors but controllers emit num_motors={} PWMs",
					ab.num_rotors(), num_motors)));
			}
			if controllers[0].action_repeat_n() != 1 {
				// The kernel composes-then-holds; the CPU twin cannot see hold
				// boundaries through c.step's held raw output. Refuse loudly.
				return Err(pyo3::exceptions::PyValueError::new_err(
					"alloc residual on the CPU scorer requires action_repeat == 1".to_string()));
			}
			Some(ab)
		}
		None => None,
	};
	// Clone out of the (non-Send) PyRefs into owned WnnControllers so rayon can roll
	// them out across threads. WnnController::Clone DEEP-copies cells — and a heavy mode
	// (TERNARY accumulates ~30× QUAD's cells: 17.5M vs 0.5M) cloned across the WHOLE
	// population at once was the 15/07 40GB OOM. But rollout_one is READ-ONLY on cells,
	// so we clone+score in CHUNKS sized to a memory budget: the light modes still take
	// the whole population in one chunk (full parallelism), heavy modes only a few
	// genomes at a time. Bit-identical scores — only the clone lifetime changes.
	let max_cells = controllers.iter().map(|c| c.total_cells()).max().unwrap_or(0);
	const CLONE_BUDGET_BYTES: usize = 6 * 1024 * 1024 * 1024; // ≤6GB of live clones
	// CLONE-ONLY bytes per cell: what one read-only rollout copy of the cells costs
	// (DashMap(FxHasher) entry + load-factor overhead). Deliberately NOT the same
	// quantity as evaluator.py's bytes_per_cell (1300), which is the train+score
	// PEAK per cell and therefore also carries the training working set. Keep the
	// distinction explicit — the two were silently 4x apart before it was labelled.
	const BYTES_PER_CELL: usize = 160;
	let per_genome = max_cells.saturating_mul(BYTES_PER_CELL).max(1);
	let n_ctrl = controllers.len();
	let chunk = (CLONE_BUDGET_BYTES / per_genome).clamp(1, n_ctrl.max(1));
	let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n_ctrl);
	for chunk_ctrls in controllers.chunks(chunk) {
		// Clone THIS chunk under the GIL, roll it out GIL-free, drop the clones before
		// the next chunk → peak clone memory = chunk × per-genome, not N × per-genome.
		let mut owned: Vec<WnnController> = chunk_ctrls.iter().map(|c| (**c).clone()).collect();
		let chunk_rows: Vec<Vec<f64>> = py.allow_threads(|| {
		owned
			.par_iter_mut()
			.map(|c| {
				// Cooperative cancel: a genome rayon starts AFTER a SIGTERM skips its
				// whole rollout (13-wide zero row, discarded on the unwinding resume).
				if ram_core::cancel::check_cancel() {
					return vec![0.0f64; 13];
				}
				rollout_one(
					c, &q0, &omega0, num_episodes, steps, dt, arm_length, k_thrust, k_drag,
					inertia, gravity, target, dist_enabled, dist_tau_bias, dist_gust_sigma,
					dist_gust_tau_c, dist_motor_asym, dist_gyro_sigma, dist_gyro_bias_walk,
					dist_accel_sigma, dist_seed,
					dist_dropout_prob, dist_dropout_len_steps, dist_obs_delay_steps,
					dist_torque_scale_jitter, levels, num_motors,
					geometry.as_deref(), rotor_asym.as_deref(),
					alloc.as_ref(), residual_scale, residual_clamp,
				)
				.to_vec()
			})
			.collect()
	});
		rows.extend(chunk_rows);
	}
	drop(controllers); // release the Python borrows
	Ok(rows)
}
