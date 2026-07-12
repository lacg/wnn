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
//! err_rad, stable, jerk, mono — mirroring eval_closed_loop_rs. The 7 transient/display
//! metrics (steady, rise, settle×2, itae, iae, ise) are left 0.0 here; the held-out
//! REPORT (once per stage) still uses the GPU scorer for those.

use crate::controller::{
	compute_reward, disturbance_episode_seed, monotonicity_violations_core, yaw_from_quat_rs,
	AttitudeSim, WnnController,
};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Roll out ONE controller over `num_eps` episodes with EXPLICIT initial conditions
/// (q0/omega0, same contract as score_controllers_metal) and return a 12-metric row.
/// Mirrors eval_closed_loop_rs's inner loop; the caller supplies a clone it may mutate.
#[allow(clippy::too_many_arguments)]
fn rollout_one(
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
	levels_per_motor: usize,
	num_motors: usize,
	// Overactuated Phase 1: None = the legacy quad sim (bit-identical
	// pre-geometry path via step()). Some(rows) ⇒ step_n on the N-rotor
	// geometry (rows already validated + rotor_asym applied by the caller).
	geometry: Option<&[[f32; 9]]>,
	rotor_asym: Option<&[f32]>,
) -> [f64; 12] {
	let mut sim = AttitudeSim::new(dt, arm, k_thrust, k_drag, inertia, gravity);
	if let Some(rows) = geometry {
		// Validated in score_controllers_cpu before the rayon fan-out; these
		// cannot fail here (non-empty rows, asym len == N).
		sim.set_geometry_core(rows.to_vec()).expect("validated geometry");
		sim.set_rotor_asym_core(rotor_asym.map(|a| a.to_vec())).expect("validated rotor_asym");
	}
	let stable_thresh_rad = 5.0_f64.to_radians();
	let mut sum_reward = 0.0f64;
	let mut sum_err = 0.0f64;
	let mut sum_jerk = 0.0f64; // Σ over steps of sqrt(Σ_m (Δpwm_m)²) — matches the GPU kernel
	let mut jerk_count = 0usize; // steps with a previous pwm to diff against
	let mut sum_mono = 0.0f64;
	let mut total_steps = 0usize;
	let mut n_stable = 0usize;

	for ep in 0..num_eps {
		let q = [q0[ep * 4], q0[ep * 4 + 1], q0[ep * 4 + 2], q0[ep * 4 + 3]];
		let om = [omega0[ep * 3], omega0[ep * 3 + 1], omega0[ep * 3 + 2]];
		c.reset(yaw_from_quat_rs(q)); // yaw-anchor: seed heading from the episode's true yaw
		sim.reset(Some(q), Some(om));
		if dist_enabled {
			let eps_seed = disturbance_episode_seed(dist_seed, ep as u64);
			sim.set_disturbance(
				dist_tau_bias, dist_gust_sigma, dist_gust_tau_c, dist_motor_asym,
				dist_gyro_sigma, dist_gyro_bias_walk, dist_accel_sigma, eps_seed,
			);
		}

		let mut ep_sum_err = 0.0f64;
		let mut prev_pwm = vec![0.5f32; num_motors];
		let mut first_step = true;
		let mut ep_steps = 0usize;
		let mut diverged = false;
		for _t in 0..steps {
			if sim.is_unstable() {
				diverged = true;
				break;
			}
			let (gyro, accel) = sim.read_imu();
			let pwm = c.step(gyro, accel, target);

			// Motor jerk: mean over steps of sqrt(Σ_m (Δpwm_m)²) (the L2 norm of the
			// per-step motor-delta vector) — EXACTLY the GPU kernel's formula
			// (controller_rollout.metal: sum_jerk += sqrt(dj); jerk_count++). First
			// step has no prev to diff against.
			if !first_step {
				let mut step_jerk = 0.0f64;
				for m in 0..num_motors {
					let d = (pwm[m] - prev_pwm[m]) as f64;
					step_jerk += d * d;
				}
				sum_jerk += step_jerk.sqrt();
				jerk_count += 1;
			}
			prev_pwm.copy_from_slice(&pwm);
			first_step = false;

			if let Ok(mv) = monotonicity_violations_core(&c.get_last_output_cells(), levels_per_motor, num_motors) {
				sum_mono += mv as f64;
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
			ep_steps += 1;
		}
		total_steps += ep_steps;
		let mean_err = ep_sum_err / ep_steps.max(1) as f64;
		sum_err += mean_err;
		if !diverged && mean_err <= stable_thresh_rad {
			n_stable += 1;
		}
	}

	let n = num_eps.max(1) as f64;
	let s = total_steps.max(1) as f64;
	// Row order matches metal_controller.rs: [reward, err_rad, stable, jerk, mono,
	// steady, rise, settle_abs, settle_rel, itae, iae, ise]. Transient/display metrics 0.
	[
		sum_reward / n,
		sum_err / n,
		n_stable as f64 / n,
		sum_jerk / jerk_count.max(1) as f64, // mean over steps-with-prev (GPU normalization)
		sum_mono / s,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
	// Overactuated Phase 1 — None = legacy quad sim. Same contract as
	// score_controllers_metal: rows [px,py,pz, ax,ay,az, spin, k_thrust,
	// k_drag] (pass the PERTURBED table for tilt/pos error); rotor_asym =
	// per-rotor thrust multipliers (N-rotor D3 twin).
	geometry = None,
	rotor_asym = None,
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
	geometry: Option<Vec<[f32; 9]>>,
	rotor_asym: Option<Vec<f32>>,
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
	// Clone out of the (non-Send) PyRefs into owned WnnControllers so rayon can roll
	// them out across threads. WnnController: Clone deep-copies cells+connectivity;
	// each clone gets its own mutable eval state (reset per episode anyway).
	let mut owned: Vec<WnnController> = controllers.iter().map(|c| (**c).clone()).collect();
	drop(controllers); // release the Python borrows before the GIL-free section
	let rows: Vec<Vec<f64>> = py.allow_threads(|| {
		owned
			.par_iter_mut()
			.map(|c| {
				rollout_one(
					c, &q0, &omega0, num_episodes, steps, dt, arm_length, k_thrust, k_drag,
					inertia, gravity, target, dist_enabled, dist_tau_bias, dist_gust_sigma,
					dist_gust_tau_c, dist_motor_asym, dist_gyro_sigma, dist_gyro_bias_walk,
					dist_accel_sigma, dist_seed, levels, num_motors,
					geometry.as_deref(), rotor_asym.as_deref(),
				)
				.to_vec()
			})
			.collect()
	});
	Ok(rows)
}
