// Scoring the FULL-STATE (position-cascade) teacher — scope C stage 2 chunk B.
//
// WHAT THIS ANSWERS: can the classical teacher hover to a POINT, and how well?
// That is chunk B's pre-registered bar (docs/scope_c_stage2_chunk_b_teacher.md),
// and it must be measured BEFORE the WNN is asked to imitate the teacher —
// DAgger cannot teach position control from an expert that has none.
//
// It reports Euclidean position error in METRES because that is Molchanov et
// al. 2019's headline metric (0.11 / 0.19 / 0.21 / 0.24 m across their
// configurations), and being able to emit that number at all is the reason
// scope C exists. The attitude triple rides along so a teacher that buys
// position accuracy by thrashing its attitude cannot hide.
//
// DELIBERATELY A SIBLING of score_classical_baseline, not a refactor of it: that
// scorer is the published attitude comparator and a live chain imports this
// wheel. Same per-episode reset, same disturbance derivation, same 80%-tail
// steady window — the ONLY additions are the translation plant and the two
// outer loops.

use pyo3::prelude::*;

use crate::altitude_pd::AltitudePd;
use crate::controller::disturbance_episode_seed;
use crate::dagger_train::AirframeRs;
use crate::position_loop::PositionLoop;

/// Held-out position-hold score for ONE classical teacher flying the full
/// cascade. Returns (mean position error m, final position error m,
/// stable_rate, attitude err deg, attitude steady deg).
///
/// `init_p` is 3 floats per episode (x, y, z offsets from the target in metres):
/// the episode STARTS off-target and the teacher must fly back, which is what
/// makes this a position-hold measurement rather than a hover check.
///
/// `use_estimator` follows the 13/08/2026 rule — comparison rows read a Mahony
/// estimate of the same noisy IMU, never the true quaternion. Position and
/// velocity are handed over directly: this is the TEACHER's bar, and a
/// position estimator is a separate question (no GPS/vision model exists here),
/// so that assumption must be DISCLOSED rather than buried.
#[pyfunction]
#[pyo3(signature = (teacher_id, init_qs, init_omegas, init_p, mass, steps,
	stable_deg = 5.0,
	pos_omega_n = 1.0, pos_zeta = 1.0, pos_max_tilt_deg = 30.0,
	alt_omega_n = 2.0, alt_zeta = 1.0, alt_max_delta = 0.25,
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
pub fn score_position_teacher(
	teacher_id: u8,
	init_qs: Vec<f32>,
	init_omegas: Vec<f32>,
	init_p: Vec<f32>,
	mass: f32,
	steps: usize,
	stable_deg: f64,
	pos_omega_n: f64,
	pos_zeta: f64,
	pos_max_tilt_deg: f64,
	alt_omega_n: f64,
	alt_zeta: f64,
	alt_max_delta: f64,
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
	af_arm_length: f32,
	af_k_thrust: f32,
	af_k_drag: f32,
	af_inertia: [f32; 3],
	af_gravity: f32,
	af_dt: f32,
	af_pid_att: [f64; 12],
	af_pid_rate: [f64; 12],
	af_pid_out_limit_n: f64,
	af_pid_hover_n: f64,
	af_pid_attitude_hz: f64,
	af_pid_lpf_hz: f64,
	use_estimator: bool,
	est_kp: f64,
	est_ki: f64,
) -> PyResult<(f64, f64, f64, f64, f64)>
{
	let err = |m: String| pyo3::exceptions::PyValueError::new_err(m);
	if init_qs.len() % 4 != 0 || init_omegas.len() % 3 != 0 || init_p.len() % 3 != 0
	{
		return Err(err(
			"score_position_teacher: init_qs (4/ep), init_omegas (3/ep) and \
		                init_p (3/ep) must each divide evenly"
				.into(),
		));
	}
	let num_episodes = init_qs.len() / 4;
	if init_omegas.len() / 3 != num_episodes || init_p.len() / 3 != num_episodes
	{
		return Err(err(format!(
			"score_position_teacher: episode counts differ — q {}, omega {}, p {}",
			num_episodes,
			init_omegas.len() / 3,
			init_p.len() / 3
		)));
	}
	if !(mass > 0.0) || !mass.is_finite()
	{
		return Err(err(format!(
			"score_position_teacher: mass must be positive, got {mass}"
		)));
	}

	let pos = PositionLoop::from_plant(
		af_gravity as f64,
		pos_omega_n,
		pos_zeta,
		pos_max_tilt_deg.to_radians(),
	)
	.map_err(err)?;
	let pd = AltitudePd::from_plant(
		mass as f64,
		af_gravity as f64,
		af_k_thrust as f64,
		alt_omega_n,
		alt_zeta,
		alt_max_delta,
	)
	.map_err(err)?;

	let af = AirframeRs {
		dt: af_dt,
		arm_length: af_arm_length,
		k_thrust: af_k_thrust,
		k_drag: af_k_drag,
		inertia: af_inertia,
		gravity: af_gravity,
		pid_fw: crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(
			af_pid_att,
			af_pid_rate,
			af_pid_out_limit_n,
			af_pid_hover_n,
			af_k_thrust as f64,
			(1.0 / af_dt.max(1e-9)).round() as u32,
			af_pid_attitude_hz,
			af_pid_lpf_hz,
		),
	};
	let mut sim = af.sim();
	// ANCHOR AT TRUE HOVER, not the attitude teachers' legacy 0.5 neutral. With
	// gravity simulated, a 0.5-anchored teacher is short by (hover − 0.5) of
	// collective and the integral-free altitude PD can only supply it by sitting
	// permanently below target — a 1.372 m droop on cf21. Teacher-side twin of
	// stage 1's collective_anchor.
	let hover_pwm = (mass as f64 * af_gravity as f64 / (4.0 * af_k_thrust as f64)).sqrt();
	let mut teacher = crate::optimal::Teacher::from_id_with_hover(
		teacher_id,
		af_dt,
		af_arm_length,
		af_k_thrust,
		af_k_drag,
		af_inertia,
		af_gravity,
		hover_pwm,
	);
	let mut est = if use_estimator
	{
		Some(crate::estimator::MahonyFilter::new(
			af_dt as f64,
			est_kp,
			est_ki,
		))
	}
	else
	{
		None
	};

	let stable_thresh_rad = stable_deg.to_radians();
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	let (mut sum_pos_err, mut sum_final_err) = (0.0f64, 0.0f64);
	let (mut sum_att_err, mut sum_steady) = (0.0f64, 0.0f64);
	let (mut n_stable, mut steady_eps) = (0usize, 0usize);

	for ep in 0..num_episodes
	{
		let init_q = [
			init_qs[ep * 4],
			init_qs[ep * 4 + 1],
			init_qs[ep * 4 + 2],
			init_qs[ep * 4 + 3],
		];
		let init_om = [
			init_omegas[ep * 3],
			init_omegas[ep * 3 + 1],
			init_omegas[ep * 3 + 2],
		];
		let (px, py, pz) = (init_p[ep * 3], init_p[ep * 3 + 1], init_p[ep * 3 + 2]);
		teacher.reset();
		sim.reset(Some(init_q), Some(init_om));
		// Translation AFTER reset (reset zeroes the states) — cpu_score's order.
		sim.set_translation_core(mass).map_err(err)?;
		sim.set_vertical_state(pz, 0.0);
		sim.set_horizontal_state(px, py, 0.0, 0.0);
		if let Some(e) = est.as_mut()
		{
			e.reset(Some(init_q));
		}
		if dist_enabled
		{
			let ep_seed = disturbance_episode_seed(dist_seed, ep as u64);
			sim.set_disturbance(
				dist_tau_bias,
				dist_gust_sigma,
				dist_gust_tau_c,
				dist_motor_asym,
				dist_gyro_sigma,
				dist_gyro_bias_walk,
				dist_accel_sigma,
				ep_seed,
				dist_dropout_prob,
				dist_dropout_len_steps,
				dist_obs_delay_steps,
				dist_torque_scale_jitter,
			);
		}

		let (mut ep_pos, mut ep_att) = (0.0f64, 0.0f64);
		let (mut tail_sum, mut tail_cnt) = (0.0f64, 0usize);
		let mut done = 0usize;
		let mut last_applied = [0.5f64; 4];
		let mut diverged = false;
		for t in 0..steps
		{
			if sim.is_unstable()
			{
				diverged = true;
				break;
			}
			let (gyro, accel) = sim.read_imu();
			let q = match est.as_mut()
			{
				Some(e) => e.update(gyro, accel),
				None => sim.quaternion(),
			};
			// Target is the ORIGIN; the episode starts displaced from it, so the
			// error is just −position.
			let (x, y) = sim.position_xy();
			let (vx, vy) = sim.velocity_xy();
			teacher.observe(gyro, last_applied);
			let cmd = teacher.step_full_state(
				q,
				gyro,
				0.0,
				&pos,
				&pd,
				-(x as f64),
				vx as f64,
				-(y as f64),
				vy as f64,
				-(sim.altitude_rs() as f64),
				sim.vertical_velocity_rs() as f64,
			);
			let pwm = [cmd[0] as f32, cmd[1] as f32, cmd[2] as f32, cmd[3] as f32];
			sim.step(pwm);
			last_applied = cmd;
			let (x, y) = sim.position_xy();
			let pe = ((x * x + y * y + sim.altitude_rs() * sim.altitude_rs()) as f64).sqrt();
			let ae = sim.attitude_error(None) as f64;
			ep_pos += pe;
			ep_att += ae;
			if t >= tail_start
			{
				tail_sum += ae;
				tail_cnt += 1;
			}
			done += 1;
		}
		let d = done.max(1) as f64;
		sum_pos_err += ep_pos / d;
		let (fx, fy) = sim.position_xy();
		sum_final_err += ((fx * fx + fy * fy + sim.altitude_rs() * sim.altitude_rs()) as f64).sqrt();
		let mean_att = ep_att / d;
		sum_att_err += mean_att;
		if !diverged && mean_att <= stable_thresh_rad
		{
			n_stable += 1;
		}
		if tail_cnt > 0
		{
			sum_steady += tail_sum / tail_cnt as f64;
			steady_eps += 1;
		}
	}

	let n = num_episodes.max(1) as f64;
	Ok((
		sum_pos_err / n,
		sum_final_err / n,
		n_stable as f64 / n,
		(sum_att_err / n).to_degrees(),
		if steady_eps > 0
		{
			(sum_steady / steady_eps as f64).to_degrees()
		}
		else
		{
			f64::NAN
		},
	))
}
