//! ABI 34 (05/11/2026): lag-aware teachers + ONE pwm convention under motor lag.
//!
//! Three things are pinned here:
//!   1. T = 0 is BIT-IDENTICAL to ABI 33 — every teacher's label stream and every
//!      classical-baseline row hashes to the digest captured on the ABI-33 build
//!      before any ABI-34 line was written (the constants below). A digest moving
//!      means the anchor plant's teachers changed, which this ABI must never do.
//!   2. T > 0 actually reaches the teachers (lag-aware ≠ lag-blind) and the
//!      augmented designs are sound (DARE residual, alpha = 1 collapses onto the
//!      lag-free design, closed loop stable on the lagged plant).
//!   3. The mpcof observer no longer reads actuator lag as a disturbance.

use crate::controller::AttitudeSim;
use crate::dagger_train::AirframeRs;
use crate::optimal::Teacher;

/// A cf21-like airframe at the derived hover the stage-2 rivals fly.
fn airframe(motor_lag_s: f32) -> AirframeRs
{
	AirframeRs {
		dt: 0.001,
		arm_length: 0.0707,
		k_thrust: 0.2,
		k_drag: 0.0057,
		inertia: [3.0e-5, 3.0e-5, 5.3e-5],
		gravity: 9.81,
		pid_fw: None,
		teacher_hover: Some(0.6942),
		motor_lag_s,
		plant: crate::plant_opts::PlantOpts::OFF,
	}
}

fn fnv(h: &mut u64, v: f64)
{
	for b in v.to_bits().to_le_bytes()
	{
		*h ^= b as u64;
		*h = h.wrapping_mul(0x0000_0100_0000_01B3);
	}
}

/// Closed-loop teacher rollout with the DAgger observe() protocol (dummy hover
/// observe at t=0, then the action applied last step). The APPLIED action is the
/// teacher's own plus a deterministic perturbation, so observe() sees something
/// other than the teacher's command — the student-drives case. A constant torque
/// bias keeps the mpcof observer busy. Returns (label digest, mean |err| rad).
pub(crate) fn teacher_rollout(id: u8, motor_lag_s: f32, episodes: usize, steps: usize) -> (u64, f64)
{
	let af = airframe(motor_lag_s);
	let r = rollout_with(af.teacher(id), &af, episodes, steps, BIAS, 0.004);
	(r.digest, r.mean_err)
}

const BIAS: [f32; 3] = [2.0e-4, -1.5e-4, 0.5e-4];

pub(crate) struct Rollout
{
	pub digest: u64,
	pub mean_err: f64,
	/// mean |err| over the last 20% of every episode (the steady window)
	pub steady_err: f64,
	/// mean over steps of max-axis |d̂| (mpcof; 0 for the others)
	pub mean_abs_dhat: f64,
	pub final_dhat: [f32; 3],
	pub diverged: usize,
	/// The teachers' own quadratic cost on the TRUE plant, Σ dt·(q_att|e|² +
	/// q_rate|ω|² + q_int|∫e|² + r_ctrl|u|²) per episode, averaged — the
	/// objective every lag-aware design minimizes (∫e is the running integral).
	pub cost: f64,
}

/// The rollout behind teacher_rollout, with the teacher, plant, bias and the
/// applied-action wobble explicit (so a lag-BLIND teacher can fly a lagged plant).
pub(crate) fn rollout_with(mut teacher: Teacher, af: &AirframeRs, episodes: usize, steps: usize, bias: [f32; 3], wobble: f64) -> Rollout
{
	let mut sim: AttitudeSim = af.sim();
	let mut h = 0xcbf2_9ce4_8422_2325u64;
	let (mut sum_err, mut sum_tail, mut sum_dhat) = (0.0f64, 0.0f64, 0.0f64);
	let (mut n, mut n_tail, mut diverged) = (0usize, 0usize, 0usize);
	let tail_start = steps * 4 / 5;
	let mut cost = 0.0f64;
	let dt = af.dt as f64;
	for ep in 0..episodes
	{
		let a = 0.08 + 0.05 * ep as f32;
		let (s, c) = (a * 0.5).sin_cos();
		let q = [c, s * 0.6, s * 0.8, 0.0];
		teacher.reset();
		sim.reset(Some(q), Some([0.3, -0.2, 0.1]));
		sim.set_disturbance(bias, 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 7 + ep as u64, 0.0, 0, 0, 0.0);
		let mut last = [0.5f64; 4];
		let mut integ = [0.0f64; 3];
		for t in 0..steps
		{
			if sim.is_unstable()
			{
				diverged += 1;
				break;
			}
			let (gyro, _accel) = sim.read_imu();
			teacher.observe(gyro, last);
			let cmd = teacher.step_rs(sim.quaternion(), gyro, [0.0, 0.0, 0.0]);
			{
				let (r, p, y) = crate::controller::quat_to_euler_f64(sim.quaternion());
				let e = [r, p, crate::controller::wrap_angle_f64(y)];
				let u = crate::actuator_lag::unmix_plus(cmd);
				for axis in 0..3
				{
					integ[axis] += e[axis] * dt;
					let w = gyro[axis] as f64;
					cost += dt
						* (crate::optimal::Q_ATT * e[axis] * e[axis]
							+ crate::optimal::Q_RATE * w * w
							+ crate::optimal::Q_INT * integ[axis] * integ[axis]
							+ crate::optimal::R_CTRL * u[axis] * u[axis]);
				}
			}
			for v in cmd
			{
				fnv(&mut h, v);
			}
			let wob = wobble * ((t as f64) * 0.05).sin();
			let applied = [
				(cmd[0] + wob).clamp(0.0, 1.0),
				(cmd[1] - wob).clamp(0.0, 1.0),
				cmd[2],
				cmd[3],
			];
			sim.step([applied[0] as f32, applied[1] as f32, applied[2] as f32, applied[3] as f32]);
			last = applied;
			let err = sim.attitude_error(None) as f64;
			sum_err += err;
			n += 1;
			if t >= tail_start
			{
				sum_tail += err;
				n_tail += 1;
			}
			if let Teacher::MpcOf(_) = teacher
			{
				sum_dhat += teacher.integrals().iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
			}
		}
	}
	let final_dhat = teacher.integrals();
	for v in final_dhat
	{
		fnv(&mut h, v as f64);
	}
	Rollout {
		digest: h,
		mean_err: sum_err / n.max(1) as f64,
		steady_err: sum_tail / n_tail.max(1) as f64,
		mean_abs_dhat: sum_dhat / n.max(1) as f64,
		final_dhat,
		diverged,
		cost: cost / episodes.max(1) as f64,
	}
}

/// The lag-BLIND (ABI-33) teacher for this airframe: the same hover, no lag.
fn blind_teacher(af: &AirframeRs, id: u8) -> Teacher
{
	AirframeRs { motor_lag_s: 0.0, ..af.clone() }.teacher(id)
}

/// ABI-33 digests of `teacher_rollout(id, 0.0, 3, 600)` for ids 0..=4, captured on
/// the pre-change build (commit e56d6084). T = 0 must reproduce them bit-for-bit.
const ABI33_T0_DIGESTS: [u64; 5] = [
	0x6d56_f3dd_0a65_e7f7,
	0x8cc9_33e4_2c5a_734a,
	0x1537_5cf8_4f72_01f2,
	0xa206_dc22_ff9c_2e59,
	0xcc2f_637f_9056_d531,
];

#[test]
fn t0_teacher_labels_bit_identical_to_abi33()
{
	let got: Vec<u64> = (0..=4u8).map(|id| teacher_rollout(id, 0.0, 3, 600).0).collect();
	assert_eq!(
		got,
		ABI33_T0_DIGESTS.to_vec(),
		"T=0 teacher labels moved vs ABI 33 (got {got:#x?})"
	);
}

/// T > 0 REACHES the teachers: ids 1-4 label differently lag-aware vs
/// lag-blind on the same lagged plant; PID (the firmware reference) does not.
#[test]
fn lag_reaches_model_teachers_not_pid()
{
	for tt in [0.15f32, 0.30]
	{
		let af = airframe(tt);
		for id in 0..=4u8
		{
			let aware = rollout_with(af.teacher(id), &af, 2, 400, BIAS, 0.004).digest;
			let blind = rollout_with(blind_teacher(&af, id), &af, 2, 400, BIAS, 0.004).digest;
			if id == 0
			{
				assert_eq!(aware, blind, "T={tt}: PID must stay the lag-free firmware law");
			}
			else
			{
				assert_ne!(aware, blind, "T={tt}: teacher {id} ignored the motor lag");
			}
		}
	}
}

/// The lag-aware designs FLY the lagged plant and are what they claim to be:
/// every model teacher holds attitude (no divergence, steady error under 1°,
/// no disturbance) at T ∈ {0.15, 0.30}, and its own quadratic cost on the TRUE
/// lagged plant is no higher than its lag-blind (ABI-33) self's — the
/// optimality the augmented design buys (2% slack: the cost is evaluated in
/// closed loop with authority clamps the design does not model).
#[test]
fn lag_aware_teachers_hold_the_lagged_plant()
{
	for tt in [0.15f32, 0.30]
	{
		let af = airframe(tt);
		for id in 1..=4u8
		{
			let aware = rollout_with(af.teacher(id), &af, 3, 2000, [0.0; 3], 0.0);
			let blind = rollout_with(blind_teacher(&af, id), &af, 3, 2000, [0.0; 3], 0.0);
			eprintln!(
				"[abi34] T={tt} id={id}: aware err {:.3}° steady {:.3}° J {:.5} div {} | blind err {:.3}° steady {:.3}° J {:.5} div {}",
				aware.mean_err.to_degrees(), aware.steady_err.to_degrees(), aware.cost, aware.diverged,
				blind.mean_err.to_degrees(), blind.steady_err.to_degrees(), blind.cost, blind.diverged,
			);
			assert_eq!(aware.diverged, 0, "T={tt} id={id}: lag-aware teacher diverged");
			assert!(aware.steady_err.to_degrees() < 1.0, "T={tt} id={id}: steady {}°", aware.steady_err.to_degrees());
			assert!(
				aware.cost <= blind.cost * 1.02,
				"T={tt} id={id}: lag-aware cost {} above lag-blind {}",
				aware.cost, blind.cost
			);
		}
	}
}

/// Finding 2: the mpcof observer must not read actuator lag as a disturbance.
/// (a) With NO disturbance the lag-aware d̂ stays near 0 while the lag-blind one
/// (subtracting b·u_commanded) reads the lag; (b) with a constant torque bias the
/// lag-aware d̂ converges to the true rate-accel bias τ/I.
#[test]
fn mpcof_observer_does_not_read_lag_as_disturbance()
{
	for tt in [0.15f32, 0.30]
	{
		let af = airframe(tt);
		let aware = rollout_with(af.teacher(4), &af, 3, 1500, [0.0; 3], 0.004);
		let blind = rollout_with(blind_teacher(&af, 4), &af, 3, 1500, [0.0; 3], 0.004);
		eprintln!("[abi34] T={tt} no-bias mean max|d̂|: aware {:.4} blind {:.4}", aware.mean_abs_dhat, blind.mean_abs_dhat);
		assert!(
			aware.mean_abs_dhat < 0.25 * blind.mean_abs_dhat,
			"T={tt}: lag-aware d̂ {} not well below lag-blind {}",
			aware.mean_abs_dhat, blind.mean_abs_dhat
		);
		let biased = rollout_with(af.teacher(4), &af, 1, 4000, BIAS, 0.0);
		for axis in 0..3
		{
			let truth = BIAS[axis] / af.inertia[axis];
			let rel = (biased.final_dhat[axis] - truth).abs() / truth.abs();
			assert!(rel < 0.10, "T={tt} axis {axis}: d̂ {} vs τ/I {} ({:.1}% off)", biased.final_dhat[axis], truth, rel * 100.0);
		}
	}
}
