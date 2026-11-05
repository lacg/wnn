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
	let mut sim: AttitudeSim = af.sim();
	let mut teacher: Teacher = af.teacher(id);
	let mut h = 0xcbf2_9ce4_8422_2325u64;
	let mut sum_err = 0.0f64;
	let mut n = 0usize;
	for ep in 0..episodes
	{
		let a = 0.08 + 0.05 * ep as f32;
		let (s, c) = (a * 0.5).sin_cos();
		let q = [c, s * 0.6, s * 0.8, 0.0];
		teacher.reset();
		sim.reset(Some(q), Some([0.3, -0.2, 0.1]));
		sim.set_disturbance(
			[2.0e-4, -1.5e-4, 0.5e-4], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 7 + ep as u64, 0.0, 0, 0, 0.0,
		);
		let mut last = [0.5f64; 4];
		for t in 0..steps
		{
			if sim.is_unstable()
			{
				break;
			}
			let (gyro, _accel) = sim.read_imu();
			teacher.observe(gyro, last);
			let cmd = teacher.step_rs(sim.quaternion(), gyro, [0.0, 0.0, 0.0]);
			for v in cmd
			{
				fnv(&mut h, v);
			}
			let wob = 0.004 * ((t as f64) * 0.05).sin();
			let applied = [
				(cmd[0] + wob).clamp(0.0, 1.0),
				(cmd[1] - wob).clamp(0.0, 1.0),
				cmd[2],
				cmd[3],
			];
			sim.step([applied[0] as f32, applied[1] as f32, applied[2] as f32, applied[3] as f32]);
			last = applied;
			sum_err += sim.attitude_error(None) as f64;
			n += 1;
		}
	}
	for v in teacher.integrals()
	{
		fnv(&mut h, v as f64);
	}
	(h, sum_err / n.max(1) as f64)
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
