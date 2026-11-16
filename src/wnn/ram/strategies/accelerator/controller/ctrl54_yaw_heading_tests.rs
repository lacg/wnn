//! Lever 1 (ABI 54, Luiz 16/11/2026): the anchored yaw heading from the student's own Mahony
//! quaternion (`set_yaw_from_att_est`) instead of the body-rate sum gyro_z·dt.
//!
//! Pinned: refusals; off ⇒ the features of a never-touched twin bit for bit; on an exact
//! rigid-body trajectory (constant body rate, tilted start, perfect IMU) the swapped heading
//! tracks the true Euler yaw while gyro_z·dt does not (ψ̇ ≠ r under tilt).

use crate::controller::{quat_to_euler_f64, WnnController};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt};
use ram_core::neuron_memory::BINARY;

fn student(est: bool) -> WnnController
{
	let mut c = yaw_student(0, 10, 7, YawOpt::Off, BINARY);
	if est
	{
		c.enable_att_estimator(2.0, 0.1).unwrap();
	}
	c
}

/// q ⊗ exp(ω dt / 2), f64 — the exact body-rate propagation.
fn propagate(q: [f64; 4], w: [f64; 3], dt: f64) -> [f64; 4]
{
	let n = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
	let (s, c) = ((n * dt * 0.5).sin(), (n * dt * 0.5).cos());
	let d = [c, s * w[0] / n, s * w[1] / n, s * w[2] / n];
	let r = [
		q[0] * d[0] - q[1] * d[1] - q[2] * d[2] - q[3] * d[3],
		q[0] * d[1] + q[1] * d[0] + q[2] * d[3] - q[3] * d[2],
		q[0] * d[2] - q[1] * d[3] + q[2] * d[0] + q[3] * d[1],
		q[0] * d[3] + q[1] * d[2] - q[2] * d[1] + q[3] * d[0],
	];
	let m = r.iter().map(|v| v * v).sum::<f64>().sqrt();
	r.map(|v| v / m)
}

/// Specific force (body) of a hovering IMU: R(q)ᵀ·(0, 0, g).
fn accel(q: [f64; 4], g: f64) -> [f32; 3]
{
	let [w, x, y, z] = q;
	[(2.0 * (x * z - w * y) * g) as f32, (2.0 * (y * z + w * x) * g) as f32, ((1.0 - 2.0 * (x * x + y * y)) * g) as f32]
}

/// Fly `steps` of the exact trajectory; returns (max |heading − true yaw|, final features).
fn fly(c: &mut WnnController, steps: usize) -> (f64, Vec<f32>)
{
	let dt = crate::abi38_pins::DT as f64;
	let (h0, h1) = ((0.06f64 * 0.5).cos(), (0.06f64 * 0.5).sin()); // 0.06 rad roll start
	let mut q = [h0, h1, 0.0, 0.0];
	let w = [0.4, -0.3, 0.25]; // rad/s body rate — keeps rolling/pitching while yawing
	let q32 = q.map(|v| v as f32);
	c.reset(quat_to_euler_f64(q32).2 as f32);
	c.seed_att_estimator(q32);
	let (mut worst, mut f) = (0.0f64, Vec::new());
	for _ in 0..steps
	{
		f = c.compute_features([w[0] as f32, w[1] as f32, w[2] as f32], accel(q, 9.81), [0.0; 3]);
		let yaw_true = quat_to_euler_f64(q.map(|v| v as f32)).2;
		worst = worst.max((c.yaw_heading_est() as f64 - yaw_true).abs());
		q = propagate(q, w, dt);
	}
	(worst, f)
}

#[test]
fn yaw_from_att_est_refusals_and_default()
{
	let mut c = student(false);
	assert!(!c.yaw_from_att_est(), "default off");
	assert!(c.set_yaw_from_att_est(true).is_err(), "needs obs_att_estimator");
	c.set_yaw_from_att_est(false).unwrap();
	let mut e = student(true);
	e.set_yaw_from_att_est(true).unwrap();
	assert!(e.yaw_from_att_est());
}

#[test]
fn yaw_from_att_est_off_is_bit_identical()
{
	let mut a = student(true);
	let mut b = student(true);
	b.set_yaw_from_att_est(false).unwrap();
	let (wa, fa) = fly(&mut a, 1500);
	let (wb, fb) = fly(&mut b, 1500);
	assert_eq!(wa.to_bits(), wb.to_bits());
	assert_eq!(fa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), fb.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn yaw_from_att_est_tracks_euler_yaw_under_tilt()
{
	let (dead, _) = fly(&mut student(true), 1500);
	let mut s = student(true);
	s.set_yaw_from_att_est(true).unwrap();
	let (swap, _) = fly(&mut s, 1500);
	assert!(dead > 5e-3, "gyro_z·dt must drift from the Euler yaw under tilt: {dead}");
	assert!(swap < 1e-3 && swap * 10.0 < dead, "Mahony heading {swap} vs dead-reckon {dead}");
}
