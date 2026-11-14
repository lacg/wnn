//! CTRL-46 item 3 (ABI 46, `--obs-att-bias-pred`): an ONBOARD predictor of the
//! Mahony estimator's specific-force bias, as two student input features.
//!
//! THE BIAS (docs/ctrl_steady_gap_teacher.md §2). Under the ABI-32 specific-force IMU
//! the accelerometer measures f = Rᵀ(a − g_w): its direction is the true body "up"
//! u = Rᵀẑ only when the vehicle does not accelerate. To first order
//!     f/|f| = u + δ,      δ = Rᵀ(a_h / g)   (the horizontal acceleration, body frame)
//! and Mahony pulls its estimate v toward f/|f|. Linearised (small angles, the gyro
//! integration common to v and u cancels), the estimate error ε = v − u obeys
//!     ε̇ = kp·(δ − ε) + J,      J̇ = ki·(δ − ε)
//! — a 2nd-order low-pass of the acceleration with the filter's OWN gains. Rotor drag
//! does not enter: f = Rᵀ(a − g_w) holds whatever forces make up a. At constant
//! velocity δ = 0 and the bias decays; a translating hold (the tilt reference the
//! position loop asks for IS an acceleration demand) keeps it alive — the measured
//! r = +0.68 with the tilt reference and 0.52° of the teacher's 0.68° steady.
//!
//! THE PREDICTOR integrates exactly that model, driven by the ONBOARD velocity
//! estimate: δ·dt = Rᵀ(Δv/g) (one difference per physical step — no 1/dt blow-up),
//! rotated into the body by the ESTIMATOR's own quaternion. Emits the predicted bias
//! of the student's two attitude readings, in the compute_features conventions
//! (roll_est = atan2(up_y, up_z), pitch_est = atan2(−up_x, ·)):
//!     abp_r = ε_y,   abp_p = −ε_x        (rad, predicted estimate − truth)
//!
//! LEAK ARGUMENT. Inputs, all available on a real Crazyflie:
//!   * the horizontal velocity estimate — in sim, the SAME `horiz_obs[2..4]` signal
//!     the student already reads as `--obs-vel-xy` (onboard: the Kalman estimator's
//!     velocity from flow / Lighthouse / LPS). The predictor is refused without
//!     `obs_vel_xy`, so it can never see a signal the student is not already given;
//!   * the Mahony estimate quaternion (the student's own `obs_att_estimator` filter);
//!   * the filter's gains kp, ki and the step dt (firmware constants), g (nominal).
//! It never reads the true quaternion, the true acceleration or any plant state:
//! `step` takes exactly (v_est, q_est). The truth is used only by the TEST below to
//! measure how well it predicts.
//!
//! TRAIN = DEPLOY: the rollout records the two values per step (TrajectoryRs.att_bias →
//! ReplayObs.att_bias) and a replay re-applies them (the cascade / att_up pattern).

/// Nominal gravity the predictor scales by (the thermometer ladder is a quantile fit,
/// so only the sign/shape matters; 9.81 keeps the feature in radians).
pub(crate) const ABP_GRAVITY: f32 = 9.81;

/// Features the block appends ([abp_r, abp_p]).
pub(crate) const ABP_FEATURES: usize = 2;

#[derive(Clone, Debug)]
pub(crate) struct AttBiasPredictor
{
	kp: f32,
	ki: f32,
	dt: f32,
	/// ε (body x, y) and the integral J of the linearised Mahony error dynamics.
	eps: [f32; 2],
	j: [f32; 2],
	v_prev: [f32; 2],
	have_prev: bool,
	last: [f32; 2],
	replay: Option<[f32; 2]>,
}

impl AttBiasPredictor
{
	pub(crate) fn new(kp: f32, ki: f32, dt: f32) -> Self
	{
		AttBiasPredictor { kp, ki, dt, eps: [0.0; 2], j: [0.0; 2], v_prev: [0.0; 2], have_prev: false, last: [0.0; 2], replay: None }
	}

	/// Episode start (the estimator is warm-started at the episode's attitude, so its
	/// bias starts at 0 too).
	pub(crate) fn reset(&mut self)
	{
		self.eps = [0.0; 2];
		self.j = [0.0; 2];
		self.v_prev = [0.0; 2];
		self.have_prev = false;
		self.last = [0.0; 2];
		self.replay = None;
	}

	/// Replay: the next `step` returns this recorded value and advances nothing.
	pub(crate) fn set_replay(&mut self, v: [f32; 2])
	{
		self.replay = Some(v);
	}

	/// A replay value is pending (the next `step` ignores its inputs).
	pub(crate) fn has_replay(&self) -> bool
	{
		self.replay.is_some()
	}

	/// What the last `step` emitted (what a rollout records).
	pub(crate) fn last(&self) -> [f32; 2]
	{
		self.last
	}

	/// One physical step on (onboard world-frame horizontal velocity, estimate
	/// quaternion w,x,y,z) → [abp_r, abp_p]. f32 throughout: the Metal twin
	/// (controller_rollout.metal `abp_step`) evaluates the same expressions.
	pub(crate) fn step(&mut self, v_world: [f32; 2], q_est: [f32; 4]) -> [f32; 2]
	{
		if let Some(r) = self.replay.take()
		{
			self.last = r;
			return r;
		}
		// δ·dt = Rᵀ(Δv / g), horizontal world vector (vz component 0).
		let (ux, uy) = if self.have_prev
		{
			((v_world[0] - self.v_prev[0]) / ABP_GRAVITY, (v_world[1] - self.v_prev[1]) / ABP_GRAVITY)
		}
		else
		{
			(0.0, 0.0)
		};
		self.v_prev = v_world;
		self.have_prev = true;
		let d = rot_t_xy(q_est, ux, uy);
		for a in 0..2
		{
			let innov = d[a] - self.dt * self.eps[a]; // (δ − ε)·dt
			self.eps[a] += self.kp * innov + self.dt * self.j[a];
			self.j[a] += self.ki * innov;
		}
		self.last = [self.eps[1], -self.eps[0]];
		self.last
	}
}

/// Body x, y of Rᵀ·(wx, wy, 0) for q = (w, x, y, z) (R = body → world), f32.
#[inline]
pub(crate) fn rot_t_xy(q: [f32; 4], wx: f32, wy: f32) -> [f32; 2]
{
	let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
	let r00 = 1.0 - 2.0 * (y * y + z * z);
	let r01 = 2.0 * (x * y - w * z);
	let r10 = 2.0 * (x * y + w * z);
	let r11 = 1.0 - 2.0 * (x * x + z * z);
	[r00 * wx + r10 * wy, r01 * wx + r11 * wy]
}

#[cfg(test)]
mod tests
{
	use super::*;
	use crate::att_estimator::up_from_quat;
	use crate::dagger_train::AirframeRs;
	use crate::episode_regime::OuterLoops;
	use crate::estimator::MahonyFilter;

	fn rp(up: [f32; 3]) -> [f32; 2]
	{
		[up[1].atan2(up[2]), (-up[0]).atan2((up[1] * up[1] + up[2] * up[2]).sqrt())]
	}

	/// Rᵀ is the inverse rotation: rotating the world z axis gives up_from_quat, and a
	/// pure yaw ψ maps world x to body (cos ψ, −sin ψ).
	#[test]
	fn rotation_conventions()
	{
		let psi = 0.4f32;
		let q = [(psi * 0.5).cos(), 0.0, 0.0, (psi * 0.5).sin()];
		let b = rot_t_xy(q, 1.0, 0.0);
		assert!((b[0] - psi.cos()).abs() < 1e-6 && (b[1] + psi.sin()).abs() < 1e-6, "{b:?}");
		let q = [0.97f32, 0.1, -0.2, 0.08];
		let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
		let q = q.map(|v| v / n);
		let up = up_from_quat(q);
		let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
		let r20 = 2.0 * (x * z - w * y);
		assert!((up[0] - r20).abs() < 1e-6);
	}

	/// Replay is one-shot and advances no state; reset forgets everything.
	#[test]
	fn replay_and_reset()
	{
		let mut p = AttBiasPredictor::new(2.0, 0.1, 0.001);
		let q = [1.0, 0.0, 0.0, 0.0];
		p.step([0.0, 0.0], q);
		let a = p.step([0.01, 0.0], q);
		assert!(a[1] != 0.0, "an x acceleration must move the pitch bias");
		p.set_replay([0.5, -0.5]);
		assert_eq!(p.step([9.0, 9.0], q), [0.5, -0.5]);
		assert_eq!(p.last(), [0.5, -0.5]);
		p.reset();
		assert_eq!(p.step([3.0, 3.0], q), [0.0, 0.0], "first step after reset has no Δv");
	}

	/// Steady acceleration a along world x at level attitude: Mahony converges to the
	/// accelerometer direction — the predictor's fixed point is ε_x = a/g ⇒
	/// abp_p = −a/g (proportional-only filter, so it settles in a few 1/kp).
	#[test]
	fn steady_acceleration_fixed_point()
	{
		let (dt, a) = (0.001f32, 0.5f32);
		let mut p = AttBiasPredictor::new(2.0, 0.0, dt);
		let q = [1.0, 0.0, 0.0, 0.0];
		let (mut out, mut v) = ([0.0; 2], 0.0f64);
		for _ in 0..5000
		{
			out = p.step([v as f32, 0.0], q);
			v += (a * dt) as f64;
		}
		let want = -a / ABP_GRAVITY;
		assert!(((out[1] - want) / want).abs() < 0.01, "{out:?} vs {want}");
		assert!(out[0].abs() < 1e-6);
	}

	/// THE QUALITY PROBE (truth used ONLY here, to score the prediction): the firmware
	/// PID cascade flies the translating cf21 regime (xy 0.5 m, specific-force IMU,
	/// rotor drag 0.78, L4C-like noise); a Mahony filter runs on the observed IMU
	/// exactly like the student's, and the predictor sees only (velocity, estimate q).
	/// The predicted roll/pitch bias must explain most of the measured bias variance.
	#[test]
	fn predicts_the_mahony_specific_force_bias()
	{
		let mut cfg = crate::abi38_tests::l4c_cfg(0.0, 0);
		cfg.imu_specific_force = true;
		cfg.rotor_drag = 0.78;
		let af = AirframeRs::from_cfg(&cfg);
		let loops = OuterLoops::for_cfg(&cfg);
		let (mut sxx, mut sxy, mut syy) = ([0f64; 2], [0f64; 2], [0f64; 2]);
		let mut n = 0usize;
		for ep in 0..6u64
		{
			let mut teacher = af.teacher(0);
			let mut sim = af.sim();
			let q0 = [0.998f32, 0.03 - 0.01 * ep as f32, -0.04, 0.05];
			sim.reset(Some(q0), Some([0.1, -0.1, 0.05]));
			sim.set_translation_core(cfg.af_mass).unwrap();
			sim.set_vertical_state(0.1, 0.0);
			let s = if ep % 2 == 0 { 0.5 } else { -0.5 };
			sim.set_horizontal_state(s, -0.6 * s + 0.1 * ep as f32 - 0.2, 0.0, 0.0);
			crate::dagger_train::arm_cfg_disturbance(&mut sim, &cfg, 4600 + ep);
			let mut mahony = MahonyFilter::new(cfg.dt, 2.0, 0.1);
			mahony.reset(Some(q0));
			let mut pred = AttBiasPredictor::new(2.0, 0.1, cfg.dt as f32);
			for t in 0..2000
			{
				if sim.is_unstable()
				{
					break;
				}
				let (g, acc) = sim.read_imu();
				let q_est = mahony.update(g, acc);
				let p = pred.step(sim.velocity_xy_rs(), q_est);
				let q_true = sim.quaternion();
				if t >= 300
				{
					let (re, rt) = (rp(up_from_quat(q_est)), rp(up_from_quat(q_true)));
					for a in 0..2
					{
						let bias = (re[a] - rt[a]) as f64;
						let x = p[a] as f64;
						sxx[a] += x * x;
						sxy[a] += x * bias;
						syy[a] += bias * bias;
					}
					n += 1;
				}
				let cmd = loops.expert(&mut teacher, &sim, q_true, g, [0.0; 3], (cfg.target_altitude - sim.altitude_rs(), sim.vertical_velocity_rs()));
				sim.step(cmd.map(|v| v as f32));
			}
		}
		assert!(n > 5000, "probe flew too little ({n} samples)");
		for a in 0..2
		{
			// Uncentred R² of bias ≈ prediction (no intercept: the model has none), and
			// the slope must be ~1 (the predictor is in the bias's own units).
			let r2 = sxy[a] * sxy[a] / (sxx[a] * syy[a]);
			let slope = sxy[a] / sxx[a];
			let rms_deg = (syy[a] / n as f64).sqrt().to_degrees();
			eprintln!("axis {a}: R2 {r2:.3} slope {slope:.3} bias rms {rms_deg:.3} deg");
			assert!(rms_deg > 0.05, "axis {a}: regime too quiet to test ({rms_deg} deg)");
			assert!(r2 > 0.8, "axis {a}: predictor explains only R2 {r2:.3} of the bias");
			assert!((0.7..1.3).contains(&slope), "axis {a}: predictor scale off (slope {slope:.3})");
		}
	}
}
