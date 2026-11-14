//! ABI 39 (CTRL-33, Luiz 06/11/2026): the cascade's OUTER-loop MPC for one body axis.
//! Its prediction model IS the closed inner loop: the firmware rate PID (sourced
//! cf21bl gains, never re-derived) + the first-order motor lag + the rigid body,
//! discretised at the OUTER sample time (action_repeat ticks). The decision variable
//! is the body-rate setpoint the rate loop will be handed, not a torque.
//!
//! WHY (the ABI-38 failure, logs/controller/ct38/rivals_*.json). The ABI-38 mpcof CTBR
//! label planned TORQUE on the motor-level model (as if it commanded the rotors) and
//! read the body rate its plan reached N_o steps ahead. The real rate loop
//! (closed-loop bandwidth ~76 rad/s at lag 0, much less behind the 50 ms rotor pole)
//! adds phase that plan never modelled, and under lag the motor lag was counted twice
//! (once in the plan, once again in the loop that tracks the rate). MPCOF>rate(h10):
//! 18.8% stable at lag 0.20 against PID>rate 100%.
//!
//! THE MODEL, per axis, at the plant tick dt (1 kHz), state
//!   s = [θ, ω, ua, I, f1, f2, pm, Fh]
//!   θ  attitude error (Euler angle − reference, rad — the firmware attitude stage's
//!      own error; small-angle θ' = ω about the BODY axis),
//!   ω  body rate (rad/s, the gyro),
//!   ua the LAGGED axis force (N) — the rotors' actual differential thrust,
//!   I, f1, f2, pm, Fh the rate PID's integrator (rad), its 30 Hz lpf2p delay line,
//!      its last measured rate and its held output force (N) — pid.c, verbatim.
//! One tick (the order RateStageFw::step → AttitudeSim::step runs it):
//!   if rate-update tick (every `decimation` ticks, dt_r = 1/attitude_hz):
//!     e = sp − ω;  I ← I + e·dt_r;  raw = −(ω − pm)/dt_r;
//!     d0 = raw − a1·f1 − a2·f2;  D = b0·d0 + b1·f1 + b2·f2;  (f2, f1) ← (f1, d0);
//!     pm ← ω;  Fh ← kp·e + ki·I + kd·D
//!   ua ← ua + α·(Fh − ua)          (Molchanov lag, α = motor_lag_alpha; α = 1 at T = 0)
//!   ω  ← ω + dt·(c·ua + d);  θ ← θ + dt·ω + ½dt²·(c·ua + d)
//! with c = L/Jx, L/Jy, 4·k_drag/Jz (AttitudeSim::body_torque through the firmware '+'
//! mixer: roll/pitch force r ⇒ t3 − t1 = r, yaw force y ⇒ t0 − t1 + t2 − t3 = 4y) and d
//! the input disturbance (rad/s²) the offset-free observer estimates. Linear except the
//! out/i-limit clamps, which the model omits (the shadow loop the state is read from
//! keeps them). The thrust is linear in the mixer's force by construction
//! (pwm = sqrt(t/kt), t = kt·pwm²); the lag acts on pwm, which is linearised here.
//!
//! The OUTER map (A, B) is the exact composition of `n_o` such ticks from the rate
//! loop's current phase, built column by column from `tick` itself, so the model and
//! the tick can never disagree. Offset-free: for a constant d the closed loop rests at
//! θ = ω = 0, sp = 0, Fh = ua = −d/c, I = −d/(c·ki) (the rate integrator carries the
//! trim — measured: it absorbs the ±20% motor asymmetry, 6.38° → 0.80°); the MPC
//! regulates x − x_ss(d̂) to 0, so its plan is unbiased by any d̂. (ki = 0: no
//! integrator, the trim is a setpoint offset sp_ss = −d/(c·kp).)
//!
//! COST per outer step q_att·θ² + q_rate·ω² + r·sp², terminal cost the DARE solution
//! of the same model (so the unconstrained law is the infinite-horizon LQR on the
//! cascade), box |sp| ≤ the bank's rate range, solved exactly (box_qp.rs).

use crate::box_qp::BoxQpSolver;
use crate::optimal::{condense_horizon, dare_doubling, CondensedHorizon, Mat};

/// Per-axis prediction state dimension: [θ, ω, ua, I, f1, f2, pm, Fh].
pub(crate) const NS: usize = 8;
pub(crate) const TH: usize = 0;
pub(crate) const W: usize = 1;
pub(crate) const UA: usize = 2;
pub(crate) const IN: usize = 3;
pub(crate) const F1: usize = 4;
pub(crate) const F2: usize = 5;
pub(crate) const PM: usize = 6;
pub(crate) const FH: usize = 7;

/// One axis of the closed inner loop, in SI.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AxisPlant
{
	pub kp: f64,
	pub ki: f64,
	pub kd: f64,
	/// The rate-D lpf2p coefficients (b0, b1, b2, a1, a2); None = filter off.
	pub lpf: Option<[f64; 5]>,
	/// Plant tick (s).
	pub dt: f64,
	/// Rate-loop sample period 1/attitude_hz (s) — the PID's own dt.
	pub dt_rate: f64,
	/// Plant ticks per rate update.
	pub decimation: u32,
	/// Body angular acceleration per newton of axis force (rad/s² per N).
	pub c: f64,
	/// Motor-lag filter coefficient per tick (1 = no lag).
	pub alpha: f64,
}

impl AxisPlant
{
	/// One plant tick of the linear closed inner loop (module doc). `update` = this
	/// tick runs the rate PID.
	pub(crate) fn tick(&self, s: &[f64; NS], sp: f64, d: f64, update: bool) -> [f64; NS]
	{
		let mut n = *s;
		if update
		{
			let e = sp - s[W];
			n[IN] = s[IN] + e * self.dt_rate;
			let raw = -(s[W] - s[PM]) / self.dt_rate;
			let deriv = match self.lpf
			{
				Some([b0, b1, b2, a1, a2]) =>
				{
					let d0 = raw - s[F1] * a1 - s[F2] * a2;
					n[F2] = s[F1];
					n[F1] = d0;
					d0 * b0 + s[F1] * b1 + s[F2] * b2
				}
				None => raw,
			};
			n[PM] = s[W];
			n[FH] = self.kp * e + self.ki * n[IN] + self.kd * deriv;
		}
		n[UA] = s[UA] + self.alpha * (n[FH] - s[UA]);
		let acc = self.c * n[UA] + d;
		n[W] = s[W] + self.dt * acc;
		n[TH] = s[TH] + self.dt * s[W] + 0.5 * self.dt * self.dt * acc;
		n
	}

	/// `n_o` ticks from rate-loop phase `phase` (ticks since the last update), with
	/// the setpoint held: s_{k+1} = A·s_k + B·sp_k at the outer sample time.
	pub(crate) fn outer_map(&self, n_o: usize, phase: u32) -> (Mat, Mat)
	{
		let dec = self.decimation.max(1) as usize;
		let run = |s0: [f64; NS], sp: f64| {
			let mut s = s0;
			for j in 0..n_o
			{
				s = self.tick(&s, sp, 0.0, (phase as usize + j) % dec == 0);
			}
			s
		};
		let mut a = Mat::zeros(NS, NS);
		for j in 0..NS
		{
			let mut e = [0.0f64; NS];
			e[j] = 1.0;
			let col = run(e, 0.0);
			for (i, v) in col.iter().enumerate()
			{
				a.set(i, j, *v);
			}
		}
		let colb = run([0.0; NS], 1.0);
		let mut b = Mat::zeros(NS, 1);
		for (i, v) in colb.iter().enumerate()
		{
			b.set(i, 0, *v);
		}
		(a, b)
	}

	/// The rest point under a constant input disturbance d (module doc): the state
	/// and the setpoint that hold θ = ω = 0.
	pub(crate) fn equilibrium(&self, d: f64) -> ([f64; NS], f64)
	{
		let f = -d / self.c;
		let mut x = [0.0f64; NS];
		x[UA] = f;
		x[FH] = f;
		if self.ki != 0.0
		{
			x[IN] = f / self.ki;
			(x, 0.0)
		}
		else
		{
			(x, f / self.kp)
		}
	}
}

/// Stage-cost weights of one axis.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AxisWeights
{
	pub q_att: f64,
	pub q_rate: f64,
	pub r: f64,
}

/// The condensed horizon QP of one axis for one (n_o, phase).
pub(crate) struct CtbrAxisMpc
{
	n: usize,
	bound: f64,
	ustar_map: Mat, // N×NS: unconstrained U* = ustar_map · x̃0
	g_map: Mat,     // N×NS: g = g_map · x̃0
	qp: BoxQpSolver,
	warm: Vec<f64>,
	u_buf: Vec<f64>,
	g_buf: Vec<f64>,
}

impl CtbrAxisMpc
{
	pub(crate) fn build(plant: &AxisPlant, n_o: usize, phase: u32, w: AxisWeights, horizon: usize, bound: f64) -> Self
	{
		assert!(plant.kp > 0.0 && plant.c > 0.0, "CtbrAxisMpc: the rate loop needs kp > 0 and a torque gain c > 0");
		assert!(horizon >= 1 && bound > 0.0, "CtbrAxisMpc: horizon >= 1 and bound > 0");
		let (a, b) = plant.outer_map(n_o, phase);
		let mut qm = Mat::zeros(NS, NS);
		qm.set(TH, TH, w.q_att);
		qm.set(W, W, w.q_rate);
		let mut rm = Mat::zeros(1, 1);
		rm.set(0, 0, w.r);
		let qf = dare_doubling(&a, &b, &qm, &rm);
		let CondensedHorizon { h, g_map, ustar_map } = condense_horizon(&a, &b, &qm, &rm, &qf, horizon);
		CtbrAxisMpc {
			n: horizon,
			bound,
			ustar_map,
			g_map,
			qp: BoxQpSolver::new(horizon, h.data().to_vec(), bound),
			warm: vec![0.0; horizon],
			u_buf: vec![0.0; horizon],
			g_buf: vec![0.0; horizon],
		}
	}

	/// The unconstrained law's FIRST setpoint as a row over the shifted state x̃0:
	/// sp = row · x̃0 (box inactive) — the linear gains the L2y input is built from
	/// (yaw_trim_err.rs).
	pub(crate) fn gain_row(plant: &AxisPlant, n_o: usize, phase: u32, w: AxisWeights, horizon: usize, bound: f64) -> [f64; NS]
	{
		let m = CtbrAxisMpc::build(plant, n_o, phase, w, horizon, bound);
		std::array::from_fn(|j| m.ustar_map.get(0, j))
	}

	pub(crate) fn reset(&mut self)
	{
		self.warm.iter_mut().for_each(|v| *v = 0.0);
	}

	/// The first setpoint of the optimal plan from the shifted state x̃0 = x0 − x_ss.
	pub(crate) fn solve(&mut self, x: &[f64; NS]) -> f64
	{
		self.ustar_map.mul_vec_into(x, &mut self.u_buf);
		let a = self.bound;
		if !self.u_buf.iter().all(|u| u.abs() <= a + 1e-9)
		{
			self.g_map.mul_vec_into(x, &mut self.g_buf);
			let u = self.qp.solve(&self.g_buf, &self.warm);
			self.u_buf.copy_from_slice(u);
		}
		let n = self.n;
		self.warm[..n - 1].copy_from_slice(&self.u_buf[1..]);
		self.warm[n - 1] = self.u_buf[n - 1];
		self.u_buf[0]
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	fn cf21_roll(alpha: f64) -> AxisPlant
	{
		let lpf = crate::pid_firmware::Lpf2p::new(500.0, 30.0).coeffs();
		AxisPlant {
			kp: 0.03497110216713654,
			ki: 0.06994220433427308,
			kd: 0.00043713877708920673,
			lpf: Some(lpf),
			dt: 0.001,
			dt_rate: 0.002,
			decimation: 2,
			c: 0.070710678 / 3.004e-5,
			alpha,
		}
	}

	/// The equilibrium is a fixed point of the tick under its disturbance, on update
	/// and hold ticks alike — the offset-free shift is exact.
	#[test]
	fn equilibrium_is_a_fixed_point()
	{
		for alpha in [1.0, 0.02]
		{
			let p = cf21_roll(alpha);
			let d = 37.0;
			let (x, sp) = p.equilibrium(d);
			for upd in [true, false]
			{
				let n = p.tick(&x, sp, d, upd);
				for i in 0..NS
				{
					assert!((n[i] - x[i]).abs() < 1e-12, "alpha {alpha} upd {upd} state {i}: {} vs {}", n[i], x[i]);
				}
			}
		}
	}

	/// The outer map is the tick composition: A·s + B·sp == n_o ticks of `tick`.
	#[test]
	fn outer_map_is_the_tick_composition()
	{
		let p = cf21_roll(0.02);
		for phase in [0u32, 1]
		{
			let (a, b) = p.outer_map(10, phase);
			let s0 = [0.03, -0.4, 0.002, 0.01, 0.5, -0.3, -0.35, 0.001];
			let mut s = s0;
			for j in 0..10usize
			{
				s = p.tick(&s, 0.7, 0.0, (phase as usize + j) % 2 == 0);
			}
			for i in 0..NS
			{
				let mut v = b.get(i, 0) * 0.7;
				for j in 0..NS
				{
					v += a.get(i, j) * s0[j];
				}
				assert!((v - s[i]).abs() < 1e-12, "phase {phase} state {i}: {v} vs {}", s[i]);
			}
		}
	}

	/// The planned setpoint opposes the attitude error, and the unconstrained law
	/// (terminal = DARE) closes a stable loop on the model.
	#[test]
	fn plan_levels_the_model()
	{
		for alpha in [1.0, 4.0 * 0.001 / 0.20]
		{
			let p = cf21_roll(alpha);
			let w = AxisWeights { q_att: 100.0, q_rate: 0.5, r: 1.0 };
			let mut m = CtbrAxisMpc::build(&p, 10, 0, w, 20, 4.2);
			let mut s = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
			let sp0 = m.solve(&s);
			assert!(sp0 < -0.1, "alpha {alpha}: +θ must command −rate, got {sp0}");
			for _ in 0..200
			{
				let sp = m.solve(&s);
				for j in 0..10usize
				{
					s = p.tick(&s, sp, 0.0, j % 2 == 0);
				}
			}
			assert!(s[TH].abs() < 1e-4 && s[W].abs() < 1e-3, "alpha {alpha}: not levelled after 2 s: {s:?}");
		}
	}
}
