//! ABI 39 (CTRL-33, Luiz 06/11/2026): the cascade's MPCOF OUTER-loop teacher — the
//! offset-free MPC whose prediction model is the closed inner loop (firmware rate PID
//! + motor lag + rigid body, ctbr_axis_mpc.rs) and whose decision is the body-rate
//! setpoint it hands that loop. Replaces the ABI-38 label (the motor-level MPC's
//! planned rate N_o steps ahead), which did not know the rate loop existed.
//!
//! STATE THE PLAN STARTS FROM, per axis, at every outer decision (BODY axes):
//!   θ, ω   the Euler-angle error and the gyro (optimal::state_error — the firmware
//!          attitude stage's own convention: Euler error per axis → that axis' rate);
//!   ua     MEASURED: the pwm the plant was told to apply, filtered through the
//!          plant's own lag (the ActuatorLag prime/seed protocol, per motor) and
//!          turned into differential thrust t = kt·pwm² — the rotors' actual force;
//!   I, f1, f2, pm, Fh  the rate loop's internals, read from a SHADOW copy of the
//!          firmware rate stage that this teacher runs on the measured gyro with its
//!          own held setpoints. When the teacher flies (the C4 rival, expert_drives,
//!          calibration, the recorder) the shadow IS the inner loop, state for state.
//!          Under DAgger the student's own setpoints drive the real loop; the shadow is
//!          then the loop the teacher's policy would have produced from the same gyro
//!          history, and any force mismatch reaches the plan through the observer.
//! OFFSET-FREE OBSERVER (attitude level, input disturbance on ω'): every tick
//!   d̂ ← d̂ + l·(ω'_measured − c·ua − d̂),
//! ω' from consecutive gyro reads, ua the measured lagged force above — so d̂ is the
//! torque the rotors' nominal thrust does NOT explain (motor asymmetry, torque-scale
//! jitter, rotor drag, gusts). The plan regulates x − x_ss(d̂) (ctbr_axis_mpc.rs).
//!
//! COLLECTIVE: not planned here. Teacher::ctbr_label passes the caller's collective
//! deviation through unchanged (the altitude PD, as for the PID outer teacher) — the
//! rate loop's mixer adds it uniformly, which leaves every axis force as planned.
//!
//! TIMING: one decision per `n_o` ticks (the student's action_repeat), held between —
//! the setpoint stream the student must reproduce and the one CtbrOverRate flies. The
//! decision ticks are the teacher's own tick % n_o == 0 since reset, which every host
//! aligns with its hold (all reset at the episode start and step every tick).

use crate::ctbr_axis_mpc::{AxisPlant, AxisWeights, CtbrAxisMpc, NS};
use crate::pid_firmware::RateStageFw;

/// |sp| bound per axis (rad/s) = the CTBR banks' default rate range
/// (wnn/control/evaluator.py CTBR_DEFAULT_RATE_MAX): a label beyond it is clipped by
/// the student's encoder anyway, so the plan respects it instead of being clipped.
pub(crate) const CTBR_MPC_RATE_MAX: [f64; 3] = [4.2, 4.2, 2.0];
/// Horizon in OUTER steps (terminal cost = the DARE solution of the same model).
pub(crate) const CTBR_MPC_HORIZON: usize = 20;
/// Stage weights (θ in rad, ω and sp in rad/s), all three axes.
pub(crate) const CTBR_MPC_WEIGHTS: AxisWeights = AxisWeights { q_att: 100.0, q_rate: 0.5, r: 1.0 };
/// Observer gain per plant tick.
pub(crate) const CTBR_MPC_L_GAIN: f64 = 0.02;

pub(crate) struct CtbrMpcOf
{
	axes: [AxisPlant; 3],
	k_thrust: f64,
	dt: f64,
	/// Lazily built per (n_o, rate-loop phase at the decision tick).
	planners: Vec<(usize, u32, Vec<CtbrAxisMpc>)>,
	shadow: RateStageFw,
	// measured actuator: per-motor lagged pwm, ActuatorLag's prime/seed protocol
	lag_pwm: [f64; 4],
	alpha: f64,
	primed: bool,
	seeded: bool,
	dhat: [f64; 3],
	dhat_clamp: [f64; 3],
	last_gyro: Option<[f64; 3]>,
	tick: usize,
	held: [f64; 3],
	/// A CTBR label has been asked for (the host is a cascade) — integrals() then
	/// reports this observer, not the motor-level one.
	active: bool,
}

impl CtbrMpcOf
{
	/// `rate` = the airframe's firmware rate stage (AttitudePidFirmwareRs::rate_stage —
	/// the same gains every CTBR inner loop is built from).
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn new(
		rate: RateStageFw,
		dt: f32,
		arm_length: f32,
		k_drag: f32,
		inertia: [f32; 3],
		k_thrust: f32,
		motor_lag_s: f32,
	) -> Self
	{
		let alpha = crate::actuator_lag::motor_lag_alpha(motor_lag_s, dt);
		let lpf = rate.lpf_coeffs();
		let (arm, kd) = (arm_length as f64, k_drag as f64);
		let c = [arm / inertia[0] as f64, arm / inertia[1] as f64, 4.0 * kd / inertia[2] as f64];
		let axis = |i: usize| {
			let g = rate.axis_gains(i);
			AxisPlant {
				kp: g.kp,
				ki: g.ki,
				kd: g.kd,
				lpf,
				dt: dt as f64,
				dt_rate: rate.dt_rate(),
				decimation: rate.decimation(),
				c: c[i],
				alpha,
			}
		};
		let out = rate.out_limit_n();
		let mut shadow = rate;
		shadow.reset();
		CtbrMpcOf {
			axes: [axis(0), axis(1), axis(2)],
			k_thrust: k_thrust as f64,
			dt: dt as f64,
			planners: Vec::new(),
			shadow,
			lag_pwm: [0.0; 4],
			alpha,
			primed: false,
			seeded: false,
			dhat: [0.0; 3],
			dhat_clamp: [c[0] * out, c[1] * out, c[2] * out],
			last_gyro: None,
			tick: 0,
			held: [0.0; 3],
			active: false,
		}
	}

	pub(crate) fn reset(&mut self)
	{
		self.shadow.reset();
		self.lag_pwm = [0.0; 4];
		self.primed = false;
		self.seeded = false;
		self.dhat = [0.0; 3];
		self.last_gyro = None;
		self.tick = 0;
		self.held = [0.0; 3];
		for (_, _, p) in self.planners.iter_mut()
		{
			p.iter_mut().for_each(|m| m.reset());
		}
	}

	pub(crate) fn active(&self) -> bool
	{
		self.active
	}

	/// The (gyro just read, pwm applied last step) pair — the DAgger observe protocol
	/// (first call after reset = the t=0 placeholder: prime only).
	pub(crate) fn observe(&mut self, gyro: [f32; 3], applied_pwm: [f64; 4])
	{
		let g = [gyro[0] as f64, gyro[1] as f64, gyro[2] as f64];
		if let (Some(ua), Some(prev)) = (self.track_lag(applied_pwm), self.last_gyro)
		{
			for a in 0..3
			{
				let rate_dot = (g[a] - prev[a]) / self.dt;
				let residual = rate_dot - self.axes[a].c * ua[a] - self.dhat[a];
				let lim = self.dhat_clamp[a];
				self.dhat[a] = (self.dhat[a] + CTBR_MPC_L_GAIN * residual).clamp(-lim, lim);
			}
		}
		self.last_gyro = Some(g);
	}

	/// Advance the measured-actuator filter; the lagged axis forces once seeded.
	fn track_lag(&mut self, applied: [f64; 4]) -> Option<[f64; 3]>
	{
		if !self.primed
		{
			self.primed = true;
			return None;
		}
		let cmd = applied.map(|v| v.clamp(0.0, 1.0));
		if !self.seeded
		{
			self.lag_pwm = cmd;
			self.seeded = true;
		}
		else
		{
			for m in 0..4
			{
				self.lag_pwm[m] += self.alpha * (cmd[m] - self.lag_pwm[m]);
			}
		}
		Some(self.lagged_force())
	}

	/// The rotors' differential thrust in the rate stage's axis-force coordinates.
	fn lagged_force(&self) -> [f64; 3]
	{
		let t = self.lag_pwm.map(|p| self.k_thrust * p * p);
		[t[3] - t[1], t[2] - t[0], (t[0] - t[1] + t[2] - t[3]) * 0.25]
	}

	/// The body-rate setpoint (rad/s, BODY axes) for this tick: a fresh plan on
	/// decision ticks, the held one between. Call every physical tick.
	pub(crate) fn label(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3], n_o: usize) -> [f64; 3]
	{
		self.active = true;
		let n_o = n_o.max(1);
		if self.tick % n_o == 0
		{
			self.held = self.decide(q, gyro, target_rpy, n_o);
		}
		self.tick += 1;
		self.shadow.step(self.held, [gyro[0] as f64, gyro[1] as f64, gyro[2] as f64], 0.0);
		self.held
	}

	fn decide(&mut self, q: [f32; 4], gyro: [f32; 3], target_rpy: [f32; 3], n_o: usize) -> [f64; 3]
	{
		let x = crate::optimal::state_error(q, gyro, target_rpy);
		let idx = self.planner_index(n_o, self.shadow.phase());
		let ua = if self.seeded { Some(self.lagged_force()) } else { None };
		let mut sp = [0.0f64; 3];
		for a in 0..3
		{
			let st = self.shadow.axis_state(a);
			let s0 = [
				x[a],
				x[3 + a],
				// Before anything has flown the plant seeds its lag AT the first
				// command, i.e. the rotors will be at the loop's force.
				ua.map_or(st.held, |u| u[a]),
				st.integ,
				st.lpf[0],
				st.lpf[1],
				// pid.c's first update has no D term: as if the last reading were now.
				if st.first { x[3 + a] } else { st.prev_measured },
				st.held,
			];
			let (xss, sp_ss) = self.axes[a].equilibrium(self.dhat[a]);
			let mut xt = [0.0f64; NS];
			for i in 0..NS
			{
				xt[i] = s0[i] - xss[i];
			}
			let bound = CTBR_MPC_RATE_MAX[a];
			sp[a] = (self.planners[idx].2[a].solve(&xt) + sp_ss).clamp(-bound, bound);
		}
		sp
	}

	fn planner_index(&mut self, n_o: usize, phase: u32) -> usize
	{
		if let Some(i) = self.planners.iter().position(|(n, p, _)| *n == n_o && *p == phase)
		{
			return i;
		}
		let axes = self.axes;
		let p = (0..3)
			.map(|a| CtbrAxisMpc::build(&axes[a], n_o, phase, CTBR_MPC_WEIGHTS, CTBR_MPC_HORIZON, CTBR_MPC_RATE_MAX[a]))
			.collect();
		self.planners.push((n_o, phase, p));
		self.planners.len() - 1
	}

	/// The observer state d̂ (rad/s²) — the Option-A integral-target analogue, as the
	/// motor-level MPCOF's.
	pub(crate) fn integrals(&self) -> [f32; 3]
	{
		self.dhat.map(|v| v as f32)
	}

	/// d̂'s clamp: the most torque the rate loop's output limit can cancel (c·out_limit).
	pub(crate) fn i_clamps(&self) -> [f32; 3]
	{
		self.dhat_clamp.map(|v| v as f32)
	}
}
