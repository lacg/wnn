//! ABI 38 (CTRL-27, Luiz 06/11/2026): the CASCADE controller's output contract.
//!
//! "Collective thrust + body rates" (CTBR): the WNN is the OUTER (attitude) loop and
//! emits four banks — [collective deviation (pwm), p_sp, q_sp, r_sp (rad/s, body
//! frame)] — decoded ABSOLUTELY (no hidden accumulator) on a per-bank range and γ
//! alphabet, held across action-repeat. The INNER loop is the firmware rate PID
//! (`pid_firmware::RateStageFw`, sourced cf21bl gains, never re-derived) running at
//! its own 500 Hz on the measured gyro, so the 50 ms rotor lag sits inside a classical
//! loop with phase lead instead of inside the WNN's delta accumulator
//! (docs/ctrl_cascade_design.md §0, §1).
//!
//! THE COORDINATES. Every bank is a DEVIATION around 0: the rate banks are the
//! setpoint itself, the collective bank is the pwm deviation the teacher's altitude
//! PD adds to its own hover (`AltitudePd::delta`). The inner stage composes the
//! collective exactly as the PID teacher does — mixer at its hover, then a uniform
//! pwm offset — so a setpoint stream equal to the PID teacher's labels reproduces the
//! PID teacher bit for bit (label-replay identity, abi38_tests). Under translation the
//! student rides on the episode's COMMANDED collective (the motor path's anchor): the
//! offset becomes (anchor − inner hover) + deviation — the D0 transport rule.
//!
//! One definition of the bank alphabet lives here (`CtbrSpec::decode` / `encode`):
//! the CPU decode, the labels of every trainer (sn=0 and sn>0 EDRA-BPTT, CPU and GPU
//! walk via host-prepared targets, the state-split trainer) and the Metal kernel's
//! uniforms all read it.

use crate::pid_firmware::RateStageFw;
use pyo3::prelude::*;

/// Banks: [collective deviation, p, q, r].
pub const CTBR_BANKS: usize = 4;

/// The CTBR outer/inner contract a cascade WnnController is built with. Python builds
/// it from the airframe's firmware cascade (EpisodeConfig.cascade_kwargs — the SI
/// conversion happens there and nowhere else) plus the ControllerSpec's ranges/γ.
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct CtbrSpec
{
	/// Rate loop, [roll, pitch, yaw] x [kp, ki, kd, i_limit] in SI (N per rad/s).
	#[pyo3(get)]
	pub rate_gains: [f64; 12],
	/// int16 output saturation in newtons.
	#[pyo3(get)]
	pub out_limit_n: f64,
	/// Per-motor hover thrust m·g/4 (N) — the inner stage's mixer operating point.
	#[pyo3(get)]
	pub hover_n: f64,
	#[pyo3(get)]
	pub k_thrust: f64,
	/// The plant tick rate (1/dt) and the firmware ATTITUDE_RATE (500): the rate loop
	/// updates every main_loop_hz/attitude_hz ticks and holds in between.
	#[pyo3(get)]
	pub main_loop_hz: u32,
	#[pyo3(get)]
	pub attitude_hz: f64,
	/// Rate-D lpf2p cutoff (Hz; 0 = off). The firmware's own 30 Hz facility.
	#[pyo3(get)]
	pub lpf_hz: f64,
	/// Per-bank range: [collective pwm, p, q, r rad/s]. The bank decodes to
	/// ±range around 0.
	#[pyo3(get)]
	pub range: [f32; CTBR_BANKS],
	/// Per-bank γ of the |t|^γ alphabet (1 = linear). γ = 2 on the rate banks
	/// keeps the smallest step below the steady-error budget (design §1.3).
	#[pyo3(get)]
	pub gamma: [f32; CTBR_BANKS],
	/// CTRL-38 a1 (`--obs-rate-i`): append the inner rate loop's integrator I/i_limit
	/// per axis (cascade_obs.rs). false ⇒ the ABI-40 feature layout.
	#[pyo3(get)]
	pub obs_rate_i: bool,
	/// CTRL-38 a2 (`--obs-cascade-dhat`): the airframe constants [arm_length, k_drag,
	/// Ixx, Iyy, Izz] the student's copy of the teacher's offset-free observer needs
	/// (its torque gains c; k_thrust and out_limit come from the fields above). Some ⇒
	/// append d̂ roll/pitch/yaw; None ⇒ the ABI-40 feature layout.
	#[pyo3(get)]
	pub obs_dhat_plant: Option<[f64; 5]>,
	/// CTRL-48 a3 (`--obs-cascade-dhat-residual`, ABI 45): the same airframe constants,
	/// for the observer the UNABSORBED-disturbance block runs on (cascade_obs.rs). Some
	/// ⇒ append (d̂ + c·ki·I)/(c·ki·i_limit) per axis, independent of a2 (must equal
	/// obs_dhat_plant when both are set); None ⇒ the ABI-44 feature layout.
	#[pyo3(get)]
	pub obs_dhat_residual_plant: Option<[f64; 5]>,
}

impl CtbrSpec
{
	pub(crate) fn validate(&self) -> Result<(), String>
	{
		if !(self.attitude_hz > 0.0) || self.main_loop_hz == 0
		{
			return Err(format!(
				"CtbrSpec: the inner rate loop needs the firmware cascade's rates (attitude_hz={}, main_loop_hz={}) — \
				 the airframe registers no cascade gains?",
				self.attitude_hz, self.main_loop_hz
			));
		}
		if !(self.k_thrust > 0.0 && self.hover_n > 0.0 && self.out_limit_n > 0.0)
		{
			return Err(format!(
				"CtbrSpec: k_thrust={}, hover_n={}, out_limit_n={} must all be > 0",
				self.k_thrust, self.hover_n, self.out_limit_n
			));
		}
		if self.rate_gains.iter().any(|v| !v.is_finite())
		{
			return Err("CtbrSpec: rate gains must be finite".into());
		}
		for p in [self.obs_dhat_plant, self.obs_dhat_residual_plant].into_iter().flatten()
		{
			if p.iter().any(|v| !(v.is_finite() && *v > 0.0))
			{
				return Err(format!(
					"CtbrSpec: obs_dhat_plant [arm_length, k_drag, Ixx, Iyy, Izz] = {p:?} must all be positive and finite"
				));
			}
		}
		if let (Some(a), Some(b)) = (self.obs_dhat_plant, self.obs_dhat_residual_plant)
		{
			if a != b
			{
				return Err(format!(
					"CtbrSpec: obs_dhat_plant {a:?} and obs_dhat_residual_plant {b:?} must name the same airframe"
				));
			}
		}
		for m in 0..CTBR_BANKS
		{
			let (r, g) = (self.range[m], self.gamma[m]);
			if !(r.is_finite() && r > 0.0)
			{
				return Err(format!("CtbrSpec: range[{m}] = {r} must be a positive finite number"));
			}
			if !(g.is_finite() && g > 0.0)
			{
				return Err(format!("CtbrSpec: gamma[{m}] = {g} must be a positive finite number"));
			}
		}
		Ok(())
	}

	/// The inner stage, fresh (reset). Built through the SAME constructor as the PID
	/// teacher's cascade (`AttitudePidFirmwareRs::from_si_arrays`), so the two can
	/// never hold different gains, filters or decimation.
	pub(crate) fn rate_stage(&self) -> RateStageFw
	{
		RateStageFw::from_si_arrays(
			self.rate_gains,
			self.out_limit_n,
			self.hover_n,
			self.k_thrust,
			self.main_loop_hz,
			self.attitude_hz,
			self.lpf_hz,
		)
		.expect("CtbrSpec::validate guarantees attitude_hz > 0")
	}

	/// Bank `m`'s ABSOLUTE decode: raw decode ∈ [0, 1] → setpoint ∈ [−range, +range],
	/// the untrained neutral ↦ 0 (zero rate, the anchor's collective).
	#[inline]
	pub(crate) fn decode(&self, m: usize, decoded: f32, neutral: f32) -> f32
	{
		crate::controller::decoded_to_delta(decoded, self.range[m], neutral, self.gamma[m])
	}

	/// Inverse of `decode` (label → raw decode target), clipped at ±range like any
	/// out-of-range teacher command.
	#[inline]
	pub(crate) fn encode(&self, m: usize, setpoint: f32, neutral: f32) -> f32
	{
		crate::controller::delta_to_decoded(setpoint, self.range[m], neutral, self.gamma[m])
	}

	/// The delta-law ablation (`--delta-control` under CTBR): the per-step increment
	/// alphabet is `delta_max` of each bank's RANGE, with the bank's γ.
	#[inline]
	pub(crate) fn decode_delta(&self, m: usize, decoded: f32, neutral: f32, delta_max: f32) -> f32
	{
		crate::controller::decoded_to_delta(decoded, delta_max * self.range[m], neutral, self.gamma[m])
	}

	#[inline]
	pub(crate) fn encode_delta(&self, m: usize, delta: f32, neutral: f32, delta_max: f32) -> f32
	{
		crate::controller::delta_to_decoded(delta, delta_max * self.range[m], neutral, self.gamma[m])
	}

	/// The provenance token's CTBR half. The CTRL-38 blocks appear only when on, so
	/// every ABI-40 token is unchanged.
	pub(crate) fn provenance_line(&self) -> String
	{
		let mut s = format!(
			"inner=pidfw_rate rate_hz={} range={:?} gamma={:?}",
			self.attitude_hz, self.range, self.gamma
		);
		if self.obs_rate_i
		{
			s.push_str(" obs=rate_i");
		}
		if let Some(p) = self.obs_dhat_plant
		{
			s.push_str(&format!(" obs=dhat(l={},plant={p:?})", crate::ctbr_mpcof::CTBR_MPC_L_GAIN));
		}
		if self.obs_dhat_residual_plant.is_some()
		{
			s.push_str(" obs=dhat_resid");
		}
		s
	}

	/// CTRL-38/48: the features the cascade blocks append (0 when all are off).
	pub(crate) fn cascade_feature_count(&self) -> usize
	{
		3 * (self.obs_rate_i as usize)
			+ 3 * (self.obs_dhat_plant.is_some() as usize)
			+ 3 * (self.obs_dhat_residual_plant.is_some() as usize)
	}

	/// CTRL-48 a3: the residual's scale from this spec's rate loop and airframe.
	fn residual_scale(&self, d: &crate::cascade_dhat::CascadeDhat) -> crate::cascade_obs::CasResidual
	{
		let rs = self.rate_stage();
		let ki_lim = std::array::from_fn(|i| (rs.axis_gains(i).ki, rs.axis_gains(i).i_limit));
		crate::cascade_obs::CasResidual::new(d.torque_gain(), ki_lim, d.clamp())
	}

	/// CTRL-38: the student's cascade feature block, None when both halves are off.
	/// The observer is the teacher's (cascade_dhat.rs) on this spec's airframe:
	/// torque gains from the plant constants, k_thrust and the rate loop's output
	/// limit from the contract; lag off until a host sets the plant's
	/// (set_observer_motor_lag), exactly like the motor-level observer.
	pub(crate) fn cascade_obs(&self, dt: f32) -> Option<crate::cascade_obs::CascadeObs>
	{
		if self.cascade_feature_count() == 0
		{
			return None;
		}
		let dhat = self.obs_dhat_plant.or(self.obs_dhat_residual_plant).map(|p| {
			let c = crate::cascade_dhat::torque_gains(p[0] as f32, p[1] as f32, [p[2] as f32, p[3] as f32, p[4] as f32]);
			crate::cascade_dhat::CascadeDhat::new(c, self.k_thrust, dt, 0.0, self.out_limit_n)
		});
		let resid = match (self.obs_dhat_residual_plant, dhat.as_ref())
		{
			(Some(_), Some(d)) => Some(self.residual_scale(d)),
			_ => None,
		};
		Some(crate::cascade_obs::CascadeObs::new(self.obs_rate_i, dhat, self.obs_dhat_plant.is_some(), resid))
	}
}

#[pymethods]
impl CtbrSpec
{
	#[new]
	#[pyo3(signature = (rate_gains, out_limit_n, hover_n, k_thrust, main_loop_hz, attitude_hz, lpf_hz, range, gamma,
	                    obs_rate_i = false, obs_dhat_plant = None, obs_dhat_residual_plant = None))]
	#[allow(clippy::too_many_arguments)]
	fn py_new(
		rate_gains: [f64; 12],
		out_limit_n: f64,
		hover_n: f64,
		k_thrust: f64,
		main_loop_hz: u32,
		attitude_hz: f64,
		lpf_hz: f64,
		range: [f32; CTBR_BANKS],
		gamma: [f32; CTBR_BANKS],
		obs_rate_i: bool,
		obs_dhat_plant: Option<[f64; 5]>,
		obs_dhat_residual_plant: Option<[f64; 5]>,
	) -> PyResult<Self>
	{
		let s = CtbrSpec {
			rate_gains,
			out_limit_n,
			hover_n,
			k_thrust,
			main_loop_hz,
			attitude_hz,
			lpf_hz,
			range,
			gamma,
			obs_rate_i,
			obs_dhat_plant,
			obs_dhat_residual_plant,
		};
		s.validate().map_err(pyo3::exceptions::PyValueError::new_err)?;
		Ok(s)
	}

	/// The inner stage's hover pwm (the PID teacher's hover, through the same
	/// clamp-and-sqrt): the operating point of an un-translated cascade.
	#[getter]
	fn hover_pwm(&self) -> f64
	{
		self.rate_stage().hover_pwm()
	}

	fn __repr__(&self) -> String
	{
		format!("CtbrSpec({})", self.provenance_line())
	}
}

/// The cascade's runtime state inside a WnnController: the inner stage, the HELD
/// setpoint of the last decision and the collective transport.
#[derive(Clone, Debug)]
pub(crate) struct CtbrInner
{
	pub spec: CtbrSpec,
	rate: RateStageFw,
	/// [collective deviation, p, q, r] of the last DECISION step, flown on every
	/// physical step until the next decision (action-repeat hold). f64 so a test can
	/// hold a teacher label EXACTLY (the label-replay identity); a decoded f32
	/// setpoint widens losslessly.
	held: [f64; CTBR_BANKS],
	/// (collective anchor − inner hover), pwm: the D0 transport onto the episode's
	/// commanded collective. 0 until a translating host anchors the controller, so
	/// an attitude-only cascade flies the inner stage at its own hover.
	transport: f64,
}

impl CtbrInner
{
	pub(crate) fn new(spec: CtbrSpec) -> Result<Self, String>
	{
		spec.validate()?;
		let rate = spec.rate_stage();
		Ok(CtbrInner { spec, rate, held: [0.0; CTBR_BANKS], transport: 0.0 })
	}

	/// Episode start: the rate loop forgets its integrals/filters/tick and the held
	/// setpoint returns to the untrained decode (zero rate, zero deviation). The
	/// transport is the anchor's — it is re-set by set_collective_anchor, which hosts
	/// call AFTER reset.
	pub(crate) fn reset(&mut self)
	{
		self.rate.reset();
		self.held = [0.0; CTBR_BANKS];
	}

	pub(crate) fn set_anchor(&mut self, anchor_pwm: f32)
	{
		self.transport = anchor_pwm as f64 - self.rate.hover_pwm();
	}

	pub(crate) fn hold(&mut self, sp: [f32; CTBR_BANKS])
	{
		self.held = [sp[0] as f64, sp[1] as f64, sp[2] as f64, sp[3] as f64];
	}

	/// TEST-ONLY: hold an exact (f64) setpoint — the label-replay identity.
	#[cfg(test)]
	pub(crate) fn hold_exact(&mut self, sp: [f64; CTBR_BANKS])
	{
		self.held = sp;
	}

	pub(crate) fn held(&self) -> [f64; CTBR_BANKS]
	{
		self.held
	}

	/// The inner rate loop itself (a copy: DAgger hands it to the teacher's plan).
	pub(crate) fn rate_stage_copy(&self) -> RateStageFw
	{
		self.rate
	}

	/// The inner rate loop's axis state (integrator, filter, held force).
	pub(crate) fn rate_axis_state(&self, i: usize) -> crate::pid_firmware::RateAxisState
	{
		self.rate.axis_state(i)
	}

	/// CTRL-38 a1: the integrator I per axis over the firmware's own clamp i_limit
	/// (∈ [−1, 1]); an unclamped axis (i_limit = 0) reports I itself.
	pub(crate) fn rate_integral_norm(&self) -> [f32; 3]
	{
		std::array::from_fn(|i| {
			let lim = self.rate.axis_gains(i).i_limit;
			let integ = self.rate.axis_state(i).integ;
			(if lim > 0.0 { integ / lim } else { integ }) as f32
		})
	}

	/// CTRL-38: adopt the state of the rate loop that is ACTUALLY flying (a teacher's
	/// ExpertPilot) — the rate-I feature then reads the vehicle's integrator when the
	/// student's own loop does not drive the plant (calibration, recorder,
	/// expert_drives). The held setpoint and transport are untouched.
	pub(crate) fn adopt_rate_stage(&mut self, flown: &RateStageFw)
	{
		self.rate = *flown;
	}

	/// One physical tick of the inner loop on the held setpoint and the measured gyro.
	pub(crate) fn fly(&mut self, gyro: [f32; 3]) -> [f32; 4]
	{
		let h = self.held;
		// transport + deviation; the transport is exactly 0 off translation, and
		// 0.0 + x == x, so an un-anchored cascade adds the bare deviation.
		let p = self.rate.step([h[1], h[2], h[3]], [gyro[0] as f64, gyro[1] as f64, gyro[2] as f64], self.transport + h[0]);
		[p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
	}
}

/// A classical outer law flown over the cascade's inner stage: the C2 rival
/// (firmware attitude PID → firmware rate PID), the C4 rival (MPCOF label → rate
/// PID), and the teacher-driven rollouts of a CTBR run (expert_drives, threshold
/// calibration, the MEMORY recorder) — so a CTBR student's training cascade is
/// flown by its teacher THROUGH the same inner loop the student flies.
///
/// `hold` = physical ticks per outer decision (the student's action_repeat for a
/// matched outer rate, 1 for the firmware's own attitude rate). The teacher label
/// is computed every tick (the teacher's internal clocks run as in training); the
/// setpoint the inner stage tracks is refreshed on decision ticks only.
#[derive(Clone, Debug)]
pub(crate) struct CtbrOverRate
{
	rate: RateStageFw,
	hold: usize,
	tick: usize,
	held: [f64; CTBR_BANKS],
}

impl CtbrOverRate
{
	pub(crate) fn new(rate: RateStageFw, hold: usize) -> Self
	{
		let mut rate = rate;
		rate.reset();
		CtbrOverRate { rate, hold: hold.max(1), tick: 0, held: [0.0; CTBR_BANKS] }
	}

	pub(crate) fn reset(&mut self)
	{
		self.rate.reset();
		self.tick = 0;
		self.held = [0.0; CTBR_BANKS];
	}

	/// The rate loop as it stands (CTRL-38: the flown integrator a teacher-driven
	/// rollout lends the student's rate-I feature).
	pub(crate) fn rate(&self) -> &RateStageFw
	{
		&self.rate
	}

	/// One physical tick: take the outer label on a decision tick, fly the held one.
	pub(crate) fn fly(&mut self, label: [f64; CTBR_BANKS], gyro: [f32; 3]) -> [f64; 4]
	{
		if self.tick % self.hold == 0
		{
			self.held = label;
		}
		self.tick += 1;
		let h = self.held;
		self.rate.step([h[1], h[2], h[3]], [gyro[0] as f64, gyro[1] as f64, gyro[2] as f64], h[0])
	}
}
