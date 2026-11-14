//! L3 (CTRL-48 block 10, ABI 51, `--alt-integral {off, pi, pole3}`): one integrator
//! of the altitude error per episode, in front of the UNCHANGED altitude PD
//! (docs/ctrl48_block10_levers_spec.md §L3).
//!
//! WHY. The altitude PD (altitude_pd.rs) has no integral, so a constant
//! thrust-to-weight error ε (the per-episode mass draw against the nominal hover)
//! parks the vehicle ε·g/ω² off target for the whole episode — the hold-window dz of
//! docs/ctrl_gap_after_block9.md §C.5.
//!
//! THE LAW. I ← I + e·dt (e = target − z in m, I in m·s). The PD is linear before its
//! clamp, az = ω²·e − 2ζω·vz, so an integral term is a reshaping of its two INPUTS:
//!   * `pi`:    e_eff = e + (k_i/ω²)·I, vz_eff = vz, k_i = ω³/4. The PD's gains are
//!              untouched; Routh needs k_i < 2ζω³, so this sits at 1/8 of the limit
//!              for ζ = 1.
//!   * `pole3`: e_eff = 3e + ω·I, vz_eff = 1.5·vz, i.e. k_p = 3ω², k_d = 3ω, k_i = ω³:
//!              the three closed-loop poles at −ω. Defined for ζ = 1 only.
//!
//! ANTI-WINDUP. (a) |I| ≤ 0.4·g/k_i, so the integral never asks for more than 0.4 g;
//! (b) I holds on a step where the PD's unclamped command is at its limit
//! (|az| ≥ b_z·max_delta, AltitudePd::law).
//!
//! ONE TYPE, EVERY HOST. The classical rivals, the DAgger label (OuterLoops) and the
//! student's `alt_err` observation all read `shape`; the Metal rollout kernel carries
//! the f32 twin (same expressions, same order — `gpu_gains`). f32 throughout because
//! the student's observation is f32 and the kernel is; the rival hosts widen the
//! OUTPUT to f64, exactly as they widened the raw (target − z) before.
//!
//! ORDER WITHIN A STEP. `shape` returns the inputs built from the integral of the
//! PREVIOUS steps' errors, then advances I with this step's error. Off returns its
//! arguments untouched and holds no state: every pre-ABI-51 path is bit-identical.

use crate::altitude_pd::AltitudePd;

/// Share of g the integral term may command (anti-windup clamp (a)).
const INTEGRAL_ACCEL_SHARE: f64 = 0.4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AltIntegralMode
{
	Off,
	Pi,
	Pole3,
}

impl AltIntegralMode
{
	/// The wire value every host carries: 0 off, 1 pi, 2 pole3.
	pub fn from_u8(v: u8) -> Result<Self, String>
	{
		match v
		{
			0 => Ok(AltIntegralMode::Off),
			1 => Ok(AltIntegralMode::Pi),
			2 => Ok(AltIntegralMode::Pole3),
			_ => Err(format!("alt_integral = {v}: expected 0 (off), 1 (pi) or 2 (pole3)")),
		}
	}
}

#[derive(Clone, Copy, Debug)]
pub struct AltitudeIntegrator
{
	mode: AltIntegralMode,
	/// e_eff = k_e·e + k_i·I, vz_eff = k_v·vz.
	k_e: f32,
	k_i: f32,
	k_v: f32,
	/// Clamp on |I| (m·s).
	i_max: f32,
	/// The PD's unclamped law az = w2·e_eff − c·vz_eff and the |az| at which its
	/// collective clamp binds.
	w2: f32,
	c: f32,
	az_limit: f32,
	dt: f32,
	integral: f32,
}

impl AltitudeIntegrator
{
	/// No integrator: `shape` is the identity.
	pub const OFF: AltitudeIntegrator = AltitudeIntegrator {
		mode: AltIntegralMode::Off,
		k_e: 1.0,
		k_i: 0.0,
		k_v: 1.0,
		i_max: 0.0,
		w2: 0.0,
		c: 0.0,
		az_limit: 0.0,
		dt: 0.0,
		integral: 0.0,
	};

	/// Derive the gains from the altitude loop's own shape (`pd.law()`), gravity and
	/// the physical step. Off needs nothing and validates nothing.
	pub fn new(mode: AltIntegralMode, pd: &AltitudePd, gravity: f64, dt: f32) -> Result<Self, String>
	{
		if mode == AltIntegralMode::Off
		{
			return Ok(Self::OFF);
		}
		let (w, zeta, az_limit) = pd.law();
		if !(w > 0.0 && w.is_finite() && gravity > 0.0 && gravity.is_finite() && dt > 0.0 && dt.is_finite())
		{
			return Err(format!(
				"AltitudeIntegrator: omega/gravity/dt must be positive-finite, got {w}/{gravity}/{dt}"
			));
		}
		let (k_e, k_i_gain, k_v, k_i) = match mode
		{
			AltIntegralMode::Pi => (1.0, w / 4.0, 1.0, w * w * w / 4.0),
			_ =>
			{
				if (zeta - 1.0).abs() > 1e-9
				{
					return Err(format!(
						"AltitudeIntegrator: pole3 places three poles at -omega and is defined for \
						 zeta = 1 only, got zeta = {zeta}"
					));
				}
				(3.0, w, 1.5, w * w * w)
			}
		};
		Ok(AltitudeIntegrator {
			mode,
			k_e: k_e as f32,
			k_i: k_i_gain as f32,
			k_v: k_v as f32,
			i_max: (INTEGRAL_ACCEL_SHARE * gravity / k_i) as f32,
			w2: (w * w) as f32,
			c: (2.0 * zeta * w) as f32,
			az_limit: az_limit as f32,
			dt,
			integral: 0.0,
		})
	}

	/// Build from a host's kwargs (the wire mode + the altitude PD's plant and shape).
	/// Mode 0 returns OFF without reading — or validating — anything else, so a host
	/// that never had an altitude PD (the WNN scorers) is untouched at the default.
	#[allow(clippy::too_many_arguments)]
	pub fn from_plant(
		mode: u8,
		mass: f64,
		gravity: f64,
		k_thrust: f64,
		omega_n: f64,
		zeta: f64,
		max_delta: f64,
		dt: f32,
	) -> Result<Self, String>
	{
		let mode = AltIntegralMode::from_u8(mode)?;
		if mode == AltIntegralMode::Off
		{
			return Ok(Self::OFF);
		}
		let pd = AltitudePd::from_plant(mass, gravity, k_thrust, omega_n, zeta, max_delta)
			.map_err(|e| format!("alt_integral needs the altitude PD's plant and shape: {e}"))?;
		Self::new(mode, &pd, gravity, dt)
	}

	pub fn is_on(&self) -> bool
	{
		self.mode != AltIntegralMode::Off
	}

	/// The integral I (m·s) as it stands (after the last `shape`).
	#[cfg(test)]
	pub fn integral(&self) -> f32
	{
		self.integral
	}

	/// [k_e, k_i, k_v, i_max, w2, c, az_limit] for the Metal twin (dt is the kernel's
	/// own P.dt — the same f32). Meaningful only when `is_on`.
	pub fn gpu_gains(&self) -> [f32; 7]
	{
		[self.k_e, self.k_i, self.k_v, self.i_max, self.w2, self.c, self.az_limit]
	}

	/// Episode start.
	pub fn reset(&mut self)
	{
		self.integral = 0.0;
	}

	/// One physical step: (alt_err, vz) → the altitude PD's (alt_err_eff, vz_eff), then
	/// advance I by this step's error. `alt_err` = target − z (m), `vz` = +up (m/s).
	#[inline]
	pub fn shape(&mut self, alt_err: f32, vz: f32) -> (f32, f32)
	{
		if self.mode == AltIntegralMode::Off
		{
			return (alt_err, vz);
		}
		let e_eff = self.k_e * alt_err + self.k_i * self.integral;
		let vz_eff = self.k_v * vz;
		let az = self.w2 * e_eff - self.c * vz_eff;
		if az.abs() < self.az_limit
		{
			self.integral = (self.integral + alt_err * self.dt).clamp(-self.i_max, self.i_max);
		}
		(e_eff, vz_eff)
	}
}
