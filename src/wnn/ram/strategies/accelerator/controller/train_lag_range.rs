//! CTRL-87 (ABI 68): the per-episode TRAINING motor-lag spread (prereg
//! docs/ctrl87_nominal_mass_wide_spread_prereg.md, Amendment 1).
//!
//! The plant's motor lag is otherwise ONE number (`motor_lag_s`, the 2% settling time T)
//! for every rollout. With a range [lo, hi] armed, each TRAINING episode — DAgger rollouts,
//! the per-round eval, the calibration sampler, the MEMORY recorder (all through
//! `dagger_train::arm_cfg_disturbance`) — and each GA-search scoring episode (both WNN
//! scorers, `scorer_episode_lags`) flies its own plant lag T_e ~ U(lo, hi), while every
//! MODEL of the actuator stays at the nominal design lag `motor_lag_s`: the WNN's actuator
//! twin (d̂ / obs_pwm / cascade d̂ — armed once from the nominal plant before any episode
//! draws) and the teachers (built from the nominal AirframeRs). The TEST units, the VAL
//! stage-select and the report scorer never pass a range: they fly the fixed preset lag.
//!
//! THE DRAW. T_e = lo + (hi − lo)·u, u = ram_core::counter_rng::uniform(ep_seed, 0, 0,
//! DIST_CH_MOTOR_LAG, 0, 0), keyed on the episode's WEATHER seed — the trainer's per-episode
//! ep_seed, the scorers' disturbance_episode_seed(dist_seed, e) — so it is a pure function of
//! (seed, episode), never touches a loop rng (the draw order of everything else is unchanged),
//! and is computed ONCE on the host for both scorers: the CPU rollout and the Metal kernel
//! (buffer 30, 7th float) read the same f32 per episode.
//!
//! ABI 69 (CTRL-83 cell `all`, Luiz 28/11): the same draw is also a REPORT-time stress — the
//! joint-OOD stress cell passes the range to both WNN scorers AND score_classical_baseline (the
//! rivals, the row-B probe), so every controller flies the identical per-episode plant lag.
//! Outside that cell the TEST/VAL/report sentence below still holds. lo == hi == motor_lag_s is the fixed
//! lag bit for bit (pinned by ctrl87_train_spread_tests).
//!
//! Scope: a symmetric, floor-free actuator (the cf21 lineage); the CTRL-86 extended actuator
//! has no single T to spread and is refused, like DesignModel. Needs the weather (the draw's
//! seed) and translation (the per-episode plant blob). Controller-only physics: nothing here
//! is shared with the IDS worker (not ram_core).

use crate::actuator_spec::ActuatorSpec;
use crate::controller::{disturbance_episode_seed, DIST_CH_MOTOR_LAG};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainLagRange
{
	/// Lower / upper 2% settling time T (s), lo <= hi.
	pub lo_s: f32,
	pub hi_s: f32,
}

impl TrainLagRange
{
	pub fn new(lo_s: f32, hi_s: f32) -> Result<Self, String>
	{
		if !(lo_s.is_finite() && hi_s.is_finite() && lo_s > 0.0 && lo_s <= hi_s)
		{
			return Err(format!(
				"train_motor_lag_range = ({lo_s}, {hi_s}) must be finite settling times with 0 < lo <= hi (s)"
			));
		}
		Ok(TrainLagRange { lo_s, hi_s })
	}

	pub fn from_kwarg(range: Option<(f32, f32)>) -> Result<Option<Self>, String>
	{
		range.map(|(lo, hi)| Self::new(lo, hi)).transpose()
	}

	/// What a host must fly for the range to mean anything: a lag-on, symmetric, floor-free
	/// nominal actuator (the design value the twins keep), the weather (the draw's seed) and
	/// translation (the scorers carry the drawn T in the per-episode plant blob).
	pub fn require_host(&self, nominal: ActuatorSpec, translation: bool, dist_enabled: bool, site: &str) -> Result<(), String>
	{
		if !nominal.lag_on()
		{
			return Err(format!("{site}: train_motor_lag_range needs motor_lag_s > 0 (the nominal design lag the twins keep)"));
		}
		if nominal.extended()
		{
			return Err(format!(
				"{site}: train_motor_lag_range needs a symmetric, floor-free actuator (motor_lag_down_s / pwm_min are set)"
			));
		}
		if !translation
		{
			return Err(format!("{site}: train_motor_lag_range needs translation=True"));
		}
		if !dist_enabled
		{
			return Err(format!("{site}: train_motor_lag_range needs the weather on (its per-episode seed keys the draw)"));
		}
		Ok(())
	}

	/// The plant lag of the episode whose weather seed is `ep_seed`.
	pub fn draw(&self, ep_seed: u64) -> f32
	{
		let u = ram_core::counter_rng::uniform(ep_seed, 0, 0, DIST_CH_MOTOR_LAG as u64, 0, 0) as f32;
		self.lo_s + (self.hi_s - self.lo_s) * u
	}

	/// The scorers' per-episode lags for base weather seed `dist_seed` (episode e's weather
	/// seed is disturbance_episode_seed(dist_seed, e), the CPU scorer's and the kernel's).
	pub fn per_episode(&self, dist_seed: u64, n: usize) -> Vec<f32>
	{
		(0..n)
			.map(|e| self.draw(disturbance_episode_seed(dist_seed, e as u64)))
			.collect()
	}
}

/// The WNN scorers' entry (CPU and Metal, one spelling): the kwarg → the per-episode plant
/// lags for the Stage1Cfg (empty = off, every existing call unchanged), validated.
pub fn scorer_episode_lags(
	range: Option<(f32, f32)>,
	nominal: ActuatorSpec,
	translation: bool,
	dist_enabled: bool,
	dist_seed: u64,
	num_episodes: usize,
	site: &str,
) -> Result<Vec<f32>, String>
{
	let Some(r) = TrainLagRange::from_kwarg(range).map_err(|e| format!("{site}: {e}"))?
	else
	{
		return Ok(Vec::new());
	};
	r.require_host(nominal, translation, dist_enabled, site)?;
	Ok(r.per_episode(dist_seed, num_episodes))
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn validates_and_refuses_hosts()
	{
		assert!(TrainLagRange::new(0.0, 0.3).is_err());
		assert!(TrainLagRange::new(0.3, 0.2).is_err());
		assert!(TrainLagRange::new(0.15, f32::NAN).is_err());
		assert_eq!(TrainLagRange::from_kwarg(None).unwrap(), None);
		let r = TrainLagRange::new(0.15, 0.30).unwrap();
		let sym = ActuatorSpec::symmetric(0.20);
		assert!(r.require_host(sym, true, true, "t").is_ok());
		assert!(r.require_host(ActuatorSpec::OFF, true, true, "t").is_err(), "lag off");
		assert!(r.require_host(ActuatorSpec::new(0.2, 0.3, 0.0).unwrap(), true, true, "t").is_err(), "extended");
		assert!(r.require_host(sym, false, true, "t").is_err(), "no translation");
		assert!(r.require_host(sym, true, false, "t").is_err(), "no weather");
		assert_eq!(scorer_episode_lags(None, ActuatorSpec::OFF, false, false, 1, 8, "t").unwrap(), Vec::<f32>::new());
	}

	#[test]
	fn degenerate_range_is_the_fixed_lag()
	{
		let r = TrainLagRange::new(0.20, 0.20).unwrap();
		assert!(r.per_episode(777, 64).iter().all(|t| t.to_bits() == 0.20f32.to_bits()));
	}
}
