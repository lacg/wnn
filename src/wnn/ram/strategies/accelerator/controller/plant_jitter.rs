//! CTRL-89 (ABI 70): two per-episode PLANT perturbations the robustness literature has and
//! the sim lacked — body-INERTIA scaling and CONTROL LATENCY — for the combined
//! out-of-distribution ("golden") stress cell and, optionally, the training spread.
//!
//! INERTIA. One multiplicative draw s_e ~ U(1 − j, 1 + j) scales the WHOLE diagonal
//! (Ixx, Iyy, Izz)·s_e — jointly, not per axis:
//!   * Zhang et al. 2022 (docs/papers/zhang2022_single_nearhover_controller.md, Table I) tie
//!     Ixx = Iyy and carry Izz with it; an independent per-axis draw can violate the
//!     rigid-body triangle inequality (Izz ≤ Ixx + Iyy), i.e. fly a body that cannot exist.
//!   * The per-AXIS gain mismatch is already covered by the D7 torque-scale jitter
//!     (dist_torque_scale_jitter, independent per axis, ±0.20 under L4C). A joint scale is
//!     the orthogonal axis: the controller's authority is uniformly off by 1/s_e (with the
//!     gyroscopic term ω×Iω unchanged, since I⁻¹(ω×Iω) is invariant to a joint scale).
//!   * Uniform, like Zhang's ranges and our other ± draws (motor wear, torque scale).
//! Every MODEL stays at the nominal inertia: the teachers' airframe (AirframeRs.inertia), the
//! MPCOF/LQI designs, the cascade d̂ observer constants (CtbrSpec.obs_dhat_plant) — only the
//! plant's `AttitudeSim::inertia` moves (the CTRL-83 D1 rule: AirframeRs::designed).
//! The Metal kernel flies the algebraically identical torque form, τ/s_e on the nominal I
//! (I'⁻¹(τ − ω×I'ω) = I⁻¹(τ/s − ω×Iω) for I' = s·I): a second derivatives() on I' moved the
//! flag-off codegen. CPU == GPU is pinned in ctrl89_plant_jitter_tests.
//!
//! LATENCY. A pure transport delay of n_e physics steps between the controller's command and
//! the plant: the plant at step t receives the command issued at t − n_e (the episode's first
//! command before then — the vehicle is already flying at it, the motor-lag seeding rule).
//! n_e is drawn per episode, uniformly over the INTEGER step counts in
//! [round(lo/dt), round(hi/dt)] (lo, hi in ms; "quantized to the physics step"). It sits in
//! the plant (AttitudeSim::actuate / the kernel's plant block), AFTER every controller-side
//! record of the command (the WNN's pwm_applied and actuator twin, the teachers'
//! last_applied), so it is the same for every controller and no controller is told.
//! Peng et al. 2018 randomize the action timestep per episode for the same reason.
//!
//! THE DRAWS are pure functions of the episode's WEATHER seed (the trainer's ep_seed, the
//! scorers' disturbance_episode_seed(dist_seed, e)) on fresh ram_core::counter_rng layers
//! DIST_CH_INERTIA (9) and DIST_CH_LATENCY (10): no loop rng moves, CPU and Metal read the
//! same host-drawn values, and the WNN and rival scorers draw identically from the same
//! kwargs. j = 0 and hi = 0 is OFF: no setter runs, every rollout is the ABI-68 one.
//!
//! Controller-only physics (not ram_core): nothing here is shared with the IDS worker.

use crate::controller::{disturbance_episode_seed, AttitudeSim, DIST_CH_INERTIA, DIST_CH_LATENCY};

/// The command ring's capacity (physics steps): 32 ms at the 1 kHz plant. The kernel's
/// per-thread ring has the same size (MAX_LAT_STEPS in controller_rollout.metal).
pub const MAX_LATENCY_STEPS: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantJitter
{
	/// Inertia half-width j: s_e ~ U(1 − j, 1 + j); 0 ⇒ the nominal inertia.
	pub inertia_jitter: f32,
	/// Latency range in physics steps (lo <= hi); hi = 0 ⇒ no delay.
	pub latency_lo_steps: u32,
	pub latency_hi_steps: u32,
}

impl PlantJitter
{
	#[cfg_attr(not(test), allow(dead_code))] // the Rust callers that pass OFF are tests
	pub const OFF: PlantJitter = PlantJitter {
		inertia_jitter: 0.0,
		latency_lo_steps: 0,
		latency_hi_steps: 0,
	};

	/// Validated from the kwargs: j in [0, 1), latency (lo, hi) ms with 0 <= lo <= hi and
	/// round(hi/dt) <= MAX_LATENCY_STEPS.
	pub fn new(inertia_jitter: f32, latency_ms: Option<(f32, f32)>, dt: f32) -> Result<Self, String>
	{
		if !(inertia_jitter.is_finite() && (0.0..1.0).contains(&inertia_jitter))
		{
			return Err(format!("inertia_jitter must be in [0, 1), got {inertia_jitter}"));
		}
		let (lo, hi) = match latency_ms
		{
			None => (0, 0),
			Some((lo_ms, hi_ms)) =>
			{
				if !(lo_ms.is_finite() && hi_ms.is_finite() && 0.0 <= lo_ms && lo_ms <= hi_ms)
				{
					return Err(format!("control latency ({lo_ms}, {hi_ms}) ms must be finite with 0 <= lo <= hi"));
				}
				if !(dt.is_finite() && dt > 0.0)
				{
					return Err(format!("control latency needs a physics step dt > 0, got {dt}"));
				}
				let (lo, hi) = (ms_to_steps(lo_ms, dt), ms_to_steps(hi_ms, dt));
				if hi > MAX_LATENCY_STEPS
				{
					return Err(format!(
						"control latency {hi_ms} ms = {hi} steps at dt {dt} s exceeds the ring ({MAX_LATENCY_STEPS} steps)"
					));
				}
				(lo, hi)
			}
		};
		Ok(PlantJitter { inertia_jitter, latency_lo_steps: lo, latency_hi_steps: hi })
	}

	pub fn inertia_on(&self) -> bool
	{
		self.inertia_jitter != 0.0
	}

	pub fn latency_on(&self) -> bool
	{
		self.latency_hi_steps > 0
	}

	pub fn is_off(&self) -> bool
	{
		!self.inertia_on() && !self.latency_on()
	}

	/// What a host must fly for the draws to mean anything: the weather (their seed) and
	/// translation (the scorers carry them in the per-episode Stage1Cfg / buffer-30 record).
	pub fn require_host(&self, translation: bool, dist_enabled: bool, site: &str) -> Result<(), String>
	{
		if self.is_off()
		{
			return Ok(());
		}
		if !translation
		{
			return Err(format!("{site}: inertia_jitter / control latency need translation=True"));
		}
		if !dist_enabled
		{
			return Err(format!("{site}: inertia_jitter / control latency need the weather on (its per-episode seed keys the draws)"));
		}
		Ok(())
	}

	/// Episode `ep_seed`'s inertia scale s_e (exactly 1.0 when off; the D7 expression order).
	pub fn inertia_scale(&self, ep_seed: u64) -> f32
	{
		if !self.inertia_on()
		{
			return 1.0;
		}
		let u = ram_core::counter_rng::uniform(ep_seed, 0, 0, DIST_CH_INERTIA as u64, 0, 0) as f32;
		1.0 - self.inertia_jitter + 2.0 * self.inertia_jitter * u
	}

	/// Episode `ep_seed`'s delay n_e (steps), uniform over the integers in [lo, hi].
	pub fn latency_steps(&self, ep_seed: u64) -> u32
	{
		if !self.latency_on()
		{
			return 0;
		}
		let span = self.latency_hi_steps - self.latency_lo_steps;
		let u = ram_core::counter_rng::uniform(ep_seed, 0, 0, DIST_CH_LATENCY as u64, 0, 0);
		self.latency_lo_steps + ((u * (span + 1) as f64) as u32).min(span)
	}

	/// The trainer hook (arm_cfg_disturbance): this episode's plant. Off calls nothing.
	pub fn arm(&self, sim: &mut AttitudeSim, ep_seed: u64)
	{
		if self.inertia_on()
		{
			sim.set_inertia_scale_core(self.inertia_scale(ep_seed));
		}
		if self.latency_on()
		{
			sim.set_control_latency_core(self.latency_steps(ep_seed));
		}
	}

	/// The scorers' per-episode draws for base weather seed `dist_seed`; OFF ⇒ empty.
	pub fn per_episode(&self, dist_seed: u64, n: usize) -> EpisodePlant
	{
		let seeds = (0..n).map(|e| disturbance_episode_seed(dist_seed, e as u64));
		EpisodePlant {
			inertia_scale: if self.inertia_on() { seeds.clone().map(|s| self.inertia_scale(s)).collect() } else { Vec::new() },
			latency_steps: if self.latency_on() { seeds.map(|s| self.latency_steps(s)).collect() } else { Vec::new() },
		}
	}
}

/// ms → physics steps, rounded (f64, so 10 ms at 1 ms is exactly 10).
fn ms_to_steps(ms: f32, dt: f32) -> u32
{
	((ms as f64) * 1e-3 / (dt as f64)).round() as u32
}

/// The per-episode draws a scorer pool flies (Stage1Cfg.ep_plant). Each Vec is empty
/// (that perturbation off) or one entry per episode; both empty ⇒ the ABI-68 rollout.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EpisodePlant
{
	pub inertia_scale: Vec<f32>,
	pub latency_steps: Vec<u32>,
}

impl EpisodePlant
{
	pub const OFF: EpisodePlant = EpisodePlant { inertia_scale: Vec::new(), latency_steps: Vec::new() };

	pub fn is_on(&self) -> bool
	{
		!self.inertia_scale.is_empty() || !self.latency_steps.is_empty()
	}

	pub fn validate(&self, num_episodes: usize) -> Result<(), String>
	{
		for (name, len) in [("inertia_scale", self.inertia_scale.len()), ("latency_steps", self.latency_steps.len())]
		{
			if len != 0 && len != num_episodes
			{
				return Err(format!("Stage1Cfg.ep_plant.{name} has {len} entries but there are {num_episodes} episodes"));
			}
		}
		if let Some(bad) = self.inertia_scale.iter().find(|s| !s.is_finite() || **s <= 0.0)
		{
			return Err(format!("Stage1Cfg.ep_plant.inertia_scale must be finite and > 0, found {bad}"));
		}
		if let Some(bad) = self.latency_steps.iter().find(|n| **n > MAX_LATENCY_STEPS)
		{
			return Err(format!("Stage1Cfg.ep_plant.latency_steps must be <= {MAX_LATENCY_STEPS}, found {bad}"));
		}
		Ok(())
	}

	/// Episode `ep`'s (inertia scale, latency steps) — (1.0, 0) for an off perturbation.
	pub fn at(&self, ep: usize) -> (f32, u32)
	{
		(self.inertia_scale.get(ep).copied().unwrap_or(1.0), self.latency_steps.get(ep).copied().unwrap_or(0))
	}

	/// Arm episode `ep` on a scorer's sim (after its reset). Off calls nothing.
	pub fn apply(&self, sim: &mut AttitudeSim, ep: usize)
	{
		if let Some(&s) = self.inertia_scale.get(ep)
		{
			sim.set_inertia_scale_core(s);
		}
		if let Some(&n) = self.latency_steps.get(ep)
		{
			sim.set_control_latency_core(n);
		}
	}
}

/// The scorers' entry (WNN CPU / Metal and the classical rivals, one spelling): the kwargs →
/// the validated per-episode draws (EpisodePlant::OFF when both are off).
#[allow(clippy::too_many_arguments)]
pub fn scorer_episode_plant(
	inertia_jitter: f32,
	control_latency_ms: Option<(f32, f32)>,
	dt: f32,
	translation: bool,
	dist_enabled: bool,
	dist_seed: u64,
	num_episodes: usize,
	site: &str,
) -> Result<EpisodePlant, String>
{
	let j = PlantJitter::new(inertia_jitter, control_latency_ms, dt).map_err(|e| format!("{site}: {e}"))?;
	if j.is_off()
	{
		return Ok(EpisodePlant::OFF);
	}
	j.require_host(translation, dist_enabled, site)?;
	Ok(j.per_episode(dist_seed, num_episodes))
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn validates_and_quantizes()
	{
		assert!(PlantJitter::new(-0.1, None, 0.001).is_err());
		assert!(PlantJitter::new(1.0, None, 0.001).is_err());
		assert!(PlantJitter::new(0.2, Some((5.0, 2.0)), 0.001).is_err(), "lo > hi");
		assert!(PlantJitter::new(0.2, Some((-1.0, 2.0)), 0.001).is_err(), "negative");
		assert!(PlantJitter::new(0.0, Some((0.0, 40.0)), 0.001).is_err(), "beyond the ring");
		let j = PlantJitter::new(0.0, Some((8.0, 12.0)), 0.001).unwrap();
		assert_eq!((j.latency_lo_steps, j.latency_hi_steps), (8, 12));
		let j = PlantJitter::new(0.0, Some((8.0, 12.0)), 0.002).unwrap();
		assert_eq!((j.latency_lo_steps, j.latency_hi_steps), (4, 6), "quantized to the 2 ms step");
		assert!(PlantJitter::new(0.0, Some((0.0, 0.0)), 0.001).unwrap().is_off(), "[0, 0] ms is off");
		assert!(PlantJitter::new(0.0, Some((0.0, 0.4)), 0.001).unwrap().is_off(), "rounds to 0 steps: off");
		assert!(PlantJitter::OFF.require_host(false, false, "t").is_ok());
		let on = PlantJitter::new(0.2, None, 0.001).unwrap();
		assert!(on.require_host(false, true, "t").is_err(), "no translation");
		assert!(on.require_host(true, false, "t").is_err(), "no weather");
		assert_eq!(scorer_episode_plant(0.0, None, 0.001, false, false, 1, 8, "t").unwrap(), EpisodePlant::OFF);
		assert_eq!(scorer_episode_plant(0.0, Some((0.0, 0.0)), 0.001, false, false, 1, 8, "t").unwrap(), EpisodePlant::OFF);
	}

	#[test]
	fn draws_cover_the_range_and_are_pure()
	{
		let j = PlantJitter::new(0.30, Some((2.0, 12.0)), 0.001).unwrap();
		let n = 20_000u64;
		let seeds: Vec<u64> = (0..n).map(|s| disturbance_episode_seed(0xC789, s)).collect();
		let sc: Vec<f32> = seeds.iter().map(|&s| j.inertia_scale(s)).collect();
		let (mn, mx) = sc.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
		assert!(mn >= 0.70 && mx <= 1.30 && mn < 0.701 && mx > 1.299, "inertia scale [{mn}, {mx}]");
		let mean = sc.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
		assert!((mean - 1.0).abs() < 0.01, "inertia mean {mean}");
		let mut hist = [0u32; 13];
		for &s in &seeds
		{
			hist[j.latency_steps(s) as usize] += 1;
		}
		assert_eq!(hist[0] + hist[1], 0, "below lo");
		let expect = n as f64 / 11.0;
		for (k, &c) in hist.iter().enumerate().skip(2)
		{
			assert!((c as f64 - expect).abs() < 0.08 * expect, "latency {k} steps: {c} vs {expect}");
		}
		let a = j.per_episode(777, 8);
		assert_eq!(a, j.per_episode(777, 8), "not deterministic");
		assert_eq!(a.inertia_scale[..4], j.per_episode(777, 4).inertia_scale[..], "episode e depends on the pool size");
		assert_ne!(a, j.per_episode(778, 8), "the base seed does not key the draws");
		a.validate(8).unwrap();
		assert!(a.validate(9).is_err());
		eprintln!("[ctrl89-draw] inertia [{mn:.4}, {mx:.4}] mean {mean:.4} latency hist {hist:?}");
	}
}
