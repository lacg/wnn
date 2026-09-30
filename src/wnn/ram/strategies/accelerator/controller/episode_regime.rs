//! The per-episode REGIME every trainer-side rollout flies (CTRL-17, 26/09/2026).
//!
//! WHY THIS FILE EXISTS. The DAgger training rollout (`rollout_and_label_rs`) was
//! the only rollout that set up a translating episode correctly: the mass draw,
//! the vertical and horizontal starts, the collective anchor, the live per-step
//! observations and the full outer-loop cascade on the teacher. The three rollouts
//! AROUND it — the per-round checkpoint eval, the threshold-calibration sampler and
//! the MEMORY address recorder — each carried their own partial copy (or none), so
//! they flew a different world than the one the student is trained and scored in
//! (docs/ctrl17_stage2_trainer_audit.md G1-G4). Every one of them now calls these
//! helpers, so there is ONE definition of "a training episode" and a new channel
//! cannot reach one rollout and miss the others.
//!
//! Every helper is inert when `cfg.translation` is off: no rng draw, no call on
//! the sim or controller. Attitude-only runs are therefore byte-identical.

use crate::altitude_pd::AltitudePd;
use crate::controller::{compute_reward, compute_reward_stage2, AttitudeSim, WnnController};
use crate::dagger_train::RewardGatedConfigPacked;
use crate::optimal::Teacher;
use crate::position_loop::PositionLoop;
use rand::rngs::SmallRng;
use rand::Rng;

/// A symmetric jitter draw, U(-mag, +mag). mag == 0 draws NOTHING from the rng —
/// that is what keeps a translation-off run's random sequence untouched.
#[inline]
pub(crate) fn jitter_sym(rng: &mut SmallRng, mag: f32) -> f32
{
	if mag == 0.0
	{
		return 0.0;
	}
	rng.gen_range(-mag..mag)
}

/// The teacher's outer position loop. None unless the config arms the horizontal
/// channel (translation && xy_offset > 0), keeping every stage-1 teacher identical.
pub(crate) fn pos_loop_for(cfg: &RewardGatedConfigPacked) -> Option<PositionLoop>
{
	if !cfg.translation || cfg.xy_offset <= 0.0
	{
		return None;
	}
	Some(
		PositionLoop::from_plant(
			cfg.af_gravity as f64,
			cfg.pos_omega as f64,
			cfg.pos_zeta as f64,
			cfg.pos_max_tilt_rad as f64,
		)
		.expect("stage-2 position loop must derive from the config's plant"),
	)
}

/// The teacher's outer altitude PD, derived from the config's own plant (never
/// guessed). None when translation is off.
pub(crate) fn alt_pd_for(cfg: &RewardGatedConfigPacked) -> Option<AltitudePd>
{
	if !cfg.translation
	{
		return None;
	}
	AltitudePd::from_plant(
		cfg.af_mass as f64,
		cfg.af_gravity as f64,
		cfg.af_k_thrust as f64,
		cfg.alt_pd_omega as f64,
		cfg.alt_pd_zeta as f64,
		cfg.alt_pd_max_delta as f64,
	)
	.ok()
}

/// This episode's translation PLANT draw and starts, taken from the loop rng in the
/// order the trainer has always used (mass, z0, vz0, [x0, y0], collective jitter).
/// Call AFTER `sim.reset` (reset zeroes the translational state). Returns the
/// episode's commanded collective in ABSOLUTE pwm — 0.0 when translation is off,
/// in which case nothing is drawn and the sim is untouched.
pub(crate) fn draw_translation_episode(
	sim: &mut AttitudeSim,
	cfg: &RewardGatedConfigPacked,
	rng: &mut SmallRng,
) -> f32
{
	if !cfg.translation
	{
		return 0.0;
	}
	let m = cfg.af_mass * (1.0 + jitter_sym(rng, cfg.mass_jitter));
	sim
		.set_translation_core(m)
		.expect("stage-1 mass must be positive");
	sim.set_vertical_state(
		jitter_sym(rng, cfg.alt_offset),
		jitter_sym(rng, cfg.init_vz),
	);
	// STAGE 2: displaced at rest. Gated on xy_offset > 0 so a stage-1 config draws
	// nothing here and its rng sequence is untouched.
	if cfg.xy_offset > 0.0
	{
		let x0 = jitter_sym(rng, cfg.xy_offset);
		let y0 = jitter_sym(rng, cfg.xy_offset);
		sim.set_horizontal_state(x0, y0, 0.0, 0.0);
	}
	let hover = (m * cfg.af_gravity / (4.0 * cfg.af_k_thrust)).sqrt();
	(hover * (1.0 + jitter_sym(rng, cfg.collective_jitter))).clamp(0.0, 1.0)
}

/// Anchor the delta accumulator at the episode's commanded collective. Call AFTER
/// `controller.reset` (reset seeds the accumulators from the current anchor).
pub(crate) fn anchor_controller(
	controller: &mut WnnController,
	cfg: &RewardGatedConfigPacked,
	collective_pwm: f32,
)
{
	if cfg.translation
	{
		controller.set_collective_anchor(collective_pwm);
	}
}

/// Hand the controller THIS step's live vertical and horizontal observation, read
/// at the same start-of-step snapshot as the IMU — the exact twin of the scorers.
/// Target is the origin at `target_altitude`, so the errors are target − state.
pub(crate) fn feed_live_obs(
	controller: &mut WnnController,
	sim: &AttitudeSim,
	cfg: &RewardGatedConfigPacked,
	collective_pwm: f32,
)
{
	if !cfg.translation
	{
		return;
	}
	controller.set_vertical_obs(
		collective_pwm,
		cfg.target_altitude - sim.altitude_rs(),
		sim.vertical_velocity_rs(),
	);
	let [hx, hy] = sim.position_xy_rs();
	let [hvx, hvy] = sim.velocity_xy_rs();
	controller.set_horizontal_obs(-hx, -hy, hvx, hvy);
}

/// The teacher's outer loops for a config: the altitude PD under translation and
/// the position loop when the horizontal channel is armed.
pub(crate) struct OuterLoops
{
	alt_pd: Option<AltitudePd>,
	pos: Option<PositionLoop>,
}

impl OuterLoops
{
	pub(crate) fn for_cfg(cfg: &RewardGatedConfigPacked) -> Self
	{
		OuterLoops {
			alt_pd: alt_pd_for(cfg),
			pos: pos_loop_for(cfg),
		}
	}

	/// The teacher's command at the current sim state: the full disclosed cascade
	/// (position → tilt ref → attitude law, collective on top) when both loops
	/// exist, the stage-1 cascade with only the altitude PD, else the attitude law.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn expert(
		&self,
		teacher: &mut Teacher,
		sim: &AttitudeSim,
		q: [f32; 4],
		gyro: [f32; 3],
		target: [f32; 3],
		target_altitude: f32,
	) -> [f64; 4]
	{
		let alt_err = (target_altitude - sim.altitude_rs()) as f64;
		let vz = sim.vertical_velocity_rs() as f64;
		match (self.pos.as_ref(), self.alt_pd.as_ref())
		{
			(Some(pl), Some(pd)) =>
			{
				let [hx, hy] = sim.position_xy_rs();
				let [hvx, hvy] = sim.velocity_xy_rs();
				teacher.step_full_state(
					q,
					gyro,
					target[2],
					pl,
					pd,
					(-hx) as f64,
					hvx as f64,
					(-hy) as f64,
					hvy as f64,
					alt_err,
					vz,
				)
			}
			(None, Some(pd)) => teacher.step_with_collective(q, gyro, target, pd, alt_err, vz),
			_ => teacher.step_rs(q, gyro, target),
		}
	}
}

/// G4: the per-step reward the DAgger gate AND the per-round checkpoint rank on.
/// Attitude² plus λ_alt·alt_err² plus λ_pos·(e_x²+e_y²) under translation; the
/// attitude-only reward otherwise. compute_reward_stage2 short-circuits each term
/// at λ = 0, so a run with both lambdas 0 is bit-identical to the legacy reward.
/// Call on the POST-step sim state (the reward the legacy loop computed).
pub(crate) fn step_reward(sim: &AttitudeSim, cfg: &RewardGatedConfigPacked, attitude_err: f32) -> f64
{
	if !cfg.translation
	{
		return compute_reward(attitude_err, 0.0, 0, 0.0, 0.0) as f64;
	}
	let [hx, hy] = sim.position_xy_rs();
	compute_reward_stage2(
		attitude_err,
		0.0,
		0,
		0.0,
		0.0,
		cfg.target_altitude - sim.altitude_rs(),
		cfg.lambda_alt,
		// target = origin ⇒ err = −pos; squared, so the sign is moot
		hx,
		hy,
		cfg.lambda_pos,
	) as f64
}
