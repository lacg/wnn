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

use crate::altitude_integrator::AltitudeIntegrator;
use crate::altitude_pd::AltitudePd;
use crate::att_err_ref::AttErrRef;
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

/// L3 (ABI 51): the episode's altitude integrator for a config — built on the SAME
/// altitude PD the teacher's outer loop uses (alt_pd_for), so the label, the rivals
/// and the student's observation share one law. OFF when cfg.alt_integral == 0 or
/// translation is off (validate_alt_integral refuses the second with a mode set).
pub(crate) fn alt_integrator_for(cfg: &RewardGatedConfigPacked) -> AltitudeIntegrator
{
	let (Some(pd), Ok(mode)) = (alt_pd_for(cfg), crate::altitude_integrator::AltIntegralMode::from_u8(cfg.alt_integral))
	else
	{
		return AltitudeIntegrator::OFF;
	};
	AltitudeIntegrator::new(mode, &pd, cfg.af_gravity as f64, cfg.dt as f32)
		.expect("alt_integral validated at construction (validate_alt_integral)")
}

/// L3: THIS step's altitude-law inputs (alt_err_eff, vz_eff), at the start-of-step
/// snapshot — the ONE read that feeds both the student's `alt_err` observation
/// (feed_live_obs) and the teacher's altitude PD (OuterLoops::expert / expert_ctbr).
/// Advances the episode's integrator: call exactly once per physical step, before
/// either consumer. With the integrator off this is the raw (target − z, vz) every
/// host computed before ABI 51, bit-for-bit.
#[inline]
pub(crate) fn sense_altitude(alt_int: &mut AltitudeIntegrator, sim: &AttitudeSim, target_altitude: f32) -> (f32, f32)
{
	alt_int.shape(target_altitude - sim.altitude_rs(), sim.vertical_velocity_rs())
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
	// STAGE 2: displaced, at rest unless init_vxy > 0. Gated on xy_offset > 0 so a
	// stage-1 config draws nothing here and its rng sequence is untouched.
	if cfg.xy_offset > 0.0
	{
		let x0 = jitter_sym(rng, cfg.xy_offset);
		let y0 = jitter_sym(rng, cfg.xy_offset);
		// ABI 48: initial horizontal velocity (training-only, idea #4). Drawn AFTER
		// x0/y0; init_vxy == 0 draws nothing, so the ABI-47 sequence is intact.
		let vx0 = jitter_sym(rng, cfg.init_vxy);
		let vy0 = jitter_sym(rng, cfg.init_vxy);
		sim.set_horizontal_state(x0, y0, vx0, vy0);
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
/// `alt_err` is this step's `sense_altitude(..).0` — the altitude error the altitude
/// law reads (L3: reshaped by the integrator when on, target − z otherwise). The
/// vz observation stays the raw velocity.
pub(crate) fn feed_live_obs(
	controller: &mut WnnController,
	sim: &AttitudeSim,
	cfg: &RewardGatedConfigPacked,
	collective_pwm: f32,
	alt_err: f32,
)
{
	if !cfg.translation
	{
		return;
	}
	controller.set_vertical_obs(collective_pwm, alt_err, sim.vertical_velocity_rs());
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
	/// `alt_law` = this step's `sense_altitude` (alt_err_eff, vz_eff).
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn expert(
		&self,
		teacher: &mut Teacher,
		sim: &AttitudeSim,
		q: [f32; 4],
		gyro: [f32; 3],
		target: [f32; 3],
		alt_law: (f32, f32),
	) -> [f64; 4]
	{
		let alt_err = alt_law.0 as f64;
		let vz = alt_law.1 as f64;
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

	/// ABI 38 (CTRL-27): the CTBR twin of `expert` — the same outer loops (position
	/// → tilt reference, altitude PD → collective deviation), with the attitude
	/// teacher's OUTER decision as the label instead of its motor command:
	/// [collective deviation, p, q, r] (Teacher::ctbr_label). `outer_steps` is the
	/// student's outer period in physical ticks (its action_repeat).
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn expert_ctbr(
		&self,
		teacher: &mut Teacher,
		sim: &AttitudeSim,
		q: [f32; 4],
		gyro: [f32; 3],
		target: [f32; 3],
		alt_law: (f32, f32),
		outer_steps: usize,
	) -> [f64; 4]
	{
		let alt_err = alt_law.0 as f64;
		let vz = alt_law.1 as f64;
		let coll = self.alt_pd.as_ref().map_or(0.0, |pd| pd.delta(alt_err, vz));
		let tgt = match self.pos.as_ref()
		{
			// The full cascade: step_full_state's tilt reference, verbatim.
			Some(pl) if self.alt_pd.is_some() =>
			{
				let [hx, hy] = sim.position_xy_rs();
				let [hvx, hvy] = sim.velocity_xy_rs();
				let (roll_ref, pitch_ref) = pl.tilt_ref((-hx) as f64, hvx as f64, (-hy) as f64, hvy as f64);
				[roll_ref as f32, pitch_ref as f32, target[2]]
			}
			_ => target,
		};
		teacher
			.ctbr_label(q, gyro, tgt, outer_steps, coll)
			.expect("CTBR teacher validated before the first episode (validate_train_regime)")
	}
}

/// ABI 38 (CTRL-27): what a training-cascade rollout FLIES when the teacher drives
/// it — expert_drives (pure BC), the threshold-calibration sampler and the MEMORY
/// recorder. On the motor path, the teacher's own motor command (`expert`). For a
/// CTBR student, the teacher's CTBR label flown through the student's OWN inner
/// stage at the student's outer rate (ctbr::CtbrOverRate) — so the cascade the
/// student is calibrated, recorded and cloned on is the one it flies.
pub(crate) struct ExpertPilot
{
	ctbr: Option<(crate::ctbr::CtbrOverRate, usize)>,
}

impl ExpertPilot
{
	pub(crate) fn for_controller(controller: &WnnController) -> Self
	{
		ExpertPilot {
			ctbr: controller.ctbr_ref().map(|c| {
				let hold = controller.action_repeat_n().max(1);
				(crate::ctbr::CtbrOverRate::new(c.spec.rate_stage(), hold), hold)
			}),
		}
	}

	/// Episode start (the inner stage forgets its integrals, filters and clocks).
	pub(crate) fn reset(&mut self)
	{
		if let Some((p, _)) = self.ctbr.as_mut()
		{
			p.reset();
		}
	}

	/// CTRL-38 a1: before the student's step(), hand it the rate loop that is
	/// FLYING the vehicle (this pilot's), so its rate-I feature reads the vehicle's
	/// integrator — what it reads from its own loop when it flies (deploy). Called
	/// by every host where the teacher drives (calibration, recorder,
	/// expert_drives). No-op on the motor path and with obs_rate_i off.
	pub(crate) fn lend_flown_rate_stage(&self, controller: &mut WnnController)
	{
		if let Some((p, _)) = self.ctbr.as_ref()
		{
			controller.ctbr_adopt_flown_rate_stage(p.rate());
		}
	}

	/// The teacher's LABEL for this step (motor command, or CTBR bank vector) and
	/// the pwm the teacher's cascade puts on the motors.
	#[allow(clippy::too_many_arguments)]
	pub(crate) fn command(
		&mut self,
		loops: &OuterLoops,
		teacher: &mut Teacher,
		sim: &AttitudeSim,
		q: [f32; 4],
		gyro: [f32; 3],
		target: [f32; 3],
		alt_law: (f32, f32),
	) -> ([f64; 4], [f64; 4])
	{
		match self.ctbr.as_mut()
		{
			Some((pilot, hold)) =>
			{
				let label = loops.expert_ctbr(teacher, sim, q, gyro, target, alt_law, *hold);
				(label, pilot.fly(label, gyro))
			}
			None =>
			{
				let pwm = loops.expert(teacher, sim, q, gyro, target, alt_law);
				(pwm, pwm)
			}
		}
	}
}

/// ABI 38: the refusals a CTBR student adds to a training config — run BEFORE the
/// first episode so no rollout can hit an undefined label mid-run.
///   * every teacher the config can select must define a rate-level decision
///     (Teacher::supports_ctbr: the firmware PID cascade, or mpcof);
///   * the student's inner stage must BE the airframe's firmware rate loop — the
///     config's own af_pid_* cascade (gains, saturation, hover, k_thrust, rate,
///     filter): a student trained over one rate loop and labelled by a teacher
///     whose cascade is another would imitate a controller it does not fly.
pub(crate) fn validate_ctbr_regime(controller: &WnnController, cfg: &RewardGatedConfigPacked) -> Result<(), String>
{
	let Some(c) = controller.ctbr_ref()
	else
	{
		return Ok(());
	};
	let s = &c.spec;
	let main_hz = (1.0 / cfg.dt.max(1e-9)).round() as u32;
	let same = s.rate_gains == cfg.af_pid_rate
		&& s.out_limit_n == cfg.af_pid_out_limit_n
		&& s.hover_n == cfg.af_pid_hover_n
		// the packed config carries k_thrust as f32 (the plant's own precision)
		&& s.k_thrust as f32 == cfg.af_k_thrust
		&& s.attitude_hz == cfg.af_pid_attitude_hz
		&& s.lpf_hz == cfg.af_pid_lpf_hz
		&& s.main_loop_hz == main_hz;
	if !same
	{
		return Err(format!(
			"CTBR: the student's inner rate loop ({}) is not the training config's firmware cascade \
			 (af_pid_rate/out_limit/hover_n/k_thrust/attitude_hz/lpf_hz, main loop {main_hz} Hz) — build the \
			 CtbrSpec from the SAME airframe cascade the config carries",
			s.provenance_line()
		));
	}
	// CTRL-48: the residual block's observer must fly the same airframe too.
	if let Some(p) = s.obs_dhat_plant.or(s.obs_dhat_residual_plant)
	{
		let cfg_plant = [
			cfg.af_arm_length as f64,
			cfg.af_k_drag as f64,
			cfg.af_inertia[0] as f64,
			cfg.af_inertia[1] as f64,
			cfg.af_inertia[2] as f64,
		];
		// f32 compare: the packed config carries the plant at f32 precision.
		if p.map(|v| v as f32) != cfg_plant.map(|v| v as f32)
		{
			return Err(format!(
				"CTBR: the student's d̂ observer plant {p:?} [arm, k_drag, Ixx, Iyy, Izz] is not the training \
				 config's airframe {cfg_plant:?} — the student would estimate a vehicle it is not flying"
			));
		}
	}
	let af = crate::dagger_train::AirframeRs::from_cfg(cfg);
	let mut ids: Vec<u8> = vec![cfg.teacher];
	ids.extend(cfg.teacher_schedule.iter().copied());
	ids.extend(cfg.teacher_blend.iter().copied());
	ids.sort_unstable();
	ids.dedup();
	for id in ids
	{
		if !af.teacher(id).supports_ctbr()
		{
			return Err(format!(
				"CTBR: teacher id {id} has no rate-level decision — the cascade's outer teachers are \
				 0 (the firmware attitude PID, needs af_pid_* cascade gains) and 4 (mpcof)"
			));
		}
	}
	Ok(())
}

/// G5 (ABI 32): the attitude-error reference a config scores against — the
/// position loop's tilt reference (the teacher's own loop, pos_loop_for) when
/// cfg.att_err_tilt_ref, else LEVEL. validate_abi32 guarantees the horizontal
/// channel is armed whenever the flag is on.
pub(crate) fn att_err_ref_for(cfg: &RewardGatedConfigPacked) -> AttErrRef
{
	if !cfg.att_err_tilt_ref
	{
		return AttErrRef::LEVEL;
	}
	AttErrRef::tilt(
		cfg.af_gravity as f64,
		cfg.pos_omega as f64,
		cfg.pos_zeta as f64,
		cfg.pos_max_tilt_rad as f64,
	)
	.expect("att_err_tilt_ref: the position-loop shape must derive from the config's plant")
}

/// G5: the per-step attitude error (rad) the gate, checkpoint and L4 priority
/// read. LEVEL is the legacy `sim.attitude_error(None)`, bit-for-bit.
#[inline]
pub(crate) fn scored_att_err(sim: &AttitudeSim, cfg: &RewardGatedConfigPacked) -> f32
{
	if cfg.att_err_tilt_ref
	{
		att_err_ref_for(cfg).error(sim)
	}
	else
	{
		sim.attitude_error(None)
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
