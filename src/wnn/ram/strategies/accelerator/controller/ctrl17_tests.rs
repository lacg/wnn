//! CTRL-17 behaviour tests (26/09/2026, docs/ctrl17_stage2_trainer_audit.md).
//!
//! Each test names the gap it pins. The byte-identity side (attitude-only and the
//! training rollout unchanged) lives in ctrl17_pins.rs.

use crate::controller::{AttitudePidRs, AttitudeSim, WnnController};
use crate::ctrl17_pins::{pin_student, pin_translation_cfg};
use crate::dagger_train::{
	eval_closed_loop_rs, rollout_and_label_rs, unmix_motors_to_controls,
	AirframeRs, RewardGatedConfigPacked, TrajectoryRs,
};
use crate::record_ops::{record_address_universe, CascadeDriver, Driver, RecorderStage1};
use crate::stage1::Stage1Cfg;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// The stage-2 cf21 config with weather off and a level, at-rest attitude start —
/// so every lateral command comes from the position loop.
fn quiet_stage2_cfg(steps: usize) -> RewardGatedConfigPacked
{
	let mut c = pin_translation_cfg(0.5);
	c.dist_enabled = false;
	c.steps_per_episode = steps;
	c.max_initial_yaw_rad = 0.0;
	c.max_initial_body_rate = 0.0;
	c.max_initial_yaw_rate = 0.0;
	c.full_tilt_deg = 0.0;
	c.easy_tilt_deg = 0.0;
	c
}

/// One rollout of `cfg` from rng seed `seed` (fresh student + teacher + sim).
fn one_rollout(cfg: &RewardGatedConfigPacked, seed: u64) -> (TrajectoryRs, AttitudeSim)
{
	let mut c = pin_student(true, true);
	let af = AirframeRs::from_cfg(cfg);
	let mut teacher = af.teacher(cfg.teacher);
	let mut sim = af.sim();
	let mut rng = SmallRng::seed_from_u64(seed);
	let t = rollout_and_label_rs(&mut c, &mut teacher, &mut sim, cfg, 0.0, &mut rng, [0.0; 3]);
	(t, sim)
}

/// A rollout seed whose x0 draw is at least 0.2 m away from the origin.
fn seed_with_offset(cfg: &RewardGatedConfigPacked) -> (u64, TrajectoryRs, AttitudeSim)
{
	for seed in 1..64u64
	{
		let (t, sim) = one_rollout(cfg, seed);
		if t.horiz_obs[0][0].abs() > 0.2
		{
			return (seed, t, sim);
		}
	}
	panic!("no seed in 1..64 drew |x0| > 0.2 m at xy_offset 0.5");
}

/// G11: the TRAINER's lateral teacher (rollout_and_label_rs with xy_offset > 0),
/// which no test exercised. Under expert_drives the teacher flies home: the early
/// pitch torque opposes x0 (+pitch accelerates +x), the recorded horizontal
/// observation is live, x ends well inside x0, and altitude holds.
///
/// teacher_hover_mode = 0 (no D0 re-base): the D0 variant of expert_drives is
/// pinned separately by g13_expert_drives_flies_raw_teacher_pwm_under_d0.
#[test]
fn g11_trainer_lateral_teacher_flies_home()
{
	let mut cfg = quiet_stage2_cfg(2000);
	cfg.expert_drives = true;
	cfg.teacher_hover_mode = 0;
	let (_seed, t, sim) = seed_with_offset(&cfg);
	let x0 = -t.horiz_obs[0][0];
	assert!(t.steps == 2000, "the teacher-driven episode must not diverge ({} steps)", t.steps);
	assert!(t.horiz_obs.iter().any(|h| h[2] != 0.0), "v_x must be observed live, not held at 0");
	let tau_pitch: f32 = t.pid_pwms[..50].iter().map(|p| unmix_motors_to_controls(*p)[2]).sum::<f32>() / 50.0;
	assert!(
		tau_pitch.signum() == -x0.signum() && tau_pitch.abs() > 1e-6,
		"early pitch torque {tau_pitch} must oppose x0 {x0}"
	);
	let x_end = sim.position_xy_rs()[0];
	assert!(
		x_end.abs() < 0.7 * x0.abs(),
		"the lateral teacher must fly toward the origin: x0 {x0} -> x_end {x_end}"
	);
	assert!(sim.altitude_rs().abs() < 0.15, "the cascade must hold altitude ({})", sim.altitude_rs());
}

/// G13: expert_drives + D0 derived hover + translation + a delta student. The
/// LABEL is re-based on the teacher's hover (≈ neutral 0.5), but the SIM must fly
/// the teacher's raw pwm (hover 0.694 on cf21). Before the fix the sim flew the
/// label and the vehicle fell ~1.2 m in 2 s.
#[test]
fn g13_expert_drives_flies_raw_teacher_pwm_under_d0()
{
	let mut cfg = quiet_stage2_cfg(2000);
	cfg.expert_drives = true;
	cfg.teacher_hover_mode = crate::dagger_train::TEACHER_HOVER_DERIVED;
	let (_seed, t, sim) = seed_with_offset(&cfg);
	assert_eq!(t.steps, 2000, "the teacher-driven episode must not diverge");
	assert!(
		sim.altitude_rs().abs() < 0.15,
		"expert_drives must fly the teacher's pwm, not the re-based label (alt {})",
		sim.altitude_rs()
	);
	// The label stays re-based (≈ neutral at hover) — only the APPLIED action changes.
	let mean_label: f32 = t.pid_pwms[1999].iter().sum::<f32>() / 4.0;
	let mean_applied: f32 = t.student_pwms[1999].iter().sum::<f32>() / 4.0;
	assert!((mean_label - 0.5).abs() < 0.1, "label must stay re-based ({mean_label})");
	assert!((mean_applied - 0.694).abs() < 0.1, "applied must be the teacher's hover ({mean_applied})");
}

/// G4: the rollout's gate score carries λ_alt·alt² and λ_pos·radial². The labels
/// (hence training) do not depend on λ; the score strictly drops when a λ is on.
#[test]
fn g4_gate_score_reads_lambda_alt_and_lambda_pos()
{
	let base = quiet_stage2_cfg(400);
	let (seed, t0, _) = seed_with_offset(&base);
	let mut pos = base.clone();
	pos.lambda_pos = 1.0;
	let mut alt = base.clone();
	alt.lambda_alt = 1.0;
	let (tp, _) = one_rollout(&pos, seed);
	let (ta, _) = one_rollout(&alt, seed);
	assert_eq!(t0.pid_pwms, tp.pid_pwms, "λ must not change the episode or its labels");
	assert!(tp.cumulative_reward < t0.cumulative_reward, "λ_pos must lower the gate score");
	assert!(ta.cumulative_reward < t0.cumulative_reward, "λ_alt must lower the gate score");
}

/// The per-round eval for `cfg` from a fixed rng seed.
fn eval_once(c: &mut WnnController, cfg: &RewardGatedConfigPacked, seed: u64) -> (f64, f64, f64, f64, f64)
{
	let mut sim = AirframeRs::from_cfg(cfg).sim();
	let mut rng = SmallRng::seed_from_u64(seed);
	let lvls = c.levels_per_motor();
	eval_closed_loop_rs(c, &mut sim, cfg, &mut rng, [0.0; 3], 0.05, 4, lvls)
}

/// G3: the checkpoint eval draws the training starts (off-origin xy, alt offset)
/// and ranks on the weighted reward — the λ terms move its fitness, and only it
/// (err/stable are attitude metrics and do not change).
#[test]
fn g3_eval_draws_training_starts_and_ranks_weighted()
{
	let mut base = quiet_stage2_cfg(300);
	base.eval_episodes = 3;
	let mut pos = base.clone();
	pos.lambda_pos = 1.0;
	let mut alt = base.clone();
	alt.lambda_alt = 1.0;
	let mut c = pin_student(true, true);
	let r0 = eval_once(&mut c, &base, 5);
	let rp = eval_once(&mut c, &pos, 5);
	let ra = eval_once(&mut c, &alt, 5);
	assert!(rp.0 < r0.0, "an off-origin start must cost λ_pos reward ({} vs {})", rp.0, r0.0);
	assert!(ra.0 < r0.0, "an off-target altitude must cost λ_alt reward ({} vs {})", ra.0, r0.0);
	assert_eq!((rp.1, rp.2), (r0.1, r0.2), "λ must not move the attitude metrics");
}

/// G3: the eval addresses on LIVE observations. Before the fix it never called
/// set_vertical_obs/set_horizontal_obs, and reset() does not clear them, so every
/// eval step addressed on whatever the previous rollout left behind (its last-step
/// e, v, alt_err, vz), held constant — a phantom offset. Seed the slots with junk:
/// after the eval they must hold the eval's own live state, not the junk.
#[test]
fn g3_eval_feeds_live_observations()
{
	let mut cfg = quiet_stage2_cfg(300);
	cfg.eval_episodes = 2;
	let mut c = pin_student(true, true);
	let junk_v = [9.0f32, 9.0, 9.0];
	let junk_h = [9.0f32, -9.0, 9.0, -9.0];
	c.set_vertical_obs(junk_v[0], junk_v[1], junk_v[2]);
	c.set_horizontal_obs(junk_h[0], junk_h[1], junk_h[2], junk_h[3]);
	eval_once(&mut c, &cfg, 21);
	let (v, h) = (c.vert_obs(), c.horiz_obs());
	assert!(v != junk_v && h != junk_h, "the eval addressed on stale observations: {v:?} {h:?}");
	assert!(h[0].abs() <= cfg.xy_offset && h[1].abs() <= cfg.xy_offset, "live e_xy must lie in the start box: {h:?}");
	assert!((v[0] - c.collective_anchor()).abs() < 1e-6, "collective_cmd must be the episode's anchor");
}

/// G1: the calibration sampler flies the training cascade, so every horizontal
/// ladder has real spread — the position errors span the start draws and the
/// velocities the teacher's fly-back (the old fitter sampled all four as 0.0).
/// The vertical channel is controlled (alt_err returns toward 0), not a drift.
#[test]
fn g1_sampler_ladders_are_not_degenerate()
{
	let mut cfg = quiet_stage2_cfg(1500);
	cfg.full_tilt_deg = 5.0;
	let mut c = pin_student(true, true);
	let s = crate::calib_sampler::sample_calibration_features_rs(&mut c, &cfg, 4, 3);
	assert_eq!(s.len(), 16, "9 base + 3 vertical + 4 horizontal features");
	let span = |v: &Vec<f32>| v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min);
	for (f, min_span) in [(12usize, 0.2f32), (13, 0.2), (14, 0.02), (15, 0.02)]
	{
		assert!(span(&s[f]) > min_span, "feature {f} span {} <= {min_span}: degenerate ladder", span(&s[f]));
	}
	assert!(span(&s[10]) > 0.1, "alt_err must span the vertical starts");
	let tail_alt = s[10][1400..1500].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
	assert!(tail_alt < 0.1, "the altitude PD must bring alt_err home in episode 0 (tail |e| {tail_alt})");
}

/// Recorder fixture: one episode per start, cf21 plant, alt offset 0.3, xy iff given.
fn recorder_draws(xy: Option<(f32, f32)>) -> Stage1Cfg
{
	Stage1Cfg {
		target_altitude: 0.0,
		lambda_alt: 0.0,
		init_z: vec![0.3],
		init_vz: vec![0.0],
		mass: vec![0.0393],
		collective_frac: vec![0.0],
		lambda_pos: 0.0,
		init_x: xy.map(|p| vec![p.0]).unwrap_or_default(),
		init_y: xy.map(|p| vec![p.1]).unwrap_or_default(),
	}
}

/// Record one 1500-step episode; returns (output universe, final sim).
fn record_one(cfg: &RewardGatedConfigPacked, draws: &Stage1Cfg, cascade: bool) -> (Vec<(usize, u64)>, AttitudeSim)
{
	let af = AirframeRs::from_cfg(cfg);
	let mut sim = af.sim();
	let mut c = pin_student(true, true);
	let mut teacher = af.teacher(cfg.teacher);
	let mut pid = AttitudePidRs::new_default();
	let mut d = if cascade
	{
		Driver::Cascade(CascadeDriver::new(&mut teacher, cfg))
	}
	else
	{
		Driver::Pid(&mut pid)
	};
	let s1 = RecorderStage1 {
		cfg: draws,
		gravity: cfg.af_gravity,
		k_thrust: cfg.af_k_thrust,
	};
	let q = [[1.0f32, 0.0, 0.0, 0.0]];
	let om = [[0.0f32; 3]];
	let (_s, o) = record_address_universe(&mut c, &mut sim, &mut d, &q, &om, [0.0; 3], 1500, Some(&s1));
	(o, sim)
}

/// G2: the recorder flies the training cascade. The legacy 0.5-hover PID does not
/// hover cf21 (hover pwm 0.694), so it recorded a falling vehicle; the cascade
/// holds altitude, applies the horizontal start, and flies it home — and arming
/// xy changes the recorded universe (the channel is live, not a false green).
#[test]
fn g2_recorder_flies_the_training_cascade()
{
	let cfg = quiet_stage2_cfg(1500);
	let (_, legacy) = record_one(&cfg, &recorder_draws(None), false);
	assert!(legacy.altitude_rs() < -0.5, "fixture: the legacy PID must fall on cf21 ({})", legacy.altitude_rs());
	let (o_z, hover) = record_one(&cfg, &recorder_draws(None), true);
	assert!(hover.altitude_rs().abs() < 0.1, "the cascade must hold altitude ({})", hover.altitude_rs());
	let (o_xy, flown) = record_one(&cfg, &recorder_draws(Some((0.4, -0.4))), true);
	let [x, y] = flown.position_xy_rs();
	assert!(x.abs() < 0.3 && y.abs() < 0.3, "the xy start must be applied and flown home ({x}, {y})");
	assert!(x != 0.0 || y != 0.0, "the xy start was never applied");
	assert_ne!(o_xy, o_z, "arming xy must change the recorded universe");
}
