//! L3 of block 10 (ABI 51, `--alt-integral {off, pi, pole3}`): the altitude
//! integrator in front of the altitude PD (altitude_integrator.rs,
//! docs/ctrl48_block10_levers_spec.md §L3).
//!
//! Pinned here: off is the identity; the law, its clamp and its conditional
//! integration; the droop of a constant thrust-to-weight error goes to ~0 on the
//! translating plant for both modes while off keeps ε·g/ω²; one integrator per
//! episode; the rival scorer and the trainer's OuterLoops fly the same altitude law;
//! the student's observation is the law's input; CPU scorer == Metal scorer.

use crate::abi38_pins::{DT, G, KT, MASS};
use crate::altitude_integrator::{AltIntegralMode, AltitudeIntegrator};
use crate::altitude_pd::AltitudePd;
use crate::controller::WnnController;
use crate::dagger_train::RewardGatedConfigPacked;

const OMEGA: f64 = 2.0;

fn pd() -> AltitudePd
{
	AltitudePd::from_plant(MASS as f64, G as f64, KT as f64, OMEGA, 1.0, 0.25).unwrap()
}

fn integ(mode: AltIntegralMode) -> AltitudeIntegrator
{
	AltitudeIntegrator::new(mode, &pd(), G as f64, DT).unwrap()
}

/// The trainer config of the cf21 translation pins with the integrator mode set.
fn cfg(mode: u8, xy: f32) -> RewardGatedConfigPacked
{
	let mut c = crate::ctrl17_pins::pin_translation_cfg(xy);
	c.teacher = 0;
	c.alt_integral = mode;
	c.validate_alt_integral().expect("valid alt_integral");
	c
}

#[test]
fn off_is_the_identity_and_holds_no_state()
{
	let mut off = AltitudeIntegrator::OFF;
	for (e, v) in [(0.3f32, -0.2f32), (-0.0, 0.0), (f32::NAN, f32::INFINITY), (1e-40, -1e30), (5.0, 9.0)]
	{
		let (a, b) = off.shape(e, v);
		assert_eq!((a.to_bits(), b.to_bits()), (e.to_bits(), v.to_bits()), "off must return its inputs untouched");
	}
	assert_eq!(off.integral(), 0.0);
	assert!(!off.is_on());
	// Mode 0 reads nothing else: a host with no altitude PD at all stays valid.
	let z = AltitudeIntegrator::from_plant(0, f64::NAN, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0).expect("mode 0 validates nothing");
	assert!(!z.is_on());
	assert!(AltitudeIntegrator::new(AltIntegralMode::Off, &pd(), G as f64, DT).is_ok_and(|i| !i.is_on()));
	assert!(AltIntegralMode::from_u8(3).is_err());
	assert_eq!([0u8, 1, 2].map(|m| AltIntegralMode::from_u8(m).unwrap()), [AltIntegralMode::Off, AltIntegralMode::Pi, AltIntegralMode::Pole3]);
}

/// e_eff / vz_eff are the spec's law on the integral of the PREVIOUS steps' errors.
#[test]
fn shape_is_the_spec_law()
{
	for (mode, ke, ki, kv) in [(AltIntegralMode::Pi, 1.0, OMEGA / 4.0, 1.0), (AltIntegralMode::Pole3, 3.0, OMEGA, 1.5)]
	{
		let mut it = integ(mode);
		let mut i_ref = 0.0f64;
		for t in 0..3000
		{
			let e = 0.08 * ((t as f32) * 0.002).cos();
			let vz = 0.1 * ((t as f32) * 0.003).sin();
			let (e_eff, vz_eff) = it.shape(e, vz);
			let want_e = ke * e as f64 + ki * i_ref;
			assert!((e_eff as f64 - want_e).abs() < 1e-5, "{mode:?} step {t}: e_eff {e_eff} vs {want_e}");
			assert!((vz_eff as f64 - kv * vz as f64).abs() < 1e-6, "{mode:?} step {t}: vz_eff");
			i_ref += e as f64 * DT as f64;
			assert!((it.integral() as f64 - i_ref).abs() < 1e-5, "{mode:?} step {t}: I {} vs {i_ref}", it.integral());
		}
		assert!(i_ref.abs() > 1e-3, "fixture never integrated");
	}
}

/// Ideal vertical plant z̈ = az − d (the PD's linear law, no clamp reached): off parks
/// at d/ω²; pi and pole3 drive the error to zero, and pole3's poles all sit at −ω.
#[test]
fn ideal_plant_droop_is_removed()
{
	let d = 1.2f64; // m/s² of missing thrust
	let fly = |mode: AltIntegralMode, secs: f64| {
		let mut it = integ(mode);
		let (mut z, mut vz) = (0.0f64, 0.0f64);
		let dt = DT as f64;
		let mut worst_late = 0.0f64;
		let n = (secs / dt) as usize;
		for t in 0..n
		{
			let (e, v) = it.shape(-z as f32, vz as f32);
			let az = OMEGA * OMEGA * e as f64 - 2.0 * OMEGA * v as f64;
			vz += (az - d) * dt;
			z += vz * dt;
			if t > n * 9 / 10
			{
				worst_late = worst_late.max(z.abs());
			}
		}
		worst_late
	};
	let droop = d / (OMEGA * OMEGA);
	let off = fly(AltIntegralMode::Off, 20.0);
	assert!((off - droop).abs() < 0.02 * droop, "off must park at d/omega^2 = {droop}, got {off}");
	for mode in [AltIntegralMode::Pi, AltIntegralMode::Pole3]
	{
		let late = fly(mode, 20.0);
		assert!(late < 0.01 * droop, "{mode:?}: the droop must vanish, {late} m left of {droop}");
	}
	// pole3 = three poles at −ω: after 4 s the error is far inside 1% (t²·e^{−ωt}).
	assert!(fly(AltIntegralMode::Pole3, 6.0) < 0.01 * droop);
}

/// Anti-windup (a): k_i·|I| <= 0.4·g, both signs.
#[test]
fn integral_is_clamped_to_0p4_g()
{
	for (mode, k_i) in [(AltIntegralMode::Pi, OMEGA.powi(3) / 4.0), (AltIntegralMode::Pole3, OMEGA.powi(3))]
	{
		for sign in [1.0f32, -1.0]
		{
			let mut it = integ(mode);
			// A small error (the PD never saturates), held long past the clamp.
			for _ in 0..400_000
			{
				it.shape(sign * 0.02, 0.0);
			}
			let accel = k_i * it.integral() as f64;
			assert!((accel - sign as f64 * 0.4 * G as f64).abs() < 1e-3, "{mode:?} sign {sign}: integral accel {accel} m/s², want ±0.4 g");
		}
	}
}

/// Anti-windup (b): no integration on a step where the PD's unclamped command is at
/// its limit — through the error, through the velocity, and released below it.
#[test]
fn no_integration_while_the_pd_is_at_its_limit()
{
	let (_, _, az_limit) = pd().law();
	for mode in [AltIntegralMode::Pi, AltIntegralMode::Pole3]
	{
		let mut it = integ(mode);
		let e_sat = (az_limit / (OMEGA * OMEGA)) as f32 * 1.5;
		for _ in 0..2000
		{
			it.shape(e_sat, 0.0);
		}
		assert_eq!(it.integral(), 0.0, "{mode:?}: integrated while the error saturates the PD");
		let vz_sat = (az_limit / (2.0 * OMEGA)) as f32 * 1.5;
		for _ in 0..2000
		{
			it.shape(0.01, vz_sat);
		}
		assert_eq!(it.integral(), 0.0, "{mode:?}: integrated while the velocity saturates the PD");
		for _ in 0..2000
		{
			it.shape(0.05, 0.0);
		}
		assert!((it.integral() - 0.05 * 2000.0 * DT).abs() < 1e-4, "{mode:?}: must integrate once below the limit, I = {}", it.integral());
		it.reset();
		assert_eq!(it.integral(), 0.0, "reset must clear the integral");
	}
}

#[test]
fn refusals()
{
	let zeta07 = AltitudePd::from_plant(MASS as f64, G as f64, KT as f64, OMEGA, 0.7, 0.25).unwrap();
	assert!(AltitudeIntegrator::new(AltIntegralMode::Pole3, &zeta07, G as f64, DT).is_err(), "pole3 is defined for zeta = 1 only");
	assert!(AltitudeIntegrator::new(AltIntegralMode::Pi, &zeta07, G as f64, DT).is_ok());
	assert!(AltitudeIntegrator::new(AltIntegralMode::Pi, &pd(), G as f64, 0.0).is_err());
	// The trainer config: needs translation, a known mode, zeta = 1 for pole3.
	let mut c = crate::ctrl17_pins::pin_translation_cfg(0.5);
	assert_eq!(c.alt_integral, 0, "default off");
	assert!(c.validate_alt_integral().is_ok());
	c.alt_integral = 3;
	assert!(c.validate_alt_integral().is_err(), "unknown mode");
	c.alt_integral = 2;
	c.alt_pd_zeta = 0.7;
	assert!(c.validate_alt_integral().is_err(), "pole3 with zeta != 1");
	c.alt_pd_zeta = 1.0;
	assert!(c.validate_alt_integral().is_ok());
	let mut att = crate::ctrl17_pins::pin_attitude_cfg();
	att.alt_integral = 1;
	assert!(att.validate_alt_integral().is_err(), "no altitude loop without translation");
}

/// One rival episode set (firmware PID cascade, oracle-fed, no weather) on the cf21
/// translating plant. `xy` empty ⇒ stage 1.
#[allow(clippy::type_complexity)]
fn rival(
	c: &RewardGatedConfigPacked,
	mode: u8,
	hold: usize,
	steps: usize,
	q0: &[f32],
	w0: &[f32],
	z0: &[f32],
	mass: &[f32],
	xy: &[(f32, f32)],
) -> pyo3::PyResult<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)>
{
	let n = mass.len();
	crate::dagger_train::score_classical_baseline(
		0, q0.to_vec(), w0.to_vec(), steps, 5.0,
		false, [0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0,
		c.af_arm_length, c.af_k_thrust, c.af_k_drag, c.af_inertia, c.af_gravity, c.dt as f32,
		c.af_pid_att, c.af_pid_rate, c.af_pid_out_limit_n, c.af_pid_hover_n, c.af_pid_attitude_hz, c.af_pid_lpf_hz,
		false, 2.0, 0.1,
		true, c.target_altitude, z0.to_vec(), vec![0.0; n], mass.to_vec(), vec![0.0; n],
		c.alt_pd_omega as f64, c.alt_pd_zeta as f64, c.alt_pd_max_delta as f64,
		xy.iter().map(|p| p.0).collect(), xy.iter().map(|p| p.1).collect(),
		c.pos_omega as f64, c.pos_zeta as f64, c.pos_max_tilt_rad as f64,
		0.0, false, 0.0, false, hold, mode,
	)
}

const LEVEL_Q: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// THE spec test: a constant thrust-to-weight error (a heavy mass draw against the
/// nominal hover) on the translating plant. Off holds ε·g/ω² below target for the
/// whole episode; pi and pole3 bring the hold-window |dz| to ~0 — on the motor-level
/// rival and on the cascade (CTBR over the rate loop) rival.
#[test]
fn rival_droop_is_removed_on_the_translating_plant()
{
	let c = cfg(0, 0.0);
	let eps = 0.15f32;
	let droop = eps as f64 * G as f64 / (OMEGA * OMEGA);
	for hold in [0usize, 1]
	{
		let hold_dz = |mode: u8| rival(&c, mode, hold, 12_000, &LEVEL_Q, &[0.0; 3], &[0.0], &[MASS * (1.0 + eps)], &[]).expect("rival scores").9;
		let off = hold_dz(0);
		assert!(off > 0.6 * droop && off < 1.4 * droop, "hold {hold}: off must keep the droop ≈ ε·g/ω² = {droop:.3} m, got {off:.3}");
		for mode in [1u8, 2]
		{
			let on = hold_dz(mode);
			assert!(on < 0.05 * off, "hold {hold} mode {mode}: hold |dz| {on:.4} m is not ~0 (off {off:.3})");
		}
	}
	// Refused where there is no altitude loop.
	assert!(crate::dagger_train::score_classical_baseline(
		0, LEVEL_Q.to_vec(), vec![0.0; 3], 100, 5.0,
		false, [0.0; 3], 0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0,
		c.af_arm_length, c.af_k_thrust, c.af_k_drag, c.af_inertia, c.af_gravity, c.dt as f32,
		c.af_pid_att, c.af_pid_rate, c.af_pid_out_limit_n, c.af_pid_hover_n, c.af_pid_attitude_hz, c.af_pid_lpf_hz,
		false, 2.0, 0.1,
		false, 0.0, vec![], vec![], vec![], vec![],
		2.0, 1.0, 0.25, vec![], vec![], 1.0, 1.0, 0.5236,
		0.0, false, 0.0, false, 0, 1,
	)
	.is_err(), "alt_integral without translation must be refused");
}

/// One integrator PER EPISODE: two identical episodes score exactly like one (a
/// carried integral would make the second one differ).
#[test]
fn rival_integrator_resets_per_episode()
{
	let c = cfg(0, 0.0);
	let m = MASS * 1.12;
	for mode in [1u8, 2]
	{
		let one = rival(&c, mode, 0, 3000, &LEVEL_Q, &[0.0; 3], &[0.1], &[m], &[]).unwrap();
		let two = rival(&c, mode, 0, 3000, &[LEVEL_Q, LEVEL_Q].concat(), &[0.0; 6], &[0.1, 0.1], &[m, m], &[]).unwrap();
		assert_eq!(format!("{one:?}"), format!("{two:?}"), "mode {mode}: the second episode saw the first one's integral");
	}
}

/// What the rival scorer reports for |dz| (4th), the 3-D error (5th) and their
/// hold-window means, recomputed by flying the TRAINER's cascade (OuterLoops +
/// sense_altitude, the DAgger label path) on the same episode.
fn outer_loops_flight(c: &RewardGatedConfigPacked, hold: usize, steps: usize, q0: [f32; 4], w0: [f32; 3], z0: f32, mass: f32, xy: (f32, f32)) -> [f64; 4]
{
	use crate::episode_regime::{alt_integrator_for, sense_altitude, OuterLoops};
	let af = crate::dagger_train::AirframeRs::from_cfg(c);
	let (mut sim, mut teacher) = (af.sim(), af.teacher(0));
	let loops = OuterLoops::for_cfg(c);
	let mut alt_int = alt_integrator_for(c);
	let mut over = (hold > 0).then(|| crate::ctbr::CtbrOverRate::new(af.pid_fw.as_ref().unwrap().rate_stage(), hold));
	teacher.reset();
	sim.reset(Some(q0), Some(w0));
	sim.set_translation_core(mass).unwrap();
	sim.set_vertical_state(z0, 0.0);
	sim.set_horizontal_state(xy.0, xy.1, 0.0, 0.0);
	let tail_start = ((steps as f64) * 0.80).ceil() as usize;
	let (mut s, mut tail, mut n_tail, mut done) = ([0.0f64; 2], [0.0f64; 2], 0usize, 0usize);
	let mut last = [0.5f32; 4];
	for t in 0..steps
	{
		if sim.is_unstable()
		{
			break;
		}
		let (gyro, _) = sim.read_imu();
		let q = sim.quaternion();
		let alt_law = sense_altitude(&mut alt_int, &sim, c.target_altitude);
		teacher.observe(gyro, last.map(|v| v as f64));
		let cmd = match over.as_mut()
		{
			Some(o) => o.fly(loops.expert_ctbr(&mut teacher, &sim, q, gyro, [0.0; 3], alt_law, hold), gyro),
			None => loops.expert(&mut teacher, &sim, q, gyro, [0.0; 3], alt_law),
		};
		last = cmd.map(|v| v as f32);
		sim.step(last);
		let dz = (c.target_altitude - sim.altitude_rs()) as f64;
		let (x, y) = sim.position_xy();
		let p = ((x as f64).powi(2) + (y as f64).powi(2) + dz * dz).sqrt();
		s[0] += dz.abs();
		s[1] += p;
		if t >= tail_start
		{
			tail[0] += p;
			tail[1] += dz.abs();
			n_tail += 1;
		}
		done += 1;
	}
	[s[0] / done as f64, s[1] / done as f64, tail[0] / n_tail as f64, tail[1] / n_tail as f64]
}

/// The rival scorer and the trainer's OuterLoops (the label) give the SAME collective
/// on the same state history: flown closed-loop on one episode, any difference in the
/// altitude law at any step would separate the two trajectories. Stage 2 (the full
/// cascade), motor-level and CTBR, all three modes.
#[test]
fn rival_and_outer_loops_fly_the_same_altitude_law()
{
	let (q0, w0) = ([0.9987f32, 0.03, -0.04, 0.0], [0.2f32, -0.1, 0.0]);
	let (z0, mass, xy) = (-0.12f32, MASS * 1.1, (0.35f32, -0.25f32));
	let mut moved = Vec::new();
	for hold in [0usize, 1]
	{
		for mode in [0u8, 1, 2]
		{
			let c = cfg(mode, 0.5);
			let r = rival(&c, mode, hold, 2500, &q0, &w0, &[z0], &[mass], &[xy]).expect("rival scores");
			let got = [r.3, r.4, r.8, r.9];
			let want = outer_loops_flight(&c, hold, 2500, q0, w0, z0, mass, xy);
			for i in 0..4
			{
				assert!(
					(got[i] - want[i]).abs() <= 1e-9 * want[i].abs().max(1e-6),
					"hold {hold} mode {mode} metric {i}: rival {} vs OuterLoops {}", got[i], want[i]
				);
			}
			moved.push(got[3]);
		}
	}
	// The mode reaches both hosts: the hold |dz| differs between off, pi and pole3.
	assert!((moved[0] - moved[1]).abs() > 1e-4 && (moved[0] - moved[2]).abs() > 1e-4 && (moved[1] - moved[2]).abs() > 1e-6, "alt_integral is dead: {moved:?}");
}

/// The student's alt_err observation IS the altitude law's input (Hard Rule 1): in
/// the DAgger rollout the recorded vert stream carries e_eff — equal to the raw error
/// at step 0 under pi (I = 0), 3× it under pole3 — and the vz observation stays raw.
#[test]
fn student_observation_is_the_law_input()
{
	use crate::episode_regime::{alt_integrator_for, feed_live_obs, sense_altitude};
	for mode in [0u8, 1, 2]
	{
		let c = cfg(mode, 0.5);
		let af = crate::dagger_train::AirframeRs::from_cfg(&c);
		let mut sim = af.sim();
		sim.reset(Some(LEVEL_Q), Some([0.0; 3]));
		sim.set_translation_core(MASS).unwrap();
		sim.set_vertical_state(-0.2, 0.3);
		let mut student = crate::ctrl17_pins::pin_student(true, true);
		let mut alt_int = alt_integrator_for(&c);
		let mut twin = integ(AltIntegralMode::from_u8(mode).unwrap());
		for t in 0..50
		{
			let law = sense_altitude(&mut alt_int, &sim, c.target_altitude);
			feed_live_obs(&mut student, &sim, &c, 0.69, law.0);
			let want = twin.shape(c.target_altitude - sim.altitude_rs(), sim.vertical_velocity_rs());
			assert_eq!(law, want, "mode {mode} step {t}: the config's integrator is not the spec's on this plant");
			let v = student.vert_obs();
			assert_eq!(v[1].to_bits(), law.0.to_bits(), "mode {mode} step {t}: alt_err obs is not e_eff");
			assert_eq!(v[2].to_bits(), sim.vertical_velocity_rs().to_bits(), "mode {mode} step {t}: vz obs must stay raw");
			sim.step([0.66; 4]); // sinking slowly: a growing error
		}
		assert_eq!(alt_int.is_on(), mode != 0);
	}
	// And through the real rollout: the recorded vert stream differs from the off run
	// exactly from the first step whose law input differs.
	let run = |mode: u8| {
		use rand::SeedableRng;
		let c = cfg(mode, 0.5);
		let af = crate::dagger_train::AirframeRs::from_cfg(&c);
		let (mut teacher, mut sim) = (af.teacher(0), af.sim());
		let mut student = crate::ctrl17_pins::pin_student(true, true);
		let mut rng = rand::rngs::SmallRng::seed_from_u64(51);
		crate::dagger_train::rollout_and_label_rs(&mut student, &mut teacher, &mut sim, &c, 0.1, &mut rng, [0.0; 3])
	};
	let (off, pi, p3) = (run(0), run(1), run(2));
	assert_eq!(off.vert_obs[0], pi.vert_obs[0], "pi: I = 0 at step 0, so e_eff = e");
	assert!((p3.vert_obs[0][1] - 3.0 * off.vert_obs[0][1]).abs() < 1e-6, "pole3: e_eff = 3e at step 0");
	assert!(off.vert_obs.iter().zip(&pi.vert_obs).any(|(a, b)| a[1] != b[1]), "pi never reached the student's observation");
	assert_ne!(off.pid_pwms, pi.pid_pwms, "pi never reached the label");
}

/// Every training host flies the integrator and is untouched at mode 0: the
/// calibration sampler (the ladder is fitted on e_eff), the DAgger rollouts + the
/// per-round checkpoint eval (whole trainer) and the MEMORY recorder.
#[test]
fn every_training_host_applies_it()
{
	use crate::calib_sampler::sample_calibration_features_rs;
	const ALT_ERR: usize = 10; // pin_student(true, true): 9 base, collective, alt_err, vz, 4 horizontal
	let calib = |c: &RewardGatedConfigPacked| sample_calibration_features_rs(&mut crate::ctrl17_pins::pin_student(true, true), c, 2, 9);
	let (base, pi, p3) = (calib(&cfg(0, 0.5)), calib(&cfg(1, 0.5)), calib(&cfg(2, 0.5)));
	assert_eq!(base, calib(&crate::ctrl17_pins::pin_translation_cfg(0.5)), "mode 0 calibration moved");
	assert_ne!(base[ALT_ERR], pi[ALT_ERR], "the calibration sampler ignored the integrator");
	// First sample of the first episode: I = 0, so pole3's e_eff is exactly 3e.
	assert!((p3[ALT_ERR][0] - 3.0 * base[ALT_ERR][0]).abs() < 1e-6 && base[ALT_ERR][0] != 0.0, "calibration alt_err is not e_eff");
	assert_eq!(base[ALT_ERR + 1][0].to_bits(), p3[ALT_ERR + 1][0].to_bits(), "the vz sample must stay raw");

	let train = |c: &RewardGatedConfigPacked| {
		let mut s = crate::ctrl17_pins::pin_student(true, true);
		let st = crate::dagger_train::dagger_train_inplace_with(&mut s, c, [0.0; 3], 11, false);
		(st.iter_mean_err_deg.clone(), st.iter_fitness.clone(), crate::abi38_pins::cells_and_stats_digest(&s, Some(&st)))
	};
	let t0 = train(&crate::ctrl17_pins::pin_translation_cfg(0.5));
	assert_eq!(train(&cfg(0, 0.5)), t0, "mode 0 trainer moved");
	assert_ne!(train(&cfg(1, 0.5)), t0, "the trainer (rollouts + per-round eval) ignored the integrator");

	let rec = |c: &RewardGatedConfigPacked| crate::ctrl17_tests::record_universe_for(c, &crate::ctrl17_pins::pin_student(true, true));
	let r0 = rec(&crate::ctrl17_pins::pin_translation_cfg(0.5));
	assert_eq!(rec(&cfg(0, 0.5)), r0, "mode 0 recorder moved");
	assert_ne!(rec(&cfg(2, 0.5)), r0, "the MEMORY recorder ignored the integrator");
}

/// A stage-2 student wired to its alt_err input (feature 12 of the abi32 layout: 9
/// base + roll/pitch err + 3 vertical + 4 horizontal), every address written.
#[cfg(target_os = "macos")]
fn alt_student(seed: u64) -> WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn, nf) = (4usize, 3usize, 2usize, 8usize, 18usize);
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..nf * bpf)
		.map(|i| match i / bpf
		{
			9 | 10 => rng.gen_range(-0.3f32..0.3),
			12 => rng.gen_range(-0.25f32..0.25),
			f if f >= 11 => rng.gen_range(-0.6f32..0.6),
			_ => rng.gen_range(-5.0f32..5.0),
		})
		.collect();
	let num_out = 4 * levels;
	let oconn: Vec<i64> = (0..num_out * obpn)
		.map(|k| if k % 2 == 0 { (12 * bpf + rng.gen_range(0..bpf)) as i64 } else { rng.gen_range(0..nf * bpf) as i64 })
		.collect();
	let c = WnnController::new_core(
		4, levels, bpf, window, 0, 0, obpn, thresholds, Vec::new(), oconn,
		true, 0.05, 0.98, 1.0,
		false, false, true, false, false, false, false, false,
		0.99, 1.0, DT, false, 1, ram_core::neuron_memory::BINARY, Some(crate::cell_mode::DECODE_ANTAGONIST), None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("alt student must construct");
	for n in 0..num_out
	{
		for k in 0..(1u64 << obpn)
		{
			c.write_output_cell_internal(n, k, if rng.gen_bool(0.5) { 1 } else { 0 });
		}
	}
	c
}

/// CPU scorer == Metal score kernel with the reshaped alt_err, both modes, and the
/// integrator MOVES the result on both sides (the kernel reads the new uniforms).
#[cfg(target_os = "macos")]
#[test]
fn alt_integral_score_cpu_matches_gpu()
{
	use crate::abi32_tests::{assert_parity_steps, cfg as s1cfg, differs, gpu, tilt_ref};
	let plant = crate::plant_opts::PlantOpts::new(true, 0.78).unwrap();
	let with = |mode: AltIntegralMode| {
		let mut s = s1cfg(plant, tilt_ref());
		s.alt_int = integ(mode);
		s
	};
	let c = alt_student(51);
	let off = assert_parity_steps(&c, &with(AltIntegralMode::Off), 300, "alt off");
	let pi = assert_parity_steps(&c, &with(AltIntegralMode::Pi), 300, "alt pi");
	let p3 = assert_parity_steps(&c, &with(AltIntegralMode::Pole3), 300, "alt pole3");
	differs(&off, &pi, &[0, 1, 13, 14], "alt_integral pi");
	differs(&off, &p3, &[0, 1, 13, 14], "alt_integral pole3");
	if metal::Device::system_default().is_some()
	{
		let rows = [AltIntegralMode::Off, AltIntegralMode::Pi, AltIntegralMode::Pole3].map(|m| gpu(&c, &with(m), 300).expect("GPU evaluator"));
		for on in [&rows[1], &rows[2]]
		{
			assert!([0usize, 1, 13, 14].iter().any(|&i| (rows[0][i] - on[i]).abs() > 1e-9), "GPU kernel ignored the altitude integrator");
		}
	}
}
