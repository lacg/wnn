//! L1 of block 10 (ABI 51, `--obs-tilt-ref-err`): the student's roll / pitch error
//! measured against the position loop's tilt reference, bias-corrected
//! (tilt_ref_err.rs, docs/ctrl48_block10_levers_spec.md §L1).
//!
//! Pinned here: the refusals; the two inputs (and their integrals) equal the
//! definition on a hand-built state while every other input — the trailing bias
//! prediction included — is untouched by the reordering; a replay reproduces the live
//! frames (train = deploy) from the streams it already re-applies; CPU scorer == Metal
//! score kernel; Metal split record/train == the CPU split replays; the whole DAgger
//! trainer on the CPU walk == the GPU walk.

use crate::controller::WnnController;
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

/// The cf21 position loop of the pin configs: b_xy = g, ω_n, ζ, max tilt (rad).
const LAW: (f32, f32, f32, f32) = (9.81, 1.0, 1.0, 0.5236);

/// Layout: 9 base, roll_err / pitch_err (9, 10), roll_i / pitch_i (11, 12), 3 vertical
/// (13..16), 4 horizontal [e_x, e_y, v_x, v_y] (16..20), abp_r / abp_p (20, 21).
const ROLL_ERR: usize = 9;
const PITCH_ERR: usize = 10;
const ROLL_I: usize = 11;
const PITCH_I: usize = 12;
const ABP: usize = 20;
const BASE_NF: usize = 20;

/// A stage-2 student with the estimator, per-axis P + I (yaw dropped), the vertical
/// and horizontal channels and the bias predictor; `on` adds the tilt-reference error.
/// Half the output bits sit on the four inputs L1 changes, so it steers the policy.
fn tre_student(seed: u64, sn: usize, on: bool, cells: usize, mode: u8) -> WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn, sbpn) = (4usize, 3usize, 2usize, 8usize, 6usize);
	let nf = BASE_NF + 2;
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..BASE_NF * bpf)
		.map(|i| match i / bpf
		{
			ROLL_ERR | PITCH_ERR => rng.gen_range(-0.08f32..0.08),
			ROLL_I | PITCH_I => rng.gen_range(-2.0f32..2.0),
			f if f >= 13 => rng.gen_range(-0.6f32..0.6),
			_ => rng.gen_range(-5.0f32..5.0),
		})
		.collect();
	let num_out = 4 * levels;
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..window * nf * bpf + sn) as i64).collect();
	let oconn: Vec<i64> = (0..num_out * obpn)
		.map(|k| if k % 2 == 0 { ((ROLL_ERR + (k / 2) % 4) * bpf + rng.gen_range(0..bpf)) as i64 } else { rng.gen_range(0..nf * bpf + sn) as i64 })
		.collect();
	let decode = if mode == BINARY { Some(crate::cell_mode::DECODE_ANTAGONIST) } else { None };
	let mut c = WnnController::new_core(
		4, levels, bpf, window, sn, if sn > 0 { sbpn } else { 0 }, obpn, thresholds, sconn, oconn,
		true, 0.05, 0.98, 1.0,
		false, false, true, true, false, false, false, false,
		0.99, 1.0, 0.001, false, 1, mode, decode, None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("tre student must construct");
	c.enable_att_estimator(2.0, 0.1).expect("estimator on");
	let thr: Vec<f32> = (0..2 * bpf).map(|i| [-0.004f32, 0.0, 0.004][i % 3]).collect();
	c.enable_att_bias_pred(thr).expect("abp on");
	assert_eq!(c.num_features(), nf);
	if on
	{
		c.enable_tilt_ref_err(LAW.0, LAW.1, LAW.2, LAW.3).expect("tilt-ref error on");
		assert_eq!(c.num_features(), nf, "L1 must not change the layout");
	}
	for n in 0..num_out
	{
		for k in 0..cells
		{
			let addr = if cells >= (1 << obpn) { k as u64 } else { rng.gen_range(0..(1u64 << obpn)) };
			c.write_output_cell_internal(n, addr, if rng.gen_bool(0.5) { 1 } else { 0 });
		}
	}
	c
}

/// The stage-2 cf21 trainer config with the realistic IMU (the bias the predictor
/// models exists only under the specific-force accelerometer).
fn tre_cfg() -> crate::dagger_train::RewardGatedConfigPacked
{
	let mut cfg = crate::ctrl17_pins::pin_translation_cfg(0.5);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	cfg
}

#[test]
fn tre_refusals_and_default_off()
{
	let bare = |pa_p: bool, pos: bool, vel: bool| {
		WnnController::new_core(
			4, 4, 3, 2, 0, 0, 8, vec![0.0; (9 + 2 * pa_p as usize + 3 + 2 * pos as usize + 2 * vel as usize) * 3], Vec::new(), vec![0; 16 * 8],
			true, 0.05, 0.98, 1.0,
			false, false, pa_p, false, false, false, false, false,
			0.99, 1.0, 0.001, false, 1, BINARY, None, None, 0.05, false, 0.30,
			true, true, true, pos, vel, false, 1,
		)
		.expect("bare student")
	};
	// No bias predictor (here: no estimator at all).
	let mut c = bare(true, true, true);
	assert!(c.enable_tilt_ref_err(LAW.0, LAW.1, LAW.2, LAW.3).is_err(), "needs obs_att_bias_pred");
	// Predictor on, but no position-error input.
	let mut c = bare(true, false, true);
	c.enable_att_estimator(2.0, 0.1).unwrap();
	c.enable_att_bias_pred(vec![0.0; 6]).unwrap();
	assert!(c.enable_tilt_ref_err(LAW.0, LAW.1, LAW.2, LAW.3).is_err(), "needs obs_pos_err_xy");
	// A degenerate loop, then the valid one, then twice.
	let mut c = tre_student(1, 0, false, 6, BINARY);
	assert_eq!(c.tilt_ref_err_shape(), None, "default off");
	assert_eq!(c.obs_tilt_ref_err(), None);
	assert!(c.enable_tilt_ref_err(0.0, 1.0, 1.0, 0.5).is_err(), "gravity 0");
	assert!(c.enable_tilt_ref_err(9.81, 1.0, 1.0, 2.0).is_err(), "tilt past horizontal");
	assert!(c.enable_tilt_ref_err(LAW.0, LAW.1, LAW.2, LAW.3).is_ok());
	assert_eq!(c.obs_tilt_ref_err(), Some(LAW));
	assert!(c.enable_tilt_ref_err(LAW.0, LAW.1, LAW.2, LAW.3).is_err(), "enabled twice");
	// The trainer refuses a student whose law is not the config's position loop.
	let cfg = tre_cfg();
	assert!(crate::dagger_train::validate_train_regime(&c, &cfg).is_ok());
	let mut other = tre_student(1, 0, false, 6, BINARY);
	other.enable_tilt_ref_err(LAW.0, 1.5, LAW.2, LAW.3).unwrap();
	assert!(crate::dagger_train::validate_train_regime(&other, &cfg).is_err(), "a different omega_n must be refused");
	assert!(crate::dagger_train::validate_train_regime(&c, &crate::ctrl17_pins::pin_translation_cfg(0.0)).is_err(), "no xy channel");
}

/// A hand-built state stream: (gyro, accel, target, horizontal obs) per step, with a
/// changing velocity (so the bias prediction is alive) and a non-zero target.
fn hand_state(t: usize) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 4])
{
	let s = t as f32;
	(
		[0.02 * (0.01 * s).sin(), -0.015 * (0.013 * s).cos(), 0.003],
		[0.35 * (0.004 * s).sin(), -0.25 * (0.005 * s).cos(), 9.81],
		[0.02, -0.01, 0.0],
		[0.4 - 0.001 * s, -0.3 + 0.0007 * s, 0.25 * (0.02 * s).sin(), -0.2 * (0.015 * s).cos()],
	)
}

/// The two inputs equal the DEFINITION — (target + PositionLoop::tilt_ref + this
/// step's bias prediction) − estimate — their integrals integrate that value, and
/// NOTHING else moves: the trailing abp inputs (the reordering), the raw targets, the
/// vertical / horizontal channels. Off is the ABI-50 definition, target − estimate.
#[test]
fn tre_input_equals_the_definition()
{
	let (mut on, mut off) = (tre_student(7, 0, true, 6, BINARY), tre_student(7, 0, false, 6, BINARY));
	let q0 = [0.998f32, 0.04, -0.05, 0.0];
	for c in [&mut on, &mut off]
	{
		c.reset(0.0);
		c.seed_att_estimator(q0);
	}
	let pl = crate::position_loop::PositionLoop::from_plant(LAW.0 as f64, LAW.1 as f64, LAW.2 as f64, LAW.3 as f64).unwrap();
	let (mut acc, mut acc_off) = ([0.0f32; 2], [0.0f32; 2]);
	let (mut moved, mut abp_alive, mut ref_alive) = (0.0f32, false, false);
	for t in 0..600
	{
		let (g, a, tg, h) = hand_state(t);
		on.set_horizontal_obs(h[0], h[1], h[2], h[3]);
		off.set_horizontal_obs(h[0], h[1], h[2], h[3]);
		let (f, fo) = (on.compute_features(g, a, tg), off.compute_features(g, a, tg));
		for i in (0..f.len()).filter(|i| !(ROLL_ERR..=PITCH_I).contains(i))
		{
			assert_eq!(f[i].to_bits(), fo[i].to_bits(), "step {t}: input {i} moved (only roll/pitch err and their integrals may)");
		}
		assert_eq!(&f[6..9], &tg[..], "step {t}: the raw target inputs");
		assert_eq!(&f[ABP..], &on.att_bias_last()[..], "step {t}: the trailing abp inputs are this step's prediction");
		// The estimate both controllers read (identical filters on identical IMU).
		let up = on.att_up_obs();
		assert_eq!(up, off.att_up_obs());
		let roll_est = up[1].atan2(up[2]);
		let pitch_est = (-up[0]).atan2((up[1] * up[1] + up[2] * up[2]).sqrt());
		// Off: the ABI-50 input, bit for bit.
		assert_eq!(fo[ROLL_ERR].to_bits(), (tg[0] - roll_est).to_bits(), "step {t}: off roll_err");
		assert_eq!(fo[PITCH_ERR].to_bits(), (tg[1] - pitch_est).to_bits(), "step {t}: off pitch_err");
		// On: the definition, against the f64 position law.
		let (roll_ref, pitch_ref) = pl.tilt_ref(h[0] as f64, h[2] as f64, h[1] as f64, h[3] as f64);
		let want_r = (tg[0] as f64 + roll_ref + f[ABP] as f64) - roll_est as f64;
		let want_p = (tg[1] as f64 + pitch_ref + f[ABP + 1] as f64) - pitch_est as f64;
		assert!((f[ROLL_ERR] as f64 - want_r).abs() < 2e-6, "step {t}: roll_err {} vs definition {want_r}", f[ROLL_ERR]);
		assert!((f[PITCH_ERR] as f64 - want_p).abs() < 2e-6, "step {t}: pitch_err {} vs definition {want_p}", f[PITCH_ERR]);
		// The integrals integrate the value the input carries (leak 0.99, scale 1).
		for (k, (e, eo)) in [(f[ROLL_ERR], fo[ROLL_ERR]), (f[PITCH_ERR], fo[PITCH_ERR])].into_iter().enumerate()
		{
			acc[k] = 0.99 * acc[k] + e;
			acc_off[k] = 0.99 * acc_off[k] + eo;
			assert_eq!(f[ROLL_I + k].to_bits(), acc[k].to_bits(), "step {t}: integral {k} (on)");
			assert_eq!(fo[ROLL_I + k].to_bits(), acc_off[k].to_bits(), "step {t}: integral {k} (off)");
		}
		moved = moved.max((f[ROLL_ERR] - fo[ROLL_ERR]).abs());
		abp_alive |= f[ABP].abs() > 1e-5 && f[ABP + 1].abs() > 1e-5;
		ref_alive |= roll_ref.abs() > 1e-3 && pitch_ref.abs() > 1e-3;
	}
	assert!(moved > 0.01 && abp_alive && ref_alive, "vacuous fixture (moved {moved}, abp {abp_alive}, ref {ref_alive})");
}

/// TRAIN = DEPLOY: a replay recomputes the whole frame — the new inputs included —
/// from the streams it already re-applies (horiz, att_up, att_bias); nothing new is
/// recorded. Blanking the horizontal stream breaks it (the reference is really read).
#[test]
fn tre_replay_equals_live()
{
	use crate::dagger_train::{rollout_and_label_inner, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = tre_cfg();
	cfg.steps_per_episode = 400;
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(0), af.sim());
	let mut c = tre_student(3, 0, true, 6, BINARY);
	let mut rng = rand::rngs::SmallRng::seed_from_u64(51);
	let mut cap = crate::imitation_probe::ImitCapture::feature_sampler();
	let tr = rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], Some(&mut cap));
	let n = tr.steps;
	assert!(n >= 50 && cap.feats.len() == n, "the capture must hold every physical step ({} of {n})", cap.feats.len());
	let ro = crate::controller::ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_att_bias(&tr.att_bias, 0, n);
	let replay = |ro: &crate::controller::ReplayObs| {
		let mut rep = tre_student(3, 0, true, 6, BINARY);
		rep.replay_obs_guard(Some(ro), n, "test");
		rep.reset(tr.init_yaw);
		(0..n)
			.map(|t| {
				rep.apply_replay_obs(Some(ro), t);
				rep.compute_features(tr.gyros[t], tr.accels[t], tr.targets[t])
			})
			.collect::<Vec<_>>()
	};
	let frames = replay(&ro);
	for t in 0..n
	{
		let (a, b): (Vec<u32>, Vec<u32>) = (frames[t].iter().map(|v| v.to_bits()).collect(), cap.feats[t].iter().map(|v| v.to_bits()).collect());
		assert_eq!(a, b, "step {t}: the replayed frame is not the live one");
	}
	let mut blank = ro.clone();
	blank.horiz.iter_mut().for_each(|h| *h = [0.0; 4]);
	let moved = replay(&blank).iter().zip(&cap.feats).filter(|(a, b)| a[ROLL_ERR] != b[ROLL_ERR]).count();
	assert!(moved > n / 2, "the tilt reference barely reads the horizontal stream ({moved}/{n})");
	// And the option itself moved the live inputs on this flight.
	let off_live = {
		let (mut teacher, mut sim) = (af.teacher(0), af.sim());
		let mut c = tre_student(3, 0, false, 6, BINARY);
		let mut rng = rand::rngs::SmallRng::seed_from_u64(51);
		let mut cap = crate::imitation_probe::ImitCapture::feature_sampler();
		rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], Some(&mut cap));
		cap.feats
	};
	assert!(off_live.iter().zip(&cap.feats).any(|(a, b)| (a[ROLL_ERR] - b[ROLL_ERR]).abs() > 0.01), "L1 never moved the live roll_err");
}

/// CPU scorer == Metal score kernel with the option on, and it MOVES the result on
/// both sides (the kernel reads the new uniforms, not a dead flag).
#[cfg(target_os = "macos")]
#[test]
fn tre_score_cpu_matches_gpu()
{
	use crate::abi32_tests::{assert_parity_steps, cfg, differs, gpu, tilt_ref};
	let plant = crate::plant_opts::PlantOpts::new(true, 0.78).unwrap();
	let off = assert_parity_steps(&tre_student(19, 0, false, 256, BINARY), &cfg(plant, tilt_ref()), 60, "tre off");
	let on = assert_parity_steps(&tre_student(19, 0, true, 256, BINARY), &cfg(plant, tilt_ref()), 60, "tre on");
	differs(&off, &on, &[0, 1, 13], "obs_tilt_ref_err");
	if metal::Device::system_default().is_some()
	{
		let s1 = cfg(plant, tilt_ref());
		let g_off = gpu(&tre_student(19, 0, false, 256, BINARY), &s1, 60).expect("GPU evaluator");
		let g_on = gpu(&tre_student(19, 0, true, 256, BINARY), &s1, 60).expect("GPU evaluator");
		assert!([0usize, 1, 13].iter().any(|&i| (g_off[i] - g_on[i]).abs() > 1e-9), "GPU kernel ignored obs_tilt_ref_err");
	}
	// Long horizon, sparse (mostly hovering) policy.
	assert_parity_steps(&tre_student(23, 0, true, 6, BINARY), &cfg(plant, tilt_ref()), 400, "tre on 400");
}

/// Metal split record/train == the CPU split replays with the option on (real
/// rollouts, sn > 0), the replay kernels really recompute the input (the rows differ
/// from the option-off student's on the same streams), and blanking a stream it is
/// built from changes the CPU rows.
#[cfg(target_os = "macos")]
#[test]
fn tre_gpu_split_matches_cpu()
{
	let mut cfg = crate::ctrl40_tests::zero_yaw(crate::abi38_tests::l4c_cfg(0.0, 0), 0.0);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	let mut c = tre_student(29, 4, true, 6, QUAD_WEIGHTED);
	let trs = crate::ctrl40_tests::rollouts(&mut c, &cfg, 3);
	let Some((rec_mism, recs, cell_mism, cells, sens)) =
		crate::ctrl40_tests::parity(&c, &trs, |o| o.horiz.iter_mut().for_each(|v| *v = [0.0; 4]))
	else
	{
		assert!(metal::Device::system_default().is_none() || std::env::var("WNN_SKIP_GPU_TESTS").is_ok(), "GPU present but parity skipped");
		return;
	};
	assert!(recs > 0 && cells > 0, "vacuous split parity");
	assert_eq!(rec_mism, 0, "{rec_mism}/{recs} record rows differ (tilt-ref error)");
	assert_eq!(cell_mism, 0, "{cell_mism}/{cells} trained cells differ (tilt-ref error)");
	assert!(sens, "blanking the horizontal stream did not change the CPU rows");
	// The option-off twin (same seed ⇒ same connections and thresholds) records
	// different rows on the SAME streams: the replay is not ignoring the option.
	let obs: Vec<crate::controller::ReplayObs> = trs.iter().map(crate::ctrl40_tests::replay_obs).collect();
	let (g, a, t, p) = crate::ctrl40_tests::streams(&trs);
	let (rows_on, ..) = c.clone().split_record_pub(g.clone(), a.clone(), t.clone(), p.clone(), Some(&obs));
	let (rows_off, ..) = tre_student(29, 4, false, 6, QUAD_WEIGHTED).split_record_pub(g, a, t, p, Some(&obs));
	assert_ne!(rows_on, rows_off, "the split replay records the same rows with and without the option");
}

/// The whole DAgger trainer with the option on: the GPU walk == the CPU walk (cells,
/// egress order, every statistic), and the option reaches the trainer (the off twin
/// trains a different memory).
#[cfg(target_os = "macos")]
#[test]
fn tre_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	// QUAD_WEIGHTED (the default mode) and BINARY (the mode block 10 flies).
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		tre_trainer_parity_in(mode);
	}
}

#[cfg(target_os = "macos")]
fn tre_trainer_parity_in(mode: u8)
{
	let cfg = tre_cfg();
	let run = |on: bool, gpu: bool| {
		let mut c = tre_student(41, 4, on, 6, mode);
		let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
		assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
		let ex = c.export_cells();
		(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
	};
	let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
	let gpu = run(true, true);
	assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "GPU walk never ran");
	assert_eq!(run(true, false), gpu, "GPU walk diverged from the CPU walk with obs_tilt_ref_err");
	assert_ne!(run(false, false), gpu, "obs_tilt_ref_err never reached the trainer");
}
