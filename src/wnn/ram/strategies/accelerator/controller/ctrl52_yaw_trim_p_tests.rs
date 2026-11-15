//! L2y' of block 10 (ABI 52, `--obs-yaw-trim-err-p`, Luiz 15/11/2026): ONLY the
//! proportional yaw input carries the trim shift, yaw_err − κ·cresid_z; the yaw integral
//! keeps integrating the RAW heading error (docs/ctrl48_block10_2x2_s1_diagnosis.md:
//! L2y's integral of the shifted error was the defect).
//!
//! Pinned here: refusals (and L2y / L2y' are alternatives); the proportional input equals
//! the definition while the integral is BIT-identical to the option-off twin's and every
//! other input is untouched; replay == live; CPU scorer == the FC_YAW_TRIM_P Metal
//! rollout; the specialised Metal split record/train == the CPU split; the whole trainer's
//! GPU walk == its CPU walk (QUAD_WEIGHTED and BINARY).

use crate::abi38_tests::l4c_cfg;
use crate::controller::ReplayObs;
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt, YAW_ERR, YAW_I};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

fn student(sn: usize, ar: usize, seed: u64, opt: YawOpt) -> crate::controller::WnnController
{
	yaw_student(sn, ar, seed, opt, BINARY)
}

#[test]
fn ytp_refusals_and_alternatives()
{
	let mut c = student(0, 10, 1, YawOpt::Off);
	assert_eq!(c.yaw_trim_p_kappa(), None, "default off");
	assert_eq!(c.obs_yaw_trim_err_p(), None);
	c.enable_yaw_trim_err_p().unwrap();
	let k = c.obs_yaw_trim_err_p().expect("on");
	assert_eq!(Some(k), student(0, 10, 1, YawOpt::L2y).yaw_trim_kappa(), "the same κ as L2y");
	assert!(c.enable_yaw_trim_err_p().is_err(), "enabled twice");
	assert!(c.enable_yaw_trim_err().is_err(), "L2y on top of L2y' is refused");
	assert!(student(0, 10, 1, YawOpt::L2y).enable_yaw_trim_err_p().is_err(), "L2y' on top of L2y is refused");
	let mut no_resid = crate::ctrl38_tests::obs_student3(0, 10, 1, false, true, false);
	assert!(no_resid.enable_yaw_trim_err_p().is_err(), "needs the residual block");
}

/// The proportional input is (target − heading) − κ·cresid_z; the integral is the
/// off twin's BIT for bit (raw heading error); every other input is untouched.
#[test]
fn ytp_proportional_shifted_integral_raw()
{
	let (mut on, mut off) = (student(0, 1, 7, YawOpt::L2yP), student(0, 1, 7, YawOpt::Off));
	let kappa = on.yaw_trim_p_kappa().unwrap();
	for c in [&mut on, &mut off]
	{
		c.reset(0.01);
	}
	let resid = on.cascade_obs_ref().and_then(|o| o.residual()).unwrap();
	let (mut moved, mut alive) = (0.0f32, false);
	for t in 0..600
	{
		let s = t as f32;
		let gyro = [0.02 * (0.01 * s).sin(), -0.015 * (0.013 * s).cos(), 0.05 * (0.03 * s).sin() + 0.002];
		let accel = [0.3 * (0.004 * s).sin(), -0.2 * (0.005 * s).cos(), 9.81];
		let tg = [0.0, 0.0, 0.003];
		let (f, fo) = (on.compute_features(gyro, accel, tg), off.compute_features(gyro, accel, tg));
		for i in (0..f.len()).filter(|&i| i != YAW_ERR)
		{
			assert_eq!(f[i].to_bits(), fo[i].to_bits(), "step {t}: input {i} moved (only the proportional yaw input may; the integral is {YAW_I})");
		}
		let r = resid.of(&on.cascade_obs_last())[2];
		assert_eq!(f[YAW_ERR].to_bits(), (fo[YAW_ERR] - kappa * r).to_bits(), "step {t}: yaw_err {} vs raw − κ·cresid_z", f[YAW_ERR]);
		moved = moved.max((f[YAW_ERR] - fo[YAW_ERR]).abs());
		alive |= r.abs() > 1e-3;
	}
	assert!(moved > 1e-4 && alive, "vacuous fixture (moved {moved}, residual alive {alive})");
}

/// TRAIN = DEPLOY: the replay rebuilds every frame from the cascade stream it already
/// re-applies; blanking that stream moves the proportional input (the trim is read).
#[test]
fn ytp_replay_equals_live()
{
	use crate::dagger_train::{rollout_and_label_inner, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = l4c_cfg(0.20, 4);
	cfg.steps_per_episode = 400;
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut c = student(0, 10, 3, YawOpt::L2yP);
	c.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = rand::rngs::SmallRng::seed_from_u64(52);
	let mut cap = crate::imitation_probe::ImitCapture::feature_sampler();
	let tr = rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], Some(&mut cap));
	let n = tr.steps;
	assert!(n >= 50 && cap.feats.len() == n);
	let ro = ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_cascade(&tr.cascade_obs, 0, n);
	let replay = |ro: &ReplayObs| {
		let mut rep = student(0, 10, 3, YawOpt::L2yP);
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
	blank.cascade.iter_mut().for_each(|v| *v = [0.0; 6]);
	let moved = replay(&blank).iter().zip(&cap.feats).filter(|(a, b)| a[YAW_ERR] != b[YAW_ERR]).count();
	assert!(moved > n / 2, "the trim barely reads the cascade stream ({moved}/{n})");
}

/// CPU scorer == the FC_YAW_TRIM_P Metal rollout, trained, both holds; the option moves
/// the result on both sides, and the result differs from L2y's (the integral differs).
#[cfg(target_os = "macos")]
#[test]
fn ytp_score_cpu_matches_gpu()
{
	use crate::abi38_pins::{cpu_rows, gpu_rows};
	use crate::dagger_train::dagger_train_inplace_with;
	for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
	{
		let train = |opt: YawOpt| {
			let mut c = student(0, ar, 0x5221 + ar as u64, opt);
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
			c
		};
		let (on, off, l2y) = (train(YawOpt::L2yP), train(YawOpt::Off), train(YawOpt::L2y));
		let (cpu, cpu_off, cpu_l2y) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag), cpu_rows(&l2y, 800, lag));
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "ar {ar}: L2y' did not move the CPU result");
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_l2y[i]).abs() > 1e-9), "ar {ar}: L2y' flies exactly like L2y");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_off = gpu_rows(&off, 800, lag).unwrap();
		eprintln!("[l2y'] ar={ar} T={lag}: err cpu {:.6} gpu {:.6} (off {:.6} / {:.6})", cpu[1], gpu[1], cpu_off[1], gpu_off[1]);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
		assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "ar {ar}: the GPU ignored obs_yaw_trim_err_p");
	}
}

/// The FC_YAW_TRIM_P controller_record / controller_train == the CPU split replays; the
/// off twin records different rows on the SAME streams.
#[cfg(target_os = "macos")]
#[test]
fn ytp_gpu_split_matches_cpu()
{
	let cfg = crate::ctrl40_tests::zero_yaw(l4c_cfg(0.20, 4), 0.20);
	let mut c = student(4, 1, 0x5202, YawOpt::L2yP);
	let trs = crate::ctrl40_tests::rollouts(&mut c, &cfg, 3);
	let fresh = student(4, 1, 0x5202, YawOpt::L2yP);
	crate::ctrl40_tests::check("yaw-trim proportional (L2y')", crate::ctrl40_tests::parity(&fresh, &trs, |o| o.cascade.iter_mut().for_each(|v| *v = [0.0; 6])));
	let obs: Vec<ReplayObs> = trs.iter().map(crate::ctrl40_tests::replay_obs).collect();
	let (g, a, t, p) = crate::ctrl40_tests::streams(&trs);
	let (rows_on, ..) = fresh.clone().split_record_pub(g.clone(), a.clone(), t.clone(), p.clone(), Some(&obs));
	let (rows_off, ..) = student(4, 1, 0x5202, YawOpt::Off).split_record_pub(g, a, t, p, Some(&obs));
	assert_ne!(rows_on, rows_off, "the split replay records the same rows with and without the option");
}

/// The whole DAgger trainer: GPU walk == CPU walk with the option, both memory modes.
#[cfg(target_os = "macos")]
#[test]
fn ytp_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |opt: YawOpt, gpu: bool| {
			let mut c = yaw_student(4, 1, 0x5241, opt, mode);
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(YawOpt::L2yP, true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(YawOpt::L2yP, false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with obs_yaw_trim_err_p");
		assert_ne!(run(YawOpt::Off, false), gpu, "mode {mode}: obs_yaw_trim_err_p never reached the trainer");
	}
}
