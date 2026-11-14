//! L2y of block 10 (ABI 51, `--obs-yaw-trim-err`): the student's yaw error measured
//! against the yaw target shifted by the trim the cascade's yaw rate loop rides on,
//! target − κ·cresid_z (yaw_trim_err.rs, docs/ctrl48_block10_levers_spec.md §L2y).
//!
//! Pinned here: the refusals; the yaw inputs (and the integral) equal the definition on
//! a hand-driven state while every other input — the trailing residual block included —
//! is untouched by ticking the cascade block early; a replay reproduces the live frames
//! (train = deploy) from the cascade stream it already re-applies; CPU scorer == Metal
//! score kernel; Metal split record/train == the CPU split replays; the whole DAgger
//! trainer on the CPU walk == the GPU walk.

use crate::abi38_tests::l4c_cfg;
use crate::controller::{ReplayObs, WnnController};
use crate::ctrl38_tests::obs_student3;
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

/// obs_student3 layout: 9 base, roll/pitch err (9, 10), roll/pitch I (11, 12),
/// yaw_err (13), yaw_err_i (14), 3 vertical, 4 horizontal, then the residual block
/// (its last slot is cresid_z).
const YAW_ERR: usize = 13;
const YAW_I: usize = 14;

/// The production-shaped cascade student (obs_student3's features, widths and
/// residual block) with the yaw inputs made to MATTER: fine cuts on yaw_err /
/// yaw_err_i and half the output bits on them, so the option steers the policy (a
/// random ±0.4 ladder would leave a 0.01-rad shift inside one bin). `on` adds L2y.
fn yte_student_mode(sn: usize, ar: usize, seed: u64, on: bool, mode: u8) -> WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn, sbpn) = (8usize, 3usize, 2usize, 10usize, 6usize);
	let base_nf = 22usize;
	let nf = base_nf + 3;
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..base_nf * bpf)
		.map(|i| match (i / bpf, i % bpf)
		{
			(YAW_ERR, k) => [-0.006f32, 0.0, 0.006][k],
			(YAW_I, k) => [-0.3f32, 0.0, 0.3][k],
			_ => rng.gen_range(-0.4f32..0.4),
		})
		.collect();
	let (state_in, out_in) = (window * nf * bpf + sn, nf * bpf + sn);
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..state_in) as i64).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn)
		.map(|k| if k % 2 == 0 { ((YAW_ERR + (k / 2) % 2) * bpf + rng.gen_range(0..bpf)) as i64 } else { rng.gen_range(0..out_in) as i64 })
		.collect();
	let decode = (mode == BINARY).then_some(crate::cell_mode::DECODE_ANTAGONIST);
	let mut c = WnnController::new_core(
		4, levels, bpf, window, sn, if sn > 0 { sbpn } else { 0 }, obpn, thresholds, sconn, oconn,
		true, 0.1, 0.95, 1.0,
		false, false, true, true, false, false, true, true,
		0.99, 1.0, crate::abi38_pins::DT, false, ar,
		mode, decode, None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("yte student must construct");
	let resid_thr: Vec<f32> = (0..3 * bpf).map(|i| [-0.25f32, 0.0, 0.25][i % 3]).collect();
	c.enable_ctbr_with_obs(crate::ctrl38_tests::obs_spec3(false, false, true), resid_thr).expect("cascade + residual");
	assert_eq!(c.num_features(), nf);
	if on
	{
		c.enable_yaw_trim_err().expect("L2y on");
		assert_eq!(c.num_features(), nf, "L2y must not change the layout");
	}
	c
}

fn yte_student(sn: usize, ar: usize, seed: u64, on: bool) -> WnnController
{
	yte_student_mode(sn, ar, seed, on, BINARY)
}

#[test]
fn yte_refusals_and_default_off()
{
	// No residual block (a2 alone): refused.
	let mut c = obs_student3(0, 10, 1, false, true, false);
	assert!(c.enable_yaw_trim_err().is_err(), "needs the residual block");
	// Default off, then on, then twice.
	let mut c = yte_student(0, 10, 1, false);
	assert_eq!(c.yaw_trim_kappa(), None, "default off");
	assert_eq!(c.obs_yaw_trim_err(), None);
	c.enable_yaw_trim_err().unwrap();
	let k = c.obs_yaw_trim_err().expect("on");
	assert!(k > 0.03 && k < 0.06, "kappa {k}");
	assert!(c.enable_yaw_trim_err().is_err(), "enabled twice");
	// κ follows the student's hold (the plan's sample time).
	assert_ne!(yte_student(0, 1, 1, true).yaw_trim_kappa(), Some(k));
	// No yaw-error input at all: a motor-path student without the yaw channel.
	let mut bare = WnnController::new_core(
		4, 4, 3, 2, 0, 0, 8, vec![0.0; (9 + 2) * 3], Vec::new(), vec![0; 16 * 8],
		true, 0.05, 0.98, 1.0,
		false, false, true, false, false, false, false, false,
		0.99, 1.0, 0.001, false, 1, BINARY, None, None, 0.05, false, 0.30,
		false, false, false, false, false, false, 1,
	)
	.expect("bare student");
	assert!(bare.enable_yaw_trim_err().is_err(), "needs a yaw-error input and the cascade");
}

/// The yaw input equals the DEFINITION — (target − κ·cresid_z) − heading, with cresid_z
/// this step's residual — its integral integrates that value, and NOTHING else moves
/// (the residual block itself, ticked earlier now, is bit-identical). Off is the ABI-50
/// input, target − heading.
#[test]
fn yte_input_equals_the_definition()
{
	let (mut on, mut off) = (yte_student(0, 1, 7, true), yte_student(0, 1, 7, false));
	let kappa = on.yaw_trim_kappa().unwrap();
	for c in [&mut on, &mut off]
	{
		c.reset(0.01);
	}
	let resid = on.cascade_obs_ref().and_then(|o| o.residual()).unwrap();
	let (mut acc, mut heading) = (0.0f32, 0.01f32);
	let (mut moved, mut alive) = (0.0f32, false);
	for t in 0..600
	{
		let s = t as f32;
		let gyro = [0.02 * (0.01 * s).sin(), -0.015 * (0.013 * s).cos(), 0.05 * (0.03 * s).sin() + 0.002];
		let accel = [0.3 * (0.004 * s).sin(), -0.2 * (0.005 * s).cos(), 9.81];
		let tg = [0.0, 0.0, 0.003];
		let (f, fo) = (on.compute_features(gyro, accel, tg), off.compute_features(gyro, accel, tg));
		for i in (0..f.len()).filter(|&i| i != YAW_ERR && i != YAW_I)
		{
			assert_eq!(f[i].to_bits(), fo[i].to_bits(), "step {t}: input {i} moved (only yaw_err and its integral may)");
		}
		heading += gyro[2] * crate::abi38_pins::DT; // the anchored heading: dt-integrated gyro z
		let r = resid.of(&on.cascade_obs_last())[2];
		assert_eq!(f[f.len() - 1].to_bits(), r.to_bits(), "step {t}: the tail is this step's cresid_z");
		assert_eq!(fo[YAW_ERR].to_bits(), (tg[2] - heading).to_bits(), "step {t}: off yaw_err");
		let want = (tg[2] - kappa * r) - heading;
		assert_eq!(f[YAW_ERR].to_bits(), want.to_bits(), "step {t}: yaw_err {} vs definition {want}", f[YAW_ERR]);
		acc = 0.99 * acc + f[YAW_ERR];
		assert_eq!(f[YAW_I].to_bits(), acc.to_bits(), "step {t}: the integral integrates the input");
		moved = moved.max((f[YAW_ERR] - fo[YAW_ERR]).abs());
		alive |= r.abs() > 1e-3;
	}
	assert!(moved > 1e-4 && alive, "vacuous fixture (moved {moved}, residual alive {alive})");
}

/// TRAIN = DEPLOY: a replay recomputes the whole frame — the new input included — from
/// the cascade stream it already re-applies; nothing new is recorded. Blanking that
/// stream breaks it (the trim is really read).
#[test]
fn yte_replay_equals_live()
{
	use crate::dagger_train::{rollout_and_label_inner, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = l4c_cfg(0.20, 4);
	cfg.steps_per_episode = 400;
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut c = yte_student(0, 10, 3, true);
	c.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = rand::rngs::SmallRng::seed_from_u64(52);
	let mut cap = crate::imitation_probe::ImitCapture::feature_sampler();
	let tr = rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], Some(&mut cap));
	let n = tr.steps;
	assert!(n >= 50 && cap.feats.len() == n, "the capture must hold every physical step ({} of {n})", cap.feats.len());
	let ro = ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_cascade(&tr.cascade_obs, 0, n);
	let replay = |ro: &ReplayObs| {
		let mut rep = yte_student(0, 10, 3, true);
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

/// CPU scorer == Metal score kernel with the option on, after training, for both
/// holds; and it MOVES the result on both sides (the kernel reads the new uniforms).
#[cfg(target_os = "macos")]
#[test]
fn yte_score_cpu_matches_gpu()
{
	use crate::abi38_pins::{cpu_rows, gpu_rows};
	use crate::dagger_train::dagger_train_inplace_with;
	for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
	{
		let train = |on: bool| {
			let mut c = yte_student(0, ar, 0x5121 + ar as u64, on);
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
			c
		};
		let (on, off) = (train(true), train(false));
		let (cpu, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag));
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "ar {ar}: L2y did not move the CPU result");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_off = gpu_rows(&off, 800, lag).unwrap();
		eprintln!("[l2y] ar={ar} T={lag}: err cpu {:.6} gpu {:.6} (off: {:.6} / {:.6})", cpu[1], gpu[1], cpu_off[1], gpu_off[1]);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
		assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "ar {ar}: the GPU kernel ignored obs_yaw_trim_err");
	}
}

/// Metal split record/train == the CPU split replays with the option on, the replay
/// kernels really recompute the input (the option-off twin records different rows on
/// the SAME streams), and blanking the cascade stream changes the CPU rows.
#[cfg(target_os = "macos")]
#[test]
fn yte_gpu_split_matches_cpu()
{
	let cfg = crate::ctrl40_tests::zero_yaw(l4c_cfg(0.20, 4), 0.20);
	let mut c = yte_student(4, 1, 0x5102, true);
	let trs = crate::ctrl40_tests::rollouts(&mut c, &cfg, 3);
	let fresh = yte_student(4, 1, 0x5102, true);
	crate::ctrl40_tests::check("yaw-trim error", crate::ctrl40_tests::parity(&fresh, &trs, |o| o.cascade.iter_mut().for_each(|v| *v = [0.0; 6])));
	let obs: Vec<ReplayObs> = trs.iter().map(crate::ctrl40_tests::replay_obs).collect();
	let (g, a, t, p) = crate::ctrl40_tests::streams(&trs);
	let (rows_on, ..) = fresh.clone().split_record_pub(g.clone(), a.clone(), t.clone(), p.clone(), Some(&obs));
	let (rows_off, ..) = yte_student(4, 1, 0x5102, false).split_record_pub(g, a, t, p, Some(&obs));
	assert_ne!(rows_on, rows_off, "the split replay records the same rows with and without the option");
}

/// The whole DAgger trainer with the option on: the GPU walk == the CPU walk (cells,
/// egress order, every statistic), and the option reaches the trainer.
#[cfg(target_os = "macos")]
#[test]
fn yte_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |on: bool, gpu: bool| {
			let mut c = yte_student_mode(4, 1, 0x5141, on, mode);
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(true, true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(true, false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with obs_yaw_trim_err");
		assert_ne!(run(false, false), gpu, "mode {mode}: obs_yaw_trim_err never reached the trainer");
	}
}
