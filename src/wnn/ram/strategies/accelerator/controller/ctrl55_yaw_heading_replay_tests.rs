//! Lever 1 plumbing (ABI 55, Luiz 16/11/2026): `obs_yaw_from_att_est` end to end — the
//! rollout records the Mahony heading (TrajectoryRs / ReplayObs.yaw_heading), every replay
//! applies it verbatim, the Metal score kernel computes it from its own filter and the
//! train / record kernels read it from yh_rec.
//!
//! Pinned: replay == live (CPU, bit for bit) and the stream is read; CPU scorer == Metal
//! scorer (float tolerance: the CPU filter is f64, the kernel's f32 — as every estimator
//! row) and the option moves both; Metal record / train == CPU split replays (bit for bit);
//! the whole trainer's GPU walk == its CPU walk in QUAD_WEIGHTED and BINARY. Off is the
//! ABI-54 path: every pre-existing pin/parity test is untouched by construction.

use crate::abi38_tests::l4c_cfg;
use crate::controller::{ReplayObs, WnnController};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt, YAW_ERR};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

fn est_student(sn: usize, ar: usize, seed: u64, on: bool, mode: u8) -> WnnController
{
	let mut c = yaw_student(sn, ar, seed, YawOpt::Off, mode);
	c.enable_att_estimator(2.0, 0.1).unwrap();
	c.set_yaw_from_att_est(on).unwrap();
	c
}

#[test]
fn yfe_replay_equals_live()
{
	use crate::dagger_train::rollout_and_label_inner;
	use crate::dagger_train::AirframeRs;
	use rand::SeedableRng;
	let mut cfg = l4c_cfg(0.20, 4);
	cfg.steps_per_episode = 400;
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut c = est_student(0, 10, 3, true, BINARY);
	c.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = rand::rngs::SmallRng::seed_from_u64(55);
	let mut cap = crate::imitation_probe::ImitCapture::feature_sampler();
	let tr = rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], Some(&mut cap));
	let n = tr.steps;
	assert!(n >= 50 && cap.feats.len() == n && tr.yaw_heading.len() == n, "heading stream {} for {n} steps", tr.yaw_heading.len());
	let ro = crate::ctrl40_tests::replay_obs(&tr);
	let replay = |ro: &ReplayObs| {
		let mut rep = est_student(0, 10, 3, true, BINARY);
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
	blank.yaw_heading.iter_mut().for_each(|v| *v += 0.05);
	let moved = replay(&blank).iter().zip(&cap.feats).filter(|(a, b)| a[YAW_ERR] != b[YAW_ERR]).count();
	assert_eq!(moved, n, "the replay does not read the heading stream ({moved}/{n})");
	// And the off twin refuses nothing / needs nothing: its guard passes without the stream.
	let mut off = est_student(0, 10, 3, false, BINARY);
	let mut no_yh = ro.clone();
	no_yh.yaw_heading.clear();
	off.replay_obs_guard(Some(&no_yh), n, "test");
	off.reset(tr.init_yaw);
}

#[test]
#[should_panic(expected = "obs_yaw_from_att_est is on but the replay heading stream")]
fn yfe_replay_guard_refuses_missing_stream()
{
	let c = est_student(0, 10, 3, true, BINARY);
	let ro = ReplayObs { vert: vec![[0.0; 3]; 4], horiz: vec![[0.0; 4]; 4], att_up: vec![[0.0, 0.0, 1.0]; 4], cascade: vec![[0.0; 6]; 4], ..Default::default() };
	c.replay_obs_guard(Some(&ro), 4, "test");
}

/// CPU scorer vs the Metal rollout with the option, both holds; the option moves the
/// result on both sides.
#[cfg(target_os = "macos")]
#[test]
fn yfe_score_cpu_matches_gpu()
{
	use crate::abi38_pins::{cpu_rows, gpu_rows};
	use crate::dagger_train::dagger_train_inplace_with;
	for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
	{
		let train = |on: bool| {
			let mut c = est_student(0, ar, 0x5501 + ar as u64, on, BINARY);
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
			c
		};
		let (on, off) = (train(true), train(false));
		let (cpu, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag));
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "ar {ar}: the option did not move the CPU result");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_off = gpu_rows(&off, 800, lag).unwrap();
		eprintln!("[yfe] ar={ar} T={lag}: err cpu {:.6} gpu {:.6} (off {:.6} / {:.6})", cpu[1], gpu[1], cpu_off[1], gpu_off[1]);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
		assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "ar {ar}: the GPU ignored obs_yaw_from_att_est");
	}
}

/// controller_record / controller_train with yh_rec == the CPU split replays, bit for bit;
/// the replays read the stream.
#[cfg(target_os = "macos")]
#[test]
fn yfe_gpu_split_matches_cpu()
{
	let cfg = crate::ctrl40_tests::zero_yaw(l4c_cfg(0.20, 4), 0.20);
	let mut c = est_student(4, 1, 0x5502, true, BINARY);
	let trs = crate::ctrl40_tests::rollouts(&mut c, &cfg, 3);
	assert!(trs.iter().all(|t| t.yaw_heading.len() == t.gyros.len()), "rollout did not record the heading");
	let fresh = est_student(4, 1, 0x5502, true, BINARY);
	crate::ctrl40_tests::check("Mahony heading (lever 1)", crate::ctrl40_tests::parity(&fresh, &trs, |o| o.yaw_heading.iter_mut().for_each(|v| *v += 0.05)));
}

/// The whole DAgger trainer: GPU walk == CPU walk with the option, both memory modes, and
/// the option reaches the trainer (BINARY; see below).
#[cfg(target_os = "macos")]
#[test]
fn yfe_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |on: bool, gpu: bool| {
			let mut c = est_student(4, 1, 0x5541, on, mode);
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(true, true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(true, false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with obs_yaw_from_att_est");
		// Sensitivity is asserted on BINARY (arm C's --memory-mode): on this 4-episode fixture
		// the QUAD nudge settles the few differing writes to the same cells (off == on),
		// while BINARY's last-write cells differ. Walk parity is asserted in both modes.
		if mode == BINARY
		{
			assert_ne!(run(false, false), gpu, "mode {mode}: obs_yaw_from_att_est never reached the trainer");
		}
	}
}
