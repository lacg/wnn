//! CTRL-40 (ABI 43): the Metal state-split train/record path for the student's
//! applied-actuator inputs — the motor-level d̂ (obs_dhat), the CTRL-38 cascade blocks
//! (rate-I, cascade d̂) and the lagged obs_pwm. Until ABI 42 the first two forced the
//! CPU split (dagger_train use_gpu_split). The kernels now re-apply the RECORDED values
//! (TrainBatch.dhat_rec / cas_rec / pwm_dev) exactly as the CPU replay does
//! (ReplayObs → apply_replay_obs), so:
//!
//!   * record parity: the Metal record kernel's per-step output-layer inputs equal the
//!     CPU split_record's, bit for bit, on REAL rollouts of each block;
//!   * train parity: the Metal train kernel writes the same cells as the CPU
//!     split_retrain_output (the run_controller_train_parity_test discipline);
//!   * sensitivity: zeroing the block's stream changes the CPU rows (not vacuous);
//!   * the bptt replay switch to the recorded d̂ is a no-op: recomputing d̂ from
//!     student_pwms (Fix A) and re-applying the recorded value give identical features.

use crate::abi38_pins::{ARM, DT, G, INERTIA, KD, KT};
use crate::abi38_tests::l4c_cfg;
use crate::controller::{ReplayObs, WnnController};
use crate::dagger_train::{flatten_gated, rollout_and_label_rs, AirframeRs, RewardGatedConfigPacked, TrajectoryRs};
use crate::metal_controller::{ControllerTrainer, TrainBatch};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// A motor-level delta student (sn = 4) with obs_pwm + the L1 d̂ observer.
fn motor_student(seed: u64) -> WnnController
{
	let (levels, bpf, window, obpn, sn, sbpn, nf) = (8usize, 3usize, 2usize, 10usize, 4usize, 6usize, 16usize);
	let mut rng = SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..nf * bpf)
		.map(|i| if (13..16).contains(&(i / bpf)) { rng.gen_range(-20.0f32..20.0) } else { rng.gen_range(-0.2f32..0.2) })
		.collect();
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..window * nf * bpf + sn) as i64).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..nf * bpf + sn) as i64).collect();
	let b = crate::controller::calibrate_control_gains_rs(DT, ARM, KT, KD, INERTIA, G, 0.5, 0.05);
	WnnController::new_core(
		4, levels, bpf, window, sn, sbpn, obpn, thresholds, sconn, oconn,
		true, 0.1, 0.95, 1.0,
		false, false, false, false, true, true, false, false,
		0.99, 1.0, DT, false, 1,
		ram_core::neuron_memory::QUAD_WEIGHTED, None, Some(b), 0.05, false, 0.30,
		false, false, false, false, false, false, 1,
	)
	.expect("ctrl40 motor student")
}

/// Zero-yaw starts (the split replays seed yaw from init_q; the CPU split_record_pub
/// reference has no per-episode yaw seed) — every other draw is the config's.
fn zero_yaw(mut cfg: RewardGatedConfigPacked, lag: f32) -> RewardGatedConfigPacked
{
	cfg.max_initial_yaw_rad = 0.0;
	cfg.motor_lag_s = lag;
	cfg.steps_per_episode = 250;
	cfg
}

fn motor_cfg(lag: f32) -> RewardGatedConfigPacked
{
	let mut cfg = l4c_cfg(lag, 4);
	cfg.translation = false;
	cfg.xy_offset = 0.0;
	cfg.att_err_tilt_ref = false;
	zero_yaw(cfg, lag)
}

fn rollouts(c: &mut WnnController, cfg: &RewardGatedConfigPacked, n: usize) -> Vec<TrajectoryRs>
{
	let af = AirframeRs::from_cfg(cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	c.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = SmallRng::seed_from_u64(0x40);
	(0..n).map(|_| rollout_and_label_rs(c, &mut teacher, &mut sim, cfg, 0.08, &mut rng, [0.0; 3])).collect()
}

fn replay_obs(t: &TrajectoryRs) -> ReplayObs
{
	ReplayObs {
		vert: t.vert_obs.clone(),
		horiz: t.horiz_obs.clone(),
		pwm_dev: t.pwm_dev.clone(),
		att_up: t.att_up.clone(),
		cascade: t.cascade_obs.clone(),
		dhat: t.dhat_obs.clone(),
	}
}

type Streams = (Vec<Vec<[f32; 3]>>, Vec<Vec<[f32; 3]>>, Vec<Vec<[f32; 3]>>, Vec<Vec<[f32; 4]>>);

fn streams(trs: &[TrajectoryRs]) -> Streams
{
	(
		trs.iter().map(|t| t.gyros.clone()).collect(),
		trs.iter().map(|t| t.accels.clone()).collect(),
		trs.iter().map(|t| t.targets.clone()).collect(),
		trs.iter().map(|t| t.pid_pwms.clone()).collect(),
	)
}

/// GPU record + train vs the CPU split replays on `trs`. Returns (record mismatches,
/// records, train cell mismatches, cells, sensitivity) or None without a GPU.
fn parity(c: &WnnController, trs: &[TrajectoryRs], blank: impl Fn(&mut ReplayObs)) -> Option<(usize, usize, usize, usize, bool)>
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return None;
	}
	let gated: Vec<&TrajectoryRs> = trs.iter().collect();
	let fb = flatten_gated(&gated);
	let batch = TrainBatch {
		ep_base: &fb.ep_base,
		ep_count: &fb.ep_count,
		step_base: &fb.step_base,
		step_count: &fb.step_count,
		gyros: &fb.gyros,
		accels: &fb.accels,
		targets: &fb.targets,
		pid_pwms: &fb.pids,
		init_q: &fb.init_q,
		vert_obs: &fb.vert_obs,
		horiz_obs: &fb.horiz_obs,
		pwm_dev: &fb.pwm_dev,
		att_up: &fb.att_up,
		dhat_rec: &fb.dhat_rec,
		cas_rec: &fb.cas_rec,
		selective: false,
		target_rpy: [0.0; 3],
	};
	let trainer = ControllerTrainer::new().expect("trainer");
	let obs: Vec<ReplayObs> = trs.iter().map(replay_obs).collect();
	let (g, a, t, p) = streams(trs);
	// record
	let gpu_rec = trainer.record(&[c], &batch).expect("gpu record");
	let (cpu_rows, ..) = c.clone().split_record_pub(g.clone(), a.clone(), t.clone(), p.clone(), Some(&obs));
	let rec_mism = gpu_rec.iter().zip(&cpu_rows).filter(|(gr, cr)| gr.0 != **cr).count() + gpu_rec.len().abs_diff(cpu_rows.len());
	// sensitivity
	let mut blanked = obs.clone();
	blanked.iter_mut().for_each(&blank);
	let (rows_blank, ..) = c.clone().split_record_pub(g.clone(), a.clone(), t.clone(), p.clone(), Some(&blanked));
	// train
	let gpu = trainer.train_seeded(&[c], &batch).expect("gpu train");
	let mut cc = c.clone();
	cc.split_retrain_output_pub(&g, &a, &t, &p, Some(&obs), false);
	// Apply the GPU cells exactly as split_train_loop_gpu does (set_output_cell —
	// the canonicalising write path), then compare the two memories cell for cell.
	let gc = c.clone();
	for (n, gpu_n) in gpu.iter().enumerate()
	{
		for &(addr, val) in gpu_n
		{
			gc.set_output_cell(n, addr, val);
		}
	}
	let (mut mism, mut cells) = (0usize, 0usize);
	for n in 0..gpu.len()
	{
		let (ge, ce) = (gc.output_entries(n), cc.output_entries(n));
		let mut all: std::collections::BTreeSet<u64> = ge.iter().map(|&(a, _)| a).collect();
		all.extend(ce.iter().map(|&(a, _)| a));
		cells += all.len();
		mism += all.iter().filter(|&&ad| gc.output_cell(n, ad) != cc.output_cell(n, ad)).count();
	}
	Some((rec_mism, cpu_rows.len(), mism, cells, rows_blank != cpu_rows))
}

fn check(name: &str, r: Option<(usize, usize, usize, usize, bool)>)
{
	let Some((rm, rn, tm, tn, sens)) = r
	else
	{
		return;
	};
	eprintln!("[ctrl40] {name}: record {rm}/{rn} mismatched, train {tm}/{tn} cells mismatched, stream-sensitive {sens}");
	assert!(rn > 0 && tn > 0, "{name}: vacuous fixture");
	assert_eq!(rm, 0, "{name}: Metal record != CPU split_record");
	assert_eq!(tm, 0, "{name}: Metal train != CPU split_retrain_output");
	assert!(sens, "{name}: the replayed stream does not reach the features (vacuous)");
}

#[test]
fn gpu_split_matches_cpu_motor_dhat_and_lagged_obs_pwm()
{
	for lag in [0.20f32, 0.0]
	{
		let cfg = motor_cfg(lag);
		let mut c = motor_student(0x4001);
		let trs = rollouts(&mut c, &cfg, 3);
		let fresh = motor_student(0x4001);
		check(&format!("motor d̂ lag {lag}"), parity(&fresh, &trs, |o| o.dhat.iter_mut().for_each(|v| *v = [0.0; 3])));
		check(&format!("obs_pwm lag {lag}"), parity(&fresh, &trs, |o| o.pwm_dev.iter_mut().for_each(|v| *v = [0.3; 4])));
	}
}

#[test]
fn gpu_split_matches_cpu_cascade_blocks()
{
	for (ri, dh) in [(false, false), (true, false), (false, true), (true, true)]
	{
		let cfg = zero_yaw(l4c_cfg(0.20, 4), 0.20);
		let mut c = crate::ctrl38_tests::obs_student(4, 1, 0x4002, ri, dh);
		let trs = rollouts(&mut c, &cfg, 3);
		let fresh = crate::ctrl38_tests::obs_student(4, 1, 0x4002, ri, dh);
		let blank = |o: &mut ReplayObs| o.cascade.iter_mut().for_each(|v| *v = [0.0; 6]);
		let r = parity(&fresh, &trs, blank);
		if !(ri || dh)
		{
			eprintln!("[ctrl40] cascade blocks OFF reference: {r:?}");
			continue;
		}
		check(&format!("cascade rate_i={ri} dhat={dh}"), r);
	}
}

/// The bptt replay now re-applies the recorded d̂ when the stream is present; that
/// must equal the Fix-A re-derivation from student_pwms bit for bit, so the switch
/// leaves every obs_dhat bptt lineage unchanged.
#[test]
fn bptt_recorded_dhat_equals_fix_a_rederivation()
{
	for lag in [0.20f32, 0.0]
	{
		let cfg = motor_cfg(lag);
		let mut c = motor_student(0x4003);
		let tr = rollouts(&mut c, &cfg, 1).remove(0);
		let n = tr.steps;
		let run = |with_stream: bool| {
			let mut r = motor_student(0x4003);
			r.set_observer_motor_lag_core(lag);
			r.reset(tr.init_yaw);
			let mut o = replay_obs(&tr);
			if !with_stream
			{
				o.dhat.clear();
			}
			(0..n)
				.map(|t| {
					r.apply_replay_obs(Some(&o), t);
					let f = r.compute_features(tr.gyros[t], tr.accels[t], tr.targets[t]);
					r.observe_applied(tr.student_pwms[t]);
					f
				})
				.collect::<Vec<_>>()
		};
		let (rec, fix_a) = (run(true), run(false));
		assert_eq!(rec, fix_a, "lag {lag}: recorded d̂ != Fix-A re-derivation — the bptt lineage would move");
		assert!(tr.dhat_obs.iter().any(|v| v[0] != 0.0), "lag {lag}: trivially-zero d̂ stream");
	}
}

/// The state-split replays REQUIRE the d̂ stream (no applied-pwm stream there).
#[test]
fn split_replay_refuses_a_dhat_student_without_the_stream()
{
	let cfg = motor_cfg(0.0);
	let mut c = motor_student(0x4004);
	let trs = rollouts(&mut c, &cfg, 1);
	let mut o = replay_obs(&trs[0]);
	o.dhat.clear();
	let (g, a, t, p) = streams(&trs);
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		motor_student(0x4004).split_record_pub(g, a, t, p, Some(&[o]))
	}));
	assert!(r.is_err(), "split_record must refuse an obs_dhat replay without the recorded d̂");
}

/// A rough speed probe (ignored by default): CPU vs Metal split train on the cascade
/// fixture. `cargo test ... ctrl40_speed -- --ignored --nocapture`.
#[test]
#[ignore]
fn ctrl40_speed_cpu_vs_gpu_split()
{
	let cfg = zero_yaw(l4c_cfg(0.20, 4), 0.20);
	let mut c = crate::ctrl38_tests::obs_student(4, 1, 0x4005, true, true);
	let mut cfg_long = cfg.clone();
	cfg_long.steps_per_episode = 2000;
	let trs = rollouts(&mut c, &cfg_long, 16);
	let fresh = crate::ctrl38_tests::obs_student(4, 1, 0x4005, true, true);
	let t0 = std::time::Instant::now();
	let _ = parity(&fresh, &trs, |_| {});
	eprintln!("[ctrl40] parity (gpu record+train + cpu record+train x2) {:?}", t0.elapsed());
	let gated: Vec<&TrajectoryRs> = trs.iter().collect();
	let fb = flatten_gated(&gated);
	let batch = TrainBatch {
		ep_base: &fb.ep_base, ep_count: &fb.ep_count, step_base: &fb.step_base, step_count: &fb.step_count,
		gyros: &fb.gyros, accels: &fb.accels, targets: &fb.targets, pid_pwms: &fb.pids, init_q: &fb.init_q,
		vert_obs: &fb.vert_obs, horiz_obs: &fb.horiz_obs, pwm_dev: &fb.pwm_dev, att_up: &fb.att_up,
		dhat_rec: &fb.dhat_rec, cas_rec: &fb.cas_rec, selective: false, target_rpy: [0.0; 3],
	};
	let trainer = ControllerTrainer::new().expect("trainer");
	let _ = trainer.train(&[&fresh], &batch);
	let t1 = std::time::Instant::now();
	let _ = trainer.train(&[&fresh], &batch);
	let gpu = t1.elapsed();
	let obs: Vec<ReplayObs> = trs.iter().map(replay_obs).collect();
	let (g, a, t, p) = streams(&trs);
	let t2 = std::time::Instant::now();
	fresh.clone().split_retrain_output_pub(&g, &a, &t, &p, Some(&obs), false);
	let cpu = t2.elapsed();
	eprintln!("[ctrl40] output retrain over {} steps: CPU {cpu:?}  Metal {gpu:?}", fb.gyros.len() / 3);
}

