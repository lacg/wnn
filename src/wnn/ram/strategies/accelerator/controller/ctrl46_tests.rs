//! CTRL-46 (ABI 46): the vote write rule (item 1) and estimate-built DAgger labels
//! (item 2). Flags-off bit-identity is held by the existing pins (abi35_tests
//! cpu_walk_bit_identical_to_pre_abi35 / cpu_dagger_bit_identical_to_pre_abi35,
//! ctrl17_pins, abi38_pins, ctrl38_pins), which run with both defaults.

use crate::abi35_tests::{cells_digest, plan_episodes, walk_controller, walk_episodes, WalkShape};
use crate::bptt_plan::{BpttL4, BpttWindowPlan};
use crate::controller::WnnController;
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED, TERNARY};
use ram_core::vote_share::VoteShare;

/// One vote round over `plans` on the CPU walk: (sorted tally, cells digest after the
/// commit, prev_state after the walk).
fn cpu_vote_round(base: &WnnController, plans: &[&BpttWindowPlan], share: VoteShare) -> (Vec<(bool, u32, u64, i64)>, u64, Vec<u8>)
{
	let mut c = base.clone();
	c.vote_begin_round(share).unwrap();
	for p in plans
	{
		let (s, o) = c.bptt_walk_cpu(p, 4, false, &BpttL4::off());
		assert_eq!((s, o), (0, 0), "a vote walk writes nothing");
	}
	let tally = c.vote_tally_ref().unwrap().sorted();
	let prev = c.bptt_prev_state().to_vec();
	c.vote_commit_round();
	(tally, cells_digest(&c), prev)
}

fn shapes() -> Vec<WalkShape>
{
	let mut v = Vec::new();
	for (i, &mode) in [TERNARY, BINARY].iter().enumerate()
	{
		v.push(WalkShape { mode, sn: 4, sbpn: 8, obpn: 8, levels: 8, bpf: 3, window_k: 2, prod: false, seed: 0x46A + i as u64, tail: 0.3 });
		v.push(WalkShape { mode, sn: 8, sbpn: 24, obpn: 24, levels: 64, bpf: 16, window_k: 2, prod: true, seed: 0x46B + i as u64, tail: 0.3 });
	}
	v
}

/// The vote round's memory is frozen, so the cells it produces cannot depend on the
/// order the round's episodes are walked in. sn=0 (the recipe): every window is
/// independent, so even the window order is free. sn>0: windows of one episode carry
/// the recurrent state, so episodes are permuted as units.
#[test]
fn vote_round_is_order_independent()
{
	let share = VoteShare::from_theta(0.3).unwrap();
	for mode in [TERNARY, BINARY]
	{
		// sn = 0, the production single-layer path: reverse every window.
		let s0 = WalkShape { mode, sn: 0, sbpn: 0, obpn: 24, levels: 64, bpf: 16, window_k: 1, prod: true, seed: 0x460, tail: 0.0 };
		let base = walk_controller(&s0);
		let plans = plan_episodes(&base, &walk_episodes(s0.seed, 4, 90), 32);
		let fwd: Vec<&BpttWindowPlan> = plans.iter().collect();
		let rev: Vec<&BpttWindowPlan> = plans.iter().rev().collect();
		let (a, b) = (cpu_vote_round(&base, &fwd, share), cpu_vote_round(&base, &rev, share));
		assert!(!a.0.is_empty(), "mode {mode}: vacuous sn=0 vote round");
		assert_eq!(a, b, "mode {mode} sn=0: window order changed the vote");
		// sn > 0: permute whole episodes.
		let s = WalkShape { mode, sn: 4, sbpn: 8, obpn: 8, levels: 8, bpf: 3, window_k: 2, prod: false, seed: 0x461, tail: 0.3 };
		let base = walk_controller(&s);
		let eps = walk_episodes(s.seed, 3, 70);
		let per_ep: Vec<Vec<BpttWindowPlan>> = eps.iter().map(|e| plan_episodes(&base, std::slice::from_ref(e), 32)).collect();
		let order = |idx: &[usize]| -> Vec<&BpttWindowPlan> { idx.iter().flat_map(|&i| per_ep[i].iter()).collect() };
		let a = cpu_vote_round(&base, &order(&[0, 1, 2]), share);
		let b = cpu_vote_round(&base, &order(&[2, 0, 1]), share);
		assert_eq!(a.0, b.0, "mode {mode} sn=4: episode order changed the tally");
		assert_eq!(a.1, b.1, "mode {mode} sn=4: episode order changed the cells");
	}
}

/// Last-write-wins IS order-dependent on the same plans (the property the vote fixes):
/// guards the order test above against being vacuous.
#[test]
fn last_write_is_order_dependent_on_the_same_plans()
{
	let s0 = WalkShape { mode: BINARY, sn: 0, sbpn: 0, obpn: 24, levels: 64, bpf: 16, window_k: 1, prod: true, seed: 0x460, tail: 0.0 };
	let base = walk_controller(&s0);
	let plans = plan_episodes(&base, &walk_episodes(s0.seed, 4, 90), 32);
	let run = |rev: bool| {
		let mut c = base.clone();
		let it: Vec<&BpttWindowPlan> = if rev { plans.iter().rev().collect() } else { plans.iter().collect() };
		for p in it
		{
			c.bptt_walk_cpu(p, 4, false, &BpttL4::off());
		}
		cells_digest(&c)
	};
	assert_ne!(run(false), run(true), "fixture too easy: last-write-wins came out order-free");
}

/// Hand check on the sn=0 walk: the committed cell of every visited (neuron, address)
/// is ram_core's vote_net_to_cell of (own·#TRUE − other·#FALSE) over the round.
#[test]
fn sn0_vote_commits_the_min_share_sign()
{
	let s0 = WalkShape { mode: BINARY, sn: 0, sbpn: 0, obpn: 6, levels: 8, bpf: 2, window_k: 1, prod: false, seed: 0x462, tail: 0.0 };
	let base = walk_controller(&s0);
	let plans = plan_episodes(&base, &walk_episodes(s0.seed, 3, 60), 16);
	let share = VoteShare::from_theta(0.25).unwrap();
	let mut c = base.clone();
	c.vote_begin_round(share).unwrap();
	for p in &plans
	{
		c.bptt_walk_cpu(p, 4, false, &BpttL4::off());
	}
	let tally = c.vote_tally_ref().unwrap().sorted();
	c.vote_commit_round();
	let mut contested = 0;
	for &(state, n, a, t) in &tally
	{
		assert!(!state);
		// own:other = 3:1 ⇒ t ≡ 3·T − F with T, F ≥ 0 counts.
		let want = ram_core::vote_share::vote_net_to_cell(t, ram_core::neuron_memory::FALSE_U8);
		// BINARY's FALSE is a deleted entry; a miss reads EMPTY on the raw read.
		let got = match c.output_cell(n as usize, a)
		{
			ram_core::neuron_memory::EMPTY_U8 => ram_core::neuron_memory::FALSE_U8,
			v => v,
		};
		assert_eq!(got, want, "neuron {n} addr {a:#x} tally {t}");
		if t % 3 != 0 && t.abs() < 3 * 4
		{
			contested += 1;
		}
	}
	assert!(contested > 0, "fixture never mixed directions on one address");
}

/// THE CPU/GPU gate for item 1: the GPU walk in a vote round produces the same tally,
/// the same committed cells and the same carried state as the CPU walk, over
/// multi-window resident walks, chunk budgets 1/64/256 and Option-A overrides.
#[cfg(target_os = "macos")]
#[test]
fn vote_walk_cpu_matches_gpu()
{
	if metal::Device::system_default().is_none()
	{
		return;
	}
	let mut fails = Vec::new();
	let mut n = 0;
	for shape in shapes()
	{
		for &(theta, budget) in &[(0.5, 256usize), (0.2, 1), (0.37, 64)]
		{
			n += 1;
			let share = VoteShare::from_theta(theta).unwrap();
			let base = walk_controller(&shape);
			let plans = plan_episodes(&base, &walk_episodes(shape.seed, 3, 64), 32);
			let refs: Vec<&BpttWindowPlan> = plans.iter().collect();
			let cpu = cpu_vote_round(&base, &refs, share);
			let mut g = base.clone();
			g.vote_begin_round(share).unwrap();
			let w = crate::bptt_walk_gpu::walk_plans_gpu_with(&mut g, &plans, 4, false, budget);
			if let Err(e) = w
			{
				fails.push(format!("{shape:?}: GPU refused: {e}"));
				continue;
			}
			assert_eq!(w.unwrap(), (0, 0), "a vote GPU walk writes nothing");
			let tally = g.vote_tally_ref().unwrap().sorted();
			let prev = g.bptt_prev_state().to_vec();
			g.vote_commit_round();
			let gpu = (tally, cells_digest(&g), prev);
			if cpu.0.is_empty() || !cpu.0.iter().any(|e| e.0)
			{
				fails.push(format!("{shape:?}: vacuous (no state-layer visits)"));
			}
			if cpu != gpu
			{
				fails.push(format!("{shape:?} θ={theta} budget={budget}: cpu tally {} keys digest {:#x} != gpu {} keys {:#x}", cpu.0.len(), cpu.1, gpu.0.len(), gpu.1));
			}
		}
	}
	assert!(fails.is_empty(), "{}/{} vote walk cases diverged: {fails:#?}", fails.len(), n);
}

fn vote_cfg(theta: Option<f64>) -> crate::dagger_train::RewardGatedConfigPacked
{
	let mut cfg = crate::ctrl17_pins::pin_translation_cfg(0.0);
	cfg.write_vote = true;
	cfg.vote_share = theta;
	cfg
}

fn dagger_run(mode: u8, sn: usize, cfg: &crate::dagger_train::RewardGatedConfigPacked, gpu: bool) -> (u64, Vec<usize>, String)
{
	let mut c = crate::abi35_tests::dagger_student(mode, sn);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut c, cfg, [0.0; 3], 0x5EED, gpu);
	let fp = format!("{:?} {:?} {:?}", st.iter_fitness, st.iter_mean_err_deg, st.iter_n_trained);
	(cells_digest(&c), st.iter_cells_written.clone(), fp)
}

/// Whole reward-gated DAgger under the vote: CPU walk == GPU walk (cells, per-round
/// writes, every eval statistic — later rounds fly the committed memory).
#[cfg(target_os = "macos")]
#[test]
fn vote_dagger_train_cpu_matches_gpu()
{
	if metal::Device::system_default().is_none()
	{
		return;
	}
	for &(mode, theta) in &[(TERNARY, None), (BINARY, Some(0.3))]
	{
		let cfg = vote_cfg(theta);
		let cpu = dagger_run(mode, 4, &cfg, false);
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = dagger_run(mode, 4, &cfg, true);
		let after = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		assert!(after > before, "mode {mode}: the GPU walk never ran");
		assert!(cpu.1.iter().sum::<usize>() > 0, "mode {mode}: vacuous vote dagger_train");
		assert_eq!(cpu, gpu, "mode {mode}: GPU vote dagger_train diverged from the CPU trainer");
	}
}

/// sn=0 DAgger (the recipe's topology): the vote trains (cells change), is
/// deterministic, and differs from last-write-wins.
#[test]
fn vote_dagger_train_sn0_runs_and_differs_from_last_write()
{
	let last = dagger_run(BINARY, 0, &crate::ctrl17_pins::pin_translation_cfg(0.0), false);
	let v1 = dagger_run(BINARY, 0, &vote_cfg(Some(0.4)), false);
	let v2 = dagger_run(BINARY, 0, &vote_cfg(Some(0.4)), false);
	assert_eq!(v1, v2, "vote dagger_train is not deterministic");
	assert!(v1.1.iter().sum::<usize>() > 0, "vote round committed nothing");
	assert_ne!(v1.0, last.0, "the vote rule changed nothing vs last-write-wins");
}

/// The config + trainer refusals of item 1.
#[test]
fn vote_config_refusals()
{
	use crate::dagger_train::validate_train_regime;
	// vote_share without the vote rule, off-grid, out of range.
	let mut c = crate::ctrl17_pins::pin_translation_cfg(0.0);
	c.vote_share = Some(0.3);
	assert!(c.write_vote_share().is_err(), "vote_share without write_vote");
	for bad in [0.0, 0.51, 0.123]
	{
		assert!(vote_cfg(Some(bad)).write_vote_share().is_err(), "θ={bad}");
	}
	assert_eq!(vote_cfg(None).write_vote_share(), Ok(Some(VoteShare::MAJORITY)));
	// QUAD: its nudge already tallies — refused, not silently ignored.
	let quad = crate::abi35_tests::dagger_student(QUAD_WEIGHTED, 4);
	assert!(validate_train_regime(&quad, &vote_cfg(None)).is_err(), "QUAD + vote");
	let bin = crate::abi35_tests::dagger_student(BINARY, 4);
	assert!(validate_train_regime(&bin, &vote_cfg(None)).is_ok());
	let mut p = vote_cfg(None);
	p.protect_learned = true;
	assert!(validate_train_regime(&bin, &p).is_err(), "protect_learned + vote");
	let mut p = vote_cfg(None);
	p.write_priority_err = true;
	assert!(validate_train_regime(&bin, &p).is_err(), "write_priority_err + vote");
	// Flags off: the default config is the LastWrite writer.
	assert_eq!(crate::ctrl17_pins::pin_translation_cfg(0.0).write_vote_share(), Ok(None));
}

// ---------------------------------------------------------------- item 2 -----

fn label_rollout(estimate: bool, xy: f32) -> crate::dagger_train::TrajectoryRs
{
	use crate::dagger_train::{rollout_and_label_rs, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = crate::ctrl17_pins::pin_translation_cfg(xy);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.35;
	cfg.dagger_label_estimate = estimate;
	let mut c = crate::abi32_tests::controller(31, true);
	let af = AirframeRs::from_cfg(&cfg);
	let mut teacher = af.teacher(cfg.teacher);
	let mut sim = af.sim();
	let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
	rollout_and_label_rs(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3])
}

/// Only the LABEL changes: the student flies the identical trajectory (same IMU,
/// same observations, same applied commands), while the recorded labels move —
/// the specific-force IMU makes the Mahony estimate differ from the truth.
#[test]
fn estimate_label_changes_only_the_label()
{
	let t = label_rollout(false, 0.5);
	let e = label_rollout(true, 0.5);
	assert_eq!(t.steps, e.steps);
	assert_eq!(t.gyros, e.gyros, "the flown trajectory changed");
	assert_eq!(t.accels, e.accels);
	assert_eq!(t.horiz_obs, e.horiz_obs);
	assert_eq!(t.att_up, e.att_up, "the student's observation changed");
	assert_eq!(t.student_pwms, e.student_pwms, "the applied command changed");
	assert_eq!(t.cumulative_reward.to_bits(), e.cumulative_reward.to_bits(), "the gate reward changed");
	let moved = t.pid_pwms.iter().zip(&e.pid_pwms).filter(|(a, b)| a != b).count();
	assert!(moved > t.steps / 2, "labels barely moved ({moved}/{} steps)", t.steps);
}

/// Item 2 refusals: no estimator ⇒ nothing to build the label from; expert_drives ⇒
/// the label would also fly. Default off is the true quaternion.
#[test]
fn estimate_label_refusals()
{
	use crate::dagger_train::validate_train_regime;
	let mut cfg = crate::ctrl17_pins::pin_translation_cfg(0.5);
	cfg.dagger_label_estimate = true;
	let no_est = crate::abi32_tests::controller(31, false);
	assert!(validate_train_regime(&no_est, &cfg).is_err(), "estimate label without an estimator");
	let est = crate::abi32_tests::controller(31, true);
	assert!(validate_train_regime(&est, &cfg).is_ok());
	cfg.expert_drives = true;
	assert!(validate_train_regime(&est, &cfg).is_err(), "estimate label under expert_drives");
	assert!(!crate::ctrl17_pins::pin_translation_cfg(0.5).dagger_label_estimate);
}

/// A whole DAgger run with estimate labels trains (no panic on the live-estimate
/// expect — every step of every rollout had a live estimate) and its trainer
/// consumed different targets than the true-quaternion run.
#[test]
fn estimate_label_dagger_train_runs()
{
	let run = |estimate: bool| {
		let mut cfg = crate::ctrl17_pins::pin_translation_cfg(0.5);
		cfg.imu_specific_force = true;
		cfg.dagger_label_estimate = estimate;
		let mut c = crate::abi32_tests::controller(31, true);
		let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 0x46, false);
		(cells_digest(&c), st.iter_cells_written)
	};
	let (t, e) = (run(false), run(true));
	assert!(e.1.iter().sum::<usize>() > 0);
	// The 4-level fixture quantises most label moves away, so the per-round write
	// count (every walked target, FALSE no-ops included) is the observable that the
	// trainer consumed different labels; the label-level effect is pinned above.
	assert_ne!(t, e, "estimate labels reached the trainer as the true labels");
}

// ---------------------------------------------------------------- item 3 -----

/// A student with the estimator, the stage-1/2 channels and (optionally) the bias
/// predictor: 9 base + roll/pitch err + 3 vertical + 4 horizontal (+ 2 abp). Output
/// connections are drawn over the FINAL layout, so the abp bits are wired;
/// `cells` random output cells per neuron (256 = dense: reacts to every bit).
fn abp_student(seed: u64, sn: usize, abp: bool, cells: usize, mode: u8) -> WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn, sbpn) = (4usize, 3usize, 2usize, 8usize, 6usize);
	let base_nf = 18usize;
	let nf = base_nf + 2 * abp as usize;
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..base_nf * bpf)
		.map(|i| match i / bpf
		{
			9 | 10 => rng.gen_range(-0.3f32..0.3),
			f if f >= 11 => rng.gen_range(-0.6f32..0.6),
			_ => rng.gen_range(-5.0f32..5.0),
		})
		.collect();
	let num_out = 4 * levels;
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..window * nf * bpf + sn) as i64).collect();
	// Half the output bits on the last two features when the block is on, so the
	// prediction visibly steers the policy (sensitivity halves of the parity tests).
	let oconn: Vec<i64> = (0..num_out * obpn)
		.map(|k| if abp && k % 2 == 0 { ((base_nf + (k / 2) % 2) * bpf + rng.gen_range(0..bpf)) as i64 } else { rng.gen_range(0..nf * bpf + sn) as i64 })
		.collect();
	let decode = if mode == BINARY { Some(crate::cell_mode::DECODE_ANTAGONIST) } else { None };
	let mut c = WnnController::new_core(
		4, levels, bpf, window, sn, if sn > 0 { sbpn } else { 0 }, obpn, thresholds, sconn, oconn,
		true, 0.05, 0.98, 1.0,
		false, false, true, false, false, false, false, false,
		0.99, 1.0, 0.001, false, 1, mode, decode, None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("abp student must construct");
	c.enable_att_estimator(2.0, 0.1).expect("estimator on");
	if abp
	{
		let thr: Vec<f32> = (0..2 * bpf).map(|i| [-0.004f32, 0.0, 0.004][i % 3]).collect();
		c.enable_att_bias_pred(thr).expect("abp on");
		assert_eq!(c.num_features(), nf);
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

/// The block refuses to run on a signal the student is not already given, or
/// without the estimator it models; off is the ABI-45 layout exactly.
#[test]
fn abp_refusals_and_layout()
{
	let mut c = crate::abi32_tests::controller(31, false); // no estimator
	assert!(c.enable_att_bias_pred(vec![0.0; 6]).is_err(), "abp without the estimator");
	let mut c = crate::abi35_tests::dagger_student(BINARY, 0); // no estimator, no vel_xy
	assert!(c.enable_att_bias_pred(vec![0.0; 6]).is_err());
	let mut c = abp_student(1, 0, false, 6, BINARY);
	assert!(c.enable_att_bias_pred(vec![0.0; 5]).is_err(), "wrong threshold count");
	let before = c.num_features();
	c.enable_att_bias_pred(vec![0.0; 6]).unwrap();
	assert_eq!(c.num_features(), before + 2);
	assert!(c.enable_att_bias_pred(vec![0.0; 6]).is_err(), "enabled twice");
}

/// The emitted features are the predictor's (live), it is the LAST block, and
/// replaying the recorded stream reproduces every frame bit-for-bit; a replay without
/// the stream is refused.
#[test]
fn abp_live_features_record_and_replay()
{
	use crate::dagger_train::{rollout_and_label_rs, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = crate::ctrl17_pins::pin_translation_cfg(0.5);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	cfg.steps_per_episode = 400;
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(0), af.sim());
	let mut c = abp_student(3, 0, true, 6, BINARY);
	let mut rng = rand::rngs::SmallRng::seed_from_u64(46);
	let tr = rollout_and_label_rs(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3]);
	let n = tr.steps;
	assert_eq!(tr.att_bias.len(), n, "the rollout must record the block");
	assert!(tr.att_bias.iter().any(|v| v[0].abs() > 1e-4) && tr.att_bias.iter().any(|v| v[1].abs() > 1e-4), "trivially zero");
	let ro = crate::controller::ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_att_bias(&tr.att_bias, 0, n);
	let mut rep = abp_student(3, 0, true, 6, BINARY);
	rep.replay_obs_guard(Some(&ro), n, "test");
	rep.reset(tr.init_yaw);
	for t in 0..n
	{
		rep.apply_replay_obs(Some(&ro), t);
		let f = rep.compute_features(tr.gyros[t], tr.accels[t], tr.targets[t]);
		assert_eq!(&f[f.len() - 2..], &tr.att_bias[t][..], "step {t}: replayed frame tail");
	}
	let mut stale = ro.clone();
	stale.att_bias.clear();
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| abp_student(3, 0, true, 6, BINARY).replay_obs_guard(Some(&stale), n, "test")));
	assert!(r.is_err(), "replay_obs_guard must refuse an abp replay without the stream");
}

/// Live features == the standalone model on the controller's OWN onboard inputs: a
/// hand-flown deploy loop where a separate Mahony filter (same gains, same seed) and
/// a separate AttBiasPredictor see only (IMU, horizontal velocity obs).
#[test]
fn abp_live_equals_the_model_on_onboard_inputs()
{
	use crate::dagger_train::AirframeRs;
	let mut cfg = crate::ctrl17_pins::pin_translation_cfg(0.5);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	let af = AirframeRs::from_cfg(&cfg);
	let mut sim = af.sim();
	let q0 = [0.995f32, 0.05, -0.08, 0.02];
	sim.reset(Some(q0), Some([0.1, -0.2, 0.0]));
	sim.set_translation_core(cfg.af_mass).unwrap();
	sim.set_horizontal_state(0.4, -0.3, 0.0, 0.0);
	let mut c = abp_student(5, 0, true, 6, BINARY);
	c.reset(0.0);
	c.seed_att_estimator(q0);
	let mut mahony = crate::estimator::MahonyFilter::new(0.001, 2.0, 0.1);
	mahony.reset(Some(q0));
	let mut pred = crate::att_bias_pred::AttBiasPredictor::new(2.0, 0.1, 0.001);
	let mut moved = false;
	for t in 0..800
	{
		let (g, a) = sim.read_imu();
		let [x, y] = sim.position_xy_rs();
		let v = sim.velocity_xy_rs();
		c.set_horizontal_obs(-x, -y, v[0], v[1]);
		let pwm = c.step(g, a, [0.0; 3]);
		let want = pred.step(v, mahony.update(g, a));
		assert_eq!(c.att_bias_last(), want, "step {t}");
		assert_eq!(&c.last_feature_vector_ref()[c.num_features() - 2..], &want[..], "step {t}: not the layout tail");
		moved |= want[0].abs() > 1e-4;
		sim.step([pwm[0], pwm[1], pwm[2], pwm[3]]);
	}
	assert!(moved, "the fixture never accelerated");
}

/// CPU scorer == Metal score kernel with the block on (short dense horizon, the
/// abi32 discipline), and the block MOVES the result (not dead plumbing).
#[cfg(target_os = "macos")]
#[test]
fn abp_score_cpu_matches_gpu()
{
	use crate::abi32_tests::{assert_parity_steps, cfg, differs, tilt_ref};
	let plant = crate::plant_opts::PlantOpts::new(true, 0.78).unwrap();
	let off = assert_parity_steps(&abp_student(19, 0, false, 256, BINARY), &cfg(plant, tilt_ref()), 60, "abp off");
	let on = assert_parity_steps(&abp_student(19, 0, true, 256, BINARY), &cfg(plant, tilt_ref()), 60, "abp on");
	differs(&off, &on, &[0, 1, 13], "obs_att_bias_pred");
	// The GPU must really run the block: its own off/on rows move too.
	if metal::Device::system_default().is_some()
	{
		let s1 = cfg(plant, tilt_ref());
		let g_off = crate::abi32_tests::gpu(&abp_student(19, 0, false, 256, BINARY), &s1, 60).expect("GPU evaluator");
		let g_on = crate::abi32_tests::gpu(&abp_student(19, 0, true, 256, BINARY), &s1, 60).expect("GPU evaluator");
		assert!([0usize, 1, 13].iter().any(|&i| (g_off[i] - g_on[i]).abs() > 1e-9), "GPU kernel ignored the abp block");
	}
	// Long horizon, sparse (mostly hovering) policy.
	assert_parity_steps(&abp_student(23, 0, true, 6, BINARY), &cfg(plant, tilt_ref()), 400, "abp on 400");
}

/// Metal split record/train == the CPU split replays with the block on (real
/// rollouts), and blanking the recorded stream changes the CPU rows (not vacuous).
#[cfg(target_os = "macos")]
#[test]
fn abp_gpu_split_matches_cpu()
{
	let mut cfg = crate::ctrl40_tests::zero_yaw(crate::abi38_tests::l4c_cfg(0.0, 0), 0.0);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	let mut c = abp_student(29, 4, true, 6, QUAD_WEIGHTED);
	let trs = crate::ctrl40_tests::rollouts(&mut c, &cfg, 3);
	assert!(trs.iter().all(|t| t.att_bias.len() == t.steps));
	let Some((rec_mism, recs, cell_mism, cells, sens)) =
		crate::ctrl40_tests::parity(&c, &trs, |o| o.att_bias.iter_mut().for_each(|v| *v = [0.0; 2]))
	else
	{
		assert!(metal::Device::system_default().is_none() || std::env::var("WNN_SKIP_GPU_TESTS").is_ok(), "GPU present but parity skipped");
		return;
	};
	assert!(recs > 0 && cells > 0, "vacuous split parity");
	assert_eq!(rec_mism, 0, "{rec_mism}/{recs} record rows differ (abp)");
	assert_eq!(cell_mism, 0, "{cell_mism}/{cells} trained cells differ (abp)");
	assert!(sens, "blanking the abp stream did not change the CPU rows — the block is not wired");
}
