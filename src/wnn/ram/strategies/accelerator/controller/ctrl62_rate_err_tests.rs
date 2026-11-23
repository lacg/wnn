//! ABI 62 (CTRL-69): the cascade student's MODEL-FREE rate-loop inputs —
//! a4 the flown rate loop's tracking error, one-pole low-passed (`--obs-rate-err`,
//! rate_err_lpf.rs), and a1 rate-I on an axis subset (`--obs-rate-i-axes`).
//!
//!   * contract: one feature per emitted axis, canonical order rate-I, rate error, d̂,
//!     residual; refusals; provenance unchanged with the options at their defaults;
//!   * live: the feature IS the low-pass of the rate loop's own last-update error
//!     (sp − gyro), checked against a standalone filter on the controller's onboard
//!     state, and the rate-error block needs no observer / plant (model-free);
//!   * train = deploy: the replay re-applies the recorded rate-error stream bit for bit,
//!     and a replay without it is refused;
//!   * teacher-driven rollouts (calibration) read the FLOWN loop (non-degenerate ladder);
//!   * CPU scorer vs Metal FC_RATE_LOOP_OBS kernel (and the kernel really reads it), and
//!     the Metal split record/train vs the CPU split (CTRL-40 discipline).

use crate::abi38_pins::{cpu_rows, gpu_rows, gpu_rows_many, DT};
use crate::abi38_tests::{l4c_cfg, spec};
use crate::controller::{ReplayObs, WnnController};
use crate::ctbr::CtbrSpec;
use crate::ctrl38_tests::{arm_plant, asym_cfg, cf21_sim, obs_spec3, obs_student_spec};
use crate::ctrl40_tests::{check, parity, rollouts, zero_yaw};
use crate::dagger_train::{dagger_train_inplace_with, rollout_and_label_rs, AirframeRs};
use crate::rate_err_lpf::{axes_count, RateErrLpf, AXES_ALL};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const BPF: usize = 3; // obs_student's bits_per_feature
const ROLL_PITCH: u8 = 0b011;
const YAW: u8 = 0b100;

/// A spec with the rate-loop inputs: `ri` = rate-I's axes (None ⇒ off), `re` = the
/// rate error's (None ⇒ off, τ 0.05 s), plus the CTRL-38/48 observer blocks.
fn rl_spec(ri: Option<u8>, re: Option<u8>, dhat: bool, resid: bool) -> CtbrSpec
{
	let mut s = obs_spec3(ri.is_some(), dhat, resid);
	s.obs_rate_i_axes = ri.unwrap_or(AXES_ALL);
	s.obs_rate_err_tau = re.map(|_| 0.05);
	s.obs_rate_err_axes = re.unwrap_or(AXES_ALL);
	s
}

/// Ladder rows for the blocks in canonical order. The rate error is a fraction of the
/// bank range (a few %).
fn rl_thresholds(s: &CtbrSpec, seed: u64) -> Vec<f32>
{
	let mut rng = SmallRng::seed_from_u64(seed ^ 0x62_7E);
	let mut th = Vec::new();
	let mut rows = |n: usize, lo: f32, hi: f32, th: &mut Vec<f32>| th.extend((0..n * BPF).map(|_| rng.gen_range(lo..hi)));
	if s.obs_rate_i
	{
		rows(axes_count(s.obs_rate_i_axes), -0.3, 0.3, &mut th);
	}
	if s.obs_rate_err_tau.is_some()
	{
		rows(axes_count(s.obs_rate_err_axes), -0.03, 0.03, &mut th);
	}
	if s.obs_dhat_plant.is_some()
	{
		rows(3, -12.0, 12.0, &mut th);
	}
	if s.obs_dhat_residual_plant.is_some()
	{
		rows(3, -0.3, 0.3, &mut th);
	}
	th
}

fn rl_student(sn: usize, ar: usize, seed: u64, s: CtbrSpec) -> WnnController
{
	let th = rl_thresholds(&s, seed);
	obs_student_spec(sn, ar, seed, s, th, ram_core::neuron_memory::BINARY)
}

/// The FIRST ARM (Luiz 22/11): rate error on roll/pitch, rate-I on yaw, no observer.
fn first_arm() -> CtbrSpec
{
	rl_spec(Some(YAW), Some(ROLL_PITCH), false, false)
}

/// Hand-flown deploy loop under motor asymmetry; returns per step (rate-I six, rate-err three).
fn fly(c: &mut WnnController, steps: usize) -> Vec<([f32; 6], [f32; 3])>
{
	let mut sim = cf21_sim(0.0);
	arm_plant(&mut sim, [0.0; 3], [1.06, 0.95, 1.03, 0.97]);
	c.set_observer_motor_lag_core(sim.motor_lag());
	sim.reset(Some([1.0, 0.0, 0.0, 0.0]), Some([0.05, -0.04, 0.02]));
	c.reset(0.0);
	let mut rec = Vec::with_capacity(steps);
	for _ in 0..steps
	{
		let (gyro, accel) = sim.read_imu();
		let p = c.step(gyro, accel, [0.0; 3]);
		rec.push((c.cascade_obs_last(), c.cascade_rate_err_last()));
		sim.step([p[0], p[1], p[2], p[3]]);
	}
	rec
}

#[test]
fn rate_loop_contract_counts_order_and_refusals()
{
	let base = rl_student(0, 10, 2, spec()).num_features();
	for (ri, re, add) in [(None, Some(AXES_ALL), 3usize), (None, Some(ROLL_PITCH), 2), (Some(YAW), None, 1), (Some(YAW), Some(ROLL_PITCH), 3), (Some(AXES_ALL), Some(AXES_ALL), 6)]
	{
		let c = rl_student(0, 10, 2, rl_spec(ri, re, false, false));
		assert_eq!(c.num_features(), base + add, "ri={ri:?} re={re:?}");
		assert_eq!(c.ctbr_spec().unwrap().cascade_feature_count(), add);
	}
	// the first arm: [rate_i_z, rerr_x, rerr_y] are the tail; model-free (no observer).
	let mut c = rl_student(0, 1, 3, first_arm());
	assert!(c.cascade_obs_ref().unwrap().observer().is_none(), "the first arm must run no observer");
	let rec = fly(&mut c, 200);
	let (six, re) = rec[rec.len() - 1];
	let f = c.last_feature_vector_ref();
	assert_eq!(&f[f.len() - 3..], &[six[2], re[0], re[1]][..], "first-arm tail order");
	// all four blocks: rate-I(3), rate error(3), d̂(3), residual(3) — the residual stays last.
	let mut c = rl_student(0, 1, 3, rl_spec(Some(AXES_ALL), Some(AXES_ALL), true, true));
	let rec = fly(&mut c, 200);
	let (six, re) = rec[rec.len() - 1];
	let f = c.last_feature_vector_ref().to_vec();
	let n = f.len();
	assert_eq!(&f[n - 12..n - 9], &six[0..3]);
	assert_eq!(&f[n - 9..n - 6], &re[..]);
	assert_eq!(&f[n - 6..n - 3], &six[3..6]);
	assert_eq!(&f[n - 3..], &c.cascade_obs_ref().unwrap().residual().unwrap().of(&six)[..]);
	// refusals
	for (ri_axes, re_axes, tau, ri_on) in [(0u8, AXES_ALL, None, true), (8, AXES_ALL, None, true), (YAW, AXES_ALL, None, false), (AXES_ALL, YAW, None, false), (AXES_ALL, AXES_ALL, Some(0.0), false), (AXES_ALL, AXES_ALL, Some(f64::NAN), false)]
	{
		let mut s = spec();
		s.obs_rate_i = ri_on;
		s.obs_rate_i_axes = ri_axes;
		s.obs_rate_err_axes = re_axes;
		s.obs_rate_err_tau = tau;
		assert!(s.validate().is_err(), "must refuse ri_axes={ri_axes} re_axes={re_axes} tau={tau:?} rate_i={ri_on}");
	}
	assert!(rl_student(0, 1, 4, first_arm()).enable_ctbr_with_obs(first_arm(), vec![0.0; 2]).is_err());
	// provenance: the defaults leave every ABI-61 token unchanged
	assert!(!spec().provenance_line().contains("rate_err"));
	assert!(obs_spec3(true, false, false).provenance_line().ends_with(" obs=rate_i"));
	let p = first_arm().provenance_line();
	assert!(p.contains(" obs=rate_i(axes=z)") && p.contains(" obs=rate_err(tau=0.05)(axes=xy)"), "{p}");
	assert!(crate::dagger_train::validate_train_regime(&rl_student(0, 1, 6, first_arm()), &l4c_cfg(0.2, 4)).is_ok());
}

/// LIVE = the model on the onboard state: every step the feature is exactly a
/// standalone RateErrLpf fed the student's own rate loop's last-update error, and that
/// error is the loop's `held setpoint − gyro` of its last update tick.
#[test]
fn rate_err_live_is_the_lowpassed_flown_error()
{
	let mut c = rl_student(0, 10, 7, rl_spec(None, Some(AXES_ALL), false, false));
	let s = c.ctbr_spec().unwrap();
	let mut model = RateErrLpf::new(0.05, DT, [s.range[1], s.range[2], s.range[3]], AXES_ALL);
	let mut sim = cf21_sim(0.0);
	arm_plant(&mut sim, [0.0; 3], [1.06, 0.95, 1.03, 0.97]);
	sim.reset(Some([1.0, 0.0, 0.0, 0.0]), Some([0.05, -0.04, 0.02]));
	c.reset(0.0);
	let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
	for t in 0..1500
	{
		let (gyro, accel) = sim.read_imu();
		let ct = c.ctbr_ref().unwrap();
		let raw = ct.rate_err_raw();
		let want = model.step(raw);
		let updates = ct.rate_stage_copy().phase() == 0;
		let p = c.step(gyro, accel, [0.0; 3]);
		assert_eq!(c.cascade_rate_err_last(), want, "step {t}");
		let f = c.last_feature_vector_ref();
		assert_eq!(&f[f.len() - 3..], &want[..], "step {t}: not the layout tail");
		if updates
		{
			// this tick ran the rate loop on the held setpoint and this gyro
			let ct = c.ctbr_ref().unwrap();
			let h = ct.held();
			for a in 0..3
			{
				assert_eq!(ct.rate_axis_state(a).err, h[1 + a] - gyro[a] as f64, "step {t} axis {a}: last_err");
			}
		}
		for a in 0..3
		{
			lo[a] = lo[a].min(want[a]);
			hi[a] = hi[a].max(want[a]);
		}
		sim.step([p[0], p[1], p[2], p[3]]);
	}
	eprintln!("[ctrl62] live rate-err range lo {lo:?} hi {hi:?}");
	for a in 0..3
	{
		assert!(hi[a] - lo[a] > 1e-4, "axis {a}: rate-error feature is constant");
	}
}

/// TRAIN = DEPLOY: the replay re-applies the recorded rate-error stream (first arm,
/// a DAgger rollout under asymmetry), bit for bit; a replay without it is refused.
#[test]
fn rate_err_replay_equals_live()
{
	let cfg = asym_cfg([1.06, 0.95, 1.03, 0.97], 0.20);
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut student = rl_student(0, 10, 9, first_arm());
	student.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = SmallRng::seed_from_u64(62);
	let mut short = cfg.clone();
	short.steps_per_episode = 400;
	let tr = rollout_and_label_rs(&mut student, &mut teacher, &mut sim, &short, 0.05, &mut rng, [0.0; 3]);
	let n = tr.steps;
	assert_eq!(tr.rate_err.len(), n, "the rollout must record the block");
	assert!(tr.rate_err.iter().any(|v| v[0] != 0.0) && tr.rate_err.iter().any(|v| v[1] != 0.0), "stream is trivially zero");
	let ro = ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_cascade(&tr.cascade_obs, 0, n)
		.with_rate_err(&tr.rate_err, 0, n);
	let mut rep = rl_student(0, 10, 9, first_arm());
	rep.replay_obs_guard(Some(&ro), n, "test");
	rep.reset(tr.init_yaw);
	for t in 0..n
	{
		rep.apply_replay_obs(Some(&ro), t);
		let f = rep.compute_features(tr.gyros[t], tr.accels[t], tr.targets[t]);
		assert_eq!(rep.cascade_rate_err_last(), tr.rate_err[t], "step {t}: replay != rollout");
		assert_eq!(&f[f.len() - 3..], &[tr.cascade_obs[t][2], tr.rate_err[t][0], tr.rate_err[t][1]][..], "step {t}: replayed tail");
	}
	let mut stale = ro.clone();
	stale.rate_err.clear();
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rl_student(0, 10, 9, first_arm()).replay_obs_guard(Some(&stale), n, "test")));
	assert!(r.is_err(), "replay_obs_guard must refuse a rate-error replay without the stream");
}

/// Calibration flies the TEACHER through the cascade and lends the student the flown
/// loop: the rate-error columns are fitted on a real spread (a student reading its own
/// unflown loop would see its zero setpoint minus the teacher's rates instead).
#[test]
fn rate_err_calibration_samples_the_flown_loop()
{
	let mut cfg = l4c_cfg(0.20, 0);
	cfg.steps_per_episode = 600;
	let mut c = rl_student(0, 10, 10, first_arm());
	let s = crate::calib_sampler::sample_calibration_features_rs(&mut c, &cfg, 3, 0x62);
	let nf = c.num_features();
	for f in nf - 3..nf
	{
		let col = &s[f];
		let (lo, hi) = col.iter().fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
		eprintln!("[ctrl62] calib feature {f}: [{lo:.5}, {hi:.5}] over {}", col.len());
		assert!(hi > lo, "feature {f}: degenerate calibration column");
	}
}

/// CPU scorer vs the Metal FC_RATE_LOOP_OBS kernel on trained students, and the kernel
/// really reads the block (flattening the rate-error cuts moves CPU AND GPU rows).
#[test]
fn rate_loop_cpu_scorer_matches_gpu()
{
	for (name, s) in [("first arm", first_arm()), ("rerr all", rl_spec(None, Some(AXES_ALL), false, false)), ("all blocks", rl_spec(Some(ROLL_PITCH), Some(YAW), true, true))]
	{
		for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
		{
			let mut c = rl_student(0, ar, 0x6211 + ar as u64, s.clone());
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
			let cpu = cpu_rows(&c, 800, lag);
			let Some(gpu) = gpu_rows(&c, 800, lag)
			else
			{
				return;
			};
			eprintln!("[ctrl62] {name} ar={ar} T={lag}: err cpu {:.6} gpu {:.6} | stable {:.3}/{:.3}", cpu[1], gpu[1], cpu[2], gpu[2]);
			for (i, m) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
			{
				let tol = 2e-3 * cpu[i].abs().max(1e-3);
				assert!((gpu[i] - cpu[i]).abs() <= tol, "{name} ar={ar} T={lag} {m}: gpu {} cpu {}", gpu[i], cpu[i]);
			}
		}
	}
	// sensitivity: the same trained cells, rate-error cuts moved out of range
	let mut on = rl_student(0, 1, 0x6299, first_arm());
	let _ = dagger_train_inplace_with(&mut on, &l4c_cfg(0.0, 4), [0.0; 3], 23, false);
	let mut flat = on.clone();
	let th = flat.thresholds_mut_for_test();
	let n = th.len();
	th[n - 2 * BPF..].iter_mut().for_each(|v| *v = 10.0); // the first arm's rerr_x, rerr_y rows
	let (c_on, c_flat) = (cpu_rows(&on, 800, 0.0), cpu_rows(&flat, 800, 0.0));
	crate::abi32_tests::differs(&c_on, &c_flat, &[0, 1, 2, 3], "obs_rate_err (CPU)");
	if let (Some(g_on), Some(g_flat)) = (gpu_rows(&on, 800, 0.0), gpu_rows(&flat, 800, 0.0))
	{
		assert!([0usize, 1, 2, 3].iter().any(|&i| (g_on[i] - g_flat[i]).abs() > 1e-9), "GPU kernel ignored the rate-error block");
	}
}

/// Metal split record/train == the CPU split replays with the rate-loop inputs on
/// (real rollouts), and blanking the recorded rate-error stream changes the CPU rows.
#[test]
fn rate_loop_gpu_split_matches_cpu()
{
	for (name, s) in [("first arm", first_arm()), ("all blocks", rl_spec(Some(ROLL_PITCH), Some(YAW), true, true))]
	{
		let cfg = zero_yaw(l4c_cfg(0.20, 4), 0.20);
		let mut c = rl_student(4, 1, 0x6202, s.clone());
		let trs = rollouts(&mut c, &cfg, 3);
		assert!(trs.iter().all(|t| t.rate_err.len() == t.steps));
		let fresh = rl_student(4, 1, 0x6202, s);
		let blank = |o: &mut ReplayObs| o.rate_err.iter_mut().for_each(|v| *v = [0.0; 3]);
		check(&format!("rate loop {name}"), parity(&fresh, &trs, blank));
	}
}

/// A POPULATION whose members carry different filter STATE (one flown, one fresh) is one
/// lineage: the GPU scorer's uniform contract compares the constants only, and each
/// member's row equals its solo row (the smoke caught a state-sensitive comparison).
#[test]
fn rate_err_population_is_one_lineage_on_gpu()
{
	let mut a = rl_student(0, 10, 0x6231, first_arm());
	let _ = dagger_train_inplace_with(&mut a, &l4c_cfg(0.20, 4), [0.0; 3], 23, false);
	let mut b = a.clone();
	let _ = fly(&mut b, 300); // leaves b's rate-error filter state non-zero
	assert!(b.cascade_rate_err_last() != [0.0; 3]);
	let Some(rows) = gpu_rows_many(&[&a, &b], 400, 0.20)
	else
	{
		return;
	};
	let solo = gpu_rows(&a, 400, 0.20).unwrap();
	for (i, r) in rows.iter().enumerate()
	{
		assert_eq!(r, &solo, "member {i}: the dispatch must reset the filter per episode (row != solo row)");
	}
}
