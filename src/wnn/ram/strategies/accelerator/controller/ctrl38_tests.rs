//! CTRL-38 (ABI 41): the cascade student's rate-I (a1) and d̂ (a2) input blocks —
//! contract, physics, train = deploy, CPU/GPU parity. The both-OFF byte identity is
//! ctrl38_pins (cascade) + abi38_pins (motor path).
//!
//!   * contract: +3 features per block in canonical order (LAST), the threshold-tail
//!     split, the refusals (wrong ladder length, bad plant, plant ≠ training airframe);
//!   * a1 physics: under a constant torque bias the untrained cascade (zero-rate hold)
//!     settles with ki·I = −τ/g, so I/i_limit → −τ/(g·ki·i_limit) per axis, sign
//!     included (g = arm for roll/pitch, 4·k_drag for yaw — the plant's own torques);
//!   * a2 physics: under rotor asymmetry d̂ → the TRUE unmodelled angular acceleration
//!     (Δτ/I from the plant's own asym torque law) in a hand-flown DEPLOY loop AND in
//!     a DAgger training rollout's recorded stream, and the two agree;
//!   * train = deploy: the replay re-applies the recorded stream bit for bit, and the
//!     replay guard refuses a stream-less replay;
//!   * calibration: the sampler's rate-I column reads the FLOWN loop (non-degenerate);
//!   * CPU scorer vs Metal kernel on trained students, each block alone and both.

use crate::abi38_pins::{cpu_rows, gpu_rows, prod_student, ARM, INERTIA, KD};
use crate::abi38_tests::{l4c_cfg, spec};
use crate::controller::{AttitudeSim, ReplayObs, WnnController};
use crate::ctbr::CtbrSpec;
use crate::dagger_train::{dagger_train_inplace_with, rollout_and_label_rs, AirframeRs, RewardGatedConfigPacked};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const BPF: usize = 3; // prod_student's bits_per_feature

fn plant() -> [f64; 5]
{
	[ARM as f64, KD as f64, INERTIA[0] as f64, INERTIA[1] as f64, INERTIA[2] as f64]
}

fn obs_spec(rate_i: bool, dhat: bool) -> CtbrSpec
{
	let mut s = spec();
	s.obs_rate_i = rate_i;
	s.obs_dhat_plant = dhat.then(plant);
	s
}

/// Thermometer rows for the blocks: rate-I lives in [−1, 1], d̂ in rad/s² (~±15).
fn obs_thresholds(rate_i: bool, dhat: bool, seed: u64) -> Vec<f32>
{
	let mut rng = SmallRng::seed_from_u64(seed ^ 0x38_7E);
	let mut th = Vec::new();
	if rate_i
	{
		th.extend((0..3 * BPF).map(|_| rng.gen_range(-0.3f32..0.3)));
	}
	if dhat
	{
		th.extend((0..3 * BPF).map(|_| rng.gen_range(-12.0f32..12.0)));
	}
	th
}

/// The production-shaped cascade student (abi38_pins::prod_student's feature set and
/// widths) with the CTRL-38 blocks switched as asked. Unlike prod_student the
/// connections are drawn over the FULL frame, blocks included — otherwise no neuron
/// would read the new bits and every test below would pass on an inert feature.
pub(crate) fn obs_student(sn: usize, ar: usize, seed: u64, rate_i: bool, dhat: bool) -> WnnController
{
	let (levels, window, obpn, sbpn) = (8usize, 2usize, 10usize, 6usize);
	let base_nf = 9 + 2 + 2 + 2 + 3 + 4;
	let nf = base_nf + 3 * (rate_i as usize) + 3 * (dhat as usize);
	let mut rng = SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..base_nf * BPF).map(|_| rng.gen_range(-0.4f32..0.4)).collect();
	let (state_in, out_in) = (window * nf * BPF + sn, nf * BPF + sn);
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..state_in) as i64).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..out_in) as i64).collect();
	let mut c = WnnController::new_core(
		4, levels, BPF, window, sn, if sn > 0 { sbpn } else { 0 }, obpn, thresholds, sconn, oconn,
		true, 0.1, 0.95, 1.0,
		false, false, true, true, false, false, true, true,
		0.99, 1.0, crate::abi38_pins::DT, false, ar,
		ram_core::neuron_memory::BINARY, Some(crate::cell_mode::DECODE_ANTAGONIST), None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("ctrl38 student must construct");
	c.enable_ctbr_with_obs(obs_spec(rate_i, dhat), obs_thresholds(rate_i, dhat, seed))
		.expect("ctrl38 student");
	assert_eq!(c.num_features(), nf);
	c
}

/// The cf21 attitude-only plant of the stage-2 config (lag `lag`).
fn cf21_sim(lag: f32) -> AttitudeSim
{
	AirframeRs::from_cfg(&l4c_cfg(lag, 4)).sim()
}

/// A plant with only motor asymmetry (or only a torque bias) — no noise.
fn arm_plant(sim: &mut AttitudeSim, tau: [f32; 3], asym: [f32; 4])
{
	sim.set_disturbance(tau, 0.0, 0.1, asym, 0.0, 0.0, 0.0, 1, 0.0, 0, 0, 0.0);
}

/// Hand-flown DEPLOY loop (the scorer's protocol): the student flies its own command.
/// Returns (cascade_obs_last per step, last command).
fn fly(c: &mut WnnController, sim: &mut AttitudeSim, steps: usize) -> (Vec<[f32; 6]>, [f32; 4])
{
	c.set_observer_motor_lag_core(sim.motor_lag());
	sim.reset(Some([1.0, 0.0, 0.0, 0.0]), Some([0.05, -0.04, 0.02]));
	c.reset(0.0);
	let (mut rec, mut last) = (Vec::with_capacity(steps), [0.0f32; 4]);
	for _ in 0..steps
	{
		let (gyro, accel) = sim.read_imu();
		let p = c.step(gyro, accel, [0.0; 3]);
		last = [p[0], p[1], p[2], p[3]];
		rec.push(c.cascade_obs_last());
		sim.step(last);
	}
	(rec, last)
}

fn mean_tail(rec: &[[f32; 6]], k: usize, i: usize) -> f64
{
	let n = rec.len().min(k);
	rec[rec.len() - n..].iter().map(|v| v[i] as f64).sum::<f64>() / n as f64
}

/// The plant's own unmodelled angular acceleration at a steady command: Δτ/I with
/// τ = AttitudeSim::body_torque_asym − body_torque (the same expressions).
fn true_dhat(pwm: [f32; 4], asym: [f32; 4]) -> [f64; 3]
{
	let t: Vec<f64> = (0..4).map(|m| 0.2 * (pwm[m] as f64).powi(2)).collect();
	let d: Vec<f64> = (0..4).map(|m| (asym[m] as f64 - 1.0) * t[m]).collect();
	let (l, k) = (ARM as f64, KD as f64);
	let tau = [l * (-d[1] + d[3]), l * (-d[0] + d[2]), k * (d[0] - d[1] + d[2] - d[3])];
	[tau[0] / INERTIA[0] as f64, tau[1] / INERTIA[1] as f64, tau[2] / INERTIA[2] as f64]
}

#[test]
fn contract_counts_order_and_refusals()
{
	let base = prod_student(0, 10, false, 1).num_features();
	for (ri, dh, add) in [(false, false, 0usize), (true, false, 3), (false, true, 3), (true, true, 6)]
	{
		let c = obs_student(0, 10, 2, ri, dh);
		assert_eq!(c.num_features(), base + add, "rate_i={ri} dhat={dh}");
		assert_eq!(c.cascade_obs_ref().is_some(), ri || dh);
		assert_eq!(c.ctbr_spec().unwrap().cascade_feature_count(), add);
	}
	// both on: the blocks are the LAST six features, rate-I first (the ladder tail).
	let mut c = obs_student(0, 1, 3, true, true);
	let mut sim = cf21_sim(0.0);
	arm_plant(&mut sim, [0.0; 3], [1.06, 0.95, 1.03, 0.97]);
	let _ = fly(&mut c, &mut sim, 40);
	let f = c.last_feature_vector_ref();
	let last = c.cascade_obs_last();
	assert_eq!(&f[f.len() - 6..], &last[..], "the blocks must be the last 6 features, rate-I then d̂");
	// refusals: a ladder of the wrong length, a bad plant, a plant that is not the trainer's
	let mut c = prod_student(0, 1, false, 4);
	assert!(c.enable_ctbr_with_obs(obs_spec(true, false), vec![0.0; 2]).unwrap_err().contains("thermometer"));
	let mut bad = obs_spec(false, true);
	bad.obs_dhat_plant = Some([ARM as f64, 0.0, 1e-5, 1e-5, 1e-5]);
	assert!(bad.validate().is_err(), "a zero k_drag must be refused");
	let mut wrong = obs_spec(false, true);
	wrong.obs_dhat_plant = Some([ARM as f64 * 1.1, KD as f64, INERTIA[0] as f64, INERTIA[1] as f64, INERTIA[2] as f64]);
	let mut c = prod_student(0, 1, false, 5);
	c.enable_ctbr_with_obs(wrong, obs_thresholds(false, true, 5)).expect("constructs");
	let e = crate::dagger_train::validate_train_regime(&c, &l4c_cfg(0.2, 4)).unwrap_err();
	assert!(e.contains("observer plant"), "{e}");
	assert!(crate::dagger_train::validate_train_regime(&obs_student(0, 1, 6, true, true), &l4c_cfg(0.2, 4)).is_ok());
	// provenance: unchanged with both off, names the blocks when on
	assert!(!spec().provenance_line().contains("obs="));
	assert!(obs_spec(true, true).provenance_line().contains("obs=rate_i obs=dhat("));
}

/// a1: under a constant torque bias the untrained cascade holds zero rate through its
/// firmware rate loop; at rest Fh = ki·I cancels the torque: g·ki·I = −τ (g = arm on
/// roll/pitch, 4·k_drag on yaw), so the feature I/i_limit → −τ/(g·ki·i_limit).
#[test]
fn rate_i_converges_to_trim_under_torque_bias()
{
	let tau = [8.0e-4f32, -6.0e-4, 5.0e-5];
	let mut c = obs_student(0, 10, 7, true, false);
	let mut sim = cf21_sim(0.0);
	arm_plant(&mut sim, tau, [1.0; 4]);
	// 40 s: the yaw integrator's time constant is kp/ki ≈ 7 s (roll/pitch ≈ 0.5 s).
	let (rec, _) = fly(&mut c, &mut sim, 40000);
	let g = [ARM as f64, ARM as f64, 4.0 * KD as f64];
	let rate = crate::abi38_pins::PID_RATE;
	for a in 0..3
	{
		let (ki, lim) = (rate[a * 4 + 1], rate[a * 4 + 3]);
		let want = -(tau[a] as f64) / (g[a] * ki * lim);
		let got = mean_tail(&rec, 500, a);
		eprintln!("[ctrl38] rate-I axis {a}: got {got:.5} want {want:.5}");
		assert!(want.abs() > 0.1 && want.abs() < 0.9, "fixture: axis {a} trim {want} not inside the clamp");
		assert!((got - want).abs() <= 0.03 * want.abs(), "axis {a}: I/i_limit {got} != trim {want}");
	}
}

/// The DAgger rollout config whose plant is the deploy loop's: fixed asymmetry, no
/// noise, the student flies (no expert, no exploration).
fn asym_cfg(asym: [f32; 4], lag: f32) -> RewardGatedConfigPacked
{
	let mut cfg = l4c_cfg(lag, 4);
	cfg.dist_motor_asym = asym;
	cfg.dist_motor_asym_mag = 0.0;
	cfg.dist_torque_scale_jitter = 0.0;
	cfg.dist_gyro_sigma = 0.0;
	cfg.dist_gyro_bias_walk = 0.0;
	cfg.dist_accel_sigma = 0.0;
	cfg.expert_drives = false;
	cfg.steps_per_episode = 3000;
	cfg
}

/// a2: d̂ → the true unmodelled angular acceleration of an asymmetric airframe, in a
/// hand-flown deploy loop AND in a DAgger rollout's recorded stream (lag 0.20, so the
/// student's actuator filter is exercised), and the two agree.
#[test]
fn cascade_dhat_converges_in_deploy_and_dagger_rollout()
{
	let asym = [1.06f32, 0.95, 1.03, 0.97];
	let lag = 0.20;
	let mut c = obs_student(0, 10, 8, false, true);
	let mut sim = cf21_sim(lag);
	arm_plant(&mut sim, [0.0; 3], asym);
	let (rec, pwm) = fly(&mut c, &mut sim, 3000);
	let want_deploy = true_dhat(pwm, asym);
	let cfg = asym_cfg(asym, lag);
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim2) = (af.teacher(cfg.teacher), af.sim());
	let mut student = obs_student(0, 10, 8, false, true);
	student.set_observer_motor_lag_core(sim2.motor_lag());
	let mut rng = SmallRng::seed_from_u64(38);
	let tr = rollout_and_label_rs(&mut student, &mut teacher, &mut sim2, &cfg, 0.02, &mut rng, [0.0; 3]);
	assert_eq!(tr.cascade_obs.len(), tr.steps, "the rollout must record the cascade stream");
	let want_train = true_dhat(tr.student_pwms[tr.steps - 1], asym);
	for a in 0..3
	{
		let (dep, trn) = (mean_tail(&rec, 300, 3 + a), mean_tail(&tr.cascade_obs, 300, 3 + a));
		eprintln!(
			"[ctrl38] d̂ axis {a}: deploy {dep:.4} (true {:.4}) | dagger {trn:.4} (true {:.4})",
			want_deploy[a], want_train[a]
		);
		assert!(want_deploy[a].abs() > 1.0, "fixture: axis {a} asymmetry too small to measure");
		assert!((dep - want_deploy[a]).abs() <= 0.03 * want_deploy[a].abs(), "deploy axis {a}: {dep} vs {}", want_deploy[a]);
		assert!((trn - want_train[a]).abs() <= 0.03 * want_train[a].abs(), "dagger axis {a}: {trn} vs {}", want_train[a]);
		// The two rollouts hover at slightly different thrust (the trainer draws a
		// mass), so the TRUE d̂ differs a few %; what must agree is the estimate as a
		// fraction of the truth.
		let (rd, rt) = (dep / want_deploy[a], trn / want_train[a]);
		assert!((rd - rt).abs() <= 0.01, "axis {a}: deploy d̂/true {rd} vs training-rollout d̂/true {rt}");
	}
}

/// TRAIN = DEPLOY in the replay: the trainer's replay (apply_replay_obs →
/// compute_features) produces EXACTLY the values the rollout recorded, and a replay
/// without the stream is refused (fail-loud guard).
#[test]
fn replay_reapplies_the_recorded_cascade_stream()
{
	let cfg = asym_cfg([1.06, 0.95, 1.03, 0.97], 0.20);
	let af = AirframeRs::from_cfg(&cfg);
	let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
	let mut student = obs_student(0, 10, 9, true, true);
	student.set_observer_motor_lag_core(sim.motor_lag());
	let mut rng = SmallRng::seed_from_u64(39);
	let mut cfg_short = cfg.clone();
	cfg_short.steps_per_episode = 400;
	let tr = rollout_and_label_rs(&mut student, &mut teacher, &mut sim, &cfg_short, 0.05, &mut rng, [0.0; 3]);
	let n = tr.steps;
	let ro = ReplayObs::slice(&tr.vert_obs, &tr.horiz_obs, &tr.pwm_dev, &tr.att_up, 0, n)
		.expect("obs")
		.with_cascade(&tr.cascade_obs, 0, n);
	let mut rep = obs_student(0, 10, 9, true, true);
	rep.replay_obs_guard(Some(&ro), n, "test");
	rep.reset(tr.init_yaw);
	for t in 0..n
	{
		rep.apply_replay_obs(Some(&ro), t);
		let f = rep.compute_features(tr.gyros[t], tr.accels[t], tr.targets[t]);
		assert_eq!(rep.cascade_obs_last(), tr.cascade_obs[t], "step {t}: replay != rollout");
		assert_eq!(&f[f.len() - 6..], &tr.cascade_obs[t][..], "step {t}: replayed frame tail");
	}
	assert!(tr.cascade_obs.iter().any(|v| v[0] != 0.0) && tr.cascade_obs.iter().any(|v| v[3] != 0.0), "stream is trivially zero");
	let mut stale = ro.clone();
	stale.cascade.clear();
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		obs_student(0, 10, 9, true, true).replay_obs_guard(Some(&stale), n, "test")
	}));
	assert!(r.is_err(), "replay_obs_guard must refuse a cascade-block replay without the stream");
}

/// Calibration: the sampler lends the flown (teacher) rate loop, so the rate-I ladder
/// is fitted on a real spread inside [−1, 1], and d̂ on the plant's real input.
#[test]
fn calibration_samples_the_flown_loop()
{
	let mut cfg = l4c_cfg(0.20, 0);
	cfg.steps_per_episode = 600;
	let mut c = obs_student(0, 10, 10, true, true);
	let s = crate::calib_sampler::sample_calibration_features_rs(&mut c, &cfg, 3, 0x38);
	let nf = c.num_features();
	for f in nf - 6..nf
	{
		let col = &s[f];
		let (lo, hi) = col.iter().fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
		eprintln!("[ctrl38] calib feature {f}: [{lo:.4}, {hi:.4}] over {}", col.len());
		assert!(hi > lo, "feature {f}: degenerate calibration column");
		if f < nf - 3
		{
			assert!(lo >= -1.0 && hi <= 1.0, "rate-I outside its clamp");
		}
	}
}

/// CPU scorer vs Metal kernel on trained students (lag 0 / 0.20, action_repeat 1 / 10),
/// each block alone and both together — the cpu_fallback_matches_gpu discipline.
#[test]
fn cascade_blocks_cpu_scorer_matches_gpu()
{
	for (ri, dh) in [(true, false), (false, true), (true, true)]
	{
		for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
		{
			let mut c = obs_student(0, ar, 0x3811 + ar as u64, ri, dh);
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
			let cpu = cpu_rows(&c, 800, lag);
			let Some(gpu) = gpu_rows(&c, 800, lag)
			else
			{
				return;
			};
			eprintln!(
				"[ctrl38] rate_i={ri} dhat={dh} ar={ar} T={lag}: err cpu {:.6} gpu {:.6} | stable {:.3}/{:.3} | alt {:.5}/{:.5}",
				cpu[1], gpu[1], cpu[2], gpu[2], cpu[14], gpu[14]
			);
			for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
			{
				let tol = 2e-3 * cpu[i].abs().max(1e-3);
				assert!((gpu[i] - cpu[i]).abs() <= tol, "rate_i={ri} dhat={dh} ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
			}
		}
	}
}

/// The MEMORY recorder: the training-cascade driver lends the flown rate loop and
/// feeds the observer what flew; the legacy PID driver (no firmware rate loop to
/// lend) refuses a rate-I student loudly instead of recording an unflown integrator.
#[test]
fn recorder_lends_the_flown_loop_and_refuses_legacy_drivers()
{
	let cfg = l4c_cfg(0.20, 4);
	let student = obs_student(0, 10, 13, true, true);
	let u = crate::ctrl17_tests::record_universe_for(&cfg, &student);
	assert!(!u.is_empty(), "the cascade recorder recorded nothing");
	let mut c = student.clone();
	let mut sim = AirframeRs::from_cfg(&cfg).sim();
	let mut pid = crate::controller::AttitudePidRs::new_default();
	let mut d = crate::record_ops::Driver::Pid(&mut pid);
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		crate::record_ops::record_address_universe(&mut c, &mut sim, &mut d, &[[1.0, 0.0, 0.0, 0.0]], &[[0.0; 3]], [0.0; 3], 50, None)
	}));
	assert!(r.is_err(), "the legacy PID recorder must refuse a rate-I student");
}
