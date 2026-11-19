//! ABI 60 (Luiz 19/11/2026, --yaw-p-ff [SCALE], docs/ctrl59_yaw_pterm_spec.md): the PROPORTIONAL
//! yaw-rate feed-forward on top of the ABI 58 trim FF,
//!     sp_r = clamp(bank_r + cT·cresid_z + K_P·e_yaw, ±range_r),   K_P = −K_θ·scale,
//! e_yaw = target_yaw − the student's anchored heading (the float behind the yaw_err input).
//! The bank keeps its own value (pwm[3]); bank 3's DAgger label is sp_T − (FF_trim + FF_p).
//!
//! Pinned: K_P on cf21 at n_o = 10 (= −cT/κ, linear in the scale); refusals (no trim FF, no
//! obs_yaw_err, L2y / L2y', twice, bad scale); the FF rides on the SETPOINT only; the recorded
//! label is the off twin's minus the FF of the same tick; the kernels' feature index is the
//! yaw_err input bit for bit; CPU scorer == Metal scorer with the flag ON (also stacked on
//! --delta-max-yaw), incl. steady / hold / mid rows, and the P-term moves both; the trainer's
//! GPU walk == its CPU walk; flag off = the ABI 58 student (yaw_ff_value's trim branch).

use crate::abi38_pins::{cpu_rows, gpu_rows};
use crate::abi38_tests::l4c_cfg;
use crate::controller::{WnnController, YAW_BANK};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt, YAW_ERR};
use crate::yaw_trim_err::YawTrimErr;
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

/// The trim-FF student, with the P-term at `scale` when Some.
fn student(ar: usize, seed: u64, mode: u8, pff: Option<f32>) -> WnnController
{
	let mut c = yaw_student(0, ar, seed, YawOpt::Off, mode);
	c.enable_yaw_trim_ff().unwrap();
	if let Some(s) = pff
	{
		c.enable_yaw_p_ff(s).unwrap();
	}
	c
}

/// yaw_student's cascade + residual contract WITHOUT the obs_yaw_err input (yaw_err_i kept, so
/// the heading is still anchored): the trim FF is legal on it, the P-term is not.
fn student_without_yaw_err() -> WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn) = (8usize, 3usize, 2usize, 10usize);
	let base_nf = 21usize;
	let nf = base_nf + 3;
	let mut rng = rand::rngs::SmallRng::seed_from_u64(60);
	let thresholds: Vec<f32> = (0..base_nf * bpf).map(|_| rng.gen_range(-0.4f32..0.4)).collect();
	let oconn: Vec<i64> = (0..4 * levels * obpn).map(|_| rng.gen_range(0..nf * bpf) as i64).collect();
	let mut c = WnnController::new_core(
		4, levels, bpf, window, 0, 0, obpn, thresholds, Vec::new(), oconn,
		true, 0.1, 0.95, 1.0,
		false, false, true, true, false, false, false, true,
		0.99, 1.0, crate::abi38_pins::DT, false, 10,
		BINARY, Some(crate::cell_mode::DECODE_ANTAGONIST), None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("no-yaw_err student must construct");
	let resid_thr: Vec<f32> = (0..3 * bpf).map(|i| [-0.25f32, 0.0, 0.25][i % 3]).collect();
	c.enable_ctbr_with_obs(crate::ctrl38_tests::obs_spec3(false, false, true), resid_thr).expect("cascade + residual");
	c
}

#[test]
fn ypf_gain_on_cf21_is_minus_k_theta_and_scales()
{
	let spec = crate::ctrl38_tests::obs_spec3(false, false, true);
	let dt = crate::abi38_pins::DT;
	let kp = YawTrimErr::p_gain(&spec, dt, 10, 1.0).unwrap();
	eprintln!("[ypf] K_P(scale 1) = {kp:.5} rad/s per rad");
	assert!(kp > 9.0 && kp < 10.2, "K_P {kp} outside the derived 9.59");
	// K_θ = cT / κ (κ = K_I·L / K_θ, cT = K_I·L) — the SAME row.
	let ct = YawTrimErr::ff_gain(&spec, dt, 10).unwrap();
	let kappa = YawTrimErr::from_spec(&spec, dt, 10).unwrap().kappa();
	assert!((kp - (-(ct / kappa))).abs() < 1e-4 * kp, "K_P {kp} vs -cT/kappa {}", -(ct / kappa));
	for scale in [0.5f32, 0.25, 2.0]
	{
		let k = YawTrimErr::p_gain(&spec, dt, 10, scale).unwrap();
		assert!((k - scale * kp).abs() < 1e-5 * kp, "scale {scale}: K_P {k} vs {}", scale * kp);
	}
	assert_eq!(student(10, 1, BINARY, Some(1.0)).yaw_p_ff_gain(), Some(kp));
	assert_eq!(student(10, 1, BINARY, Some(0.5)).yaw_p_ff_gain(), Some(YawTrimErr::p_gain(&spec, dt, 10, 0.5).unwrap()));
	assert!(student(10, 1, BINARY, None).yaw_p_ff_gain().is_none(), "default off");
}

#[test]
fn ypf_refusals()
{
	let mut no_ff = yaw_student(0, 10, 1, YawOpt::Off, BINARY);
	assert!(no_ff.enable_yaw_p_ff(1.0).is_err(), "needs the trim FF first");
	let mut no_yaw_err = student_without_yaw_err();
	no_yaw_err.enable_yaw_trim_ff().expect("the trim FF is legal without obs_yaw_err");
	assert!(no_yaw_err.enable_yaw_p_ff(1.0).is_err(), "needs obs_yaw_err");
	assert!(no_yaw_err.yaw_err_feature_index().is_none());
	let mut motor = crate::abi32_tests::controller(1, false);
	assert!(motor.enable_yaw_p_ff(1.0).is_err(), "needs the cascade (and the trim FF)");
	let mut c = student(10, 1, BINARY, Some(1.0));
	assert!(c.enable_yaw_p_ff(1.0).is_err(), "enabled twice");
	assert!(c.enable_yaw_trim_err().is_err() && c.enable_yaw_trim_err_p().is_err(), "P-term then L2y / L2y'");
	for opt in [YawOpt::L2y, YawOpt::L2yP]
	{
		let mut l2 = yaw_student(0, 10, 1, opt, BINARY);
		assert!(l2.enable_yaw_p_ff(1.0).is_err(), "L2y / L2y' then the P-term");
	}
	for bad in [0.0f32, -1.0, 2.0001, f32::NAN, f32::INFINITY]
	{
		let mut c = student(10, 1, BINARY, None);
		assert!(c.enable_yaw_p_ff(bad).is_err(), "scale {bad} must be refused");
		assert!(c.yaw_p_ff_gain().is_none(), "a refused scale must leave the option off");
	}
	assert!(student(10, 1, BINARY, None).enable_yaw_p_ff(2.0).is_ok(), "scale 2 is the inclusive bound");
}

/// The kernels read e_yaw as sensors[ypf_idx]: on the CPU the feature vector's entry at
/// yaw_err_feature_index() IS yaw_err_last, bit for bit, at every step; and it is the yaw_err
/// input of the fixture's layout.
#[test]
fn ypf_feature_index_is_the_yaw_err_input()
{
	let mut c = student(10, 2, BINARY, Some(1.0));
	let idx = c.yaw_err_feature_index().expect("obs_yaw_err is on");
	assert_eq!(idx, YAW_ERR, "the fixture's yaw_err input");
	c.reset(0.03);
	let mut moved = 0usize;
	for t in 0..400
	{
		let s = t as f32;
		let gyro = [0.02 * (0.03 * s).sin(), -0.015 * (0.02 * s).cos(), 0.4 * (0.05 * s).sin() + 0.1];
		c.step(gyro, [0.3 * (0.01 * s).sin(), -0.2, 9.81], [0.0, 0.0, 0.004]);
		let f = c.last_feature_vector_ref()[idx];
		assert_eq!(f.to_bits(), c.yaw_err_last().to_bits(), "t {t}: feature[{idx}] {f} vs e_yaw {}", c.yaw_err_last());
		moved += (f != 0.0) as usize;
	}
	assert!(moved > 300, "vacuous: e_yaw zero on {} of 400 steps", 400 - moved);
}

/// The FF lives on the SETPOINT only: at every decision sp_r = clamp(bank_r + cT·cresid_z_t +
/// K_P·e_yaw_t, ±r), where bank_r (pwm[3], what the delta law's leak reads) never contains
/// either term. yaw_ff_value is the total and splits into the trim twin's value + K_P·e_yaw.
#[test]
fn ypf_rides_on_the_setpoint_only()
{
	let mut base = yaw_student(0, 1, 3, YawOpt::Off, BINARY);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut base, &l4c_cfg(0.0, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
	let mut on = base.clone();
	on.enable_yaw_trim_ff().unwrap();
	on.enable_yaw_p_ff(1.0).unwrap();
	let (ct, kp) = (on.yaw_trim_ff_gain().unwrap(), on.yaw_p_ff_gain().unwrap());
	let r = on.ctbr_spec().unwrap().range[YAW_BANK];
	let (_, _, leak, _) = on.delta_params();
	on.reset(0.01);
	let (mut p_seen, mut clamped, mut bank_prev) = (0.0f32, 0usize, on.bank_value(YAW_BANK));
	for t in 0..300
	{
		let s = t as f32;
		let gyro = [0.02 * (0.03 * s).sin(), -0.015 * (0.02 * s).cos(), 0.04 * (0.05 * s).sin() + 0.01];
		on.step(gyro, [0.3 * (0.01 * s).sin(), -0.2, 9.81], [0.0, 0.0, 0.004]);
		let sn = on.ctbr_held_setpoint().unwrap();
		let resid = on.cascade_obs_ref().and_then(|o| o.residual()).unwrap();
		let trim = ct * resid.of(&on.cascade_obs_ref().unwrap().last())[2];
		let ff = on.yaw_ff_value();
		assert_eq!(ff.to_bits(), (trim + kp * on.yaw_err_last()).to_bits(), "t {t}: the total FF is trim + K_P*e_yaw");
		let want = (on.bank_value(YAW_BANK) + ff).clamp(-r, r);
		assert_eq!((sn[YAW_BANK] as f32).to_bits(), want.to_bits(), "t {t}: sp_r {} vs bank {} + FF {ff}", sn[YAW_BANK], on.bank_value(YAW_BANK));
		// The leak acts on the bank's own value only: the bank moves by at most one delta from
		// leak·(its previous value), whatever the FF is.
		let (dm, bank) = (on.delta_max_bank(YAW_BANK) * r, on.bank_value(YAW_BANK));
		assert!((bank - leak * bank_prev).abs() <= dm * 1.0001 + 1e-6, "t {t}: bank {bank} vs leak*prev {} (delta_max {dm}) — the FF leaked into the bank", leak * bank_prev);
		bank_prev = bank;
		p_seen = p_seen.max((kp * on.yaw_err_last()).abs());
		clamped += (want.abs() == r) as usize;
	}
	assert!(p_seen > 1e-2, "vacuous: the P-term never moved ({p_seen})");
	assert!(clamped < 300, "vacuous: the setpoint sat on the clamp throughout");
}

/// expert_drives: the teacher flies, so the plant, the teacher's labels and the student's
/// streams are identical with and without the P-term; bank 3's recorded label is the no-FF
/// twin's minus (cT·cresid_z + K_P·e_yaw) of the SAME tick, banks 0..2 identical. e_yaw is
/// rebuilt from the recorded stream exactly as compute_features builds it (dead-reckoned
/// heading seeded to the episode's initial yaw).
#[test]
fn ypf_label_is_the_remainder()
{
	use crate::dagger_train::{rollout_and_label_inner, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = l4c_cfg(0.20, 4);
	cfg.steps_per_episode = 400;
	cfg.expert_drives = true;
	let fly = |ff: bool| {
		let af = AirframeRs::from_cfg(&cfg);
		let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
		let mut c = yaw_student(0, 10, 4, YawOpt::Off, BINARY);
		if ff
		{
			c.enable_yaw_trim_ff().unwrap();
			c.enable_yaw_p_ff(1.0).unwrap();
		}
		c.set_observer_motor_lag_core(sim.motor_lag());
		let mut rng = rand::rngs::SmallRng::seed_from_u64(60);
		(rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], None), c)
	};
	let ((off, _), (on, c)) = (fly(false), fly(true));
	assert_eq!(off.steps, on.steps);
	assert_eq!(off.cascade_obs, on.cascade_obs, "the plant / student stream differs (expert_drives broken?)");
	assert_eq!(off.student_pwms.len(), on.student_pwms.len());
	let (ct, kp) = (c.yaw_trim_ff_gain().unwrap(), c.yaw_p_ff_gain().unwrap());
	let resid = c.cascade_obs_ref().and_then(|o| o.residual()).unwrap();
	let mut heading = on.init_yaw;
	let (mut moved, mut p_max) = (0usize, 0.0f32);
	for t in 0..on.steps
	{
		heading += on.gyros[t][2] * crate::abi38_pins::DT;
		let e_yaw = on.targets[t][2] - heading;
		for m in 0..3
		{
			assert_eq!(off.pid_pwms[t][m].to_bits(), on.pid_pwms[t][m].to_bits(), "t {t}: bank {m} label moved");
		}
		let ff = ct * resid.of(&on.cascade_obs[t])[2] + kp * e_yaw;
		assert_eq!(on.pid_pwms[t][YAW_BANK].to_bits(), (off.pid_pwms[t][YAW_BANK] - ff).to_bits(), "t {t}: label {} vs teacher {} - FF {ff}", on.pid_pwms[t][YAW_BANK], off.pid_pwms[t][YAW_BANK]);
		moved += (ff != 0.0) as usize;
		p_max = p_max.max((kp * e_yaw).abs());
	}
	assert!(moved > on.steps / 2, "vacuous: FF zero on {} of {} ticks", on.steps - moved, on.steps);
	assert!(p_max > 1e-3, "vacuous: the P-term never moved the label ({p_max})");
}

/// CPU scorer == Metal scorer with the P-term ON: every row the fitness ranks, plus steady,
/// the hold columns and the mid row; and the P-term moves both (against the trim-FF twin).
#[cfg(target_os = "macos")]
#[test]
fn ypf_score_cpu_matches_gpu()
{
	use crate::dagger_train::dagger_train_inplace_with;
	let (_, dm, _, _) = yaw_student(0, 10, 5, YawOpt::Off, BINARY).delta_params();
	for (ar, lag, dmy, scale) in [(1usize, 0.0f32, None, 1.0f32), (10, 0.20, None, 1.0), (10, 0.20, Some(0.5 * dm), 0.5)]
	{
		let train = |pff: Option<f32>| {
			let mut c = student(ar, 0x6001 + ar as u64, BINARY, pff);
			c.set_delta_max_yaw(dmy).unwrap();
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
			c
		};
		let (on, off) = (train(Some(scale)), train(None));
		let (cpu, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag));
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "ar {ar}: the P-term did not move the CPU result");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_off = gpu_rows(&off, 800, lag).unwrap();
		eprintln!(
			"[ypf] ar={ar} T={lag} dmy={dmy:?} scale={scale}: err cpu {:.6} gpu {:.6} (trim-only {:.6} / {:.6}) steady {:.6}/{:.6} mid {:.6}/{:.6}",
			cpu[1], gpu[1], cpu_off[1], gpu_off[1], cpu[5], gpu[5], cpu[19], gpu[19]
		);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (5, "steady"), (13, "pos"), (14, "alt"), (17, "pos_hold"), (18, "dz_hold"), (19, "mid")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
		assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "ar {ar}: the GPU ignored yaw_p_ff");
	}
}

/// A batch that mixes P-term gains is refused by the Metal scorer's guard (one RolloutParams
/// per launch) — the guard reads yaw_p_ff_gain, so two scales differ.
#[test]
fn ypf_batch_gain_is_what_the_guard_compares()
{
	let (a, b, off) = (student(10, 1, BINARY, Some(1.0)), student(10, 1, BINARY, Some(0.5)), student(10, 1, BINARY, None));
	assert_ne!(a.yaw_p_ff_gain(), b.yaw_p_ff_gain());
	assert_ne!(a.yaw_p_ff_gain(), off.yaw_p_ff_gain());
	assert_eq!(a.yaw_trim_ff_gain(), off.yaw_trim_ff_gain(), "the trim FF guard alone cannot tell them apart");
}

#[cfg(target_os = "macos")]
#[test]
fn ypf_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |pff: bool, gpu: bool| {
			let mut c = yaw_student(4, 1, 0x6041, YawOpt::Off, mode);
			c.enable_yaw_trim_ff().unwrap();
			if pff
			{
				c.enable_yaw_p_ff(1.0).unwrap();
			}
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(true, true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(true, false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with yaw_p_ff");
		if mode == BINARY
		{
			assert_ne!(run(false, false), gpu, "mode {mode}: yaw_p_ff never reached the trainer");
		}
	}
}

/// Flag off = the ABI 58 student: a trim-FF student that never enabled the P-term returns the
/// trim value alone (the unchanged branch), and stepping it is bit-identical whether or not
/// e_yaw is being tracked.
#[test]
fn ypf_off_is_the_trim_ff_student()
{
	let mut base = yaw_student(0, 1, 3, YawOpt::Off, BINARY);
	crate::dagger_train::dagger_train_inplace_with(&mut base, &l4c_cfg(0.0, 4), [0.0; 3], 23, false);
	let mut off = base.clone();
	off.enable_yaw_trim_ff().unwrap();
	let ct = off.yaw_trim_ff_gain().unwrap();
	off.reset(0.01);
	for t in 0..200
	{
		let s = t as f32;
		off.step([0.02 * (0.03 * s).sin(), 0.0, 0.04 * (0.05 * s).sin() + 0.01], [0.1, -0.2, 9.81], [0.0, 0.0, 0.004]);
		let resid = off.cascade_obs_ref().and_then(|o| o.residual()).unwrap();
		let trim = ct * resid.of(&off.cascade_obs_ref().unwrap().last())[2];
		assert_eq!(off.yaw_ff_value().to_bits(), trim.to_bits(), "t {t}: flag off must return the trim FF alone");
	}
	assert!(off.yaw_err_last() != 0.0, "vacuous: e_yaw never moved, so the off branch was not exercised against a live P input");
}
