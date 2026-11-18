//! ABI 58 (Luiz 17/11/2026, --yaw-trim-ff): the yaw-rate feed-forward sp_r = clamp(bank_r +
//! cT·cresid_z, ±range_r), cT = K_I·L from the teacher's own gain row; the bank keeps its own
//! value (pwm[3]) and the DAgger label of bank 3 is recorded as sp_T − FF.
//!
//! Pinned: cT on cf21 at n_o = 10 (−0.39927) and its consistency with κ; refusals (no cascade,
//! no residual, L2y / L2y' are alternatives, both orders); the FF rides on the SETPOINT only —
//! the bank value and banks 0..2 are those of the off twin while sp_r − bank_r = FF; the recorded
//! label is exactly the off twin's minus FF (expert_drives ⇒ identical plant and teacher); CPU
//! scorer == Metal scorer with the FF (also stacked on --delta-max-yaw) and the FF moves both;
//! the trainer's GPU walk == its CPU walk with the FF.

use crate::abi38_pins::{cpu_rows, gpu_rows};
use crate::abi38_tests::l4c_cfg;
use crate::controller::{WnnController, YAW_BANK};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

fn student(ar: usize, seed: u64, mode: u8, ff: bool) -> WnnController
{
	let mut c = yaw_student(0, ar, seed, YawOpt::Off, mode);
	if ff
	{
		c.enable_yaw_trim_ff().unwrap();
	}
	c
}

#[test]
fn yff_gain_on_cf21_and_consistent_with_kappa()
{
	let spec = crate::ctrl38_tests::obs_spec3(false, false, true);
	let ct = crate::yaw_trim_err::YawTrimErr::ff_gain(&spec, crate::abi38_pins::DT, 10).unwrap();
	eprintln!("[yff] cT = {ct:.5} rad/s per unit cresid_z");
	assert!((ct - (-0.39927)).abs() < 2e-4, "cT {ct} vs the spec's -0.39927");
	// κ = K_I·L / K_θ ⇒ cT = κ·K_θ; K_θ from κ's own derivation.
	let kappa = crate::yaw_trim_err::YawTrimErr::from_spec(&spec, crate::abi38_pins::DT, 10).unwrap().kappa();
	let k_theta = ct / kappa;
	assert!(k_theta < -9.0 && k_theta > -10.2, "implied K_theta {k_theta}");
	assert_eq!(student(10, 1, BINARY, true).yaw_trim_ff_gain(), Some(ct));
}

#[test]
fn yff_refusals()
{
	let mut no_resid = crate::ctrl38_tests::obs_student3(0, 10, 1, false, true, false);
	assert!(no_resid.enable_yaw_trim_ff().is_err(), "needs the residual block");
	let mut motor = crate::abi32_tests::controller(1, false);
	assert!(motor.enable_yaw_trim_ff().is_err(), "needs the cascade");
	let mut c = student(10, 1, BINARY, true);
	assert!(c.enable_yaw_trim_ff().is_err(), "enabled twice");
	assert!(c.enable_yaw_trim_err().is_err() && c.enable_yaw_trim_err_p().is_err(), "FF then L2y / L2y'");
	assert!(student(10, 1, BINARY, false).yaw_trim_ff_gain().is_none(), "default off");
	let mut l2y = yaw_student(0, 10, 1, YawOpt::L2y, BINARY);
	assert!(l2y.enable_yaw_trim_ff().is_err(), "L2y then FF");
	let mut l2yp = yaw_student(0, 10, 1, YawOpt::L2yP, BINARY);
	assert!(l2yp.enable_yaw_trim_ff().is_err(), "L2y' then FF");
}

/// The FF lives on the SETPOINT only: at every decision sp_r = clamp(bank_r + cT·cresid_z_t, ±r)
/// where bank_r (pwm[3], what the delta law's leak reads) never contains the FF. Before any FF
/// exists (the first decision) the student is the off twin bit for bit. (The FF does reach the
/// features afterwards, correctly: the student's own inner rate loop integrates sp − gyro, and
/// the cascade block reads that integrator.)
#[test]
fn yff_rides_on_the_setpoint_only()
{
	let mut base = student(1, 3, BINARY, false);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut base, &l4c_cfg(0.0, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
	let (mut off, mut on) = (base.clone(), base.clone());
	on.enable_yaw_trim_ff().unwrap();
	let r = on.ctbr_spec().unwrap().range[YAW_BANK];
	for c in [&mut off, &mut on]
	{
		c.reset(0.01);
	}
	let mut ff_seen = 0.0f32;
	for t in 0..300
	{
		let s = t as f32;
		let gyro = [0.02 * (0.03 * s).sin(), -0.015 * (0.02 * s).cos(), 0.04 * (0.05 * s).sin() + 0.01];
		let accel = [0.3 * (0.01 * s).sin(), -0.2, 9.81];
		off.step(gyro, accel, [0.0, 0.0, 0.004]);
		on.step(gyro, accel, [0.0, 0.0, 0.004]);
		let sn = on.ctbr_held_setpoint().unwrap();
		if t == 0
		{
			let so = off.ctbr_held_setpoint().unwrap();
			for m in 0..4
			{
				assert_eq!(off.bank_value(m).to_bits(), on.bank_value(m).to_bits(), "first decision: bank {m} value moved");
				if m != YAW_BANK
				{
					assert_eq!(so[m].to_bits(), sn[m].to_bits(), "first decision: setpoint {m} moved");
				}
			}
		}
		let ff = on.yaw_ff_value();
		let want = (on.bank_value(YAW_BANK) + ff).clamp(-r, r);
		assert_eq!((sn[YAW_BANK] as f32).to_bits(), want.to_bits(), "t {t}: sp_r {} vs bank {} + FF {ff}", sn[YAW_BANK], on.bank_value(YAW_BANK));
		ff_seen = ff_seen.max(ff.abs());
	}
	assert!(ff_seen > 1e-3, "vacuous: the FF never moved ({ff_seen})");
}

/// expert_drives: the teacher flies, so the plant, the teacher's labels and the student's cascade
/// stream are identical with and without the FF; bank 3's recorded label is the off twin's minus
/// cT·cresid_z at that tick, banks 0..2 identical.
#[test]
fn yff_label_is_the_remainder()
{
	use crate::dagger_train::{rollout_and_label_inner, AirframeRs};
	use rand::SeedableRng;
	let mut cfg = l4c_cfg(0.20, 4);
	cfg.steps_per_episode = 400;
	cfg.expert_drives = true;
	let fly = |ff: bool| {
		let af = AirframeRs::from_cfg(&cfg);
		let (mut teacher, mut sim) = (af.teacher(cfg.teacher), af.sim());
		let mut c = student(10, 4, BINARY, ff);
		c.set_observer_motor_lag_core(sim.motor_lag());
		let mut rng = rand::rngs::SmallRng::seed_from_u64(58);
		(rollout_and_label_inner(&mut c, &mut teacher, &mut sim, &cfg, 0.1, &mut rng, [0.0; 3], None), c)
	};
	let ((off, _), (on, c)) = (fly(false), fly(true));
	assert_eq!(off.steps, on.steps);
	assert_eq!(off.cascade_obs, on.cascade_obs, "the plant / student stream differs (expert_drives broken?)");
	let ct = c.yaw_trim_ff_gain().unwrap();
	let resid = c.cascade_obs_ref().and_then(|o| o.residual()).unwrap();
	let mut moved = 0usize;
	for t in 0..on.steps
	{
		for m in 0..3
		{
			assert_eq!(off.pid_pwms[t][m].to_bits(), on.pid_pwms[t][m].to_bits(), "t {t}: bank {m} label moved");
		}
		let ff = ct * resid.of(&on.cascade_obs[t])[2];
		assert_eq!(on.pid_pwms[t][YAW_BANK].to_bits(), (off.pid_pwms[t][YAW_BANK] - ff).to_bits(), "t {t}: label {} vs teacher {} - FF {ff}", on.pid_pwms[t][YAW_BANK], off.pid_pwms[t][YAW_BANK]);
		moved += (ff != 0.0) as usize;
	}
	assert!(moved > on.steps / 2, "vacuous: FF zero on {} of {} ticks", on.steps - moved, on.steps);
}

#[cfg(target_os = "macos")]
#[test]
fn yff_score_cpu_matches_gpu()
{
	use crate::dagger_train::dagger_train_inplace_with;
	let (_, dm, _, _) = yaw_student(0, 10, 5, YawOpt::Off, BINARY).delta_params();
	for (ar, lag, dmy) in [(1usize, 0.0f32, None), (10, 0.20, None), (10, 0.20, Some(0.5 * dm))]
	{
		let train = |ff: bool| {
			let mut c = student(ar, 0x5801 + ar as u64, BINARY, ff);
			c.set_delta_max_yaw(dmy).unwrap();
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
			c
		};
		let (on, off) = (train(true), train(false));
		let (cpu, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag));
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "ar {ar}: the FF did not move the CPU result");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_off = gpu_rows(&off, 800, lag).unwrap();
		eprintln!("[yff] ar={ar} T={lag} dmy={dmy:?}: err cpu {:.6} gpu {:.6} (off {:.6} / {:.6})", cpu[1], gpu[1], cpu_off[1], gpu_off[1]);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
		assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "ar {ar}: the GPU ignored yaw_trim_ff");
	}
}

#[cfg(target_os = "macos")]
#[test]
fn yff_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |ff: bool, gpu: bool| {
			let mut c = yaw_student(4, 1, 0x5841, YawOpt::Off, mode);
			if ff
			{
				c.enable_yaw_trim_ff().unwrap();
			}
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(true, true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(true, false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with yaw_trim_ff");
		if mode == BINARY
		{
			assert_ne!(run(false, false), gpu, "mode {mode}: yaw_trim_ff never reached the trainer");
		}
	}
}
