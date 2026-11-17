//! ABI 57 (Luiz 17/11/2026, --delta-max-yaw): the yaw bank's (m = 3) own delta_max.
//!
//! Pinned: refusals; the override set to the shared delta_max is the flag-off controller BIT
//! for bit (scorer rows); with a different value ONLY the yaw bank's decode moves, by exactly
//! the gain ratio, and the label encoder round-trips on the yaw alphabet; CPU scorer == Metal
//! scorer with the override (float tolerance, as every cascade row) and the override moves
//! both; the trainer's GPU walk == its CPU walk with the override, and it reaches the trainer.

use crate::abi38_pins::{cpu_rows, gpu_rows};
use crate::abi38_tests::l4c_cfg;
use crate::controller::{WnnController, YAW_BANK};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

fn student(ar: usize, seed: u64, mode: u8, dmy: Option<f32>) -> WnnController
{
	let mut c = yaw_student(0, ar, seed, YawOpt::Off, mode);
	c.set_delta_max_yaw(dmy).unwrap();
	c
}

fn bits(r: &[f64]) -> Vec<u64>
{
	r.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn dmy_refusals()
{
	let mut c = yaw_student(0, 10, 1, YawOpt::Off, BINARY);
	assert!(c.set_delta_max_yaw(Some(0.0)).is_err() && c.set_delta_max_yaw(Some(f32::NAN)).is_err());
	assert!(c.delta_max_yaw().is_none());
	// The raw motor path has no yaw bank (abi32 fixture: delta law, no CTBR / decouple).
	let mut motor = crate::abi32_tests::controller(1, false);
	assert!(motor.set_delta_max_yaw(Some(0.05)).is_err(), "raw motor path must refuse a yaw-bank gain");
}

#[test]
fn dmy_equal_to_shared_is_bit_identical()
{
	let (_, dm, _, _) = yaw_student(0, 10, 2, YawOpt::Off, BINARY).delta_params();
	let mut off = yaw_student(0, 10, 2, YawOpt::Off, BINARY);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut off, &l4c_cfg(0.20, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
	let mut same = off.clone();
	same.set_delta_max_yaw(Some(dm)).unwrap();
	assert_eq!(bits(&cpu_rows(&off, 600, 0.2)), bits(&cpu_rows(&same, 600, 0.2)));
}

/// The decode each bank applies (CtbrSpec::decode_delta at the bank's delta_max_bank): for
/// the same raw decode, banks 0..2 are BIT-identical with and without the override and bank
/// 3 scales by exactly dmy / dm (decoded_to_delta is linear in its delta_max).
#[test]
fn dmy_moves_only_the_yaw_bank()
{
	let (_, dm, _, _) = yaw_student(0, 1, 3, YawOpt::Off, BINARY).delta_params();
	let dmy = 0.5 * dm;
	let (a, b) = (student(1, 3, BINARY, None), student(1, 3, BINARY, Some(dmy)));
	let spec = a.ctbr_spec().unwrap();
	let n = a.neutral_pub();
	for raw in [0.0f32, 0.2, 0.45, 0.55, 0.8, 1.0]
	{
		for m in 0..4
		{
			let (da, db) = (spec.decode_delta(m, raw, n, a.delta_max_bank(m)), spec.decode_delta(m, raw, n, b.delta_max_bank(m)));
			if m == YAW_BANK
			{
				assert!(db == 0.0 && da == 0.0 || ((db / da) - 0.5).abs() < 1e-6, "raw {raw}: yaw {da} -> {db}");
				if raw != n { assert!(da != 0.0, "vacuous: raw {raw} decodes 0"); }
			}
			else
			{
				assert_eq!(da.to_bits(), db.to_bits(), "raw {raw}: bank {m} moved");
			}
		}
	}
	assert_eq!(a.delta_max_bank(YAW_BANK), dm);
	assert_eq!(b.delta_max_bank(YAW_BANK), dmy);
}

#[test]
fn dmy_label_encoder_round_trips_on_the_yaw_alphabet()
{
	let (_, dm, _, _) = yaw_student(0, 10, 4, YawOpt::Off, BINARY).delta_params();
	let c = student(10, 4, BINARY, Some(0.4 * dm));
	let spec = c.ctbr_spec().unwrap();
	let r = spec.range[YAW_BANK];
	for k in -8..=8
	{
		let delta = 0.4 * dm * r * k as f32 / 8.5;
		let dec = c.delta_label_target_pub(YAW_BANK, delta);
		let back = spec.decode_delta(YAW_BANK, dec, c.neutral_pub(), 0.4 * dm);
		assert!((back - delta).abs() < 1e-5, "k {k}: {delta} -> {dec} -> {back}");
	}
}

#[cfg(target_os = "macos")]
#[test]
fn dmy_score_cpu_matches_gpu()
{
	use crate::dagger_train::dagger_train_inplace_with;
	let (_, dm, _, _) = yaw_student(0, 10, 5, YawOpt::Off, BINARY).delta_params();
	for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
	{
		let train = |dmy: Option<f32>| {
			let mut c = student(ar, 0x5701 + ar as u64, BINARY, dmy);
			let st = dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
			c
		};
		let (on, off) = (train(Some(0.5 * dm)), train(None));
		let (cpu, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag));
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "ar {ar}: the override did not move the CPU result");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_off = gpu_rows(&off, 800, lag).unwrap();
		eprintln!("[dmy] ar={ar} T={lag}: err cpu {:.6} gpu {:.6} (off {:.6} / {:.6})", cpu[1], gpu[1], cpu_off[1], gpu_off[1]);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
		assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "ar {ar}: the GPU ignored delta_max_yaw");
	}
}

#[cfg(target_os = "macos")]
#[test]
fn dmy_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	let (_, dm, _, _) = yaw_student(4, 1, 6, YawOpt::Off, BINARY).delta_params();
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |dmy: Option<f32>, gpu: bool| {
			let mut c = yaw_student(4, 1, 0x5741, YawOpt::Off, mode);
			c.set_delta_max_yaw(dmy).unwrap();
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(Some(0.5 * dm), true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(Some(0.5 * dm), false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with delta_max_yaw");
		if mode == BINARY
		{
			assert_ne!(run(None, false), gpu, "mode {mode}: delta_max_yaw never reached the trainer");
		}
	}
}
