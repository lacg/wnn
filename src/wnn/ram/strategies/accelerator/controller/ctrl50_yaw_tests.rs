//! ARM 3 (Luiz 12/11/2026, docs/ctrl_steady_gap_block6.md §3-4): `--yaw-integral-leak`,
//! a separate leak for the YAW integral inputs (yaw_err_i, yaw_i_pa) only. Off = the
//! integral_leak for yaw too — flags-off bit-identity is held by the existing pins and
//! the phased_ga goldens; these pin the on-path: only the two yaw integrals move, the
//! leak is validated, and the Metal score / split-train kernels match the CPU.

use crate::controller::WnnController;
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

/// Feature indices in the yaw student's layout (compute_features order): 9 base,
/// peraxis_p roll/pitch/yaw (9..12), peraxis_i roll_i/pitch_i/yaw_i_pa (12..15),
/// yaw_err (15), yaw_err_i (16), then the 7 translation channels.
const YAW_I_PA: usize = 14;
const YAW_ERR_I: usize = 16;
const BASE_NF: usize = 24;

fn yaw_thresholds(bpf: usize, rng: &mut impl rand::Rng) -> Vec<f32>
{
	(0..BASE_NF * bpf)
		.map(|i| match i / bpf
		{
			9..=11 | 15 => rng.gen_range(-0.3f32..0.3),
			12..=14 | 16 => rng.gen_range(-3.0f32..3.0),
			f if f >= 17 => rng.gen_range(-0.6f32..0.6),
			_ => rng.gen_range(-5.0f32..5.0),
		})
		.collect()
}

/// A translation student with BOTH yaw integrals on (per-axis with yaw, and the yaw
/// anchor), half its output bits on the yaw features so the leak steers the policy.
fn yaw_student(seed: u64, sn: usize, leak: Option<f32>, cells: usize, mode: u8) -> WnnController
{
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn, sbpn) = (4usize, 3usize, 2usize, 8usize, 6usize);
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let thresholds = yaw_thresholds(bpf, &mut rng);
	let num_out = 4 * levels;
	let sconn: Vec<i64> = (0..sn * sbpn).map(|_| rng.gen_range(0..window * BASE_NF * bpf + sn) as i64).collect();
	let yaw_feats = [11usize, YAW_I_PA, 15, YAW_ERR_I];
	let oconn: Vec<i64> = (0..num_out * obpn)
		.map(|k| if k % 2 == 0 { (yaw_feats[(k / 2) % 4] * bpf + rng.gen_range(0..bpf)) as i64 } else { rng.gen_range(0..BASE_NF * bpf + sn) as i64 })
		.collect();
	let decode = if mode == BINARY { Some(crate::cell_mode::DECODE_ANTAGONIST) } else { None };
	let mut c = WnnController::new_core(
		4, levels, bpf, window, sn, if sn > 0 { sbpn } else { 0 }, obpn, thresholds, sconn, oconn,
		true, 0.05, 0.98, 1.0,
		false, false, true, true, true, false, true, true,
		0.99, 1.0, 0.001, false, 1, mode, decode, None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("yaw student must construct");
	assert_eq!(c.num_features(), BASE_NF);
	c.set_yaw_integral_leak(leak).expect("valid leak");
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

/// A deterministic IMU/target stream with a constant yaw-rate bias, so the yaw
/// error grows and the yaw integrals separate under different leaks.
fn feature_trace(c: &mut WnnController, steps: usize) -> Vec<Vec<f32>>
{
	c.reset(0.0);
	(0..steps)
		.map(|t| {
			let s = (t as f32 * 0.01).sin();
			c.compute_features([0.02 * s, -0.01 * s, 0.003], [0.1 * s, 0.05, 9.81], [0.0; 3])
		})
		.collect()
}

#[test]
fn yaw_leak_default_is_integral_leak()
{
	let mut off = yaw_student(1, 0, None, 6, QUAD_WEIGHTED);
	assert_eq!(off.yaw_integral_leak_value(), 0.99);
	let mut same = yaw_student(1, 0, Some(0.99), 6, QUAD_WEIGHTED);
	assert_eq!(feature_trace(&mut off, 500), feature_trace(&mut same, 500), "leak == integral_leak must be bit-identical");
}

#[test]
fn yaw_leak_moves_only_the_yaw_integrals()
{
	let off = feature_trace(&mut yaw_student(2, 0, None, 6, QUAD_WEIGHTED), 800);
	let on = feature_trace(&mut yaw_student(2, 0, Some(0.9995), 6, QUAD_WEIGHTED), 800);
	for (t, (a, b)) in off.iter().zip(&on).enumerate()
	{
		for f in 0..BASE_NF
		{
			if f != YAW_I_PA && f != YAW_ERR_I
			{
				assert_eq!(a[f].to_bits(), b[f].to_bits(), "step {t}: feature {f} moved");
			}
		}
	}
	let last = off.len() - 1;
	for f in [YAW_I_PA, YAW_ERR_I]
	{
		assert!(on[last][f].abs() > 2.0 * off[last][f].abs(), "feature {f}: the 0.9995 integral must outgrow 0.99 ({} vs {})", on[last][f], off[last][f]);
	}
}

#[test]
fn yaw_leak_refusals()
{
	let mut c = yaw_student(3, 0, None, 6, QUAD_WEIGHTED);
	for bad in [0.0f32, -0.5, 1.5, f32::NAN]
	{
		assert!(c.set_yaw_integral_leak(Some(bad)).is_err(), "leak {bad} must be refused");
	}
	assert!(c.set_yaw_integral_leak(Some(1.0)).is_ok(), "1.0 (a pure sum) is allowed");
}

/// CPU scorer == Metal score kernel with the yaw leak on, and the leak MOVES the
/// result on both (the kernel reads the new Params field, not integral_leak).
#[cfg(target_os = "macos")]
#[test]
fn yaw_leak_score_cpu_matches_gpu()
{
	use crate::abi32_tests::{assert_parity_steps, cfg, differs, tilt_ref};
	let plant = crate::plant_opts::PlantOpts::new(true, 0.78).unwrap();
	let off = assert_parity_steps(&yaw_student(19, 0, None, 256, BINARY), &cfg(plant, tilt_ref()), 300, "yaw leak off");
	let on = assert_parity_steps(&yaw_student(19, 0, Some(0.9995), 256, BINARY), &cfg(plant, tilt_ref()), 300, "yaw leak on");
	differs(&off, &on, &[0, 1, 13], "yaw_integral_leak");
	if metal::Device::system_default().is_some()
	{
		let s1 = cfg(plant, tilt_ref());
		let g_off = crate::abi32_tests::gpu(&yaw_student(19, 0, None, 256, BINARY), &s1, 300).expect("GPU evaluator");
		let g_on = crate::abi32_tests::gpu(&yaw_student(19, 0, Some(0.9995), 256, BINARY), &s1, 300).expect("GPU evaluator");
		assert!([0usize, 1, 13].iter().any(|&i| (g_off[i] - g_on[i]).abs() > 1e-9), "GPU kernel ignored the yaw leak");
	}
}

/// Metal split record/train == the CPU split replays with the yaw leak on (real
/// rollouts, sn > 0 so the state layer reads the yaw integrals too).
#[cfg(target_os = "macos")]
#[test]
fn yaw_leak_gpu_split_matches_cpu()
{
	let mut cfg = crate::ctrl40_tests::zero_yaw(crate::abi38_tests::l4c_cfg(0.0, 0), 0.0);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	let mut c = yaw_student(29, 4, Some(0.9995), 6, QUAD_WEIGHTED);
	let trs = crate::ctrl40_tests::rollouts(&mut c, &cfg, 3);
	let Some((rec_mism, recs, cell_mism, cells, _sens)) = crate::ctrl40_tests::parity(&c, &trs, |_o| {})
	else
	{
		assert!(metal::Device::system_default().is_none() || std::env::var("WNN_SKIP_GPU_TESTS").is_ok(), "GPU present but parity skipped");
		return;
	};
	assert!(recs > 0 && cells > 0, "vacuous split parity");
	assert_eq!(rec_mism, 0, "{rec_mism}/{recs} record rows differ (yaw leak)");
	assert_eq!(cell_mism, 0, "{cell_mism}/{cells} trained cells differ (yaw leak)");
}
