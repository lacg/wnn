//! ABI 63 (Luiz 25/11/2026, Plane CTRL-74, --delta-leak-yaw): the yaw bank's (m = 3) own
//! output-accumulator leak under the delta law (docs/ctrl73_gap_analysis_gt.md lever L1).
//!
//! Pinned: refusals; the override set to the shared delta_leak is the flag-off controller BIT
//! for bit (scorer rows); with a different value ONLY the yaw bank's accumulator leaks
//! differently — banks 0..2 bit-identical, bank 3 = leak_yaw · prev exactly; the DAgger label
//! baseline (label_baselines, what every delta label is measured from) is the very baseline
//! the decoder added its delta to, bank by bank, step by step (a decoder/label leak mismatch
//! is the past bug class this guards); CPU scorer == Metal scorer with the override (CTBR and
//! decouple accumulators) and the override moves both; the trainer's GPU walk == its CPU walk
//! with the override, and it reaches the trainer.

use crate::abi38_pins::{cpu_rows, gpu_rows};
use crate::abi38_tests::l4c_cfg;
use crate::controller::{WnnController, YAW_BANK};
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt};
use ram_core::neuron_memory::{BINARY, QUAD_WEIGHTED};

/// The yaw_student fixture's shared delta_leak is 0.95; 0.7 is the G+T global (the flag's
/// use is the reverse — a yaw leak ABOVE the global — but any value ≠ shared exercises it).
const LEAK_YAW: f32 = 0.7;

fn student(ar: usize, seed: u64, mode: u8, dly: Option<f32>) -> WnnController
{
	let mut c = yaw_student(0, ar, seed, YawOpt::Off, mode);
	c.set_delta_leak_yaw(dly).unwrap();
	c
}

fn bits(r: &[f64]) -> Vec<u64>
{
	r.iter().map(|v| v.to_bits()).collect()
}

fn trained(ar: usize, seed: u64, mode: u8, dly: Option<f32>, lag: f32) -> WnnController
{
	let mut c = student(ar, seed, mode, dly);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
	c
}

#[test]
fn dly_refusals_and_default_off()
{
	let mut c = yaw_student(0, 10, 1, YawOpt::Off, BINARY);
	assert!(c.delta_leak_yaw().is_none(), "default off");
	for bad in [0.0f32, -0.5, 1.0001, f32::NAN, f32::INFINITY]
	{
		assert!(c.set_delta_leak_yaw(Some(bad)).is_err(), "leak {bad} must be refused");
	}
	assert!(c.delta_leak_yaw().is_none());
	c.set_delta_leak_yaw(Some(1.0)).expect("1.0 (pure integrator) is allowed");
	// The raw motor path has no yaw bank (abi32 fixture: delta law, no CTBR / decouple).
	let mut motor = crate::abi32_tests::controller(1, false);
	assert!(motor.set_delta_leak_yaw(Some(0.95)).is_err(), "raw motor path must refuse a yaw-bank leak");
}

#[test]
fn dly_equal_to_shared_is_bit_identical()
{
	let (_, _, leak, _) = yaw_student(0, 10, 2, YawOpt::Off, BINARY).delta_params();
	let off = trained(10, 2, BINARY, None, 0.20);
	let mut same = off.clone();
	same.set_delta_leak_yaw(Some(leak)).unwrap();
	assert_eq!(same.delta_leak_bank(YAW_BANK).to_bits(), leak.to_bits());
	assert_eq!(bits(&cpu_rows(&off, 600, 0.2)), bits(&cpu_rows(&same, 600, 0.2)));
	// And trained under the equal override from scratch: same cells, same rows.
	let same_tr = trained(10, 2, BINARY, Some(leak), 0.20);
	assert_eq!(bits(&cpu_rows(&off, 600, 0.2)), bits(&cpu_rows(&same_tr, 600, 0.2)));
}

/// Drive a trained student step by step (ar = 1: every step decides) and check, at EVERY
/// step and for EVERY bank, that
///   (a) the label baseline recorded for the DAgger delta label (label_baselines) is
///       anchor (0 under CTBR) + leak_m · (the previous step's bank value), with leak_3 the
///       override and leak_0..2 the shared leak, bit for bit;
///   (b) the decoder's output is clamp(that SAME baseline + decode_delta(raw)) bit for bit,
///       so a label measured from label_baselines is measured from the decoder's baseline.
#[test]
fn dly_label_baseline_is_the_decoder_baseline()
{
	let c0 = trained(1, 3, BINARY, Some(LEAK_YAW), 0.0);
	let (_, _, shared, _) = c0.delta_params();
	assert!(shared != LEAK_YAW);
	let mut c = c0.clone();
	c.reset(0.0);
	let spec = c.ctbr_spec().unwrap().clone();
	let n = c.neutral_pub();
	let mut prev = [0.0f32; 4];
	let mut yaw_moved = 0usize;
	for t in 0..300
	{
		let tf = t as f32;
		let gyro = [0.02 * (0.05 * tf).sin(), -0.015 * (0.07 * tf).cos(), 0.03 * (0.031 * tf).sin()];
		c.step(gyro, [0.3, -0.2, 9.7], [0.0, 0.0, 0.1 * (0.013 * tf).sin()]);
		let base = c.label_baselines();
		for m in 0..4
		{
			let leak = if m == YAW_BANK { LEAK_YAW } else { shared };
			assert_eq!(c.delta_leak_bank(m).to_bits(), leak.to_bits());
			let want_base = 0.0 + leak * (prev[m] - 0.0);
			assert_eq!(base[m].to_bits(), want_base.to_bits(), "t {t} bank {m}: label base {} vs leak·prev {}", base[m], want_base);
			let r = spec.range[m];
			let delta = spec.decode_delta(m, c.raw_bank_decode_pub(m), n, c.delta_max_bank(m));
			let want = (base[m] + delta).clamp(-r, r);
			assert_eq!(c.pwm_bank_pub(m).to_bits(), want.to_bits(), "t {t} bank {m}: decoder {} vs base+delta {}", c.pwm_bank_pub(m), want);
		}
		if prev[YAW_BANK] != 0.0
		{
			yaw_moved += 1;
		}
		for (m, p) in prev.iter_mut().enumerate()
		{
			*p = c.pwm_bank_pub(m);
		}
	}
	assert!(yaw_moved > 50, "vacuous: the yaw bank sat at 0 ({yaw_moved} steps moved)");
}

/// The label encoder round-trips a delta measured from the per-bank baseline on the yaw
/// alphabet: a teacher target equal to the student's own output encodes to the student's
/// own raw decode (up to the alphabet), with the override on.
#[test]
fn dly_teacher_equal_to_student_labels_its_own_decode()
{
	let mut c = trained(1, 4, BINARY, Some(LEAK_YAW), 0.0);
	c.reset(0.0);
	let spec = c.ctbr_spec().unwrap().clone();
	let n = c.neutral_pub();
	let mut checked = 0usize;
	for t in 0..200
	{
		let tf = t as f32;
		c.step([0.01 * (0.05 * tf).sin(), 0.0, 0.02 * (0.03 * tf).cos()], [0.0, 0.0, 9.81], [0.0, 0.0, 0.05]);
		let base = c.label_baselines();
		let m = YAW_BANK;
		let own = c.pwm_bank_pub(m);
		let r = spec.range[m];
		if own.abs() >= 0.999 * r
		{
			continue; // clamped: the delta is not recoverable
		}
		let label = c.delta_label_target_pub(m, own - base[m]);
		let back = spec.decode_delta(m, label, n, c.delta_max_bank(m));
		let emitted = spec.decode_delta(m, c.raw_bank_decode_pub(m), n, c.delta_max_bank(m));
		assert!((back - emitted).abs() < 1e-5, "t {t}: label decodes {back}, student emitted {emitted}");
		checked += 1;
	}
	assert!(checked > 100);
}

#[cfg(target_os = "macos")]
#[test]
fn dly_score_cpu_matches_gpu()
{
	for (ar, lag) in [(1usize, 0.0f32), (10, 0.20)]
	{
		let (on, off) = (trained(ar, 0x7401 + ar as u64, BINARY, Some(LEAK_YAW), lag), trained(ar, 0x7401 + ar as u64, BINARY, None, lag));
		let (cpu, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag));
		assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "ar {ar}: the override did not move the CPU result");
		let Some(gpu) = gpu_rows(&on, 800, lag)
		else
		{
			return;
		};
		let gpu_off = gpu_rows(&off, 800, lag).unwrap();
		eprintln!("[dly] ar={ar} T={lag}: err cpu {:.6} gpu {:.6} (off {:.6} / {:.6})", cpu[1], gpu[1], cpu_off[1], gpu_off[1]);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
		assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "ar {ar}: the GPU ignored delta_leak_yaw");
	}
}

/// Stacked on --delta-max-yaw (the G+T recipe flies --delta-max-yaw 0.5): both yaw-bank
/// overrides at once, CPU == Metal.
#[cfg(target_os = "macos")]
#[test]
fn dly_stacked_on_delta_max_yaw_cpu_matches_gpu()
{
	let (_, dm, _, _) = yaw_student(0, 10, 7, YawOpt::Off, BINARY).delta_params();
	let mut c = student(10, 0x7407, BINARY, Some(LEAK_YAW));
	c.set_delta_max_yaw(Some(0.5 * dm)).unwrap();
	let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &l4c_cfg(0.20, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
	let cpu = cpu_rows(&c, 800, 0.20);
	let Some(gpu) = gpu_rows(&c, 800, 0.20)
	else
	{
		return;
	};
	for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (13, "pos")]
	{
		let tol = 2e-3 * cpu[i].abs().max(1e-3);
		assert!((gpu[i] - cpu[i]).abs() <= tol, "{name}: gpu {} cpu {}", gpu[i], cpu[i]);
	}
}

#[cfg(target_os = "macos")]
#[test]
fn dly_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for mode in [QUAD_WEIGHTED, BINARY]
	{
		let cfg = l4c_cfg(0.20, 4);
		let run = |dly: Option<f32>, gpu: bool| {
			let mut c = yaw_student(4, 1, 0x7441, YawOpt::Off, mode);
			c.set_delta_leak_yaw(dly).unwrap();
			let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 51, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous trainer run");
			let ex = c.export_cells();
			(crate::abi38_pins::cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(Some(LEAK_YAW), true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "mode {mode}: GPU walk never ran");
		assert_eq!(run(Some(LEAK_YAW), false), gpu, "mode {mode}: GPU walk diverged from the CPU walk with delta_leak_yaw");
		if mode == BINARY
		{
			assert_ne!(run(None, false), gpu, "mode {mode}: delta_leak_yaw never reached the trainer");
		}
	}
}

/// A decouple_outputs student (banks T, τ_roll, τ_pitch, τ_yaw; no CTBR) on yaw_student's
/// layout minus the residual block — the second Metal accumulator site (the decouple one).
fn decouple_student(ar: usize, seed: u64, mode: u8, dly: Option<f32>) -> WnnController
{
	use crate::ctrl51_yaw_trim_tests::{YAW_ERR, YAW_I};
	use rand::{Rng, SeedableRng};
	let (levels, bpf, window, obpn) = (8usize, 3usize, 2usize, 10usize);
	let nf = 22usize;
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..nf * bpf)
		.map(|i| match (i / bpf, i % bpf)
		{
			(YAW_ERR, k) => [-0.006f32, 0.0, 0.006][k],
			(YAW_I, k) => [-0.3f32, 0.0, 0.3][k],
			_ => rng.gen_range(-0.4f32..0.4),
		})
		.collect();
	let out_in = nf * bpf;
	let oconn: Vec<i64> = (0..4 * levels * obpn)
		.map(|k| if k % 2 == 0 { ((YAW_ERR + (k / 2) % 2) * bpf + rng.gen_range(0..bpf)) as i64 } else { rng.gen_range(0..out_in) as i64 })
		.collect();
	let decode = (mode == BINARY).then_some(crate::cell_mode::DECODE_ANTAGONIST);
	let mut c = WnnController::new_core(
		4, levels, bpf, window, 0, 0, obpn, thresholds, Vec::new(), oconn,
		true, 0.1, 0.95, 1.0,
		false, false, true, true, false, false, true, true,
		0.99, 1.0, crate::abi38_pins::DT, true, ar,
		mode, decode, None, 0.05, false, 0.30,
		true, true, true, true, true, false, 1,
	)
	.expect("decouple student must construct");
	c.set_delta_leak_yaw(dly).unwrap();
	c
}

#[cfg(target_os = "macos")]
#[test]
fn dly_decouple_score_cpu_matches_gpu()
{
	let lag = 0.20f32;
	let train = |dly: Option<f32>| {
		let mut c = decouple_student(10, 0x74d1, BINARY, dly);
		let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &l4c_cfg(lag, 4), [0.0; 3], 23, false);
		assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "vacuous decouple trainer run");
		c
	};
	let (on, off) = (train(Some(LEAK_YAW)), train(None));
	let (cpu, cpu_off) = (cpu_rows(&on, 800, lag), cpu_rows(&off, 800, lag));
	assert!([0usize, 1, 13].iter().any(|&i| (cpu[i] - cpu_off[i]).abs() > 1e-9), "decouple: the override did not move the CPU result");
	let Some(gpu) = gpu_rows(&on, 800, lag)
	else
	{
		return;
	};
	let gpu_off = gpu_rows(&off, 800, lag).unwrap();
	eprintln!("[dly-dec] err cpu {:.6} gpu {:.6} (off {:.6} / {:.6})", cpu[1], gpu[1], cpu_off[1], gpu_off[1]);
	for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
	{
		let tol = 2e-3 * cpu[i].abs().max(1e-3);
		assert!((gpu[i] - cpu[i]).abs() <= tol, "decouple {name}: gpu {} cpu {}", gpu[i], cpu[i]);
	}
	assert!([0usize, 1, 13].iter().any(|&i| (gpu[i] - gpu_off[i]).abs() > 1e-9), "decouple: the GPU ignored delta_leak_yaw");
}
