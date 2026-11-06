//! ABI 38 (CTRL-27): the CASCADE controller (output=ctbr) — behaviour, identity and
//! CPU/GPU parity. The motor-level path is pinned separately (abi38_pins).
//!
//!   * contract: decode/encode, refusals, the untrained cascade holds zero rate;
//!   * LABEL-REPLAY IDENTITY (design §2.1): a cascade student whose decode equals the
//!     PID teacher's CTBR label flies bit-identical to the PID teacher alone — on the
//!     controller (CPU) and as the C2 rival (outer hold 1 == the motor-level PID row);
//!     the GPU half lives in metal_controller::tests (abi38_gpu_*);
//!   * the mpcof label is a rate-level decision with the right sign;
//!   * the CPU scorer and the Metal kernel agree on a cascade student (lag 0 / 0.20,
//!     action_repeat 1 / 10, L4C-like weather, stage 2);
//!   * training (sn = 0 direct write, sn = 4 EDRA-BPTT, the state-split trainer)
//!     writes CTBR labels and the GPU walk / GPU split are bit-identical to the CPU.

use crate::abi38_pins::{
	cells_and_stats_digest, cpu_rows, gpu_rows, prod_cfg, prod_student, KT, PID_ATT_HZ, PID_HOVER_N, PID_LPF_HZ,
	PID_OUT_LIMIT_N, PID_RATE,
};
use crate::controller::{AttitudeSim, WnnController};
use crate::ctbr::CtbrSpec;
use crate::dagger_train::{arm_cfg_disturbance, dagger_train_inplace_with, AirframeRs, RewardGatedConfigPacked, TrajectoryRs};
use crate::episode_regime::OuterLoops;

pub(crate) const RANGE: [f32; 4] = [0.25, 4.2, 4.2, 1.6];
pub(crate) const GAMMA: [f32; 4] = [1.0, 2.0, 2.0, 2.0];

pub(crate) fn spec() -> CtbrSpec
{
	CtbrSpec {
		rate_gains: PID_RATE,
		out_limit_n: PID_OUT_LIMIT_N,
		hover_n: PID_HOVER_N,
		k_thrust: KT as f64,
		main_loop_hz: 1000,
		attitude_hz: PID_ATT_HZ,
		lpf_hz: PID_LPF_HZ,
		range: RANGE,
		gamma: GAMMA,
	}
}

/// The cf21 stage-2 trainer config with the REAL L4C weather (training.py preset
/// 'L4C': no torque bias/gust, ±20% per-episode motor asymmetry, gyro σ 0.0107 +
/// bias walk, accel σ 0.126, torque-scale jitter 0.2). abi38_pins::prod_cfg keeps
/// the synthetic-plant torque bias it was pinned with (it is a digest fixture).
pub(crate) fn l4c_cfg(lag: f32, teacher: u8) -> RewardGatedConfigPacked
{
	let mut c = prod_cfg(lag, teacher);
	c.dist_tau_bias = [0.0; 3];
	c.dist_gust_sigma = 0.0;
	c.dist_motor_asym = [1.0; 4];
	c.dist_motor_asym_mag = 0.2;
	c.dist_gyro_sigma = 0.01073277832;
	c.dist_gyro_bias_walk = 3.8785e-05;
	c.dist_accel_sigma = 0.1264912;
	c.dist_torque_scale_jitter = 0.2;
	c
}

/// The production-shaped feature set without obs_pwm, made a cascade.
pub(crate) fn ctbr_student(sn: usize, ar: usize, seed: u64) -> WnnController
{
	let mut c = prod_student(sn, ar, false, seed);
	c.enable_ctbr(spec()).expect("ctbr student");
	c
}

#[test]
fn decode_encode_roundtrip_and_neutral_is_zero()
{
	let s = spec();
	for n in [0.5f32, 0.75]
	{
		for m in 0..4
		{
			assert_eq!(s.decode(m, n, n), 0.0, "bank {m}: the untrained decode is not zero");
			for &v in &[-0.9f32, -0.3, -0.01, 0.0, 0.02, 0.4, 0.95]
			{
				let sp = v * RANGE[m];
				let back = s.decode(m, s.encode(m, sp, n), n);
				assert!((back - sp).abs() <= 1e-5 * RANGE[m].max(1.0), "bank {m} n {n}: {sp} -> {back}");
			}
			// out-of-range labels clip at ±range
			assert_eq!(s.decode(m, s.encode(m, 10.0 * RANGE[m], n), n), RANGE[m]);
		}
	}
	// γ = 2 on the rate banks: the finest step at 64 levels is (1/32)²·4.2 (design §1.3)
	let finest = s.decode(1, 0.5 + 0.5 / 32.0, 0.5);
	assert!((finest - 4.2 / 1024.0).abs() < 1e-6, "finest rate step {finest}");
}

#[test]
fn refusals()
{
	let mut c = prod_student(0, 1, true, 1);
	assert!(c.enable_ctbr(spec()).unwrap_err().contains("obs_pwm"));
	let mut bad = spec();
	bad.range[2] = 0.0;
	assert!(prod_student(0, 1, false, 1).enable_ctbr(bad).is_err());
	let mut bad = spec();
	bad.attitude_hz = 0.0;
	assert!(prod_student(0, 1, false, 1).enable_ctbr(bad).is_err());
	let mut c = ctbr_student(0, 1, 1);
	assert!(c.set_delta_label_scale_core(2.0).is_err(), "label scale under CTBR");
	assert_eq!(c.output_mode(), "ctbr");
	assert_eq!(prod_student(0, 1, true, 1).output_mode(), "pwm");
	// The trainer refuses a teacher with no rate-level decision, and an inner loop
	// that is not the config's firmware cascade.
	let mut cfg = prod_cfg(0.0, 1); // lqr
	assert!(crate::dagger_train::validate_train_regime(&c, &cfg).unwrap_err().contains("teacher id 1"));
	cfg.teacher = 4;
	assert!(crate::dagger_train::validate_train_regime(&c, &cfg).is_ok());
	cfg.af_pid_rate[0] *= 1.01;
	assert!(crate::dagger_train::validate_train_regime(&c, &cfg).unwrap_err().contains("inner rate loop"));
	let _ = &mut c;
}

/// One closed-loop run: the PID teacher flying its own motor command vs a cascade
/// student whose every decision is the teacher twin's CTBR label. Bit-identical
/// motor commands every tick, hence bit-identical trajectories.
fn identity_run(lag: f32, translation: bool) -> usize
{
	let mut cfg = l4c_cfg(lag, 0);
	if !translation
	{
		cfg.translation = false;
		cfg.xy_offset = 0.0;
		cfg.att_err_tilt_ref = false;
		cfg.af_mass = 0.0;
	}
	let af = AirframeRs::from_cfg(&cfg);
	let loops = OuterLoops::for_cfg(&cfg);
	let mut checked = 0usize;
	for ep in 0..3u64
	{
		let mut t_motor = af.teacher(0);
		let mut t_label = af.teacher(0);
		let mut s1 = af.sim();
		let mut s2 = af.sim();
		let q0 = [0.995f32, 0.06 - 0.03 * ep as f32, -0.07, 0.02];
		for s in [&mut s1, &mut s2]
		{
			s.reset(Some(q0), Some([0.3, -0.2, 0.1]));
			if translation
			{
				s.set_translation_core(cfg.af_mass).unwrap();
				s.set_vertical_state(0.15, 0.0);
				s.set_horizontal_state(0.3, -0.25, 0.0, 0.0);
			}
			arm_cfg_disturbance(s, &cfg, 1000 + ep);
		}
		let mut c = ctbr_student(0, 1, 7);
		c.reset(0.0);
		for t in 0..1500
		{
			if s1.is_unstable()
			{
				break;
			}
			let (g1, _a1) = s1.read_imu();
			let (g2, a2) = s2.read_imu();
			let cmd = loops.expert(&mut t_motor, &s1, s1.quaternion(), g1, [0.0; 3], cfg.target_altitude);
			let label = loops.expert_ctbr(&mut t_label, &s2, s2.quaternion(), g2, [0.0; 3], cfg.target_altitude, 1);
			c.ctbr_inject_setpoint(label);
			let pwm = c.step(g2, a2, [0.0; 3]);
			let want = cmd.map(|v| v as f32);
			for m in 0..4
			{
				assert_eq!(pwm[m].to_bits(), want[m].to_bits(), "lag {lag} tr {translation} ep {ep} t {t} m {m}: {} vs {}", pwm[m], want[m]);
			}
			s1.step(want);
			s2.step([pwm[0], pwm[1], pwm[2], pwm[3]]);
			checked += 1;
		}
		assert_eq!(s1.quaternion().map(f32::to_bits), s2.quaternion().map(f32::to_bits));
	}
	checked
}

#[test]
fn label_replay_identity_cpu()
{
	for lag in [0.0f32, 0.20]
	{
		for tr in [false, true]
		{
			let n = identity_run(lag, tr);
			eprintln!("[abi38] identity lag {lag} translation {tr}: {n} ticks bit-identical");
			assert!(n > 3000, "vacuous identity run ({n} ticks)");
		}
	}
}

/// The C2 rival at outer hold 1 IS the firmware cascade: its row equals the
/// motor-level PID rival's bit for bit (stage 2, estimator, L4C-like, lag 0 / 0.20).
#[test]
fn c2_rival_hold1_equals_the_pid_row()
{
	for lag in [0.0f32, 0.20]
	{
		let pid = crate::abi38_pins::rival(0, lag);
		let c2 = rival_ctbr(0, lag, 1);
		assert_eq!(format!("{pid:?}"), format!("{c2:?}"), "lag {lag}");
		let c4 = rival_ctbr(4, lag, 10);
		let mpcof = crate::abi38_pins::rival(4, lag);
		eprintln!("[abi38] rivals (pin fixture: synthetic torque bias) lag {lag}: PID {:.3}/{:.2}° | C2(h1) {:.3}/{:.2}° | MPCOF {:.3}/{:.2}° | C4 mpcof>rate(h10) {:.3}/{:.2}°", pid.0, pid.1, c2.0, c2.1, mpcof.0, mpcof.1, c4.0, c4.1);
		assert!(c4.1.is_finite(), "C4 rival row not finite");
	}
	assert!(rival_ctbr_res(1, 0.0, 1).is_err(), "lqr has no CTBR decision");
}

pub(crate) fn rival_ctbr(teacher: u8, lag: f32, hold: usize) -> (f64, f64, f64, f64, f64, f64, f64, f64)
{
	rival_ctbr_res(teacher, lag, hold).unwrap_or_else(|_| panic!("ctbr rival {teacher} refused"))
}

fn rival_ctbr_res(teacher: u8, lag: f32, hold: usize) -> pyo3::PyResult<(f64, f64, f64, f64, f64, f64, f64, f64)>
{
	use crate::abi38_pins::*;
	let (q0, w0) = ics();
	let s1 = stage2();
	crate::dagger_train::score_classical_baseline(
		teacher, q0, w0, 600, 5.0,
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777, 0.0, 0, 0, 0.2,
		ARM, KT, KD, INERTIA, G, DT,
		PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, PID_ATT_HZ, PID_LPF_HZ,
		true, 2.0, 0.1,
		true, 0.0, s1.init_z.clone(), s1.init_vz.clone(), s1.mass.clone(), s1.collective_frac.clone(),
		2.0, 1.0, 0.25,
		s1.init_x.clone(), s1.init_y.clone(), 1.0, 1.0, 0.5236,
		lag, false, 0.0, true, hold,
	)
}

/// mpcof's CTBR label is a body rate that opposes the attitude error, and zero at
/// level and still; lag-aware at T > 0 (the plan reaches a different rate).
#[test]
fn mpcof_label_is_a_rate_decision()
{
	for lag in [0.0f32, 0.20]
	{
		let cfg = l4c_cfg(lag, 4);
		let af = AirframeRs::from_cfg(&cfg);
		let mut t = af.teacher(4);
		let level = t.ctbr_label([1.0, 0.0, 0.0, 0.0], [0.0; 3], [0.0; 3], 10, 0.0).unwrap();
		assert!(level.iter().all(|v| v.abs() < 1e-9), "lag {lag}: level & still ⇒ zero rate, got {level:?}");
		t.reset();
		let (s, c) = (0.05f32).sin_cos(); // roll +0.1 rad
		let l = t.ctbr_label([c, s, 0.0, 0.0], [0.0; 3], [0.0; 3], 10, 0.07).unwrap();
		assert!(l[1] < -0.05, "lag {lag}: positive roll error must command negative p, got {l:?}");
		assert!(l[2].abs() < 0.2 * l[1].abs() && l[0] == 0.07, "lag {lag}: {l:?}");
	}
}

/// DAgger on a cascade (06/11/2026 fix): when the STUDENT flies, the MPCOF teacher
/// must plan from the student's REAL inner rate loop, not from its shadow loop driven
/// by the teacher's own (unflown) setpoints. An untrained student flies zero rate, so
/// before the fix the shadow integrators wandered off the real ones by ~1-2× their
/// size (measured: |I_sh − I| 0.16-0.40 vs |I| 0.13-0.19, imitation_probe_ctbr);
/// with the sync the only residual is the one rate update the shadow takes on the
/// teacher's held setpoint after the decision (≈ dt_rate·|Δsp|).
#[test]
fn dagger_teacher_plans_from_the_flown_inner_loop()
{
	use rand::SeedableRng;
	for (lag, ar) in [(0.20f32, 10usize), (0.0, 1)]
	{
		let mut cfg = l4c_cfg(lag, 4);
		cfg.expert_drives = false;
		let mut c = ctbr_student(0, ar, 7);
		let mut teachers = crate::dagger_train::TeacherBank::new(AirframeRs::from_cfg(&cfg));
		let mut sim = crate::dagger_train::sim_default(&cfg);
		let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
		let mut cap = crate::imitation_probe::ImitCapture::new(false, false);
		for _ in 0..3
		{
			crate::dagger_train::rollout_and_label_inner(
				&mut c, teachers.get_mut(4), &mut sim, &cfg, 15f64.to_radians(), &mut rng, [0.0; 3], Some(&mut cap),
			);
		}
		assert!(cap.hidden.len() > 50, "lag {lag}: too few decision records ({})", cap.hidden.len());
		// Both loops start this tick from the SAME (synced) state and take at most one
		// rate update, on the teacher's label vs the untrained student's zero setpoint:
		// |I_sh − I| ≤ dt_rate·|label| exactly (clamping only shrinks it). Without the
		// sync the difference is the whole episode's integral of the label.
		let dt_rate = 1.0 / spec().attitude_hz;
		let (mut size, mut worst) = (0.0f64, 0.0f64);
		for ((h, r), l) in cap.hidden.iter().zip(&cap.inner).zip(&cap.label)
		{
			for a in 0..3
			{
				size += r[a].abs();
				let excess = (h[3 + a] - r[a]).abs() - dt_rate * (l[a + 1] as f64).abs();
				worst = worst.max(excess);
			}
		}
		assert!(size > 1e-3, "lag {lag}: the inner integrators never moved");
		assert!(worst < 1e-9, "lag {lag} ar {ar}: the teacher planned from a loop that is not the flown one (excess {worst:.3e})");
	}
}

/// The untrained cascade holds zero body rate at the anchor's collective over the
/// inner loop (design §2.4): from a tilted, spinning start at T = 0.20 the body rates
/// die out and the attitude stays where it was — no tumble, no leveling.
#[test]
fn untrained_cascade_holds_zero_rate()
{
	for lag in [0.0f32, 0.20]
	{
		let mut cfg = l4c_cfg(lag, 4);
		cfg.translation = false;
		cfg.xy_offset = 0.0;
		cfg.att_err_tilt_ref = false;
		cfg.dist_enabled = false;
		let af = AirframeRs::from_cfg(&cfg);
		let mut s = af.sim();
		s.reset(Some([0.995, 0.07, -0.05, 0.0]), Some([0.3, -0.2, 0.1]));
		let e0 = s.attitude_error(None);
		let mut c = ctbr_student(0, 10, 3);
		c.reset(0.0);
		for _ in 0..1500
		{
			assert!(!s.is_unstable(), "lag {lag}: the untrained cascade tumbled");
			let (g, a) = s.read_imu();
			let p = c.step(g, a, [0.0; 3]);
			s.step([p[0], p[1], p[2], p[3]]);
		}
		let (g, _) = s.read_imu();
		let e = s.attitude_error(None);
		assert!(g.iter().all(|w| w.abs() < 0.01), "lag {lag}: body rates did not die out: {g:?}");
		assert!((e - e0).abs() < 0.02, "lag {lag}: attitude drifted {e0} -> {e}");
	}
}

/// A cascade student with written cells, CPU scorer vs the Metal kernel.
fn trained_ctbr(ar: usize, lag: f32) -> WnnController
{
	let mut c = ctbr_student(0, ar, 0xC7B8 + ar as u64);
	let cfg = l4c_cfg(lag, 4);
	let st = dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "ctbr trainer wrote nothing");
	c
}

#[test]
fn ctbr_scorer_cpu_matches_gpu()
{
	for (ar, lag) in [(1usize, 0.0f32), (1, 0.20), (10, 0.0), (10, 0.20)]
	{
		let c = trained_ctbr(ar, lag);
		let cpu = cpu_rows(&c, 800, lag);
		let Some(gpu) = gpu_rows(&c, 800, lag)
		else
		{
			return;
		};
		eprintln!("[abi38] ctbr ar={ar} T={lag}: err cpu {:.6} gpu {:.6} | stable {:.3}/{:.3} | alt {:.5}/{:.5} | jerk {:.5}/{:.5}", cpu[1], gpu[1], cpu[2], gpu[2], cpu[14], gpu[14], cpu[3], gpu[3]);
		for (i, name) in [(1usize, "err"), (0, "reward"), (2, "stable"), (3, "jerk"), (13, "pos"), (14, "alt")]
		{
			let tol = 2e-3 * cpu[i].abs().max(1e-3);
			assert!((gpu[i] - cpu[i]).abs() <= tol, "ar={ar} T={lag} {name}: gpu {} cpu {}", gpu[i], cpu[i]);
		}
	}
}

/// The label stream the trainer records under CTBR is the bank vector (bounded
/// rates, the PD's collective deviation) and sn = 0 / sn = 4 both train on it;
/// the GPU EDRA-BPTT walk is bit-identical to the CPU walk (with per-episode
/// training asymmetry on — the ABI-38 trainer).
#[test]
fn ctbr_trainers_cpu_matches_gpu_walk()
{
	let cfg = l4c_cfg(0.20, 4);
	// labels
	let mut c = ctbr_student(0, 10, 5);
	let batch = batch_of(&cfg, &mut c);
	let mut rates: Vec<f32> = Vec::new();
	for t in &batch
	{
		for l in &t.pid_pwms
		{
			assert!(l[0].abs() <= 0.25 + 1e-6 && l.iter().all(|v| v.is_finite()), "label {l:?}");
			rates.push(l[1].abs().max(l[2].abs()));
		}
	}
	rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let (p50, p95, mx) = (rates[rates.len() / 2], rates[rates.len() * 95 / 100], rates[rates.len() - 1]);
	eprintln!("[abi38] mpcof CTBR rate labels |p,q|: p50 {p50:.3} p95 {p95:.3} max {mx:.3} rad/s (range ±4.2)");
	assert!(p50 > 1e-3 && p95 < 4.2, "rate labels look wrong: p50 {p50} p95 {p95}");
	for (sn, ar) in [(0usize, 1usize), (0, 10), (4, 1), (4, 10)]
	{
		let run = |gpu: bool| {
			let mut c = ctbr_student(sn, ar, 0xBB + sn as u64);
			let st = dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 29, gpu);
			assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "sn {sn} ar {ar}: nothing written");
			let ex = c.export_cells();
			(cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let cpu = run(false);
		if sn == 0 || std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
		{
			continue;
		}
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "GPU walk never ran");
		assert_eq!(cpu, gpu, "sn {sn} ar {ar}: GPU walk diverged from the CPU walk on CTBR labels");
	}
}

fn batch_of(cfg: &RewardGatedConfigPacked, student: &mut WnnController) -> Vec<TrajectoryRs>
{
	crate::train_asym_tests::asym_batch(cfg, student)
}

/// The state-split trainer on CTBR labels. (a) The conflict scan reads decode-space
/// labels (split_scan_label) on a real CTBR rollout. (b) CPU and GPU plant the same
/// cells on the planting split-parity fixture with the cascade output law (QUAD and
/// BINARY-antagonist) — the real-rollout fixtures find no clean separator, so they
/// cannot exercise the resolve + retrain machinery.
#[test]
fn ctbr_split_trainer_cpu_matches_gpu()
{
	let cfg = l4c_cfg(0.20, 4);
	let mut rec = ctbr_student(4, 1, 0x51);
	let batch = batch_of(&cfg, &mut rec);
	let c = ctbr_student(4, 1, 0x52);
	let l = batch[0].pid_pwms[100];
	let scan = c.split_scan_label(l);
	for m in 0..4
	{
		assert!((0.0..=1.0).contains(&scan[m]), "scan label not in decode space: {scan:?}");
		assert_eq!(scan[m], c.output_decode_target(m, l[m]));
	}
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	use crate::metal_controller::{controller_split_train_loop_parity_variant, FixtureVariant};
	for (mode, decode) in [(2u8, None), (3u8, Some(crate::cell_mode::DECODE_ANTAGONIST))]
	{
		let (s_m, o_m, planted, s_a, o_a) =
			controller_split_train_loop_parity_variant(false, 8, crate::metal_controller::split_parity_salt(8), mode, decode, FixtureVariant::Ctbr).expect("parity run");
		eprintln!("[abi38] ctbr split mode {mode}: planted {planted} state {s_a} addrs / {s_m} mismatches, output {o_a} / {o_m}");
		assert!(planted > 0 && o_a > 0, "mode {mode}: vacuous split parity");
		assert_eq!((s_m, o_m), (0, 0), "mode {mode}: GPU split diverged on CTBR labels");
	}
}

/// Bug fixed alongside (06/11/2026): the GPU split trained the UNSCALED label when
/// --delta-label-scale != 1 (odt_train never scaled; the CPU split does). Motor path,
/// delta student, s = 2, on the planting fixture: CPU and GPU now plant the same cells.
#[test]
fn gpu_split_honours_delta_label_scale()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	use crate::metal_controller::{controller_split_train_loop_parity_variant, FixtureVariant};
	let (s_m, o_m, planted, _s_a, o_a) =
		controller_split_train_loop_parity_variant(false, 8, crate::metal_controller::split_parity_salt(8), 2, None, FixtureVariant::DeltaScale).expect("parity run");
	eprintln!("[abi38] delta-label-scale split: planted {planted}, output {o_a} addrs / {o_m} mismatches, state mismatches {s_m}");
	assert!(planted > 0 && o_a > 0, "vacuous split parity");
	assert_eq!((s_m, o_m), (0, 0), "GPU split ignored delta_label_scale");
}

/// The teacher-driven hosts fly the teacher THROUGH the student's inner stage: the
/// calibration sampler's features and the recorder's universe differ from the
/// motor-level student's on the same config.
#[test]
fn teacher_driven_hosts_fly_the_cascade()
{
	let cfg = l4c_cfg(0.20, 4);
	let mut a = ctbr_student(0, 10, 9);
	let ca = crate::calib_sampler::sample_calibration_features_rs(&mut a, &cfg, 2, 4);
	assert!(ca.iter().all(|col| col.iter().all(|v| v.is_finite())) && !ca[0].is_empty());
	let mut sim: AttitudeSim = AirframeRs::from_cfg(&cfg).sim();
	let _ = &mut sim;
	let u = crate::ctrl17_tests::record_universe_for(&cfg, &ctbr_student(0, 10, 9));
	assert!(!u.is_empty(), "the CTBR recorder visited nothing");
}


/// DIAGNOSTIC: the mpcof CTBR label's planning offset N_o vs the outer hold, on an
/// attitude-only L4C plant, 20 episodes: mean err (deg) and stable fraction.
#[test]
#[ignore]
fn diag_mpcof_rate_label_offset()
{
	for lag in [0.0f32, 0.20]
	{
		let mut cfg = l4c_cfg(lag, 4);
		cfg.translation = false;
		cfg.xy_offset = 0.0;
		cfg.att_err_tilt_ref = false;
		cfg.af_mass = 0.0;
		let af = AirframeRs::from_cfg(&cfg);
		for (tid, n_o, hold) in diag_cases()
		{
			let (mut se, mut st, mut n) = (0.0f64, 0usize, 0usize);
			for ep in 0..20u64
			{
				let mut t = af.teacher(tid);
				let mut o = crate::ctbr::CtbrOverRate::new(af.pid_fw.as_ref().unwrap().rate_stage(), hold);
				let mut s = af.sim();
				let a = 0.04 + 0.004 * ep as f32;
				s.reset(Some([(a).cos(), (a).sin() * 0.6, (a).sin() * 0.8, 0.0]), Some([0.3, -0.2, 0.1]));
				arm_cfg_disturbance(&mut s, &cfg, 50 + ep);
				let mut last = [0.5f64; 4];
				let mut e_sum = 0.0f64;
				let mut k = 0usize;
				for _ in 0..2000
				{
					if s.is_unstable()
					{
						break;
					}
					let (g, _) = s.read_imu();
					t.observe(g, last);
					let l = t.ctbr_label(s.quaternion(), g, [0.0; 3], n_o, 0.0).unwrap();
					let p = o.fly(l, g);
					last = p;
					s.step(p.map(|v| v as f32));
					e_sum += s.attitude_error(None) as f64;
					k += 1;
				}
				let me = e_sum / k.max(1) as f64;
				se += me;
				st += (k == 2000 && me < 5f64.to_radians()) as usize;
				n += 1;
			}
			eprintln!("[c4diag] lag {lag} teacher {tid} N_o {n_o:2} hold {hold:2}: err {:.5}° stable {}/{}", (se / n as f64).to_degrees(), st, n);
		}
	}
}

#[cfg(test)]
fn diag_cases() -> Vec<(u8, usize, usize)>
{
	match std::env::var("WNN_C4DIAG_PID").ok()
	{
		Some(_) => vec![(0, 1, 1), (0, 1, 2), (0, 1, 10), (0, 1, 20)],
		None => vec![(0, 1, 1), (0, 1, 10), (4, 1, 1), (4, 10, 1), (4, 25, 1), (4, 10, 10), (4, 25, 10)],
	}
}

/// DIAGNOSTIC (CTRL-27 brief: "the rate PID's integral should absorb motor
/// asymmetry — measure whether it does"). The C2 composition (firmware attitude
/// stage over the firmware rate PID, outer hold 10 like the student) on the
/// attitude-only cf21 plant with a FIXED per-episode motor asymmetry drawn as the
/// scorer draws it (±20%), no other weather. Steady error (last 20%) with the
/// rate-loop and attitude-loop integrators on/off, vs the symmetric plant.
#[test]
#[ignore]
fn diag_rate_integral_absorbs_asymmetry()
{
	use crate::abi38_pins::{PID_ATT, PID_RATE};
	for lag in [0.0f32, 0.20]
	{
		let mut cfg = l4c_cfg(lag, 0);
		cfg.translation = false;
		cfg.xy_offset = 0.0;
		cfg.att_err_tilt_ref = false;
		cfg.af_mass = 0.0;
		let af = AirframeRs::from_cfg(&cfg);
		for (rate_ki, att_ki) in [(true, true), (false, true), (true, false), (false, false)]
		{
			let mut att = PID_ATT;
			let mut rate = PID_RATE;
			for ax in 0..3
			{
				if !att_ki
				{
					att[ax * 4 + 1] = 0.0;
				}
				if !rate_ki
				{
					rate[ax * 4 + 1] = 0.0;
				}
			}
			for asym_on in [false, true]
			{
				let (mut steady, mut n) = (0.0f64, 0usize);
				let mut worst = 0.0f64;
				for ep in 0..40u64
				{
					let mut p = crate::pid_firmware::AttitudePidFirmwareRs::from_si_arrays(att, rate, PID_OUT_LIMIT_N, PID_HOVER_N, KT as f64, 1000, PID_ATT_HZ, PID_LPF_HZ).unwrap();
					let mut o = crate::ctbr::CtbrOverRate::new(p.rate_stage(), 10);
					let mut s = af.sim();
					s.reset(Some([0.999, 0.03, -0.02, 0.0]), Some([0.0; 3]));
					let asym = if asym_on { crate::controller::train_motor_asym([1.0; 4], 0.2, 7000 + ep) } else { [1.0; 4] };
					s.set_disturbance([0.0; 3], 0.0, 0.1, asym, 0.0, 0.0, 0.0, 1, 0.0, 0, 0, 0.0);
					let steps = 4000usize;
					let mut tail = 0.0f64;
					for t in 0..steps
					{
						let (g, _) = s.read_imu();
						let q = s.quaternion();
						let sp = p.att_stage_step([q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64], [0.0; 3]);
						let pwm = o.fly([0.0, sp[0], sp[1], sp[2]], g);
						s.step(pwm.map(|v| v as f32));
						if t >= steps * 4 / 5
						{
							tail += s.attitude_error(None) as f64;
						}
					}
					let e = (tail / (steps / 5) as f64).to_degrees();
					steady += e;
					worst = worst.max(e);
					n += 1;
				}
				eprintln!(
					"[asymdiag] lag {lag:.2} rate_ki {rate_ki:5} att_ki {att_ki:5} asym {}: steady mean {:.3}° worst {:.3}°",
					if asym_on { "±20%" } else { "none" },
					steady / n as f64,
					worst
				);
			}
		}
	}
}

/// DIAGNOSTIC (design R6): the CTBR label histogram per bank on the stage-2 L4C
/// regime, for both outer teachers — sizes the r_sp range (provisional ±1.6).
#[test]
#[ignore]
fn diag_ctbr_label_histogram()
{
	for (teacher, name) in [(0u8, "pid"), (4u8, "mpcof")]
	{
		for lag in [0.0f32, 0.20]
		{
			let cfg = l4c_cfg(lag, teacher);
			let mut c = ctbr_student(0, 10, 0x6157);
			let batch = batch_of(&cfg, &mut c);
			let mut cols: [Vec<f32>; 4] = Default::default();
			for t in &batch
			{
				for (k, l) in t.pid_pwms.iter().enumerate()
				{
					if k % 10 == 0
					{
						for m in 0..4
						{
							cols[m].push(l[m].abs());
						}
					}
				}
			}
			let q = |v: &mut Vec<f32>, f: f32| {
				v.sort_by(|a, b| a.partial_cmp(b).unwrap());
				v[((v.len() - 1) as f32 * f) as usize]
			};
			let mut line = String::new();
			for (m, bank) in ["T", "p", "q", "r"].iter().enumerate()
			{
				let (p50, p95, p99, mx) = (q(&mut cols[m], 0.5), q(&mut cols[m], 0.95), q(&mut cols[m], 0.99), q(&mut cols[m], 1.0));
				line += &format!(" |{bank}| p50 {p50:.3} p95 {p95:.3} p99 {p99:.3} max {mx:.3};");
			}
			eprintln!("[labelhist] teacher {name} lag {lag:.2}:{line}");
		}
	}
}
