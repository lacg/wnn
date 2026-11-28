//! CTRL-86 S1/S2 (ABI 65): cf21 FLAG-OFF BYTE-IDENTITY PINS.
//!
//! S1 (asymmetric motor lag, `motor_lag_down_s`) and S2 (armed-idle floor, `pwm_min`)
//! are default-OFF plant fields. With both off — and with S1 spelled out as down == up —
//! the cf21 lineage must not move by one bit. Every digest below was CAPTURED on the
//! ABI-64 build (before any CTRL-86 line was written) by running this very file there.
//! The regime is the flown l2b recipe shrunk to a fixture:
//!
//!   * the whole DAgger trainer on a cascade (output=ctbr) student — mpcof teacher at
//!     the derived hover, L4C weather, lag 0.20, the ABI-32 real-life plant
//!     (specific-force IMU + rotor drag 0.78), tilt reference;
//!   * the per-step feature vector + motor command of a hand-flown episode on the
//!     trainer's own plant;
//!   * the CPU scorer and the Metal scorer on the trained genome (stage 2, real-life
//!     plant);
//!   * the classical rivals PID (firmware cascade) and mpcof on the same episodes;
//!   * the threshold-calibration sampler on the cascade.
//!
//! A digest moving means the cf21 lineage changed. Do not re-pin without a lineage
//! decision from Luiz.

use crate::abi38_pins::{cells_and_stats_digest, ics, stage2, Fnv, ARM, ASYM, DT, G, INERTIA, KD, KT};
use crate::abi38_pins::{PID_ATT, PID_ATT_HZ, PID_HOVER_N, PID_LPF_HZ, PID_OUT_LIMIT_N, PID_RATE};
use crate::abi38_tests::{ctbr_student, l4c_cfg};
use crate::controller::WnnController;
use crate::dagger_train::{dagger_train_inplace_with, AirframeRs, RewardGatedConfigPacked};
use crate::plant_opts::PlantOpts;
use crate::stage1::Stage1Cfg;

pub(crate) const LAG: f32 = 0.20;
pub(crate) const ROTOR_DRAG: f32 = 0.78;

/// How the (unchanged) cf21 actuator is spelled: the ABI-64 field set alone, or S1
/// written out as down == up.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Spelling
{
	Legacy,
	DownEqualsUp,
}

/// The l2b-shaped cf21 trainer config with the real-life plant.
pub(crate) fn l2b_cfg(teacher: u8, spelling: Spelling) -> RewardGatedConfigPacked
{
	let mut c = l4c_cfg(LAG, teacher);
	c.imu_specific_force = true;
	c.rotor_drag = ROTOR_DRAG;
	spell_cfg(&mut c, spelling);
	c
}

fn spell_cfg(c: &mut RewardGatedConfigPacked, spelling: Spelling)
{
	if let Spelling::DownEqualsUp = spelling
	{
		c.motor_lag_down_s = LAG;
	}
}

pub(crate) fn real_life_plant(spelling: Spelling) -> PlantOpts
{
	let p = PlantOpts::new(true, ROTOR_DRAG).expect("plant");
	match spelling
	{
		Spelling::Legacy => p,
		Spelling::DownEqualsUp => p.with_actuator(LAG, LAG, 0.0).expect("down == up"),
	}
}

fn stage2_real_life(spelling: Spelling) -> Stage1Cfg
{
	stage2_with(real_life_plant(spelling))
}

/// The abi38 stage-2 episode set on plant `p`.
pub(crate) fn stage2_with(p: PlantOpts) -> Stage1Cfg
{
	let mut s1 = stage2();
	s1.plant = p;
	s1
}

fn trained(spelling: Spelling) -> (WnnController, u64)
{
	let cfg = l2b_cfg(4, spelling);
	trained_on(&cfg)
}

/// A cascade student trained by the cascade mpcof teacher on `cfg`, + its digest.
pub(crate) fn trained_on(cfg: &RewardGatedConfigPacked) -> (WnnController, u64)
{
	let mut c = ctbr_student(0, 10, 0x8600);
	let st = dagger_train_inplace_with(&mut c, cfg, [0.0; 3], 86, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "ctrl86 pin trainer wrote nothing");
	let d = cells_and_stats_digest(&c, Some(&st));
	(c, d)
}

fn feature_stream_digest(c: &WnnController, spelling: Spelling, steps: usize) -> u64
{
	let af = AirframeRs::from_cfg(&l2b_cfg(4, spelling));
	let mut sim = af.sim();
	let mut c = c.clone();
	match spelling
	{
		Spelling::Legacy => c.set_observer_motor_lag_core(sim.motor_lag()),
		Spelling::DownEqualsUp => c.set_observer_actuator_core(sim.actuator()),
	}
	sim.set_rotor_asym_core(Some(ASYM.to_vec())).expect("asym");
	sim.reset(Some([0.9990482, 0.0436194, 0.0, 0.0]), Some([0.2, -0.1, 0.05]));
	c.reset(0.0);
	let mut h = Fnv::new();
	for _ in 0..steps
	{
		let (gyro, accel) = sim.read_imu();
		let pwm = c.step(gyro, accel, [0.0; 3]);
		h.f64s(&c.last_feature_vector_ref().iter().map(|&v| v as f64).collect::<Vec<_>>());
		h.f64s(&pwm.iter().map(|&v| v as f64).collect::<Vec<_>>());
		sim.step([pwm[0], pwm[1], pwm[2], pwm[3]]);
	}
	h.0
}

pub(crate) fn rows_digest(r: &[f64]) -> u64
{
	let r = &r[..r.len().min(crate::cpu_score::PRE_ABI51_ROW_METRICS)];
	let mut h = Fnv::new();
	h.f64s(r);
	h.0
}

fn cpu_rows(c: &WnnController, spelling: Spelling, steps: usize) -> u64
{
	rows_digest(&cpu_rows_on(c, &stage2_real_life(spelling), steps))
}

/// The CPU scorer's row on the stage-2 episodes `s1` (L4C-like weather, lag LAG).
pub(crate) fn cpu_rows_on(c: &WnnController, s1: &Stage1Cfg, steps: usize) -> [f64; crate::cpu_score::ROW_METRICS]
{
	let (q0, w0) = ics();
	let mut cc = c.clone();
	crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, crate::abi38_pins::EPS, steps, DT, ARM, KT, KD, INERTIA, G, LAG, [0.0; 3],
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777,
		0.0, 0, 0, 0.2, c.levels_per_motor(), 4, None, None, None, 1.0, 0.4, None, Some(s1), None, None,
	)
}

fn gpu_rows(c: &WnnController, spelling: Spelling, steps: usize) -> Option<u64>
{
	gpu_rows_on(c, &stage2_real_life(spelling), steps).map(|r| rows_digest(&r))
}

/// The Metal scorer's row on the same episodes (None without a GPU / WNN_SKIP_GPU_TESTS).
pub(crate) fn gpu_rows_on(c: &WnnController, s1: &Stage1Cfg, steps: usize) -> Option<Vec<f64>>
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok()
	{
		return None;
	}
	let ev = crate::metal_controller::ControllerRolloutEvaluator::new().ok()?;
	let (q0, w0) = ics();
	let d = crate::controller::Disturbance {
		tau_bias: [0.0; 3],
		gust_sigma: 0.0,
		gust_tau_c: 0.1,
		motor_asym: ASYM,
		gyro_sigma: 0.0107,
		gyro_bias_walk: 0.0,
		accel_sigma: 0.0,
		dropout_prob: 0.0,
		dropout_len_steps: 0,
		obs_delay_steps: 0,
		torque_scale_jitter: 0.2,
		seed: 777,
	};
	let mut rows = ev
		.score(&[c], &q0, &w0, crate::abi38_pins::EPS, steps, (DT, ARM, KT, INERTIA, G), KD, LAG, [0.0; 3], Some(d), None, None, None, Some(s1))
		.expect("gpu score");
	Some(rows.swap_remove(0))
}

/// One classical rival on the stage-2 episodes with the real-life plant.
fn rival(teacher: u8, spelling: Spelling) -> u64
{
	let r = rival_on(teacher, &stage2_real_life(spelling));
	rows_digest(&[r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7])
}

/// One classical cascade rival (outer hold 10) on the stage-2 episodes `s1`.
pub(crate) fn rival_on(teacher: u8, s1: &Stage1Cfg) -> (f64, f64, f64, f64, f64, f64, f64, f64)
{
	let (q0, w0) = ics();
	crate::dagger_train::score_classical_baseline(
		teacher, q0, w0, 600, 5.0,
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777, 0.0, 0, 0, 0.2,
		ARM, KT, KD, INERTIA, G, DT,
		PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, PID_ATT_HZ, PID_LPF_HZ,
		true, 2.0, 0.1,
		true, 0.0, s1.init_z.clone(), s1.init_vz.clone(), s1.mass.clone(), s1.collective_frac.clone(),
		2.0, 1.0, 0.25,
		s1.init_x.clone(), s1.init_y.clone(), 1.0, 1.0, 0.5236,
		LAG, true, ROTOR_DRAG, true, 10, 0, None, s1.plant.motor_lag_down_s, s1.plant.pwm_min,
	)
	.map(crate::dagger_train::rival_first8)
	.expect("rival must score")
}

fn calib_digest(spelling: Spelling) -> u64
{
	let mut cfg = l2b_cfg(0, spelling);
	cfg.steps_per_episode = 250;
	let mut c = ctbr_student(0, 10, 0x86CA);
	let s = crate::calib_sampler::sample_calibration_features_rs(&mut c, &cfg, 3, 0x86CB);
	let mut h = Fnv::new();
	for col in &s
	{
		h.f64s(&col.iter().map(|&v| v as f64).collect::<Vec<_>>());
	}
	h.0
}

/// [train, feature stream, cpu rows, gpu rows, rival pid, rival mpcof, calib] for one spelling.
pub(crate) fn digests(spelling: Spelling) -> [u64; 7]
{
	let (c, d_train) = trained(spelling);
	let fs = feature_stream_digest(&c, spelling, 600);
	let cpu = cpu_rows(&c, spelling, 500);
	let gpu = gpu_rows(&c, spelling, 500).unwrap_or(ABI64_CF21[3]);
	[d_train, fs, cpu, gpu, rival(0, spelling), rival(4, spelling), calib_digest(spelling)]
}

/// Captured on the ABI-64 build (see the module note).
const ABI64_CF21: [u64; 7] = [
	0xf1ab_d7ec_1545_872c, 0xe4ef_46d7_9d1b_f6b6, 0xade4_e37f_d045_c6e0, 0x4749_3be3_a07a_4c5d,
	0xdaa0_45a3_4a4a_ab0c, 0x964c_6f34_ad92_948f, 0xa899_da8f_a410_606f,
];

#[test]
fn ctrl86_cf21_flag_off_bit_identical_to_abi64()
{
	let legacy = digests(Spelling::Legacy);
	eprintln!("[ctrl86-pin] legacy {legacy:#018x?}");
	assert_eq!(legacy, ABI64_CF21, "the cf21 lineage moved with the CTRL-86 actuator fields OFF");
}

/// S1 spelled out as down == up (and S2's floor at 0) is the same plant: every digest equal.
#[test]
fn ctrl86_cf21_down_equals_up_bit_identical_to_abi64()
{
	let spelled = digests(Spelling::DownEqualsUp);
	eprintln!("[ctrl86-pin] down==up {spelled:#018x?}");
	assert_eq!(spelled, ABI64_CF21, "down == up moved the cf21 lineage");
}
