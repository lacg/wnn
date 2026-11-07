//! CTRL-38 (ABI 41): CASCADE BYTE-IDENTITY PINS with both new feature blocks OFF.
//!
//! `--obs-rate-i` and `--obs-cascade-dhat` are default-OFF input blocks of the cascade
//! (output=ctbr) student. With both off the cascade lineage must not move by one bit.
//! Every digest below was CAPTURED on the ABI-40 build (before any CTRL-38 line was
//! written) by running this very file there. They cover:
//!
//!   * the whole DAgger trainer on a cascade student (mpcof teacher — whose observer
//!     CTRL-38 factors out into cascade_dhat.rs — L4C weather, lag 0.20, sn = 0 with
//!     action_repeat 10 and sn = 4 with action_repeat 1);
//!   * the per-step feature vector + motor command of a scored episode (the student's
//!     compute_features → inner rate loop → plant, flown by hand);
//!   * the CPU scorer and the Metal scorer on the trained sn = 0 genome;
//!   * the threshold-calibration sampler on the cascade (teacher-driven, ExpertPilot).
//!
//! A digest moving means the cascade lineage changed. Do not re-pin without a lineage
//! decision from Luiz.

use crate::abi38_pins::{cells_and_stats_digest, cpu_rows, gpu_rows, Fnv};
use crate::abi38_tests::{ctbr_student, l4c_cfg};
use crate::controller::WnnController;
use crate::dagger_train::{dagger_train_inplace_with, AirframeRs};

/// A cascade student trained by the cascade mpcof teacher (sn, action_repeat) at lag 0.20.
pub(crate) fn trained_cascade(c: &mut WnnController) -> u64
{
	let cfg = l4c_cfg(0.20, 4);
	let st = dagger_train_inplace_with(c, &cfg, [0.0; 3], 31, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "ctrl38 pin trainer wrote nothing");
	cells_and_stats_digest(c, Some(&st))
}

/// Fly `c` by hand on the training airframe (asymmetric rotors, lag 0.20) and hash
/// every step's feature vector and motor command.
pub(crate) fn feature_stream_digest(c: &WnnController, steps: usize) -> u64
{
	let cfg = l4c_cfg(0.20, 4);
	let af = AirframeRs::from_cfg(&cfg);
	let mut sim = af.sim();
	let mut c = c.clone();
	c.set_observer_motor_lag_core(sim.motor_lag());
	sim.set_rotor_asym_core(Some(vec![1.12, 0.9, 1.05, 0.95])).expect("asym");
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

/// The calibration sampler on a cascade student (CTRL-17 G1 path).
pub(crate) fn calib_digest(c: &WnnController) -> u64
{
	let mut cfg = l4c_cfg(0.20, 0);
	cfg.steps_per_episode = 250;
	let mut c = c.clone();
	let s = crate::calib_sampler::sample_calibration_features_rs(&mut c, &cfg, 3, 0x38CA);
	let mut h = Fnv::new();
	for col in &s
	{
		h.f64s(&col.iter().map(|&v| v as f64).collect::<Vec<_>>());
	}
	h.0
}

fn rows_digest(r: &[f64]) -> u64
{
	let mut h = Fnv::new();
	h.f64s(r);
	h.0
}

/// [train sn0 ar10, train sn4 ar1, feature stream, cpu rows, gpu rows, calib].
const ABI40_CASCADE: [u64; 6] = [
	0xbdf2_ab6c_1cc2_e7ca, 0x505b_6b49_9425_7f16, 0xc562_13c5_2e45_9f07,
	0x58f4_b328_7119_b9fe, 0xd9e7_e9d4_455a_432a, 0x9356_96e7_8530_fb91,
];

#[test]
fn ctrl38_cascade_off_bit_identical_to_abi40()
{
	let mut a = ctbr_student(0, 10, 0x3800);
	let d_a = trained_cascade(&mut a);
	let mut b = ctbr_student(4, 1, 0x3804);
	let d_b = trained_cascade(&mut b);
	let fs = feature_stream_digest(&a, 600);
	let cpu = rows_digest(&cpu_rows(&a, 500, 0.20));
	let gpu = gpu_rows(&a, 500, 0.20).map_or(ABI40_CASCADE[4], |r| rows_digest(&r));
	let cal = calib_digest(&ctbr_student(0, 10, 0x38CB));
	let got = [d_a, d_b, fs, cpu, gpu, cal];
	eprintln!("[ctrl38-pin] cascade off {got:#018x?}");
	assert_eq!(got, ABI40_CASCADE, "the cascade lineage moved with both CTRL-38 blocks OFF");
}
