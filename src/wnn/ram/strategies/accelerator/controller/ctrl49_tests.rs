//! ABI 49: the STUDENT feature sampler behind `--threshold-refit-from-student`
//! (calib_sampler::sample_student_features_rs).
//!
//! The Python collector it replaces flew an attitude-only Python plant and refused
//! the ABI-32 plant, so on every real-life recipe the refit silently fell back to the
//! teacher ladder (docs/ctrl_dres_fix_offline_tests.md A.4). These pins fly the
//! rf_c200_dres runtime surface (scorer_chunk_tests::dres_student: 25-feature cascade
//! student, Mahony estimator, delta law, action_repeat 10, lag 0.20, specific-force
//! IMU + rotor drag 0.78, translation, L4C weather) and check that the sampler
//!   * runs on that plant at all,
//!   * flies the STUDENT (its columns differ from the teacher sampler's on the same
//!     cfg + seed, which draws the identical episodes),
//!   * reads memory and never writes it,
//!   * is deterministic in its seed and samples every physical step (the teacher
//!     sampler's convention).

use crate::abi38_pins::cells_and_stats_digest;
use crate::abi38_tests::l4c_cfg;
use crate::calib_sampler::{sample_calibration_features_rs, sample_student_features_rs};
use crate::dagger_train::RewardGatedConfigPacked;
use crate::scorer_chunk_tests::dres_student;

const EPS: usize = 3;
const STEPS: usize = 300;
const AR: usize = 10;

/// The reference config as Python builds it (reference_packed_config: expert_drives on).
fn dres_reference_cfg() -> RewardGatedConfigPacked
{
	let mut cfg = l4c_cfg(0.20, 4);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	cfg.expert_drives = true;
	cfg.steps_per_episode = STEPS;
	cfg
}

#[test]
fn student_sampler_runs_on_the_dres_plant_and_never_writes()
{
	let c = dres_student();
	let cfg = dres_reference_cfg();
	let before = cells_and_stats_digest(&c, None);
	let mut a = c.clone();
	let s = sample_student_features_rs(&mut a, &cfg, EPS, 0x49);
	assert_eq!(cells_and_stats_digest(&a, None), before, "the student sampler wrote cells");
	assert_eq!(s.len(), c.num_features(), "one column per feature");
	let n = s[0].len();
	assert!(n > EPS * STEPS / AR && n <= EPS * STEPS, "every physical step: {n} samples for {EPS}x{STEPS} at ar {AR}");
	assert!(s.iter().all(|col| col.len() == n), "ragged columns");
	let mut b = c.clone();
	assert_eq!(sample_student_features_rs(&mut b, &cfg, EPS, 0x49), s, "not deterministic in its seed");
}

#[test]
fn student_sampler_flies_the_student_not_the_teacher()
{
	let c = dres_student();
	let cfg = dres_reference_cfg();
	let s = sample_student_features_rs(&mut c.clone(), &cfg, EPS, 0x49);
	let t = sample_calibration_features_rs(&mut c.clone(), &cfg, EPS, 0x49);
	// Same draws, different pilot: the attitude-error columns must move.
	let moved = (0..c.num_features()).filter(|&f| t[f] != s[f]);
	assert!(moved.count() > c.num_features() / 2, "the student sampler reproduced the teacher's states");
}
