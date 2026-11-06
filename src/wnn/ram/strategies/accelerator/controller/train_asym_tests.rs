//! ABI 38 (Luiz 06/11/2026, "let's train the same way, asymmetric"): the trainer
//! draws the D3 motor-asymmetry δ per TRAINING episode (controller::train_motor_asym,
//! RewardGatedConfigPacked.dist_motor_asym_mag) with the scorer's distribution
//! instead of flying the fixed multipliers alone.
//!
//! Pinned here:
//!   * mag = 0 is the ABI-37 trainer bit for bit (also abi38_pins, which never sets it);
//!   * the draw is the scorer's distribution: base·(1 + U(−mag, +mag)) iid per motor,
//!     a fresh airframe every episode, reproducible from the episode seed;
//!   * every training host flies it — DAgger rollouts, the per-round eval, the
//!     calibration sampler, the MEMORY recorder;
//!   * the GPU sn>0 EDRA-BPTT walk stays bit-identical to the CPU under it. No Metal kernel simulates a TRAINING plant (they replay
//!     the CPU rollouts' recorded streams), so the asymmetric plant reaches the GPU
//!     only through those streams — which is exactly what these parity tests cover.

use crate::abi38_pins::{cells_and_stats_digest, prod_cfg, prod_student};
use crate::controller::{train_motor_asym, AttitudeSim, WnnController};
use crate::dagger_train::{dagger_train_inplace_with, rollout_and_label_rs, AirframeRs, RewardGatedConfigPacked, TrajectoryRs};
use rand::rngs::SmallRng;
use rand::SeedableRng;

const MAG: f32 = 0.2; // L4C: ±20%

#[test]
fn zero_magnitude_is_the_fixed_multiplier_bit_for_bit()
{
	let base = [1.12f32, 0.9, 1.05, 0.95];
	for seed in [0u64, 1, 0xDEAD_BEEF, u64::MAX]
	{
		assert_eq!(train_motor_asym(base, 0.0, seed).map(f32::to_bits), base.map(f32::to_bits));
	}
}

/// The scorer's distribution (training.DisturbanceConfig.resolved_motor_asym):
/// δ ~ U(−mag, +mag) iid per motor, applied multiplicatively on the fixed part.
#[test]
fn draw_is_uniform_per_motor_and_fresh_per_episode()
{
	let base = [1.0f32, 1.0, 1.0, 1.0];
	let n = 20_000u64;
	let mut sum = [0.0f64; 4];
	let mut sq = [0.0f64; 4];
	let mut cross = 0.0f64;
	let (mut lo, mut hi) = (f32::MAX, f32::MIN);
	for ep in 0..n
	{
		let seed = crate::controller::disturbance_episode_seed(0x5EED, ep);
		let a = train_motor_asym(base, MAG, seed);
		assert_eq!(a, train_motor_asym(base, MAG, seed), "not reproducible from the seed");
		for m in 0..4
		{
			let d = (a[m] - 1.0) as f64;
			sum[m] += d;
			sq[m] += d * d;
			lo = lo.min(a[m]);
			hi = hi.max(a[m]);
		}
		cross += ((a[0] - 1.0) * (a[1] - 1.0)) as f64;
	}
	let var_u = (2.0 * MAG as f64).powi(2) / 12.0;
	for m in 0..4
	{
		let mean = sum[m] / n as f64;
		let var = sq[m] / n as f64 - mean * mean;
		assert!(mean.abs() < 0.004, "motor {m}: mean δ {mean}");
		assert!((var / var_u - 1.0).abs() < 0.05, "motor {m}: var {var} vs uniform {var_u}");
	}
	assert!((cross / n as f64).abs() < 0.001, "motors 0/1 correlated: {}", cross / n as f64);
	assert!(lo >= 1.0 - MAG && hi <= 1.0 + MAG && lo < 1.0 - 0.99 * MAG && hi > 1.0 + 0.99 * MAG, "support [{lo}, {hi}]");
	// Multiplicative on the fixed part, as the scorer resolves it.
	let b = [1.12f32, 0.9, 1.05, 0.95];
	let s = crate::controller::disturbance_episode_seed(7, 3);
	let one = train_motor_asym([1.0; 4], MAG, s);
	let got = train_motor_asym(b, MAG, s);
	for m in 0..4
	{
		assert_eq!(got[m].to_bits(), (b[m] * one[m]).to_bits(), "motor {m}: not base × (1 + δ)");
	}
}

fn cfg_mag(lag: f32, mag: f32) -> RewardGatedConfigPacked
{
	let mut c = prod_cfg(lag, 4);
	c.dist_motor_asym_mag = mag;
	c
}

/// The trainer flies it: the same rollout seed with mag > 0 changes the trajectory
/// (the plant moved), mag = 0 set explicitly is the default config bit for bit.
#[test]
fn training_rollout_flies_the_per_episode_airframe()
{
	let roll = |cfg: &RewardGatedConfigPacked| -> Vec<[f32; 3]> {
		let mut c = prod_student(0, 1, true, 0xA5);
		let af = AirframeRs::from_cfg(cfg);
		let mut teacher = af.teacher(cfg.teacher);
		let mut sim: AttitudeSim = af.sim();
		let mut rng = SmallRng::seed_from_u64(3);
		let mut g = Vec::new();
		for _ in 0..3
		{
			g.extend(rollout_and_label_rs(&mut c, &mut teacher, &mut sim, cfg, 0.1, &mut rng, [0.0; 3]).gyros);
		}
		g
	};
	let base = roll(&prod_cfg(0.0, 4));
	assert_eq!(roll(&cfg_mag(0.0, 0.0)), base, "mag = 0 is not the ABI-37 rollout");
	assert_ne!(roll(&cfg_mag(0.0, MAG)), base, "mag > 0 never reached the training plant");
}

/// The per-round checkpoint eval, the calibration sampler and the MEMORY recorder
/// fly it too (each moves with mag > 0 and is untouched at mag = 0).
#[test]
fn every_training_host_flies_it()
{
	use crate::calib_sampler::sample_calibration_features_rs;
	let calib = |cfg: &RewardGatedConfigPacked| {
		let mut c = prod_student(0, 1, true, 0xA6);
		sample_calibration_features_rs(&mut c, cfg, 2, 9)
	};
	assert_eq!(calib(&cfg_mag(0.2, 0.0)), calib(&prod_cfg(0.2, 4)));
	assert_ne!(calib(&cfg_mag(0.2, MAG)), calib(&prod_cfg(0.2, 4)), "calibration sampler ignored the training asymmetry");

	let train = |cfg: &RewardGatedConfigPacked| {
		let mut c = prod_student(0, 1, true, 0xA7);
		let st = dagger_train_inplace_with(&mut c, cfg, [0.0; 3], 11, false);
		(st.iter_mean_err_deg.clone(), cells_and_stats_digest(&c, Some(&st)))
	};
	let (e0, d0) = train(&prod_cfg(0.2, 4));
	assert_eq!(train(&cfg_mag(0.2, 0.0)), (e0.clone(), d0), "mag = 0 trainer moved");
	let (e1, _) = train(&cfg_mag(0.2, MAG));
	assert_ne!(e1, e0, "the per-round eval / training rollouts ignored the asymmetry");

	let rec = |cfg: &RewardGatedConfigPacked| crate::ctrl17_tests::record_universe_for(cfg, &prod_student(0, 1, true, 0xA8));
	assert_eq!(rec(&cfg_mag(0.0, 0.0)), rec(&prod_cfg(0.0, 4)), "mag = 0 recorder moved");
	assert_ne!(rec(&cfg_mag(0.0, MAG)), rec(&prod_cfg(0.0, 4)), "the MEMORY recorder ignored the asymmetry");
}

/// sn>0 EDRA-BPTT: the GPU-resident walk equals the CPU walk under the asymmetric
/// trainer (whole dagger_train, cells + stats + egress order).
#[test]
fn asymmetric_sn4_trainer_cpu_matches_gpu_walk()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok() || metal::Device::system_default().is_none()
	{
		return;
	}
	for lag in [0.0f32, 0.20]
	{
		let run = |gpu: bool| {
			let mut c = prod_student(4, 1, true, 0xA9);
			let st = dagger_train_inplace_with(&mut c, &cfg_mag(lag, MAG), [0.0; 3], 13, gpu);
			let ex = c.export_cells();
			(cells_and_stats_digest(&c, Some(&st)), crate::abi35_tests::cells_order_digest(&ex.0, &ex.1))
		};
		let before = crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed);
		let gpu = run(true);
		assert!(crate::dagger_train::GPU_BPTT_ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > before, "GPU walk never ran");
		assert_eq!(run(false), gpu, "lag {lag}: GPU walk diverged from the CPU walk under per-episode asymmetry");
	}
}

/// Rollouts of the asymmetric trainer (the gated batch the split trainer sees).
pub(crate) fn asym_batch(cfg: &RewardGatedConfigPacked, student: &mut WnnController) -> Vec<TrajectoryRs>
{
	let af = AirframeRs::from_cfg(cfg);
	let mut teacher = af.teacher(cfg.teacher);
	let mut sim: AttitudeSim = af.sim();
	let mut rng = SmallRng::seed_from_u64(17);
	(0..6).map(|_| rollout_and_label_rs(student, &mut teacher, &mut sim, cfg, 0.1, &mut rng, [0.0; 3])).collect()
}

// The state-split trainer replays the recorded streams; its CPU/GPU parity is
// stream-agnostic and is exercised on the planting fixture (metal_controller
// parity_split_train_loop, abi38_tests::ctbr_split_trainer_cpu_matches_gpu). The
// real asymmetric rollouts find no clean separator, so a parity check on them would
// be vacuous — removed rather than kept as a test that cannot fail.
