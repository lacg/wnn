//! ABI 59 (CTRL-58): the GA's near-fail tail export and the native four-input gate.
//!
//! Pinned here: the per-episode tail sink is measurement only (the scorer row bit-identical
//! with and without it); its values are the per-episode mean errors the scorer averages (their
//! mean reproduces err when nothing diverged) and a diverged episode carries the cap (finite —
//! the ZRank refuses inf); the Metal scorer's host-built list equals the CPU sink episode for
//! episode (float tolerance, as every CPU/GPU row) and so does the p99. The gate's own tests
//! live with THE gate in ram_core::fitness (golden over random pools, four-bound Deb order).

use crate::abi38_pins::{ics, stage2, ARM, ASYM, DT, EPS, G, INERTIA, KD, KT};
use crate::controller::WnnController;
use crate::ctrl51_yaw_trim_tests::{yaw_student, YawOpt};
use crate::episode_tail::{percentile_inverted_cdf, EpisodeErrSink, TAIL_P};
use ram_core::neuron_memory::BINARY;

fn cpu(c: &WnnController, steps: usize, sink: Option<&mut EpisodeErrSink>) -> [f64; 19]
{
	let (q0, w0) = ics();
	let s1 = stage2();
	let mut cc = c.clone();
	crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, EPS, steps, DT, ARM, KT, KD, INERTIA, G, 0.2, [0.0; 3],
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, 777,
		0.0, 0, 0, 0.2, c.levels_per_motor(), 4, None, None, None, 1.0, 0.4, None, Some(&s1), None, sink,
	)
}

fn trained() -> WnnController
{
	let mut c = yaw_student(0, 10, 0x5901, YawOpt::Off, BINARY);
	let st = crate::dagger_train::dagger_train_inplace_with(&mut c, &crate::abi38_tests::l4c_cfg(0.20, 4), [0.0; 3], 23, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0);
	c
}

fn bits(r: &[f64]) -> Vec<u64>
{
	r.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn tail_sink_is_inert_and_reproduces_err()
{
	let c = trained();
	let off = cpu(&c, 800, None);
	let mut sink = EpisodeErrSink::new(8.0);
	let on = cpu(&c, 800, Some(&mut sink));
	assert_eq!(bits(&off), bits(&on), "the tail sink moved the scorer row");
	assert_eq!(sink.errs.len(), EPS);
	// The identity needs no CAPPED (diverged) episode; a high-err fixture is fine.
	assert!(sink.errs.iter().all(|v| *v != 8.0), "fixture diverged: {:?}", sink.errs);
	let mean_deg = sink.errs.iter().sum::<f64>() / EPS as f64;
	assert!((mean_deg - off[1].to_degrees()).abs() < 1e-9, "per-episode mean {mean_deg} vs err {}", off[1].to_degrees());
	let p99 = percentile_inverted_cdf(&sink.errs, TAIL_P).unwrap();
	assert!(p99 >= mean_deg && p99.is_finite());
	assert_eq!(sink.row_tail()[0], EPS as f64);
}

/// A chaotic every-address student diverges; its diverged episodes carry the cap (8.0°), so
/// the tail is finite and at least the cap.
#[test]
fn tail_caps_diverged_episodes()
{
	let c = crate::ctrl51_alt_integral_tests::alt_student_cells(0x5902, 1 << 8);
	let mut sink = EpisodeErrSink::new(8.0);
	let row = cpu(&c, 2000, Some(&mut sink));
	assert!(sink.errs.iter().all(|v| v.is_finite()), "{:?}", sink.errs);
	let capped = sink.errs.iter().filter(|v| **v == 8.0).count();
	eprintln!("[tail59] chaotic student: stable {} errs {:?}", row[2], sink.errs);
	assert!(row[2] < 1.0 && capped > 0, "vacuous: nothing diverged (stable {})", row[2]);
	assert!(percentile_inverted_cdf(&sink.errs, TAIL_P).unwrap() >= 8.0);
}

/// CPU sink == the Metal scorer's host-built list (score_with_episodes →
/// per_genome_from_episode_rows), episode for episode, and so the p99.
#[cfg(target_os = "macos")]
#[test]
fn tail_cpu_matches_gpu()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok()
	{
		return;
	}
	let Ok(ev) = crate::metal_controller::ControllerRolloutEvaluator::new()
	else
	{
		return;
	};
	let c = trained();
	let mut sink = EpisodeErrSink::new(8.0);
	cpu(&c, 800, Some(&mut sink));
	let (q0, w0) = ics();
	let s1 = stage2();
	let d = crate::controller::Disturbance {
		tau_bias: [0.0; 3], gust_sigma: 0.0, gust_tau_c: 0.1, motor_asym: ASYM, gyro_sigma: 0.0107,
		gyro_bias_walk: 0.0, accel_sigma: 0.0, dropout_prob: 0.0, dropout_len_steps: 0,
		obs_delay_steps: 0, torque_scale_jitter: 0.2, seed: 777,
	};
	let (_, episodes) = ev
		.score_with_episodes(&[&c], &q0, &w0, EPS, 800, (DT, ARM, KT, INERTIA, G), KD, 0.2, [0.0; 3], Some(d), None, None, None, Some(&s1))
		.expect("gpu score");
	let gpu = &crate::episode_tail::per_genome_from_episode_rows(&episodes, 1, 8.0)[0];
	assert_eq!(gpu.len(), sink.errs.len());
	for (k, (g, c)) in gpu.iter().zip(&sink.errs).enumerate()
	{
		assert!((g - c).abs() <= 2e-3 * c.abs().max(1e-3), "episode {k}: gpu {g} cpu {c}");
	}
	let (pg, pc) = (percentile_inverted_cdf(gpu, TAIL_P).unwrap(), percentile_inverted_cdf(&sink.errs, TAIL_P).unwrap());
	eprintln!("[tail59] p99 cpu {pc:.6} gpu {pg:.6}");
	assert!((pg - pc).abs() <= 2e-3 * pc.abs().max(1e-3));
}
