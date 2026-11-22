//! CTRL-54: END-TO-END CPU ↔ Metal scorer agreement on TRAINED genomes — the measured
//! tolerance, enforced.
//!
//! WHAT DIFFERS AND WHY (measured 22/11/2026 on this fixture; root cause also in the
//! docstrings of cpu_score.rs and score_controllers_metal):
//!   * Training is device-independent: WNN_CONTROLLER_GPU_EVAL toggles only the closed-loop
//!     SCORER (the CPU DAgger trainer runs in both modes; the GPU sn>0 trainer is the separate
//!     WNN_CONTROLLER_GPU_BPTT knob). The divergence is entirely in SCORING.
//!   * Both scorers fly the same f32 plant and controller, but not the same f32 OPERATIONS:
//!     the CPU Mahony estimator is f64 (estimator.rs MahonyFilter) vs f32 in the kernel (Metal
//!     has no double); Rust/libm vs Metal transcendentals (Box-Muller gyro noise, atan2/asin/acos
//!     in features + plant); the rollout library is compiled fast-math (approximate
//!     transcendentals, FMA contraction); the kernel accumulates per-episode sums in f32, the
//!     CPU in f64. Per step that is an ulp-level floor (≤ ~3e-6 relative on an episode's mean
//!     err here, fast-math and strict alike).
//!   * A WNN reads features through THRESHOLDS, so a rare ulp difference that lands on a
//!     threshold flips an address bit and that EPISODE forks (here: 2/48 episodes of the sn=4
//!     estimator-off genome, by 1.4e-3 and 2.8e-3 relative; 0/48 for the others). Fast-math OFF
//!     does not remove the forks (same episodes, same magnitude) — parity is all-or-nothing,
//!     only bit-identical per-step arithmetic on both sides would give it.
//!   * Production scale (hundreds of neurons, 100+ episodes, 5 folds) makes some fork per batch
//!     near-certain: the 17/11 probe measured reward +5e-4..+7e-4, err −0.0010..−0.0014°,
//!     steady −0.0013..−0.0019° — the same fork mechanism, ~1e-3 relative.
//!
//! WHAT THIS PINS (tighter than the 2e-2 rollout sweeps in metal_controller.rs, which cannot
//! see a 1e-3 drift): every non-forked episode agrees to EPISODE_REL; forks stay rare
//! (≤ MAX_FORK_FRAC of episodes, each ≤ FORK_REL); the aggregate row agrees to AGG_REL_NO_FORK
//! when no episode forked and to AGG_REL_FORK otherwise; `stable` and the episode lengths agree
//! exactly. A semantic CPU/GPU mismatch (a dropped feature, a wrong window, a mis-plumbed plant
//! option) moves EVERY episode far past EPISODE_REL and fails here.

use crate::abi38_pins::{prod_cfg, prod_student, stage2, ARM, ASYM, DT, EPS, G, INERTIA, KD, KT};
use crate::controller::{Disturbance, WnnController};
use crate::cpu_score::ROW_METRICS;
use crate::metal_controller::ControllerRolloutEvaluator;
use rand::{Rng, SeedableRng};

/// Episodes per genome and steps per episode (the production episode length).
const N_EPS: usize = 48;
const STEPS: usize = 2000;
const LAG: f32 = 0.20;
const DIST_SEED: u64 = 777;

/// A non-forked episode's mean err: CPU vs Metal, relative (measured worst 3.0e-6).
const EPISODE_REL: f64 = 1e-5;
/// A forked episode's mean err: relative bound (measured worst 2.8e-3).
const FORK_REL: f64 = 1e-2;
/// At most this fraction of a genome's episodes may fork (measured 2/48).
const MAX_FORK_FRAC: f64 = 0.10;
/// Aggregate err / steady / mid / jerk / reward, relative, with no fork (measured ≤ 1.7e-6).
const AGG_REL_NO_FORK: f64 = 1e-5;
/// The same with forks in the batch (measured ≤ 7.2e-4 here, jerk; ~1e-3 at production scale).
const AGG_REL_FORK: f64 = 2e-3;

/// Scorer rows compared on the aggregate: reward, err, jerk, steady, mid.
const AGG_ROWS: [(usize, &str); 5] = [(0, "reward"), (1, "err"), (3, "jerk"), (5, "steady"), (19, "mid")];

/// Per-episode Metal row layout (score_with_episodes): [.., 4 = mean_err_rad, .., 10 = steps].
const EP_ERR: usize = 4;
const EP_STEPS: usize = 10;

struct Case
{
	sn: usize,
	action_repeat: usize,
	estimator: bool,
}

/// The production-shaped genomes: estimator on (the ABI-32+ base) at sn 0/4 and action
/// repeat 10, plus the estimator-off sn=4 genome whose episodes DO fork (the tolerance's
/// teeth — a fork-free fixture could never exercise the fork branch).
const CASES: [Case; 4] = [
	Case { sn: 0, action_repeat: 1, estimator: true },
	Case { sn: 4, action_repeat: 1, estimator: true },
	Case { sn: 0, action_repeat: 10, estimator: true },
	Case { sn: 4, action_repeat: 1, estimator: false },
];

/// TRAIN half of the end-to-end path: the CPU DAgger trainer (the only trainer either scoring
/// device runs behind) on the abi38 cf21 stage-2 config.
fn trained(case: &Case) -> WnnController
{
	let mut c = prod_student(case.sn, case.action_repeat, true, 0xC738 + case.sn as u64 * 7 + case.action_repeat as u64);
	if case.estimator
	{
		c.enable_att_estimator(2.0, 0.1).expect("estimator on");
	}
	let st = crate::dagger_train::dagger_train_inplace_rs(&mut c, &prod_cfg(LAG, 4), [0.0; 3], 4243);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "fixture trainer wrote no cells");
	c
}

fn initial_conditions() -> (Vec<f32>, Vec<f32>)
{
	let mut rng = rand::rngs::SmallRng::seed_from_u64(0x54);
	let (mut q0, mut w0) = (Vec::new(), Vec::new());
	for _ in 0..N_EPS
	{
		let ang: f32 = rng.gen_range(-0.09..0.09);
		let (hx, hy): (f32, f32) = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
		let n = (hx * hx + hy * hy).sqrt().max(1e-6);
		let h = ang / 2.0;
		q0.extend_from_slice(&[h.cos(), h.sin() * hx / n, h.sin() * hy / n, 0.0]);
		w0.extend_from_slice(&[rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), 0.0]);
	}
	(q0, w0)
}

/// The abi38 stage-2 episode set widened to N_EPS (draws cycle, starts offset per cycle).
fn stage2_wide() -> crate::stage1::Stage1Cfg
{
	let mut s = stage2();
	let off = |e: usize| (e / EPS) as f32 * 0.01;
	s.init_z = (0..N_EPS).map(|e| s.init_z[e % EPS] + off(e)).collect();
	s.init_vz = vec![0.0; N_EPS];
	s.mass = (0..N_EPS).map(|e| s.mass[e % EPS]).collect();
	s.collective_frac = (0..N_EPS).map(|e| s.collective_frac[e % EPS]).collect();
	s.init_x = (0..N_EPS).map(|e| s.init_x[e % EPS] - off(e)).collect();
	s.init_y = (0..N_EPS).map(|e| s.init_y[e % EPS] + off(e)).collect();
	s.validate(N_EPS).expect("ctrl54 stage-2 fixture must validate");
	s
}

/// L4C-like weather: motor asymmetry, gyro noise, torque-scale jitter.
fn weather() -> Disturbance
{
	Disturbance {
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
		seed: DIST_SEED,
	}
}

/// CPU scorer (cpu_score::rollout_one — score_controllers_cpu's per-genome body) with the
/// per-episode err trace.
fn cpu_score(c: &WnnController, s1: &crate::stage1::Stage1Cfg) -> ([f64; ROW_METRICS], Vec<Vec<f64>>)
{
	let (q0, w0) = initial_conditions();
	let d = weather();
	let mut trace = Vec::new();
	let mut cc = c.clone();
	let row = crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, N_EPS, STEPS, DT, ARM, KT, KD, INERTIA, G, LAG, [0.0; 3],
		true, d.tau_bias, d.gust_sigma, d.gust_tau_c, d.motor_asym, d.gyro_sigma, d.gyro_bias_walk, d.accel_sigma, d.seed,
		d.dropout_prob, d.dropout_len_steps, d.obs_delay_steps, d.torque_scale_jitter,
		c.levels_per_motor(), 4, None, None, None, 1.0, 0.4, Some(&mut trace), Some(s1), None, None,
	);
	(row, trace)
}

/// Metal scorer: aggregate row + per-episode rows.
fn gpu_score(ev: &ControllerRolloutEvaluator, c: &WnnController, s1: &crate::stage1::Stage1Cfg) -> (Vec<f64>, Vec<Vec<f64>>)
{
	let (q0, w0) = initial_conditions();
	let (agg, eps) = ev
		.score_with_episodes(&[c], &q0, &w0, N_EPS, STEPS, (DT, ARM, KT, INERTIA, G), KD, LAG, [0.0; 3], Some(weather()), None, None, None, Some(s1))
		.expect("gpu score");
	(agg[0].clone(), eps)
}

fn rel(a: f64, b: f64) -> f64
{
	(a - b).abs() / b.abs().max(1e-12)
}

/// Per-episode agreement; returns the number of forked episodes.
fn check_episodes(name: &str, trace: &[Vec<f64>], eps: &[Vec<f64>]) -> usize
{
	assert_eq!(eps.len(), N_EPS, "{name}: per-episode rows");
	let mut forks = 0usize;
	for (e, (t, g)) in trace.iter().zip(eps).enumerate()
	{
		assert_eq!(t.len() as f64, g[EP_STEPS], "{name} ep {e}: episode length (a divergence decided differently)");
		let r = rel(g[EP_ERR], t.iter().sum::<f64>() / t.len().max(1) as f64);
		assert!(r <= FORK_REL, "{name} ep {e}: mean err rel {r:.3e} > FORK_REL {FORK_REL:.0e}");
		forks += (r > EPISODE_REL) as usize;
	}
	let frac = forks as f64 / N_EPS as f64;
	assert!(frac <= MAX_FORK_FRAC, "{name}: {forks}/{N_EPS} episodes forked (> {MAX_FORK_FRAC})");
	forks
}

fn check_aggregate(name: &str, cpu: &[f64], gpu: &[f64], forks: usize)
{
	assert_eq!(gpu.len(), ROW_METRICS);
	assert_eq!(gpu[2], cpu[2], "{name}: stable rate");
	let tol = if forks == 0 { AGG_REL_NO_FORK } else { AGG_REL_FORK };
	for (i, metric) in AGG_ROWS
	{
		let r = rel(gpu[i], cpu[i]);
		eprintln!("[ctrl54] {name}: {metric:6} cpu {:.9e} gpu {:.9e} rel {r:.2e} (tol {tol:.0e})", cpu[i], gpu[i]);
		assert!(r <= tol, "{name} {metric}: gpu {} cpu {} rel {r:.3e} > {tol:.0e} ({forks} forks)", gpu[i], cpu[i]);
	}
}

/// CTRL-54: train (CPU) → score on BOTH devices, per episode and aggregate, within the
/// measured tolerance. Training determinism is pinned too: the same genome trained twice
/// carries identical cells (so any CPU/GPU difference left is the scorer's).
#[cfg(target_os = "macos")]
#[test]
fn ctrl54_train_and_score_cpu_matches_gpu_within_measured_tolerance()
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok()
	{
		return;
	}
	let Ok(ev) = ControllerRolloutEvaluator::new()
	else
	{
		assert!(metal::Device::system_default().is_none(), "GPU present but the rollout evaluator failed to build");
		return;
	};
	let s1 = stage2_wide();
	let mut total_forks = 0usize;
	for case in &CASES
	{
		let name = format!("sn{} ar{} est{}", case.sn, case.action_repeat, case.estimator);
		let c = trained(case);
		let again = trained(case);
		assert_eq!(crate::abi35_tests::cells_digest(&c), crate::abi35_tests::cells_digest(&again), "{name}: training is not deterministic");
		let (cpu, trace) = cpu_score(&c, &s1);
		let (gpu, eps) = gpu_score(&ev, &c, &s1);
		let forks = check_episodes(&name, &trace, &eps);
		eprintln!("[ctrl54] {name}: {forks}/{N_EPS} episodes forked");
		check_aggregate(&name, &cpu, &gpu, forks);
		total_forks += forks;
	}
	assert!(total_forks > 0, "vacuous: no fixture episode forked, the fork tolerance is untested");
}
