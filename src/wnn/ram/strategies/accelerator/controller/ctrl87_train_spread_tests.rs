//! CTRL-87 (ABI 68): the per-episode TRAINING motor-lag spread (train_lag_range.rs).
//!
//!   * FLAG-OFF PIN: with the field at None the ctrl86 / ctrl83 pins (ABI64_CF21, which
//!     ABI 65 and 66 are pinned to) hold on this build — trainer, CPU and Metal scorer rows,
//!     rivals, calibration — and the Metal flag-off pipeline is the ABI-66 one (FC_EP_LAG
//!     undefined).
//!   * IN-RANGE PIN: the range spelled out at the nominal lag, [0.20, 0.20], is the fixed
//!     lag bit for bit on the trainer, the CPU scorer and the calibration sampler. On Metal
//!     it runs the FC_EP_LAG specialization, whose lag filter reads T from buffer 30 — a
//!     different codegen of the same arithmetic (measured: rows agree to ~1e-7 relative,
//!     not bit for bit, the CTRL-86 FC_ACTUATOR_EXT lesson) — held to 1e-6 relative (f32 scale).
//!   * the spread reaches every host it should (the trainer digest and the scorer rows move);
//!   * CPU == GPU with the spread on (`cpu_fallback_matches_gpu_ctrl87_*`);
//!   * the draw covers [lo, hi], is a pure function of the episode seed, and the trainer's
//!     plant takes it per episode while the teachers and the student's actuator twin stay at
//!     the nominal 0.20 s.

use crate::abi38_pins::{Fnv, DT, EPS};
use crate::abi38_tests::ctbr_student;
use crate::actuator_spec::lag_alpha;
use crate::controller::WnnController;
use crate::ctrl86_flag_off_pins::{
	cpu_rows_lag, gpu_rows_lag, l2b_cfg, real_life_plant, rows_digest, stage2_with, trained_on, Spelling, ABI64_CF21, LAG,
};
use crate::dagger_train::{AirframeRs, RewardGatedConfigPacked};
use crate::stage1::Stage1Cfg;
use crate::train_lag_range::TrainLagRange;

/// The proposed l2bnmw training range (driver ab_nm87.sh).
const LO: f32 = 0.15;
const HI: f32 = 0.30;
const STEPS: usize = 500;
/// The fixture's weather seed (ctrl86_flag_off_pins cpu_rows_lag / gpu_rows_lag).
const DIST_SEED: u64 = 777;

fn spread() -> TrainLagRange
{
	TrainLagRange::new(LO, HI).unwrap()
}

fn cfg_with(range: Option<(f32, f32)>) -> RewardGatedConfigPacked
{
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.train_motor_lag_range = range;
	c.validate_lag_range().expect("the l2b fixture is a valid lag-range host");
	c
}

/// The stage-2 real-life episodes, each flying its own plant lag when `lags` is non-empty.
fn episodes(lags: Vec<f32>) -> Stage1Cfg
{
	let mut s1 = stage2_with(real_life_plant(Spelling::Legacy));
	s1.ep_motor_lag = lags;
	s1.validate(EPS).expect("fixture");
	s1
}

/// ctrl86_flag_off_pins' calibration digest (PID teacher, 3 episodes) with `range` armed.
fn calib_digest(range: Option<(f32, f32)>) -> u64
{
	let mut cfg = l2b_cfg(0, Spelling::Legacy);
	cfg.train_motor_lag_range = range;
	cfg.validate_lag_range().expect("calib fixture");
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

fn close(gpu: f64, cpu: f64, rel: f64, floor: f64, what: &str)
{
	let tol = floor.max(cpu.abs() * rel);
	assert!((gpu - cpu).abs() <= tol, "{what}: gpu={gpu} cpu={cpu} tol={tol}");
}

/// CPU == GPU on reward, err, pos, alt, dx, dy, mid (the ctrl83 / ctrl86 tolerance policy).
fn assert_parity(c: &WnnController, s1: &Stage1Cfg, what: &str) -> Vec<f64>
{
	let cr = cpu_rows_lag(&mut c.clone(), s1, STEPS, LAG);
	if let Some(gr) = gpu_rows_lag(c, s1, STEPS, LAG)
	{
		close(gr[1], cr[1], 2e-2, 1e-4, &format!("{what} err"));
		close(gr[0], cr[0], 5e-2, 1e-3, &format!("{what} reward"));
		for (i, name) in [(13, "pos"), (14, "alt"), (15, "dx"), (16, "dy")]
		{
			close(gr[i], cr[i], 5e-2, 1e-3, &format!("{what} {name}"));
		}
		close(gr[19], cr[19], 2e-2, 1e-4, &format!("{what} mid"));
		eprintln!("[ctrl87-parity] {what}: cpu err {:.6} gpu err {:.6} | cpu alt {:.5} gpu alt {:.5}", cr[1], gr[1], cr[14], gr[14]);
	}
	cr.to_vec()
}

fn moved(a: &[f64], b: &[f64], what: &str)
{
	assert!(a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9), "{what}: the spread did not move the row — dead plumbing");
}

/// The range spelled out at the nominal lag is the fixed-lag lineage, digest for digest.
#[test]
fn ctrl87_degenerate_range_is_the_fixed_lag()
{
	let cfg = cfg_with(Some((LAG, LAG)));
	assert_eq!(l2b_cfg(4, Spelling::Legacy).train_motor_lag_range, None, "the spread defaults OFF");
	let (c, d_train) = trained_on(&cfg);
	assert_eq!(d_train, ABI64_CF21[0], "trainer, [0.20, 0.20]");
	let s1 = episodes(vec![LAG; EPS]);
	assert_eq!(rows_digest(&cpu_rows_lag(&mut c.clone(), &s1, STEPS, LAG)), ABI64_CF21[2], "CPU scorer, [0.20, 0.20]");
	if let (Some(g), Some(fixed)) = (gpu_rows_lag(&c, &s1, STEPS, LAG), gpu_rows_lag(&c, &episodes(vec![]), STEPS, LAG))
	{
		assert_eq!(rows_digest(&fixed), ABI64_CF21[3], "Metal scorer, fixed lag (FC_EP_LAG undefined)");
		for (i, (a, b)) in g.iter().zip(&fixed).enumerate().take(20)
		{
			close(*a, *b, 1e-6, 1e-9, &format!("Metal [0.20, 0.20] vs fixed lag, metric {i}"));
		}
	}
	assert_eq!(calib_digest(Some((LAG, LAG))), ABI64_CF21[6], "calibration sampler, [0.20, 0.20]");
}

/// The spread reaches the trainer (DAgger + per-round eval), the calibration sampler and
/// both scorers: everything moves off the pin.
#[test]
fn ctrl87_spread_moves_trainer_calibration_and_scorers()
{
	let cfg = cfg_with(Some((LO, HI)));
	let (_, d_train) = trained_on(&cfg);
	assert_ne!(d_train, ABI64_CF21[0], "the trainer never saw the lag spread");
	assert_ne!(calib_digest(Some((LO, HI))), ABI64_CF21[6], "the calibration sampler never saw the lag spread");
	let (c, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	let off = cpu_rows_lag(&mut c.clone(), &episodes(vec![]), STEPS, LAG);
	let on = cpu_rows_lag(&mut c.clone(), &episodes(spread().per_episode(DIST_SEED, EPS)), STEPS, LAG);
	moved(&off, &on, "CPU scorer");
}

/// CPU == GPU with every episode on its own plant lag (cascade + motor-level obs_pwm, whose
/// twin is read every step — a twin on the drawn lag would split the two).
#[test]
fn cpu_fallback_matches_gpu_ctrl87_lag_spread()
{
	let lags = spread().per_episode(DIST_SEED, EPS);
	let (cascade, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	let base = assert_parity(&cascade, &episodes(vec![]), "cascade fixed lag");
	let on = assert_parity(&cascade, &episodes(lags.clone()), "cascade lag spread");
	moved(&base, &on, "cascade lag spread");
	let (motor, _) = crate::abi38_pins::train(0, 1, LAG, 4, true);
	assert_parity(&motor, &episodes(lags), "motor+obs_pwm lag spread");
}

/// The draw: inside [lo, hi], covering it, uniform-ish, a pure function of the seed.
#[test]
fn training_draw_covers_the_range_and_is_deterministic()
{
	let r = spread();
	let n = 20_000u64;
	let draws: Vec<f32> = (0..n).map(|s| r.draw(crate::controller::disturbance_episode_seed(0xC787, s))).collect();
	let (mn, mx) = draws.iter().fold((f32::MAX, f32::MIN), |(a, b), &t| (a.min(t), b.max(t)));
	let mean = draws.iter().map(|&t| t as f64).sum::<f64>() / n as f64;
	assert!(mn >= LO && mx <= HI, "draw outside [{LO}, {HI}]: [{mn}, {mx}]");
	let w = (HI - LO) as f64;
	assert!((mn - LO) as f64 <= 0.002 * w && (HI - mx) as f64 <= 0.002 * w, "range not covered: [{mn}, {mx}]");
	assert!((mean - 0.5 * (LO + HI) as f64).abs() < 0.01 * w, "mean {mean} far from the midpoint");
	let below_nominal = draws.iter().filter(|&&t| t < LAG).count() as f64 / n as f64;
	assert!((below_nominal - (LAG - LO) as f64 / w).abs() < 0.02, "P(T < 0.20) = {below_nominal}");
	assert_eq!(r.per_episode(DIST_SEED, 8), r.per_episode(DIST_SEED, 8), "not deterministic");
	assert_eq!(r.per_episode(DIST_SEED, 8)[..4], r.per_episode(DIST_SEED, 4)[..], "episode e's lag depends on the pool size");
	assert_ne!(r.per_episode(DIST_SEED, 8), r.per_episode(DIST_SEED + 1, 8), "the base seed does not key the draw");
	eprintln!("[ctrl87-draw] n={n} min {mn:.5} max {mx:.5} mean {mean:.5} P(T<0.20) {below_nominal:.4}");
}

/// The trainer's plant takes T_e per episode (from the episode's weather seed); the teachers
/// and the student's twin are built on the nominal 0.20 s.
#[test]
fn trainer_plant_draws_per_episode_while_models_stay_nominal()
{
	let cfg = cfg_with(Some((LO, HI)));
	let af = AirframeRs::from_cfg(&cfg);
	assert_eq!(af.motor_lag_s, LAG, "the teachers' airframe is the nominal one");
	let mut sim = af.sim();
	assert_eq!(sim.motor_lag(), LAG, "the sim is built at the nominal lag");
	let mut seen = Vec::new();
	for ep_seed in [1u64, 2, 3, 99, 0xDEAD_BEEF]
	{
		crate::dagger_train::arm_cfg_disturbance(&mut sim, &cfg, ep_seed);
		assert_eq!(sim.motor_lag(), spread().draw(ep_seed), "episode {ep_seed}: the plant is not on its draw");
		seen.push(sim.motor_lag());
	}
	assert!(seen.iter().any(|&t| (t - seen[0]).abs() > 1e-4), "every episode flew one lag: {seen:?}");
	// Off: arming the weather leaves the plant at the nominal lag.
	let off = l2b_cfg(4, Spelling::Legacy);
	let mut sim_off = AirframeRs::from_cfg(&off).sim();
	crate::dagger_train::arm_cfg_disturbance(&mut sim_off, &off, 1);
	assert_eq!(sim_off.motor_lag(), LAG);
	// The student's twin after a whole training run with the spread: still the nominal lag.
	let (c, _) = trained_on(&cfg);
	assert_eq!(c.observer_lag_alpha(), lag_alpha(LAG, DT), "the trainer moved the student's twin");
}

/// The search scorer: each episode flies its draw, the twin it arms is the nominal lag.
#[test]
fn search_scorer_twin_stays_nominal_under_the_spread()
{
	let (motor, _) = crate::abi38_pins::train(0, 1, LAG, 4, true);
	let mut cc = motor.clone();
	cpu_rows_lag(&mut cc, &episodes(spread().per_episode(DIST_SEED, EPS)), 100, LAG);
	assert_eq!(cc.observer_lag_alpha(), lag_alpha(LAG, DT), "the scorer armed the twin off the nominal lag");
}

/// Refusals at the trainer config (the scorers share scorer_episode_lags, unit-tested there).
#[test]
fn trainer_config_refuses_bad_hosts()
{
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.train_motor_lag_range = Some((0.3, 0.2));
	assert!(c.validate_lag_range().is_err(), "lo > hi");
	c.train_motor_lag_range = Some((LO, HI));
	c.dist_enabled = false;
	assert!(c.validate_lag_range().is_err(), "no weather");
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.train_motor_lag_range = Some((LO, HI));
	c.motor_lag_s = 0.0;
	assert!(c.validate_lag_range().is_err(), "no nominal lag");
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.train_motor_lag_range = Some((LO, HI));
	c.motor_lag_down_s = 0.3;
	assert!(c.validate_lag_range().is_err(), "extended actuator");
}

