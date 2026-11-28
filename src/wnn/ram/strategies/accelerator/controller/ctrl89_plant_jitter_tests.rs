//! CTRL-89 (ABI 70): per-episode plant INERTIA scaling + CONTROL LATENCY (plant_jitter.rs).
//!
//!   * FLAG-OFF: the ctrl86 / ctrl83 / ctrl87 pins (ABI64_CF21) hold on this build untouched;
//!     here the zeros SPELLED OUT (j = 0, latency [0, 0] ms) are the same lineage on the
//!     trainer, the CPU scorer and the rivals, and resolve to EpisodePlant::OFF so the Metal
//!     scorer keeps the ABI-68 pipeline (FC_EP_JIT undefined).
//!   * the FC_EP_JIT specialization fed trivial draws (s = 1, n = 0) is the fixed plant on
//!     the CPU bit for bit and on Metal to f32 scale (the FC_EP_LAG lesson);
//!   * CPU == GPU with both on (cascade and motor-level obs_pwm, symmetric and extended
//!     actuator — the ring sits in both kernel paths);
//!   * a latency of exactly n steps delays the command by exactly n steps;
//!   * the trainer's plant takes the inertia / latency draws per episode while the
//!     teachers' airframe (and so every design model) stays at the nominal inertia;
//!   * the jitter reaches the trainer, the scorers and the rivals (no dead plumbing).

use crate::abi38_pins::{ics, Fnv, ARM, ASYM, DT, EPS, G, INERTIA, KD, KT};
use crate::abi38_pins::{PID_ATT, PID_ATT_HZ, PID_HOVER_N, PID_LPF_HZ, PID_OUT_LIMIT_N, PID_RATE};
use crate::abi38_tests::ctbr_student;
use crate::controller::{AttitudeSim, WnnController};
use crate::ctrl86_flag_off_pins::{
	cpu_rows_lag, gpu_rows_lag, l2b_cfg, real_life_plant, rows_digest, stage2_with, trained_on, Spelling, ABI64_CF21, LAG,
	ROTOR_DRAG,
};
use crate::dagger_train::{AirframeRs, RewardGatedConfigPacked};
use crate::plant_jitter::{scorer_episode_plant, EpisodePlant, PlantJitter};
use crate::plant_opts::PlantOpts;
use crate::stage1::Stage1Cfg;

/// A wide training-arm-sized spread (wider than the proposed golden cell, ±0.20 / 8-12 ms, so
/// every path is exercised hard): inertia ±30 %, latency 2-12 ms.
const J: f32 = 0.30;
const LAT_MS: (f32, f32) = (2.0, 12.0);
const STEPS: usize = 500;
/// The fixture's weather seed (ctrl86_flag_off_pins cpu_rows_lag / gpu_rows_lag / rival).
const DIST_SEED: u64 = 777;

fn jitter() -> PlantJitter
{
	PlantJitter::new(J, Some(LAT_MS), DT).unwrap()
}

fn draws() -> EpisodePlant
{
	jitter().per_episode(DIST_SEED, EPS)
}

fn cfg_with(j: f32, lat: Option<(f32, f32)>) -> RewardGatedConfigPacked
{
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.train_inertia_jitter = j;
	c.train_latency_ms = lat;
	c.validate_plant_jitter().expect("the l2b fixture is a valid jitter host");
	c
}

/// The stage-2 real-life episodes on plant `p`, each flying `ep_plant`.
fn episodes_on(p: PlantOpts, ep_plant: EpisodePlant) -> Stage1Cfg
{
	let mut s1 = stage2_with(p);
	s1.ep_plant = ep_plant;
	s1.validate(EPS).expect("fixture");
	s1
}

fn episodes(ep_plant: EpisodePlant) -> Stage1Cfg
{
	episodes_on(real_life_plant(Spelling::Legacy), ep_plant)
}

/// ctrl86_flag_off_pins::rival_kw with the CTRL-89 kwargs (stress off).
fn rival_j(teacher: u8, s1: &Stage1Cfg, j: f32, lat: Option<(f32, f32)>) -> (f64, f64, f64, f64, f64, f64, f64, f64)
{
	let (q0, w0) = ics();
	crate::dagger_train::score_classical_baseline(
		teacher, q0, w0, 600, 5.0,
		true, [0.0; 3], 0.0, 0.1, ASYM, 0.0107, 0.0, 0.0, DIST_SEED, 0.0, 0, 0, 0.2,
		ARM, KT, KD, INERTIA, G, DT,
		PID_ATT, PID_RATE, PID_OUT_LIMIT_N, PID_HOVER_N, PID_ATT_HZ, PID_LPF_HZ,
		true, 2.0, 0.1,
		true, 0.0, s1.init_z.clone(), s1.init_vz.clone(), s1.mass.clone(), s1.collective_frac.clone(),
		2.0, 1.0, 0.25,
		s1.init_x.clone(), s1.init_y.clone(), 1.0, 1.0, 0.5236,
		LAG, true, ROTOR_DRAG, true, if teacher == 3 { 0 } else { 10 }, 0, None, s1.plant.motor_lag_down_s, s1.plant.pwm_min,
		None, None, 1.0,
		j, lat,
	)
	.map(crate::dagger_train::rival_first8)
	.expect("rival must score")
}

fn r8(r: (f64, f64, f64, f64, f64, f64, f64, f64)) -> u64
{
	rows_digest(&[r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7])
}

fn close(gpu: f64, cpu: f64, rel: f64, floor: f64, what: &str)
{
	let tol = floor.max(cpu.abs() * rel);
	assert!((gpu - cpu).abs() <= tol, "{what}: gpu={gpu} cpu={cpu} tol={tol}");
}

/// CPU == GPU on reward, err, pos, alt, dx, dy, mid (the ctrl83 / ctrl86 / ctrl87 policy).
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
		eprintln!("[ctrl89-parity] {what}: cpu err {:.6} gpu err {:.6} | cpu alt {:.5} gpu alt {:.5}", cr[1], gr[1], cr[14], gr[14]);
	}
	cr.to_vec()
}

fn moved(a: &[f64], b: &[f64], what: &str)
{
	assert!(a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-9), "{what}: the jitter did not move the row — dead plumbing");
}

/// The zeros spelled out are the ABI-68 lineage on every host, and resolve to OFF.
#[test]
fn ctrl89_zeros_spelled_out_are_the_abi68_lineage()
{
	let base = l2b_cfg(4, Spelling::Legacy);
	assert_eq!((base.train_inertia_jitter, base.train_latency_ms), (0.0, None), "the jitter defaults OFF");
	let cfg = cfg_with(0.0, Some((0.0, 0.0)));
	assert!(cfg.plant_jitter().is_off());
	let (c, d_train) = trained_on(&cfg);
	assert_eq!(d_train, ABI64_CF21[0], "trainer, j = 0, latency [0, 0] ms");
	let ep = scorer_episode_plant(0.0, Some((0.0, 0.0)), DT, true, true, DIST_SEED, EPS, "t").unwrap();
	assert_eq!(ep, EpisodePlant::OFF, "zeros must resolve to OFF (no FC_EP_JIT specialization)");
	let s1 = episodes(ep);
	assert_eq!(rows_digest(&cpu_rows_lag(&mut c.clone(), &s1, STEPS, LAG)), ABI64_CF21[2], "CPU scorer, zeros");
	if let Some(g) = gpu_rows_lag(&c, &s1, STEPS, LAG)
	{
		assert_eq!(rows_digest(&g), ABI64_CF21[3], "Metal scorer, zeros (FC_EP_JIT undefined)");
	}
	let s0 = stage2_with(real_life_plant(Spelling::Legacy));
	assert_eq!(r8(rival_j(0, &s0, 0.0, Some((0.0, 0.0)))), ABI64_CF21[4], "rival PID, zeros");
	assert_eq!(r8(rival_j(4, &s0, 0.0, Some((0.0, 0.0)))), ABI64_CF21[5], "rival mpcof, zeros");
}

/// Trivial draws (s = 1, n = 0) forced through the FC_EP_JIT path: CPU bit for bit, Metal
/// to f32 scale (a different codegen of the same arithmetic).
#[test]
fn ctrl89_trivial_draws_are_the_fixed_plant()
{
	let (c, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	let trivial = EpisodePlant { inertia_scale: vec![1.0; EPS], latency_steps: vec![0; EPS] };
	let s1 = episodes(trivial);
	assert!(s1.ep_jit_on() && s1.gpu_stride() == 8);
	assert_eq!(rows_digest(&cpu_rows_lag(&mut c.clone(), &s1, STEPS, LAG)), ABI64_CF21[2], "CPU scorer, s = 1, n = 0");
	if let (Some(g), Some(fixed)) = (gpu_rows_lag(&c, &s1, STEPS, LAG), gpu_rows_lag(&c, &episodes(EpisodePlant::OFF), STEPS, LAG))
	{
		assert_eq!(rows_digest(&fixed), ABI64_CF21[3], "Metal scorer, off");
		for (i, (a, b)) in g.iter().zip(&fixed).enumerate().take(20)
		{
			close(*a, *b, 1e-6, 1e-9, &format!("Metal s = 1, n = 0 vs off, metric {i}"));
		}
	}
}

/// CPU == GPU with every episode on its own inertia and delay: cascade and motor-level
/// obs_pwm (its twin is read every step), symmetric and extended actuator, and each
/// perturbation alone.
#[test]
fn cpu_fallback_matches_gpu_ctrl89_plant_jitter()
{
	let (cascade, _) = trained_on(&l2b_cfg(4, Spelling::Legacy));
	let base = assert_parity(&cascade, &episodes(EpisodePlant::OFF), "cascade off");
	let on = assert_parity(&cascade, &episodes(draws()), "cascade inertia+latency");
	moved(&base, &on, "cascade inertia+latency");
	let inertia_only = PlantJitter::new(J, None, DT).unwrap().per_episode(DIST_SEED, EPS);
	moved(&base, &assert_parity(&cascade, &episodes(inertia_only), "cascade inertia only"), "inertia only");
	let lat_only = PlantJitter::new(0.0, Some(LAT_MS), DT).unwrap().per_episode(DIST_SEED, EPS);
	moved(&base, &assert_parity(&cascade, &episodes(lat_only), "cascade latency only"), "latency only");
	let ext = real_life_plant(Spelling::Legacy).with_actuator(LAG, 0.30, 0.05).expect("extended actuator");
	assert_parity(&cascade, &episodes_on(ext, draws()), "cascade extended actuator + jitter");
	let (motor, _) = crate::abi38_pins::train(0, 1, LAG, 4, true);
	assert_parity(&motor, &episodes(draws()), "motor+obs_pwm inertia+latency");
	// With the CTRL-87 lag spread too: the record is [.., T_e, s_e, n_e] (stride 9).
	let mut both = episodes(draws());
	both.ep_motor_lag = crate::train_lag_range::TrainLagRange::new(0.15, 0.30).unwrap().per_episode(DIST_SEED, EPS);
	assert_eq!(both.gpu_stride(), 9);
	assert_parity(&cascade, &both, "cascade lag spread + jitter");
}

/// A latency of exactly n steps: the plant receives command t − n (the first command
/// before then); n = 0 is the identity; reset re-seeds the ring.
#[test]
fn latency_delays_the_command_by_exactly_n_steps()
{
	let cmds: Vec<[f32; 4]> = (0..8).map(|t| [0.1 + 0.1 * t as f32, 0.2, 0.3 - 0.01 * t as f32, 0.4]).collect();
	let mut sim = AttitudeSim::new(DT, ARM, KT, KD, INERTIA, G); // lag off: the actuator is clamp + delay
	assert_eq!(sim.control_latency_steps(), 0);
	for c in &cmds
	{
		assert_eq!(sim.actuate_for_test(*c), *c, "n = 0 must be the identity");
	}
	for n in [1u32, 3, 32]
	{
		sim.set_control_latency_core(n);
		assert_eq!(sim.control_latency_steps(), n);
		for (t, c) in cmds.iter().enumerate()
		{
			let want = cmds[t.saturating_sub(n as usize)];
			let want = if t < n as usize { cmds[0] } else { want };
			assert_eq!(sim.actuate_for_test(*c), want, "n = {n}, step {t}");
		}
		// A new episode: the ring re-seeds with that episode's first command.
		sim.reset(None, None);
		let fresh = [0.9, 0.8, 0.7, 0.6];
		assert_eq!(sim.actuate_for_test(fresh), fresh, "n = {n}: reset did not re-seed the ring");
		sim.reset(None, None);
	}
	// One step of delay through the whole plant: the delayed sim at step t+1 is the
	// undelayed sim fed command t.
	let (mut a, mut b) = (AttitudeSim::new(DT, ARM, KT, KD, INERTIA, G), AttitudeSim::new(DT, ARM, KT, KD, INERTIA, G));
	b.set_control_latency_core(1);
	a.reset(None, None);
	b.reset(None, None);
	a.step(cmds[0]);
	for t in 0..cmds.len() - 1
	{
		b.step(cmds[t]);
		assert_eq!(a.quaternion(), b.quaternion(), "step {t}: not a one-step delay");
		a.step(cmds[t]);
	}
}

/// The inertia draw covers its range; the trainer's plant takes it (and the latency) per
/// episode while the teachers' airframe stays at the nominal inertia; reuse never compounds.
#[test]
fn trainer_plant_draws_per_episode_while_models_stay_nominal()
{
	let cfg = cfg_with(J, Some(LAT_MS));
	let af = AirframeRs::from_cfg(&cfg);
	assert_eq!(af.inertia, cfg.af_inertia, "the teachers' airframe is the nominal one");
	let mut sim = af.sim();
	assert_eq!(sim.plant_inertia(), cfg.af_inertia, "the sim is built at the nominal inertia");
	let j = cfg.plant_jitter();
	let (mut smin, mut smax, mut lats) = (f32::MAX, f32::MIN, std::collections::BTreeSet::new());
	for k in 0..4000u64
	{
		let ep_seed = crate::controller::disturbance_episode_seed(0xC789, k);
		crate::dagger_train::arm_cfg_disturbance(&mut sim, &cfg, ep_seed);
		let s = j.inertia_scale(ep_seed);
		let n = cfg.af_inertia;
		assert_eq!(sim.plant_inertia(), [n[0] * s, n[1] * s, n[2] * s], "episode {k}: plant not on its inertia draw");
		assert_eq!(sim.control_latency_steps(), j.latency_steps(ep_seed), "episode {k}: plant not on its latency draw");
		(smin, smax) = (smin.min(s), smax.max(s));
		lats.insert(sim.control_latency_steps());
	}
	assert!(smin >= 1.0 - J && smax <= 1.0 + J && smin < 1.0 - 0.99 * J && smax > 1.0 + 0.99 * J, "inertia [{smin}, {smax}]");
	assert_eq!(lats.into_iter().collect::<Vec<_>>(), (2..=12).collect::<Vec<u32>>(), "latency does not cover 2..=12 steps");
	// The design models after the training run: built from the same nominal airframe.
	assert_eq!(AirframeRs::from_cfg(&cfg).inertia, INERTIA);
	// Off: arming the weather leaves the plant nominal and undelayed.
	let off = l2b_cfg(4, Spelling::Legacy);
	let mut sim_off = AirframeRs::from_cfg(&off).sim();
	crate::dagger_train::arm_cfg_disturbance(&mut sim_off, &off, 1);
	assert_eq!((sim_off.plant_inertia(), sim_off.control_latency_steps()), (INERTIA, 0));
}

/// The jitter reaches the trainer (DAgger + per-round eval), the calibration sampler, the
/// WNN scorer and the rivals: everything moves off the pin.
#[test]
fn ctrl89_jitter_moves_trainer_calibration_scorers_and_rivals()
{
	let (_, d_train) = trained_on(&cfg_with(J, Some(LAT_MS)));
	assert_ne!(d_train, ABI64_CF21[0], "the trainer never saw the jitter");
	let mut cal = cfg_with(J, Some(LAT_MS));
	cal.teacher = 0;
	cal.steps_per_episode = 250;
	let mut c = ctbr_student(0, 10, 0x86CA);
	let s = crate::calib_sampler::sample_calibration_features_rs(&mut c, &cal, 3, 0x86CB);
	let mut h = Fnv::new();
	for col in &s
	{
		h.f64s(&col.iter().map(|&v| v as f64).collect::<Vec<_>>());
	}
	assert_ne!(h.0, ABI64_CF21[6], "the calibration sampler never saw the jitter");
	let s0 = stage2_with(real_life_plant(Spelling::Legacy));
	for (teacher, pin) in [(0u8, ABI64_CF21[4]), (4, ABI64_CF21[5])]
	{
		assert_ne!(r8(rival_j(teacher, &s0, J, Some(LAT_MS))), pin, "rival {teacher} never saw the jitter");
		assert_ne!(r8(rival_j(teacher, &s0, J, None)), pin, "rival {teacher}: inertia alone");
		assert_ne!(r8(rival_j(teacher, &s0, 0.0, Some(LAT_MS))), pin, "rival {teacher}: latency alone");
	}
}

/// Refusals at every host: no translation, no weather, a ring overflow.
#[test]
fn refusals()
{
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.train_inertia_jitter = J;
	c.dist_enabled = false;
	assert!(c.validate_plant_jitter().is_err(), "no weather");
	let mut c = l2b_cfg(4, Spelling::Legacy);
	c.train_latency_ms = Some(LAT_MS);
	c.translation = false;
	assert!(c.validate_plant_jitter().is_err(), "no translation");
	assert!(scorer_episode_plant(0.0, Some((0.0, 40.0)), DT, true, true, 1, 4, "t").is_err(), "beyond the ring");
	assert!(scorer_episode_plant(J, None, DT, false, true, 1, 4, "t").is_err(), "scorer without translation");
	assert!(scorer_episode_plant(J, None, DT, true, false, 1, 4, "t").is_err(), "scorer without weather");
}
