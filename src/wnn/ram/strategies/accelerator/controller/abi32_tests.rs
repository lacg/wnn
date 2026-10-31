//! ABI 32 behaviour + CPU/Metal parity tests (31/10/2026): the specific-force
//! IMU, linear rotor drag, the G5 tilt-reference attitude error, the dx/dy
//! position metrics and the WNN's Mahony attitude estimator.
//!
//! Every parity test has two halves, the house discipline: (a) the CPU rollout
//! (cpu_score::rollout_one) and the Metal kernel agree with the option ON, and
//! (b) the option MOVES the result — otherwise (a) could pass with the option
//! silently dead on both sides.

use crate::att_err_ref::AttErrRef;
use crate::controller::{AttitudeSim, ReplayObs, WnnController};
use crate::metal_controller::ControllerRolloutEvaluator;
use crate::plant_opts::PlantOpts;
use crate::stage1::Stage1Cfg;
use rand::{rngs::SmallRng, Rng, SeedableRng};

const DT: f32 = 0.001;
// cf21-like plant (the stage-2 airframe scale): arm, k_thrust, k_drag, inertia.
const ARM: f32 = 0.0707;
const KT: f32 = 0.2;
const KD: f32 = 0.0057;
const INERTIA: [f32; 3] = [3.0e-5, 3.0e-5, 5.3e-5];
const G: f32 = 9.81;
const MASS: f32 = 0.0393;
const EPS: usize = 4;
const STEPS: usize = 400;

/// sn=0 stage-2 controller with the per-axis attitude features (roll/pitch err,
/// yaw dropped) so the estimator has something to feed: 9 base + 2 + 3 vertical
/// + 4 horizontal = 18 features.
fn controller(seed: u64, estimator: bool) -> WnnController
{
	controller_with(seed, estimator, 6)
}

/// `cells` random output cells per neuron: 6 = a mostly-hovering policy (stable
/// over STEPS, so CPU/GPU float differences stay small); 256 = every address
/// written, so the policy reacts to EVERY connected bit (used on a short
/// horizon, before the chaotic random policy amplifies f32 differences).
fn controller_with(seed: u64, estimator: bool, cells: usize) -> WnnController
{
	let (num_motors, levels, bpf, window, obpn) = (4usize, 4usize, 3usize, 2usize, 8usize);
	let num_features = 18usize;
	let frame_bits = num_features * bpf;
	let mut rng = SmallRng::seed_from_u64(seed);
	let thresholds: Vec<f32> = (0..frame_bits)
		.map(|i| match i / bpf
		{
			9 | 10 => rng.gen_range(-0.3f32..0.3),
			f if f >= 11 => rng.gen_range(-0.6f32..0.6),
			_ => rng.gen_range(-5.0f32..5.0),
		})
		.collect();
	let num_out = num_motors * levels;
	let output_connections: Vec<i64> = (0..num_out * obpn)
		.map(|_| rng.gen_range(0..frame_bits) as i64)
		.collect();
	let mut c = WnnController::new_core(
		num_motors, levels, bpf, window, 0, 0, obpn, thresholds, Vec::new(), output_connections,
		true, 0.05, 0.98, 1.0, // delta control (the production student)
		false, false, true, false, false, false, false, false, // obs_peraxis_p, yaw dropped
		0.99, 1.0, DT, false, 1, ram_core::neuron_memory::BINARY, None, None, 0.05, false, 0.30,
		true, true, true, // stage-1 vertical
		true, true,       // stage-2 horizontal
		false, 1,
	)
	.expect("abi32 controller must construct");
	for n in 0..num_out
	{
		for k in 0..cells
		{
			let addr = if cells >= (1 << obpn) { k as u64 } else { rng.gen_range(0..(1u64 << obpn)) };
			c.write_output_cell_internal(n, addr, if rng.gen_bool(0.5) { 1 } else { 0 });
		}
	}
	if estimator
	{
		c.enable_att_estimator(2.0, 0.1).expect("estimator on");
	}
	c
}

fn ics() -> (Vec<f32>, Vec<f32>)
{
	let mut rng = SmallRng::seed_from_u64(91);
	let (mut q0, mut w0) = (Vec::new(), Vec::new());
	for _ in 0..EPS
	{
		let ang: f32 = rng.gen_range(-0.12..0.12);
		let (hx, hy): (f32, f32) = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
		let n = (hx * hx + hy * hy).sqrt().max(1e-6);
		let h = ang / 2.0;
		q0.extend_from_slice(&[h.cos(), h.sin() * hx / n, h.sin() * hy / n, 0.0]);
		w0.extend_from_slice(&[rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), 0.0]);
	}
	(q0, w0)
}

fn cfg(plant: PlantOpts, att_ref: AttErrRef) -> Stage1Cfg
{
	let c = Stage1Cfg {
		target_altitude: 0.0,
		lambda_alt: 1.0,
		init_z: (0..EPS).map(|e| -0.1 + 0.05 * e as f32).collect(),
		init_vz: vec![0.0; EPS],
		mass: vec![MASS; EPS],
		collective_frac: vec![0.0; EPS],
		lambda_pos: 2.0,
		init_x: (0..EPS).map(|e| 0.4 - 0.25 * e as f32).collect(),
		init_y: (0..EPS).map(|e| -0.45 + 0.3 * e as f32).collect(),
		plant,
		att_ref,
	};
	c.validate(EPS).expect("abi32 fixture must validate");
	c
}

fn tilt_ref() -> AttErrRef
{
	AttErrRef::tilt(G as f64, 1.0, 1.0, 0.5236).unwrap()
}

fn cpu(c: &WnnController, s1: &Stage1Cfg, steps: usize) -> [f64; 17]
{
	let (q0, w0) = ics();
	let mut c = c.clone();
	crate::cpu_score::rollout_one(
		&mut c, &q0, &w0, EPS, steps, DT, ARM, KT, KD, INERTIA, G, 0.0, [0.0; 3], false, [0.0; 3],
		0.0, 0.1, [1.0; 4], 0.0, 0.0, 0.0, 0, 0.0, 0, 0, 0.0, 4, 4, None, None, None, 1.0, 0.4, None,
		Some(s1),
	)
}

fn gpu(c: &WnnController, s1: &Stage1Cfg, steps: usize) -> Option<Vec<f64>>
{
	let (q0, w0) = ics();
	let ev = ControllerRolloutEvaluator::new().ok()?;
	let rows = ev
		.score(&[c], &q0, &w0, EPS, steps, (DT, ARM, KT, INERTIA, G), KD, 0.0, [0.0; 3], None, None, None, None, Some(s1))
		.expect("gpu score");
	Some(rows[0].clone())
}

fn close(gpu: f64, cpu: f64, rel: f64, floor: f64, what: &str)
{
	let tol = floor.max(cpu.abs() * rel);
	assert!((gpu - cpu).abs() <= tol, "{what}: gpu={gpu} cpu={cpu} tol={tol}");
}

/// CPU == GPU on err, reward, pos_err, alt, dx, dy (rows 1, 0, 13, 14, 15, 16).
fn assert_parity(c: &WnnController, s1: &Stage1Cfg, what: &str) -> [f64; 17]
{
	assert_parity_steps(c, s1, STEPS, what)
}

fn assert_parity_steps(c: &WnnController, s1: &Stage1Cfg, steps: usize, what: &str) -> [f64; 17]
{
	let cr = cpu(c, s1, steps);
	if let Some(gr) = gpu(c, s1, steps)
	{
		assert_eq!(gr.len(), 17, "{what}: GPU row must carry dx/dy (17 metrics)");
		close(gr[1], cr[1], 2e-2, 1e-4, &format!("{what} err"));
		close(gr[0], cr[0], 5e-2, 1e-3, &format!("{what} reward"));
		for (i, name) in [(13, "pos"), (14, "alt"), (15, "dx"), (16, "dy")]
		{
			close(gr[i], cr[i], 5e-2, 1e-3, &format!("{what} {name}"));
		}
	}
	cr
}

fn differs(a: &[f64; 17], b: &[f64; 17], idx: &[usize], what: &str)
{
	assert!(
		idx.iter().any(|&i| (a[i] - b[i]).abs() > 1e-9),
		"{what}: the option did not move the result — dead plumbing"
	);
}

/// dx/dy are the horizontal components of pos: each positive with displaced
/// starts, and dx² + dy² + alt² bounds the per-step 3-D mean from above-ish —
/// here the simple sanity: dx, dy < pos and > 0.
#[test]
fn dx_dy_are_the_horizontal_components()
{
	let c = controller(7, false);
	let r = assert_parity(&c, &cfg(PlantOpts::OFF, AttErrRef::LEVEL), "dxdy");
	assert!(r[15] > 0.0 && r[16] > 0.0, "dx/dy zero with displaced starts — not wired");
	assert!(r[15] < r[13] && r[16] < r[13], "dx/dy must each be below the 3-D error");
}

#[test]
fn parity_imu_specific_force()
{
	let c = controller(11, false);
	let off = assert_parity(&c, &cfg(PlantOpts::OFF, AttErrRef::LEVEL), "isf off");
	let on = assert_parity(&c, &cfg(PlantOpts::new(true, 0.0).unwrap(), AttErrRef::LEVEL), "isf on");
	differs(&off, &on, &[0, 1, 13], "imu_specific_force");
}

#[test]
fn parity_rotor_drag()
{
	let c = controller(13, false);
	let off = assert_parity(&c, &cfg(PlantOpts::OFF, AttErrRef::LEVEL), "drag off");
	let on = assert_parity(&c, &cfg(PlantOpts::new(false, 0.35).unwrap(), AttErrRef::LEVEL), "drag on");
	differs(&off, &on, &[13, 15, 16], "rotor_drag");
}

#[test]
fn parity_att_err_tilt_ref()
{
	let c = controller(17, false);
	let lvl = assert_parity(&c, &cfg(PlantOpts::OFF, AttErrRef::LEVEL), "ref level");
	let tilt = assert_parity(&c, &cfg(PlantOpts::OFF, tilt_ref()), "ref tilt");
	differs(&lvl, &tilt, &[1, 0], "att_err_tilt_ref");
	// Same trajectory (the reference only re-scores it): position is untouched.
	assert_eq!(lvl[13].to_bits(), tilt[13].to_bits(), "G5 must not change the flight");
}

#[test]
fn parity_att_estimator_full_abi32_plant()
{
	// Dense policy (reacts to every bit, incl. roll/pitch) on a short horizon.
	let plant = PlantOpts::new(true, 0.35).unwrap();
	let raw = assert_parity_steps(&controller_with(19, false, 256), &cfg(plant, tilt_ref()), 60, "accel-only");
	let est = assert_parity_steps(&controller_with(19, true, 256), &cfg(plant, tilt_ref()), 60, "estimator");
	differs(&raw, &est, &[0, 1, 13], "obs_att_estimator");
	// And the long-horizon stable policy with the estimator on, full plant.
	assert_parity(&controller(19, true), &cfg(plant, tilt_ref()), "estimator 400");
}

/// Under specific force the accel-only roll/pitch features are near-blind (the
/// accelerometer reads ≈ thrust axis); the estimator recovers the tilt. Fly one
/// tilted, translating episode and compare each feature to the TRUE roll.
#[test]
fn estimator_sees_tilt_the_specific_force_accelerometer_cannot()
{
	let mut sim = AttitudeSim::new(DT, ARM, KT, KD, INERTIA, G);
	PlantOpts::new(true, 0.0).unwrap().apply(&mut sim);
	let roll = 0.2f32;
	let q0 = [(roll * 0.5).cos(), (roll * 0.5).sin(), 0.0, 0.0];
	sim.reset(Some(q0), Some([0.0; 3]));
	sim.set_translation_core(MASS).unwrap();
	let mut est = controller(23, true);
	let mut raw = controller(23, false);
	est.reset(0.0);
	est.seed_att_estimator(q0);
	raw.reset(0.0);
	let hover = (MASS * G / (4.0 * KT)).sqrt();
	let (mut e_est, mut e_raw) = (0.0f32, 0.0f32);
	// 100 ms: while the vehicle ACCELERATES the Mahony accel correction drags
	// the estimate toward the thrust axis at ~kp (2 rad/s) — the real-world
	// behaviour; in unaccelerated flight specific force = −g_body again.
	for _ in 0..100
	{
		let (gy, ac) = sim.read_imu();
		let fe = est.compute_features(gy, ac, [0.0; 3]);
		let fr = raw.compute_features(gy, ac, [0.0; 3]);
		// roll_err = target(0) − roll_est ⇒ roll_est = −feature 9.
		let true_roll = sim.attitude_error(None); // pure-roll episode: |angle| = roll
		e_est = e_est.max((-fe[9] - true_roll).abs());
		e_raw = e_raw.max((-fr[9] - true_roll).abs());
		sim.step([hover; 4]);
	}
	assert!(e_est < 0.06, "estimator roll error {e_est} rad — it must track the true tilt");
	assert!(e_raw > 0.15, "accel-only roll error {e_raw} rad — specific force should blind it");
}

/// Replay parity: a live rollout records att_up_obs() per step; a fresh
/// controller replaying with ReplayObs.att_up produces the SAME features
/// bit-for-bit, and the guard refuses a replay without the stream.
#[test]
fn att_estimator_replay_is_bit_identical_and_guarded()
{
	let mut sim = AttitudeSim::new(DT, ARM, KT, KD, INERTIA, G);
	PlantOpts::new(true, 0.35).unwrap().apply(&mut sim);
	let q0 = [0.995f32, 0.07, -0.05, 0.02];
	sim.reset(Some(q0), Some([0.2, -0.1, 0.0]));
	sim.set_translation_core(MASS).unwrap();
	sim.set_horizontal_state(0.3, -0.2, 0.0, 0.0);
	let mut live = controller(29, true);
	live.reset(0.0);
	live.seed_att_estimator(q0);
	let n = 150usize;
	let (mut g, mut a, mut up, mut feats) = (vec![], vec![], vec![], vec![]);
	for _ in 0..n
	{
		let (gy, ac) = sim.read_imu();
		feats.push(live.compute_features(gy, ac, [0.0; 3]));
		up.push(live.att_up_obs());
		g.push(gy);
		a.push(ac);
		sim.step([0.5, 0.52, 0.5, 0.48]);
	}
	let ro = ReplayObs { vert: vec![[0.0; 3]; n], horiz: vec![[0.0; 4]; n], pwm_dev: vec![], att_up: up };
	let mut rep = controller(29, true);
	rep.reset(0.0);
	rep.replay_obs_guard(Some(&ro), n, "test");
	for t in 0..n
	{
		rep.apply_replay_obs(Some(&ro), t);
		let f = rep.compute_features(g[t], a[t], [0.0; 3]);
		for i in 9..11
		{
			assert_eq!(f[i].to_bits(), feats[t][i].to_bits(), "replay diverged at t={t} f={i}");
		}
	}
	let short = ReplayObs { vert: vec![[0.0; 3]; n], horiz: vec![[0.0; 4]; n], pwm_dev: vec![], att_up: vec![] };
	let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		controller(29, true).replay_obs_guard(Some(&short), n, "test")
	}));
	assert!(r.is_err(), "the guard must refuse an estimator replay without att_up");
}

/// The estimator refuses to be switched on without an attitude feature to feed.
#[test]
fn estimator_without_attitude_features_is_refused()
{
	let mut c = WnnController::new_core(
		4, 4, 3, 2, 0, 0, 8, vec![0.0; 27], Vec::new(), vec![0; 4 * 4 * 8], false, 0.15, 0.98, 1.0,
		false, false, false, false, false, false, false, false, 0.99, 1.0, DT, false, 1,
		ram_core::neuron_memory::BINARY, None, None, 0.05, false, 0.30, false, false, false, false,
		false, false, 1,
	)
	.unwrap();
	assert!(c.enable_att_estimator(2.0, 0.1).is_err());
}

// ---- trainer side (RewardGatedConfigPacked → AirframeRs / episode_regime) ----

fn rollout_reward(
	cfg: &crate::dagger_train::RewardGatedConfigPacked,
	est: bool,
) -> (f64, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)
{
	use crate::dagger_train::{rollout_and_label_rs, AirframeRs};
	let mut c = crate::ctrl17_pins::pin_student(true, true);
	if est
	{
		c = controller(31, true);
	}
	let af = AirframeRs::from_cfg(cfg);
	let mut teacher = af.teacher(cfg.teacher);
	let mut sim = af.sim();
	let mut rng = SmallRng::seed_from_u64(5);
	let t = rollout_and_label_rs(&mut c, &mut teacher, &mut sim, cfg, 0.1, &mut rng, [0.0; 3]);
	(t.cumulative_reward, t.att_up, t.horiz_obs, t.accels)
}

/// The trainer flies the ABI-32 plant (AirframeRs::from_cfg arms it), the gate
/// reward re-scores under G5, and the rollout records the estimator stream.
#[test]
fn trainer_reads_the_abi32_fields()
{
	let base = crate::ctrl17_pins::pin_translation_cfg(0.5);
	let (r0, up0, h0, a0) = rollout_reward(&base, false);
	assert!(up0.iter().all(|u| *u == [0.0; 3]), "estimator off ⇒ zero att_up stream");
	let mut plant = crate::ctrl17_pins::pin_translation_cfg(0.5);
	plant.imu_specific_force = true;
	plant.rotor_drag = 0.35;
	let (_, _, h1, a1) = rollout_reward(&plant, false);
	// The pin student is a constant-hover policy, so the attitude reward cannot
	// move — the PLANT does: drag moves the horizontal state, specific force the
	// recorded accelerometer.
	assert_ne!(h0, h1, "the trainer ignored rotor_drag (horizontal obs unchanged)");
	assert_ne!(a0, a1, "the trainer ignored imu_specific_force (accel unchanged)");
	let mut g5 = crate::ctrl17_pins::pin_translation_cfg(0.5);
	g5.att_err_tilt_ref = true;
	let (r2, ..) = rollout_reward(&g5, false);
	assert_ne!(r0.to_bits(), r2.to_bits(), "the gate reward ignored att_err_tilt_ref");
	let (_, up3, ..) = rollout_reward(&plant, true);
	assert!(up3.iter().any(|u| *u != [0.0; 3]), "estimator on ⇒ att_up recorded");
}

/// validate_abi32: the options are refused where they would be silently inert.
#[test]
fn trainer_config_refuses_inert_abi32_options()
{
	let mut c = crate::ctrl17_pins::pin_attitude_cfg();
	c.imu_specific_force = true;
	assert!(c.validate_abi32().is_err(), "specific force without translation");
	let mut c = crate::ctrl17_pins::pin_translation_cfg(0.0);
	c.att_err_tilt_ref = true;
	assert!(c.validate_abi32().is_err(), "tilt reference without the horizontal channel");
	let mut c = crate::ctrl17_pins::pin_translation_cfg(0.5);
	c.att_err_tilt_ref = true;
	c.rotor_drag = 0.3;
	c.imu_specific_force = true;
	assert!(c.validate_abi32().is_ok());
}
