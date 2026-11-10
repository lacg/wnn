//! CPU scorer vs Metal kernel on the rf_c200_dres recipe, ACROSS the score
//! chunk boundary (09/11/2026, docs/ctrl_steady_checks.md "Bugs found").
//!
//! The bug: `ControllerRolloutEvaluator::score` dispatches episodes in chunks of
//! `episodes_per_chunk()` (25) and hands the kernel q0/omega0 SLICED per chunk,
//! but binds the per-episode stage-1/2 blob (buffer 30: z0, vz0, mass,
//! collective, x0, y0) WHOLE. The kernel indexed that blob by the chunk-local
//! episode, so every episode >= 25 flew episode (e mod 25)'s translational draw.
//! Every parity test before this one scored <= 4 episodes — one chunk — so the
//! mis-index was invisible to all of them; the s1 dres winner showed it as
//! CPU 1.322/0.924/0.373 vs GPU 1.315/0.973/0.357 (err°/steady°/pos m).
//!
//! The fixture is the recipe's runtime surface: the 25-feature cascade student
//! (per-axis p+i, yaw pair, vertical + horizontal channels, the d̂ residual
//! block), Mahony estimator features, delta law, action_repeat 10, motor lag
//! 0.20 s, specific-force IMU + rotor drag 0.78, tilt reference, L4C weather,
//! trained by the DAgger trainer. Receptive fields / rf-cross / calib-episodes
//! are host-side (connection indices and thresholds only — receptive_fields.rs
//! has no runtime twin), so they need no kernel coverage beyond the indices.

use crate::abi38_pins::{ARM, DT, G, INERTIA, KD, KT, MASS};
use crate::abi38_tests::l4c_cfg;
use crate::att_err_ref::AttErrRef;
use crate::controller::{Disturbance, WnnController};
use crate::ctrl38_tests::obs_student3;
use crate::dagger_train::dagger_train_inplace_with;
use crate::metal_controller::{episodes_per_chunk, ControllerRolloutEvaluator};
use crate::plant_opts::PlantOpts;
use crate::stage1::Stage1Cfg;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// > 2 chunks at the default chunk size, with a partial last chunk.
const EPS: usize = 60;
const STEPS: usize = 600;
const LAG: f32 = 0.20;
const ASYM: [f32; 4] = [1.12, 0.9, 1.05, 0.95];

/// Per-episode attitude ICs (the scorer's ±tilt draw shape).
fn ics() -> (Vec<f32>, Vec<f32>)
{
	let mut rng = SmallRng::seed_from_u64(0xC4_0C);
	let (mut q0, mut w0) = (Vec::new(), Vec::new());
	for _ in 0..EPS
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

/// Translation ON with a DISTINCT draw per episode in every slot of the blob —
/// a repeated draw is exactly what the chunk mis-index produced, so the fixture
/// must make every episode's z/vz/mass/collective/x/y its own.
fn stage2_full() -> Stage1Cfg
{
	let mut rng = SmallRng::seed_from_u64(0x5C_0E);
	let mut u = |lo: f32, hi: f32| -> Vec<f32> { (0..EPS).map(|_| rng.gen_range(lo..hi)).collect() };
	let c = Stage1Cfg {
		target_altitude: 0.0,
		lambda_alt: 1.0,
		init_z: u(-0.15, 0.15),
		init_vz: u(-0.1, 0.1),
		mass: u(MASS * 0.9, MASS * 1.1),
		collective_frac: u(-0.04, 0.04),
		lambda_pos: 2.0,
		init_x: u(-0.5, 0.5),
		init_y: u(-0.5, 0.5),
		plant: PlantOpts::new(true, 0.78).unwrap(),
		att_ref: AttErrRef::tilt(G as f64, 1.0, 1.0, 0.5236).unwrap(),
	};
	c.validate(EPS).expect("chunk fixture must validate");
	c
}

fn dist() -> Disturbance
{
	Disturbance {
		tau_bias: [0.0; 3],
		gust_sigma: 0.0,
		gust_tau_c: 0.1,
		motor_asym: ASYM,
		gyro_sigma: 0.01073277832,
		gyro_bias_walk: 3.8785e-05,
		accel_sigma: 0.1264912,
		dropout_prob: 0.0,
		dropout_len_steps: 0,
		obs_delay_steps: 0,
		torque_scale_jitter: 0.2,
		seed: 0xD5E5,
	}
}

fn cpu_rows(c: &WnnController, s1: &Stage1Cfg) -> [f64; 17]
{
	let (q0, w0) = ics();
	let d = dist();
	let mut cc = c.clone();
	crate::cpu_score::rollout_one(
		&mut cc, &q0, &w0, EPS, STEPS, DT, ARM, KT, KD, INERTIA, G, LAG, [0.0; 3],
		true, d.tau_bias, d.gust_sigma, d.gust_tau_c, d.motor_asym, d.gyro_sigma, d.gyro_bias_walk,
		d.accel_sigma, d.seed, 0.0, 0, 0, d.torque_scale_jitter, c.levels_per_motor(), 4,
		None, None, None, 1.0, 0.4, None, Some(s1),
	)
}

fn gpu_rows(c: &WnnController, s1: &Stage1Cfg) -> Option<Vec<f64>>
{
	if std::env::var("WNN_SKIP_GPU_TESTS").is_ok()
	{
		return None;
	}
	let ev = ControllerRolloutEvaluator::new().ok()?;
	let (q0, w0) = ics();
	let rows = ev
		.score(&[c], &q0, &w0, EPS, STEPS, (DT, ARM, KT, INERTIA, G), KD, LAG, [0.0; 3], Some(dist()), None, None, None, Some(s1))
		.expect("gpu score");
	Some(rows[0].clone())
}

/// The recipe's student, trained on the recipe's plant (estimator before training,
/// so train = deploy).
pub(crate) fn dres_student() -> WnnController
{
	let mut c = obs_student3(0, 10, 0xC5C0, false, false, true);
	c.enable_att_estimator(2.0, 0.1).expect("estimator on");
	let mut cfg = l4c_cfg(LAG, 4);
	cfg.imu_specific_force = true;
	cfg.rotor_drag = 0.78;
	let st = dagger_train_inplace_with(&mut c, &cfg, [0.0; 3], 49, false);
	assert!(st.iter_cells_written.iter().sum::<usize>() > 0, "trainer wrote nothing");
	c
}

#[test]
fn dres_recipe_cpu_scorer_matches_gpu_across_chunks()
{
	assert!(EPS > 2 * episodes_per_chunk(), "fixture must span > 2 score chunks (chunk={})", episodes_per_chunk());
	let c = dres_student();
	let s1 = stage2_full();
	let cpu = cpu_rows(&c, &s1);
	let Some(gpu) = gpu_rows(&c, &s1)
	else
	{
		return;
	};
	for (i, name) in [
		(1usize, "err"),
		(0, "reward"),
		(2, "stable"),
		(5, "steady"),
		(13, "pos"),
		(14, "alt"),
		(15, "dx"),
		(16, "dy"),
	]
	{
		eprintln!("[chunk] {name:6}: cpu {:.7} gpu {:.7}", cpu[i], gpu[i]);
		let tol = 2e-3 * cpu[i].abs().max(1e-3);
		assert!((gpu[i] - cpu[i]).abs() <= tol, "dres recipe {name}: gpu {} cpu {} (tol {tol})", gpu[i], cpu[i]);
	}
}

/// The per-episode translational draw must reach episodes past the first chunk:
/// replacing ONLY the last chunk's draws must move the GPU row (it did not move
/// before the fix — those episodes read chunk 0's slots).
#[test]
fn gpu_reads_per_episode_draws_past_the_first_chunk()
{
	let c = dres_student();
	let s1 = stage2_full();
	let Some(base) = gpu_rows(&c, &s1)
	else
	{
		return;
	};
	let mut moved = stage2_full();
	let k = episodes_per_chunk();
	for e in 2 * k..EPS
	{
		moved.init_x[e] = -moved.init_x[e];
		moved.init_y[e] = 0.5 * moved.init_y[e];
		moved.mass[e] *= 1.05;
	}
	let alt = gpu_rows(&c, &moved).expect("gpu");
	assert!((alt[13] - base[13]).abs() > 1e-6, "last-chunk draws did not reach the kernel: pos {} vs {}", alt[13], base[13]);
	// And the CPU agrees on the moved draw too.
	let cpu = cpu_rows(&c, &moved);
	let tol = 2e-3 * cpu[13].abs().max(1e-3);
	assert!((alt[13] - cpu[13]).abs() <= tol, "moved draw pos: gpu {} cpu {}", alt[13], cpu[13]);
}
