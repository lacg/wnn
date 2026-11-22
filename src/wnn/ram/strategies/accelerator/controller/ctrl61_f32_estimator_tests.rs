//! ABI 61 (22/11/2026, Luiz plan "A + B"): the WNN's Mahony estimator and the ABI-55 heading
//! it feeds are f32 on the CPU, op for op the Metal kernel's — so the ESTIMATOR part of CPU
//! and GPU must agree BIT FOR BIT, every step: the quaternion, the "up" vector the attitude
//! features read, and the anchored heading.
//!
//! HOW. The Metal side is the rollout library exactly as the scorer compiles it (fast-math
//! options; the estimator functions carry their own strict-FP scope) running
//! controller_att_est_probe, which calls the SAME inline mahony_update / mahony_up /
//! mahony_heading the score kernel calls. The CPU side is the production path: AttEstimator
//! (seed, up, last_quat) + controller::mahony_heading. Streams: an AttitudeSim flown open
//! loop with a moving differential (the specific-force IMU the scorer reads), noisy random
//! IMU, accel dropouts (the zero-norm branch), and fast yaw spins that wrap the heading
//! through ±π many times.
//!
//! WHAT IS NOT COVERED (and stays a CTRL-54 difference): the roll / pitch / tilt features
//! are atan2 of the up vector through libm (CPU) vs Metal's atan2 — features, not the
//! estimator; and the yaw-anchor's initial heading (yaw_from_quat), a host-side seed.

use crate::att_estimator::{up_from_quat, AttEstimator};
use crate::controller::{mahony_heading, AttitudeSim};
use crate::mahony_f32::atan2_det;
use crate::metal_controller::ControllerRolloutEvaluator;
use metal::*;
use rand::{Rng, SeedableRng};

const DT: f32 = 0.001;
const KP: f32 = 2.0;
const KI: f32 = 0.1;
const STEPS: usize = 2000;

/// One stream: initial q0, initial heading, then per step (gyro, accel).
struct Stream
{
	q0: [f32; 4],
	h0: f32,
	imu: Vec<([f32; 3], [f32; 3])>,
}

fn sim_stream(q0: [f32; 4], w0: [f32; 3], pwm: [f32; 4]) -> Stream
{
	let mut sim = AttitudeSim::new(DT, 0.0707, 0.2, 0.0057, [1.66e-5, 1.66e-5, 2.93e-5], 9.81);
	sim.reset(Some(q0), Some(w0));
	let q0 = sim.quaternion();
	let mut imu = Vec::with_capacity(STEPS);
	for _ in 0..STEPS
	{
		imu.push(sim.read_imu());
		sim.step(pwm);
	}
	Stream { q0, h0: crate::controller::yaw_from_quat(q0), imu }
}

fn random_stream(seed: u64, yaw_rate: f32, dropout: f64) -> Stream
{
	let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
	let mut q0 = [1.0f32, rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3), rng.gen_range(-1.0..1.0)];
	let n = q0.iter().map(|v| v * v).sum::<f32>().sqrt();
	q0 = q0.map(|v| v / n);
	let mut imu = Vec::with_capacity(STEPS);
	for t in 0..STEPS
	{
		let g = [rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0), yaw_rate + rng.gen_range(-1.0..1.0)];
		let a = if rng.gen_bool(dropout)
		{
			[0.0, 0.0, 0.0]
		}
		else
		{
			[rng.gen_range(-4.0..4.0), rng.gen_range(-4.0..4.0), 9.81 + rng.gen_range(-3.0..3.0) + (t as f32) * 1e-4]
		};
		imu.push((g, a));
	}
	Stream { q0, h0: rng.gen_range(-3.1..3.1), imu }
}

fn streams() -> Vec<Stream>
{
	vec![
		sim_stream([0.996, 0.05, -0.05, 0.03], [0.3, -0.2, 0.1], [0.70, 0.69, 0.70, 0.69]),
		sim_stream([0.9, 0.3, 0.2, -0.25], [2.0, -1.5, 4.0], [0.72, 0.60, 0.75, 0.62]),
		random_stream(61, 0.0, 0.0),
		random_stream(62, 0.0, 0.05),
		random_stream(63, 12.0, 0.01), // ~3.8 turns: the heading wraps through ±π
		random_stream(64, -20.0, 0.0),
	]
}

/// CPU: the production path — AttEstimator.up() then mahony_heading on its last_quat.
fn cpu_run(s: &Stream) -> Vec<[f32; 8]>
{
	let mut e = AttEstimator::new(DT, KP, KI).unwrap();
	e.seed(s.q0);
	let mut h = s.h0;
	s.imu
		.iter()
		.map(|(g, a)| {
			let up = e.up(*g, *a);
			let q = e.last_quat().expect("live step");
			assert_eq!(up, up_from_quat(q));
			h = mahony_heading(h, q);
			[q[0], q[1], q[2], q[3], up[0], up[1], up[2], h]
		})
		.collect()
}

fn shared_buf<T>(dev: &Device, v: &[T]) -> Buffer
{
	dev.new_buffer_with_data(
		v.as_ptr() as *const _,
		(std::mem::size_of_val(v).max(4)) as u64,
		MTLResourceOptions::StorageModeShared,
	)
}

fn run_probe(ev: &ControllerRolloutEvaluator, name: &str, bufs: &[&Buffer], threads: u64)
{
	let (dev, queue, lib) = ev.test_parts();
	let f = lib.get_function(name, Some(FunctionConstantValues::new())).expect("probe kernel");
	let pso = dev.new_compute_pipeline_state_with_function(&f).expect("probe pipeline");
	let cmd = queue.new_command_buffer();
	let enc = cmd.new_compute_command_encoder();
	enc.set_compute_pipeline_state(&pso);
	for (i, b) in bufs.iter().enumerate()
	{
		enc.set_buffer(i as u64, Some(b), 0);
	}
	enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(threads.min(32), 1, 1));
	enc.end_encoding();
	cmd.commit();
	cmd.wait_until_completed();
	assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed, "{name} dispatch failed");
}

fn gpu_run(ev: &ControllerRolloutEvaluator, ss: &[Stream]) -> Vec<f32>
{
	let (dev, _, _) = ev.test_parts();
	let imu: Vec<f32> = ss.iter().flat_map(|s| s.imu.iter().flat_map(|(g, a)| g.iter().chain(a.iter()).copied())).collect();
	let seed: Vec<f32> = ss.iter().flat_map(|s| s.q0.iter().copied().chain([s.h0])).collect();
	let out = vec![0.0f32; ss.len() * STEPS * 8];
	let (b_imu, b_seed, b_out) = (shared_buf(dev, &imu), shared_buf(dev, &seed), shared_buf(dev, &out));
	let b_p = shared_buf(dev, &[ss.len() as u32, STEPS as u32]);
	let b_g = shared_buf(dev, &[KP, KI, DT, 0.0f32]);
	run_probe(ev, "controller_att_est_probe", &[&b_imu, &b_seed, &b_out, &b_p, &b_g], ss.len() as u64);
	unsafe { std::slice::from_raw_parts(b_out.contents() as *const f32, out.len()) }.to_vec()
}

const FIELDS: [&str; 8] = ["q.w", "q.x", "q.y", "q.z", "up.x", "up.y", "up.z", "heading"];

/// THE PIN: CPU == Metal, bit for bit, every step of every stream.
#[test]
fn estimator_cpu_gpu_bit_identical()
{
	let ev = ControllerRolloutEvaluator::new().expect("Metal device");
	let ss = streams();
	let gpu = gpu_run(&ev, &ss);
	let mut wraps = 0usize;
	for (si, s) in ss.iter().enumerate()
	{
		let cpu = cpu_run(s);
		for (t, row) in cpu.iter().enumerate()
		{
			let g = &gpu[(si * STEPS + t) * 8..(si * STEPS + t + 1) * 8];
			for k in 0..8
			{
				assert_eq!(
					row[k].to_bits(),
					g[k].to_bits(),
					"stream {si} step {t} {}: cpu {:e} gpu {:e}",
					FIELDS[k],
					row[k],
					g[k]
				);
			}
		}
		let last = cpu.last().unwrap()[7];
		wraps += (last.abs() / std::f32::consts::PI) as usize;
	}
	assert!(wraps >= 4, "the yaw-spin streams must carry the heading past ±π (got {wraps})");
}

/// atan2_det CPU == Metal on a random + edge-case sweep (axes, diagonals, the tan(π/8)
/// reduction boundary, tiny and huge magnitudes, signed zeros).
#[test]
fn atan2_det_cpu_gpu_bit_identical()
{
	let ev = ControllerRolloutEvaluator::new().expect("Metal device");
	let mut rng = rand::rngs::SmallRng::seed_from_u64(0x61);
	let mut yx: Vec<[f32; 2]> = (0..200_000)
		.map(|_| {
			let e = rng.gen_range(-20..20) as f32;
			[rng.gen_range(-1.0f32..1.0) * e.exp2(), rng.gen_range(-1.0f32..1.0) * e.exp2()]
		})
		.collect();
	let t8 = 0.41421357f32;
	for (y, x) in [
		(0.0, 1.0),
		(1.0, 0.0),
		(-1.0, 0.0),
		(0.0, -1.0),
		(-0.0, -1.0),
		(0.0, 0.0),
		(1.0, 1.0),
		(-1.0, -1.0),
		(t8, 1.0),
		(f32::from_bits(t8.to_bits() + 1), 1.0),
		(f32::from_bits(t8.to_bits() - 1), 1.0),
		(1e-30, 1.0),
		(1e30, -1.0),
	]
	{
		yx.push([y, x]);
	}
	let (dev, _, _) = ev.test_parts();
	let out = vec![0.0f32; yx.len()];
	let (b_in, b_out, b_n) = (shared_buf(dev, &yx), shared_buf(dev, &out), shared_buf(dev, &[yx.len() as u32]));
	run_probe(&ev, "controller_atan2_det_probe", &[&b_in, &b_out, &b_n], yx.len() as u64);
	let got = unsafe { std::slice::from_raw_parts(b_out.contents() as *const f32, yx.len()) };
	for (i, p) in yx.iter().enumerate()
	{
		let want = atan2_det(p[0], p[1]);
		assert_eq!(want.to_bits(), got[i].to_bits(), "atan2_det({:e}, {:e}): cpu {want:e} gpu {:e}", p[0], p[1], got[i]);
	}
}
