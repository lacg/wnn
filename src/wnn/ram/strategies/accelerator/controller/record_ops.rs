//! Reference-rollout RECORDERS, in Rust.
//!
//! Ports the last two per-STEP Python loops in the controller stack:
//!
//!   ga_memory.record_address_universe   -> record_address_universe
//!   arch_adaptation.record_input_entropy -> record_input_entropy
//!
//! Both drove a PID reference rollout from Python — 12 eps x 1500 steps and
//! 6 eps x 800 steps respectively — with a PyO3 `c.step()` per step, and then
//! either materialised a Python tuple per visited address per step, or ran a
//! Python loop over every sensor BIT per step (~10^6 interpreter operations).
//!
//! Initial conditions are INJECTED, not drawn here: `_sample_initial_state` runs
//! on numpy's PCG64 in Python and the (q0, omega0) pairs are passed in. That is
//! this codebase's established parity convention for episode ICs (see
//! dagger_train's pre-drawn ICs) and it is appropriate at this volume — a dozen
//! episodes, drawn once, not 10^9 per-cell draws. So these two ports are
//! BIT-EXACT: same ICs, same sim, same controller, same accumulation.

use crate::controller::{AttitudeSim, AttitudePidRs, WnnController};
use crate::optimal::AllocLqrRs;

/// Shannon entropy of a Bernoulli(p), in bits. Mirrors
/// arch_adaptation._binary_entropy including its 0/1 short-circuit.
#[inline]
fn binary_entropy(p: f64) -> f64 {
	if p <= 0.0 || p >= 1.0 {
		return 0.0;
	}
	-(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}

/// Drive one episode of a PID reference rollout, calling `on_step` after each
/// controller step. Returns the number of steps actually taken (an episode ends
/// early if the sim goes unstable).
///
/// The ORDER matters and mirrors the Python exactly: read IMU, read quaternion,
/// step the CONTROLLER (which caches its layer inputs/addresses), observe, then
/// advance the sim with the PID's action.
/// The reference driver that holds the sim in its operating region while the
/// controller runs forward at each visited state. PID for the quad path;
/// allocator-LQR on the TRUE rotor table for the overactuated path (which must
/// go through step_n, not the 4-motor step).
pub enum Driver<'a> {
	Pid(&'a mut AttitudePidRs),
	Alloc(&'a mut AllocLqrRs),
}

impl Driver<'_> {
	fn reset(&mut self) {
		if let Driver::Pid(p) = self {
			p.reset();
		}
	}
	/// Advance the sim one step with the driver's action.
	fn drive(&mut self, sim: &mut AttitudeSim, q: [f32; 4], gyro: [f32; 3], target: [f32; 3]) {
		match self {
			Driver::Pid(p) => sim.step(p.step_pub(q, gyro, target)),
			Driver::Alloc(a) => {
				let pwm: Vec<f32> = a.step_alloc_rs(q, gyro, target)
					.into_iter().map(|x| x as f32).collect();
				let _ = sim.step_n_core(&pwm);
			}
		}
	}
}

fn run_episode<F: FnMut(&WnnController)>(
	c: &mut WnnController, sim: &mut AttitudeSim, driver: &mut Driver<'_>,
	q0: [f32; 4], om0: [f32; 3], target: [f32; 3], steps: usize,
	mut on_step: F,
) -> usize {
	sim.reset(Some(q0), Some(om0));
	driver.reset();
	c.reset(0.0);
	let mut n = 0usize;
	for _ in 0..steps {
		if sim.is_unstable() {
			break;
		}
		let (gyro, accel) = sim.read_imu();
		let q = sim.quaternion();
		c.step(gyro, accel, target);
		on_step(c);
		n += 1;
		driver.drive(sim, q, gyro, target);
	}
	n
}

/// Cells the controller VISITS along reference rollouts, as sorted-unique
/// (neuron, address) pairs per layer.
///
/// Mirrors `record_address_universe`: the universe is what a MEMORY-phase genome
/// evolves over, so it must be deterministic — the returned vectors are sorted,
/// exactly as the Python `sorted(state_set)` was.
pub fn record_address_universe(
	c: &mut WnnController, sim: &mut AttitudeSim, driver: &mut Driver<'_>,
	init_q: &[[f32; 4]], init_om: &[[f32; 3]],
	target: [f32; 3], steps: usize,
) -> (Vec<(usize, u64)>, Vec<(usize, u64)>) {
	use std::collections::HashSet;
	let mut state_set: HashSet<(usize, u64)> = HashSet::new();
	let mut out_set: HashSet<(usize, u64)> = HashSet::new();

	for ep in 0..init_q.len().min(init_om.len()) {
		run_episode(c, sim, driver, init_q[ep], init_om[ep], target, steps, |c| {
			state_set.extend(c.last_state_addresses_pub());
			out_set.extend(c.last_output_addresses_pub());
		});
	}
	let mut s: Vec<(usize, u64)> = state_set.into_iter().collect();
	let mut o: Vec<(usize, u64)> = out_set.into_iter().collect();
	s.sort_unstable();
	o.sort_unstable();
	(s, o)
}

/// Per-input-bit activation ENTROPY over a PID reference rollout.
///
/// Mirrors `record_input_entropy`, including the empty-rollout contract: if no
/// step ran, return all-zero entropies rather than dividing by zero.
pub fn record_input_entropy(
	c: &mut WnnController, init_q: &[[f32; 4]], init_om: &[[f32; 3]],
	target: [f32; 3], steps: usize, sensor_window: usize, sensor_frame: usize,
) -> (Vec<f64>, Vec<f64>) {
	let mut sim = AttitudeSim::new(0.001, 0.075, 2.4, 0.05, [0.0023, 0.0023, 0.0046], 9.81);
	let mut pid = AttitudePidRs::new_default();
	let mut driver = Driver::Pid(&mut pid);
	let mut s_act = vec![0usize; sensor_window];
	let mut o_act = vec![0usize; sensor_frame];
	let mut nsteps = 0usize;

	for ep in 0..init_q.len().min(init_om.len()) {
		nsteps += run_episode(c, &mut sim, &mut driver, init_q[ep], init_om[ep], target, steps, |c| {
			let si = c.last_state_layer_input_ref();
			for i in 0..sensor_window.min(si.len()) {
				if si[i] { s_act[i] += 1; }
			}
			let oi = c.last_output_layer_input_ref();
			for i in 0..sensor_frame.min(oi.len()) {
				if oi[i] { o_act[i] += 1; }
			}
		});
	}
	if nsteps == 0 {
		return (vec![0.0; sensor_window], vec![0.0; sensor_frame]);
	}
	let n = nsteps as f64;
	(s_act.iter().map(|&a| binary_entropy(a as f64 / n)).collect(),
	 o_act.iter().map(|&a| binary_entropy(a as f64 / n)).collect())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn binary_entropy_endpoints_and_peak() {
		assert_eq!(binary_entropy(0.0), 0.0);
		assert_eq!(binary_entropy(1.0), 0.0);
		assert_eq!(binary_entropy(-0.5), 0.0, "out-of-range must clamp to 0");
		assert!((binary_entropy(0.5) - 1.0).abs() < 1e-12, "p=0.5 must be exactly 1 bit");
		// symmetric
		assert!((binary_entropy(0.25) - binary_entropy(0.75)).abs() < 1e-12);
	}
}
