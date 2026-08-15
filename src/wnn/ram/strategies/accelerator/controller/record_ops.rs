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
use crate::stage1::Stage1Cfg;

/// SCOPE C STAGE 1: what the recorder needs to fly the VERTICAL channel.
///
/// WHY THE RECORDER NEEDS THIS AT ALL (13/08/2026). The MEMORY phase evolves
/// cells over the address universe this module records. With translation off,
/// obs_collective_cmd sits at the anchor and obs_alt_err/obs_vz sit at zero for
/// every step — so the recorded universe covers ONE degenerate slice of the
/// vertical feature space, and every address the controller actually visits
/// once altitude moves is MISSING from it. The memory GA then cannot reach the
/// cells that decide stage-1 behaviour. Same failure shape as the degenerate
/// thermometer ladder (ed921bac) and the L2 GPU plant omission: a rollout that
/// silently flies a different aircraft than the one being optimised.
///
/// `gravity`/`k_thrust` come from the airframe and are only used to derive the
/// per-episode hover collective (Stage1Cfg::collective_pwm), exactly as
/// cpu_score does — one implementation of that formula, not two.
pub struct RecorderStage1<'a> {
	pub cfg: &'a Stage1Cfg,
	pub gravity: f32,
	pub k_thrust: f32,
}

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
	ep: usize, s1: Option<&RecorderStage1<'_>>,
	mut on_step: F,
) -> usize {
	sim.reset(Some(q0), Some(om0));
	driver.reset();
	c.reset(0.0);
	// STAGE 1: per-episode PLANT draw + vertical ICs, in cpu_score::score_one's
	// EXACT order — set_translation AFTER reset (reset zeroes z/vz), then the
	// anchor AFTER c.reset (which seeds the accumulators from the current
	// anchor). None ⇒ none of this runs ⇒ bit-identical to attitude-only.
	if let Some(s) = s1 {
		sim.set_translation_core(s.cfg.mass[ep]).expect("validated stage1 mass");
		sim.set_vertical_state(s.cfg.init_z[ep], s.cfg.init_vz[ep]);
		c.set_collective_anchor(s.cfg.collective_pwm(ep, s.gravity, s.k_thrust));
	}
	let mut n = 0usize;
	for _ in 0..steps {
		if sim.is_unstable() {
			break;
		}
		let (gyro, accel) = sim.read_imu();
		// STAGE 1: same start-of-step snapshot the IMU is read at, matching
		// cpu_score::score_one so the recorded addresses are the ones scoring
		// will actually visit.
		if let Some(s) = s1 {
			c.set_vertical_obs(s.cfg.collective_pwm(ep, s.gravity, s.k_thrust),
			                   s.cfg.target_altitude - sim.altitude_rs(),
			                   sim.vertical_velocity_rs());
		}
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
	target: [f32; 3], steps: usize, s1: Option<&RecorderStage1<'_>>,
) -> (Vec<(usize, u64)>, Vec<(usize, u64)>) {
	use std::collections::HashSet;
	let mut state_set: HashSet<(usize, u64)> = HashSet::new();
	let mut out_set: HashSet<(usize, u64)> = HashSet::new();

	for ep in 0..init_q.len().min(init_om.len()) {
		run_episode(c, sim, driver, init_q[ep], init_om[ep], target, steps, ep, s1, |c| {
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
		// Entropy profiling is attitude-only by construction (it ranks SENSOR bits
		// on the legacy plant); no stage-1 channel ⇒ None keeps it bit-identical.
		nsteps += run_episode(c, &mut sim, &mut driver, init_q[ep], init_om[ep], target, steps, ep, None, |c| {
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

	/// SCOPE C STAGE 1 regression (13/08/2026). The MEMORY phase evolves cells
	/// over the universe this module records. Before the fix the recorder flew a
	/// NON-TRANSLATING plant, so obs_collective_cmd sat at the anchor and
	/// obs_alt_err/obs_vz sat at zero for every step of every episode: the
	/// universe covered ONE degenerate slice, and every address reached once
	/// altitude moved was missing from it.
	///
	/// The assertion is the property that matters, not a golden count: turning
	/// the vertical channel on must CHANGE which addresses are visited. If these
	/// two universes were ever equal again, the channel would be dead and the
	/// memory stage silently blind to it.
	#[test]
	fn stage1_vertical_channel_changes_the_recorded_universe() {
		let (mass, g, k) = (0.0393f32, 9.81f32, 0.2f32);
		let eps = 3usize;
		let cfg = Stage1Cfg {
			target_altitude: 0.0, lambda_alt: 0.0,
			// Start each episode OFF the target with vertical motion, which is
			// exactly what a degenerate recorder never sees.
			init_z: vec![0.5, -0.5, 0.0],
			init_vz: vec![0.0, 0.0, 0.4],
			mass: vec![mass; eps],
			collective_frac: vec![0.0, 0.1, -0.1],
			lambda_pos: 0.0, init_x: vec![], init_y: vec![],
		};
		cfg.validate(eps).expect("fixture must be well-formed");

		let init_q: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 0.0]; eps];
		let init_om: Vec<[f32; 3]> = vec![[0.05, -0.05, 0.0]; eps];

		let record = |s1: Option<&RecorderStage1<'_>>| {
			let mut c = stage1_recorder_controller();
			let mut sim = AttitudeSim::new(0.001, 0.0707, k, 0.0057,
				[1.66e-5, 1.66e-5, 2.93e-5], g);
			let mut pid = AttitudePidRs::new_default();
			let mut d = Driver::Pid(&mut pid);
			record_address_universe(&mut c, &mut sim, &mut d, &init_q, &init_om,
				[0.0; 3], 400, s1)
		};

		let (s_off, o_off) = record(None);
		let s1 = RecorderStage1 { cfg: &cfg, gravity: g, k_thrust: k };
		let (s_on, o_on) = record(Some(&s1));

		assert!(!o_off.is_empty() && !o_on.is_empty(),
			"both rollouts must actually visit addresses, got {} / {}",
			o_off.len(), o_on.len());
		assert_ne!(o_on, o_off,
			"the stage-1 vertical channel must change the recorded OUTPUT universe \
			 — identical means the recorder is flying a non-translating plant again, \
			 which is the bug that made the MEMORY phase meaningless for stage 1");
		// sn=0 here, so the state universe is empty on both sides; assert that
		// rather than let a future sn>0 change slip past unnoticed.
		assert!(s_off.is_empty() && s_on.is_empty(),
			"fixture is sn=0; a non-empty state universe means the fixture changed");
	}

	/// sn=0, 12-feature (9 base + the three vertical) controller for the
	/// recorder test. Thresholds straddle the vertical ranges or the thermometer
	/// saturates and the test goes blind — same care as the Metal parity fixture.
	fn stage1_recorder_controller() -> WnnController {
		let (num_motors, levels, bpf, window, obpn) = (4usize, 4usize, 3usize, 2usize, 8usize);
		let frame_bits = 12 * bpf;
		let thresholds: Vec<f32> = (0..frame_bits)
			.map(|i| {
				let f = i / bpf;
				let step = (i % bpf) as f32 - 1.0;
				if f >= 9 { step * 0.2 } else { step * 2.0 }
			})
			.collect();
		let output_connections: Vec<i64> = (0..num_motors * levels * obpn)
			.map(|i| (i * 7 % frame_bits) as i64)
			.collect();
		WnnController::new_core(
			num_motors, levels, bpf, window, 0, 0, obpn,
			thresholds, Vec::new(), output_connections,
			false, 0.15, 0.98, 1.0,
			false, false, false, false, false,
			false, false, false,
			0.99, 1.0, 0.001, false, 1,
			ram_core::neuron_memory::BINARY, None,
			None, 0.05, false, 0.30,
			true, true, true,   // stage-1 vertical channel ON
		
			false, false,   // stage-2 horizontal channel OFF
			false,              // arm D output_full_window OFF (legacy)
		1,                  // frame_stride = 1 (legacy every-step window)
		).expect("recorder fixture must construct")
	}

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
