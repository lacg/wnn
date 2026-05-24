"""Parity: Rust AttitudePidRs == Python AttitudePID, step for step.

The Rust PID (controller.rs::AttitudePidRs) is the teacher inside the Rust
DAGGER rollout. It must produce bit-for-bit (within f32/f64 tolerance) the
SAME action as the Python reference pid.py::AttitudePID, or the Rust-trained
controller would imitate a different teacher than the validated Python one.

Drives one sim with the Python PID; at each step feeds the SAME (q, gyro,
target) to the Rust PID and compares all 4 PWMs. Both integrators stay in
sync because they see the identical input sequence.

Run:  python tests/test_pid_parity.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ram_accelerator import AttitudeSim, AttitudePidRs
from wnn.control.pid import AttitudePID, AttitudePIDConfig
from wnn.control.training import _sample_initial_state


def main():
	rng = np.random.default_rng(0)
	target = (0.0, 0.0, 0.0)
	worst = 0.0
	n_steps_total = 0

	for ep in range(10):
		ep_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
		init_q, init_omega = _sample_initial_state(
			ep_rng, math.radians(40.0), math.radians(30.0), 0.6, 0.4)
		sim = AttitudeSim()
		sim.reset(q=list(init_q), omega=list(init_omega))
		py = AttitudePID(AttitudePIDConfig()); py.reset()
		rs = AttitudePidRs(); rs.reset()

		for _ in range(2000):
			if sim.is_unstable():
				break
			gyro, accel = sim.read_imu()
			q = sim.quaternion
			py_pwm = py.step(q, gyro, target)
			rs_pwm = rs.step(list(q), list(gyro), list(target))
			d = max(abs(a - b) for a, b in zip(py_pwm, rs_pwm))
			worst = max(worst, d)
			n_steps_total += 1
			sim.step(list(py_pwm))  # both PIDs saw the same q-sequence

	print(f"Compared {n_steps_total} steps across 10 episodes.")
	print(f"Worst per-motor PWM diff (Rust vs Python): {worst:.3e}")
	TOL = 1e-4
	if worst < TOL:
		print(f"PARITY OK (< {TOL:.0e}). Rust PID matches Python teacher.")
		return 0
	print(f"PARITY FAIL (>= {TOL:.0e}). Rust PID diverges from Python teacher.")
	return 1


if __name__ == "__main__":
	sys.exit(main())
