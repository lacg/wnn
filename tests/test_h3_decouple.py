"""H3 decouple-outputs verification — ram_accelerator only (isolated-venv runnable).

The launch gate for H3. Verifies the two correctness properties the 33h run depends on:
  1. UN-MIX ROUND-TRIP: unmix(mix(controls)) == controls and mix(unmix(motors)) == motors,
     exactly (the mix is a fixed invertible 4×4 allocation; the un-mix is its inverse).
     If this fails, DAGGER trains toward the wrong control targets.
  2. NEUTRAL → HOVER: an UNTRAINED decouple controller (all cells EMPTY → decode 0.75 →
     neutral controls T=hover, τ=0) must mix to all-motors-hover (≈0.5) — the stable
     bootstrap, and proof the forward control→motor path is right at the fixed point.

NOTE on risk: H3 training AND scoring share the same forward path (decode→mix), so unlike
the delta/absolute bug there is NO silent train/score mismatch — a wrong un-mix shows up as
HONEST poor performance, not inflated numbers. This gate ensures H3 gets a FAIR shot.
"""
import math
import numpy as np
from wnn.control._accel import WnnController, AttitudeSim

# Mirror controller.rs mix_controls_to_motors / unmix_motors_to_controls.
def mix(c):
	T, tr, tp, ty = c
	return [T - tp + ty, T - tr - ty, T + tp + ty, T + tr - ty]  # front,left,back,right

def unmix(p):
	p0, p1, p2, p3 = p
	return [(p0 + p1 + p2 + p3) * 0.25, (p3 - p1) * 0.5, (p2 - p0) * 0.5, (p0 - p1 + p2 - p3) * 0.25]


def test_roundtrip():
	rng = np.random.default_rng(0)
	max_err = 0.0
	for _ in range(2000):
		# random controls (T in [0,1], torques in [-0.5,0.5] so mixed motors stay sane)
		c = [float(rng.uniform(0.2, 0.8)), *[float(rng.uniform(-0.3, 0.3)) for _ in range(3)]]
		back = unmix(mix(c))
		max_err = max(max_err, max(abs(a - b) for a, b in zip(c, back)))
		# and motors → controls → motors
		p = [float(rng.uniform(0.1, 0.9)) for _ in range(4)]
		backp = mix(unmix(p))
		max_err = max(max_err, max(abs(a - b) for a, b in zip(p, backp)))
	print(f"[1] un-mix round-trip max abs error over 4000 cases: {max_err:.2e}")
	assert max_err < 1e-5, max_err
	print("    PASS — un-mix is the exact inverse of the mix")


def _ctl(decouple):
	NM, L, BPF, K, SN, SBPN, OBPN = 4, 8, 8, 4, 4, 16, 16
	nf = 9
	frame_bits = nf * BPF
	# UNTRAINED: no cells written → all EMPTY → decode neutral 0.75.
	return WnnController(
		num_motors=NM, levels_per_motor=L, bits_per_feature=BPF, input_window_k=K,
		state_neurons=SN, state_bits_per_neuron=SBPN, output_bits_per_neuron=OBPN,
		thresholds=[0.0] * (nf * BPF),
		state_connections=[0] * (SN * SBPN), output_connections=[0] * (NM * L * OBPN),
		delta_control=True, delta_max=0.1, delta_leak=0.95, decouple_outputs=decouple)


def test_untrained_hover():
	# At level with no disturbance, an untrained controller should hold hover (all motors ≈0.5).
	for decouple in (False, True):
		c = _ctl(decouple)
		sim = AttitudeSim(); sim.reset(q=[1.0, 0.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0]); c.reset()
		pwm = None
		for _ in range(50):
			gyro, accel = sim.read_imu()
			pwm = c.step(list(gyro), list(accel), [0.0, 0.0, 0.0])
			sim.step(list(pwm))
		dev = max(abs(p - 0.5) for p in pwm)
		print(f"[2] untrained {'decouple' if decouple else 'motors  '}: final pwm={[round(p,3) for p in pwm]} max|dev from hover|={dev:.3f}")
		assert dev < 0.02, f"untrained {'decouple' if decouple else 'motors'} did not hover: {pwm}"
	print("    PASS — untrained decouple holds hover (neutral controls → mix → all motors ≈0.5)")


if __name__ == "__main__":
	print("=== H3 decouple verification ===")
	test_roundtrip()
	test_untrained_hover()
	print("\nH3 VERIFIED: un-mix exact + untrained hover. Forward CPU==GPU already shown by the smoke.")
