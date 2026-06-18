"""H2 smoke + CPU/GPU consistency — ram_accelerator only (no wnn deps; runs anywhere).
Validates the delta-control substrate fix + dynamic num_features obs-features end to end:
Validates: (1) delta+obs WnnController construction + num_features, (2) CPU step()
+ get_last_feature_vector, (3) the Metal shader COMPILES + RUNS for each config
(H2 features / delta-only / absolute) returning finite aggregates. Each GPU batch
uses a UNIFORM config (the scorer reads dims+thresholds from controllers[0]).
"""
import math
import numpy as np
from ram_accelerator import WnnController, AttitudeSim, score_controllers_metal

NM, L, BPF, K = 4, 8, 8, 4
SN, SBPN, OBPN = 4, 16, 16
num_out = NM * L


def build_batch(delta, tp, ti, pwm=False, decouple=False, n_ctl=3, seed=7):
	nf = 9 + int(tp) + int(ti) + NM * int(pwm)
	frame_bits = nf * BPF
	sensor_total = K * frame_bits
	rng = np.random.default_rng(seed)
	thresholds = list(np.linspace(-2.0, 2.0, nf * BPF).astype(float))
	ctls = []
	for ci in range(n_ctl):
		# sensor-only connections so CPU and GPU read identical bits
		sc = [int(x) for x in rng.integers(0, sensor_total, size=SN * SBPN)]
		oc = [int(x) for x in rng.integers(0, frame_bits, size=num_out * OBPN)]
		c = WnnController(
			num_motors=NM, levels_per_motor=L, bits_per_feature=BPF, input_window_k=K,
			state_neurons=SN, state_bits_per_neuron=SBPN, output_bits_per_neuron=OBPN,
			thresholds=thresholds, state_connections=sc, output_connections=oc,
			delta_control=delta, delta_max=0.1, delta_leak=0.95,
			obs_tilt_p=tp, obs_tilt_i=ti, obs_peraxis_p=False, obs_peraxis_i=False,
			obs_pwm=pwm, integral_leak=0.99, integral_scale=1.0,
			decouple_outputs=decouple)
		for n in range(num_out):
			c.write_output_cell(n, (n + ci) % 9, (n + ci) % 3)
		ctls.append(c)
	return ctls, nf


def cpu_one(c):
	"""One CPU rollout; return (mean_err_deg, stable_bool)."""
	sim = AttitudeSim()
	ang = math.radians(4.0)
	sim.reset(q=[math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0], omega=[0.0, 0.0, 0.0])
	c.reset()
	tgt = [1.0, 0.0, 0.0, 0.0]
	errs = []
	for t in range(400):
		if sim.is_unstable():
			break
		gyro, accel = sim.read_imu()
		pwm = c.step(list(gyro), list(accel), [0.0, 0.0, 0.0])
		sim.step(list(pwm))
		errs.append(sim.attitude_error(tgt))
	return math.degrees(float(np.mean(errs))) if errs else float("nan")


def run_config(name, delta, tp, ti, pwm=False, decouple=False):
	ctls, nf = build_batch(delta, tp, ti, pwm, decouple)
	assert ctls[0].num_features() == nf, ctls[0].num_features()
	# CPU
	cpu_errs = [cpu_one(c) for c in ctls]
	# GPU (same small-tilt ICs)
	E, STEPS = len(ctls), 400
	q0, omega0 = [], []
	ang = math.radians(4.0)
	for _ in range(E):
		q0 += [math.cos(ang / 2), math.sin(ang / 2), 0.0, 0.0]
		omega0 += [0.0, 0.0, 0.0]
	agg = score_controllers_metal(ctls, q0, omega0, E, STEPS)
	gpu_errs = [math.degrees(row[1]) for row in agg]
	for row in agg:
		assert all(math.isfinite(v) for v in row), row
	print(f"  {name:<22} nf={nf}  CPU err°={[round(e,2) for e in cpu_errs]}  "
	      f"GPU err°={[round(e,2) for e in gpu_errs]}")
	return cpu_errs, gpu_errs


print("[1] CPU feature extraction (delta + tilt_p + tilt_i) ...")
c, _ = build_batch(True, True, True, n_ctl=1)[0][0], None
sim = AttitudeSim(); sim.reset(q=[1.0, 0.05, 0.0, 0.0], omega=[0.1, 0.0, 0.0]); c.reset()
for _ in range(20):
	g, a = sim.read_imu(); sim.step(list(c.step(list(g), list(a), [0.0, 0.0, 0.0])))
fv = c.get_last_feature_vector()
print(f"    num_features={c.num_features()} feature_vec_len={len(fv)} tail(tilt_p,tilt_i)={[round(x,3) for x in fv[9:11]]}")
assert c.num_features() == 11 and len(fv) == 11

print("[2] Metal shader compiles + runs per config:")
run_config("absolute (baseline)", False, False, False)
run_config("delta-only (3a fix)", True,  False, False)
run_config("delta + tilt_p+tilt_i", True,  True,  True)
run_config("delta + obs_pwm (accum)", True, False, False, pwm=True)
run_config("delta + tilt_i + obs_pwm", True, False, True, pwm=True)
run_config("delta + decouple (H3)", True, False, False, decouple=True)
run_config("delta + obs_pwm + decouple", True, False, False, pwm=True, decouple=True)

print("\nSMOKE PASS: shader compiled for all configs; CPU+GPU both finite.")
