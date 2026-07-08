"""Smoke: the Rust DAGGER teacher selector (cfg.teacher 0=pid,1=lqr,2=mpc) is wired
end-to-end. Trains a tiny controller through ra.dagger_train_inplace with each teacher
and confirms (a) no crash, (b) finite stats, (c) the teacher SWITCH is active — LQR
changes the trained controller vs PID, and MPC is measurably slower (its per-step QP
runs). NOTE: at extreme ICs PID and MPC both saturate to ±authority → identical clamped
labels; at tiny ICs all three quantize to the same near-hover PWM bins. So bit-identity
between two teachers in this coarse smoke is a saturation/quantization artifact, not a
wiring bug — the per-step parity test (optimal_rs_parity.py) is the teacher-fidelity proof."""
import math
import os
import time

os.environ.setdefault("WNN_RUST_DAGGER", "1")
os.environ.setdefault("WNN_STATE_SPLIT", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "2")

from wnn.control.evaluator import ControllerSpec, fit_thresholds_from_pid_rollouts, random_connectivity
from wnn.control import _accel as ra


def build_cfg(teacher_id):
	return ra.RewardGatedConfigPacked(
		num_rounds=2, episodes_per_round=3, steps_per_episode=300,
		eval_episodes=3, teacher=teacher_id,
		dt=0.001, max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)


def train(teacher_id, spec, thr, sc, oc, seed=20260609):
	ctrl = ra.WnnController(
		num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons, state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron, thresholds=thr,
		state_connections=sc, output_connections=oc,
		delta_control=spec.delta_control,
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i,
		obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i,
	)
	t0 = time.time()
	ts = ra.dagger_train_inplace(ctrl, build_cfg(teacher_id), [0.0, 0.0, 0.0], seed)
	return list(ts.iter_fitness), list(ts.iter_stable_rate), (time.time() - t0) * 1000.0


def main():
	spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
		input_window_k=4, state_neurons=16, state_bits_per_neuron=32, output_bits_per_neuron=32,
		delta_control=False, obs_tilt_p=True, obs_tilt_i=True,
		obs_peraxis_p=True, obs_peraxis_i=True)
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=5, seed=20260609)
	sc, oc = random_connectivity(spec, seed=20260609)

	results, times = {}, {}
	for name, tid in [("pid", 0), ("lqr", 1), ("mpc", 2)]:
		fit, stab, ms = train(tid, spec, thr, sc, oc)
		ok = all(math.isfinite(f) for f in fit)
		print(f"[smoke] teacher={name:3s} (id={tid})  iter_fitness={[round(f,3) for f in fit]}  "
		      f"stable={[round(s,2) for s in stab]}  finite={ok}  {ms:.0f}ms")
		results[name] = fit; times[name] = ms

	print("\n[smoke] ===== VERDICT =====")
	finite = all(all(math.isfinite(f) for f in results[n]) for n in results)
	switch_active = results["lqr"] != results["pid"]           # teacher switch changes labels
	mpc_qp_runs = times["mpc"] > times["pid"] * 1.5            # MPC's per-step QP is active
	print(f"[smoke] all finite            : {finite}")
	print(f"[smoke] lqr≠pid (switch active): {switch_active}")
	print(f"[smoke] mpc QP active (slower) : {mpc_qp_runs}  (pid {times['pid']:.0f}ms → mpc {times['mpc']:.0f}ms)")
	print(f"[smoke] {'PASS ✅' if (finite and switch_active and mpc_qp_runs) else 'CHECK ❌'}")


if __name__ == "__main__":
	main()
