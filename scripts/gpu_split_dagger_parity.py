"""Parity: dagger_train_inplace with the GPU state-split offload (Task 3,
WNN_CONTROLLER_GPU_TRAIN=1) trains the SAME controller cells as the CPU split
(WNN_CONTROLLER_GPU_TRAIN=0).

Rollouts, teacher, and sim are all CPU + deterministic (seed_from_u64), so both
runs see identical gated batches; the ONLY difference is whether the split loop
runs on Metal or rayon-CPU. split_train_loop_gpu is bit-exact vs split_train_loop
(controller_split_train_loop_parity), so the final exported cells must match
cell-for-cell. The saturation/wish_bits GA-handshake signal is intentionally
absent on the GPU path — this test checks the trained MEMORY, not that signal.

Run only when the GPU is free (contention with the IDS worker just slows it)."""
import math
import os

os.environ.setdefault("WNN_RUST_DAGGER", "1")
os.environ["WNN_STATE_SPLIT"] = "1"          # split path is the thing under test
os.environ.setdefault("RAYON_NUM_THREADS", "4")

from wnn.control.evaluator import ControllerSpec, fit_thresholds_from_pid_rollouts, random_connectivity
from wnn.control import _accel as ra

SEED = 20260709
TEACHER = 1  # lqr


def build_cfg():
	return ra.RewardGatedConfigPacked(
		num_rounds=3, episodes_per_round=4, steps_per_episode=400,
		eval_episodes=4, teacher=TEACHER,
		dt=0.001, max_initial_yaw_rad=math.radians(5.0),
		max_initial_body_rate=0.5, max_initial_yaw_rate=0.3,
	)


def fresh_controller(spec, thr, sc, oc):
	return ra.WnnController(
		num_motors=spec.num_motors, levels_per_motor=spec.levels_per_motor,
		bits_per_feature=spec.bits_per_feature, input_window_k=spec.input_window_k,
		state_neurons=spec.state_neurons, state_bits_per_neuron=spec.state_bits_per_neuron,
		output_bits_per_neuron=spec.output_bits_per_neuron, thresholds=thr,
		state_connections=sc, output_connections=oc,
		delta_control=spec.delta_control,
		obs_tilt_p=spec.obs_tilt_p, obs_tilt_i=spec.obs_tilt_i,
		obs_peraxis_p=spec.obs_peraxis_p, obs_peraxis_i=spec.obs_peraxis_i,
	)


def train_and_export(spec, thr, sc, oc, gpu):
	os.environ["WNN_CONTROLLER_GPU_TRAIN"] = "1" if gpu else "0"
	ctrl = fresh_controller(spec, thr, sc, oc)
	ts = ra.dagger_train_inplace(ctrl, build_cfg(), [0.0, 0.0, 0.0], SEED)
	state_cells, output_cells = ctrl.export_cells()
	# Normalize to sets of (neuron, addr, value) — order-independent compare.
	return set(state_cells), set(output_cells), list(ts.iter_fitness)


def main():
	spec = ControllerSpec(num_motors=4, levels_per_motor=16, bits_per_feature=8,
		input_window_k=4, state_neurons=16, state_bits_per_neuron=32, output_bits_per_neuron=32,
		delta_control=False, obs_tilt_p=True, obs_tilt_i=True,
		obs_peraxis_p=True, obs_peraxis_i=True)
	thr = fit_thresholds_from_pid_rollouts(spec, num_episodes=5, seed=SEED)
	sc, oc = random_connectivity(spec, seed=SEED)

	cpu_s, cpu_o, cpu_fit = train_and_export(spec, thr, sc, oc, gpu=False)
	gpu_s, gpu_o, gpu_fit = train_and_export(spec, thr, sc, oc, gpu=True)

	s_only_cpu, s_only_gpu = cpu_s - gpu_s, gpu_s - cpu_s
	o_only_cpu, o_only_gpu = cpu_o - gpu_o, gpu_o - cpu_o
	state_ok = not s_only_cpu and not s_only_gpu
	out_ok = not o_only_cpu and not o_only_gpu

	print(f"[parity] state cells: cpu={len(cpu_s)} gpu={len(gpu_s)} "
	      f"cpu_only={len(s_only_cpu)} gpu_only={len(s_only_gpu)}  {'OK' if state_ok else 'MISMATCH'}")
	print(f"[parity] output cells: cpu={len(cpu_o)} gpu={len(gpu_o)} "
	      f"cpu_only={len(o_only_cpu)} gpu_only={len(o_only_gpu)}  {'OK' if out_ok else 'MISMATCH'}")
	print(f"[parity] iter_fitness cpu={[round(f,4) for f in cpu_fit]}")
	print(f"[parity] iter_fitness gpu={[round(f,4) for f in gpu_fit]}")
	print(f"[parity] {'PASS ✅' if (state_ok and out_ok) else 'FAIL ❌'}")
	if not (state_ok and out_ok):
		for tag, s in [("state_cpu_only", s_only_cpu), ("state_gpu_only", s_only_gpu),
		               ("out_cpu_only", o_only_cpu), ("out_gpu_only", o_only_gpu)]:
			if s:
				print(f"  {tag} (first 5): {sorted(s)[:5]}")


if __name__ == "__main__":
	main()
