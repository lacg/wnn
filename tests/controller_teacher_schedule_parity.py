"""Parity gate for hybrid teachers (task #11, teacher-hybrids-roadmap).

The roadmap's hard gate: a constant teacher_schedule=[X]*N must be BIT-EXACT
identical to the legacy scalar --teacher X path (same cells, same stats), so
enabling the hybrid plumbing provably changes nothing for existing runs.

Cases:
  B  teacher_schedule=[lqr,lqr] over 4 rounds  ≡ scalar lqr   (constant + last-entry extension)
  C  teacher_blend=[lqr]                        ≡ scalar lqr   (single-entry blend)
  E  teacher_schedule=[pid,lqr] ≡ [pid,lqr,lqr,lqr]            (extension semantics on a MIXED schedule)
  D  teacher_blend=[lqr,pid] runs and actually changes labels  (cells differ from pure lqr)

Run: source wnn/bin/activate && PYTHONPATH=src/wnn python tests/controller_teacher_schedule_parity.py
"""
import os
os.environ["WNN_STATE_SPLIT"] = "0"          # exercise the per-traj BPTT path (fast, deterministic)
os.environ.pop("WNN_CONTROLLER_GPU_TRAIN", None)

from wnn.control import _accel as ra
from wnn.control.evaluator import ControllerSpec, ControllerGenome, build_controller, random_connectivity

SEED = 12345
ROUNDS = 4
LQR, PID = 1, 0


def make_spec() -> ControllerSpec:
	return ControllerSpec(
		levels_per_motor=8, bits_per_feature=4, input_window_k=2,
		state_neurons=12, state_bits_per_neuron=18, output_bits_per_neuron=18,
	)


def make_controller(spec: ControllerSpec):
	"""Fresh all-EMPTY controller, identical every call (fixed connectivity seed)."""
	nf = spec.num_features()
	thresholds = [
		(b + 1) / (spec.bits_per_feature + 1) * 2.0 - 1.0
		for _ in range(nf) for b in range(spec.bits_per_feature)
	]
	sc, oc = random_connectivity(spec, seed=7)
	return build_controller(ControllerGenome(spec, thresholds, sc, oc))


def make_cfg(teacher: int, schedule: list[int] = (), blend: list[int] = ()):
	return ra.RewardGatedConfigPacked(
		num_rounds=ROUNDS, episodes_per_round=6, steps_per_episode=250,
		eval_episodes=2, teacher=teacher,
		teacher_schedule=list(schedule), teacher_blend=list(blend),
	)


def train(cfg):
	spec = make_spec()
	c = make_controller(spec)
	stats = ra.dagger_train_inplace(c, cfg, [0.0, 0.0, 0.0], SEED)
	s_cells, o_cells = c.export_cells()
	return (sorted(s_cells), sorted(o_cells)), (
		list(stats.iter_fitness), list(stats.iter_mean_err_deg),
		list(stats.iter_cells_written))


def check(name: str, got, want, equal: bool):
	ok = (got == want) if equal else (got != want)
	rel = "==" if equal else "!="
	print(f"  [{'PASS' if ok else 'FAIL'}] {name}: cells+stats {rel} baseline")
	if not ok:
		raise SystemExit(f"PARITY GATE FAILED: {name}")


def main():
	print(f"Teacher-schedule parity gate ({ROUNDS} rounds x 6 episodes x 250 steps, seed {SEED})")
	base = train(make_cfg(teacher=LQR))                                   # A: legacy scalar lqr
	print("  baseline (scalar lqr) trained:",
	      f"{len(base[0][0])} state cells, {len(base[0][1])} output cells")

	# B: constant schedule (len 2 < rounds → also proves last-entry extension).
	# teacher=PID on purpose: proves the schedule, not the scalar, is in charge.
	check("B schedule=[lqr,lqr] (teacher=pid)", train(make_cfg(PID, schedule=[LQR, LQR])), base, equal=True)

	# C: single-entry blend ≡ scalar.
	check("C blend=[lqr]      (teacher=pid)", train(make_cfg(PID, blend=[LQR])), base, equal=True)

	# E: extension semantics on a MIXED schedule: [pid,lqr] over 4 rounds must
	# equal the explicit [pid,lqr,lqr,lqr].
	e1 = train(make_cfg(PID, schedule=[PID, LQR]))
	e2 = train(make_cfg(PID, schedule=[PID, LQR, LQR, LQR]))
	check("E [pid,lqr] == [pid,lqr,lqr,lqr]", e1, e2, equal=True)

	# D: a real blend must actually change the labels vs pure lqr (sanity that
	# the per-episode selector reaches the rollout).
	check("D blend=[lqr,pid] differs from pure lqr", train(make_cfg(PID, blend=[LQR, PID])), base, equal=False)

	print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
	main()
