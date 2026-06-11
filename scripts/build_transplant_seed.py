#!/usr/bin/env python
"""Build a phased-GA resume pickle from the 43%-hover 1-bit transplant seed.

The state-splitting trainer only plants structure when it starts from a
NON-random (warm) controller — a fresh/tumbling controller produces ~0
conflicts. The enabler (per project_controller_state_splitting) is the
untrained 1-bit GA-connectivity TRANSPLANT: graft a saved 2-bit hovering
genome's GA-optimized sensor SAMPLING into the current 1-bit shape
(arch_shape_from_spec hardcodes prefix_factor=1). Untrained = ~43% hover @5°;
bptt RETRAINING destroys it, so we DON'T train output naively — the splitting
trainer (WNN_STATE_SPLIT=1) takes it from here.

This writes an emergency-dump-schema pickle (stage_num=1 NEURONS) so
`run_phased_ga.py --resume-from-emergency SEED.pkl --resume-mode same` skips
the grid and seeds the NEURONS stage's GA population with the transplant.

Usage:
  python scripts/build_transplant_seed.py \
      --winner logs/controller/curriculum/armA_grid_neurons_memory_5deg_pop50_20260607_231655/winner.pkl \
      --out logs/controller/transplant_seed.pkl
"""
import argparse
import pickle
import time
from pathlib import Path

from wnn.control.recurrent_genome import RecurrentArchGenome
from wnn.control.evaluator import (
	arch_shape_from_spec,
	controller_genome_from_arch,
	build_controller,
	fit_thresholds_from_pid_rollouts,
)

from wnn.control.checkpoint_io import load_controller_checkpoint as _load_ctl_checkpoint




def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--winner", required=True, help="armA curriculum winner.pkl (2-bit hovering genome)")
	ap.add_argument("--out", required=True, help="output emergency-dump pickle for --resume-from-emergency")
	ap.add_argument("--thr-episodes", type=int, default=6, help="PID rollouts to fit output thresholds")
	args = ap.parse_args()

	src = Path(args.winner)
	if not src.exists():
		raise FileNotFoundError(src)
	d = _load_ctl_checkpoint(src)
	g = d["best_genome"]
	spec2 = d["spec"]  # 2-bit spec (provenance only)

	# Transplant: 1-bit shape (prefix_factor=1) + the GA-optimized sampling, cells=None.
	ng = RecurrentArchGenome(
		shape=arch_shape_from_spec(spec2),
		state_neurons=g.state_neurons,
		output_neurons=g.output_neurons,
		state_sampled=[list(s) for s in g.state_sampled],
		output_sampled=[list(s) for s in g.output_sampled],
		cells=None,
	)
	thr = fit_thresholds_from_pid_rollouts(spec2, num_episodes=args.thr_episodes, seed=0)
	cg = controller_genome_from_arch(ng, spec2, thr)  # cg.spec is the 1-bit spec

	# Sanity: the genome must build into a controller without error.
	build_controller(cg)
	spec1 = cg.spec
	print(f"[transplant] source winner: {src}")
	print(f"[transplant] 1-bit spec: state_neurons={spec1.state_neurons} "
	      f"bits_per_feature={spec1.bits_per_feature} input_window_k={spec1.input_window_k} "
	      f"num_motors={spec1.num_motors} levels_per_motor={spec1.levels_per_motor}")
	print(f"[transplant] genome: state_neurons={ng.state_neurons} output_neurons={ng.output_neurons} "
	      f"cells={'None (untrained)' if ng.cells is None else 'present'}")

	payload = {
		"stage_num": 1,           # 1 = NEURONS stage
		"stage_name": "NEURONS",
		"spec": spec1,            # 1-bit spec carried forward (grid skipped)
		"population": [ng],       # GA expands this seed up to --pop via mutation
		"best_genome": ng,        # pinned into gen-0 elite slate
		"generation": 0,
		"fitness_weights": {"err_sq": 1.0, "stable": 0.0, "jerk": 0.0, "mono": 0.0},
		"meta": {
			"saved_at_unix": time.time(),
			"saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
			"emergency_dump": True,
			"transplant_seed": True,
			"source_winner": str(src),
			"note": "43%-hover 1-bit GA-connectivity transplant; launch with WNN_STATE_SPLIT=1",
		},
	}
	out = Path(args.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	with open(out, "wb") as f:
		pickle.dump(payload, f)
	print(f"[transplant] wrote resume seed → {out}  "
	      f"(launch: run_phased_ga.py --resume-from-emergency {out} --resume-mode same)")


if __name__ == "__main__":
	main()
