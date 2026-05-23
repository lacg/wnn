"""CLI entry for standalone controller training (no dashboard needed).

Usage:
  python -m wnn.control train --config=configs/controller_paper1.yaml
  python -m wnn.control eval  --genome-id=NNN
  python -m wnn.control sim-demo

Once the sim physics + controller forward pass are real, this lets you
iterate on the controller without the dashboard's worker process.
The eventual integration path: add a flow template to the dashboard so
controller flows appear alongside IDS flows, but standalone CLI stays
the primary developer-iteration path.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="wnn.control")
	subparsers = parser.add_subparsers(dest="command", required=True)

	# train command
	train = subparsers.add_parser("train", help="Train a controller via GA + episodes.")
	train.add_argument("--config", required=False, default=None,
		help="YAML config file (TODO: schema TBD).")
	train.add_argument("--seeds", type=int, default=1,
		help="Number of independent GA seeds to run (cohort size).")
	train.add_argument("--generations", type=int, default=100,
		help="Maximum GA generations (with patience-based early stop).")

	# eval command
	ev = subparsers.add_parser("eval", help="Evaluate a trained controller against the benchmark suite.")
	ev.add_argument("--genome-id", type=int, required=True)

	# sim-demo command (useful for sanity-checking the sim implementation)
	subparsers.add_parser("sim-demo", help="Run the sim for a few seconds with hover-throttle output and print state.")

	args = parser.parse_args(argv)

	if args.command == "train":
		# TODO: load config, instantiate GA, run cohort, write to DB.
		print(f"[TODO] train: seeds={args.seeds}, generations={args.generations}")
		return 0

	if args.command == "eval":
		# TODO: load trained genome, run benchmark, print metrics.
		print(f"[TODO] eval genome {args.genome_id}")
		return 0

	if args.command == "sim-demo":
		from .sim import AttitudeSim
		sim = AttitudeSim()
		print(f"sim t={sim.time:.3f}s, q={sim.quaternion}, omega={sim.angular_velocity}")
		for _ in range(10):
			sim.step([0.5, 0.5, 0.5, 0.5])
		print(f"sim t={sim.time:.3f}s, q={sim.quaternion}, omega={sim.angular_velocity}")
		return 0

	return 1


if __name__ == "__main__":
	sys.exit(main())
