"""Rebuild a resumable GA population checkpoint from an experiment-end PhaseResult
`.json.gz` (the `<phase>.json.gz` that experiment.py writes via _save_checkpoint).

Why this exists
---------------
A GA crash can leave the rolling `ga_checkpoint_ga.json` overwritten or empty
(e.g. a worker that re-claimed the flow, did a fresh-start wipe, then was killed)
while the experiment-end PhaseResult gz still holds the last good population —
genomes WITH connections + cached metrics, plus the best genomes. This tool
turns that gz back into a `CheckpointManager`-compatible checkpoint so the GA's
resume path can pick the population back up.

The gz captures the population but NOT the live GA control state (current
generation, patience, threshold, best fitness). Those are supplied explicitly so
the rebuilt checkpoint resumes at the correct generation with the correct
early-stopping state — pass the values recorded for the last completed gen.

Usage
-----
    python scripts/phase_gz_to_ga_checkpoint.py \
        --gz   /path/exp_01/ga_neurons.json.gz \
        --checkpoint-dir /path/exp_01 \
        --resume-iteration 75 --patience-counter 4 --threshold 0.00296 \
        --best-ce 0.04019011287897468 --best-acc 0.9892181268024033 \
        --total-iterations 250 --phase-name "GA Neurons"
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))

from wnn.ram.strategies.connectivity.architecture_strategies import (
	CheckpointConfig,
	CheckpointManager,
)
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome


def _genome_from_gz(gd: dict) -> ClusterGenome:
	g = ClusterGenome(
		bits_per_neuron=list(gd["bits_per_neuron"]),
		neurons_per_cluster=list(gd["neurons_per_cluster"]),
		connections=list(gd["connections"]) if gd.get("connections") else None,
	)
	cm = gd.get("cached_metrics")
	if cm:
		try:
			from wnn.ram.metrics import Metrics
			g.metrics = Metrics(ce=cm.get("ce", 0.0), acc=cm.get("acc", 0.0),
			                    f1=cm.get("f1"), fpr=cm.get("fpr"))
		except Exception:
			pass
	return g


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--gz", required=True, help="Path to the experiment-end PhaseResult .json.gz")
	grp = ap.add_mutually_exclusive_group(required=True)
	grp.add_argument("--checkpoint-dir", help="Exp checkpoint dir; writes ga_checkpoint_ga.json there")
	grp.add_argument("--out", help="Explicit checkpoint file path")
	ap.add_argument("--resume-iteration", type=int, required=True,
	                help="Last completed generation (resume continues at this + 1)")
	ap.add_argument("--patience-counter", type=int, default=0)
	ap.add_argument("--threshold", type=float, default=0.0)
	ap.add_argument("--best-ce", type=float, required=True)
	ap.add_argument("--best-acc", type=float, required=True)
	ap.add_argument("--total-iterations", type=int, default=0)
	ap.add_argument("--phase-name", default="GA")
	ap.add_argument("--dry-run", action="store_true")
	args = ap.parse_args()

	with gzip.open(args.gz) as f:
		data = json.load(f)
	pr = data["phase_result"]
	pop_src = pr.get("final_population") or []
	if not pop_src:
		raise SystemExit("gz has no final_population to recover")

	population = [(_genome_from_gz(gd), (gd.get("cached_metrics") or {}).get("ce", 0.0)) for gd in pop_src]
	best_src = data.get("best_acc_genome") or pr.get("best_genome") or pop_src[0]
	best_genome = _genome_from_gz(best_src)

	if args.out:
		ckpt_dir = Path(args.out).parent
		prefix = Path(args.out).stem.replace("_ga", "")
	else:
		ckpt_dir = Path(args.checkpoint_dir)
		prefix = "ga_checkpoint"

	print(f"Source gz        : {args.gz}")
	print(f"  population     : {len(population)} genomes (connections present: "
	      f"{sum(1 for g, _ in population if g.has_connections())}/{len(population)})")
	print(f"  resume at gen  : {args.resume_iteration + 1} (of {args.total_iterations or '?'})")
	print(f"  patience       : {args.patience_counter}")
	print(f"  threshold      : {args.threshold}")
	print(f"  best_fitness   : ce={args.best_ce:.5f} acc={args.best_acc:.5%}")
	print(f"  checkpoint     : {ckpt_dir / (prefix + '_ga.json')}")

	if args.dry_run:
		print("\n[dry-run] nothing written.")
		return 0

	mgr = CheckpointManager(
		config=CheckpointConfig(enabled=True, checkpoint_dir=ckpt_dir, filename_prefix=prefix),
		phase_name=args.phase_name,
		optimizer_type="GA",
		total_iterations=args.total_iterations or (args.resume_iteration + 1),
		logger=print,
	)
	mgr.save(
		iteration=args.resume_iteration,
		population=population,
		best_genome=best_genome,
		best_fitness=(args.best_ce, args.best_acc),
		current_threshold=args.threshold,
		extra_state={
			"patience_counter": args.patience_counter,
			"recovered_from_gz": True,
			"complete": False,
		},
	)
	state = mgr.load(ClusterGenome)
	assert state["current_iteration"] == args.resume_iteration
	assert len(state["population"]) == len(population)
	print(f"\n✓ Rebuilt + verified: resume at gen {args.resume_iteration + 1} with "
	      f"{len(state['population'])} genomes, patience {args.patience_counter}, "
	      f"threshold {args.threshold}.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
