"""True held-out validation of a saved controller stage checkpoint — through the
pipeline's OWN held-out code, never a private copy of it.

Usage:
  python scripts/evaluate_winner_holdout.py \\
    --winner logs/controller/sweep_ladder/ckpt/<TAG>/stage1_neurons.yaml.gz \\
    [--report-seeds 99990101 99990102 ...]

What it does:
  1. Re-parse the run's EXACT command line — the checkpoint's meta["argv"], else
     the `[argv]` line of the run's .out (both recorded since 26/10/2026) — and
     build its EpisodeConfig with phased_ga.episode_config_from_args, the plant the
     run actually flew. A run without a recorded argv is REFUSED: rebuilding its
     flags from the .out header guesses, and the guess silently drops recipe flags
     such as --obs-pwm (measured 26/10 on CTRL-19 s5).
  2. Load the checkpoint's population; the published genome is population[0].
  3. Score it on each report seed with phased_ga._holdout_report: K=5 training
     for an arch-only genome, cells read under TRAIN-seed thresholds
     (_report_thresholds), the same code that writes every marker's held-out rows.
  4. Print the checkpoint's training-side metrics next to the held-out mean±SD.

Rewritten 26/10/2026 (Luiz: "touch it just in case"). The old version kept its own
copy of the held-out: num_eval_folds=1, thresholds re-fit on the held-out seed (the
01/08 threshold-misalignment bug, docs/threshold_misalignment_finding.md), and a
hand-built EpisodeConfig (no airframe/obs/translation, tilt 15) — three ways to
report a number for a controller other than the one the run built.
"""
from __future__ import annotations

import argparse
import shlex
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rescore_first_report_seed import _find_out, parse_out  # noqa: E402
from wnn.control.phased_ga import (_ctl_load, _holdout_report, build_arg_parser,  # noqa: E402
                                   episode_config_from_args)


def _recorded_argv(payload: dict, out_path: str) -> list[str] | None:
	"""The run's command line: checkpoint meta first, then the .out's `[argv]` line."""
	argv = (payload.get("meta") or {}).get("argv")
	if argv:
		return list(argv)
	for line in open(out_path, errors="replace"):
		if line.startswith("[argv] "):
			return shlex.split(line[len("[argv] "):])
	return None


def _mean_sd(xs: list[float]) -> str:
	return f"{statistics.mean(xs):.2f}±{statistics.pstdev(xs) if len(xs) > 1 else 0.0:.2f}"


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--winner", required=True, help="ckpt/<TAG>/stageN_<name>.yaml.gz")
	ap.add_argument("--report-seeds", type=int, nargs="+", default=None,
	                help="Held-out seeds (default: the run's own --report-seeds from its .out).")
	a = ap.parse_args()

	path = Path(a.winner)
	tag = path.parent.name
	out = _find_out(tag)
	if out is None:
		print(f"ERROR: no .out for {tag} — cannot rebuild the run's recipe", file=sys.stderr)
		return 1
	facts = parse_out(out)
	seeds = a.report_seeds or facts.get("report_seeds")
	if not seeds or "train" not in facts or "steps" not in facts:
		print(f"ERROR: {out} header incomplete (seed set / recipe line / report seeds)", file=sys.stderr)
		return 1
	payload = _ctl_load(str(path))
	argv = _recorded_argv(payload, out)
	if argv is None:
		print(f"ERROR: {tag} recorded no command line (run predates 26/10/2026). Its flags cannot "
		      f"be rebuilt exactly, and a guessed recipe scores a different controller. Refused.",
		      file=sys.stderr)
		return 1
	args = build_arg_parser().parse_args(argv)
	ec = episode_config_from_args(args)
	pop = list(payload.get("population") or [])
	genome = pop[0] if pop else payload.get("best_genome")
	if genome is None:
		print("ERROR: checkpoint has neither population nor best_genome", file=sys.stderr)
		return 1
	stage = payload.get("stage_name") or path.stem
	rows = [_holdout_report(args, ec, payload["spec"], genome, pop or None, s, facts["train"], stage_label=stage)
	        for s in seeds]

	prev = payload.get("metrics") or {}
	prev = prev if isinstance(prev, dict) else vars(prev)
	print(f"\n{'=' * 72}\n  {tag}  [{stage}]  population[0], {len(seeds)} report seeds\n{'=' * 72}")
	if prev:
		print(f"  training-side (checkpoint): stable={prev.get('acc', float('nan')) * 100:.1f}%  "
		      f"err={prev.get('mean_attitude_error_deg', float('nan')):.2f}°  "
		      f"steady={prev.get('mean_steady_error_deg') or float('nan'):.2f}°")
	print(f"  held-out:                   stable={_mean_sd([m.acc * 100 for m in rows])}%  "
	      f"err={_mean_sd([m.mean_attitude_error_deg for m in rows])}°  "
	      f"steady={_mean_sd([m.mean_steady_error_deg for m in rows if m.mean_steady_error_deg is not None])}°  "
	      f"alt={_mean_sd([m.mean_altitude_error_m for m in rows if getattr(m, 'mean_altitude_error_m', None) is not None])}m")
	return 0


if __name__ == "__main__":
	sys.exit(main())
