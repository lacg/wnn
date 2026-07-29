#!/usr/bin/env python3
"""Replay saved controller winners on extra held-out report seeds.

WHY: the study table gives each WNN cell a ±SD over TRAINING seeds on one fixed
held-out draw, while the classical baselines carry a ±SD over REPORT seeds with no
training at all. Those are different axes and cannot be read as the same error bar.
Re-scoring a frozen winner on several report seeds puts the WNN on the BASELINE's
axis (test-set variance), which is the comparison that can be stated as
"WNN a±b vs PID c±d" without an apples-to-oranges footnote.

No search and no training happen here: a MEMORY-stage winner already carries the
cells it was trained with on its own train seed, so this is rollouts only. That is
also what keeps it honest — the report seeds are never seen by any training pass.

TWO THRESHOLD VARIANTS, because the answer differs and the difference IS the point:

  per_seed  — decode thresholds fit from PID rollouts on EACH report seed. This is
              exactly what phased_ga._holdout_report does, so these numbers are
              directly comparable to the study table.
  train     — decode thresholds fit ONCE on the winner's own train seed, then held
              fixed across every report seed. Nothing about the test draw touches
              the model.

The gap between them measures the only place a report seed influences anything
before scoring. It is not label leakage (thresholds come from the PID teacher, not
from the genome's score), but it IS calibration against the test distribution, and
a reviewer is entitled to ask. If the two variants agree, the question is settled
with a measurement instead of an argument.

VALIDATION STATUS (29/07/2026) — NOT yet reproducing the study exactly. Replaying
1layer_9feat_BINARY_s31337002 on its own report seed reproduces the recorded
stable=87.0% and err=3.3670786° to full precision, but steady comes out 3.4554°
against a logged 3.47°. On 1layer_9feat_BINARY_s31337003 stable is 41.0% against a
logged 39.0%, with err agreeing to the logged 2dp. Output is bit-reproducible across
repeat runs, so this is a systematic path difference, not noise.

Ruled out so far: batch composition (scoring the winner alone vs inside the
8-genome holdout sample gives identical values), rg_config.teacher (pid and lqr give
identical values — score_genomes does no training), non-determinism, the
body-rate/yaw-rate defaults, and scoring best_genome instead of population[0]. Both
of the latter two WERE real bugs found by this validation and are fixed.

Do not treat this script's numbers as publishable until a replay reproduces a
recorded held-out triple exactly. Remaining suspect: whether the saved
population[0] + cells are byte-identical to the genome state that _holdout_report
scored, given the winner is written after the MEMORY stage completes.

Usage:
  rescore_winners.py --glob 'logs/controller/dfa1l/*_winner.yaml.gz' \
      --report-seeds 99990101 99990102 99990103 99990104 99990105 \
      --out experiments/dfa1l_markers/rescore.json
"""
import argparse
import glob as globmod
import json
import math
import os
import re
import statistics
import sys

_TAG = re.compile(r"^(?P<sub>[^_]+)_(?P<feat>[^_]+)_(?P<mode>[^_]+)_s(?P<seed>\d+)$")


def _parse_tag(path):
	"""'.../dfa_9feat_QUAD_s31337002_winner.yaml.gz' → dict of cell coordinates."""
	tag = os.path.basename(path).replace("_winner.yaml.gz", "")
	m = _TAG.match(tag)
	if not m:
		return None
	d = m.groupdict()
	d["tag"], d["seed"] = tag, int(d["seed"])
	return d


def _episode_config(a):
	from wnn.control.evaluator import EpisodeConfig
	from wnn.control.training import DisturbanceConfig
	# seed=911 matches phased_ga: the disturbance PRESET is fixed across all runs, so
	# only the episode draw varies with the report seed.
	return EpisodeConfig(
		dt=0.001, steps_per_episode=a.steps,
		max_initial_tilt_rad=math.radians(a.tilt),
		max_initial_yaw_rad=math.radians(a.tilt),
		# NOT 0.0 — phased_ga's --body-rate/--yaw-rate default to 0.5/0.3 and the study
		# never overrode them. Zeroing these silently removes the initial angular rates,
		# making every episode easier and every re-scored number wrong-but-plausible.
		max_initial_body_rate=a.body_rate, max_initial_yaw_rate=a.yaw_rate,
		disturbance=DisturbanceConfig.preset(a.disturbance, seed=911),
	)


def _thresholds(spec, seed, ec):
	from wnn.control.evaluator import fit_thresholds_from_pid_rollouts
	return fit_thresholds_from_pid_rollouts(
		spec, num_episodes=10, seed=seed,
		geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None))


def _score_once(spec, genome, ec, report_seed, thresholds, a):
	"""One frozen genome, one report seed → (stable%, err°, steady°). Rollouts only."""
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.reward_gated import RewardGatedConfig
	rg = RewardGatedConfig(seed=report_seed, episode_config=ec)
	rg.steps_per_episode, rg.progress = a.steps, False
	ev = ControllerEvaluator(spec, num_eval_episodes=a.episodes, seed=report_seed,
	                         episode_config=ec, thresholds=thresholds, rg_config=rg,
	                         num_eval_folds=5)
	m = ev.score_genomes([genome])[0]
	return (m.acc * 100.0, m.mean_attitude_error_deg,
	        getattr(m, "mean_steady_error_deg", None))


def _load_winner(path):
	"""→ (spec, genome) or None. A winner without cells cannot be score-only.

	The genome is population[0], NOT best_genome. phased_ga._holdout_report scores
	list(final_population) and reports metrics[0] — "final_population[0] = the
	during-search winner = THE RESULT" — so population[0] is the genome every number
	in the study table describes. The payload's best_genome is a DIFFERENT
	architecture (verified: differing output_sampled on a real winner file), so
	scoring it silently re-scores the wrong controller.
	"""
	from wnn.control.checkpoint_io import load_controller_checkpoint
	payload = load_controller_checkpoint(path)
	if not payload:
		return None
	pop = payload.get("population") or []
	g = pop[0] if pop else payload.get("best_genome")
	spec = payload.get("spec")
	if spec is None or g is None or getattr(g, "cells", None) is None:
		return None
	return spec, g


def _agg(triples):
	"""[(stable, err, steady), ...] → {metric: [mean, sd]}."""
	out = {}
	for i, name in enumerate(("stable", "err", "steady")):
		xs = [t[i] for t in triples if t[i] is not None]
		if xs:
			out[name] = [statistics.mean(xs),
			             statistics.pstdev(xs) if len(xs) > 1 else 0.0]
	return out


def _rescore_cell(path, meta, a, ec):
	"""Both threshold variants for one winner, across every report seed."""
	from wnn.seeds import resolve_seed_set
	loaded = _load_winner(path)
	if loaded is None:
		print(f"  SKIP {meta['tag']} — no trained cells in winner (arch-only)")
		return None
	spec, genome = loaded
	train_seed = resolve_seed_set(base=meta["seed"], run_index=0).train
	fixed = _thresholds(spec, train_seed, ec)          # variant 'train': fit ONCE
	runs = {"per_seed": [], "train": []}
	for rs in a.report_seeds:
		runs["per_seed"].append(_score_once(spec, genome, ec, rs, _thresholds(spec, rs, ec), a))
		runs["train"].append(_score_once(spec, genome, ec, rs, fixed, a))
	return {"tag": meta["tag"], "substrate": meta["sub"], "feature": meta["feat"],
	        "mode": meta["mode"], "seed": meta["seed"], "train_seed": train_seed,
	        "report_seeds": a.report_seeds,
	        "per_seed": {"agg": _agg(runs["per_seed"]), "runs": runs["per_seed"]},
	        "train": {"agg": _agg(runs["train"]), "runs": runs["train"]}}


def _print_row(r):
	def cell(v, key):
		m = v["agg"].get(key)
		return f"{m[0]:5.1f}±{m[1]:4.1f}" if m else "   —   "
	ps, tr = r["per_seed"], r["train"]
	print(f"  {r['tag']:34} {cell(ps,'stable'):>11} {cell(ps,'err'):>11} "
	      f"{cell(ps,'steady'):>11}  | {cell(tr,'stable'):>11} {cell(tr,'err'):>11} "
	      f"{cell(tr,'steady'):>11}")


def _print_table(results, a):
	print("=" * 118)
	print(f"  WINNER RE-SCORE — frozen winners replayed on {len(a.report_seeds)} report "
	      f"seeds (TEST-SET variance: the baselines' axis)")
	print(f"  seeds {a.report_seeds} | {a.episodes} ep x {a.steps} steps | "
	      f"{a.disturbance}, tilt {a.tilt}°")
	print("=" * 118)
	print(f"  {'cell':34} {'--- thresholds fit PER REPORT SEED ---':>35}  "
	      f"| {'--- thresholds fit on TRAIN SEED ---':>35}")
	print(f"  {'':34} {'stable%':>11} {'err°':>11} {'steady°':>11}  "
	      f"| {'stable%':>11} {'err°':>11} {'steady°':>11}")
	print("  " + "-" * 114)
	for r in results:
		_print_row(r)
	print("=" * 118)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--glob", default="logs/controller/dfa1l/*_winner.yaml.gz")
	ap.add_argument("--report-seeds", type=int, nargs="+",
	                default=[99990101, 99990102, 99990103, 99990104, 99990105])
	ap.add_argument("--episodes", type=int, default=100)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--tilt", type=float, default=5.0)
	# Defaults MUST track phased_ga's --body-rate/--yaw-rate, or the replayed episodes
	# are not the episodes the winner was scored on.
	ap.add_argument("--body-rate", type=float, default=0.5)
	ap.add_argument("--yaw-rate", type=float, default=0.3)
	ap.add_argument("--disturbance", default="L2D")
	ap.add_argument("--out", required=True)
	a = ap.parse_args()

	ec = _episode_config(a)
	results = []
	for path in sorted(globmod.glob(a.glob)):
		meta = _parse_tag(path)
		if meta is None:
			print(f"  SKIP {path} — unparseable tag")
			continue
		print(f"  scoring {meta['tag']} ...", flush=True)
		r = _rescore_cell(path, meta, a, ec)
		if r:
			results.append(r)
	if not results:
		print("no winners scored", file=sys.stderr)
		return 1
	_print_table(results, a)
	with open(a.out, "w") as f:
		json.dump({"meta": {"report_seeds": a.report_seeds, "episodes": a.episodes,
		                    "steps": a.steps, "tilt_deg": a.tilt,
		                    "disturbance": a.disturbance,
		                    "variance_note": "these ±SD are TEST-SET variance (frozen "
		                                     "winner, different held-out draws) — the "
		                                     "same axis as the classical baselines, NOT "
		                                     "the training-seed ±SD in the study table."},
		           "cells": results}, f, indent=1)
	print(f"# wrote {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
