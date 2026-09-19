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

  train     — decode thresholds fit ONCE on the winner's own train seed under the
              regime it was TRAINED in, then held fixed across every report seed.
              This is what phased_ga._holdout_report does (unconditionally since
              03/08/2026 — see _report_thresholds), so these numbers are the ones
              comparable to the study table. Nothing about the test draw touches
              the model: cells + thresholds together ARE the deployed controller.
  per_seed  — decode thresholds re-fit from PID rollouts on EACH report seed under
              the SCORING regime. The diagnostic: it re-quantizes the inputs, so
              cells are read at addresses they were not written to
              (docs/threshold_misalignment_finding.md).

The gap between them measures the only place a report seed influences anything
before scoring. It is not label leakage (thresholds come from the PID teacher, not
from the genome's score), but it IS calibration against the test distribution, and
a reviewer is entitled to ask. If the two variants agree, the question is settled
with a measurement instead of an argument.

CROSS-CONDITION SCORING (multi-axis programme §3 axis A "A-cross", 19/09/2026):
--disturbance / --airframe / --translation name the regime the frozen winner is
SCORED under. The regime it was TRAINED under (which fixes its address function)
defaults to the same values — a plain replay — and is overridden per field with
--trained-disturbance / --trained-airframe / --trained-translation when the two
differ, e.g. the L4C anchor winners scored at L4A:
  rescore_winners.py --glob 'logs/controller/sweep_ladder/*_hd29_winner.yaml.gz' \\
      --airframe cf21_brushless --translation --trained-disturbance L4C \\
      --disturbance L4A --out experiments/sweepladder_markers/across_L4A.json
Both regimes are assembled by phased_ga.episode_config_from_args from phased_ga's
OWN argument parser, so every knob the held-out report reads is phased_ga's
default rather than a hand copy that drifts. --dry-run prints both resolved
regimes and the matched winner files, then exits without loading anything.

WHAT THIS DOES *NOT* DO — replay cannot reproduce the study's logged triples, and
that is inherent, not a bug (resolved 29/07/2026).

Controller folds ACCUMULATE: evaluate_for_adaptation writes trained cells back onto
the genome (evaluator.py:1577), so a genome's cells keep gaining evidence as folds
are consumed. A generation's logged metric is therefore measured at THAT generation's
partial cell state, while the winner file is written after every fold has been
accumulated. The saved artifact is a strictly better-trained controller than the one
any logged number describes. score_genomes itself writes nothing, so the replay is
faithful — it is faithful to a LATER state.

The evidence is directional, not noisy. Across six independent comparisons the replay
came out equal-or-better every time, never worse:
  s31337002 held-out : stable 87.0 = 87.0,  steady 3.4554 vs 3.47
  s31337003 held-out : stable 41.0 vs 39.0, steady 9.6468 vs 9.68
  s31337003 train-seed: stable 87.5 vs 85.5, err 3.18 vs 3.30 (5-fold mean)

Ruled out along the way: batch composition, rg_config.teacher, non-determinism,
fold index (swept 0-4), K=1 vs K=5, 100 vs 200 episodes, disturbance preset drift
(L2D still matches the run banner exactly), and genome identity (no member of the
saved 57 scores the logged triple at any fold). Two REAL bugs were found and fixed:
the episode config hardcoded body/yaw rates to 0.0 against phased_ga's 0.5/0.3, and
it scored best_genome where the study reports population[0].

CONSEQUENCE FOR REPORTING: these numbers describe the SAVED controller — the artifact
you would deploy — not the row in the study table. Both are legitimate; they are
different quantities and must never be pooled or presented as the same measurement.
Cross-cell comparisons within this script's own output are valid, since every cell is
replayed at the same point in its lifecycle.

Usage:
  rescore_winners.py --glob 'logs/controller/dfa1l/*_winner.yaml.gz' \\
      --report-seeds 99990101 99990102 99990103 99990104 99990105 \\
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
from dataclasses import dataclass

_TAG = re.compile(r"^(?P<sub>[^_]+)_(?P<feat>[^_]+)_(?P<mode>[^_]+)_s(?P<seed>\d+)$")
_SEED = re.compile(r"_s(?P<seed>\d{5,})")
# The four-column surface (stable/err/steady/alt): metric attr → column.
COLUMNS = (("acc", "stable"), ("mean_attitude_error_deg", "err"),
           ("mean_steady_error_deg", "steady"), ("mean_altitude_error_m", "alt"))
DISTURBANCES = ["OFF", "L4A", "L4B", "L4C", "L1", "L2", "L3", "L2D", "L3D"]


def _parse_tag(path):
	"""Winner path → {tag, seed, sub, feat, mode}. The dfa1l cell pattern
	('dfa_9feat_QUAD_s31337002') fills the cell coordinates; any other tag with an
	'_s<seed>' token (the sweep-ladder 'SL_C_..._s31337002_hd29') keeps them None."""
	tag = os.path.basename(path).replace("_winner.yaml.gz", "")
	m = _TAG.match(tag)
	if m:
		d = m.groupdict()
	else:
		hit = _SEED.search(tag)
		if not hit:
			return None
		d = {"sub": None, "feat": None, "mode": None, "seed": hit.group("seed")}
	d["tag"], d["seed"] = tag, int(d["seed"])
	return d


@dataclass(frozen=True)
class Condition:
	"""One plant regime: the disturbance rung, the airframe, and whether the
	vertical channel is integrated. `None` airframe = phased_ga's legacy plant."""
	disturbance: str
	airframe: str | None
	translation: bool

	def label(self):
		return "%s / %s / %s" % (self.disturbance, self.airframe or "legacy-plant",
		                         "translation" if self.translation else "attitude-only")

	def argv(self):
		out = ["--disturbance", self.disturbance]
		if self.airframe:
			out += ["--airframe", self.airframe]
		if self.translation:
			out.append("--translation")
		return out


@dataclass(frozen=True)
class Regime:
	"""A Condition resolved through phased_ga: its parsed namespace + EpisodeConfig."""
	cond: Condition
	args: object
	ec: object

	@property
	def report_episodes(self):
		return self.args.report_episodes or self.args.eval_episodes


def _regime(cond, a):
	"""Resolve a Condition EXACTLY as phased_ga would: its own parser supplies every
	default the held-out report reads, episode_config_from_args assembles the plant."""
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	argv = cond.argv() + ["--steps", str(a.steps), "--tilt", str(a.tilt),
	                      "--body-rate", str(a.body_rate), "--yaw-rate", str(a.yaw_rate),
	                      "--report-episodes", str(a.episodes), "--num-eval-folds", "5"]
	args = build_arg_parser().parse_args(argv)
	ec = episode_config_from_args(args)
	if ec.translation and ec.airframe is None:   # phased_ga main()'s stage-1 guard
		raise SystemExit("--translation requires --airframe: mass is a PLANT parameter "
		                 "and the synthetic default has none.")
	return Regime(cond, args, ec)


def _pick(override, default):
	return default if override is None else override


def _conditions(a):
	"""(scoring, trained): the trained regime defaults field-wise to the scoring one."""
	score = Condition(a.disturbance, a.airframe, bool(a.translation))
	trained = Condition(_pick(a.trained_disturbance, score.disturbance),
	                    _pick(a.trained_airframe, score.airframe),
	                    bool(_pick(a.trained_translation, score.translation)))
	return score, trained


def _fit_thresholds(regime, spec, report_seed, train_seed, use_score):
	"""phased_ga's report-threshold fit (calibration ec included) on this regime.
	use_score=True → fit ONCE on the train seed (the address function the cells were
	written under); False → re-fit on the report seed (the per_seed diagnostic)."""
	from wnn.control.phased_ga import _report_thresholds
	return _report_thresholds(regime.args, regime.ec, spec, report_seed, train_seed, use_score)


def _score_once(spec, genome, score, report_seed, thresholds):
	"""One frozen genome, one report seed → the scorer's metrics object. This is the
	evaluator phased_ga._holdout_report builds, knob for knob (report episodes,
	_rg_config, K=5, train workers) — score-only, no training, nothing written back."""
	from wnn.control.evaluator import ControllerEvaluator
	from wnn.control.phased_ga import _rg_config
	ev = ControllerEvaluator(spec, num_eval_episodes=score.report_episodes,
	                         seed=report_seed, episode_config=score.ec, thresholds=thresholds,
	                         rg_config=_rg_config(score.args, score.ec, report_seed),
	                         max_train_workers=score.args.train_workers,
	                         num_eval_folds=score.args.num_eval_folds)
	return ev.score_genomes([genome])[0]


def _row(m):
	"""metrics object → the four-column surface (stable in %, absent metrics None —
	never 0: a zero altitude reads as a perfect hold, the opposite of not measured)."""
	out = {}
	for attr, col in COLUMNS:
		v = getattr(m, attr, None)
		absent = v is None or (isinstance(v, float) and math.isnan(v))
		out[col] = None if absent else (v * 100.0 if col == "stable" else float(v))
	return out


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


def _agg(rows):
	"""[{col: value|None}, ...] → {col: [mean, sd]} over the seeds that measured it."""
	out = {}
	for _, col in COLUMNS:
		xs = [r[col] for r in rows if r[col] is not None]
		if xs:
			out[col] = [statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0]
	return out


def _rescore_cell(path, meta, a, score, trained):
	"""Both threshold variants for one winner, across every report seed."""
	from wnn.seeds import resolve_seed_set
	loaded = _load_winner(path)
	if loaded is None:
		print(f"  SKIP {meta['tag']} — no trained cells in winner (arch-only)")
		return None
	spec, genome = loaded
	train_seed = resolve_seed_set(base=meta["seed"], run_index=0).train
	# variant 'train': fit ONCE, on the train seed, under the TRAINED regime.
	fixed = _fit_thresholds(trained, spec, a.report_seeds[0], train_seed, use_score=True)
	runs = {"per_seed": [], "train": []}
	for rs in a.report_seeds:
		refit = _fit_thresholds(score, spec, rs, train_seed, use_score=False)
		runs["per_seed"].append(_row(_score_once(spec, genome, score, rs, refit)))
		runs["train"].append(_row(_score_once(spec, genome, score, rs, fixed)))
	return {"tag": meta["tag"], "substrate": meta["sub"], "feature": meta["feat"],
	        "mode": meta["mode"], "seed": meta["seed"], "train_seed": train_seed,
	        "report_seeds": a.report_seeds,
	        "per_seed": {"agg": _agg(runs["per_seed"]), "runs": runs["per_seed"]},
	        "train": {"agg": _agg(runs["train"]), "runs": runs["train"]}}


def _cell(v, key):
	m = v["agg"].get(key)
	return f"{m[0]:5.1f}±{m[1]:4.1f}" if m else "   —   "


def _print_row(r):
	ps, tr = r["per_seed"], r["train"]
	cols = [_cell(tr, c) for _, c in COLUMNS] + ["|"] + [_cell(ps, c) for _, c in COLUMNS]
	print(f"  {r['tag']:48} " + " ".join(f"{c:>11}" if c != "|" else c for c in cols))


def _print_table(results, a, score, trained):
	print("=" * 150)
	print(f"  WINNER RE-SCORE — frozen winners replayed on {len(a.report_seeds)} report "
	      f"seeds (TEST-SET variance: the baselines' axis)")
	print(f"  seeds {a.report_seeds} | {a.episodes} ep x {a.steps} steps | tilt {a.tilt}°")
	print(f"  scored under : {score.cond.label()}")
	print(f"  trained under: {trained.cond.label()}  (fixes the 'train' address function)")
	print("=" * 150)
	heads = [f"{c}" for _, c in COLUMNS]
	print(f"  {'cell':48} {'--- thresholds fit ONCE on TRAIN seed / trained regime ---':>47}  "
	      f"{'--- re-fit PER REPORT SEED / scoring regime ---':>47}")
	print(f"  {'':48} " + " ".join(f"{h:>11}" for h in heads) + " | "
	      + " ".join(f"{h:>11}" for h in heads))
	print("  " + "-" * 146)
	for r in results:
		_print_row(r)
	print("=" * 150)


def _payload(results, a, score, trained):
	"""The on-disk document. Built fresh each flush so a partial file is a valid one."""
	return {"meta": {"report_seeds": a.report_seeds, "episodes": a.episodes,
	                 "steps": a.steps, "tilt_deg": a.tilt,
	                 "disturbance": a.disturbance,
	                 "scored_under": vars(score.cond), "trained_under": vars(trained.cond),
	                 "complete": False,
	                 "variance_note": "these ±SD are TEST-SET variance (frozen "
	                                  "winner, different held-out draws) — the "
	                                  "same axis as the classical baselines, NOT "
	                                  "the training-seed ±SD in the study table."},
	        "cells": results}


def _flush(results, a, score, trained, complete):
	"""Write after EVERY cell, atomically: a kill at cell 11 must not cost cells 1-10."""
	doc = _payload(results, a, score, trained)
	doc["meta"]["complete"] = complete
	tmp = a.out + ".tmp"
	with open(tmp, "w") as f:
		json.dump(doc, f, indent=1)
	os.replace(tmp, a.out)


def _describe(regime, role):
	ec = regime.ec
	af = ec.airframe
	print(f"  {role:<8} {regime.cond.label()}")
	print(f"           disturbance={regime.args.disturbance} airframe="
	      f"{af.name if af is not None else None}"
	      + (f" (mass {af.mass} kg)" if af is not None else "")
	      + f" translation={ec.translation}")
	print(f"           steps={ec.steps_per_episode} tilt={math.degrees(ec.max_initial_tilt_rad):.1f}° "
	      f"body_rate={ec.max_initial_body_rate} yaw_rate={ec.max_initial_yaw_rate} "
	      f"report_episodes={regime.report_episodes} folds={regime.args.num_eval_folds}")
	if ec.translation:
		print(f"           alt_offset={ec.max_initial_alt_offset_m} init_vz={ec.max_initial_vz} "
		      f"collective_jitter={ec.collective_cmd_jitter} mass_jitter={ec.mass_jitter} "
		      f"target_altitude={ec.target_altitude}")


def _dry_run(a, score, trained, paths):
	print("DRY RUN — resolved configuration (nothing loaded, nothing scored)")
	_describe(score, "score:")
	_describe(trained, "trained:")
	print(f"  thresholds: 'train' variant fit ONCE on the train seed under the TRAINED regime; "
	      f"'per_seed' re-fit per report seed under the SCORING regime")
	print(f"  report seeds: {a.report_seeds}")
	print(f"  out: {a.out}")
	print(f"  winners matched by {a.glob!r}: {len(paths)}")
	for p in paths:
		meta = _parse_tag(p)
		print(f"    {p}  → " + (f"tag={meta['tag']} seed={meta['seed']}" if meta else "UNPARSEABLE (skipped)"))


def _build_parser():
	from wnn.control.airframe import Airframe
	ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
	# The SCORING regime. Choices mirror phased_ga's own --disturbance/--airframe.
	ap.add_argument("--disturbance", default="L2D", choices=DISTURBANCES)
	ap.add_argument("--airframe", default=None, choices=Airframe.names(),
	                help="airframe preset the winner is SCORED on; omit for the legacy plant")
	ap.add_argument("--translation", action=argparse.BooleanOptionalAction, default=False,
	                help="integrate the vertical channel while scoring (requires --airframe)")
	# The TRAINED regime — the winner's address function. Each defaults to the
	# scoring value (plain replay); override the fields that differ for A-cross.
	ap.add_argument("--trained-disturbance", default=None, choices=DISTURBANCES,
	                help="disturbance the winner was TRAINED under (default: --disturbance)")
	ap.add_argument("--trained-airframe", default=None, choices=Airframe.names(),
	                help="airframe the winner was TRAINED on (default: --airframe)")
	ap.add_argument("--trained-translation", action=argparse.BooleanOptionalAction, default=None,
	                help="whether the winner was TRAINED with translation (default: --translation)")
	ap.add_argument("--dry-run", action="store_true",
	                help="print the resolved regimes + matched winners and exit")
	ap.add_argument("--out", required=True)
	return ap


def main():
	a = _build_parser().parse_args()
	score_cond, trained_cond = _conditions(a)
	score, trained = _regime(score_cond, a), _regime(trained_cond, a)
	paths = sorted(globmod.glob(a.glob))
	if a.dry_run:
		_dry_run(a, score, trained, paths)
		return 0
	results = []
	for path in paths:
		meta = _parse_tag(path)
		if meta is None:
			print(f"  SKIP {path} — unparseable tag")
			continue
		print(f"  scoring {meta['tag']} ...", flush=True)
		r = _rescore_cell(path, meta, a, score, trained)
		if r:
			results.append(r)
			_flush(results, a, score, trained, complete=False)
			_print_row(r)
	if not results:
		print("no winners scored", file=sys.stderr)
		return 1
	_flush(results, a, score, trained, complete=True)
	_print_table(results, a, score, trained)
	print(f"# wrote {a.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
