#!/usr/bin/env python3
"""Count a controller winner's TRUE-only keys and record them for the leaderboard.

WHY A CACHE. The deployability rule (docs/chip_selection.md, "Recipe constraint") is
"fits the STM32H743's 2 MB internal flash as TRUE-only sorted keys". The key count
is NOT in the marker: `populated` counts FALSE cells too (up to ~10-18% of a
winner), so the only honest number comes from loading the winner checkpoint —
70-370 MB gzipped, minutes each. This script does that once per winner and
appends the result to experiments/h743_keys.json; gate_distance_leaderboard.py
reads the cache and prints `h743` exactly for cached runs, as a populated-based
upper bound for the rest.

The counting is scripts/export_controller_c.py's own `true_onset`, so the number
here is the number the shipped header would carry.

Usage:
  PYTHONPATH=src/wnn python scripts/count_true_keys.py \\
      --winner logs/controller/sweep_ladder/<tag>_winner.yaml.gz [--winner ...]
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_controller_c import true_onset, true_onset_genome  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(ROOT, 'experiments', 'h743_keys.json')
WINNER_SUFFIX = '_winner.yaml.gz'


def parse_args():
	ap = argparse.ArgumentParser(description="Count TRUE-only keys of controller winners")
	ap.add_argument('--winner', action='append', default=[],
	                help='winner checkpoint (<tag>_winner.yaml.gz); repeatable')
	ap.add_argument('--headline', action='append', default=[],
	                help='sweep-ladder marker json; counts the stage-select HEADLINE genome '
	                     '(STAGE#i = final_population[i] of that stage checkpoint), cached as <tag>@headline')
	ap.add_argument('--retrain-args', default=None,
	                help="the run's exact phased_ga argv file ('#' lines ignored). With --headline, an arch-only "
	                     "(GRID) headline is trained EXACTLY as the scorer does (phased_ga._holdout_report: train "
	                     "seed, write_back) and scored on the first report seed; compare that RESULT row with the "
	                     "run's logged HEADLINE row before trusting the count")
	ap.add_argument('--ckpt-root', default=os.path.join(ROOT, 'logs', 'controller', 'sweep_ladder', 'ckpt'))
	ap.add_argument('--cache', default=DEFAULT_CACHE)
	return ap.parse_args()


def tag_of(winner: str) -> str:
	base = os.path.basename(winner)
	if not base.endswith(WINNER_SUFFIX):
		raise SystemExit(f"{winner}: expected a <tag>{WINNER_SUFFIX} file")
	return base[:-len(WINNER_SUFFIX)]


def count_one(winner: str) -> dict:
	return entry_from_onset(true_onset(winner), winner)


def entry_from_onset(m: dict, source: str) -> dict:
	keys, conn = len(m['keys']), len(m['conn'])
	return dict(
		true_keys=keys,
		populated=m['n_true'] + m['n_false'],
		neurons=m['neurons'],
		bits=m['bits'],
		# What export_controller_c.py ships: uint32 keys + uint8 connectivity.
		bytes_uint32=keys * 4 + conn,
		# Tight packing at exactly `bits` per key — the floor for a plain sorted array.
		bytes_packed=(keys * m['bits'] + 7) // 8 + conn,
		source=os.path.relpath(source, ROOT),
		counted=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
	)


def load_cache(path: str) -> dict:
	if not os.path.exists(path):
		return {}
	with open(path) as f:
		return json.load(f)


def save_cache(path: str, cache: dict) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as f:
		json.dump(dict(sorted(cache.items())), f, indent=1, sort_keys=True)
		f.write('\n')


HEADLINE_SUFFIX = '@headline'


def headline_label(marker: dict) -> "tuple[str, int]":
	"""(stage name, population index) of the marker's stage-select headline.
	`genome=CONNECTIONS#1` -> ('connections', 1); a bare label is index 0."""
	m = re.search(r'genome=([A-Z]+)(?:#(\d+))?', marker.get('headline_stage') or '')
	if m is None:
		raise SystemExit(f"{marker.get('tag')}: no stage-select headline in marker")
	return m.group(1).lower(), int(m.group(2) or 0)


def count_headline(marker_path: str, ckpt_root: str, retrain_args: "str | None") -> "tuple[str, dict]":
	from wnn.control.checkpoint_io import load_controller_population_member
	with open(marker_path) as f:
		marker = json.load(f)
	tag = marker['tag']
	stage, idx = headline_label(marker)
	hits = glob.glob(os.path.join(ckpt_root, tag, f'stage[0-9]_{stage}.yaml.gz'))
	if len(hits) != 1:
		raise SystemExit(f"{tag}: expected one stage*_{stage}.yaml.gz, found {hits}")
	got = load_controller_population_member(hits[0], idx)
	label = f"{stage.upper()}#{idx}"
	if got['genome'].cells is None:
		if retrain_args is None:
			raise SystemExit(f"{tag}: headline {label} is arch-only (no stored cells) — its memory exists "
			                 f"only when the scorer trains it; pass --retrain-args <run argv file>")
		retrain_like_scorer(got['genome'], got['spec'], retrain_args, label)
	e = entry_from_onset(true_onset_genome(got['genome'], got['spec']), hits[0])
	e['genome'] = label
	if retrain_args is not None:
		e['retrained_from_args'] = os.path.relpath(retrain_args, ROOT)
	return tag + HEADLINE_SUFFIX, e


def retrain_like_scorer(genome, spec, argv_path: str, label: str) -> None:
	"""Stamp the cells the stage-select scorer gave an arch-only headline: the SAME
	_holdout_report call (train on the train seed, write_back), scored on the first
	report seed so its RESULT row can be diffed against the run's log."""
	import shlex
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args, _holdout_report
	from wnn.seeds import resolve_seed_set
	with open(argv_path) as f:
		argv = shlex.split(" ".join(l for l in f.read().splitlines() if not l.lstrip().startswith("#")))
	args = build_arg_parser().parse_args(argv)
	ec = episode_config_from_args(args)
	train = resolve_seed_set(base=args.base_seed, run_index=0).train
	_holdout_report(args, ec, spec, genome, None, int(args.report_seeds[0]), train,
	                stage_label=f"HEADLINE-{label} (re-count)")
	if genome.cells is None:
		raise SystemExit(f"{label}: scorer left no cells after training")


def record(cache: dict, path: str, key: str, e: dict) -> None:
	cache[key] = e
	save_cache(path, cache)  # after EVERY entry: a crash mid-list loses nothing
	print(f"{key}: TRUE keys={e['true_keys']} (populated {e['populated']}, "
	      f"{e['neurons']}n x {e['bits']}b)  uint32 {e['bytes_uint32'] / 1024:.0f} KB  "
	      f"packed {e['bytes_packed'] / 1024:.0f} KB")


def main():
	args = parse_args()
	if not args.winner and not args.headline:
		raise SystemExit("give --winner and/or --headline")
	cache = load_cache(args.cache)
	for w in args.winner:
		record(cache, args.cache, tag_of(w), count_one(w))
	for mk in args.headline:
		key, e = count_headline(mk, args.ckpt_root, args.retrain_args)
		record(cache, args.cache, key, e)
	print(f"cache: {args.cache} ({len(cache)} entries)")


if __name__ == '__main__':
	sys.exit(main())
