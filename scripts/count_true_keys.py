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
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_controller_c import true_onset  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(ROOT, 'experiments', 'h743_keys.json')
WINNER_SUFFIX = '_winner.yaml.gz'


def parse_args():
	ap = argparse.ArgumentParser(description="Count TRUE-only keys of controller winners")
	ap.add_argument('--winner', action='append', required=True,
	                help='winner checkpoint (<tag>_winner.yaml.gz); repeatable')
	ap.add_argument('--cache', default=DEFAULT_CACHE)
	return ap.parse_args()


def tag_of(winner: str) -> str:
	base = os.path.basename(winner)
	if not base.endswith(WINNER_SUFFIX):
		raise SystemExit(f"{winner}: expected a <tag>{WINNER_SUFFIX} file")
	return base[:-len(WINNER_SUFFIX)]


def count_one(winner: str) -> dict:
	m = true_onset(winner)
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
		source=os.path.relpath(winner, ROOT),
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


def main():
	args = parse_args()
	cache = load_cache(args.cache)
	for w in args.winner:
		tag = tag_of(w)
		e = count_one(w)
		cache[tag] = e
		save_cache(args.cache, cache)  # after EVERY winner: a crash mid-list loses nothing
		print(f"{tag}: TRUE keys={e['true_keys']} (populated {e['populated']}, "
		      f"{e['neurons']}n x {e['bits']}b)  uint32 {e['bytes_uint32'] / 1024:.0f} KB  "
		      f"packed {e['bytes_packed'] / 1024:.0f} KB")
	print(f"cache: {args.cache} ({len(cache)} winners)")


if __name__ == '__main__':
	sys.exit(main())
