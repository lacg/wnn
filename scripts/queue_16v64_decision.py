#!/usr/bin/env python3
"""Queue the 16b-vs-64b decision cohort: 5 fresh seeds × {16b, 64b} Wb, interleaved.

UNSW-random, 500n34b (C35), weight-set b. 10 flows total. Posted in (16b, 64b) pairs
per seed so that under the worker's id-DESC pickup the two widths stay interleaved
(never 5 of one width before the other) — enabling early culling.

DRY-RUN by default; pass --execute to POST. Reuses build_flow/post from
queue_cross_dataset (experiments + template included → Behavioral Rule 2 satisfied).
"""
import argparse, secrets, sys
from queue_cross_dataset import build_flow, post, DATASETS  # same dir on PYTHONPATH

DS = "unsw-random"
WEIGHT = "b"
WIDTHS = [16, 64]
ARCH = (500, 34)            # explicit — unsw-random has no DATASET_ARCH entry (defaults 250x100)
N_SEEDS = 5

# Avoid collision with existing cohort seeds.
USED = {25608, 8188, 82096, 14675, 52015, 25694, 88021, 74627, 11760, 73300,
        85011, 40417, 84446, 76297, 25737, 13922, 35141, 49994, 42823, 91849,
        27384, 13710, 41448, 99534, 27140, 56926, 87167, 75197, 3922, 69269,
        43097, 75840, 65268, 77567, 45199, 63480}


def fresh_seeds(n):
	out = []
	while len(out) < n:
		s = secrets.randbelow(100000)
		if s not in USED and s not in out:
			out.append(s)
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--execute", action="store_true", help="Actually POST (default: dry-run)")
	ap.add_argument("--seeds", type=int, nargs="*", default=None,
	                help="Explicit seeds (default: 5 fresh random, printed for the record)")
	args = ap.parse_args()

	seeds = args.seeds or fresh_seeds(N_SEEDS)
	mode = "EXECUTE" if args.execute else "DRY-RUN"
	print(f"[{mode}] 16b-vs-64b decision cohort — {DS} {ARCH[0]}n{ARCH[1]}b W{WEIGHT}")
	print(f"  seeds: {seeds}")
	print(f"  {len(seeds)} seeds × {WIDTHS} = {len(seeds)*len(WIDTHS)} flows, interleaved (16b,64b) per seed\n")

	# Interleave: per seed, queue 16b then 64b.
	flows = []
	for s in seeds:
		for w in WIDTHS:
			flows.append(build_flow(DS, w, WEIGHT, s, arch_override=ARCH))

	for f in flows:
		if args.execute:
			post(f, True)
		else:
			print(f"    would queue: {f['name']}")

	if not args.execute:
		print("\n  DRY-RUN — nothing queued. Re-run with --execute to POST.")


if __name__ == "__main__":
	main()
