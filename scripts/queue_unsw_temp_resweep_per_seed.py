"""Reorder the in-flight UNSW-temp re-sweep into PER-SEED rounds.

User-requested 29/05/2026: instead of the default queue order (which picks
flows roughly width-by-width descending), interleave so we get a COMPLETE
(5 widths × 3 weights) picture at one seed first, then a second seed, then
the third. This lets us drop clearly-losing cells after just 1/3 of the
compute.

Round 1 (seed 88021, n=1): all 15 cells (5 widths × 3 weights) — runs first
Round 2 (seed 74627, n=2): all 15 again — runs second
Round 3 (seed 11760, n=3): all 15 again — runs third

Within each seed round, runs in Wc → Wb → Wa order (matches user
preference established earlier — CE-heavy first).

Already-completed cells (3 so far, all at 96b-Wc) are NOT re-queued.

Implementation:
  - id-DESC worker pickup means HIGHEST ids run first.
  - Queue order (lowest id first): round 3 (11760), then round 2 (74627),
    then round 1 (88021). So id-DESC picks seed 88021 first.
  - Within a seed batch (15 cells): queue with Wa first, then Wb, then Wc
    so Wc gets highest ids → runs first inside the batch.
  - Across widths within (seed, weight): queue ascending 8→96 so 96b runs
    first per id-DESC (continues the prior pattern, gets wide-first signal).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import urllib3
import requests

sys.path.insert(0, "scripts")
from queue_cross_dataset import BASE_PARAMS, EXPERIMENTS, DATASETS, DATASET_ARCH, API

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB = "/Users/lacg/wnn/db/wnn.db"

SEEDS_ORDER     = [11760, 74627, 88021]     # creation order; LAST = highest id = runs first
WEIGHTS_ORDER   = [("a", {"ce": 0.35, "acc": 0.30, "f1": 0.30, "fpr": 0.05}),
                   ("b", {"ce": 0.10, "acc": 0.20, "f1": 0.35, "fpr": 0.35}),
                   ("c", {"ce": 0.70, "acc": 0.10, "f1": 0.15, "fpr": 0.05})]
WIDTHS_ORDER    = [8, 16, 32, 64, 96]


def build_flow(n_bits: int, weight_key: str, weight_vals: dict, seed: int) -> dict:
	ds, split = DATASETS["unsw-temporal"]
	max_neurons, max_bits = DATASET_ARCH["unsw-temporal"]
	params = dict(BASE_PARAMS)
	params.update({
		"ids_dataset": ds, "ids_split": split, "ids_n_bits": n_bits, "seed": seed,
		"max_neurons": max_neurons, "max_bits": max_bits,
		"fitness_weight_ce":  weight_vals["ce"],
		"fitness_weight_acc": weight_vals["acc"],
		"fitness_weight_f1":  weight_vals["f1"],
		"fitness_weight_fpr": weight_vals["fpr"],
	})
	name = f"XDS-unsw-temporal-{n_bits}b-W{weight_key}-C35-{max_neurons}n{max_bits}b-OI-r{seed}"
	return {
		"name": name,
		"description": (
			f"UNSW-temp RE-sweep (PER-SEED ORDER) {n_bits}b W{weight_key}: "
			f"ce={weight_vals['ce']:.2f} acc={weight_vals['acc']:.2f} "
			f"f1={weight_vals['f1']:.2f} fpr={weight_vals['fpr']:.2f}."
		),
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": [dict(e) for e in EXPERIMENTS],
		"seed_checkpoint_id": None,
	}


def already_completed_cells():
	"""Return set of (width, weight_key, seed) cells already completed
	(or currently running) in the active sweep (no rename suffix)."""
	con = sqlite3.connect(DB)
	cur = con.cursor()
	cur.execute(
		"SELECT name, status FROM flows "
		"WHERE name LIKE 'XDS-unsw-temporal-%-W%-C35-500n34b-OI-r%' "
		"  AND name NOT LIKE '%PREEMP-OLD%' "
		"  AND name NOT LIKE '%REORDERED-OLD%' "
		"  AND status IN ('completed', 'running') "
		"  AND id >= 2846"
	)
	cells = set()
	import re
	pat = re.compile(r"XDS-unsw-temporal-(\d+)b-W([abc])-C35-500n34b-OI-r(\d+)$")
	for name, _ in cur.fetchall():
		m = pat.match(name)
		if m:
			cells.add((int(m.group(1)), m.group(2), int(m.group(3))))
	con.close()
	return cells


def cancel_and_rename_active_queued():
	"""Cancel + rename all currently-queued sweep flows so they don't collide
	with the re-queued ones. Marks them with -REORDERED-OLD- suffix."""
	con = sqlite3.connect(DB)
	con.execute(
		"UPDATE flows SET status='cancelled', "
		"name=REPLACE(name, '-OI-r', '-OI-REORDERED-OLD-r') "
		"WHERE name LIKE 'XDS-unsw-temporal-%-W%-C35-500n34b-OI-r%' "
		"  AND name NOT LIKE '%PREEMP-OLD%' "
		"  AND name NOT LIKE '%REORDERED-OLD%' "
		"  AND status='queued' AND id >= 2846"
	)
	con.commit()
	cancelled = con.execute(
		"SELECT COUNT(*) FROM flows WHERE name LIKE '%REORDERED-OLD%'"
	).fetchone()[0]
	con.close()
	return cancelled


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--execute", action="store_true",
	                help="Cancel old queued + POST the new flows (default is preview).")
	args = ap.parse_args()

	done = already_completed_cells()
	print(f"Already completed/running cells: {len(done)}")
	for cell in sorted(done):
		print(f"  {cell[0]:>3}b W{cell[1]}  r{cell[2]}")

	# Build the new flow list in queue-creation order.
	# Creation order (lowest id → highest id): seeds[::-1] = [11760, 74627, 88021]
	# Within each seed: weights in [(a, ..), (b, ..), (c, ..)] order; widths ascending.
	# Result: highest id = (seed=88021, Wc, 96b) → first to run per id-DESC.
	flows = []
	for seed in SEEDS_ORDER:                              # 11760, 74627, 88021
		for (wk, wv) in WEIGHTS_ORDER:                    # a, b, c
			for n_bits in WIDTHS_ORDER:                   # 8, 16, 32, 64, 96
				if (n_bits, wk, seed) in done:
					continue                              # skip cells already done
				flows.append(build_flow(n_bits, wk, wv, seed))

	# Print queue order. Run order is REVERSED (id-DESC).
	print(f"\nWill queue {len(flows)} flows (creation order; reversed for id-DESC runs):")
	print(f"  First-to-run (highest id, queued LAST): {flows[-1]['name']}")
	print(f"  Last-to-run  (lowest id,  queued FIRST): {flows[0]['name']}")
	print()

	# Show run-order summary by seed
	by_seed = {}
	for f in flows:
		seed = int(f['name'].split('-r')[-1])
		by_seed[seed] = by_seed.get(seed, 0) + 1
	print(f"Per-seed flow counts (run order = reverse of creation):")
	for s in [88021, 74627, 11760]:
		marker = "← runs FIRST" if s == 88021 else ("← runs MIDDLE" if s == 74627 else "← runs LAST")
		print(f"  seed {s}: {by_seed.get(s, 0)} flows  {marker}")

	if not args.execute:
		print(f"\nDRY-RUN — pass --execute to:")
		print(f"  1. Cancel + rename the currently-queued flows (-REORDERED-OLD- suffix)")
		print(f"  2. POST {len(flows)} new flows in per-seed order")
		return

	# Real execution.
	print(f"\nCancelling + renaming currently-queued flows...")
	cancelled = cancel_and_rename_active_queued()
	print(f"  Cancelled + renamed {cancelled} prior queued flows")

	print(f"\nPOSTing {len(flows)} new flows...")
	sess = requests.Session()
	sess.verify = False
	for f in flows:
		r = sess.post(API, json=f, timeout=30)
		if r.status_code not in (200, 201):
			print(f"  ERROR {f['name']}: {r.status_code} {r.text[:200]}")
			continue
		body = r.json()
		fid = body.get("id") or body.get("flow", {}).get("id")
		print(f"  queued flow {fid}: {f['name']}")

	# Promote pending → queued (the dashboard sometimes parks new flows as pending)
	con = sqlite3.connect(DB)
	con.execute(
		"UPDATE flows SET status='queued' "
		"WHERE name LIKE 'XDS-unsw-temporal-%-W%-C35-500n34b-OI-r%' "
		"  AND name NOT LIKE '%REORDERED-OLD%' AND name NOT LIKE '%PREEMP-OLD%' "
		"  AND status='pending'"
	)
	con.commit()
	con.close()
	print("Promoted pending → queued (if any).")


if __name__ == "__main__":
	main()
