"""Re-queue the UNSW-temp probe sweep across 3 weight sets — POST empirical-fix.

Why re-sweep now (29/05/2026):
  * `fit_empirical_threshold` brittleness fixed (min_bin_size=200 default;
    see project_empirical_brittleness_fix memory). Empirical-mode F1/FPR
    numbers stored from prior cohorts used the old noise-prone rule, so
    a fresh sweep gives the first apples-to-apples post-fix dataset.
  * Adds a third weight set Wc (CE-heavy) to test the user's hypothesis
    that the pre-fix tournament_select=CE-only bug had a "lucky bias"
    that today's harmonic-rank-based selection could re-create by
    explicitly setting w_CE high.
  * The dual-bug fix (offspring-eval-test-set + tournament_select) is
    now also in effect, so the GA's per-generation selection pressure
    is genuinely balanced — earlier sweeps had hidden CE-bias even when
    the configured weights weren't CE-heavy.

Design:
  - Widths:   8, 16, 32, 64, 96 bits (5 widths — matches prior probe)
  - Weights:
      Wa (CIC-IoT, CE-leaning):   ce 0.35  acc 0.30  f1 0.30  fpr 0.05
      Wb (paper "balanced"):      ce 0.10  acc 0.20  f1 0.35  fpr 0.35
      Wc (CE-heavy NEW):          ce 0.70  acc 0.10  f1 0.15  fpr 0.05
  - Seeds:    88021, 74627, 11760 (same trio as prior probe — width/weight
              becomes the only variable; matched splits)
  - Arch:     500n × 34b OI/QUAD (same as Cohort 2; UNSW-temp is too small
              for 250n × 100b)
  - Naming:   XDS-unsw-temporal-{N}b-W{a|b|c}-C35-500n34b-OI-r{seed}
              The naming pattern is grep-compatible with prior probe runs;
              older runs with the same name (pre-fix) are renamed to
              -PREFIX-OLD- or already cancelled.

Total: 5 × 3 × 3 = 45 new flows.

Usage:
  python scripts/queue_unsw_temp_resweep.py            # dry-run preview
  python scripts/queue_unsw_temp_resweep.py --execute  # POST to dashboard
"""
from __future__ import annotations

import argparse
import sys
import urllib3
import requests

sys.path.insert(0, "scripts")
from queue_cross_dataset import BASE_PARAMS, EXPERIMENTS, DATASETS, DATASET_ARCH, API

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SEEDS = [88021, 74627, 11760]
WIDTHS = [8, 16, 32, 64, 96]
WEIGHT_SETS = {
	"a": {"ce": 0.35, "acc": 0.30, "f1": 0.30, "fpr": 0.05},      # CIC-IoT (CE-leaning)
	"b": {"ce": 0.10, "acc": 0.20, "f1": 0.35, "fpr": 0.35},      # paper "balanced"
	"c": {"ce": 0.70, "acc": 0.10, "f1": 0.15, "fpr": 0.05},      # CE-heavy NEW
}


def build_flow(n_bits: int, weight_key: str, seed: int) -> dict:
	w = WEIGHT_SETS[weight_key]
	ds, split = DATASETS["unsw-temporal"]
	max_neurons, max_bits = DATASET_ARCH["unsw-temporal"]
	params = dict(BASE_PARAMS)
	params.update({
		"ids_dataset": ds, "ids_split": split, "ids_n_bits": n_bits, "seed": seed,
		"max_neurons": max_neurons, "max_bits": max_bits,
		"fitness_weight_ce":  w["ce"],
		"fitness_weight_acc": w["acc"],
		"fitness_weight_f1":  w["f1"],
		"fitness_weight_fpr": w["fpr"],
	})
	name = f"XDS-unsw-temporal-{n_bits}b-W{weight_key}-C35-{max_neurons}n{max_bits}b-OI-r{seed}"
	return {
		"name": name,
		"description": (
			f"UNSW-temp RE-sweep ({n_bits}b thermo, W{weight_key}): "
			f"post-empirical-fix; ce={w['ce']:.2f} acc={w['acc']:.2f} "
			f"f1={w['f1']:.2f} fpr={w['fpr']:.2f}."
		),
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": [dict(e) for e in EXPERIMENTS],
		"seed_checkpoint_id": None,
	}


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--execute", action="store_true",
	                help="POST the flows (default is dry-run preview).")
	args = ap.parse_args()

	flows = []
	for n in WIDTHS:
		for wk in ("a", "b", "c"):
			for s in SEEDS:
				flows.append(build_flow(n, wk, s))

	mode = "DRY-RUN" if not args.execute else "EXECUTING"
	print(f"{mode}: UNSW-temp RE-sweep — {len(flows)} flows "
	      f"({len(WIDTHS)} widths × 3 weight sets × {len(SEEDS)} seeds)")
	print(f"  widths:   {WIDTHS}b")
	print(f"  weights:  Wa {WEIGHT_SETS['a']}")
	print(f"            Wb {WEIGHT_SETS['b']}  ← paper")
	print(f"            Wc {WEIGHT_SETS['c']}  ← NEW (CE-heavy)")
	print(f"  seeds:    {SEEDS}  (same as prior probe)")
	print(f"  arch:     500n × 34b OI (matches paper / Cohort 2)")
	print()

	# Brief preview: count per (width, weight) group
	print(f"  Per (width, weight): 3 flows each. Total: {len(flows)}.")
	if not args.execute:
		print(f"\nDRY-RUN — pass --execute to POST to {API}.")
		print(f"First flow body sample:")
		import json
		print(json.dumps(flows[0]["config"]["params"], indent=2)[:600] + "...")
		return

	sess = requests.Session()
	sess.verify = False
	for f in flows:
		r = sess.post(API, json=f, timeout=30)
		if r.status_code not in (200, 201):
			print(f"  ERROR queueing {f['name']}: {r.status_code} {r.text[:200]}")
			continue
		body = r.json()
		fid = body.get("id") or body.get("flow", {}).get("id")
		print(f"  queued flow {fid}: {f['name']}")


if __name__ == "__main__":
	main()
