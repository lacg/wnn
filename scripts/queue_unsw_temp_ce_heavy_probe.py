"""One-off probe: UNSW-temp at CE-heavy weights, asking whether the paper's
Table 1 "Best F1‡ = 90.52 @ FPR 4.13%" requires a much heavier CE pressure
than the Cohort 2 Wa (ce=0.35) we're currently running.

Hypothesis: the paper's tight FPR came from a CE-leading weight set that
pushes the GA's calibrated thresholds tighter than the more balanced Wa
weights do. The Cohort 2 32b-Wa probe topped out at F1=89.46 @ FPR=9.21%
— ~2× wider FPR than the paper. CE-heavy weights might close that gap.

Design (29/05/2026):
  - 3 thermo widths × 3 shared seeds = 9 flows
  - Widths: 8b (paper-extreme narrow), 32b (Cohort 2 winner), 96b (Cohort
    2 widest probe). Skips 16/64 to keep the cost down; this is a
    DIAGNOSTIC probe, not a replacement cohort.
  - Weights: ce=0.7, acc=0.1, f1=0.15, fpr=0.05 (CE 2× heavier than Wa,
    drops acc/f1 substantially; keeps fpr light so it's the GA's own
    ratchet via CE that drives the calibrated threshold tightening).
  - Same architecture as Cohort 2: 500n × 34b OI, ids_n_bits per probe,
    K-fold=5 with neuron_sample_rate=0.25.
  - Naming: XDS-unsw-temporal-{N}b-WCEh-C35-500n34b-OI-r{seed}
    'WCEh' = "Weights CE-heavy" — keeps it filterable from Wa / Wb cohorts.

Usage:
  python scripts/queue_unsw_temp_ce_heavy_probe.py             # dry-run
  python scripts/queue_unsw_temp_ce_heavy_probe.py --execute   # POST
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib3
import requests

# Reuse the shared probe infrastructure from the cross-dataset script
# (BASE_PARAMS, EXPERIMENTS, DATASETS, DATASET_ARCH).
sys.path.insert(0, "scripts")
from queue_cross_dataset import BASE_PARAMS, EXPERIMENTS, DATASETS, DATASET_ARCH, API

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3 fixed seeds — shared across widths so width is the only variable.
SEEDS = [88021, 74627, 11760]                     # same trio as the Wa/Wb probes
WIDTHS = [8, 32, 96]                              # narrow / Cohort 2 winner / wide
WEIGHTS_CE_HEAVY = {"ce": 0.70, "acc": 0.10, "f1": 0.15, "fpr": 0.05}


def build_flow(n_bits: int, seed: int) -> dict:
	ds, split = DATASETS["unsw-temporal"]
	max_neurons, max_bits = DATASET_ARCH["unsw-temporal"]   # 500, 34
	params = dict(BASE_PARAMS)
	params.update({
		"ids_dataset": ds, "ids_split": split, "ids_n_bits": n_bits, "seed": seed,
		"max_neurons": max_neurons, "max_bits": max_bits,
		"fitness_weight_ce":  WEIGHTS_CE_HEAVY["ce"],
		"fitness_weight_acc": WEIGHTS_CE_HEAVY["acc"],
		"fitness_weight_f1":  WEIGHTS_CE_HEAVY["f1"],
		"fitness_weight_fpr": WEIGHTS_CE_HEAVY["fpr"],
	})
	name = f"XDS-unsw-temporal-{n_bits}b-WCEh-C35-{max_neurons}n{max_bits}b-OI-r{seed}"
	return {
		"name": name,
		"description": (
			f"UNSW-temp CE-heavy probe ({n_bits}b thermo): ce=0.70/acc=0.10/f1=0.15/"
			f"fpr=0.05 — does the paper Table 1 FPR 4.13% come from CE-heavy weights?"
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

	flows = [build_flow(n, s) for n in WIDTHS for s in SEEDS]
	print(f"{'DRY-RUN' if not args.execute else 'EXECUTING'}: UNSW-temp CE-heavy probe — "
	      f"{len(flows)} flows ({len(WIDTHS)} widths × {len(SEEDS)} seeds)")
	print(f"  weights: {WEIGHTS_CE_HEAVY}")
	print(f"  widths:  {WIDTHS}b   seeds: {SEEDS}")
	print(f"  arch:    500n × 34b OI (matches Cohort 2)")
	print()
	for f in flows:
		print(f"  {f['name']}")

	if not args.execute:
		print(f"\nDRY-RUN — pass --execute to POST {len(flows)} flows to {API}.")
		return

	# Real POST.
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
