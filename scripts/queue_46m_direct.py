#!/usr/bin/env python3
"""Queue 2 direct flows on the full 46M CIC-IoT-2023 dataset.

Differs from queue_cross_dataset.py (which is for probe→cohort cross-dataset
sweeps at 500n×34b):
- Architecture: 250n × 100b (matches the paper Table 5 Search(46M) entries
  and the 2747+2748 cohort that originally produced those numbers; large
  address space is fine on 46M because per-neuron coverage is dense).
- Weights: WEIGHTS_A (CE/Acc/F1/FPR = 0.35/0.30/0.30/0.05) — the
  native CIC-IoT weight set 2747+2748 used.
- 2 flows by default (matching 2747+2748). Configurable via --n for
  Plan B (+20) and Plan A (+50) iterations.
- No probe-cohort split: each flow is a full grid+GA run on 46M.

Naming: REPRO46M-T20-96b-C35-250n100b-OI-r{seed} — distinct from the
2747+2748 originals (WSWEEP-...) and from XDS-V2 (XDS-...).

Examples:
  python scripts/queue_46m_direct.py --n 2          # preview (dry-run)
  python scripts/queue_46m_direct.py --n 2 --execute
  python scripts/queue_46m_direct.py --n 20 --execute   # Plan B increment
  python scripts/queue_46m_direct.py --n 50 --execute   # Plan A increment
"""

from __future__ import annotations

import argparse
import secrets
import sys
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API = "https://localhost:3000/api/flows"

# CIC-IoT native weight set (CE-leading) — same as queue_cross_dataset.py's
# WEIGHTS_A and the original 2747+2748 cohort.
WEIGHTS = {"ce": 0.35, "acc": 0.30, "f1": 0.30, "fpr": 0.05}

# 250n × 100b matches 2747+2748 + the paper's Table 5 Search(46M) entries.
# Large address space is fine on 46M (dense per-neuron coverage). Switching
# to 500n × 34b is reserved for smaller datasets where 100b address space
# becomes too sparse (see queue_cross_dataset.py rationale).
BASE_PARAMS = {
	"ids_classification": "binary",
	"ids_feature_selection": "top20",
	"ids_dataset": "ciciot2023_neto_full",
	"ids_split": "random",
	"ids_n_bits": 96,
	"architecture_type": "ids",
	"min_neurons": 5, "max_neurons": 250,
	"min_bits": 4, "max_bits": 100,
	"population_size": 50, "ga_generations": 250, "patience": 5,
	"phase_order": "neurons_first",
	"fitness_calculator": "harmonic_rank",
	"ids_k_folds": 5, "ids_kfold_per_gen": 5, "ids_num_parts": 5,
	"ids_val_fraction": 0.25,
	"ids_single_cluster": True, "balance_classes": True,
	"neuron_sample_rate": 0.25,
	# 46M doesn't fit comfortably in RAM at 96b × top20 × 46M without
	# memmap — see 2747+2748 behavior. Worker auto-detects via 8GB threshold;
	# explicit memmap here is belt-and-suspenders.
	"ids_encoded_storage": "memmap",
	"min_accuracy_floor": 0, "pool_shuffle_ratio": 0.8,
	"adaptation_iterations": 50,
	"cluster_crossover_ratio": 0.5, "assortative_mating_ratio": 0.85,
	"neighbors_per_iter": 50, "context_size": 4,
	"threshold_start": 0, "threshold_step": 1, "fitness_percentile": 0.75,
	"wnn_order_independent_train": True,
	"fitness_weight_ce": WEIGHTS["ce"],
	"fitness_weight_acc": WEIGHTS["acc"],
	"fitness_weight_f1": WEIGHTS["f1"],
	"fitness_weight_fpr": WEIGHTS["fpr"],
}

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
	{"name": "GA Neurons",                   "phase_type": "ga_neurons",  "experiment_type": "ga"},
]


def build_flow(seed: int) -> dict:
	params = dict(BASE_PARAMS)
	params["seed"] = seed
	return {
		"name": f"REPRO46M-T20-96b-C35-250n100b-OI-r{seed}",
		"description": "CIC-IoT-2023 46M direct re-run (post offspring-eval + fitness_scores fix). Architecture matches paper Table 5 Search(46M).",
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": [dict(e) for e in EXPERIMENTS],
		"seed_checkpoint_id": None,
	}


def post(flow: dict, execute: bool) -> None:
	if not execute:
		return
	r = requests.post(API, json=flow, verify=False, timeout=30)
	r.raise_for_status()
	fid = r.json().get("id")
	# Flip pending → queued (POST /api/flows creates as pending by default).
	rr = requests.post(f"{API}/{fid}/restart", json={}, verify=False, timeout=15)
	rr.raise_for_status()
	print(f"    queued flow {fid}: {flow['name']}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--n", type=int, default=2,
		help="Number of 46M flows to queue (default 2 = Plan C; 20 = Plan B increment; 50 = Plan A increment)")
	ap.add_argument("--seeds", type=int, nargs="*", default=None,
		help="Explicit seeds (default: random; printed for the record)")
	ap.add_argument("--execute", action="store_true",
		help="Actually POST (default: dry-run preview)")
	args = ap.parse_args()

	if args.seeds:
		if len(args.seeds) != args.n:
			sys.exit(f"--seeds count {len(args.seeds)} != --n {args.n}")
		seeds = args.seeds
	else:
		seeds = [secrets.randbelow(100000) for _ in range(args.n)]

	mode = "EXECUTE" if args.execute else "DRY-RUN"
	print(f"[{mode}] CIC-IoT 46M direct — {args.n} flows at 250n×100b (seeds: {seeds})")
	for s in seeds:
		flow = build_flow(s)
		print(f"  {flow['name']}")
		post(flow, args.execute)

	if not args.execute:
		print("\n  DRY-RUN — nothing queued. Re-run with --execute to POST.")


if __name__ == "__main__":
	sys.exit(main())
