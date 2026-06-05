#!/usr/bin/env python3
"""Queue the cross-dataset OI cohorts (UNSW-temporal, UNSW-random, CICIDS-random).

Two phases per dataset, run sequentially dataset-by-dataset:

  Phase 1 (probe): widths {8,16,32,64,96}b × 3 shared seeds × 2 weight sets = 30 flows.
      The 3 seeds are SHARED across all 30 so width/weight is the only variable
      (matched splits). Answers "which thermo width + weight is best for this dataset".
  Phase 2 (cohort): 97 NEW-seed flows at the chosen (width, weight); together with the
      3 renamed probe winners that = a 100-flow cohort (saves re-running the winners).

Config = the canonical CIC-IoT OI cohort (250n×100b, OI/QUAD, top20, harmonic_rank,
K-fold 5×5, neuron_sample_rate 0.25, patience 5) with dataset/split/width/weights swapped.
Non-`_3way` splits + K-fold=5 (per the methodology decision: test+val merge, K-fold on
the 80% train) — matches how CIC-IoT was run.

Naming: XDS-{ds}-{split}-{N}b-W{a|b}-C35-250n100b-OI-r{seed}  (width/weight/ds/split parseable).
Seeds are passed explicitly so they're recorded in `seed_runs` (the dashboard also records
every flow's seed now). DRY-RUN by default — pass --execute to actually POST.

Examples:
  python scripts/queue_cross_dataset.py --dataset unsw-temporal --phase probe          # preview 30
  python scripts/queue_cross_dataset.py --dataset unsw-temporal --phase probe --execute
  python scripts/queue_cross_dataset.py --dataset cicids-random --phase cohort --width 16 --weight a --execute
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API = "https://localhost:3000/api/flows"
WIDTHS = [8, 16, 32, 64, 96]
DATASETS = {                       # key → (ids_dataset, ids_split)
	"unsw-temporal":     ("unsw-nb15",                "temporal"),
	"unsw-random":       ("unsw-nb15",                "random"),
	"cicids-random":     ("cicids2017",               "random"),
	"ciciot-subsample":  ("ciciot2023_neto_subsample", "random"),
	# Note: ciciot-46M (ciciot2023_neto_full) is intentionally NOT here — it
	# uses 2 direct flows (no probe-cohort structure). See
	# scripts/queue_46m_direct.py for that path.
}

# Per-dataset architecture override. Default (no entry) = BASE_PARAMS's
# 250n×100b. UNSW-temp drops to 500n×34b because at only 175K train rows
# (8× smaller than the 1.43M ciciot-subsample where 250n×100b was derived),
# 100-bit address spaces become too sparse for effective per-neuron coverage.
# Format: key → (max_neurons, max_bits).
DATASET_ARCH = {
	"unsw-temporal":     (500, 34),
	# unsw-random (1.27M): default 250×100b — close to derivation size
	# cicids-random  (2.26M): default 250×100b — larger than derivation
	# ciciot-subsample (1.14M): default 250×100b — THE derivation dataset
}

# Weight sets (ce, acc, f1, fpr). "a" = the CIC-IoT cohort weights (CE-heavy);
# "b" = each dataset's own historical F1/FPR-heavy weights;
# "c" = CE-heavy experimental probe added 29/05/2026 — fitness_weight_ce=0.70,
# the highest CE-dominance recipe. The Wc scheme dominated F1 on the
# XDS-unsw-temporal probe (peak 89.50 vs Wb's 89.08), motivating its inclusion
# in the cross-dataset comparison going forward.
WEIGHTS_A = {"ce": 0.35, "acc": 0.30, "f1": 0.30, "fpr": 0.05}
WEIGHTS_B = {
	"unsw-nb15":                 {"ce": 0.1,  "acc": 0.2,  "f1": 0.35, "fpr": 0.35},
	"cicids2017":                {"ce": 0.2,  "acc": 0.1,  "f1": 0.30, "fpr": 0.40},
	# CIC-IoT subsample weight-set B mirrors the UNSW F1/FPR-heavy pattern
	# (the "non-CE-leading" alternative); WEIGHTS_A is its CE-leading native.
	"ciciot2023_neto_subsample": {"ce": 0.1,  "acc": 0.2,  "f1": 0.35, "fpr": 0.35},
}
WEIGHTS_C = {"ce": 0.70, "acc": 0.10, "f1": 0.15, "fpr": 0.05}
# "bu" = UNIFORM Wb (= UNSW-nb15's historical Wb) applied to ANY dataset, so the
# weight scheme is held constant across datasets and only the dataset varies — the
# clean apples-to-apples cross-dataset comparison. Distinct from each dataset's
# NATIVE "b" (WEIGHTS_B[ds]); for CICIDS the two differ (native ce.20/acc.10/f1.30/
# fpr.40 vs uniform ce.10/acc.20/f1.35/fpr.35). Run both to let the data choose.
WEIGHTS_BU = {"ce": 0.1, "acc": 0.2, "f1": 0.35, "fpr": 0.35}

# Canonical OI cohort params (dataset/split/n_bits/weights/architecture filled
# per flow via DATASET_ARCH override). Default architecture in BASE_PARAMS is
# 250n×100b — the architecture originally derived from CIC-IoT subsample (1.43M)
# sweeps and used by 2747+2748 + the FIXED cohort. After the 28/05/2026
# offspring-collapse RCA proved that bug was eval-path + fitness_scores (NOT
# architecture-related), 250n×100b became the default for any dataset at or
# above the derivation size. Smaller datasets (UNSW-temp at 175K train rows is
# 8× smaller than the derivation point) drop down to 500n×34b for denser per-
# neuron coverage — see DATASET_ARCH below.
BASE_PARAMS = {
	"ids_classification": "binary",
	"ids_feature_selection": "top20",
	"architecture_type": "ids",
	"min_neurons": 5, "max_neurons": 250,
	"min_bits": 4, "max_bits": 100,
	"population_size": 50, "ga_generations": 250, "patience": 5,
	"phase_order": "neurons_first",
	"fitness_calculator": "harmonic_rank",
	"ids_k_folds": 5, "ids_kfold_per_gen": 5, "ids_num_parts": 5, "ids_val_fraction": 0.25,
	"ids_single_cluster": True, "balance_classes": True, "neuron_sample_rate": 0.25,
	# UNSW (~1.6M) / CICIDS (~2.8M) fit in RAM; memmap not needed (unlike 46M).
	"ids_encoded_storage": "memory",
	"min_accuracy_floor": 0, "pool_shuffle_ratio": 0.8, "adaptation_iterations": 50,
	"cluster_crossover_ratio": 0.5, "assortative_mating_ratio": 0.85,
	"neighbors_per_iter": 50, "context_size": 4,
	"threshold_start": 0, "threshold_step": 1, "fitness_percentile": 0.75,
	"wnn_order_independent_train": True,
}

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
	{"name": "GA Neurons", "phase_type": "ga_neurons", "experiment_type": "ga"},
]


def _weights(ds: str, weight_key: str) -> dict:
	if weight_key == "a":
		return WEIGHTS_A
	if weight_key == "c":
		return WEIGHTS_C
	if weight_key == "bu":
		return WEIGHTS_BU
	return WEIGHTS_B[ds]


def build_flow(ds_key: str, n_bits: int, weight_key: str, seed: int,
               arch_override: tuple | None = None) -> dict:
	"""Build a flow body for a single queue POST.

	Architecture resolution order (winner-most-specific):
	  1. arch_override (CLI --max-neurons/--max-bits) — for curiosity runs
	     like "UNSW-temp at 250n×100b" that deliberately override the dataset
	     default.
	  2. DATASET_ARCH[ds_key] — per-dataset override (e.g. UNSW-temp 500×34).
	  3. BASE_PARAMS defaults (250×100 for any dataset without override).

	The resolved (neurons, bits) is reflected in the flow name so any
	non-default architecture is grep-able in the DB later.
	"""
	ds, split = DATASETS[ds_key]
	w = _weights(ds, weight_key)
	if arch_override is not None:
		max_neurons, max_bits = arch_override
	else:
		max_neurons, max_bits = DATASET_ARCH.get(
			ds_key, (BASE_PARAMS["max_neurons"], BASE_PARAMS["max_bits"])
		)
	params = dict(BASE_PARAMS)
	params.update({
		"ids_dataset": ds, "ids_split": split, "ids_n_bits": n_bits, "seed": seed,
		"max_neurons": max_neurons, "max_bits": max_bits,
		"fitness_weight_ce": w["ce"], "fitness_weight_acc": w["acc"],
		"fitness_weight_f1": w["f1"], "fitness_weight_fpr": w["fpr"],
	})
	name = f"XDS-{ds_key}-{n_bits}b-W{weight_key}-C35-{max_neurons}n{max_bits}b-OI-r{seed}"
	return {
		"name": name,
		"description": f"Cross-dataset OI cohort: {ds} {split}, {n_bits}b thermo, weight-set {weight_key}.",
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": [dict(e) for e in EXPERIMENTS],
		"seed_checkpoint_id": None,
	}


def probe_flows(ds_key: str, seeds: list[int], arch_override: tuple | None = None,
                weight_keys: tuple = ("a", "b", "c")) -> list[dict]:
	"""Phase 1: widths × weight-sets × shared seeds.

	With the default 3 weight sets {a,b,c} (added 30/05/2026 after Wc was
	proven dominant on XDS-unsw-temporal): 5 × 3 × 3 = 45 flows.

	INTERLEAVED order (seeds OUTERMOST): round 1 = all (width×weight) combos at
	seed[0], round 2 = all at seed[1], round 3 = at seed[2]. With the worker's
	id-DESC pick order, every combo gets one seed before any gets a second — a
	clearly-losing combo is cullable after round 1 instead of after 3 back-to-back
	seeds. (feedback: sweeps always interleave across dimensions, 31/05/2026.)"""
	return [build_flow(ds_key, n, wk, s, arch_override=arch_override)
	        for s in seeds for n in WIDTHS for wk in weight_keys]


def cohort_flows(ds_key: str, width: int, weight_key: str, n: int,
                 arch_override: tuple | None = None) -> list[dict]:
	"""Phase 2: `n` NEW-seed flows at the chosen (width, weight). Default n=97 so with
	the 3 renamed probe winners it totals 100."""
	return [build_flow(ds_key, width, weight_key, secrets.randbelow(100000),
	                   arch_override=arch_override) for _ in range(n)]


def post(flow: dict, execute: bool) -> None:
	if not execute:
		return
	r = requests.post(API, json=flow, verify=False, timeout=30)
	r.raise_for_status()
	fid = r.json().get("id")
	# The /api/flows endpoint creates new flows in `pending` status (held back from
	# the worker). Flip to `queued` via /restart so the worker can actually pick
	# them up. Without this, queued flows look "queued" in the script output but
	# the worker starves.
	rr = requests.post(f"{API}/{fid}/restart", json={}, verify=False, timeout=15)
	rr.raise_for_status()
	print(f"    queued flow {fid}: {flow['name']}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], required=True)
	ap.add_argument("--phase", choices=["probe", "cohort"], required=True)
	ap.add_argument("--width", type=int, help="Phase 2: chosen thermo width (from probe)")
	ap.add_argument("--weight", choices=["a", "b", "bu", "c"], help="Phase 2: chosen weight set")
	ap.add_argument("--weights", nargs="+", choices=["a", "b", "bu", "c"], default=["a", "b", "c"],
		help="Phase 1: which weight sets to sweep (default a b c). Use 'bu' for the uniform "
		     "(UNSW) Wb to compare against a dataset's native 'b' in the same sweep.")
	ap.add_argument("--seeds", type=int, nargs=3, default=None,
		help="Phase 1: the 3 shared probe seeds (default: random, printed for the record)")
	ap.add_argument("--cohort-n", type=int, default=97, help="Phase 2 new-seed flow count (default 97 → +3 winners = 100)")
	ap.add_argument("--max-neurons", type=int, default=None,
		help="Override per-dataset architecture neuron cap (e.g. 250 to force 250n×100b on a dataset normally at 500n×34b — useful for curiosity experiments)")
	ap.add_argument("--max-bits", type=int, default=None,
		help="Override per-dataset architecture bit cap (must be passed together with --max-neurons)")
	ap.add_argument("--execute", action="store_true", help="Actually POST (default: dry-run preview)")
	args = ap.parse_args()

	# Resolve the architecture override (both flags or neither)
	arch_override: tuple | None = None
	if (args.max_neurons is None) != (args.max_bits is None):
		sys.exit("--max-neurons and --max-bits must be passed together (or both omitted to use DATASET_ARCH default)")
	if args.max_neurons is not None:
		arch_override = (args.max_neurons, args.max_bits)

	ds_keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
	mode = "EXECUTE" if args.execute else "DRY-RUN"

	for ds_key in ds_keys:
		if args.phase == "probe":
			seeds = args.seeds or [secrets.randbelow(100000) for _ in range(3)]
			flows = probe_flows(ds_key, seeds, arch_override=arch_override,
			                    weight_keys=tuple(args.weights))
			arch_note = f" (arch override: {arch_override[0]}n×{arch_override[1]}b)" if arch_override else ""
			print(f"[{mode}] {ds_key} PROBE — {len(flows)} flows "
			      f"({len(WIDTHS)} widths × {len(args.weights)} weights {args.weights} "
			      f"× seeds {seeds}){arch_note}")
		else:
			if args.width is None or args.weight is None:
				sys.exit("--phase cohort requires --width and --weight (the probe winner)")
			flows = cohort_flows(ds_key, args.width, args.weight, args.cohort_n, arch_override=arch_override)
			arch_note = f" (arch override: {arch_override[0]}n×{arch_override[1]}b)" if arch_override else ""
			print(f"[{mode}] {ds_key} COHORT — {len(flows)} new flows @ {args.width}b W{args.weight} "
			      f"(+3 renamed winners = {args.cohort_n + 3}){arch_note}")
		for f in flows:
			print(f"  {f['name']}")
			post(f, args.execute)
	if not args.execute:
		print(f"\n  DRY-RUN — nothing queued. Re-run with --execute to POST. Sample body:")
		print("  " + json.dumps(flows[0]["config"]["params"], indent=2).replace("\n", "\n  "))


if __name__ == "__main__":
	sys.exit(main())
