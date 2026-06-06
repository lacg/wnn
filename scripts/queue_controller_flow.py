#!/usr/bin/env python3
"""Queue a drone-CONTROLLER phased-GA flow on the dashboard (Option A path).

A controller flow runs the 5-stage phased GA (grid -> neurons -> bits ->
connections -> memory) via flow_runner._run_controller_flow, which wraps
wnn.control.phased_ga._run_one. The worker schedules it under the CPU budget,
co-resident with IDS (controller takes wnn_num_threads=3 of the reserved cores).

Param keys MUST match the phased_ga CLI arg dests (e.g. neurons_gens,
fit_weight_err_sq, base_seed) -- _build_phased_ga_args maps them by name, so a
typo silently no-ops. wnn_num_threads + check_interval are scheduler/early-stop
knobs (wnn_num_threads is consumed by the scheduler; check_interval by the GA).

DRY-RUN by default -- pass --execute to actually POST. Use --smoke for a tiny
end-to-end run (pop 6, 1 gen/stage, 2 eval episodes) to validate the dashboard
path at deploy time before launching the full recipe.

Examples:
  python scripts/queue_controller_flow.py                       # preview full recipe
  python scripts/queue_controller_flow.py --smoke --execute     # tiny E2E (deploy check)
  python scripts/queue_controller_flow.py --lamarckian --execute
  python scripts/queue_controller_flow.py --tilt 15 --pop 50 --neurons-gens 200 --execute
"""

from __future__ import annotations

import argparse
import json
import sys

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API = "https://localhost:3000/api/flows"

# The 5 dashboard experiment records, in stage order. flow_runner resolves them
# by flow-sequence (0=grid .. 4=memory) to thread the per-stage tracker.
EXPERIMENTS = [
	{"name": "Grid (state_neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
	{"name": "GA Neurons",     "phase_type": "ga_neurons",     "experiment_type": "ga"},
	{"name": "GA Bits",        "phase_type": "ga_bits",        "experiment_type": "ga"},
	{"name": "GA Connections", "phase_type": "ga_connections", "experiment_type": "ga"},
	{"name": "GA Memory",      "phase_type": "ga_memory",      "experiment_type": "ga"},
]

# Full controller recipe. Keys == phased_ga CLI dests (see _build_phased_ga_args).
# Mirrors the Stage-A lamarckian recipe (tilt 5 / pop 50 / 30-30-30-60 gens,
# patience 2, K-fold 5, multi-objective fitness). wnn_num_threads=3 keeps the
# controller within the cores IDS reserves; check_interval=5 is the controller
# early-stop cadence.
FULL_PARAMS = {
	"architecture_type": "controller",
	"wnn_num_threads": 3,
	"check_interval": 5,
	# Stage 0 grid axes.
	"grid_state_neurons": [8, 12, 16, 20, 24],
	"grid_bits": [18, 24, 30, 36],
	"levels": 16,
	# Stages 1-4 budgets + patience.
	"neurons_gens": 30, "neurons_patience": 2,
	"bits_gens": 30, "bits_patience": 2,
	"conns_gens": 30, "conns_patience": 2,
	"memory_gens": 60, "memory_patience": 2,
	# GA hyperparams.
	"pop": 50, "elitism": 0.2,
	# Episode / eval.
	"eval_episodes": 100, "universe_episodes": 8, "steps": 250,
	"tilt": 5.0, "body_rate": 0.5, "yaw_rate": 0.3,
	"num_eval_folds": 5, "train_workers": 4,
	# Multi-objective fitness (harmonic-rank activates when any non-err weight > 0).
	"fit_weight_err_sq": 0.40, "fit_weight_stable": 0.30,
	"fit_weight_jerk": 0.10, "fit_weight_mono": 0.20,
	# Seeds (base_seed -> 3-way SeedSet; report_seed -> held-out).
	"base_seed": 5005, "report_seed": 9009,
}

# Tiny end-to-end recipe for the deploy smoke test (seconds, not hours).
SMOKE_OVERRIDES = {
	"grid_state_neurons": [4, 6],
	"grid_bits": [12, 16],
	"neurons_gens": 1, "bits_gens": 1, "conns_gens": 1, "memory_gens": 1,
	"neurons_patience": 1, "bits_patience": 1, "conns_patience": 1, "memory_patience": 1,
	"pop": 6, "eval_episodes": 2, "universe_episodes": 2, "steps": 50,
	"num_eval_folds": 1,  # smoke only: K=1 to keep it seconds (real runs use 5)
	"rg_rounds": 1, "rg_episodes_per_round": 2, "rg_eval_episodes": 2,
}


def build_flow(args) -> dict:
	params = dict(FULL_PARAMS)
	if args.smoke:
		params.update(SMOKE_OVERRIDES)
	# Explicit CLI overrides (only when provided).
	for cli_key, pkey in (
		("tilt", "tilt"), ("pop", "pop"), ("base_seed", "base_seed"),
		("report_seed", "report_seed"), ("wnn_num_threads", "wnn_num_threads"),
		("check_interval", "check_interval"),
		("neurons_gens", "neurons_gens"), ("bits_gens", "bits_gens"),
		("conns_gens", "conns_gens"), ("memory_gens", "memory_gens"),
	):
		val = getattr(args, cli_key, None)
		if val is not None:
			params[pkey] = val
	params["lamarckian"] = bool(args.lamarckian)

	tag = "smoke" if args.smoke else "stageA"
	lam = "-lam" if args.lamarckian else ""
	name = (f"CTRL-{tag}{lam}-tilt{params['tilt']:g}-pop{params['pop']}-"
	        f"{params['neurons_gens']}n{params['bits_gens']}b{params['conns_gens']}c{params['memory_gens']}m"
	        f"-r{params['base_seed']}")
	return {
		"name": name,
		"description": (f"Controller phased-GA ({tag}): tilt {params['tilt']}deg, pop {params['pop']}, "
		               f"lamarckian={params['lamarckian']}, {params['wnn_num_threads']} cores."),
		"config": {"template": "controller-phased-ga", "params": params},
		"experiments": [dict(e) for e in EXPERIMENTS],
		"seed_checkpoint_id": None,
	}


def post(flow: dict, execute: bool) -> None:
	if not execute:
		return
	r = requests.post(API, json=flow, verify=False, timeout=30)
	r.raise_for_status()
	fid = r.json().get("id")
	# /api/flows creates flows as `pending` (held back); flip to `queued` so the
	# scheduler can admit it (mirrors queue_cross_dataset.py).
	rr = requests.post(f"{API}/{fid}/restart", json={}, verify=False, timeout=15)
	rr.raise_for_status()
	# Verify the experiments landed (Rule 2: a flow with 0 experiments does nothing).
	try:
		exps = requests.get(f"{API}/{fid}/experiments", verify=False, timeout=15).json()
		n_exp = len(exps)
	except Exception:
		n_exp = "?"
	print(f"    queued flow {fid} ({n_exp} experiments): {flow['name']}")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--smoke", action="store_true", help="Tiny E2E recipe (pop6, 1 gen/stage) for deploy validation")
	ap.add_argument("--lamarckian", action="store_true", help="Carry learned cells across arch-phase generations")
	ap.add_argument("--tilt", type=float, default=None, help="Initial tilt severity (deg)")
	ap.add_argument("--pop", type=int, default=None, help="Per-stage population")
	ap.add_argument("--wnn-num-threads", type=int, default=None, help="CPU cores (scheduler budget); default 3")
	ap.add_argument("--check-interval", type=int, default=None, help="Patience-check cadence (gens); default 5")
	ap.add_argument("--neurons-gens", type=int, default=None)
	ap.add_argument("--bits-gens", type=int, default=None)
	ap.add_argument("--conns-gens", type=int, default=None)
	ap.add_argument("--memory-gens", type=int, default=None)
	ap.add_argument("--base-seed", type=int, default=None)
	ap.add_argument("--report-seed", type=int, default=None)
	ap.add_argument("--execute", action="store_true", help="Actually POST (default: dry-run preview)")
	args = ap.parse_args()

	flow = build_flow(args)
	mode = "EXECUTE" if args.execute else "DRY-RUN"
	print(f"[{mode}] controller flow: {flow['name']}")
	print(f"  {len(flow['experiments'])} experiments: {[e['name'] for e in flow['experiments']]}")
	post(flow, args.execute)
	if not args.execute:
		print("\n  DRY-RUN — nothing queued. Re-run with --execute to POST. params:")
		print("  " + json.dumps(flow["config"]["params"], indent=2).replace("\n", "\n  "))


if __name__ == "__main__":
	sys.exit(main())
