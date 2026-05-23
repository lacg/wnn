#!/usr/bin/env python3
"""One-shot watcher: queue the 2× CIC-IoT-2023 46M OI flows after
flow 2452 starts running (i.e., when the 1.14M OI cohort is on its
last flow).

The trigger condition is "flow 2452 transitions to status=running"
because the worker picks queued flows by id DESC, so the
lowest-id queued flow runs last. Queueing the 46M flows at that
moment ensures they get higher ids than 2452 — they sit in the
queue while 2452 runs, then pick up automatically when 2452
completes (= cohort hits n=100).

If we queued the 46M flows earlier, their higher ids would jump
the queue and run BEFORE the remaining 1.14M cohort flows,
breaking the cohort. The watcher avoids that.

The locked 46M config matches the 1.14M OI cohort's methodology
exactly (96b thermo, 250n×100b ceiling, CE-leading weights,
K=5×5, neuron_sample_rate=0.25, OI training). See
~/.claude/projects/-Users-lacg-wnn/memory/project_46M_oi_flows_plan.md.

Usage (run detached so it survives terminal/session close):
  nohup python3 scripts/queue_46m_when_2452_runs.py \\
       > /tmp/46m-watcher.log 2>&1 &

Or one-shot manually (for testing):
  python3 scripts/queue_46m_when_2452_runs.py

The script exits as soon as it has queued the 46M flows
(success) or if it detects the trigger window was missed
(2452 already completed/cancelled/failed).
"""
import json
import secrets
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DB = Path("/Users/lacg/wnn/db/wnn.db")
DASHBOARD_API = "https://localhost:3000/api/flows"
TRIGGER_FLOW_ID = 2452
POLL_INTERVAL_S = 60
NAME_PREFIX = "WSWEEP-T20-96b-C35-250n100b-OI-46M"

# Exit if the trigger window was missed and the cohort is well past 2452.
TERMINAL_STATUSES = {"completed", "cancelled", "failed"}


def log(msg: str) -> None:
	stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
	print(f"[{stamp}] {msg}", flush=True)


def fetch_flow_status(flow_id: int) -> str | None:
	con = sqlite3.connect(DB)
	row = con.execute("SELECT status FROM flows WHERE id = ?", (flow_id,)).fetchone()
	con.close()
	return row[0] if row else None


def build_flow_body(seed: int) -> dict:
	"""Locked config — see project_46M_oi_flows_plan.md for the source of truth."""
	return {
		"name": f"{NAME_PREFIX}-r{seed}",
		"description": (
			"CIC-IoT-2023 full 46M (canonical neto-full), OI training, "
			"250n×100b ceiling, 96b thermometer, CE-leading weights, "
			"canonical TOP20. Queued by watcher after 1.14M OI cohort "
			f"reached n=99 (flow 2452 entered running state). Seed={seed}."
		),
		"config": {
			"template": "ids-binary-2-phase",
			"params": {
				"ids_dataset": "ciciot2023_neto_full",
				"ids_split": "random",
				"ids_classification": "binary",
				"ids_feature_selection": "top20",
				"architecture_type": "ids",
				"ids_n_bits": 96,
				"min_neurons": 5,
				"max_neurons": 250,
				"min_bits": 4,
				"max_bits": 100,
				"population_size": 50,
				"ga_generations": 250,
				"patience": 5,
				"phase_order": "neurons_first",
				"fitness_calculator": "harmonic_rank",
				"fitness_weight_ce": 0.35,
				"fitness_weight_acc": 0.30,
				"fitness_weight_f1": 0.30,
				"fitness_weight_fpr": 0.05,
				"ids_k_folds": 5,
				"ids_kfold_per_gen": 5,
				"ids_num_parts": 5,
				"ids_val_fraction": 0.25,
				"ids_single_cluster": True,
				"balance_classes": True,
				"neuron_sample_rate": 0.25,
				"ids_encoded_storage": "memmap",
				"ids_invalid_encoding": "single_bit",
				"min_accuracy_floor": 0,
				"pool_shuffle_ratio": 0.8,
				"adaptation_iterations": 50,
				"cluster_crossover_ratio": 0.5,
				"assortative_mating_ratio": 0.85,
				"neighbors_per_iter": 50,
				"context_size": 4,
				"threshold_start": 0,
				"threshold_step": 1,
				"fitness_percentile": 0.75,
				"wnn_order_independent_train": True,
				"seed": seed,
			},
		},
		"experiments": [
			{
				"name": "Grid Search (neurons x bits)",
				"phase_type": "grid_search",
				"experiment_type": "grid_search",
			},
			{
				"name": "GA Neurons",
				"phase_type": "ga_neurons",
				"experiment_type": "ga",
			},
		],
	}


def post_flow(body: dict) -> int:
	"""POST a flow body and return the created flow's id. Raises on failure."""
	# Use curl since the dashboard uses a self-signed cert and curl -k is the
	# project's standard pattern; avoids depending on `requests` / certifi.
	proc = subprocess.run(
		[
			"curl", "-sk", "-X", "POST",
			"-H", "Content-Type: application/json",
			"-d", json.dumps(body),
			DASHBOARD_API,
		],
		capture_output=True, text=True, check=False,
	)
	if proc.returncode != 0:
		raise RuntimeError(f"curl exited {proc.returncode}: {proc.stderr.strip()}")
	try:
		response = json.loads(proc.stdout)
	except json.JSONDecodeError as e:
		raise RuntimeError(f"Non-JSON response from API: {proc.stdout[:200]!r}") from e
	# Accept either {"id": N, ...} or {"flow": {"id": N, ...}}.
	if isinstance(response, dict):
		if "id" in response:
			return int(response["id"])
		if "flow" in response and isinstance(response["flow"], dict):
			return int(response["flow"]["id"])
	raise RuntimeError(f"Unexpected API response shape: {response!r}")


def verify_flow_has_experiments(flow_id: int) -> int:
	"""Per Behavioral Rule 2 in CLAUDE.md: flows without experiments are useless."""
	con = sqlite3.connect(DB)
	(count,) = con.execute(
		"SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (flow_id,)
	).fetchone()
	con.close()
	return count


def queue_46m_pair() -> None:
	"""POST 2 new 46M OI flows with unique random seeds. Logs each."""
	created = []
	for n in range(1, 3):
		seed = secrets.randbelow(100_000)
		body = build_flow_body(seed)
		flow_id = post_flow(body)
		exp_count = verify_flow_has_experiments(flow_id)
		log(f"  flow #{n}/2 created: id={flow_id}, name={body['name']}, "
			f"experiments={exp_count}")
		if exp_count == 0:
			log(f"  ⚠ flow {flow_id} has NO experiments — see Behavioral Rule 2. "
				"Worker will instant-complete this flow with no work. "
				"Manual fix required.")
		created.append((flow_id, body["name"]))
	log(f"Queued {len(created)} 46M OI flows: {created}")


def main() -> int:
	log(f"Watcher started. Polling flow {TRIGGER_FLOW_ID} every {POLL_INTERVAL_S}s.")
	log(f"Trigger: status='running'. Terminal-skip if status in {TERMINAL_STATUSES}.")

	while True:
		status = fetch_flow_status(TRIGGER_FLOW_ID)
		if status is None:
			log(f"⚠ flow {TRIGGER_FLOW_ID} not found in db. Exiting.")
			return 2
		if status == "running":
			log(f"✓ flow {TRIGGER_FLOW_ID} is running. Queueing 2× 46M flows...")
			try:
				queue_46m_pair()
			except Exception as e:
				log(f"✗ queue_46m_pair failed: {e}")
				return 3
			log("Watcher done. Exiting.")
			return 0
		if status in TERMINAL_STATUSES:
			log(f"✗ flow {TRIGGER_FLOW_ID} already in terminal state '{status}' — "
				"trigger window was missed. Queue the 46M flows manually if needed. "
				"Exiting.")
			return 1
		# Still queued/pending — keep waiting.
		log(f"  flow {TRIGGER_FLOW_ID} status='{status}', sleeping {POLL_INTERVAL_S}s")
		time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
	sys.exit(main())
