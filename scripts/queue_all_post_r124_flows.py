"""Create + queue all post-r124 flows in race-safe order.

Avoids a race where the worker polls (every 10s) and could grab a Phase D flow
before the PUB50 batch finishes being created. Sequence:

  1. Create both Phase D flows as PENDING (worker won't pick pending flows)
  2. Create all 112 PUB50 flows as PENDING
  3. Queue PUB50 first (transition pending → queued)  ← worker can now grab
     these; they have higher IDs (created later) so worker picks highest first
  4. Queue Phase D last  ← already-queued PUB50 has higher IDs, so Phase D
     waits behind all 112 PUB50

This script REPLACES queue_phase_d_neto_full_flows.py +
queue_pub50_neto_subsample_flows.py for post-r124 use. Used by the
auto_per_class_when_r124_done.py watcher.

IDs after this script (assuming starting from id 1688):
  Phase D r144 → 1688 (created first, lowest id)
  Phase D r145 → 1689
  PUB50 r001  → 1690
  PUB50 r002  → 1691
  ...
  PUB50 r112  → 1801 (created last, highest id)

Worker order (ORDER BY id DESC):
  PUB50 r112 → r001 → Phase D r145 → r144

Per CLAUDE.md Rule 2: ALWAYS create flows via dashboard POST /api/flows.
"""

import requests, urllib3, json, sys, sqlite3, time
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "wnn.db"

PHASE_D_SEEDS = [144, 145]
PUB50_SEEDS = list(range(1, 113))  # 1..112 inclusive

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def get_base_config(flow_name_or_id) -> dict:
	con = sqlite3.connect(str(DB_PATH))
	if isinstance(flow_name_or_id, int):
		row = con.execute("SELECT config_json FROM flows WHERE id = ?", (flow_name_or_id,)).fetchone()
	else:
		row = con.execute("SELECT config_json FROM flows WHERE name = ?", (flow_name_or_id,)).fetchone()
	con.close()
	if not row:
		raise ValueError(f"Base flow {flow_name_or_id!r} not found")
	return json.loads(row[0])


def create_pending(name: str, description: str, params: dict) -> int | None:
	"""POST /api/flows — creates flow in 'pending' status (worker doesn't pick pending)."""
	body = {
		"name": name,
		"description": description,
		"config": {"template": "ids-binary-2-phase", "params": params},
		"experiments": EXPERIMENTS,
	}
	r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
	if r.status_code in (200, 201):
		return r.json().get("id")
	print(f"  ✗ create failed for {name}: {r.status_code} {r.text[:200]}")
	return None


def queue_flow(fid: int) -> bool:
	"""POST /api/flows/{id}/restart — transitions pending → queued."""
	r = requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=10)
	return r.status_code == 200


def main():
	# Pull base configs once
	r98_params = dict(get_base_config(1156)["params"])  # Phase D base
	pub50_params = dict(get_base_config("PUB50-ciciot-random-r001")["params"])  # PUB50 base

	# Phase D overrides
	phase_d_base = dict(r98_params)
	phase_d_base["ids_dataset"] = "ciciot2023_neto_full"
	phase_d_base["ids_split"] = "random_3way"
	phase_d_base["ids_invalid_encoding"] = "single_bit"

	# PUB50 overrides
	pub50_base = dict(pub50_params)
	pub50_base["ids_dataset"] = "ciciot2023_neto_subsample"
	pub50_base["ids_split"] = "random"
	pub50_base["ids_invalid_encoding"] = "single_bit"

	# === Step 1: Create Phase D as PENDING ===
	print(f"Step 1: Creating {len(PHASE_D_SEEDS)} Phase D flows (PENDING)...")
	phase_d_ids = []
	for seed in PHASE_D_SEEDS:
		params = dict(phase_d_base); params["seed"] = seed
		name = f"PUB-neto-full-46M-8b-r{seed}"
		desc = f"Phase D / neto-full canonical / single_bit / seed={seed}. Compare vs r98 + r125+r124."
		fid = create_pending(name, desc, params)
		if fid is not None:
			phase_d_ids.append(fid)
			print(f"  Created Phase D id={fid}: {name}")

	# === Step 2: Create all 112 PUB50 as PENDING ===
	print(f"\nStep 2: Creating {len(PUB50_SEEDS)} PUB50 flows (PENDING)...")
	pub50_ids = []
	for seed in PUB50_SEEDS:
		params = dict(pub50_base); params["seed"] = seed
		name = f"PUB50-neto-sub-ciciot-random-r{seed:03d}"
		desc = f"PUB50 on neto-subsample (1.43M, 46 features) / single_bit / seed={seed}"
		fid = create_pending(name, desc, params)
		if fid is not None:
			pub50_ids.append(fid)
			if seed % 10 == 0 or seed in (1, len(PUB50_SEEDS)):
				print(f"  Created PUB50 [{seed:3d}/{len(PUB50_SEEDS)}]: id={fid}")
		time.sleep(0.05)  # gentle on the dashboard

	# === Step 3: Queue PUB50 first (highest IDs run first) ===
	print(f"\nStep 3: Queueing {len(pub50_ids)} PUB50 flows (this opens them to the worker)...")
	queued_pub50 = sum(1 for fid in pub50_ids if queue_flow(fid))
	print(f"  ✓ {queued_pub50}/{len(pub50_ids)} PUB50 queued. Worker can now start picking.")

	# === Step 4: Queue Phase D last (lower IDs — wait behind PUB50) ===
	print(f"\nStep 4: Queueing {len(phase_d_ids)} Phase D flows (will run after all PUB50)...")
	queued_phase_d = sum(1 for fid in phase_d_ids if queue_flow(fid))
	print(f"  ✓ {queued_phase_d}/{len(phase_d_ids)} Phase D queued.")

	# Summary
	total_created = len(phase_d_ids) + len(pub50_ids)
	total_queued = queued_pub50 + queued_phase_d
	print(f"\nTotal: {total_created} created, {total_queued} queued.")
	print(f"Run order (worker = ORDER BY id DESC):")
	print(f"  1. PUB50 r112 → r001 ({len(pub50_ids)} flows × ~80 min ≈ 6-7 days)")
	print(f"  2. Phase D r145 → r144 ({len(phase_d_ids)} flows × ~3 days ≈ 6 days)")


if __name__ == "__main__":
	main()
