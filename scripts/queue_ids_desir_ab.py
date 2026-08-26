"""Queue the IDS DESIRABILITY arm (26/08/2026; docs/DESIRABILITY_FITNESS_SHAPES.md).

5 flows, NOT 10 (Luiz 26/08): the zscore CONTROL already exists — the completed
IDSZ-unswt-quad-16b-Wb-CTRL-r20301..r20305 flows ARE the control arm, so only
the desirability side needs compute (~5 min/flow on unswt-16b, ~25 min total).

Each desir flow clones its paired control's config_json VERBATIM from the DB
and changes exactly ONE key: fitness_aggregation zscore -> desirability. Same
seed, same Wb weights (as exponents there), same everything — the pairing is
byte-level, not by convention.

⚠️ DEPLOY GATE: desir flows CRASH on a pre-ABI-8 worker (wnn.accel lacks
desirability_fitness_combine; the calculator refuses to degrade silently).
Run scripts/deploy_ids_desir_worker.sh at WORKER IDLE first; this script
refuses to queue if the installed wheel is stale.

PRE-REGISTERED READ (fixed before any flow runs): primary = val_cal held-out
F1 and FPR, PAIRED PER SEED against the banked IDSZ Wb-CTRL rows, control
first in absolutes then deltas; full Rule-7 five tables via the ids-security
agent; winner = paired majority across 5 seeds, read ONCE. NEVER report
during-search k-fold numbers.
"""
import json
import sqlite3
import sys
import time

import requests
import urllib3

urllib3.disable_warnings()

DASHBOARD = "https://localhost:3000"
DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
CONTROL_LIKE = "IDSZ-unswt-quad-16b-Wb-CTRL-r%"
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def worker_wheel_ready() -> bool:
	try:
		import ram_accelerator as a
		return getattr(a, "ABI_VERSION", 0) >= 8 and hasattr(a, "desirability_fitness_combine")
	except ImportError:
		return False


def main() -> int:
	if not worker_wheel_ready() and "--force" not in sys.argv:
		print("REFUSED: installed ram_accelerator lacks ABI 8 / desirability_fitness_combine.\n"
		      "Run scripts/deploy_ids_desir_worker.sh at worker idle first.")
		return 3
	con = ro()
	controls = con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE ? AND status='completed' ORDER BY name",
		(CONTROL_LIKE,)).fetchall()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'IDSD-%'")}
	con.close()
	if len(controls) != 5:
		print(f"REFUSED: expected 5 completed IDSZ Wb-CTRL controls, found {len(controls)}")
		return 4

	print(f"IDSD desir arm: {len(controls)} flows cloned from completed IDSZ Wb-CTRL configs")
	if "--dry-run" in sys.argv:
		for cname, cj in controls:
			seed = json.loads(cj)["params"]["seed"]
			print(f"  would clone {cname} (seed {seed}) -> IDSD-unswt-quad-16b-desir-r{seed}")
		print("DRY RUN — nothing created."); return 0

	created = []
	for cname, cj in controls:
		cfg = json.loads(cj)
		p = dict(cfg["params"])
		assert p.get("fitness_aggregation") == "zscore", f"{cname}: control is not zscore?"
		p["fitness_aggregation"] = "desirability"
		seed = p["seed"]
		name = f"IDSD-unswt-quad-16b-desir-r{seed}"
		if name in existing:
			print(f"  = exists {name}"); continue
		body = {
			"name": name,
			"description": f"IDS desirability arm, seed {seed} — byte-level clone of {cname} "
			               f"with ONLY fitness_aggregation changed (zscore -> desirability); "
			               f"paired control = that banked flow",
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": EXPERIMENTS,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}"); return 2
		created.append((r.json()["id"], name))
		print(f"  + {r.json()['id']:>5}  {name}  (control: {cname})")
		time.sleep(0.2)

	print(f"flipping {len(created)} pending -> queued...")
	for fid, _ in created:
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.12)

	con = ro(); bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		ok = (ne == 2 and st == "queued" and q.get("fitness_aggregation") == "desirability"
		      and q.get("patience") == 5 and q.get("ga_generations") == 250)
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne}"); bad += 1
	con.close()
	print("ALL VERIFIED" if not bad else f"{bad} flows FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
