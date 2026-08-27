"""Queue the MCS multiclass screening cohort under DESIRABILITY (26/08/2026).

15 flows = 3 arms (binary / multi / hierarchical) x 5 seeds 20401-20405, cloned
byte-for-byte from the PAUSED MCS-* flows with exactly TWO keys changed:

    fitness_aggregation          zscore -> desirability
    fitness_ce_anchor_normalized (absent) -> 0.1937

WHY THE ANCHOR IS NOT OPTIONAL HERE. The desirability CE half-anchor 0.133 is a
frozen unsw-nb15 BINARY fit. Multiclass CE on the same dataset runs ~1.90 —
14.3x that anchor — so with the binary absolute the ce column would sit at 14.3
of its 20 half-life clamp, take ~28% of the score at a weight of 0.10, and go
FLAT (clamped) for weak genomes. Worse for THIS cohort specifically: the only
variable is ids_classification, so the binary arm would be scored on a scale
that fits it and the multi/hier arms on one that does not — biasing the very
comparison the cohort exists to make. 0.1937 is the same anchor expressed in
units of each task's OWN base-rate log-loss H(p); ram_accelerator (ABI 9)
derives the absolute per task from the train labels already in the Rust cache,
so each arm — and each stage of the hierarchical arm — gets its own scale.

PRE-REGISTERED READ (fixed before any flow runs, unchanged from the original
MCS registration): primaries = macro-F1 and benign-FPR on the held-out val_cal
partition; the per-class recall table is MANDATORY (the QSR lesson: an
aggregate-F1 win bought with recall losses on 8/9 classes is NOT "detects
better"); RF/XGB per-dataset bar for unsw-nb15 temporal_3way = macro-F1 0.52.
Read ONCE. Never report during-search k-fold numbers.

The 15 original MCS-* flows are LEFT PAUSED, not deleted — they are the zscore
configuration and remain available if the anchor work is ever re-litigated.
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
SOURCE_LIKE = "MCS-unswt-quad-16b-%"
CE_ANCHOR_NORMALIZED = 0.2128
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def worker_wheel_ready() -> bool:
	"""ABI 9 = the wheel that can derive a per-task CE anchor."""
	try:
		import ram_accelerator as a
		return (getattr(a, "ABI_VERSION", 0) >= 9
		        and hasattr(a.IDSCacheWrapper, "desirability_ce_anchor"))
	except ImportError:
		return False


def main() -> int:
	if not worker_wheel_ready() and "--force" not in sys.argv:
		print("REFUSED: installed ram_accelerator lacks ABI 9 / desirability_ce_anchor.\n"
		      "Run scripts/deploy_ce_anchor_worker.sh at worker idle first.")
		return 3

	con = ro()
	sources = con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE ? ORDER BY name",
		(SOURCE_LIKE,)).fetchall()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'MCSD-%'")}
	con.close()
	if len(sources) != 15:
		print(f"REFUSED: expected 15 MCS source flows, found {len(sources)}")
		return 4

	# --only <substr>: clone a SINGLE arm first. A fresh wheel plus a fresh code
	# path gets smoke-tested on one flow before a cohort is committed to it —
	# queueing all 15 against an unproven path is how a whole cohort dies in
	# seconds. Re-running without --only then fills in the rest (already-created
	# names are skipped).
	only = None
	for i, a in enumerate(sys.argv):
		if a == "--only" and i + 1 < len(sys.argv):
			only = sys.argv[i + 1]
	if only:
		sources = [r for r in sources if only in r[0]]
		if not sources:
			print(f"REFUSED: --only {only!r} matched no MCS source flow")
			return 5
		print(f"--only {only!r}: {len(sources)} flow(s) selected")

	print(f"MCSD desirability cohort: {len(sources)} flows cloned from paused MCS configs")
	if "--dry-run" in sys.argv:
		for sname, cj in sources:
			p = json.loads(cj)["params"]
			print(f"  would clone {sname}: {p['ids_classification']:<12} seed {p['seed']} "
			      f"agg {p['fitness_aggregation']} -> desirability, anchor {CE_ANCHOR_NORMALIZED}")
		print("DRY RUN — nothing created.")
		return 0

	created = []
	for sname, cj in sources:
		cfg = json.loads(cj)
		p = dict(cfg["params"])
		assert p.get("fitness_aggregation") == "zscore", f"{sname}: source is not zscore?"
		p["fitness_aggregation"] = "desirability"
		p["fitness_ce_anchor_normalized"] = CE_ANCHOR_NORMALIZED
		name = sname.replace("MCS-", "MCSD-", 1)
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (
				f"MCS multiclass screening under DESIRABILITY — clone of {sname} with "
				f"fitness_aggregation zscore->desirability and a per-task CE anchor "
				f"({CE_ANCHOR_NORMALIZED} x H(p), derived in ram_accelerator ABI 9). "
				f"Arm: {p['ids_classification']}, seed {p['seed']}."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": EXPERIMENTS,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], name))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	print(f"flipping {len(created)} pending -> queued...")
	for fid, _ in created:
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.12)

	con = ro()
	bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		ok = (ne == 2 and st == "queued"
		      and q.get("fitness_aggregation") == "desirability"
		      and q.get("fitness_ce_anchor_normalized") == CE_ANCHOR_NORMALIZED
		      and q.get("patience") == 5 and q.get("ids_k_folds") == 5)
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} "
			      f"agg={q.get('fitness_aggregation')} anchor={q.get('fitness_ce_anchor_normalized')}")
			bad += 1
	con.close()
	print("ALL VERIFIED" if not bad else f"{bad} flows FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
