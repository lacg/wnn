"""Re-run the 5 MCSD hierarchical flows against the FIXED cascade path (26/08/2026).

The original five are VOID for the pre-registered read — not wrong, but not
measuring what the arm exists to measure. `_compute_ids_hierarchical_combined`
returned S0's BINARY metrics under the combined keys, so those runs reported a
2-class macro-F1 (~90%) in the same column as the flat multi arm's 10-class one
(~40%). They are LEFT IN PLACE, not deleted: they are the record of what the old
path produced, and the new rows are distinguishable without deleting anything —
only a fixed run writes a `hierarchical_cascade` genome_type.

Config is cloned VERBATIM from each original, including
fitness_ce_anchor_normalized = 0.1937. That constant is the miscalibrated one
(0.2128 is correct — it was fitted against the TEST partition's entropy instead
of TRAIN), and it is kept ON PURPOSE: the binary and multi arms of this cohort
already ran with 0.1937, and MCSD's only variable is ids_classification. Fixing
the anchor here would put TWO differences between hier and its own cohort and
make the arm comparison uninterpretable. The corrected constant belongs to the
next cohort, not to a repair of this one.
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
SOURCE_LIKE = "MCSD-unswt-quad-16b-hier-s%"
SUFFIX = "-v2"
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def post_with_retry(url, body):
	last = None
	for _ in range(3):
		try:
			return requests.post(url, json=body, verify=False, timeout=60)
		except requests.exceptions.RequestException as e:
			last = e
			time.sleep(5)
	raise last


def main() -> int:
	con = ro()
	sources = con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE ? AND name NOT LIKE ? ORDER BY name",
		(SOURCE_LIKE, f"%{SUFFIX}")).fetchall()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE ?", (f"%{SUFFIX}",))}
	con.close()

	if len(sources) != 5:
		print(f"REFUSED: expected 5 original hier flows, found {len(sources)}")
		return 4
	print(f"re-running {len(sources)} hier flows against the fixed cascade path")
	if "--dry-run" in sys.argv:
		for n, cj in sources:
			p = json.loads(cj)["params"]
			print(f"  would re-run {n} -> {n}{SUFFIX}  (seed {p['seed']}, arm {p['ids_classification']}, "
			      f"anchor {p.get('fitness_ce_anchor_normalized')})")
		print("DRY RUN — nothing created.")
		return 0

	created = []
	for sname, cj in sources:
		cfg = json.loads(cj)
		p = dict(cfg["params"])
		assert p.get("ids_classification") == "hierarchical", f"{sname} is not the hier arm"
		name = f"{sname}{SUFFIX}"
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (
				f"Re-run of {sname} against the FIXED hierarchical cascade "
				f"(true 10-class S0->S1 routing + per-class + confusion). Config byte-identical "
				f"to the original, anchor 0.1937 kept to match this cohort's other arms."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": EXPERIMENTS,
		}
		r = post_with_retry(f"{DASHBOARD}/api/flows", body)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], name))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	print(f"flipping {len(created)} pending -> queued...")
	for fid, name in created:
		try:
			post_with_retry(f"{DASHBOARD}/api/flows/{fid}/restart", {})
		except requests.exceptions.RequestException:
			print(f"  ! {fid} {name}: flip failed — still pending")
		time.sleep(0.4)

	con = ro()
	bad = 0
	for fid, name in created:
		st, = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()
		ne, = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		ok = ne == 2 and st == "queued" and q.get("ids_classification") == "hierarchical"
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} arm={q.get('ids_classification')}")
			bad += 1
	con.close()
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
