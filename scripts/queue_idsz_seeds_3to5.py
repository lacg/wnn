"""Extend the IDSZ weight sweep from n=2 to n=5 seeds — 23 arms x 3 new seeds = 69 flows.

WHY (20/08/2026). An earlier attempt (IDSZB, patience=999 + max_iterations=100) tried to
budget-match every arm. Luiz's ruling: patience is 5, it exists for a reason, and no
search-control config changes without discussing first. Those flows are DELETED and their
results must NOT enter any statistic.

With patience back to 5 the config is IDENTICAL to the existing IDSZ cohort, so the 46
completed IDSZ runs (seeds 20301, 20302) COUNT. Only seeds 20303-20305 are needed to reach
n=5 per arm. These are created under the IDSZ name so the readout treats them as one cohort.

Config is cloned VERBATIM from IDSZ flow 5380; only `seed` and the four fitness weights vary.
Nothing else is touched — no patience override, no generation cap.

Per CLAUDE.md Rule 2: dashboard POST /api/flows with experiments, never a direct SQL insert.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DASHBOARD = "https://localhost:3000"
DB_PATH = Path("/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db")
TEMPLATE_FLOW = 5380          # IDSZ-unswt-quad-16b-Wb-CTRL-r20301
NEW_SEEDS = [20303, 20304, 20305]
PREFIX = "IDSZ-unswt-quad-16b-"

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def purge_idszb() -> int:
	"""Delete every IDSZB flow. delete_flow cascades to experiments, iterations,
	checkpoints and stops any running process, so no patience=999 result can survive."""
	con = ro()
	ids = [r[0] for r in con.execute(
		"SELECT id FROM flows WHERE name LIKE 'IDSZB-%' ORDER BY id")]
	con.close()
	if not ids:
		print("no IDSZB flows to purge")
		return 0
	print(f"purging {len(ids)} IDSZB flows (patience=999 — must not enter statistics)")
	gone = 0
	for fid in ids:
		r = requests.delete(f"{DASHBOARD}/api/flows/{fid}", verify=False, timeout=30)
		if r.status_code in (200, 204):
			gone += 1
		else:
			print(f"  ! delete failed id={fid} ({r.status_code}) {r.text[:120]}")
		time.sleep(0.05)
	print(f"  deleted {gone}/{len(ids)}")
	return gone


def arms(con) -> list[tuple[str, float, float, float, float]]:
	out: dict[str, tuple] = {}
	for name, cfg in con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE ?", (PREFIX + "%",)):
		arm = name.replace(PREFIX, "").rsplit("-r", 1)[0]
		p = json.loads(cfg).get("params", {})
		out[arm] = (p["fitness_weight_ce"], p["fitness_weight_acc"],
		            p["fitness_weight_f1"], p["fitness_weight_fpr"])
	return sorted((a, *w) for a, w in out.items())


def main() -> int:
	con = ro()
	row = con.execute("SELECT config_json FROM flows WHERE id=?", (TEMPLATE_FLOW,)).fetchone()
	if not row:
		print(f"ERROR: template flow {TEMPLATE_FLOW} missing"); return 1
	base = dict(json.loads(row[0])["params"])
	A = arms(con)
	existing = {r[0] for r in con.execute(
		"SELECT name FROM flows WHERE name LIKE ?", (PREFIX + "%",))}
	con.close()

	# Guard: the template must carry the STANDARD search controls.
	if base.get("patience") != 5:
		print(f"ERROR: template patience is {base.get('patience')}, expected 5"); return 1
	if base.get("ga_generations") != 250:
		print(f"ERROR: template ga_generations is {base.get('ga_generations')}, expected 250"); return 1
	if len(A) != 23:
		print(f"ERROR: expected 23 arms, found {len(A)}"); return 1

	print(f"template patience={base['patience']}  ga_generations={base['ga_generations']}  (standard)")
	print(f"arms={len(A)}  new seeds={NEW_SEEDS}  -> {len(A)*len(NEW_SEEDS)} flows")
	if "--dry-run" in sys.argv:
		print("DRY RUN — nothing purged, nothing created."); return 0

	purge_idszb()

	created = []
	for seed in NEW_SEEDS:
		for arm, ce, acc, f1, fpr in A:
			name = f"{PREFIX}{arm}-r{seed}"
			if name in existing:
				print(f"  = skip {name} (exists)"); continue
			params = dict(base)
			params["seed"] = seed
			params["fitness_weight_ce"] = ce
			params["fitness_weight_acc"] = acc
			params["fitness_weight_f1"] = f1
			params["fitness_weight_fpr"] = fpr
			body = {
				"name": name,
				"description": (
					f"IDS fitness-WEIGHT sweep under the ZSCORE combine, seed extension to n=5. "
					f"{arm} = ce {ce}/acc {acc}/f1 {f1}/fpr {fpr}, seed={seed}. Config cloned "
					f"verbatim from flow {TEMPLATE_FLOW} — STANDARD patience=5, "
					f"ga_generations=250. Same cohort as seeds 20301/20302."
				),
				"config": {"template": "ids-binary-2-phase", "params": params},
				"experiments": EXPERIMENTS,
			}
			r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
			if r.status_code not in (200, 201):
				print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}"); return 2
			created.append((r.json()["id"], name))
			print(f"  + id={created[-1][0]}  {name}")
			time.sleep(0.25)

	print(f"\nflipping {len(created)} pending -> queued...")
	for fid, _ in created:
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.15)

	con = ro(); bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		cfg = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])
		pat = cfg["params"].get("patience")
		mi = con.execute("SELECT max_iterations FROM experiments WHERE flow_id=? AND name LIKE 'GA%'",
		                 (fid,)).fetchone()
		if not (ne == 2 and st in ("queued", "running") and pat == 5 and mi and mi[0] == 250):
			print(f"  x id={fid} status={st} exps={ne} patience={pat} max_iter={mi and mi[0]}  {name}")
			bad += 1
	left = con.execute("SELECT COUNT(*) FROM flows WHERE name LIKE 'IDSZB-%'").fetchone()[0]
	con.close()
	print(f"VERIFY: {len(created)-bad}/{len(created)} queued, patience=5, max_iterations=250")
	print(f"VERIFY: IDSZB flows remaining = {left} (must be 0)")
	return 0 if bad == 0 and left == 0 else 3


if __name__ == "__main__":
	sys.exit(main())
