"""Re-run the WHOLE MCSD cohort under the automatic per-task CE anchor.

All 15 flows — binary, multi and hierarchical x 5 seeds — because the anchor
changes for every one of them. The previous MCSD runs carried
fitness_ce_anchor_normalized=0.1937, a constant fitted against the TEST
partition's entropy instead of TRAIN, which made every arm's ce column ~9%
tighter than intended. That param no longer exists: the anchor is now derived
per task from its own train labels, with nothing for a caller to select.

The hierarchical arm additionally gets FOUR stage-tagged phases. worker.py
derives target_stage by parsing the phase NAME, so phases without an S0:/S1:
prefix all resolve to stage 0, the boundary never fires, and the cascade is
unreachable — which is why that arm never produced a two-stage result until it
was named correctly. Four phases means the cascade trains two models with the
same grid+GA search the flat arms get one of; total budget is 2x, per-stage
budget is equal, and any read of this cohort must say so.

Old flows are LEFT IN PLACE, never deleted.
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
SOURCE_LIKE = "MCSD-unswt-quad-16b-%"
SUFFIX = "-auto"
# THE STAGE TAGS ARE LOad-BEARING. worker.py derives target_stage by PARSING the
# experiment name: `if exp_name.startswith("S") and ":" in exp_name[:4]`. Phases
# named "Grid Search (neurons x bits)" / "GA Neurons" therefore ALL get
# target_stage=0, no stage boundary ever fires, the S1 evaluator is never swapped
# in, and the whole cascade path is unreachable. That is why the hierarchical arm
# has never produced a two-stage result — not in MCS, not in MCSD, not in the -v2
# re-run: it has always been a plain binary flow wearing a hierarchical label.
#
# Four phases, not two: the cascade trains TWO models, so each stage gets the
# same grid+GA search the flat arms get one of. Total search budget is therefore
# 2x the binary/multi arms. That is the honest matching for an architecture with
# two models — per-stage budget is equal — but it IS a budget difference and any
# read of this cohort has to say so.
FLAT_EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]
HIER_EXPERIMENTS = [
	{"name": "S0: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S0: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
	{"name": "S1: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S1: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def experiments_for(arm: str) -> list:
	return HIER_EXPERIMENTS if arm == "hierarchical" else FLAT_EXPERIMENTS


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
		"SELECT name, config_json FROM flows WHERE name LIKE ? AND name NOT LIKE '%-v_' ORDER BY name",
		(SOURCE_LIKE,)).fetchall()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE ?", (f"%{SUFFIX}",))}
	con.close()

	if len(sources) != 15:
		print(f"REFUSED: expected the 15 original MCSD flows, found {len(sources)}")
		return 4
	print(f"re-running {len(sources)} MCSD flows under the automatic per-task CE anchor")
	if "--dry-run" in sys.argv:
		for n, cj in sources:
			p = json.loads(cj)["params"]
			a = p["ids_classification"]
			print(f"  would re-run {n} -> {n}{SUFFIX}  (seed {p['seed']}, arm {a}, "
			      f"{len(experiments_for(a))} phases, anchor AUTO)")
		print("DRY RUN — nothing created.")
		return 0

	created = []
	for sname, cj in sources:
		cfg = json.loads(cj)
		p = dict(cfg["params"])
		arm = p.get("ids_classification")
		# The anchor is derived automatically now; carrying the retired param
		# would only invite someone to think it still does something.
		p.pop("fitness_ce_anchor_normalized", None)
		name = f"{sname}{SUFFIX}"
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (
				f"Re-run of {sname} ({arm}) under the AUTOMATIC per-task CE anchor "
				f"(derived from this task's own train-label base-rate entropy; the "
				f"0.1937 param is retired). Hierarchical additionally gets S0:/S1: "
				f"stage-tagged phases so the cascade actually runs."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": experiments_for(arm),
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
		# `running` is a PASS: an idle worker admits the first flow the instant it
		# is queued, and the -v2 attempt reported that as a verification failure.
		expected = len(experiments_for(q.get("ids_classification")))
		ok = (ne == expected and st in ("queued", "running")
		      and "fitness_ce_anchor_normalized" not in q)
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} arm={q.get('ids_classification')}")
			bad += 1
	con.close()
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
