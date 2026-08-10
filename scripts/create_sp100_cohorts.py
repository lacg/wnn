"""Create the SP100 cohorts: 100 CLEAN runs x 5 cohorts = 500 flows.

Decision (Luiz 09/08/2026, from the SP-abl 5-table report): lock QUAD_WEIGHTED for
all four datasets, 100 fresh flows each (NOT 90+10 — the old 10 ran on the
pre-8b839a30 positional-rank optimizer), PLUS a 100-run QSR cohort on unswr (the
one dataset where QSR beat QUAD, +0.79pp F1 / -0.46pp FPR at 15x the cost).

Config provenance: byte-copied from the reference flows (the config the report
ranked) with ONLY the seed varied — QUAD: 4404 (unswt) / 4405 (unswr) / 4406
(cicids) / 4407 (ciciot), memory_mode ABSENT = worker default QUAD_WEIGHTED;
QSR: 4405 + memory_mode="QSR" (verified the only param differing in 4656).

Seeds: each cohort = the dataset's 10 registry seeds (so 10 pairs are directly
comparable old-vs-new-optimizer — the IDS-side tie-fix A/B, free) + 90 fresh from
a per-cohort deterministic RNG. Names: SP100-{ds}-{mode}-{enc}-r{seed}.

Creation is ROUND-ROBIN across the 5 cohorts (the interleave rule: the FIFO
worker queue then yields one run of every cohort early, enabling early reads).

Flows go through POST /api/flows ONLY (CLAUDE.md Rule 2) — the DB is opened
read-only for reference configs, existence checks and post-verification.

Usage:
  create_sp100_cohorts.py --dry-run     # print the plan, POST nothing
  create_sp100_cohorts.py               # create + verify (idempotent by name)
"""
import argparse
import json
import random
import sqlite3
import sys

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API = "https://localhost:3000/api/flows"

# cohort tag -> (reference flow id, memory_mode override or None, name encoding)
COHORTS = {
	"unswt-quad":  (4404, None,  "16bWb"),
	"unswr-quad":  (4405, None,  "64bWb"),
	"cicids-quad": (4406, None,  "96bWa"),
	"ciciot-quad": (4407, None,  "96bWc"),
	"unswr-qsr":   (4405, "QSR", "64bWb"),
}
RUNS_PER_COHORT = 100

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search",
	 "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def registry_seeds(db, ref_id):
	"""The dataset's 10 canonical seeds = the 10 QUAD baseline flows the 5-table
	report identified (ids 4404-4452; a bare name-prefix match is WRONG — unswt's
	'...-n30-r%' prefix also matches the older 30-seed UNSW-temp cohort)."""
	prefix = db.execute("select name from flows where id=?", (ref_id,)).fetchone()[0]
	prefix = prefix.rsplit("-r", 1)[0]            # SP-unswt-bin-16bWb-n30
	rows = db.execute(
		"select json_extract(config_json,'$.params.seed') from flows "
		"where id between 4404 and 4452 and name like ? || '-r%' order by 1",
		(prefix,)).fetchall()
	seeds = [int(r[0]) for r in rows]
	if len(seeds) != 10:
		sys.exit(f"expected 10 registry seeds for {prefix} in ids 4404-4452, "
		         f"found {len(seeds)}")
	return seeds


def cohort_seeds(db, tag, ref_id):
	"""10 registry seeds + 90 fresh (deterministic per cohort, 5-digit, unique)."""
	seeds = registry_seeds(db, ref_id)
	rng = random.Random(f"SP100-{tag}")
	seen = set(seeds)
	while len(seeds) < RUNS_PER_COHORT:
		s = rng.randint(10000, 99999)
		if s not in seen:
			seen.add(s)
			seeds.append(s)
	return seeds


def build_flows(db):
	"""[(name, body)] in round-robin cohort order (the interleave rule)."""
	plans = {}
	for tag, (ref_id, mode, enc) in COHORTS.items():
		cfg = json.loads(db.execute(
			"select config_json from flows where id=?", (ref_id,)).fetchone()[0])
		if mode is None:
			cfg["params"].pop("memory_mode", None)   # absent = QUAD_WEIGHTED default
		else:
			cfg["params"]["memory_mode"] = mode
		ds, m = tag.rsplit("-", 1)
		plans[tag] = [(f"SP100-{ds}-{m}-{enc}-r{seed}",
		               {**cfg, "params": {**cfg["params"], "seed": seed}})
		              for seed in cohort_seeds(db, tag, ref_id)]
	flows = []
	for i in range(RUNS_PER_COHORT):
		for tag in COHORTS:
			name, cfg = plans[tag][i]
			flows.append((name, {
				"name": name,
				"description": f"SP100 cohort {tag} (post-8b839a30 optimizer), run {i + 1}/100",
				"config": cfg,
				"experiments": EXPERIMENTS,
			}))
	return flows


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dry-run", action="store_true")
	args = ap.parse_args()

	db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
	flows = build_flows(db)
	existing = {r[0] for r in db.execute(
		"select name from flows where name like 'SP100-%'").fetchall()}

	print(f"plan: {len(flows)} flows ({len(COHORTS)} cohorts x {RUNS_PER_COHORT}), "
	      f"{len(existing)} already exist (skipped)")
	if args.dry_run:
		for name, body in flows[:6]:
			p = body["config"]["params"]
			print(f"  {name}: seed={p['seed']} memory_mode={p.get('memory_mode', '(absent=QUAD)')} "
			      f"dataset={p.get('ids_dataset')}/{p.get('ids_split')} experiments={len(body['experiments'])}")
		print(f"  ... {len(flows) - 6} more")
		return

	created = failed = 0
	for name, body in flows:
		if name in existing:
			continue
		r = requests.post(API, json=body, verify=False, timeout=30)
		if r.status_code in (200, 201):
			created += 1
			# POST creates flows as "pending"; the worker polls status="queued"
			# ONLY (worker.py list_flows(status="queued")). Without this PATCH the
			# cohort sits forever looking healthy — 500 rows, all with their
			# experiments, and zero work done. Found 10/08/2026 after the first
			# 500-flow batch idled for 35 min with a live worker.
			fid = (r.json() or {}).get("id")
			if fid is None:
				failed += 1
				print(f"  {name}: created but no id returned — cannot queue")
			else:
				q = requests.patch(f"{API}/{fid}", json={"status": "queued"},
				                   verify=False, timeout=30)
				if q.status_code != 200:
					failed += 1
					print(f"  {name}: created but QUEUE failed {q.status_code}")
		else:
			failed += 1
			print(f"  FAILED {name}: {r.status_code} {r.text[:200]}")
			if failed >= 3:
				sys.exit("3 failures — aborting before flooding the queue with a bad spec")
	print(f"created {created}, skipped {len(existing)}, failed {failed}")

	# Rule 2 #5: verify every created flow actually carries its experiments.
	db2 = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
	n, bad, unq = db2.execute(
		"select count(*), sum(case when (select count(*) from experiments e "
		"where e.flow_id=f.id)=0 then 1 else 0 end), "
		"sum(case when f.status='pending' then 1 else 0 end) "
		"from flows f where f.name like 'SP100-%'").fetchone()
	print(f"verify: {n} SP100 flows, {bad or 0} with ZERO experiments, "
	      f"{unq or 0} still 'pending' (invisible to the worker)")
	if n < len(flows) or (bad or 0) > 0 or (unq or 0) > 0:
		sys.exit("VERIFY FAILED — do not trust this cohort until fixed")
	print("verify OK — cohort queued and visible to the worker")


if __name__ == "__main__":
	main()
