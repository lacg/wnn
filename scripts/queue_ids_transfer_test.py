"""Cross-dataset TRANSFER test for the CE20 weight vector — 4 datasets, 3 arms, n=5.

WHY (21/08/2026). CE20 (ce .20 / acc .10 / f1 .30 / fpr .40) beat production
Wb-CTRL on UNSW-NB15 temporal_3way 16b by +0.95pp F1 (+4.17 sigma) at IDENTICAL
search budget (both 62 mean gens), dominating 19/35 columns, with ~25% fewer
neurons. The open question is whether that TRANSFERS: production already uses a
DIFFERENT weight vector per dataset (Wb / Wa / Wc), so the project's own standing
position is that weights are dataset-specific.

DESIGN. A full 23-arm sweep per dataset costs ~12 days. The question is not
"which of 23 wins here" but "does CE20 beat the local production vector", so each
dataset gets 3 arms:
    CE20                      the challenger
    <dataset production>      the local incumbent (Wb / Wa / Wc)
    Wb-CTRL                   a COMMON reference across all four datasets
De-duplicated: on the two unswr datasets production IS Wb, so those get 2 arms.

ALL arms run under fitness_aggregation=zscore, INCLUDING the production arm. That
is deliberate: the SP100 cohorts were run on a pre-19/08 selector with a known
weight-handling defect, so their numbers are a different code era and cannot serve
as controls (measured: SP100 genome shapes 60+-92 neurons vs 378+-108 today).
Re-running production inside this cohort is the only apples-to-apples control.

Everything else is cloned VERBATIM from each dataset's SP100 template flow —
patience 5, ga_generations 250, same split/bits/features. Only `seed`, the four
fitness weights, and `fitness_aggregation` differ.

Per CLAUDE.md Rule 2: dashboard POST /api/flows with experiments, never SQL.
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
SEEDS = [20401, 20402, 20403, 20404, 20405]

CE20 = ("CE20", 0.20, 0.10, 0.30, 0.40)
WB   = ("Wb-CTRL", 0.10, 0.20, 0.35, 0.35)

# (short tag, template flow, measured min/run, local production arm)
DATASETS = [
	("unswr-quad-64b", 4787, 24.1, WB),
	("ciciot-quad-96b", 4789, 62.7, ("Wc-CTRL", 0.70, 0.10, 0.15, 0.05)),
	("cicids-quad-96b", 4788, 69.7, ("Wa-CTRL", 0.35, 0.30, 0.30, 0.05)),
	("unswr-qsr-64b",  4790, 296.5, WB),
]

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def arms_for(prod) -> list:
	"""CE20 + local production + Wb reference, de-duplicated by weight vector."""
	out, seen = [], set()
	for a in (CE20, prod, WB):
		if a[1:] in seen:
			continue
		seen.add(a[1:])
		out.append(a)
	return out


def main() -> int:
	con = ro()
	plan, total_min = [], 0.0
	for tag, tpl, mins, prod in DATASETS:
		row = con.execute("SELECT config_json FROM flows WHERE id=?", (tpl,)).fetchone()
		if not row:
			print(f"ERROR: template flow {tpl} for {tag} not found"); return 1
		base = dict(json.loads(row[0])["params"])
		if base.get("patience") != 5 or base.get("ga_generations") != 250:
			print(f"ERROR: {tag} template has patience={base.get('patience')} "
			      f"gens={base.get('ga_generations')}, expected 5/250"); return 1
		arms = arms_for(prod)
		plan.append((tag, base, arms, mins))
		total_min += len(arms) * len(SEEDS) * mins
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'IDSX-%'")}
	con.close()

	print(f"seeds={SEEDS}\n")
	n_flows = 0
	for tag, base, arms, mins in plan:
		n = len(arms) * len(SEEDS)
		n_flows += n
		print(f"  {tag:<18} arms={[a[0] for a in arms]}  {n:>2} flows  "
		      f"{mins:>5.1f} min/run  = {n*mins/60:5.1f} h")
	print(f"\n  TOTAL {n_flows} flows  ~{total_min/60:.1f} h ({total_min/1440:.1f} days)")
	if existing:
		print(f"  (skipping {len(existing)} IDSX flows that already exist)")
	if "--dry-run" in sys.argv:
		print("\nDRY RUN — nothing created."); return 0

	created = []
	for tag, base, arms, _ in plan:
		for seed in SEEDS:
			for arm, ce, acc, f1, fpr in arms:
				name = f"IDSX-{tag}-{arm}-r{seed}"
				if name in existing:
					continue
				p = dict(base)
				p["seed"] = seed
				p["fitness_aggregation"] = "zscore"
				p["fitness_zrank_clamp"] = 3.0
				p["fitness_weight_ce"], p["fitness_weight_acc"] = ce, acc
				p["fitness_weight_f1"], p["fitness_weight_fpr"] = f1, fpr
				body = {
					"name": name,
					"description": (
						f"CROSS-DATASET TRANSFER TEST for the CE20 weight vector. "
						f"{tag}, arm={arm} (ce {ce}/acc {acc}/f1 {f1}/fpr {fpr}), seed={seed}. "
						f"Config cloned verbatim from the SP100 template (patience=5, "
						f"ga_generations=250); only seed, the 4 weights and "
						f"fitness_aggregation=zscore differ. The production arm is re-run "
						f"HERE under zscore because the SP100 cohorts are a pre-19/08 code "
						f"era and cannot serve as controls. Question: does CE20's UNSW-"
						f"temporal win (+0.95pp F1, +4.17 sigma, budget-matched) transfer?"
					),
					"config": {"template": "ids-binary-2-phase", "params": p},
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
		q = cfg["params"]
		ok = (ne == 2 and st in ("queued", "running") and q.get("patience") == 5
		      and q.get("ga_generations") == 250 and q.get("fitness_aggregation") == "zscore")
		if not ok:
			print(f"  x id={fid} status={st} exps={ne} patience={q.get('patience')} "
			      f"gens={q.get('ga_generations')} agg={q.get('fitness_aggregation')}  {name}")
			bad += 1
	con.close()
	print(f"VERIFY: {len(created)-bad}/{len(created)} queued with patience=5, gens=250, agg=zscore")
	return 0 if bad == 0 else 3


if __name__ == "__main__":
	sys.exit(main())
