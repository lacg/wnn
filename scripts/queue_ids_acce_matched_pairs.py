"""AC/CE MATCHED-PAIR test across 4 IDS datasets — the falsifiable mechanism.

WHY (21/08/2026). The 23-arm IDSZ sweep at n=5 produced a FOUR-WAY TIE at the top
(CE20 89.78, B05-CE 89.58, B10-CE 89.56, CE40 89.55; MDD 0.409pp), and NO weight
axis explains the ordering — holding w_fpr fixed leaves 0.54-0.87pp of the 1.10pp
total spread intact. So "CE20's weights" is not a thing that can transfer, and a
24-arm sweep per dataset (37.8 days) would return the same tie four more times.

What DID replicate is a matched-pair contrast hiding inside that cohort:

    pair    AC arm    CE arm    delta      (held-out val_cal F1, n=5)
    B05      88.83     89.58    +0.74
    B10      88.74     89.56    +0.82
    B15      88.70     89.24    +0.54
                        mean    +0.70   -- 3/3 same sign, all > the 0.409pp bar

Identical weight STRUCTURE, only acc<->ce swapped. Mechanism: on a 44.94%-benign
set, accuracy is nearly redundant with F1, so w_acc buys no information, while CE
is a proper scoring rule that improves the RANKING the downstream threshold sweep
depends on. PRE-REGISTERED PREDICTION: CE > AC on every dataset, all three pairs.
A dataset where AC wins falsifies it.

ARMS PER DATASET (8):
  B05-AC / B05-CE, B10-AC / B10-CE, B15-AC / B15-CE   the mechanism (3 pairs)
  CE20                                                 does the UNSW leader transfer
  <local production>                                   Wb / Wb / Wc / Wa

The control is each dataset's OWN production vector, NOT Wb everywhere: Wb is
production only on the two unswr datasets. Wa == C35-CTRL (identical weights).
Running Wb on ciciot/cicids would be an extra arm, not a control.

All arms run under fitness_aggregation=zscore INCLUDING production, because the
SP100 cohorts are a pre-19/08 code era and cannot serve as controls.

Interleaved seed-major so round 1 is one seed of every dataset x arm.
Per CLAUDE.md Rule 2: dashboard POST /api/flows.
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

PAIRS = [
	("B05-AC", 0.225, 0.675, 0.05, 0.05), ("B05-CE", 0.675, 0.225, 0.05, 0.05),
	("B10-AC", 0.20,  0.60,  0.10, 0.10), ("B10-CE", 0.60,  0.20,  0.10, 0.10),
	("B15-AC", 0.175, 0.525, 0.15, 0.15), ("B15-CE", 0.525, 0.175, 0.15, 0.15),
]
CE20 = ("CE20", 0.20, 0.10, 0.30, 0.40)

# (tag, template flow, min/run, LOCAL production arm)
DATASETS = [
	("unswr-quad-64b", 4787,  24.1, ("Wb-CTRL", 0.10, 0.20, 0.35, 0.35)),
	("ciciot-quad-96b", 4789,  62.7, ("Wc-CTRL", 0.70, 0.10, 0.15, 0.05)),
	("cicids-quad-96b", 4788,  69.7, ("Wa-CTRL", 0.35, 0.30, 0.30, 0.05)),
	("unswr-qsr-64b",  4790, 296.5, ("Wb-CTRL", 0.10, 0.20, 0.35, 0.35)),
]

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def arms_for(prod):
	out, seen = [], set()
	for a in (*PAIRS, CE20, prod):
		if a[1:] not in seen:
			seen.add(a[1:]); out.append(a)
	return out


def purge_redundant() -> None:
	"""Wb-CTRL on ciciot/cicids is NOT those datasets' production — it was an
	extra arm masquerading as a control. Drop it rather than spend ~11 h on it."""
	con = ro()
	ids = [(r[0], r[1]) for r in con.execute(
		"SELECT id,name FROM flows WHERE (name LIKE 'IDSX-ciciot%Wb-CTRL%' "
		"OR name LIKE 'IDSX-cicids%Wb-CTRL%') AND status NOT IN ('completed','running')")]
	con.close()
	for fid, name in ids:
		r = requests.delete(f"{DASHBOARD}/api/flows/{fid}", verify=False, timeout=300)
		print(f"  - deleted {name} ({r.status_code})")
		time.sleep(0.1)
	print(f"  purged {len(ids)} non-control Wb flows")


def requeue_for_interleave() -> None:
	"""Delete still-QUEUED IDSX flows so the whole cohort can be recreated
	seed-major. Completed/running flows are kept — their names are skipped."""
	con = ro()
	ids = [(r[0], r[1]) for r in con.execute(
		"SELECT id,name FROM flows WHERE name LIKE 'IDSX-%' AND status='queued'")]
	con.close()
	for fid, name in ids:
		requests.delete(f"{DASHBOARD}/api/flows/{fid}", verify=False, timeout=300)
		time.sleep(0.1)
	print(f"  cleared {len(ids)} queued flows for re-interleaving")


def main() -> int:
	con = ro()
	bases, plan, total = {}, [], 0.0
	for tag, tpl, mins, prod in DATASETS:
		row = con.execute("SELECT config_json FROM flows WHERE id=?", (tpl,)).fetchone()
		if not row:
			print(f"ERROR: template {tpl} for {tag} missing"); return 1
		b = dict(json.loads(row[0])["params"])
		if b.get("patience") != 5 or b.get("ga_generations") != 250:
			print(f"ERROR: {tag} template not at patience=5/gens=250"); return 1
		bases[tag] = b
		arms = arms_for(prod)
		plan.append((tag, arms, mins))
		total += len(arms) * len(SEEDS) * mins
	con.close()

	print(f"seeds={SEEDS}   pre-registered prediction: CE > AC on all 3 pairs, every dataset\n")
	n = 0
	for tag, arms, mins in plan:
		k = len(arms) * len(SEEDS); n += k
		print(f"  {tag:<18}{len(arms)} arms x {len(SEEDS)} seeds = {k:>3} flows  "
		      f"{mins:>6.1f} min = {k*mins/60:>6.1f} h   [{', '.join(a[0] for a in arms)}]")
	print(f"\n  TOTAL {n} flows = {total/60:.0f} h = {total/1440:.1f} days")
	qsr = [p for p in plan if p[0] == 'unswr-qsr-64b'][0]
	qh = len(qsr[1]) * len(SEEDS) * qsr[2] / 60
	print(f"  of which unswr-qsr-64b: {qh:.0f} h = {qh/(total/60)*100:.0f}%")
	if "--dry-run" in sys.argv:
		print("\nDRY RUN — nothing purged, nothing created."); return 0

	print("\npurging non-control arms + clearing queue for interleave...")
	purge_redundant()
	requeue_for_interleave()

	con = ro()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'IDSX-%'")}
	con.close()
	print(f"keeping {len(existing)} completed/running flow(s): {sorted(existing)}\n")

	created = []
	for seed in SEEDS:                       # seed-major => round 1 covers everything
		for tag, arms, _ in plan:
			for arm, ce, acc, f1, fpr in arms:
				name = f"IDSX-{tag}-{arm}-r{seed}"
				if name in existing:
					continue
				p = dict(bases[tag])
				p["seed"] = seed
				p["fitness_aggregation"] = "zscore"
				p["fitness_zrank_clamp"] = 3.0
				p["fitness_weight_ce"], p["fitness_weight_acc"] = ce, acc
				p["fitness_weight_f1"], p["fitness_weight_fpr"] = f1, fpr
				body = {
					"name": name,
					"description": (
						f"AC/CE MATCHED-PAIR test. {tag}, arm={arm} "
						f"(ce {ce}/acc {acc}/f1 {f1}/fpr {fpr}), seed={seed}. "
						f"Tests the ONE contrast that replicated in IDSZ: identical weight "
						f"structure, acc<->ce swapped, CE won 3/3 pairs by +0.70pp mean "
						f"(all > the 0.409pp 80%-power bar). Control is this dataset's OWN "
						f"production vector. All arms under fitness_aggregation=zscore. "
						f"Prediction: CE > AC. A dataset where AC wins falsifies it."
					),
					"config": {"template": "ids-binary-2-phase", "params": p},
					"experiments": EXPERIMENTS,
				}
				r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
				if r.status_code not in (200, 201):
					print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}"); return 2
				created.append((r.json()["id"], name))
				time.sleep(0.2)
		print(f"  seed {seed}: {len(created)} created so far")

	print(f"\nflipping {len(created)} pending -> queued...")
	for fid, _ in created:
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.12)

	con = ro(); bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		if not (ne == 2 and st in ("queued", "running") and q.get("patience") == 5
		        and q.get("ga_generations") == 250 and q.get("fitness_aggregation") == "zscore"):
			print(f"  x id={fid} {name} status={st} exps={ne}"); bad += 1
	tot = con.execute("SELECT COUNT(*) FROM flows WHERE name LIKE 'IDSX-%'").fetchone()[0]
	con.close()
	print(f"VERIFY: {len(created)-bad}/{len(created)} queued OK; IDSX cohort now {tot} flows")
	return 0 if bad == 0 else 3


if __name__ == "__main__":
	sys.exit(main())
