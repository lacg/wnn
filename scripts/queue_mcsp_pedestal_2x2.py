"""Queue the MCSP pedestal cohort — docs/MCST_PEDESTAL_2X2_SPEC.md.

THREE arms x 5 seeds = 15 runs. The 2x2's fourth cell (A1B1) is DROPPED, not
forgotten: A1's per-class bits are 4-12, all at or below SPARSE_THRESHOLD=12, so
every group is dense; dense reads the real cell and never consults the sparse
miss default, therefore ids_coverage_aware cannot act and A1B1 would be
bit-identical to A1B0. Adjudicated with Luiz 28/08/2026. Restoring that cell
requires dense coverage tracking (a 1-bit-per-cell bitmap) first.

  A0B0  logratio 34-50, cov=off  -> control; reproduces MCST tiered3 (6005-6009)
  A0B1  logratio 34-50, cov=ON   -> route B alone, in the sparse regime where it acts
  A1B0  constfill 4-14, cov=off  -> route A alone, density instead of read-side

WHAT IS BEING TESTED: in QUAD an UNTOUCHED cell commits to WEAK_FALSE=0.25 while
a LEARNED rejection commits to FALSE=0.0, and the score is a mean over the
class's own neurons — so ignorance outranks knowledge and the emptiest class
wins argmax by abstention. UNSW Worms (97 train rows) absorbs 508x its fair
share of every misclassification; over-absorption vs train support is rho=-0.930
over ten classes. A attacks it at TRAIN time (size the space so it gets
populated), B at READ time (a miss stops voting).

PRE-REGISTERED PRIMARY READ-OUT — the MECHANISM, not the headline:
  1. Worms over-absorption per cell (A0B0 baseline ~460-530x)
  2. rho(train support, over-absorption) across the 10 classes (A0B0 ~ -0.93)
Secondary: macro-F1, benign FPR, accuracy, and the per-class recall table
(MANDATORY — QSR lesson: an aggregate win with recall losses is NOT "detects
better"). Expect B to trade macro-F1 for benign FPR: the n=1 BINARY probe
(6010 vs 6005) drained the sink 415x->10.1x but cost -2.28pp macro-F1 with
Worms TP 17->6, while gaining -5.82pp FPR. Expect A1 to risk accuracy outright:
it puts Exploits/Generic at 12 bits vs 44/45. If it does, THAT IS THE FINDING.
FALSIFIED IF: A1B0 leaves Worms over-absorption above ~100x — that would mean
evidence density is not what drives the sink.

CONFOUND: ids_coverage_aware drives GA fitness as well as decode (Luiz 28/08),
so deltas are "treatment + search trajectory", never the treatment alone.

SMOKE PROTOCOL: all three arms are new code paths (the logratio param, the
coverage flag, and the 4-14 band), so ONE seed of EACH arm is queued and the
other 12 are left pending.
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
SOURCE = "MCST-unswt-quad-16b-hier-s20401-tiered3"
SEEDS = [20401, 20402, 20403, 20404, 20405]
SMOKE_SEED = 20401
HIER_EXPERIMENTS = [
	{"name": "S0: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S0: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
	{"name": "S1: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S1: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]

ARMS = {
	"a0b0": {"ids_tier_bits_rule": "logratio", "ids_tier_bits_min": 34, "ids_tier_bits_max": 50,
	         "min_bits": 34, "max_bits": 50, "ids_coverage_aware": False},
	"a0b1": {"ids_tier_bits_rule": "logratio", "ids_tier_bits_min": 34, "ids_tier_bits_max": 50,
	         "min_bits": 34, "max_bits": 50, "ids_coverage_aware": True},
	"a1b0": {"ids_tier_bits_rule": "constant_fill", "ids_tier_bits_min": 4, "ids_tier_bits_max": 14,
	         "min_bits": 4, "max_bits": 14, "ids_coverage_aware": False},
}
BLURB = {
	"a0b0": "CONTROL — reproduces MCST tiered3 sizing (legacy logratio rule, band 34-50), default scorer.",
	"a0b1": "ROUTE B — coverage-aware scorer in the sparse regime: a miss scores 0.0, not the 0.25 pedestal.",
	"a1b0": "ROUTE A — constant-fill sizing (band 4-14): give each class a space it can populate, so there are no untouched cells to score.",
}


def main() -> int:
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	row = con.execute("SELECT config_json FROM flows WHERE name=?", (SOURCE,)).fetchone()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'MCSP-%'")}
	con.close()
	if not row:
		print(f"REFUSED: source flow {SOURCE} not found")
		return 4
	cfg = json.loads(row[0])

	plan = []
	for arm, overrides in ARMS.items():
		for seed in SEEDS:
			p = dict(cfg["params"])
			p.update(overrides)
			p["seed"] = seed
			plan.append((arm, seed, f"MCSP-{arm}-unswt-16b-hier-s{seed}", p))

	if "--dry-run" in sys.argv:
		for arm, seed, name, p in plan:
			d = {k: p[k] for k in ARMS[arm]}
			print(f"  would create {name}  {d}")
		print(f"DRY RUN — {len(plan)} flows, nothing created.")
		return 0

	created = []
	for arm, seed, name, p in plan:
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (f"MCSP pedestal cohort ({arm}). {BLURB[arm]} "
			                "Primary read-out is Worms over-absorption and rho(support, "
			                "over-absorption), NOT macro-F1. The 2x2's a1b1 cell is dropped: "
			                "A1 is entirely dense (bits<=12) so coverage_aware cannot act. "
			                "See docs/MCST_PEDESTAL_2X2_SPEC.md."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": HIER_EXPERIMENTS,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=60)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], arm, seed, name))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	for fid, arm, seed, name in created:
		if seed == SMOKE_SEED:
			requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=60)
			print(f"  -> queued SMOKE {fid} {name}  ({arm})")
		else:
			print(f"  . left pending {fid} {name}")

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = 0
	for fid, arm, seed, name in created:
		st, = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()
		ne, = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		want = "queued" if seed == SMOKE_SEED else "pending"
		ok = ne == 4 and st in (want, "running") and all(q.get(k) == v for k, v in ARMS[arm].items())
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} "
			      f"params={ {k: q.get(k) for k in ARMS[arm]} }")
			bad += 1
	con.close()
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
