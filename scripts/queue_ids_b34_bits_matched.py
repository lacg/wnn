"""B34 BITS-MATCHED arm — does QSR still beat QUAD when bit-width cannot vary?

WHY (23/08/2026). The IDSX cohort produced an unplanned finding much larger than
the AC/CE question it was designed for: QSR is the only decode that yields a
TUNABLE score. QSR vs QUAD is +0.889 +/- 0.100 pp held-out F1 and -0.566 pp FPR,
8/8 arms, while BINARY / BINARY-wide / TERNARY / QUAD all sit pinned at FPR
~1.10-1.12 as the threshold wanders 0.50-0.98 (their confusion matrix moves 4
samples out of 152,390). Adjudicated by experiment-design 23/08 -> SUPPORTED.
See memory `project_qsr_decode_tunable_score`; do NOT re-derive it.

The bits confound was already REFUTED once: a matched-34-bit GRID slice of the
existing runs gives the same +0.87 pp, and the abl2big ceiling control came back
null. This arm closes it a second time by CONSTRUCTION rather than by slicing --
`min_bits` is raised 4 -> 34 so it meets `max_bits` and every neuron is pinned at
the cap. The GA can then only move neurons, never bits, so no bit-width
difference between the two decodes can survive to explain the gap.

    10 flows: IDSX-unswr-{quad,qsr}-64b-B34-CTRL-r20401..r20405
    base:     verbatim config of IDSX-unswr-{quad,qsr}-64b-Wb-CTRL-r20401
    delta:    min_bits 4 -> 34   (and the per-run seed).  NOTHING else.

NOT COMPARABLE TO Wb-CTRL. Pinning bits collapses the grid phase to
neurons-only, so B34 is a self-contained contrast: B34-qsr vs B34-quad, paired
per seed. Reading it against the free-bits Wb-CTRL numbers would confound the
decode with the search space.

PRE-REGISTERED READ: held-out val_cal F1, genome_type best_f1, phase ga_neurons,
paired per seed, mean of the 5 paired deltas.
PREDICTION: ~+0.87 pp for QSR.
FALSIFIER:  |mean dF1| < 0.14 pp kills the substrate claim.

RUNS LAST. The worker's admit() takes min(id) among status='queued', and every
live queued flow is <= 5816, so these (5817+) drain only after the 124 remaining
IDSX flows. Cost ~24 h on top of the cohort.

Per CLAUDE.md Rule 2: created through the dashboard POST /api/flows, never by
direct SQL insert.
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
PINNED_BITS = 34
ARM = "B34-CTRL"

# (tag, source Wb-CTRL flow to clone verbatim)
SOURCES = [
	("unswr-quad-64b", "IDSX-unswr-quad-64b-Wb-CTRL-r20401"),
	("unswr-qsr-64b",  "IDSX-unswr-qsr-64b-Wb-CTRL-r20401"),
]

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def load_base(con, tag: str, src_name: str) -> dict:
	"""Clone the source arm's params verbatim, then pin bits at the cap."""
	row = con.execute("SELECT config_json FROM flows WHERE name=?", (src_name,)).fetchone()
	if not row:
		raise SystemExit(f"ERROR: source flow {src_name} missing")
	p = dict(json.loads(row[0])["params"])
	if p.get("max_bits") != PINNED_BITS:
		raise SystemExit(f"ERROR: {tag} max_bits={p.get('max_bits')} != {PINNED_BITS}; "
		                 "pinning min_bits there would not sit at the cap")
	if p.get("patience") != 5 or p.get("ga_generations") != 250:
		raise SystemExit(f"ERROR: {tag} source not at patience=5/gens=250")
	if p.get("fitness_aggregation") != "zscore":
		raise SystemExit(f"ERROR: {tag} source not under fitness_aggregation=zscore")
	p["min_bits"] = PINNED_BITS
	return p


def describe(tag: str, seed: int) -> str:
	return (
		f"B34 BITS-MATCHED arm. {tag}, min_bits=max_bits={PINNED_BITS} pins every neuron "
		f"at the bit cap, seed={seed}. Closes the bits confound on the QSR-vs-QUAD "
		f"finding BY CONSTRUCTION: the GA can move neurons only, so no bit-width "
		f"difference can explain the decode gap. Verbatim clone of "
		f"IDSX-{tag}-Wb-CTRL except min_bits 4->{PINNED_BITS}. NOT comparable to "
		f"Wb-CTRL (the grid collapses to neurons-only) -- read ONLY B34-qsr vs "
		f"B34-quad, paired per seed, held-out val_cal F1 / best_f1 / ga_neurons. "
		f"Prediction ~+0.87pp for QSR; falsifier |mean dF1| < 0.14pp."
	)


def main() -> int:
	con = ro()
	bases = {tag: load_base(con, tag, src) for tag, src in SOURCES}
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'IDSX-%B34-%'")}
	backlog = con.execute(
		"SELECT COUNT(*), COALESCE(MAX(id),0) FROM flows WHERE status IN ('queued','running')").fetchone()
	con.close()

	planned = [(tag, seed) for seed in SEEDS for tag, _ in SOURCES]   # seed-major interleave
	print(f"B34 bits-matched arm: {len(planned)} flows "
	      f"({len(SOURCES)} decodes x {len(SEEDS)} seeds), min_bits={PINNED_BITS}")
	for tag, src in SOURCES:
		b = bases[tag]
		print(f"  {tag:<16} mode={b.get('memory_mode','QUAD_WEIGHTED (default)'):<22} "
		      f"bits {b['min_bits']}-{b['max_bits']}  neurons {b['min_neurons']}-{b['max_neurons']}  "
		      f"w(ce/acc/f1/fpr)={b['fitness_weight_ce']}/{b['fitness_weight_acc']}/"
		      f"{b['fitness_weight_f1']}/{b['fitness_weight_fpr']}  <- {src}")
	print(f"  runs LAST: {backlog[0]} flow(s) already queued/running, max id {backlog[1]}; "
	      f"admit() takes min(id)")
	if existing:
		print(f"  skipping {len(existing)} already present: {sorted(existing)}")
	if "--dry-run" in sys.argv:
		print("\nDRY RUN -- nothing created."); return 0

	created = []
	for tag, seed in planned:
		name = f"IDSX-{tag}-{ARM}-r{seed}"
		if name in existing:
			continue
		p = dict(bases[tag]); p["seed"] = seed
		body = {
			"name": name,
			"description": describe(tag, seed),
			"config": {"template": "ids-binary-2-phase", "params": p},
			"experiments": EXPERIMENTS,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}"); return 2
		created.append((r.json()["id"], name))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	print(f"\nflipping {len(created)} pending -> queued (tail of the FIFO)...")
	for fid, _ in created:
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.12)

	con = ro(); bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		at = {r[0] for r in con.execute("SELECT architecture_type FROM experiments WHERE flow_id=?", (fid,))}
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		ok = (ne == 2 and st == "queued" and at == {"ids"} and q.get("min_bits") == PINNED_BITS
		      and q.get("max_bits") == PINNED_BITS and q.get("patience") == 5
		      and q.get("ga_generations") == 250 and q.get("ids_k_folds") == 5
		      and q.get("fitness_aggregation") == "zscore")
		if not ok:
			print(f"  x id={fid} {name} status={st} exps={ne} arch={at} "
			      f"bits={q.get('min_bits')}-{q.get('max_bits')}"); bad += 1
	ahead = con.execute("SELECT COUNT(*) FROM flows WHERE status IN ('queued','running') AND id<?",
	                    (min(f for f, _ in created),)).fetchone()[0] if created else 0
	con.close()
	print(f"VERIFY: {len(created)-bad}/{len(created)} queued OK, 2 experiments each, "
	      f"bits pinned {PINNED_BITS}-{PINNED_BITS}")
	print(f"        {ahead} flow(s) ahead of them in the FIFO -- B34 drains last")
	return 0 if bad == 0 else 3


if __name__ == "__main__":
	sys.exit(main())
