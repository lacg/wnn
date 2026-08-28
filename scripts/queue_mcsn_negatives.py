"""MCSN — does suppressing NEGATIVE writes drain the Worms sink?

THE QUESTION
------------
BINARY is the only thing that ever moved the sink: Worms over-absorption fell
415x -> 10.1x (n=1, flow 6010 vs 6005). BINARY differs from QUAD in TWO ways:
  (1) an unwritten cell scores 0.0 instead of the 0.25 WEAK_FALSE pedestal
  (2) NEGATIVES ARE IGNORED — no class ever writes FALSE

(1) has now been tested three ways and REFUTED as the driver. The coverage-aware
scorer silences the pedestal at read time; constant-fill sizing removes it at
train time at two different bands. Across six cells (Worms at b=34/7/4, i.e.
~0%/17%/78% coverage) over-absorption never left 508-618x.

So (2) is what is left. This arm suppresses negatives in QUAD, holding the
pedestal and everything else fixed:

  MCSN-neg0   num_negatives=0, band 34-50 legacy rule, coverage off
  control     = MCSP-a0b0 (flows 6015-6019), IDENTICAL except negatives ON

PREDICTION: if negatives drive the sink, over-absorption collapses toward the
~10x BINARY reached. FALSIFIED IF it stays in the 500x band — that would mean
neither of BINARY's two changes explains it alone, and the cause is either
their combination or something else about the 1-bit read entirely.

MECHANISM, stated so the result can be read: with negatives suppressed each
class writes only TRUE on its own rows and is never taught to reject. Every
class's score can then only rise, so the DEFENCE against a rare class is no
longer "the true class says no" but purely "the true class says yes louder".
That may drain the sink or may flood everything — both are informative.

Multiclass S1 only: the binary S0 gate already skips negatives in the trainer
(`if !is_binary && num_negatives > 0`), so the gate is untouched either way.

READ-OUT (pre-registered): Worms over-absorption and rho(train support,
over-absorption) FIRST, then macro-F1 / benign FPR / accuracy and the per-class
recall table. Expect large collateral movement — a class that cannot be
rejected is a different classifier, not a tweak.

SMOKE: s20401 only; the other four stay pending.
"""
import json, sqlite3, sys, time
import requests, urllib3
urllib3.disable_warnings()

DASHBOARD = "https://localhost:3000"
DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
SOURCE_PREFIX = "MCSP-a0b0-unswt-16b-hier-s"
SEEDS = [20401, 20402, 20403, 20404, 20405]
SMOKE_SEED = 20401
HIER = [
	{"name": "S0: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S0: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
	{"name": "S1: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S1: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def main() -> int:
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	row = con.execute("SELECT config_json FROM flows WHERE name=?", (SOURCE_PREFIX + "20401",)).fetchone()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'MCSN-%'")}
	con.close()
	if not row:
		print("REFUSED: control flow MCSP-a0b0-...-s20401 not found")
		return 4
	cfg = json.loads(row[0])

	created = []
	for seed in SEEDS:
		p = dict(cfg["params"])
		p["ids_suppress_negatives"] = True     # <- THE ONLY CHANGE vs a0b0
		p["seed"] = seed
		name = f"MCSN-neg0-unswt-16b-hier-s{seed}"
		if "--dry-run" in sys.argv:
			base = cfg["params"]
			print(f"  would create {name}  diff-vs-control: "
			      f"{ {k: (base.get(k), p[k]) for k in p if base.get(k) != p[k]} }")
			continue
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (
				"MCSN: suppress NEGATIVE writes (num_negatives=0) in QUAD, holding the 0.25 "
				"pedestal and everything else fixed. Control = MCSP-a0b0 (same seed, negatives ON). "
				"Isolates the second of BINARY's two changes; the first (the pedestal) is already "
				"REFUTED across six cells at Worms b=34/7/4. Read out on Worms over-absorption "
				"FIRST, not macro-F1. See scripts/queue_mcsn_negatives.py."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": HIER,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=60)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], name, seed))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	if "--dry-run" in sys.argv:
		print("DRY RUN — nothing created.")
		return 0

	for fid, name, seed in created:
		if seed == SMOKE_SEED:
			requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=60)
			print(f"  -> queued SMOKE {fid} {name}")
		else:
			print(f"  . left pending {fid} {name}")

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = 0
	for fid, name, seed in created:
		st, = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()
		ne, = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		want = "queued" if seed == SMOKE_SEED else "pending"
		if not (ne == 4 and st in (want, "running") and q.get("ids_suppress_negatives") is True):
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} supp={q.get('ids_suppress_negatives')}")
			bad += 1
	con.close()
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
