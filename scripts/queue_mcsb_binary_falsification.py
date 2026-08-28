"""Queue the MCSB BINARY falsification of the ABSTENTION mechanism.

WHAT IT TESTS (pre-registered, 28/08/2026)
------------------------------------------
Verified analytically: in QUAD_WEIGHTED an UNTOUCHED cell commits to
WEAK_FALSE = 0.25 (`oi_bin_to_cell`: obs==0 -> QUAD_WEAK_FALSE), while a class
that LEARNED to reject a row commits to FALSE = 0.0 (obs>=2, net<=-1). The
per-class score is a mean over that class's own neurons
(`compute_per_example_scores`: sum / actual_neurons), so an all-empty class
sits at a flat 0.25 pedestal and raw argmax hands it every row where the true
class's mean drops below 0.25. Worms (97 train rows vs a 2^50 address space)
is the emptiest class and absorbs 508x its fair share of ALL misclassifications;
over-absorption vs train support is rho = -0.930 (n=10).

BINARY (mode 3) removes the pedestal: unwritten -> FALSE -> 0.0 ("never seen ->
no vote"), TRUE -> 1.0. So EVERY class floors at 0.0 and abstention can no
longer beat a confident rejection.

PREDICTION: Worms over-absorption collapses from ~508x toward its fair share
(~1x), and rho(train support, over-absorption) weakens sharply.
FALSIFIED IF: Worms still over-absorbs by two orders of magnitude under BINARY
— that would mean the sink is driven by address-space collision, not by the
0.25 pedestal.

CONFOUND, stated up front: BINARY also IGNORES NEGATIVES (per-discriminator
classical training). It therefore removes the pedestal AND the negative writes
at once. Both changes push the SAME way, so a null result is decisive against
the abstention model while a positive result is CONSISTENT WITH it rather than
uniquely diagnostic. Do not report this as isolating the pedestal alone.

Controls = the paired MCST-*-tiered3 QUAD flows (6005-6009), same seeds, same
bits band 34-50, same neuron cap. memory_mode is the ONLY changed param.
Read out on OVER-ABSORPTION, not on macro-F1 (BINARY may simply be worse
overall; that is not what is being tested).

SMOKE PROTOCOL: queues s20401 ONLY; the other four are left pending.
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
SMOKE_SEED = 20401
HIER_EXPERIMENTS = [
	{"name": "S0: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S0: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
	{"name": "S1: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S1: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def main() -> int:
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	sources = con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE 'MCST-unswt-quad-16b-hier-%-tiered3' ORDER BY name"
	).fetchall()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'MCSB-%'")}
	con.close()
	if len(sources) != 5:
		print(f"REFUSED: expected the 5 MCST tiered3 flows, found {len(sources)}")
		return 4

	plan = []
	for sname, cj in sources:
		cfg = json.loads(cj)
		p = dict(cfg["params"])
		p["memory_mode"] = "BINARY"          # <- THE ONLY CHANGE
		name = sname.replace("MCST-unswt-quad", "MCSB-unswt-binary")
		plan.append((sname, name, cfg, p))

	if "--dry-run" in sys.argv:
		for sname, name, cfg, p in plan:
			base = json.loads(dict(sources)[sname])["params"]
			diff = {k: (base.get(k), p[k]) for k in p if base.get(k) != p[k]}
			print(f"  would create {name}  (seed {p['seed']})  diff-vs-control: {diff}")
		print("DRY RUN — nothing created.")
		return 0

	created = []
	for sname, name, cfg, p in plan:
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (
				"BINARY falsification of the ABSTENTION mechanism. Control = the "
				f"paired {sname} (QUAD, same seed/band/cap); memory_mode is the ONLY "
				"changed param. QUAD commits an untouched cell to WEAK_FALSE=0.25 while a "
				"learned rejection commits to FALSE=0.0, so an empty class wins argmax by "
				"abstention (Worms absorbs 508x its fair share; rho=-0.930 vs train support). "
				"BINARY floors every class at 0.0. PREDICTION: Worms over-absorption "
				"collapses toward ~1x. CONFOUND: BINARY also ignores negatives, so a null "
				"result refutes the model but a positive result is consistent-with, not "
				"uniquely diagnostic. READ OUT ON OVER-ABSORPTION, NOT macro-F1."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": HIER_EXPERIMENTS,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=60)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], name, p["seed"]))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	for fid, name, seed in created:
		if seed == SMOKE_SEED:
			requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=60)
			print(f"  -> queued SMOKE {fid} {name}")
		else:
			print(f"  . left pending {fid} {name} (release after the smoke passes)")

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = 0
	for fid, name, seed in created:
		st, = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()
		ne, = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		want = "queued" if seed == SMOKE_SEED else "pending"
		ok = (ne == 4 and st in (want, "running")
		      and q.get("memory_mode") == "BINARY"
		      and q.get("ids_tier_bits_max") == 50
		      and q.get("ids_tier_neuron_cap") == 150)
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} mode={q.get('memory_mode')}")
			bad += 1
	con.close()
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
