"""Queue the MCST support-tiered hier cohort (B side only — the A side is the
banked MCSD-auto cohort, ids_results.md §9; nothing re-runs).

Clones each MCSD-auto hier flow's params, adds ids_tier_sizing=true (cap 250),
keeps the four S0:/S1: stage-tagged phases. SMOKE PROTOCOL: queue s20401
ONLY; the caller pauses nothing because the other four are created PAUSED via
immediate pause after creation — release them once the smoke run shows the
[tier] centre lines, the tiered grid header, the [tie-break] line (if a tie
occurs), classnorm modes in the decode sweep, STAGE BOUNDARY, COMBINED, and
BOTH 'Cascade persisted' + 'Frozen S0 gate persisted'.
"""
import json
import sqlite3
import sys
import time

import requests
import urllib3

urllib3.disable_warnings()

DASHBOARD = "https://localhost:3000"
SUFFIX = "-tiered3"  # widened bits band 34-50 + neuron cap 150 (tiered2 pinned 40/45 at its ceiling)
DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
HIER_EXPERIMENTS = [
	{"name": "S0: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S0: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
	{"name": "S1: Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "S1: GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]


def main() -> int:
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	sources = con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE 'MCSD-unswt-quad-16b-hier-%-auto' ORDER BY name"
	).fetchall()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'MCST-%'")}
	con.close()
	if len(sources) != 5:
		print(f"REFUSED: expected the 5 MCSD-auto hier flows, found {len(sources)}")
		return 4
	if "--dry-run" in sys.argv:
		for n, cj in sources:
			p = json.loads(cj)["params"]
			print(f"  would create {n.replace('MCSD', 'MCST').replace('-auto', '-tiered')}"
			      f"  (seed {p['seed']}, tier cap 250, 4 phases)")
		print("DRY RUN — nothing created.")
		return 0

	created = []
	for sname, cj in sources:
		cfg = json.loads(cj)
		p = dict(cfg["params"])
		p.pop("fitness_ce_anchor_normalized", None)
		p["ids_tier_sizing"] = True
		p["ids_tier_neuron_cap"] = 150
		p["ids_tier_bits_min"] = 34
		p["ids_tier_bits_max"] = 50
		p["max_bits"] = 50          # the GA's global ceiling must clear the band
		p["min_bits"] = 34
		name = sname.replace("MCSD", "MCST").replace("-auto", SUFFIX)
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (
				f"MCST support-tiered re-run of {sname}: per-class neuron/bits centres "
				f"from train supports (cap 250, S1 = 250 - S0 winner), classnorm decode, "
				f"epsilon tiebreak. B side only — comparator is the banked MCSD-auto "
				f"cohort (ids_results.md §9). Pre-registered: macro-F1 + benign-FPR "
				f"primaries; neuron tiering claims EFFICIENCY ONLY."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": HIER_EXPERIMENTS,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=60)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], name))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	# SMOKE ONE: queue only s20401; the rest stay pending until released.
	for fid, name in created:
		if name.endswith('s20401' + SUFFIX):
			requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=60)
			print(f"  → queued SMOKE {fid} {name}")
		else:
			print(f"  · left pending {fid} {name} (release after the smoke run passes)")

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = 0
	for fid, name in created:
		st, = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()
		ne, = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		want = "queued" if name.endswith('s20401' + SUFFIX) else "pending"
		ok = (ne == 4 and st in (want, "running") and q.get("ids_tier_sizing") is True
		      and q.get("ids_tier_neuron_cap") == 150
		      and q.get("ids_tier_bits_max") == 50)
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} tier={q.get('ids_tier_sizing')}")
			bad += 1
	con.close()
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
