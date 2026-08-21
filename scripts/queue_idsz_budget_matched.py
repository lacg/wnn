"""Budget-matched IDS fitness-weight sweep — 23 arms x 5 seeds = 115 flows.

WHY (20/08/2026). The IDSZ sweep (flows 5380-5425) ranks arms that did NOT
search equally. The GA is patience-terminated and patience tracks EACH ARM'S
OWN fitness definition, so search budget is an OUTCOME of the treatment:

    r(GA generations, held-out F1) = 0.717
    r(w_ce, GA generations)        = 0.548
    partial r(w_ce, F1 | gens)     = -0.009   <- weight effect is ENTIRELY mediated

Observed generation counts ranged 60-100 across arms. Seed-pairing does NOT
fix this (budget asymmetry is identical at both seeds), so the fix is to hold
the budget constant and let ONLY the weight vector vary.

BUDGET-MATCHING = ga_generations 100 (fixed) + patience 999 (never fires).
  - 100 >= every arm's natural stopping point under the old patience rule, so
    no arm is truncated below what it previously got.
  - patience=999 also disables the magnitude-aware early stop: EarlyStopConfig
    has mag_rho_cap=0.0 => "use `patience` as the cap".
  - STATED ASSUMPTION: an arm that would still improve past gen 100 is capped.
    Budget-matching requires a fixed number; 100 is the defensible choice.

sigma_seed under zscore measured at 0.264pp (5 paired arms) => n=5 gives
MDD ~0.38pp raw, below the ~0.6pp residual arm effect. n=5 is adequate.

Seeds 20301/20302 are REUSED (plus 3 fresh) on purpose: the shared seeds give
a direct budget-matched-vs-unmatched contrast, isolating what the extra
generations were worth.

Queue order: `scheduler.admit` tie-breaks on the OLDEST queued id, so these
run AFTER the in-flight IDSZ round 2. Creation is seed-major, so round 1 of
this cohort is one of each arm (the standing interleave rule).

Per CLAUDE.md Rule 2: dashboard POST /api/flows, never a direct SQL insert.
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

GA_GENERATIONS = 100
PATIENCE = 999
SEEDS = [20301, 20302, 20303, 20304, 20305]
TEMPLATE_FLOW = 5380  # IDSZ-unswt-quad-16b-Wb-CTRL-r20301

# NOTE (20/08/2026): the GA generation cap comes from experiments.max_iterations,
# NOT from params["ga_generations"] at RUN time — worker.py:1694 is
#   exp_data.get("max_iterations") or params.get("ga_generations", 250)
# and max_iterations is always populated, so the params value is dead on an
# existing flow. At CREATE time the dashboard resolves max_iterations from
# exp_spec.params["generations"] first, then flow params["ga_generations"]
# (flows.rs:409). We set BOTH so the cap is explicit and cannot silently revert.
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons",
	 "params": {"generations": GA_GENERATIONS}},
]


def load_arms(con: sqlite3.Connection) -> list[tuple[str, float, float, float, float]]:
	"""The 23 arm definitions, read from the live IDSZ cohort (authoritative)."""
	arms: dict[str, tuple[float, float, float, float]] = {}
	rows = con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE 'IDSZ-unswt-quad-16b-%'"
	).fetchall()
	for name, cfg in rows:
		arm = name.replace("IDSZ-unswt-quad-16b-", "").rsplit("-r", 1)[0]
		p = json.loads(cfg).get("params", {})
		arms[arm] = (
			p["fitness_weight_ce"], p["fitness_weight_acc"],
			p["fitness_weight_f1"], p["fitness_weight_fpr"],
		)
	return sorted((a, *w) for a, w in arms.items())


def build_params(base: dict, arm_weights: tuple[float, float, float, float], seed: int) -> dict:
	ce, acc, f1, fpr = arm_weights
	params = dict(base)
	params["seed"] = seed
	params["ga_generations"] = GA_GENERATIONS
	params["patience"] = PATIENCE
	params["fitness_weight_ce"] = ce
	params["fitness_weight_acc"] = acc
	params["fitness_weight_f1"] = f1
	params["fitness_weight_fpr"] = fpr
	return params


def main() -> int:
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	row = con.execute("SELECT config_json FROM flows WHERE id = ?", (TEMPLATE_FLOW,)).fetchone()
	if not row:
		print(f"ERROR: template flow {TEMPLATE_FLOW} not found.")
		return 1
	base_params = dict(json.loads(row[0])["params"])
	arms = load_arms(con)
	con.close()

	if len(arms) != 23:
		print(f"ERROR: expected 23 arms, found {len(arms)}: {[a[0] for a in arms]}")
		return 1
	for arm, ce, acc, f1, fpr in arms:
		if abs((ce + acc + f1 + fpr) - 1.0) > 1e-9:
			print(f"ERROR: {arm} weights sum to {ce+acc+f1+fpr}, not 1.0")
			return 1

	print(f"arms={len(arms)}  seeds={SEEDS}  total={len(arms)*len(SEEDS)} flows")
	print(f"budget-match: ga_generations={GA_GENERATIONS}, patience={PATIENCE}")
	smoke = "--smoke" in sys.argv
	if smoke:
		print("SMOKE MODE: creating ONE flow only, to verify the budget-match lands.")
	if "--dry-run" in sys.argv:
		for arm, ce, acc, f1, fpr in arms:
			print(f"  {arm:<14} ce{ce}/acc{acc}/f1{f1}/fpr{fpr}")
		print("\nDRY RUN — nothing created.")
		return 0

	# Idempotent: an already-created name is skipped, so a re-run after a partial
	# failure (or after --smoke) tops up the cohort instead of duplicating it.
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	existing = {r[0] for r in con.execute(
		"SELECT name FROM flows WHERE name LIKE 'IDSZB-unswt-quad-16b-%'")}
	con.close()
	if existing:
		print(f"skipping {len(existing)} already-created flow(s)")

	created = []
	plan = [(seed, a) for seed in SEEDS for a in arms
	        if f"IDSZB-unswt-quad-16b-{a[0]}-r{seed}" not in existing]
	if smoke:
		plan = plan[:1]
	for seed, (arm, ce, acc, f1, fpr) in plan:
		if True:
			name = f"IDSZB-unswt-quad-16b-{arm}-r{seed}"
			body = {
				"name": name,
				"description": (
					f"BUDGET-MATCHED IDS fitness-weight sweep under the ZSCORE combine. "
					f"{arm} = ce {ce}/acc {acc}/f1 {f1}/fpr {fpr}, seed={seed}. "
					f"ga_generations={GA_GENERATIONS} FIXED, patience={PATIENCE} (early stop "
					f"disabled) so every arm searches equally — removes the search-budget "
					f"confound that made the IDSZ ranking uninterpretable "
					f"(partial r(w_ce,F1|gens) = -0.009). Recipe otherwise identical to "
					f"IDSZ (template flow {TEMPLATE_FLOW}). Seeds 20301-20305, n=5."
				),
				"config": {
					"template": "ids-binary-2-phase",
					"params": build_params(base_params, (ce, acc, f1, fpr), seed),
				},
				"experiments": EXPERIMENTS,
			}
			resp = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
			if resp.status_code not in (200, 201):
				print(f"  x FAILED {name} ({resp.status_code}): {resp.text[:200]}")
				return 2
			fid = resp.json()["id"]
			created.append((fid, name))
			print(f"  + id={fid}  {name}")
			time.sleep(0.3)

	print(f"\nFlipping {len(created)} pending -> queued...")
	for fid, name in created:
		r = requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		if r.status_code not in (200, 201, 204):
			print(f"  ! restart failed for {fid} ({r.status_code})")
		time.sleep(0.2)

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		mi = con.execute(
			"SELECT max_iterations FROM experiments WHERE flow_id=? AND name LIKE 'GA%'", (fid,)
		).fetchone()
		cfg = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])
		pat = cfg["params"].get("patience")
		ok = (ne == 2 and st in ("queued", "running")
		      and mi and mi[0] == GA_GENERATIONS and pat == PATIENCE)
		if not ok:
			print(f"  x id={fid} status={st} exps={ne} max_iterations={mi and mi[0]} "
			      f"patience={pat}  {name}")
			bad += 1
	con.close()
	if smoke and bad == 0:
		fid = created[0][0]
		print(f"\nSMOKE OK: id={fid} has max_iterations={GA_GENERATIONS}, patience={PATIENCE}.")
		print("Re-run WITHOUT --smoke to create the remaining flows "
		      "(existing names are re-created, so delete this one first if you re-run in full).")
	print(f"\nVERIFY: {len(created)-bad}/{len(created)} flows queued with 2 experiments each.")
	return 0 if bad == 0 else 3


if __name__ == "__main__":
	sys.exit(main())
