#!/usr/bin/env python3
"""BINARY granularity ablations — n=10 each for all 4 datasets (Luiz order
13/07/2026). Completes the QUAD-vs-TERNARY-vs-BINARY paired analysis: the QUAD
arm is the production `SP-{ds}-bin-{tag}-n30` cohort, TERNARY is `abl3s`
(3-state), and this adds BINARY as `abl2s` (2-state antagonist pairs).

Each BINARY flow is seed-matched to the FIRST 10 flows of its dataset's SP bin
cohort — clone that flow's config verbatim, set memory_mode=BINARY, rename
bin→abl2s — so QUAD/TERNARY/BINARY form seed-matched triples per dataset.
Interleaved across datasets (round k = one seed of each ds), per the sweep rule.

Usage: create_binary_ablations.py [--dry-run] [--queue]
"""
import argparse, copy, json, sqlite3, ssl, urllib.request

DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API_URL = "https://127.0.0.1:3000/api/flows"
COHORTS = {  # ds -> (bin name LIKE, tag)  — all 4 datasets, matching the TERNARY arm
	"unswt": ("SP-unswt-bin-16bWb-n30-%", "16bWb"),
	"unswr": ("SP-unswr-bin-64bWb-n30-%", "64bWb"),
	"cicids": ("SP-cicids-bin-96bWa-n30-%", "96bWa"),
	"ciciot": ("SP-ciciot-bin-96bWc-n30-%", "96bWc"),
}
N_ABL = 10
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
	{"name": "GA Neurons", "phase_type": "ga_neurons", "experiment_type": "ga"},
]


def _ctx():
	ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
	return ctx


def post(body):
	data = json.dumps(body).encode()
	req = urllib.request.Request(API_URL, data=data,
		headers={"Content-Type": "application/json"}, method="POST")
	with urllib.request.urlopen(req, context=_ctx()) as r:
		return json.loads(r.read().decode())


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--queue", action="store_true")
	args = ap.parse_args()
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	# Guard: refuse to double-create.
	existing = con.execute("SELECT COUNT(*) FROM flows WHERE name LIKE 'SP-%abl2s%'").fetchone()[0]
	if existing:
		raise SystemExit(f"{existing} abl2s flows already exist — refusing to re-create")
	per_ds = {}
	for ds, (like, tag) in COHORTS.items():
		rows = con.execute(
			"SELECT name, config_json FROM flows WHERE name LIKE ? ORDER BY id LIMIT ?",
			(like, N_ABL)).fetchall()
		if len(rows) != N_ABL:
			raise SystemExit(f"{ds}: expected {N_ABL} bin flows, got {len(rows)}")
		per_ds[ds] = [(name.rsplit("-r", 1)[1], json.loads(cfg), tag) for name, cfg in rows]
	# Interleaved across datasets (sweep rule): round k = one seed of each ds.
	plan = []
	for k in range(N_ABL):
		for ds in COHORTS:
			seed, cfg, tag = per_ds[ds][k]
			config = copy.deepcopy(cfg)
			config["params"]["memory_mode"] = "BINARY"
			plan.append({
				"name": f"SP-{ds}-abl2s-{tag}-n{N_ABL}-r{seed}",
				"description": (f"S&P 2027 BINARY granularity ablation for {ds} (seed-matched to "
				                f"the bin cohort's r{seed}; completes the QUAD-vs-TERNARY-vs-BINARY "
				                f"paired analysis across all 4 datasets)."),
				"config": config,
				"experiments": EXPERIMENTS,
			})
	con.close()
	print(f"plan: {len(plan)} flows ({len(COHORTS)} datasets × {N_ABL})")
	for f in plan[:8]:
		print("  ", f["name"])
	print("  ...")
	if args.dry_run:
		return
	created = []
	for f in plan:
		resp = post(f)
		fid = resp.get("id") or resp.get("flow", {}).get("id")
		if fid is None:
			raise SystemExit(f"no id for {f['name']}: {resp}")
		created.append(fid)
		if args.queue:
			data = json.dumps({"status": "queued"}).encode()
			req = urllib.request.Request(f"{API_URL}/{fid}", data=data,
				headers={"Content-Type": "application/json"}, method="PATCH")
			urllib.request.urlopen(req, context=_ctx()).read()
	print(f"created {len(created)} flows: {created[0]}..{created[-1]}")
	# Verify every flow got its experiments (Rule 2: flows without experiments do nothing).
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = [fid for fid in created if con.execute(
		"SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0] != len(EXPERIMENTS)]
	con.close()
	if bad:
		raise SystemExit(f"FLOWS WITH WRONG EXPERIMENT COUNT: {bad}")
	print("all flows have the expected experiments ✓")


main()
