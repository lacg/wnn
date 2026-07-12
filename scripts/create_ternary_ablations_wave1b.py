#!/usr/bin/env python3
"""Wave-1b: TERNARY granularity ablations for the remaining datasets
(Luiz order 12/07/2026) — n=10 each for unswr/cicids/ciciot, SEED-MATCHED to
the first 10 flows of each SP bin cohort (clone that flow's config verbatim,
add memory_mode=TERNARY, rename bin→abl3s). Mirrors the unswt abl3s arm so
the QUAD-vs-TERNARY paired analysis extends to all 4 datasets.

Usage: create_ternary_ablations_wave1b.py [--dry-run] [--queue]
"""
import argparse, copy, json, sqlite3, ssl, urllib.request

DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API_URL = "https://127.0.0.1:3000/api/flows"
COHORTS = {  # ds -> (bin name LIKE, tag)
	"unswr": ("SP-unswr-bin-64bWb-n30-%", "64bWb"),
	"cicids": ("SP-cicids-bin-96bWa-n30-%", "96bWa"),
	"ciciot": ("SP-ciciot-bin-96bWc-n30-%", "96bWc"),
}
N_ABL = 10
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
	{"name": "GA Neurons", "phase_type": "ga_neurons", "experiment_type": "ga"},
]

def post(body):
	data = json.dumps(body).encode()
	req = urllib.request.Request(API_URL, data=data,
		headers={"Content-Type": "application/json"}, method="POST")
	ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
	with urllib.request.urlopen(req, context=ctx) as r:
		return json.loads(r.read().decode())

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--queue", action="store_true")
	args = ap.parse_args()
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	plan = []
	# Interleaved across datasets (sweep rule): round k = one seed of each ds.
	per_ds = {}
	for ds, (like, tag) in COHORTS.items():
		rows = con.execute(
			"SELECT name, config_json FROM flows WHERE name LIKE ? ORDER BY id LIMIT ?",
			(like, N_ABL)).fetchall()
		if len(rows) != N_ABL:
			raise SystemExit(f"{ds}: expected {N_ABL} bin flows, got {len(rows)}")
		per_ds[ds] = [(name.rsplit("-r", 1)[1], json.loads(cfg), tag) for name, cfg in rows]
	for k in range(N_ABL):
		for ds in COHORTS:
			seed, cfg, tag = per_ds[ds][k]
			config = copy.deepcopy(cfg)
			config["params"]["memory_mode"] = "TERNARY"
			plan.append({
				"name": f"SP-{ds}-abl3s-{tag}-n{N_ABL}-r{seed}",
				"description": (f"S&P 2027 Wave-1b TERNARY granularity ablation for {ds} "
				                f"(seed-matched to the bin cohort's r{seed}; extends the "
				                f"unswt QUAD-vs-TERNARY paired analysis to all datasets)."),
				"config": config,
				"experiments": EXPERIMENTS,
			})
	con.close()
	print(f"plan: {len(plan)} flows"); [print(" ", f["name"]) for f in plan[:6]]; print("  ...")
	if args.dry_run: return
	created = []
	for f in plan:
		resp = post(f)
		fid = resp.get("id") or resp.get("flow", {}).get("id")
		if fid is None: raise SystemExit(f"no id for {f['name']}: {resp}")
		created.append(fid)
		if args.queue:
			data = json.dumps({"status": "queued"}).encode()
			req = urllib.request.Request(f"{API_URL}/{fid}", data=data,
				headers={"Content-Type": "application/json"}, method="PATCH")
			ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
			urllib.request.urlopen(req, context=ctx).read()
	print(f"created {len(created)} flows: {created[0]}..{created[-1]}")
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = [fid for fid in created if con.execute(
		"SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0] != len(EXPERIMENTS)]
	con.close()
	if bad: raise SystemExit(f"FLOWS WITH WRONG EXPERIMENT COUNT: {bad}")
	print("all flows have the expected experiments ✓")

main()
