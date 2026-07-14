#!/usr/bin/env python3
"""BINARY 250n×100b "more power" cohort — n=10 each for all 4 datasets (Luiz
order 14/07/2026). Curiosity probe: how does the 2-state (1-bit WiSARD) BINARY
arm do when given a LARGER architecture cap — max_neurons=250, max_bits=100 —
vs the standard abl2s (500n×34b for unswt/unswr/cicids; ciciot abl2s already
IS 250n×100b).

Each flow clones the dataset's SP bin-n30 cohort config verbatim, sets
memory_mode=BINARY, and overrides max_neurons=250 / max_bits=100. Names use the
`abl2big` token so they never collide with the existing `abl2s` cohort.

Seed selection avoids duplicating the running abl2s ciciot cohort (which already
is 250n×100b BINARY): the 3 currently-500n34b datasets seed-match the bin
cohort's first 10 (offset 0); ciciot takes the NEXT 10 (offset 10) so it adds
fresh samples instead of re-running identical seeds.

Interleaved across datasets (round k = one seed of each ds), per the sweep rule.

Usage: create_binary_250n100b.py [--dry-run] [--queue]
"""
import argparse, copy, json, sqlite3, ssl, urllib.request

DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API_URL = "https://127.0.0.1:3000/api/flows"
COHORTS = {  # ds -> (bin name LIKE, tag, seed_offset)
	"unswt": ("SP-unswt-bin-16bWb-n30-%", "16bWb", 0),
	"unswr": ("SP-unswr-bin-64bWb-n30-%", "64bWb", 0),
	"cicids": ("SP-cicids-bin-96bWa-n30-%", "96bWa", 0),
	"ciciot": ("SP-ciciot-bin-96bWc-n30-%", "96bWc", 10),  # fresh seeds — abl2s already covers seeds 0-9
}
N_ABL = 10
MAX_NEURONS = 250
MAX_BITS = 100
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
	existing = con.execute("SELECT COUNT(*) FROM flows WHERE name LIKE 'SP-%abl2big%'").fetchone()[0]
	if existing:
		raise SystemExit(f"{existing} abl2big flows already exist — refusing to re-create")
	per_ds = {}
	for ds, (like, tag, off) in COHORTS.items():
		rows = con.execute(
			"SELECT name, config_json FROM flows WHERE name LIKE ? ORDER BY id LIMIT ?",
			(like, off + N_ABL)).fetchall()
		if len(rows) < off + N_ABL:
			raise SystemExit(f"{ds}: expected ≥{off + N_ABL} bin flows, got {len(rows)}")
		rows = rows[off:off + N_ABL]
		per_ds[ds] = [(name.rsplit("-r", 1)[1], json.loads(cfg), tag) for name, cfg in rows]
	# Interleaved across datasets (sweep rule): round k = one seed of each ds.
	plan = []
	for k in range(N_ABL):
		for ds in COHORTS:
			seed, cfg, tag = per_ds[ds][k]
			config = copy.deepcopy(cfg)
			config["params"]["memory_mode"] = "BINARY"
			config["params"]["max_neurons"] = MAX_NEURONS
			config["params"]["max_bits"] = MAX_BITS
			plan.append({
				"name": f"SP-{ds}-abl2big-{tag}-n{N_ABL}-r{seed}",
				"description": (f"S&P 2027 BINARY 250n×100b 'more power' probe for {ds} "
				                f"(max_neurons={MAX_NEURONS}, max_bits={MAX_BITS}; seed r{seed}). "
				                f"Curiosity run: does 1-bit WiSARD BINARY improve with a larger arch cap?"),
				"config": config,
				"experiments": EXPERIMENTS,
			})
	con.close()
	print(f"plan: {len(plan)} flows ({len(COHORTS)} datasets × {N_ABL}) @ {MAX_NEURONS}n×{MAX_BITS}b BINARY")
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
