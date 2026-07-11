#!/usr/bin/env python3
"""Create the SP Wave-1 cohorts (S&P 2027, Protocol v2, 3-way splits).

Interleaved n=30 binary cohorts for 4 dataset-splits + the n=10 TERNARY
granularity-ablation arm (paired seeds with the unswt QUAD cohort). The
46M n=5 wave is created separately once streaming val-plumbing lands.

Configs are cloned from the previously blessed (now cancelled) cohort
flows — no new sweeps — with only split (-> _3way), seed, and naming
patched. Naming: SP-{ds}-{task}-{width}b{W}-n{wave}-r{seed}.

Fresh-seed guarantee: seeds are drawn to be disjoint from EVERY seed ever
recorded in flows.config_json (closes the C1.3 probe-in-cohort issue).

Usage:
	python scripts/create_sp_wave1.py --dry-run     # show plan only
	python scripts/create_sp_wave1.py               # create (pending)
	python scripts/create_sp_wave1.py --queue       # create + queue
"""

import argparse
import copy
import json
import random
import sqlite3
import ssl
import sys
import urllib.error
import urllib.request

DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API_URL = "https://127.0.0.1:3000/api/flows"
RNG_SEED = 20260711
N_WAVE = 30
N_ABLATION = 10

# ds_key -> (template_flow_id, width+weight tag) — the previously blessed configs
TEMPLATES = {
	"unswt": (4341, "16bWb"),
	"unswr": (4330, "64bWb"),
	"cicids": (4342, "96bWa"),
	"ciciot": (4343, "96bWc"),
}

EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
	{"name": "GA Neurons", "phase_type": "ga_neurons", "experiment_type": "ga"},
]


def fetch_template(con: sqlite3.Connection, flow_id: int) -> dict:
	row = con.execute("SELECT config_json FROM flows WHERE id = ?", (flow_id,)).fetchone()
	if row is None:
		raise SystemExit(f"template flow {flow_id} not found")
	return json.loads(row[0])


def used_seeds(con: sqlite3.Connection) -> set:
	rows = con.execute(
		"SELECT DISTINCT json_extract(config_json, '$.params.seed') FROM flows"
	).fetchall()
	return {int(r[0]) for r in rows if r[0] is not None}


def fresh_seeds(count: int, taken: set, rng: random.Random) -> list:
	out = []
	while len(out) < count:
		s = rng.randint(10000, 99999)
		if s not in taken:
			taken.add(s)
			out.append(s)
	return out


def build_flow(template: dict, ds: str, tag: str, seed: int, task: str, wave: int,
			   memory_mode: str | None) -> dict:
	config = copy.deepcopy(template)
	params = config["params"]
	split = params.get("ids_split", "random")
	if not split.endswith("_3way"):
		params["ids_split"] = split + "_3way"
	params["seed"] = seed
	if memory_mode is not None:
		params["memory_mode"] = memory_mode
	name = f"SP-{ds}-{task}-{tag}-n{wave}-r{seed}"
	mode_note = f", memory_mode={memory_mode}" if memory_mode else ""
	return {
		"name": name,
		"description": (
			f"S&P 2027 Wave-1 {task} cohort (Protocol v2, 3-way split: thresholds "
			f"calibrated on val, test report-only). Config cloned from the blessed "
			f"pre-reset cohort; fresh seed {seed}{mode_note}."
		),
		"config": config,
		"experiments": EXPERIMENTS,
	}


def post_flow(body: dict) -> dict:
	data = json.dumps(body).encode("utf-8")
	req = urllib.request.Request(
		API_URL, data=data,
		headers={"Content-Type": "application/json"}, method="POST",
	)
	ctx = ssl.create_default_context()
	ctx.check_hostname = False
	ctx.verify_mode = ssl.CERT_NONE
	with urllib.request.urlopen(req, context=ctx) as resp:
		return json.loads(resp.read().decode("utf-8"))


def patch_queued(flow_id: int) -> None:
	data = json.dumps({"status": "queued"}).encode("utf-8")
	req = urllib.request.Request(
		f"{API_URL}/{flow_id}", data=data,
		headers={"Content-Type": "application/json"}, method="PATCH",
	)
	ctx = ssl.create_default_context()
	ctx.check_hostname = False
	ctx.verify_mode = ssl.CERT_NONE
	urllib.request.urlopen(req, context=ctx).read()


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--queue", action="store_true", help="queue flows after creation")
	args = ap.parse_args()

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	templates = {ds: fetch_template(con, fid) for ds, (fid, _) in TEMPLATES.items()}
	taken = used_seeds(con)
	con.close()

	rng = random.Random(RNG_SEED)
	seeds = {ds: fresh_seeds(N_WAVE, taken, rng) for ds in TEMPLATES}

	# Interleaved plan: round k = one seed of each dataset (+ ablation in rounds 1..10)
	plan = []
	for k in range(N_WAVE):
		for ds, (_, tag) in TEMPLATES.items():
			plan.append(build_flow(templates[ds], ds, tag, seeds[ds][k], "bin", N_WAVE, None))
		if k < N_ABLATION:
			# TERNARY ablation: PAIRED with the unswt QUAD flow of this round (same seed)
			plan.append(build_flow(
				templates["unswt"], "unswt", TEMPLATES["unswt"][1], seeds["unswt"][k],
				"abl3s", N_ABLATION, "TERNARY",
			))

	print(f"Wave-1 plan: {len(plan)} flows "
		  f"({N_WAVE}x{len(TEMPLATES)} bin interleaved + {N_ABLATION} TERNARY ablation, paired seeds)")
	for f in plan[:6]:
		print(f"  {f['name']}  split={f['config']['params']['ids_split']}")
	print("  ...")

	if args.dry_run:
		return

	created = []
	for f in plan:
		resp = post_flow(f)
		fid = resp.get("id") or resp.get("flow", {}).get("id")
		if fid is None:
			raise SystemExit(f"no id in response for {f['name']}: {resp}")
		created.append((fid, f["name"]))
	print(f"Created {len(created)} flows: ids {created[0][0]}..{created[-1][0]}")

	# Verify experiments landed (Rule 2: a flow without experiments does nothing)
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = [fid for fid, _ in created if con.execute(
		"SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0] != len(EXPERIMENTS)]
	con.close()
	if bad:
		raise SystemExit(f"FLOWS WITH WRONG EXPERIMENT COUNT: {bad}")
	print("All flows have the expected experiments.")

	if args.queue:
		for fid, _ in created:
			patch_queued(fid)
		print(f"Queued all {len(created)} flows (FIFO order = interleaved).")


if __name__ == "__main__":
	main()
