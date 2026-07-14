#!/usr/bin/env python3
"""S&P 2027 mode-granularity ablation RERUN set (QSR/PLN/TERNARY/BINARY).

WHY: the IDS memory_mode bug (fixed 3532423e) made the eval hardcode QUAD, so every
prior abl3s/abl2s run silently trained+scored as QUAD_WEIGHTED — the results are
invalid. This rebuilds the ablation as a clean, seed-matched 4-mode comparison.

APPROACH (b, archive-and-recreate; Luiz 14/07/2026):
  - ARCHIVE the QUAD-in-disguise runs by RENAMING them (prefix so reports + the
    create-guard skip them; completed flows are already ignored by the worker, so no
    status transition is needed). Matches the oi_cohort_v2 `-OLD-` precedent.
  - KEEP the already-queued BINARY flows intact (never ran → not contaminated); the
    fixed worker will run them correctly.
  - RECREATE fresh, seed-matched flows to fill every mode to the canonical
    4 datasets x 10 seeds = 40. Each flow is cloned from that dataset's QUAD
    `SP-{ds}-bin-{tag}-n30` cohort (first 10 seeds), so QUAD/TERNARY/BINARY/QSR/PLN
    form seed-matched quintuples per dataset. Interleaved across datasets (sweep rule).

Modes -> name token: TERNARY=abl3s, BINARY=abl2s, QSR=ablqsr, PLN=ablpln (mode-explicit
so PLN(3-state) never collides with TERNARY's abl3s).

Usage: rerun_mode_ablations.py [--dry-run] [--execute] [--queue]
  --dry-run   print the archive + create plan, touch nothing (default)
  --execute   perform the renames + POST the new flows
  --queue     (with --execute) PATCH each created flow to status=queued
"""
import argparse, copy, json, re, sqlite3, ssl, urllib.request

DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
API_URL = "https://127.0.0.1:3000/api/flows"

# ds -> (QUAD clone-source cohort LIKE, config tag). Same 4 cells as the existing abl.
COHORTS = {
	"unswt":  ("SP-unswt-bin-16bWb-n30-%",  "16bWb"),
	"unswr":  ("SP-unswr-bin-64bWb-n30-%",  "64bWb"),
	"cicids": ("SP-cicids-bin-96bWa-n30-%", "96bWa"),
	"ciciot": ("SP-ciciot-bin-96bWc-n30-%", "96bWc"),
}
N_SEEDS = 10
# mode -> (memory_mode value, name token). Order = creation/report order.
MODES = [("TERNARY", "abl3s"), ("BINARY", "abl2s"), ("QSR", "ablqsr"), ("PLN", "ablpln")]
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
	{"name": "GA Neurons", "phase_type": "ga_neurons", "experiment_type": "ga"},
]
ARCHIVE_PREFIX = "zINVALIDQUAD-"   # sorts last; breaks the `SP-%abl_s%` cohort/report match
NAME_RE = re.compile(r"SP-(\w+)-abl\w+-(\w+)-n\d+-r(\d+)")


def _ctx():
	ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
	return ctx


def _req(url, body, method):
	data = json.dumps(body).encode()
	req = urllib.request.Request(url, data=data,
		headers={"Content-Type": "application/json"}, method=method)
	with urllib.request.urlopen(req, context=_ctx()) as r:
		return json.loads(r.read().decode() or "{}")


def load_clone_sources(con):
	"""ds -> ordered [(seed, config_dict)] from the QUAD bin cohort's first N_SEEDS."""
	src = {}
	for ds, (like, tag) in COHORTS.items():
		rows = con.execute(
			"SELECT name, config_json FROM flows WHERE name LIKE ? ORDER BY id LIMIT ?",
			(like, N_SEEDS)).fetchall()
		if len(rows) != N_SEEDS:
			raise SystemExit(f"{ds}: expected {N_SEEDS} QUAD clone-source flows, got {len(rows)}")
		src[ds] = [(name.rsplit("-r", 1)[1], json.loads(cfg)) for name, cfg in rows]
	return src


def load_archive_and_keep(con):
	"""Return (archive, keep_binary_combos).
	archive: [(id, old_name, ds, seed)] — the 30 TERNARY (all) + BINARY completed/cancelled.
	keep_binary_combos: set of (ds, seed) already-queued BINARY (left intact)."""
	archive, keep = [], set()
	# TERNARY abl3s 4540-4569: every flow is invalid QUAD -> archive all.
	for fid, name, status in con.execute(
			"SELECT id, name, status FROM flows WHERE id BETWEEN 4540 AND 4569"):
		m = NAME_RE.match(name)
		archive.append((fid, name, m.group(1) if m else "?", m.group(3) if m else "?"))
	# BINARY abl2s 4570-4609: archive completed+cancelled, keep queued.
	for fid, name, status in con.execute(
			"SELECT id, name, status FROM flows WHERE id BETWEEN 4570 AND 4609"):
		m = NAME_RE.match(name)
		ds, seed = (m.group(1), m.group(3)) if m else ("?", "?")
		if status in ("completed", "cancelled"):
			archive.append((fid, name, ds, seed))
		elif status == "queued":
			keep.add((ds, seed))
	return archive, keep


def build_create_plan(src, archive, keep_binary):
	"""Interleaved create plan. TERNARY/QSR/PLN: all 40. BINARY: only archived combos
	(the queued 24 are kept), reconstructed to exactly complement the kept set."""
	archived_binary = {(ds, seed) for (_, name, ds, seed) in archive if "-abl2s-" in name}
	plan = []
	for k in range(N_SEEDS):                       # round k = one seed of each ds
		for ds, (like, tag) in COHORTS.items():
			seed, cfg = src[ds][k]
			for mode, token in MODES:
				if mode == "BINARY" and (ds, seed) not in archived_binary:
					continue                       # kept queued flow already covers it
				config = copy.deepcopy(cfg)
				config["params"]["memory_mode"] = mode
				plan.append({
					"mode": mode, "ds": ds, "seed": seed,
					"name": f"SP-{ds}-{token}-{tag}-n{N_SEEDS}-r{seed}",
					"description": (f"S&P 2027 {mode} granularity ablation for {ds} "
						f"(seed-matched to the QUAD bin cohort's r{seed}; rerun after the "
						f"IDS memory_mode fix 3532423e). QUAD/TERNARY/BINARY/QSR/PLN quintuple."),
					"config": config,
					"experiments": EXPERIMENTS,
				})
	return plan, archived_binary, keep_binary


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--execute", action="store_true")
	ap.add_argument("--queue", action="store_true")
	args = ap.parse_args()
	if not args.execute:
		args.dry_run = True

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	# Guard: refuse to double-create the new stochastic arms.
	for token in ("ablqsr", "ablpln"):
		n = con.execute("SELECT COUNT(*) FROM flows WHERE name LIKE ?", (f"SP-%{token}%",)).fetchone()[0]
		if n:
			raise SystemExit(f"{n} {token} flows already exist — refusing to re-create")
	src = load_clone_sources(con)
	archive, keep_binary = load_archive_and_keep(con)
	plan, archived_binary, keep = build_create_plan(src, archive, keep_binary)
	con.close()

	# ---- report the plan ----
	from collections import Counter
	cc = Counter(p["mode"] for p in plan)
	print("=" * 70)
	print(f"ARCHIVE (rename -> '{ARCHIVE_PREFIX}<old>'): {len(archive)} flows")
	ac = Counter("TERNARY" if "-abl3s-" in n else "BINARY" for _, n, _, _ in archive)
	print(f"    TERNARY (all invalid): {ac['TERNARY']}   BINARY (completed+cancelled): {ac['BINARY']}")
	for fid, name, ds, seed in archive[:4]:
		print(f"    #{fid}  {name}  ->  {ARCHIVE_PREFIX}{name}")
	print(f"    ... ({len(archive)} total)")
	print("-" * 70)
	print(f"KEEP untouched (queued BINARY, run as-is): {len(keep)} combos")
	print("-" * 70)
	print(f"CREATE fresh: {len(plan)} flows")
	for mode, token in MODES:
		print(f"    {mode:8} ({token:7}): {cc[mode]:3}")
	print("    sample names:")
	seen = set()
	for p in plan:
		if p["mode"] not in seen:
			print(f"      {p['name']}   mm={p['config']['params']['memory_mode']}")
			seen.add(p["mode"])
	print("=" * 70)
	print(f"TOTALS: archive={len(archive)}  keep={len(keep)}  create={len(plan)}")
	print(f"  per-mode valid after: "
		f"TERNARY={cc['TERNARY']}  BINARY={cc['BINARY']}+{len(keep)}(kept)={cc['BINARY']+len(keep)}  "
		f"QSR={cc['QSR']}  PLN={cc['PLN']}")

	if args.dry_run:
		print("\n[DRY-RUN] nothing changed. Re-run with --execute [--queue] to apply.")
		return

	# ---- execute: rename archives, then POST new flows ----
	print("\n[EXECUTE] renaming archives...")
	for fid, name, _, _ in archive:
		_req(f"{API_URL}/{fid}", {"name": f"{ARCHIVE_PREFIX}{name}"}, "PATCH")
	print(f"  renamed {len(archive)} flows")
	print("[EXECUTE] creating flows...")
	created = []
	for p in plan:
		body = {"name": p["name"], "description": p["description"],
			"config": p["config"], "experiments": p["experiments"]}
		resp = _req(API_URL, body, "POST")
		fid = resp.get("id") or resp.get("flow", {}).get("id")
		if fid is None:
			raise SystemExit(f"no id for {p['name']}: {resp}")
		created.append(fid)
		if args.queue:
			_req(f"{API_URL}/{fid}", {"status": "queued"}, "PATCH")
	print(f"  created {len(created)} flows: {created[0]}..{created[-1]}")
	# Verify experiments landed (Rule 2).
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = [f for f in created if con.execute(
		"SELECT COUNT(*) FROM experiments WHERE flow_id=?", (f,)).fetchone()[0] != len(EXPERIMENTS)]
	con.close()
	if bad:
		raise SystemExit(f"FLOWS WITH WRONG EXPERIMENT COUNT: {bad}")
	print("all created flows have the expected experiments ✓")


main()
