"""Clone a banked cohort into a DESIRABILITY arm with a per-task CE anchor.

Generalised from scripts/queue_mcs_desirability.py (the MCSD one-shot, 26/08).
Given a source flow-name prefix, clones every matching flow's config VERBATIM
and changes exactly TWO keys:

    fitness_aggregation          -> desirability
    fitness_ce_anchor_normalized -> 0.1937

WHY THE ANCHOR TRAVELS WITH THE AGGREGATION. The desirability ce half-anchor
0.133 is a frozen unsw-nb15 TEMPORAL BINARY fit — the median best_ce over
18,355 IDSZ iterations. CE is the only column in the desirability vector with no
absolute meaning: it carries units and scales with the class count and the class
imbalance, so that absolute is wrong for a different class count (multiclass CE
runs ~14x it), wrong for a different split, and wrong for a different dataset.
0.1937 is the same anchor expressed in units of each task's OWN base-rate
log-loss H(p); ram_accelerator (ABI 9) derives the absolute per task from the
train labels already resident in the Rust cache. Cloning a CROSS-DATASET cohort
without it would score cicids and ciciot on unsw's CE ruler — which is exactly
the caveat the IDSD readout raised against licensing a cross-dataset cohort.

Usage:
    python scripts/queue_desirability_clone.py --source IDSX- --target IDSXD- \
        [--status paused] [--only <substr>] [--dry-run] [--expect N]

--only clones a SINGLE flow first. A cohort is never committed to an unproven
path: smoke one, verify the "[desirability] ce half-anchor" line in the worker
log, then re-run without --only to fill in the rest (existing names are skipped).

Source flows are LEFT AS THEY ARE — never deleted, never unpaused.
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
CE_ANCHOR_NORMALIZED = 0.1937
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]
POST_TIMEOUT = 60      # the dashboard is slow while a flow is training
FLIP_RETRIES = 3


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def arg(flag: str, default=None):
	for i, a in enumerate(sys.argv):
		if a == flag and i + 1 < len(sys.argv):
			return sys.argv[i + 1]
	return default


def worker_wheel_ready() -> bool:
	"""ABI 9 = the wheel that can derive a per-task CE anchor."""
	try:
		import ram_accelerator as a
		return (getattr(a, "ABI_VERSION", 0) >= 9
		        and hasattr(a.IDSCacheWrapper, "desirability_ce_anchor"))
	except ImportError:
		return False


def post_with_retry(url: str, body: dict):
	"""The dashboard read-times-out while the worker trains; retry rather than
	abandoning a half-flipped cohort (the MCSD flip died this way)."""
	last = None
	for _ in range(FLIP_RETRIES):
		try:
			return requests.post(url, json=body, verify=False, timeout=POST_TIMEOUT)
		except requests.exceptions.RequestException as e:
			last = e
			time.sleep(5)
	raise last


def main() -> int:
	source = arg("--source")
	target = arg("--target")
	if not source or not target:
		print(__doc__)
		return 64
	status = arg("--status", "paused")
	expect = arg("--expect")
	only = arg("--only")

	if not worker_wheel_ready() and "--force" not in sys.argv:
		print("REFUSED: installed ram_accelerator lacks ABI 9 / desirability_ce_anchor.\n"
		      "Run scripts/deploy_ce_anchor_worker.sh at worker idle first.")
		return 3

	con = ro()
	sources = con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE ? AND status=? ORDER BY name",
		(f"{source}%", status)).fetchall()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE ?", (f"{target}%",))}
	con.close()

	if not sources:
		print(f"REFUSED: no flows match {source!r} with status {status!r}")
		return 4
	if expect and len(sources) != int(expect):
		print(f"REFUSED: expected {expect} source flows, found {len(sources)}")
		return 4
	if only:
		sources = [r for r in sources if only in r[0]]
		if not sources:
			print(f"REFUSED: --only {only!r} matched nothing")
			return 5
		print(f"--only {only!r}: {len(sources)} flow(s) selected")

	print(f"{target} desirability cohort: {len(sources)} flow(s) from {source} [{status}]")
	if "--dry-run" in sys.argv:
		for sname, cj in sources:
			p = json.loads(cj)["params"]
			print(f"  would clone {sname}: {p.get('ids_dataset')} {p.get('ids_split')} "
			      f"{p.get('ids_n_bits')}b  agg {p.get('fitness_aggregation')} -> desirability")
		print("DRY RUN — nothing created.")
		return 0

	created = []
	for sname, cj in sources:
		cfg = json.loads(cj)
		p = dict(cfg["params"])
		if p.get("fitness_aggregation") == "desirability":
			print(f"  ! {sname} is ALREADY desirability — skipped (not a valid clone source)")
			continue
		p["fitness_aggregation"] = "desirability"
		p["fitness_ce_anchor_normalized"] = CE_ANCHOR_NORMALIZED
		name = sname.replace(source, target, 1)
		if name in existing:
			print(f"  = exists {name}")
			continue
		body = {
			"name": name,
			"description": (
				f"Desirability clone of {sname}: fitness_aggregation -> desirability with a "
				f"per-task CE anchor ({CE_ANCHOR_NORMALIZED} x H(p), derived in "
				f"ram_accelerator ABI 9). All other params byte-identical to the source."),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": p},
			"experiments": EXPERIMENTS,
		}
		r = post_with_retry(f"{DASHBOARD}/api/flows", body)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], name))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	print(f"flipping {len(created)} pending -> queued...")
	for fid, name in created:
		try:
			post_with_retry(f"{DASHBOARD}/api/flows/{fid}/restart", {})
		except requests.exceptions.RequestException:
			print(f"  ! {fid} {name}: flip failed after {FLIP_RETRIES} tries — still pending")
		time.sleep(0.5)

	con = ro()
	bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		ok = (ne == 2 and st == "queued"
		      and q.get("fitness_aggregation") == "desirability"
		      and q.get("fitness_ce_anchor_normalized") == CE_ANCHOR_NORMALIZED)
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} "
			      f"agg={q.get('fitness_aggregation')} anchor={q.get('fitness_ce_anchor_normalized')}")
			bad += 1
	con.close()
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} flows FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
