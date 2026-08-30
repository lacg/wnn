"""Re-run the completed flows whose WINNER carried >64-bit neurons, on the fixed wheel.

WHY (29/08/2026, memory: project_bits_above_64_or_fold). Every address function built
the neuron's address with `1 << (bits-1-i)` into a u64. Release builds mask the shift
count, so above 64 bits connection slot i and slot i+64 wrote the SAME address bit,
merged by OR. A "96-bit" neuron was a 64-bit neuron with 32 input pairs folded together,
and the fold was biased (P(bit=1) 0.5 -> 0.75) so its addresses clustered. Those runs are
true measurements of "64-bit OR-folded neurons"; they are invalid ONLY as evidence about
wide neurons. The fix (hash wide tuples, identity at <=64) is in ram_core and the worker
wheel is ABI 12.

BLAST RADIUS IS THE WINNER'S WIDTH, NOT THE CONFIG. Every flow in these arms was
configured max_bits=100, but the GA only sometimes CHOSE a neuron above 64. Selecting by
config would re-run 40 flows; selecting by the winner's actual widest neuron re-runs 18.
The detector below is the same one that reproduced the known IDSXD answer (flows 5888 and
5890, and only those, out of the completed ciciot-quad-96b set) before it was trusted here.

ORDER MATTERS: the worker's admit() takes min(id) among status='queued', so creation
order IS execution order. The abl2big reruns (6052-6071) were created arm-major, which is
the shape that cost IDSXD 37.7 h on a saturated cell before the other arms had any data
(memory: feedback_sweeps_always_interleave). These are emitted ROUND-ROBIN across arms, so
stopping the queue at any point leaves every arm with roughly equal n.

Sources are LEFT AS THEY ARE — never deleted, never edited. Per CLAUDE.md Rule 2 the
clones are created through POST /api/flows, never by direct SQL insert.

Usage:
    python scripts/queue_w64fix_reruns.py --arms SP-ciciot-ablpln,SP-ciciot-ablqsr,\
SP-ciciot-abl3s,SP-ciciot-bin --expect 18 [--dry-run] [--only <substr>]
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
SUFFIX = "-w64fix"
WIDTH_LIMIT = 64
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]
POST_TIMEOUT = 60      # the dashboard reads slowly while the worker trains
FLIP_RETRIES = 3


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def arg(flag: str, default=None):
	for i, a in enumerate(sys.argv):
		if a == flag and i + 1 < len(sys.argv):
			return sys.argv[i + 1]
	return default


def widest_neuron(tiers_json: str) -> int:
	"""Widest bits_per_neuron in a banked genome. Handles both shapes the column has
	carried: a dict with a bits_per_neuron list, and the older list-of-tiers."""
	try:
		t = json.loads(tiers_json)
	except (TypeError, ValueError):
		return 0
	if isinstance(t, dict):
		t = [t]
	if not isinstance(t, list):
		return 0
	widest = 0
	for entry in t:
		if not isinstance(entry, dict):
			continue
		bpn = entry.get("bits_per_neuron")
		if isinstance(bpn, list) and bpn:
			widest = max(widest, max(bpn))
		else:
			widest = max(widest, int(entry.get("bits", 0) or 0))
	return widest


def affected(con, arm: str) -> list:
	"""Completed flows in `arm` whose banked winner has a neuron wider than 64 bits."""
	rows = con.execute(
		"""SELECT f.id, f.name, f.config_json, g.tiers_json
		   FROM flows f
		   JOIN best_genomes bg ON bg.flow_id = f.id
		   JOIN genomes g ON g.id = bg.genome_id
		   WHERE f.name LIKE ? AND f.status = 'completed'""", (f"{arm}%",)).fetchall()
	per = {}
	for fid, name, cfg, tiers in rows:
		cur = per.setdefault(fid, [name, cfg, 0])
		cur[2] = max(cur[2], widest_neuron(tiers))
	return [(fid, n, c, w) for fid, (n, c, w) in sorted(per.items()) if w > WIDTH_LIMIT]


def interleave(per_arm: dict) -> list:
	"""Round-robin across arms so an early stop leaves every arm with data."""
	out, queues = [], {a: list(v) for a, v in per_arm.items()}
	while any(queues.values()):
		for arm in per_arm:                     # stable arm order = the --arms order
			if queues[arm]:
				out.append((arm, queues[arm].pop(0)))
	return out


def post_with_retry(url: str, body: dict):
	"""The dashboard read-times-out while the worker trains; retry rather than
	abandon a half-created cohort."""
	last = None
	for _ in range(FLIP_RETRIES):
		try:
			return requests.post(url, json=body, verify=False, timeout=POST_TIMEOUT)
		except requests.exceptions.RequestException as e:
			last = e
			time.sleep(5)
	raise last


def describe(src_id: int, src_name: str, widest: int) -> str:
	return (f"RERUN of flow {src_id} ({src_name}) on the ABI-12 wheel: its winner carried "
	        f"{widest}-bit neurons, which the pre-fix wheel OR-folded onto 64 address bits "
	        f"(project_bits_above_64_or_fold) — connection slots i and i+64 shared one bit "
	        f"and the fold was biased, so the banked result measures a folded 64-bit neuron, "
	        f"not a {widest}-bit one. Params byte-identical to the source. "
	        f"Created 30/08/2026; runs behind IDSXD and the abl2big reruns (FIFO by id).")


def collect(con, arms: list, only: str | None) -> list:
	per_arm = {}
	for arm in arms:
		hits = affected(con, arm)
		if only:
			hits = [h for h in hits if only in h[1]]
		per_arm[arm] = hits
		print(f"  {arm:<24} {len(hits)} of "
		      f"{con.execute('SELECT COUNT(*) FROM flows WHERE name LIKE ?', (arm + '%',)).fetchone()[0]}"
		      f" flows have a >{WIDTH_LIMIT}-bit winner")
	return interleave(per_arm)


def verify(created: list) -> int:
	con, bad = ro(), 0
	for fid, name, src_params in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		got = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?",
		                             (fid,)).fetchone()[0])["params"]
		if ne != 2 or st != "queued" or got != src_params:
			drift = {k: (src_params.get(k), got.get(k))
			         for k in set(src_params) | set(got) if src_params.get(k) != got.get(k)}
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} param_drift={drift}")
			bad += 1
	con.close()
	return bad


def main() -> int:
	arms = [a for a in (arg("--arms") or "").split(",") if a]
	if not arms:
		print(__doc__)
		return 64
	expect, only = arg("--expect"), arg("--only")

	con = ro()
	print(f"scanning {len(arms)} arm(s) for winners wider than {WIDTH_LIMIT} bits:")
	targets = collect(con, arms, only)
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE ?",
	                                      (f"%{SUFFIX}",))}
	con.close()

	if expect and len(targets) != int(expect):
		print(f"REFUSED: expected {expect} affected flows, found {len(targets)}")
		return 4
	print(f"\n{len(targets)} rerun(s), interleaved across arms:")
	for arm, (fid, name, _, widest) in targets:
		mark = "= exists" if name + SUFFIX in existing else "  new   "
		print(f"  {mark}  {fid:>5}  w={widest:>4}  {name}{SUFFIX}")
	if "--dry-run" in sys.argv:
		print("\nDRY RUN — nothing created.")
		return 0

	created = []
	for _, (fid, name, cfg_json, widest) in targets:
		target_name = name + SUFFIX
		if target_name in existing:
			print(f"  = exists {target_name}")
			continue
		cfg = json.loads(cfg_json)
		params = dict(cfg["params"])                    # byte-identical: nothing changes
		body = {
			"name": target_name,
			"description": describe(fid, name, widest),
			"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": params},
			"experiments": EXPERIMENTS,
		}
		r = post_with_retry(f"{DASHBOARD}/api/flows", body)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {target_name} ({r.status_code}) {r.text[:200]}")
			return 2
		created.append((r.json()["id"], target_name, params))
		print(f"  + {r.json()['id']:>5}  {target_name}")
		time.sleep(0.2)

	print(f"\nflipping {len(created)} pending -> queued...")
	for fid, name, _ in created:
		try:
			post_with_retry(f"{DASHBOARD}/api/flows/{fid}/restart", {})
		except requests.exceptions.RequestException:
			print(f"  ! {fid} {name}: flip failed after {FLIP_RETRIES} tries — still pending")
		time.sleep(0.5)

	bad = verify(created)
	print(f"ALL {len(created)} VERIFIED" if not bad else f"{bad} flow(s) FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
