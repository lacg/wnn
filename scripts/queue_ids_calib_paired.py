"""IDS-48 (Luiz 02/11/2026): paired check of SUBSAMPLED THRESHOLD CALIBRATION
(flow param ids_calib_sample, worker ABI 24) on UNSW-NB15 temporal, QUAD.

WHAT IT TESTS: does calibrating each genome x fold SEARCH threshold on a fixed
stratified 200k-row train subsample (instead of all ~1.6M k-fold train rows)
leave the held-out result unchanged while cutting GA time? Validation thresholds
never subsample, so any held-out difference is the SEARCH steering differently.

DESIGN: clone the 10 IDS-41 "all" unswt QUAD flows (IDS41-all-unswt-QUAD-*,
seeds as banked) into 20 flows, paired by seed, both arms on the SAME ABI-24
wheel (the IDS-41 rows are an older ABI, so they are NOT the control):
  IDS48-calibfull-unswt-QUAD-r<seed>      ids_calib_sample absent (every row)
  IDS48-calibsub200k-unswt-QUAD-r<seed>   ids_calib_sample = 200000
Every other param byte-identical to the source. SEED-MAJOR interleave (full,
sub, full, sub, ...) — the worker is FIFO min-id, so both arms of a seed run
back to back under the same box conditions.

PRE-REGISTERED READ (fixed before any IDS-48 result exists):
  n = 10 seed pairs. Per pair d = calibsub - calibfull on the HELD-OUT test
  partition (validation_summaries, genome_type best of the GA phase as in the
  standard cohort table), for threshold modes val_cal and margin_val_cal where
  present:  dF1 (pp) and dFPR (pp). Report mean d with a 95% t-interval
  (t_{0.975, 9} = 2.262) over the 10 pairs.
  ADOPT ids_calib_sample=200000 as the new default IFF, for val_cal (and
  margin_val_cal where present), the 95% CI of mean dF1 lies within
  [-0.10, +0.10] pp AND the 95% CI of mean dFPR lies within [-0.10, +0.10] pp
  (equivalence, TOST-style margin +-0.10 pp). A CI that crosses a margin =
  NOT ADOPTED (inconclusive is not equivalence).
  ALSO REPORT: GA time ratio = sum(GA-phase wall time, calibsub) /
  sum(GA-phase wall time, calibfull) over the 10 pairs (and the per-pair median
  ratio), plus grid-phase ratio. Patience may end arms at different
  generations — report generations per arm alongside the time ratio, and the
  time per generation ratio.

Usage:
  python scripts/queue_ids_calib_paired.py [--dry-run]
Idempotent: existing non-cancelled IDS48 names are skipped. Needs the ABI-24
worker wheel deployed (the waiter logs/ids/queue_ids48_after_ids47_n3.sh checks).
"""
import json
import sys
import time

import requests

sys.path.insert(0, "/Users/lacg/wnn/scripts")
import queue_param_clone as q  # noqa: E402  (ro(), DASHBOARD, EXPERIMENTS, post_with_retry)

SOURCE_LIKE = "IDS41-all-unswt-QUAD-%"
EXPECT_SOURCES = 10
CALIB_SAMPLE = 200_000
ARMS = [("calibfull", {}), ("calibsub200k", {"ids_calib_sample": CALIB_SAMPLE})]


def sources(con) -> list[tuple[int, str, dict]]:
	"""(seed, source name, config) per seed — the first non-cancelled flow of each name."""
	out, seen = [], set()
	for name, cj in con.execute(
		"SELECT name, config_json FROM flows WHERE name LIKE ? AND status != 'cancelled' ORDER BY id",
		(SOURCE_LIKE,),
	):
		if name in seen:
			continue
		seen.add(name)
		cfg = json.loads(cj)
		out.append((int(cfg["params"]["seed"]), name, cfg))
	return sorted(out)


def plan(con) -> list[tuple[str, str, dict, dict]]:
	"""(target name, source name, source config, overrides), seed-major."""
	rows = []
	for seed, sname, cfg in sources(con):
		for arm, sets in ARMS:
			rows.append((f"IDS48-{arm}-unswt-QUAD-r{seed}", sname, cfg, sets))
	return rows


def create(name: str, sname: str, cfg: dict, sets: dict) -> int:
	params = dict(cfg["params"])
	params.pop("ids_calib_sample", None)
	params.update(sets)
	body = {
		"name": name,
		"description": (f"IDS-48 paired calib check: clone of {sname}, ids_calib_sample="
			f"{sets.get('ids_calib_sample', 'absent (every row)')}. Pre-registered read in "
			"scripts/queue_ids_calib_paired.py."),
		"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": params},
		"experiments": q.EXPERIMENTS,
	}
	r = q.post_with_retry(f"{q.DASHBOARD}/api/flows", body)
	if r.status_code not in (200, 201):
		raise SystemExit(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
	return r.json()["id"]


def queue(fid: int) -> bool:
	for _ in range(5):
		r = requests.patch(f"{q.DASHBOARD}/api/flows/{fid}", json={"status": "queued"}, verify=False, timeout=30)
		if r.status_code == 200:
			return True
		time.sleep(1)
	return False


def verify(created: list[tuple[int, str, dict]]) -> int:
	con = q.ro()
	bad = 0
	for fid, name, sets in created:
		st, cj = con.execute("SELECT status, config_json FROM flows WHERE id=?", (fid,)).fetchone()
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		got = json.loads(cj)["params"].get("ids_calib_sample")
		if ne != 2 or st != "queued" or got != sets.get("ids_calib_sample"):
			print(f"  ! VERIFY FAILED {fid} {name}: status={st} exps={ne} ids_calib_sample={got}")
			bad += 1
	con.close()
	return bad


def main() -> int:
	con = q.ro()
	rows = plan(con)
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'IDS48-%' AND status != 'cancelled'")}
	con.close()
	if len(rows) != 2 * EXPECT_SOURCES:
		print(f"REFUSED: expected {EXPECT_SOURCES} source flows ({SOURCE_LIKE}), found {len(rows) // 2}")
		return 4
	print(f"IDS-48: {len(rows)} flows ({len(rows) // 2} seeds x {len(ARMS)} arms), seed-major")
	if "--dry-run" in sys.argv:
		for name, sname, _, sets in rows:
			print(f"  would create {name:44s} <- {sname}  {sets or '(every row)'}")
		return 0
	created = []
	for name, sname, cfg, sets in rows:
		if name in existing:
			print(f"  = exists {name}")
			continue
		fid = create(name, sname, cfg, sets)
		if not queue(fid):
			print(f"  ! {fid} {name}: queue PATCH failed — left pending")
		created.append((fid, name, sets))
		print(f"  + {fid}  {name}")
		time.sleep(0.2)
	time.sleep(1)
	bad = verify(created)
	print(f"created {len(created)}, verify failures {bad}")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
