#!/usr/bin/env python3
"""Create the SP 46M n=5 wave (S&P 2027, Protocol v2 streaming, 3-way).

Template = cancelled flow 4299 (ciciot2023_neto_full, 96b-Wc, 250n100b).
Queues at the BACK of the FIFO queue — the 46M wave is the last stretch,
running after the n=30 subsample cohorts drain. Requires the ABI-4 wheel
(streaming val-plumbing) — installed 11/07.

Usage:
	python scripts/create_sp_46m.py --dry-run
	python scripts/create_sp_46m.py --queue
"""

import argparse
import random
import sqlite3

from create_sp_wave1 import (
	DB_PATH, RNG_SEED, build_flow, fetch_template, fresh_seeds, patch_queued,
	post_flow, used_seeds,
)

TEMPLATE_FLOW = 4299
TAG = "96bWc"
N_WAVE = 5


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--dry-run", action="store_true")
	ap.add_argument("--queue", action="store_true")
	args = ap.parse_args()

	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	template = fetch_template(con, TEMPLATE_FLOW)
	taken = used_seeds(con)
	con.close()

	rng = random.Random(RNG_SEED + 46)
	seeds = fresh_seeds(N_WAVE, taken, rng)
	plan = [build_flow(template, "ciciot46m", TAG, s, "bin", N_WAVE, None) for s in seeds]

	for f in plan:
		print(f"  {f['name']}  split={f['config']['params']['ids_split']}")
	if args.dry_run:
		return

	created = []
	for f in plan:
		resp = post_flow(f)
		fid = resp.get("id") or resp.get("flow", {}).get("id")
		created.append((fid, f["name"]))
	con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
	bad = [fid for fid, _ in created if con.execute(
		"SELECT COUNT(*) FROM experiments WHERE flow_id = ?", (fid,)).fetchone()[0] != 2]
	con.close()
	if bad:
		raise SystemExit(f"FLOWS WITH WRONG EXPERIMENT COUNT: {bad}")
	print(f"Created {len(created)} flows: ids {created[0][0]}..{created[-1][0]}, experiments verified")
	if args.queue:
		for fid, _ in created:
			patch_queued(fid)
		print("Queued (back of FIFO — last stretch by design).")


if __name__ == "__main__":
	main()
