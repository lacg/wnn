#!/usr/bin/env python3
"""
Backfill genomes.materialized_cells — the real sparse footprint primitive.

See docs/sparse_footprint_fix.md. The dense `total_memory_bytes` column is a
fiction (i64::MAX for high-bit, 0 for ids); the real materialized-cell count was
never stored but is deterministically reproducible (connections in DB + flow seed
+ dataset). This script (re)measures it for the paper-relevant genomes.

SCOPE: `best_genomes` first (dedupes to the leaderboard genomes that matter).

STATUS: SKELETON — the per-genome measurement depends on the Rust primitive
`IDSCacheWrapper.measure_genome_memory` (step 5 of the plan), which lands with the
worker-idle accelerator rebuild. Until then:
  --dry-run  → fully works: prints the target set grouped by dataset (no measuring).
  (real run) → raises NotImplementedError at the measure step (no silent wrong data).

Usage:
	python scripts/backfill_sparse_memory.py --dry-run
	python scripts/backfill_sparse_memory.py            # after step 5 + migration
"""

import argparse
import sqlite3
import sys
from collections import defaultdict


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
	cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
	return column in cols


def _target_genomes(conn: sqlite3.Connection, skip_measured: bool) -> list[int]:
	"""Distinct genome_ids referenced by best_genomes (the paper-relevant set)."""
	# NOTE: only filter on materialized_cells once the column exists (post-migration).
	if skip_measured and _column_exists(conn, "genomes", "materialized_cells"):
		sql = (
			"SELECT DISTINCT bg.genome_id FROM best_genomes bg "
			"JOIN genomes g ON g.id = bg.genome_id "
			"WHERE g.materialized_cells IS NULL"
		)
	else:
		sql = "SELECT DISTINCT genome_id FROM best_genomes"
	return [r[0] for r in conn.execute(sql).fetchall()]


def _dataset_key(conn: sqlite3.Connection, genome_id: int) -> tuple:
	"""Group key = the IDS dataset config of the genome's flow, so each dataset
	loads ONCE (the heavy step). Reads flow params via the experiment link."""
	row = conn.execute(
		"SELECT f.config_json FROM genomes g "
		"JOIN experiments e ON e.id = g.experiment_id "
		"JOIN flows f ON f.id = e.flow_id WHERE g.id = ?",
		(genome_id,),
	).fetchone()
	if not row or not row[0]:
		return ("<unknown>",)
	import json
	params = json.loads(row[0]).get("params", {})
	return (
		params.get("ids_dataset"),
		params.get("ids_split"),
		params.get("ids_feature_selection"),
		params.get("ids_n_bits"),
	)


def _measure_genome(conn: sqlite3.Connection, genome_id: int, cache) -> int:
	"""Train the genome against the cached dataset and return materialized_cells.

	TODO(step 5): wire to the Rust primitive once built:
	    entries = cache.measure_genome_memory(genome_bits, genome_neurons, connections)
	It must reuse the already-loaded IDSCacheWrapper (zero re-upload) and return
	GenomeExport::materialized_cells() — NO Python reimplementation (no-shortcuts rule).
	"""
	raise NotImplementedError(
		"measure step needs Rust IDSCacheWrapper.measure_genome_memory (step 5, "
		"lands with the idle-window accelerator rebuild). Use --dry-run for now."
	)


def _update(conn: sqlite3.Connection, genome_id: int, cells: int) -> None:
	conn.execute("UPDATE genomes SET materialized_cells = ? WHERE id = ?", (cells, genome_id))
	conn.commit()


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--db", default="db/wnn.db")
	ap.add_argument("--scope", choices=["best_genomes"], default="best_genomes")
	ap.add_argument("--dry-run", action="store_true", help="print the plan; do not measure or write")
	ap.add_argument("--limit", type=int, default=None, help="cap the number of genomes (testing)")
	args = ap.parse_args()

	conn = sqlite3.connect(args.db)
	has_col = _column_exists(conn, "genomes", "materialized_cells")
	if not has_col:
		print("[note] genomes.materialized_cells not present yet — runs on the next "
		      "dashboard/worker restart (idle-window migration). Proceeding for planning.")

	targets = _target_genomes(conn, skip_measured=True)
	if args.limit:
		targets = targets[: args.limit]

	# Group by dataset so each IDS dataset loads once.
	by_dataset: dict[tuple, list[int]] = defaultdict(list)
	for gid in targets:
		by_dataset[_dataset_key(conn, gid)].append(gid)

	print(f"Scope={args.scope}  target genomes={len(targets)}  datasets={len(by_dataset)}")
	for key, gids in sorted(by_dataset.items(), key=lambda kv: -len(kv[1])):
		print(f"  dataset {key}: {len(gids)} genomes")

	if args.dry_run:
		print("\n[dry-run] no measurement / no writes. Re-run without --dry-run after step 5.")
		return 0

	if not has_col:
		print("\n[abort] column missing — apply the migration (restart) before a real run.")
		return 1

	done = 0
	for key, gids in by_dataset.items():
		# TODO(step 5): load the IDS dataset for `key` ONCE into an IDSCacheWrapper,
		# then loop gids measuring against it.
		cache = None  # = build_ids_cache(key)
		for gid in gids:
			cells = _measure_genome(conn, gid, cache)
			_update(conn, gid, cells)
			done += 1
	print(f"Backfilled {done} genomes.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
