#!/usr/bin/env bash
# Rebuild dfa1l per-cell markers from the .out logs.
#
# The markers are only a SUMMARY of each run — every field they hold is printed
# in the cell's own .out log, which lives under logs/controller/dfa1l/ and is
# never reaped. So a marker lost to the macOS /tmp cleaner (5-day rule) is
# recoverable, not gone. This script re-derives it, using the SAME extraction
# the driver's run_cell() uses, so a regenerated marker is byte-comparable to a
# natively-written one apart from the provenance fields.
#
# Only genuinely-complete runs get a marker, matching run_cell()'s R4 rule: a
# MEMORY-stage held-out triple must be present, else the cell is left for re-run.
#
# Usage: dfa1l_regen_markers.sh <outdir> <markdir>
set -u
OUTDIR="${1:-logs/controller/dfa1l}"
MARKDIR="${2:?markdir required}"
mkdir -p "$MARKDIR"

for out in "$OUTDIR"/*.out; do
	[ -f "$out" ] || continue
	tag=$(basename "$out" .out)
	marker="${MARKDIR}/${tag}.json"
	[ -f "$marker" ] && { echo "skip  $tag (marker exists)"; continue; }

	# tag = <sub>_<feat>_<mode>_s<seed> ; no field contains an underscore.
	sub=${tag%%_*};              rest=${tag#*_}
	feat=${rest%%_*};            rest=${rest#*_}
	mode=${rest%%_*};            seed=${rest#*_s}

	held_n=$(grep -E "RESULT — during-search winner" "$out" | sed -n '1p')
	held_m=$(grep -E "RESULT — during-search winner" "$out" | sed -n '2p')
	if [ -z "${held_m// /}" ]; then
		echo "SKIP  $tag — no MEMORY-stage held-out (incomplete run, leave for re-run)"
		continue
	fi

	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	dur=$(grep -E "[0-9.]+ real" "$out" | awk '{print $1}' | tail -1 | cut -d. -f1)
	cells=$(grep -oE "cells\[[0-9-]+ Σ[0-9]+k μ[0-9]+k\]" "$out" | tail -1)
	fpga=$(grep -E "^\[FPGA\]" "$out" | tail -1)

	# R4-equivalent for regeneration. run_cell() can read the process rc; we cannot,
	# so we test the two footers that ONLY a normal exit produces:
	#   * `N.NN real`  — the /usr/bin/time wrapper's footer, absent on SIGTERM/SIGKILL
	#   * `[FPGA] ...` — written by the driver's post-run gran_fpga_count.py step
	# This matters because a SIGTERM'd run can still print a full-looking STAGE/HELD-OUT
	# sequence with all-zero metrics (stable=0.0% err=0.00°), which would otherwise mint
	# a marker that both permanently skips the cell and poisons the table with a 0.0 row.
	# A saved _winner.yaml.gz is NOT a completeness signal — the graceful dump writes one.
	if [ -z "${dur// /}" ] || [ -z "${fpga// /}" ]; then
		echo "SKIP  $tag — no time/FPGA footer (killed mid-run, not a completion) — leave for re-run"
		continue
	fi
	done_at=$(date -u -r "$out" +%FT%TZ)

	printf '{"tag":"%s","substrate":"%s","feature":"%s","mode":"%s","seed":%s,"rc":0,"dur_s":%s,"peak_rss_bytes":%s,"cells":"%s","fpga":"%s","held_neurons":"%s","held_memory":"%s","done":"%s","regen_from_log":true}\n' \
		"$tag" "$sub" "$feat" "$mode" "$seed" "${dur:-null}" "${rss:-null}" \
		"$cells" \
		"$(echo "$fpga"   | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_n" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_m" | tr -d '"' | sed 's/  */ /g')" \
		"$done_at" > "$marker"
	echo "regen $tag  dur=${dur}s rss=${rss}"
done
