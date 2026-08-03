# shellcheck shell=bash
# run_controller_arm — run ONE phased_ga cell and write its completion marker.
#
# Extracted from run_l3d_feature_probe.sh:run_arm, which is where these rules were
# learned the hard way. The marker is a CLAIM THAT THE CELL GENUINELY FINISHED, and
# every guard below exists to stop a marker being written for a run that did not:
#
#   R1  rc=143/137 (watchdog SIGTERM/SIGKILL) -> NO marker. The box was taken from
#       us; the cell is re-runnable and must stay re-runnable.
#   R2  rc!=0 (crash)                          -> NO marker, and NO auto-retry: a
#       human should see a crash. QUAD-dfa additionally carries the attempt-3 limit.
#   R3  rc=0 but an EMPTY MEMORY-stage triple  -> NO marker. A truncated run exits
#       clean but never printed its held-out result; a marker here would silently
#       enter a hole into the study table as if it were a measurement.
#
# Marker absence is therefore always "re-run me", never "this cell scored badly" —
# which is what lets the sweep supervisor and the P2/P3 chains resume idempotently.
#
# Usage:
#   source scripts/lib/controller_arm.sh
#   run_controller_arm <tag> <markdir> <outdir> <python> <logfn> <extra-json> -- <phased_ga args...>
#
#   <logfn>       name of a shell function taking one string (the caller's logger)
#   <extra-json>  raw JSON fields spliced into the marker, e.g. '"arm":"P2","k":8'
#                 (may be empty). Caller owns their own schema beyond the common core.
#
# Returns 0 if a marker was written, non-zero otherwise.

run_controller_arm() {
	local tag="$1" markdir="$2" outdir="$3" vp="$4" logfn="$5" extra="$6"
	shift 6
	[ "${1:-}" = "--" ] && shift

	local marker="${markdir}/${tag}.json"
	local out="${outdir}/${tag}.out"
	local winner="${outdir}/${tag}_winner.yaml.gz"

	if [ -f "$marker" ]; then
		"$logfn" "$tag: marker exists — skip"
		return 0
	fi

	"$logfn" "===== START $tag ====="
	local t0=$SECONDS
	/usr/bin/time -l "$vp" -u -m wnn.control.phased_ga "$@" \
		--save-winner "$winner" > "$out" 2>&1
	local rc=$? dur=$((SECONDS - t0))

	# R1 — watchdog stop.
	if [ "$rc" = "143" ] || [ "$rc" = "137" ]; then
		"$logfn" "$tag: rc=$rc (watchdog stop) — NO marker, leaving for re-run"
		return 1
	fi
	# R2 — crash.
	if [ "$rc" != "0" ]; then
		"$logfn" "$tag: rc=$rc (crash) — NO marker, leaving for re-run"
		return 2
	fi

	local rss held_n held_m cells fpga
	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	held_n=$(grep -E "RESULT — during-search winner" "$out" | sed -n '1p')
	held_m=$(grep -E "RESULT — during-search winner" "$out" | sed -n '2p')
	[ -f "$winner" ] && "$vp" -u scripts/gran_fpga_count.py "$winner" >> "$out" 2>&1
	fpga=$(grep -E "^\[FPGA\]" "$out" | tail -1)
	cells=$(grep -oE "cells\[[0-9-]+ Σ[0-9]+k μ[0-9]+k\]" "$out" | tail -1)

	# R3 — clean exit, no MEMORY triple.
	if [ -z "${held_m// /}" ]; then
		"$logfn" "$tag: rc=0 but no MEMORY-stage held-out (truncated) — NO marker, leaving for re-run"
		return 3
	fi

	# Field ORDER matters only for byte-parity with the markers run_l3d_feature_probe.sh
	# wrote before it was migrated onto this helper; readers go through json.load.
	[ -n "$extra" ] && extra="${extra},"
	printf '{"tag":"%s",%s"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"cells":"%s","fpga":"%s","held_neurons":"%s","held_memory":"%s","fixed_thresholds":true,"done":"%s"}\n' \
		"$tag" "$extra" "$rc" "$dur" "${rss:-null}" \
		"$cells" \
		"$(echo "$fpga"   | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_n" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_m" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	"$logfn" "$tag: rc=0 dur=${dur}s — marker written"
	return 0
}
