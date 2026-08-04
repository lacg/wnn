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

	# R1 — watchdog stop: WAIT FOR MEMORY, THEN RETRY.
	#
	# Restored 04/08/2026. run_dfa_1layer_study.sh has always had this pause protocol;
	# when the P2/P3/P4 chains were migrated onto this helper it was dropped, so a
	# watchdog kill simply abandoned the run and the chain moved to the next one — the
	# work was silently lost and only a fresh chain invocation would redo it.
	#
	# That gap did not bite while it existed because the controller sat at 0.04-0.37 GB,
	# below the watchdog's CTRL_MIN_RSS efficacy gate (default 4 GB), so it rode out
	# every alarm instead of killing anything. QUAD runs changed that: the live one is at
	# 4.10 GB and P4 peaked at 6.6/8.9 GB, so the controller is now ABOVE the gate and a
	# HARD-floor trip will kill it. The watchdog is behaving correctly — killing a 4-9 GB
	# process does free memory — which is exactly why the caller must be able to survive
	# it. Losing ~1.6 h of work to a transient memory spike is not an acceptable outcome.
	#
	# Waits for a SUSTAINED recovery (3 consecutive 60s samples above 25 GB), not a
	# single reading, so a retry does not fire straight back into the pressure that
	# caused the kill. Caps at 2 retries: a run killed three times has a problem that
	# retrying will not fix, and looping forever would hide it.
	if [ "$rc" = "143" ] || [ "$rc" = "137" ]; then
		local tries="${WNN_ARM_TRIES:-0}"
		if [ "$tries" -ge 2 ]; then
			"$logfn" "$tag: rc=$rc, killed $((tries + 1))x — GIVING UP (needs a fix, not a retry). NO marker."
			return 1
		fi
		"$logfn" "$tag: rc=$rc (watchdog stop) — NO marker; waiting for memory to recover, will retry"
		local calm=0 av
		while [ "$calm" -lt 3 ]; do
			sleep 60
			av=$(vm_stat | awk '/free|inactive|speculative|purgeable/ {gsub("\\.","");s+=$NF} END {printf "%.0f", s*16384/1073741824}')
			if [ "${av:-0}" -ge 25 ]; then calm=$((calm + 1)); else calm=0; fi
			"$logfn" "$tag: waiting to retry (avail=${av}GB, calm=${calm}/3)"
		done
		"$logfn" "$tag: memory recovered — retry $((tries + 1))/2"
		WNN_ARM_TRIES=$((tries + 1)) run_controller_arm "$tag" "$markdir" "$outdir" "$vp" "$logfn" "$extra" -- "$@"
		return $?
	fi
	# R2 — crash.
	if [ "$rc" != "0" ]; then
		"$logfn" "$tag: rc=$rc (crash) — NO marker, leaving for re-run"
		return 2
	fi

	local rss held_n held_m held_nm held_mm cells fpga
	rss=$(grep -E "maximum resident set size" "$out" | awk '{print $1}' | tail -1)
	# 04/08/2026 FIX. This used to take RESULT lines 1 and 2 and call them NEURONS and
	# MEMORY. With --report-seeds N each stage emits N RESULT lines, so at N=5 lines 1
	# AND 2 are both NEURONS (report seeds 1 and 2) and "held_memory" was never the
	# memory stage at all — every marker written by this helper carried a mislabelled
	# field, and it was quoted as a MEMORY result. Anchor on the stage headers instead,
	# which hold for N=1 and N>1 alike.
	held_n=$(awk '/STAGE 1 \(NEURONS\) done/{f=1} f && /RESULT — during-search winner/{print; exit}' "$out")
	held_m=$(awk '/STAGE 4 \(MEMORY\) done/{f=1} f && /RESULT — during-search winner/{print; exit}' "$out")
	# The AUTHORITATIVE numbers when --report-seeds is used: mean±SD across report seeds
	# rather than one draw. Empty for single-seed runs, which emit no MULTI-SEED line.
	# Captured because a single RESULT line is one report seed and cannot carry variance
	# — reporting it as "the" held-out result is how a 1.99° draw got quoted against a
	# 1.73±0.06° aggregate.
	held_nm=$(grep -E "NEURONS MULTI-SEED held-out" "$out" | tail -1)
	held_mm=$(grep -E "MEMORY MULTI-SEED held-out" "$out" | tail -1)
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
	printf '{"tag":"%s",%s"rc":%s,"dur_s":%s,"peak_rss_bytes":%s,"cells":"%s","fpga":"%s","held_neurons":"%s","held_memory":"%s","held_neurons_multiseed":"%s","held_memory_multiseed":"%s","fixed_thresholds":true,"done":"%s"}\n' \
		"$tag" "$extra" "$rc" "$dur" "${rss:-null}" \
		"$cells" \
		"$(echo "$fpga"   | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_n"  | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_m"  | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_nm" | tr -d '"' | sed 's/  */ /g')" \
		"$(echo "$held_mm" | tr -d '"' | sed 's/  */ /g')" \
		"$(date -u +%FT%TZ)" > "$marker"
	"$logfn" "$tag: rc=0 dur=${dur}s — marker written"
	return 0
}
