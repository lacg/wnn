#!/bin/bash
# probe_gran_tern_bin_chain.sh — after the running QUAD probe exits, run the
# TERNARY then BINARY probes (same 8x8 recipe) so we can compare Σcells growth
# across all 3 modes and confirm/localize the TERNARY/BINARY memory blow-up.
# ONE controller at a time (waits on each probe's done-marker). Watchdog-guarded.
set -u
PROJ="/Users/lacg/wnn"
log() { echo "[probe-chain] $1 $(date -u +%FT%TZ)"; }

wait_marker() {  # $1 = marker path, up to 120 min
	local m="$1"
	for _ in $(seq 1 1440); do [ -f "$m" ] && return 0; sleep 5; done
	log "TIMEOUT waiting $m"; return 1
}
wait_no_ctrl() {  # ensure no phased_ga is alive before launching the next
	for _ in $(seq 1 360); do pgrep -f wnn.control.phased_ga >/dev/null || return 0; sleep 5; done
}

log "waiting for QUAD probe to finish (/tmp/wnn_gran_probe_done.json)…"
wait_marker /tmp/wnn_gran_probe_done.json
wait_no_ctrl

log "===== TERNARY probe ====="
bash "$PROJ/scripts/probe_gran_mode.sh" TERNARY
wait_no_ctrl

log "===== BINARY probe ====="
bash "$PROJ/scripts/probe_gran_mode.sh" BINARY

echo "{\"done\": \"$(date -u +%FT%TZ)\", \"kind\": \"gran_mode_compare\"}" > /tmp/wnn_gran_mode_compare_done.json
log "ALL 3 MODE PROBES DONE (compare Σcells)"
