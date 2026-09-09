#!/usr/bin/env bash
# Wait for the box to go IDLE (no controller, no ladder, no chain), then hand it to
# scripts/label_scale_arm_chain.sh — which re-checks every precondition itself and
# fails closed. Pure wait; never preempts. (08/09/2026, Luiz option b: the in-flight
# window-k run banks, then arm A takes the box.)
set -u
cd /Users/lacg/wnn || exit 1
LOG="/private/tmp/label_scale_arm.log"
log() { echo "[launch-when-idle] $(date -u +%FT%TZ) $*" >> "$LOG"; }
busy() { pgrep -f -- "-m wnn.control.phased_ga" >/dev/null 2>&1 || pgrep -f "scripts/sweep_ladder_gamma.sh" >/dev/null 2>&1 || pgrep -f "scripts/(mutstep_ab|queue_after_ab|leak_revisit|crn_|translation_ab|window_k)[a-z_]*chain" >/dev/null 2>&1; }
log "armed — waiting for the box to go idle before launching label_scale_arm_chain.sh"
beat=0
while busy; do
	[ $((beat % 30)) = 0 ] && log "waiting — box busy"
	beat=$((beat + 1)); sleep 60
done
sleep 120   # let the ladder bank its marker and exit cleanly
log "box idle — launching scripts/label_scale_arm_chain.sh"
exec bash scripts/label_scale_arm_chain.sh
