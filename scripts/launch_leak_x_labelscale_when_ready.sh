#!/usr/bin/env bash
# Wait until ARM A is COMPLETE (12 _ls markers, its chain gone) and the box is IDLE,
# then hand the box to scripts/leak_x_labelscale_chain.sh (which re-checks everything
# and fails closed). Pure wait; never preempts. (09/09/2026, Luiz: "queue the 2x2 leak
# x label-scale after arm A".)
set -u
cd /Users/lacg/wnn || exit 1
LOG="/private/tmp/leak_x_labelscale.log"
log() { echo "[launch-when-ready] $(date -u +%FT%TZ) $*" >> "$LOG"; }
arm_a_done() { [ "$(ls experiments/sweepladder_markers | grep -cE '_ls[248]\.json$')" -ge 12 ] && ! pgrep -f "scripts/label_scale_arm_chain.sh" >/dev/null 2>&1; }
busy() { pgrep -f -- "-m wnn.control.phased_ga" >/dev/null 2>&1 || pgrep -f "scripts/sweep_ladder_gamma.sh" >/dev/null 2>&1 || pgrep -f "scripts/[a-z_]*chain\.sh" >/dev/null 2>&1; }
log "armed — waiting for arm A (12 _ls markers) and an idle box"
beat=0
while ! arm_a_done || busy; do
	[ $((beat % 60)) = 0 ] && log "waiting — arm A $(ls experiments/sweepladder_markers | grep -cE '_ls[248]\.json$')/12, box $(busy && echo busy || echo idle)"
	beat=$((beat + 1)); sleep 60
done
sleep 120
log "arm A complete and box idle — launching scripts/leak_x_labelscale_chain.sh"
exec bash scripts/leak_x_labelscale_chain.sh
