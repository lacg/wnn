#!/usr/bin/env bash
# POST-_dl QUEUE (Luiz, 18/09/2026 15:0x EDT) — ONE arm, after post_repair_queue.sh (pid 28596)
# has finished STEP 2 (`_dl` x4) and released the GPU. Idle-gated, marker-gated, fails closed,
# honours experiments/HOLD_CONTROLLER (through the ladder). Log /private/tmp/post_dl_queue.log.
#
#   STEP 1  `_hd` RE-FLY ON ABI 29 (`_hd29`) — the stage D / arm B / label-only pairs all read
#           against `_hd` s2-s5, which flew 11-12/09 on the PRE-ABI-28 trainer (stale-altitude
#           fix). Same-seed GRID rows differ between eras, so every "arm − _hd" delta carries a
#           trainer-era component (the STRADDLE caveat). This re-flies the control recipe
#           unchanged (ladder recipe + --teacher-hover derived, no extra flags) on the installed
#           wheel so the pairs can be re-read era-clean. Paired against the ORIGINAL `_hd` as an
#           era A/B (same recipe, same seed; the only difference is the trainer). Lever line:
#           "clean control: how much of the arm gains was trainer era?"
#   STEP 2  STOP — box idle, say so.
#
# The running post_repair_queue.sh is NOT edited (bash resumes at a byte offset) — this waits for
# it to exit. FAILS CLOSED if `_dl` did not reach 4/4 (the queue aborted => a human is needed).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/post_dl_queue.log"
MARK="experiments/sweepladder_markers"
PREV_PID="${PREV_PID:-28596}"
log() { echo "[post-dl] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
count() { ls ${MARK}/SL_C_b24n256_cf21_brushless_L4C_g10_s3133700[2-5]$1.json 2>/dev/null | wc -l | tr -d ' '; }
prev_running() { kill -0 "$PREV_PID" 2>/dev/null; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null || pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null; }

log "########## ARMED — wait for post_repair_queue (pid ${PREV_PID}) -> _hd re-fly on ABI 29 (_hd29 x4, control _hd) -> STOP ##########"
beat=0
while prev_running || busy; do
	sleep 60; beat=$((beat + 1))
	[ $((beat % 30)) = 0 ] && log "waiting — previous queue still running (_dl $(count _dl)/4)"
done
log "previous queue exited; box idle (_dl $(count _dl)/4)"
[ "$(count _dl)" -ge 4 ] || { log "ABORT — _dl reached only $(count _dl)/4; the previous queue failed closed. A run needs a human. Nothing launched."; exit 1; }

if [ "$(count _hd29)" -ge 4 ]; then log "STEP 1 already complete (4/4 _hd29) — skipping"; else
	log "STEP 1 — _hd re-fly on ABI 29 (clean control)"
	ARM_SUFFIX="_hd29" ARM_LABEL="d0-derived-hover-abi29-refly" \
		ARM_EXTRA_ARGS="--teacher-hover derived" ARM_REQUIRE_FLAGS="--teacher-hover" \
		ARM_CTRL_SUFFIX="_hd" \
		ARM_MARKER_JSON=",\"refly_of\":\"_hd\",\"purpose\":\"era-clean control for the stage D / arm B / label-only pairs\"" \
		ARM_LOG="/private/tmp/hd_refly_abi29.log" \
		bash scripts/seed_arm_chain.sh
	[ "$(count _hd29)" -ge 4 ] || { log "ABORT — _hd29 chain exited with $(count _hd29)/4 markers. A run needs a human. Box left idle."; exit 1; }
fi

log "########## QUEUE COMPLETE — nothing else queued; box IDLE ##########"
