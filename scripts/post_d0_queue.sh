#!/usr/bin/env bash
# THE POST-D0 QUEUE (11/09/2026, Priority 0 landed). Replaces post_arma_queue.sh.
#   STEP 1  D0 A/B: --teacher-hover derived vs the banked legacy anchors, 4 runs _hd  (~20 h)
#   STEP 2  2x2 leak x label-scale, FORCED on s=2, derived hover, 4 runs _l090_ls2    (~20 h)
#   STEP 3  ARM B true-delta label, derived hover, controls = _hd, 4 runs _bd         (~20 h)
#           (gated on the label re-base sentinel, created by the D0 wheel deploy)
#   STEP 4  window-k FRAMED runs 2..12, derived hover                                  (~50 h)
#   STEP 5  STOP — the multi-axis programme is a written spec; its chains are not.
# Every step: idle box, marker-gated, idempotent, fails closed. Nothing preempts a run.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="${PQ_LOG:-/private/tmp/post_d0_queue.log}"
MARK="experiments/sweepladder_markers"
log() { echo "[post-d0] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
controller_pids() { pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" 2>/dev/null || true; }
chain_pids() { pgrep -f "scripts/(sweep_ladder_gamma|leak_x_labelscale_chain|arm_b_delta_label_chain|queue_after_ab_chain|d0_hover_ab_chain)\.sh" 2>/dev/null || true; }
busy() { [ -n "$(controller_pids)" ] || [ -n "$(chain_pids)" ]; }
wait_box_clear() { local b=0; while busy; do sleep 60; b=$((b+1)); [ $((b%30)) = 0 ] && log "waiting — box busy"; done; sleep 120; }
count_markers() { ls "$MARK" 2>/dev/null | grep -cE "$1"; }
run_step() {
	local name="$1" regex="$2" want="$3"; shift 3
	local have; have="$(count_markers "$regex")"
	[ "$have" -ge "$want" ] && { log "SKIP ${name} — complete (${have}/${want})."; return 0; }
	wait_box_clear
	log "===== STEP ${name}: starting (${have}/${want} markers) ====="
	"$@"; local rc=$?
	log "${name} chain exited rc=${rc}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	have="$(count_markers "$regex")"
	[ "$have" -ge "$want" ] || { log "ABORT — ${name} finished with ${have}/${want} markers. A run needs a human. Box left IDLE."; exit 1; }
	log "===== STEP ${name}: COMPLETE (${have}/${want}) ====="
}
log "########## POST-D0 QUEUE ARMED — A/B(derived hover) -> 2x2(s=2) -> arm B -> window-k -> stop ##########"
run_step "1 d0-hover-ab(derived vs legacy)" '_hd\.json$' 4 bash scripts/d0_hover_ab_chain.sh
run_step "2 2x2-leak-x-labelscale(s=2 FORCED, derived)" '_l090_ls2\.json$' 4 env LS_STAR=2 bash scripts/leak_x_labelscale_chain.sh
ARMB_GATE="experiments/labelscale_markers/LABEL_REBASE_LANDED.json"
if [ -f "$ARMB_GATE" ]; then
	run_step "3 arm-B-true-delta-label(derived, ctrl=_hd)" '_bd\.json$' 4 bash scripts/arm_b_delta_label_chain.sh
else
	log "HOLD — STEP 3 arm B SKIPPED: ${ARMB_GATE} absent (label re-base not deployed). Continuing."
fi
run_step "4 window-k-FRAMED runs 2..12 (derived)" '_win[234]\.json$' 12 bash scripts/queue_after_ab_chain.sh
log "########## QUEUE DRAINED ##########"
log "STEP 5: the multi-axis programme (docs/multi_axis_programme_spec.md) has a spec but no chains. Box IDLE."
