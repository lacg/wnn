#!/usr/bin/env bash
# PROBE HANDOFF SUPERVISOR (29/08/2026, Luiz approved: "probe first, preempt seed-2"
# + "smoke b=64 first").
#
# Waits for the ladder A/B chain's LAST round-1 marker (b=36 desir, seed 31337002),
# then hands the box over in this order, FAILING CLOSED at every gate:
#
#   1. stop the ladder chain (scripts/sweep_ladder_ab_chain.sh) so it cannot start
#      the seed-31337003 round. The chain is idempotent per marker, so relaunching
#      it later skips all 26 round-1 markers and resumes at the cull.
#   2. SMOKE b=64 on a tiny budget. b=64 is the SUBSTRATE CEILING: controller cells
#      are keyed by u64 (compute_address_sparse -> u64), so 1<<bits stops fitting
#      above 64 and 64 itself is the exact boundary. If the smoke does not exit 0,
#      the probe is NOT launched and this script stops. Fails closed by design.
#   3. launch the WIDE PROBE (b=40,48,64, both arms, seed 31337002) detached.
#   4. when the probe's 6 markers exist, relaunch the ladder chain (cull + seed-2).
#
# ONE controller at a time is preserved throughout: each step waits for the
# previous process to be gone before starting the next.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
LOG="/private/tmp/probe_handoff.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
MARKDIR="experiments/sweepladder_markers"
GATE_MARKER="${MARKDIR}/SL_A_b36n32_cf21_brushless_L4C_desir_s31337002.json"
SMOKE_OUT="/private/tmp/b64_smoke.out"
PROBE_MARKERS="40 48 64"

log() { echo "[handoff] $(date -u +%FT%TZ) $*" >> "$LOG"; }

wait_no_controller() {
	local n
	while :; do
		n=$(pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" | wc -l | tr -d ' ')
		[ "$n" = "0" ] && return 0
		sleep 10
	done
}

log "########## ARMED — waiting for $GATE_MARKER ##########"
while [ ! -f "$GATE_MARKER" ]; do sleep 20; done
log "round-1 COMPLETE: b=36 desir marker landed."

# ---- 1. stop the ladder chain (and whatever seed-2 run it may have just started)
CHAIN_PIDS=$(pgrep -f "scripts/sweep_ladder_ab_chain.sh" || true)
if [ -n "$CHAIN_PIDS" ]; then
	log "stopping ladder chain pids: $CHAIN_PIDS (seed-2 preempted; markers make it resumable)"
	# shellcheck disable=SC2086
	kill $CHAIN_PIDS 2>/dev/null || true
	sleep 3
fi
CHILD=$(pgrep -f "MacOS/Python -u -m wnn.control.phased_ga" || true)
if [ -n "$CHILD" ]; then
	log "killing in-flight seed-2 phased_ga child: $CHILD (no marker written; it re-runs later)"
	# shellcheck disable=SC2086
	kill $CHILD 2>/dev/null || true
fi
wait_no_controller
log "box is clear of controllers."

# ---- 2. SMOKE b=64 (tiny budget). Folds stay 5 — never 1, never 3.
log "SMOKE b=64 starting -> $SMOKE_OUT"
"$VP" -u -m wnn.control.phased_ga \
	--levels 16 --skip-stages neurons,bits \
	--grid-bits 64 --grid-output-neurons 32 --max-output-neurons 32 \
	--grid-state-neurons 0 --max-state-neurons 0 \
	--conns-gens 1 --conns-patience 3 \
	--memory-gens 1 --memory-patience 2 \
	--pop 8 --num-eval-folds 5 --check-interval 2 \
	--eval-episodes 5 --memory-eval-episodes 5 --report-episodes 5 \
	--steps 200 --tilt 5.0 \
	--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375 \
	--fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 \
	--runs 1 --memory-mode BINARY \
	--airframe cf21_brushless --disturbance L4C --teacher mpcof \
	--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz \
	--translation --reward-lambda-alt 0 \
	--report-seeds 99990101 --base-seed 31337002 \
	> "$SMOKE_OUT" 2>&1
SMOKE_RC=$?
if [ "$SMOKE_RC" != "0" ]; then
	log "SMOKE b=64 FAILED rc=$SMOKE_RC — PROBE NOT LAUNCHED. See $SMOKE_OUT (tail below)"
	tail -25 "$SMOKE_OUT" >> "$LOG"
	log "########## STOPPED (fail-closed). Box is idle — ladder chain NOT relaunched. ##########"
	exit 1
fi
log "SMOKE b=64 rc=0 — u64 boundary holds."
wait_no_controller

# ---- 3. launch the WIDE PROBE
log "launching WIDE PROBE (b=40,48,64 x gate|desir, seed 31337002)"
setsid_missing=1
nohup bash scripts/sweep_ladder_probe_wide.sh >/dev/null 2>&1 &
PROBE_PID=$!
log "probe pid=$PROBE_PID (log /private/tmp/sweep_ladder_probe_wide.log)"

# ---- 4. wait for all 6 probe markers, then relaunch the ladder chain
while :; do
	missing=0
	for b in $PROBE_MARKERS; do
		[ -f "${MARKDIR}/SL_A_b${b}n32_cf21_brushless_L4C_s31337002.json" ] || missing=1
		[ -f "${MARKDIR}/SL_A_b${b}n32_cf21_brushless_L4C_desir_s31337002.json" ] || missing=1
	done
	[ "$missing" = "0" ] && break
	kill -0 "$PROBE_PID" 2>/dev/null || { log "probe process gone with markers missing — NOT relaunching the ladder. Inspect first."; exit 1; }
	sleep 60
done
log "WIDE PROBE COMPLETE (6/6 markers)."
wait_no_controller
log "relaunching ladder chain (skips 26 round-1 markers, resumes at cull + seed-2)"
nohup bash scripts/sweep_ladder_ab_chain.sh >/dev/null 2>&1 &
log "ladder chain relaunched pid=$! ########## HANDOFF COMPLETE ##########"
