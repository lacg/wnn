#!/usr/bin/env bash
# POST-WINDOW-K QUEUE (13/09/2026 15:5x EDT, Luiz). Window-k stopped at s2_win4: framed1 at
# k=2 gives each neuron ONE frame with recency weights 2:1, so k=2 took ~85 of the anchor's
# 256 current-frame neurons and spent them on the previous frame (win2 79.8%/3.96° vs the
# anchor 99.6%/1.41°; win3 16.0%). The NO-STEAL arm keeps the anchor's 256 on the current
# frame and ADDS the previous frame: k=2 framed1 at n=384 = exactly 256 + 128 (deterministic
# quota). Its twin, k=1 at n=384, separates "128 extra neurons" from "the timeline".
#
#   STEP 1  s2 n384 k=2 framed1   (~5 h)   pairs: _crn 256n k=1 anchor, s2_win2 256n k=2
#   STEP 2  s2 n384 k=1           (~5 h)   the neuron-count twin
#   STEP 3  arm B re-fly x4 on the obs_pwm-fixed wheel (~20 h) — scripts/armb_refly_queue.sh
#   STEP 4  STOP — nothing else is queued; the box goes IDLE. Say so.
# Marker-gated, idempotent, never preempts, fails closed; honours the HOLD sentinel.
cd /Users/lacg/wnn || exit 1
LOG="/private/tmp/post_wink_queue.log"
SMARK="experiments/sweepladder_markers"
LADDER="scripts/sweep_ladder_gamma.sh"
log() { echo "[post-wink] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" 2>/dev/null || true; }
ladder_pids() { pgrep -f "scripts/sweep_ladder_gamm[a]\.sh" 2>/dev/null || true; }
wait_box_clear() { local b=0; while [ -n "$(controller_pids)" ] || [ -n "$(ladder_pids)" ]; do sleep 60; b=$((b+1)); [ $((b%30)) = 0 ] && log "waiting — box busy"; done; sleep 120; }
. scripts/controller_arm_lib.sh

run_arm() { # $1 neurons  $2 suffix  $3 extra args  $4 marker json  $5 window_k
	local n="$1" suf="$2" extra="$3" mj="$4" wk="$5" seed=31337002
	local tag="SL_C_b24n${n}_cf21_brushless_L4C_g10_s${seed}${suf}"
	if [ -f "${SMARK}/${tag}.json" ]; then log "SKIP ${tag} (marker exists)."; return 0; fi
	wait_box_clear
	wait_while_held log "$tag"
	log "===== START ${tag} (n=${n}, k=${wk}: ${extra:-plain}) ====="
	SL_SKIP_PHASE1=1 SL_FORCE_PHASE2_GAMMA="1.0" SL_WIDTHS="24" SL_NEURONS="$n" SL_SEED="$seed" \
		SL_SWEEP_LABEL="nosteal-window" SL_TAG_SUFFIX="$suf" SL_WINDOW_K="$wk" \
		SL_EXTRA_ARGS="$extra --teacher-hover derived" SL_EXTRA_MARKER_JSON="$mj" bash "$LADDER"
	log "ladder exited rc=$? for ${tag}"
	while [ -n "$(controller_pids)" ]; do sleep 30; done
	[ -f "${SMARK}/${tag}.json" ] || { log "ABORT — marker ${tag}.json MISSING. A run needs a human. Box left idle."; exit 1; }
	log "banked: ${tag}"
}

log "########## ARMED — no-steal window (n384 k=2, n384 k=1) -> arm B re-fly x4 -> STOP ##########"
b=0
while [ ! -f "${SMARK}/SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_win4.json" ]; do
	sleep 120; b=$((b+1)); [ $((b%30)) = 0 ] && log "waiting — s2_win4 not banked yet"
done
log "s2_win4 banked — STEP 1"
run_arm 384 "_win2" "--conn-policy framed1 --output-full-window --input-window-k 2 --frame-stride 10 --conn-mutation-scope window" \
	",\"arm\":\"nosteal_win2\",\"conn_policy\":\"framed1\",\"output_full_window\":true,\"frame_stride\":10,\"conn_mutation_scope\":\"window\",\"recency_weights\":\"2^slot\",\"frame_quota\":\"256+128\",\"control_tag\":\"SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_crn\"" 2
log "STEP 2"
run_arm 384 "" "" ",\"arm\":\"nosteal_twin_k1\",\"frame_quota\":\"384\",\"control_tag\":\"SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_crn\"" 1
log "STEP 3 — arm B re-fly (armb_refly_queue.sh waits on win4 + idle, both already true)"
bash scripts/armb_refly_queue.sh
log "########## QUEUE COMPLETE — nothing else queued; box IDLE ##########"
