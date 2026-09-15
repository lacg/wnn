#!/usr/bin/env bash
# POST-REPAIR QUEUE (Luiz, 15/09/2026 14:3x EDT) — two arms, IN ORDER, after the marker-
# repair sequence has released the GPU. Idle-gated, marker-gated, fails closed, honours
# experiments/HOLD_CONTROLLER (through the ladder). Log /private/tmp/post_repair_queue.log.
#
#   STEP 1  STAGE D PIPELINE A/B — grid -> GA-NEURONS -> MEMORY (arm, `_pipeN`) vs the banked
#           grid -> GA-CONNECTIONS -> MEMORY controls (`_hd`, derived hover). Same 5-gen /
#           patience-3 budget for the GA stage either way (the ladder's --neurons-gens equals its
#           --conns-gens). The ladder's caps would make the NEURONS GA shrink-only (grid starts AT
#           --max-output-neurons 256, and 1.8M populated cells sit over the 180k --max-cells
#           budget => growth suppressed), so the arm opens both: --max-output-neurons 512 (up to
#           128 levels/motor) and --max-cells off. Both are INERT for the control's pipeline
#           (CONNECTIONS never grows structure), so the pair differs by the GA dimension only —
#           plus the wheel era, as every pair against _hd does. Lever line: "stage D pipeline
#           A/B: does the neurons GA buy more than the connections GA after the same grid?"
#           Read: HEADLINE paired delta + CI, four columns; also report the winner's
#           levels/motor and its TRUE-key count (deployability).
#   STEP 2  ARM B AT THE CONTROL'S FOOTPRINT — --dagger-label-delta WITHOUT --obs-pwm (`_dl`),
#           vs `_hd`. Arm B's winners carry 1.8-2.1M TRUE keys (6-7x the controls' 250-390k,
#           2.6-4.0x the H743 flash): the pwm observation changes every step, so the student
#           visits ~6x more addresses. --max-cells is inert here (no structural stage), so the
#           only footprint lever inside the recipe is dropping the observation. The 13/09 smoke
#           showed +label-delta alone flies (90%/3.18 deg, tiny budget). Question: does the
#           true-delta label carry arm B's err gain (-0.39 deg, CI excl. 0) at a deployable
#           footprint? Secondary pair: _dl vs _bd (the pwm observation's own contribution).
#           obs_pwm-only (`_op`) would complete the 2x2 — NOT queued.
#   STEP 3  STOP — box idle, say so.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/post_repair_queue.log"
MARK="experiments/sweepladder_markers"
REPAIR_PID="${REPAIR_PID:-66162}"
log() { echo "[post-repair] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
count() { ls ${MARK}/SL_C_b24n256_cf21_brushless_L4C_g10_s3133700[2-5]$1.json 2>/dev/null | wc -l | tr -d ' '; }
repair_running() { kill -0 "$REPAIR_PID" 2>/dev/null || pgrep -f "recalc_headline[s].py|rescore_first_report_see[d].py" >/dev/null; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null; }

log "########## ARMED — wait for the repair sequence (pid ${REPAIR_PID}) -> stage D pipeline A/B (_pipeN x4) -> arm B label-only (_dl x4) -> STOP ##########"
beat=0
while repair_running || busy; do
	sleep 60; beat=$((beat + 1))
	[ $((beat % 30)) = 0 ] && log "waiting — repair sequence still running ($(grep -ac '→' logs/controller/recalc_headlines_connections.log 2>/dev/null)/75 on the CONNECTIONS pass)"
done
log "repair sequence exited; box idle"

if [ "$(count _pipeN)" -ge 4 ]; then log "STEP 1 already complete (4/4 _pipeN) — skipping"; else
	log "STEP 1 — stage D pipeline A/B"
	ARM_SUFFIX="_pipeN" ARM_LABEL="pipeline-neurons" \
		ARM_EXTRA_ARGS="--skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000" \
		ARM_MARKER_JSON=",\"pipeline\":\"grid-neurons-memory\",\"max_output_neurons\":512,\"max_cells\":\"off\"" \
		ARM_LOG="/private/tmp/stage_d_pipeline.log" \
		bash scripts/seed_arm_chain.sh
	[ "$(count _pipeN)" -ge 4 ] || { log "ABORT — stage D chain exited with $(count _pipeN)/4 markers. A run needs a human. Box left idle."; exit 1; }
fi

if [ "$(count _dl)" -ge 4 ]; then log "STEP 2 already complete (4/4 _dl) — skipping"; else
	log "STEP 2 — arm B at the control's footprint (label-only)"
	[ -f experiments/labelscale_markers/LABEL_REBASE_LANDED.json ] || { log "ABORT — label re-base sentinel missing; the delta label would train on 'max descend'."; exit 1; }
	ARM_SUFFIX="_dl" ARM_LABEL="delta-label-only" \
		ARM_EXTRA_ARGS="--dagger-label-delta" ARM_REQUIRE_FLAGS="--dagger-label-delta" \
		ARM_MARKER_JSON=",\"dagger_label_delta\":true,\"obs_pwm\":false,\"flag_bundle\":1,\"secondary_control_suffix\":\"_bd\"" \
		ARM_LOG="/private/tmp/arm_b_label_only.log" \
		bash scripts/seed_arm_chain.sh
	[ "$(count _dl)" -ge 4 ] || { log "ABORT — label-only chain exited with $(count _dl)/4 markers. A run needs a human. Box left idle."; exit 1; }
	log "secondary pair (_dl vs _bd = the pwm observation's own contribution):"
	PYTHONPATH=src/wnn /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python scripts/paired_power.py \
		--arm _dl --base "SL_C_b24n256_cf21_brushless_L4C_g10_s{seed}" \
		--seed 31337002 --seed 31337003 --seed 31337004 --seed 31337005 --control-suffix _bd 2>&1 | tee -a "$LOG"
fi

log "########## QUEUE COMPLETE — nothing else queued; box IDLE ##########"
