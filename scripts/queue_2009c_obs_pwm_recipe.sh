#!/bin/bash
# queue_2009c — successor to queue_2009b from STEP 6 on, with --obs-pwm ADOPTED into the recipe.
#
# Why (Luiz 26/09/2026: "if it is better, yeah, let's adopt it"): `_op30` x4 vs `_pipeN30` on the
# promotion gate's own bar (HEADLINE rows, n=4 mean paired delta): stable +1.35 pp, err −0.295° (4/4),
# steady −0.17°, alt +0.010 m → 3/4 columns favourable, err not worse → the rule FIRES. Cost: mean
# TRUE keys 1.62M vs 0.81M (all QSPI tier, which CTRL-18 already allows).
# So ANCHOR = `_op30`, RECIPE = PIPE_FLAGS + --obs-pwm.
#
# queue_2009b's supervisor was stopped (plain TERM, queue only) so it would not launch STEP 6 without
# --obs-pwm; its `_full30` chain keeps running and banks s3-s5 by itself.
#
#   STEP 5  wait for the `_full30` chain; apply the pre-registered CTRL-16 rule (scripts/ctrl16_verdict.py).
#           If `_full30` REPLACES `_pipeN30`, STOP: full-pipeline + obs-pwm is an untested combination.
#   STEP 5b CTRL-21 CRN noise probe in the controller gap (~47 min, score-only; IDS never paused for it).
#   STEP 5c PARK until the ABI-31 stage-2 trainer fixes land (option B, Luiz 26/09): experiments/ABI31_LANDED.
#   STEP 6a anchor re-fly `_op31` x5 (s2-s6) on the new trainer; s2-s5 paired vs `_op30` = trainer read.
#   STEP 6b tilt-coherence arm `_tc31` x4 (5°->5° training curriculum) paired vs `_op31` (rule: experiments/ctrl_tilt_rule.json).
#   STEP 6c PARK for the tilt decision (experiments/tilt_decision = coherent|wide, after a 15°/30° stress re-score).
#   STEP 6  LEVELS n=5 (CTRL-18) on the chosen curriculum: `_Ltc31` vs `_tc31` or `_Lop31` vs `_op31`, n384/n512 x5.
#   STEP 7  STOP. Next, in this order: CTRL-17 audit → CTRL-8 stage 2 (horizontal) → multi-axis round 1.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/queue_2009.log"
MARK="experiments/sweepladder_markers"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
BASE="SL_C_b24n256_cf21_brushless_L4C_g10"
PIPE_FLAGS="--skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
RECIPE="--obs-pwm ${PIPE_FLAGS}"
RECIPE_NAME="grid-neurons-memory+obs-pwm"
ANCHOR="_op30"
REQ="--obs-pwm --skip-stages --max-output-neurons"
SEEDS4="31337002 31337003 31337004 31337005"
SEEDS5="$SEEDS4 31337006"
log() { echo "[q2009c] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
have() { [ -f "${MARK}/${BASE}_s$1$2.json" ]; }
count() { ls ${MARK}/${BASE}_s3133700[2-5]$1.json 2>/dev/null | wc -l | tr -d ' '; }
countn() { ls ${MARK}/SL_C_b24n$1_cf21_brushless_L4C_g10_s3133700[2-6]$2.json 2>/dev/null | wc -l | tr -d ' '; }
busy() { pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null || pgrep -f "scripts/sweep_ladder_gamm[a].sh" >/dev/null || pgrep -f "scripts/seed_arm_chai[n].sh" >/dev/null; }
arm() {  # arm <suffix> <label> <seeds> <extra_args> <require_flags> <ctrl_suffix> <marker_json> [neurons] [no_control]
	ARM_SUFFIX="$1" ARM_LABEL="$2" ARM_SEEDS="$3" ARM_EXTRA_ARGS="$4" ARM_REQUIRE_FLAGS="$5" ARM_CTRL_SUFFIX="$6" \
		ARM_MARKER_JSON="$7" ARM_NEURONS="${8:-256}" ARM_NO_CONTROL="${9:-0}" \
		ARM_LOG="/private/tmp/seed_arm$1${8:+_n$8}.log" bash scripts/seed_arm_chain.sh
}
RJ=",\"recipe\":\"${RECIPE_NAME}\",\"anchor\":\"${ANCHOR}\",\"obs_pwm\":true,\"adopted\":\"CTRL-15 gate bar 26/09\""

log "########## ARMED (2009c) — wait _full30 x4 -> CTRL-16 -> CTRL-21 probe -> PARK for ABI-31 landing -> _op31 x5 -> _tc31 x4 -> PARK tilt decision -> LEVELS -> STOP ##########"

# ---- STEP 5: wait for the _full30 chain (launched by 2009b) and apply the CTRL-16 rule ------------
beat=0
while busy; do sleep 60; beat=$((beat + 1)); [ $((beat % 30)) = 0 ] && log "waiting — _full30 chain flying ($(count _full30)/4)"; done
[ -f experiments/HOLD_CONTROLLER ] && { log "HOLD present after the _full30 chain — rm it and relaunch 2009c."; exit 1; }
[ "$(count _full30)" -ge 4 ] || { log "ABORT — _full30 $(count _full30)/4 after the chain exited. A run needs a human."; exit 1; }
log "---------- CTRL-16 verdict (pre-registered, experiments/ctrl16_rule.json) ----------"
PYTHONPATH=src/wnn $VP scripts/ctrl16_verdict.py 2>&1 | tee -a "$LOG"
if $VP -c "import json,sys; sys.exit(0 if json.load(open('experiments/ctrl16_verdict.json'))['replace_anchor'] else 1)"; then
	log "STOP — CTRL-16 says _full30 REPLACES _pipeN30. Full pipeline + --obs-pwm is untested; the recipe needs Luiz. Box left idle."
	exit 0
fi

# ---- STEP 5b: CTRL-21 CRN noise probe in the controller gap (Luiz 26/09: waits for the controller
# side; IDS is never paused or stopped for it). Score-only, ~47 min, peak ~10-12 GB. A failed probe
# is logged and does NOT block LEVELS — it only feeds CTRL-20's floors.
PROBE_TAG="SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op30"
if [ -f "experiments/crn_noise_probe/PROBE_${PROBE_TAG}.json" ] && ! grep -q '"smoke": true' "experiments/crn_noise_probe/PROBE_${PROBE_TAG}.json"; then
	log "STEP 5b already done (CTRL-21 probe JSON present)"
else
	log "STEP 5b — CTRL-21: CRN noise probe on ${PROBE_TAG} MEMORY population (--frames 3); out /private/tmp/ctrl21_probe.out"
	PYTHONPATH=src/wnn /usr/bin/time -l $VP -u scripts/crn_noise_probe.py \
		--ckpt "logs/controller/sweep_ladder/ckpt/${PROBE_TAG}/stage4_memory.yaml.gz" \
		--run-args-file "experiments/crn_noise_probe/ARGS_${PROBE_TAG}.txt" \
		--run-out "logs/controller/sweep_ladder/${PROBE_TAG}.out" \
		--frames 3 > /private/tmp/ctrl21_probe.out 2>&1
	log "STEP 5b — CTRL-21 probe rc=$? (see /private/tmp/ctrl21_probe.out; JSON in experiments/crn_noise_probe/)"
fi

# ---- STEP 5c: PARK for the ABI-31 landing (option B, Luiz 26/09) ---------------------------------
# CTRL-17 stage-2 trainer fixes (G1-G4 incl. the ALTITUDE side of G3) change how every run trains, so
# they land HERE — after _full30 (CTRL-16 stays single-trainer) and before the anchor re-fly. A human
# lands (merge + controller wheel ABI 31 + python in one step), smokes ONE run, then:
#   touch experiments/ABI31_LANDED
LANDED="experiments/ABI31_LANDED"
beat=0
while [ ! -f "$LANDED" ]; do
	[ $((beat % 30)) = 0 ] && log "PARKED — waiting for the ABI-31 stage-2 trainer landing (touch ${LANDED} after the smoke)"
	sleep 60; beat=$((beat + 1))
done
ABI=$($VP -c "import ram_controller as c; print(c.ABI_VERSION)" 2>/dev/null)
[ "$ABI" = "31" ] || { log "ABORT — ${LANDED} present but ram_controller ABI=${ABI:-?}, expected 31. Box left idle."; exit 1; }
log "STEP 5c — ABI-31 landing confirmed (ram_controller ABI ${ABI})"

# ---- STEP 6a: anchor re-fly on the ABI-31 trainer: _op31 x5 ----------------------------------------
# s2-s5 pair vs _op30 (same seeds) = the old->new TRAINER read (descriptive); s6 has no ABI-30 twin.
ANCHOR="_op31"
RJ=",\"recipe\":\"${RECIPE_NAME}\",\"anchor\":\"${ANCHOR}\",\"obs_pwm\":true,\"trainer\":\"ABI31 stage-2 fixes (CTRL-17 G1-G4, altitude side incl.)\""
if [ "$(count _op31)" -ge 4 ]; then log "STEP 6a s2-s5 already banked (_op31 4/4)"; else
	log "STEP 6a — anchor re-fly: _op31 s2-s5 on the ABI-31 trainer, paired vs _op30 (trainer read)"
	arm "_op31" "anchor-obs-pwm-abi31-trainer" "$SEEDS4" "$RECIPE" "$REQ" "_op30" \
		",\"refly_of\":\"_op30\"${RJ}"
	[ "$(count _op31)" -ge 4 ] || { log "ABORT — _op31 $(count _op31)/4. A run needs a human. Box left idle."; exit 1; }
fi
if have 31337006 "$ANCHOR"; then log "STEP 6a s6 already banked"; else
	log "STEP 6a — _op31 s31337006 (5th seed of the 64-level rung, CTRL-18)"
	arm "_op31" "anchor-obs-pwm-abi31-trainer" "31337006" "$RECIPE" "$REQ" "_op30" \
		",\"purpose\":\"5th seed of the 64-level rung (CTRL-18)\"${RJ}" 256 1
	have 31337006 "$ANCHOR" || { log "ABORT — ${ANCHOR} s31337006 missing. A run needs a human. Box left idle."; exit 1; }
fi

# ---- STEP 6b: tilt-coherence arm _tc31 x4 (Luiz 27/09), paired vs _op31 ---------------------------
# DAgger training curriculum 5°->5° (= scorer --tilt 5) instead of the implicit 8°->30°. Rule
# pre-registered in experiments/ctrl_tilt_rule.json (gate bar vs _op31 + a 15°/30° stress re-score).
TILT_FLAGS="--rg-easy-tilt-deg 5 --rg-full-tilt-deg 5"
TILT_REQ="--rg-easy-tilt-deg --rg-full-tilt-deg"
if [ "$(count _tc31)" -ge 4 ]; then log "STEP 6b already banked (_tc31 4/4)"; else
	log "STEP 6b — tilt-coherence arm: _tc31 s2-s5 (${TILT_FLAGS}) paired vs _op31"
	arm "_tc31" "tilt-coherent-5-5-abi31" "$SEEDS4" "${RECIPE} ${TILT_FLAGS}" "${REQ} ${TILT_REQ}" "_op31" \
		",\"tilt_curriculum\":\"5->5 (coherent with scorer)\",\"control_curriculum\":\"8->30\",\"rule\":\"experiments/ctrl_tilt_rule.json\"${RJ}"
	[ "$(count _tc31)" -ge 4 ] || { log "ABORT — _tc31 $(count _tc31)/4. A run needs a human. Box left idle."; exit 1; }
fi

# ---- STEP 6c: PARK for the tilt decision (stress re-score at 15°/30° + rule, applied by hand) ------
DECISION="experiments/tilt_decision"
beat=0
while [ ! -f "$DECISION" ]; do
	[ $((beat % 30)) = 0 ] && log "PARKED — waiting for the tilt decision (apply experiments/ctrl_tilt_rule.json incl. the 15°/30° stress re-score; write 'coherent' or 'wide' to ${DECISION})"
	sleep 60; beat=$((beat + 1))
done
CHOICE=$(tr -d '[:space:]' < "$DECISION")
case "$CHOICE" in
	coherent) ANCHOR="_tc31"; LSUF="_Ltc31"; LEV_TILT="$TILT_FLAGS"; LEV_REQ="$TILT_REQ" ;;
	wide)     ANCHOR="_op31"; LSUF="_Lop31"; LEV_TILT=""; LEV_REQ="" ;;
	*) log "ABORT — ${DECISION} says '${CHOICE}', expected coherent|wide. Box left idle."; exit 1 ;;
esac
log "STEP 6c — tilt decision: ${CHOICE} -> LEVELS anchor ${ANCHOR}, suffix ${LSUF}"
if [ "$CHOICE" = "coherent" ] && ! have 31337006 _tc31; then
	log "STEP 6c — _tc31 s31337006 (5th seed of the 64-level rung on the coherent curriculum)"
	arm "_tc31" "tilt-coherent-5-5-abi31" "31337006" "${RECIPE} ${TILT_FLAGS}" "${REQ} ${TILT_REQ}" "_op31" \
		",\"purpose\":\"5th seed of the 64-level rung (CTRL-18)\",\"tilt_curriculum\":\"5->5\"${RJ}" 256 1
	have 31337006 _tc31 || { log "ABORT — _tc31 s31337006 missing. A run needs a human. Box left idle."; exit 1; }
fi

# ---- STEP 6: LEVELS n=5 on the chosen recipe, ABI-31 trainer, paired vs ANCHOR --------------------
for N in 384 512; do
	LV=$((N / 4))
	if [ "$(countn $N $LSUF)" -ge 5 ]; then log "STEP 6 ${LV} levels already complete (5/5)"; continue; fi
	log "STEP 6 — CTRL-18: ${LV} levels (n${N}) x5 (s2-s6) on ${RECIPE_NAME} (${CHOICE} tilt), ABI-31 trainer; off-chip tier allowed"
	# --max-output-neurons must follow the rung, not the 512 cap: last-wins after RECIPE.
	arm "$LSUF" "levels-${LV}-obs-pwm-${CHOICE}-abi31" "$SEEDS5" "--teacher-hover derived ${RECIPE} ${LEV_TILT} --max-output-neurons ${N}" \
		"--teacher-hover --motor-lag-s ${REQ} ${LEV_REQ}" "$ANCHOR" \
		",\"levels_study\":true,\"levels\":${LV},\"control\":\"${ANCHOR} n256 same seed\",\"tilt\":\"${CHOICE}\",\"deploy_tier\":\"off-chip allowed\"${RJ}" "$N" 1
	[ "$(countn $N $LSUF)" -ge 5 ] || { log "ABORT — ${LV} levels $(countn $N $LSUF)/5. A run needs a human. Box left idle."; exit 1; }
	log "---------- CTRL-18 read: ${LV} levels − 64 levels (${ANCHOR}), same seeds, n=5 ----------"
	PYTHONPATH=src/wnn $VP scripts/paired_power.py --arm "$LSUF" --base "SL_C_b24n${N}_cf21_brushless_L4C_g10_s{seed}" \
		$(for sd in $SEEDS5; do printf -- "--seed %s --control-override %s=${BASE}_s%s${ANCHOR} " "$sd" "$sd" "$sd"; done) 2>&1 | tee -a "$LOG"
done

log "########## QUEUE COMPLETE — anchor ${ANCHOR} (${RECIPE_NAME}); box IDLE. ##########"
log "NEXT (Luiz 26/09): CTRL-17 audit -> CTRL-8 stage 2 (horizontal) -> multi-axis round 1 with MA_ANCHOR_SUFFIX=${ANCHOR} + recipe flags (${RECIPE})."
