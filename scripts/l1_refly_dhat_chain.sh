#!/usr/bin/env bash
# L1-REFLY (12/08/2026) — does the d-hat FEATURE refutation survive Fix A?
#
# WHY. L1 and L1b both concluded that handing the student the mpcof disturbance
# estimate as INPUT FEATURES does not help and often hurts (refuted 4/4). Both
# were flown on code where the d-hat observer was fed a FROZEN hover accumulator
# during every training replay (bptt_train_window / split_record never advance
# self.pwm) while deploy fed it the evolving one. So the network TRAINED on a
# feature stream that does not exist at deploy, then was SCORED on the real one:
# the refutation may be an artifact of that divergence, not a property of the
# feature. Fix A (wheel 2026.212.37) threads the recorded APPLIED pwm into the
# replay, so train-time d-hat now equals deploy-time d-hat.
#
# THE 2x2 IS UNCHANGED from l1b_s16_dhat_chain.sh — same S16 weights (steady
# carries the largest weight, so the search is actually asked to keep hold
# precision), same 5-gen NEURONS cap, same seeds, same everything EXCEPT the
# wheel underneath. That is the point: a single-variable comparison against the
# banked L1b cells.
#
#   BANKED (pre-fix, l1bs16_markers)      THIS CHAIN (post-fix, l1refly_markers)
#   s16plain / s16dhat x 2 seeds          s16plain / s16dhat x 2 seeds
#
# BAR (pre-registered): the plain-vs-dhat gap on HEADLINE steady, per seed.
#   dhat beats plain on BOTH seeds  -> L1 was an ARTIFACT; the feature works and
#                                      the banked refutation must be withdrawn.
#   dhat loses on both              -> L1 CONFIRMED on honest code; the feature
#                                      genuinely does not help, and the sn>0
#                                      chain header's motivation stands.
#   split                           -> underpowered at n=2, same shape as sn>0;
#                                      report as such, do NOT pick a side.
#
# The s16plain arm carries NO observer, so it is also the clean control the DOBF
# cells lacked (both DOBF arms carry --obs-dhat).
#
# ~2 h, 4 cells, one controller at a time.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/l1_refly_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/l1refly"
MARKDIR="experiments/l1refly_markers"
AIRFRAME="${L1B_AIRFRAME:-cf21_brushless}"
DIST="${L1B_DIST:-L4C}"
SEEDS="${L1B_SEEDS:-31337002 31337003}"
# MPC-family cap, copied from the control arm. NOT a free choice.
NEURONS_GENS="${L1B_NEURONS_GENS:-5}"
LGAIN="${L1B_LGAIN:-0.05}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# nf=15 pidmix (+3 dhat when the dhat arm adds it).
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

# S16 — the whole point of this chain. steady carries the LARGEST weight.
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[l1refly] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — L1-REFLY (post Fix A) S16 x dhat 2x2 airframe=$AIRFRAME dist=$DIST seeds=[$SEEDS] gens=$NEURONS_GENS ##########"

# Never run alongside another controller: wait rather than contend, since this
# chain may be armed while something else is still finishing.
WAITED=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${WAITED}s)"
	sleep 60
	WAITED=$((WAITED + 60))
	if [ "$WAITED" -ge 43200 ]; then
		log "ABORT: box still busy after 12 h — refusing to contend."
		exit 3
	fi
done
log "box clear: controllers=0"

# run_cell <seed> <dhat:0|1>
run_cell() {
	local seed="$1" use_dhat="$2"
	local variant dhat_flags dhat_json
	if [ "$use_dhat" = "1" ]; then
		variant="s16dhat"
		dhat_flags="--obs-dhat --dhat-l-gain $LGAIN"
		dhat_json="\"obs_dhat\":true,\"dhat_l_gain\":${LGAIN},\"features\":\"pidmix+dhat\""
	else
		variant="s16plain"
		dhat_flags=""
		dhat_json="\"obs_dhat\":false,\"features\":\"pidmix\""
	fi

	# shellcheck disable=SC2086
	run_controller_arm "L1R_${variant}_mpcof_${AIRFRAME}_${DIST}_s${seed}" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"L1REFLY\",\"fix_a\":true,\"weights\":\"S16\",\"variant\":\"${variant}\",\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",${dhat_json},\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--max-cells 180000 --max-cells-strict \
		--neurons-gens "$NEURONS_GENS" --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		$S16_WEIGHTS \
		--report-episodes 100 --holdout-pop-sample 8 \
		--grid-bits 24 30 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" \
		--disturbance "$DIST" \
		--teacher mpcof \
		$FEAT_PIDMIX \
		$dhat_flags \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# INTERLEAVED: each seed flies BOTH combos before the next seed starts.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (both S16 combos) ====="
	for dh in 0 1; do
		run_cell "$seed" "$dh"
		log "seed=$seed dhat=$dh finished rc=$?"
	done
done

log "########## L1-REFLY DONE — markers in $MARKDIR ##########"
