#!/usr/bin/env bash
# CALIBRATION-PLANT A/B (13/08/2026, task #11) — should the thermometer ladder be
# fit on the AIRFRAME we fly, or on the historical synthetic plant?
#
# THE FINDING THAT MOTIVATES IT. fit_thresholds_from_pid_rollouts builds a bare
# AttitudeSim() — the synthetic default plant (k_thrust 2.4, inertia 0.0023),
# driven by the legacy PID — while every airframe run flies cf21_brushless under
# the firmware cascade. Measured 13/08 on the production regime (tilt 5°, L4C):
#
#   * 2–12% of FLOWN samples fall outside the fitted ladder (gyro_x worst at
#     11.6%) — comparable to the ACCEPTED cal-30° setting (11.3%), so this is not
#     a broken regime, just a mismatched one.
#   * 6.2% of thermometer bits change on average (gyro_x 13.6%). At 30 bits per
#     address that is 0.938^30 ≈ 0.15, i.e. ~85% of ADDRESSES move.
#   * The clearest waste: gyro_x's ladder spans −0.069..+0.013 while the flown
#     p99 is +0.143 — the whole positive half collapses into one code.
#
# It is the third member of the "calibrate on the regime you fly" family: the
# tilt regime was fixed 09/08, the disturbance 10/08, the PLANT is still open.
# Translation/stage-1 runs already use the airframe (fixed 13/08); this A/B is
# ONLY about the attitude-only lineage, which is why both arms fly attitude-only.
#
# WHY AN A/B AND NOT JUST THE FIX. "More correct" is not "better numbers": the GA
# may already have adapted its connectivity to the ladder it was handed. And
# adopting it is a LINEAGE BREAK (~85% address churn) that invalidates every
# banked memory, so it has to be earned, dated and disclosed.
#
# BAR (pre-registered): the airframe ladder must beat the synthetic one on
# HELD-OUT headline STEADY on BOTH seeds to be adopted.
#   both  -> ADOPT, as a dated lineage break with the disclosure attached.
#   split -> underpowered at n=2 (the sn>0/L1 shape); report as such, adopt
#            NOTHING, and escalate to n>=4 if the effect looks worth it.
#   none  -> the synthetic ladder is not costing us anything measurable; close
#            the question and record that the quantile fit self-corrects.
#
# The recipe is byte-identical to the banked L1-refly arms (S16 weights, mpcof,
# cf21_brushless, L4C, pidmix 15 features) so these runs are directly comparable
# to that lineage; --calib-airframe is the SINGLE variable.
#
# ~2h45m/run, 4 runs, one controller at a time.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/calib_ab_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/calibab"
MARKDIR="experiments/calibab_markers"
AIRFRAME="${CAB_AIRFRAME:-cf21_brushless}"
DIST="${CAB_DIST:-L4C}"
SEEDS="${CAB_SEEDS:-31337002 31337003}"
NEURONS_GENS="${CAB_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# Gate: wait for the lambda sweep (or whatever holds the box) to finish first.
WAIT_MARKERS="${CAB_WAIT_MARKERS:-$ROOT/experiments/stage1lambda_markers}"
WAIT_COUNT="${CAB_WAIT_COUNT:-10}"

FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[calibab] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — calibration-plant A/B, waiting for $WAIT_COUNT markers in $(basename "$WAIT_MARKERS") ##########"

# ---- gate on the preceding cohort -------------------------------------------
WAITED=0
while true; do
	N=$(ls "$WAIT_MARKERS" 2>/dev/null | wc -l | tr -d ' ')
	C=$(controllers)
	if [ "$N" -ge "$WAIT_COUNT" ] && [ "$C" -eq 0 ]; then
		log "gate open: markers=$N controllers=0 (waited ${WAITED}s)"
		break
	fi
	# 40 h ceiling: 10 sweep runs at ~2h45m is ~28 h, so this only fires if the
	# sweep died. Waiting forever would hide that.
	if [ "$WAITED" -ge 144000 ]; then
		log "ABORT: markers=$N controllers=$C after 40 h — the preceding cohort likely died. Not starting."
		exit 1
	fi
	sleep 300
	WAITED=$((WAITED + 300))
done

run_arm() {
	local arm="$1" seed="$2" flag="$3"
	local tag="CAB_${arm}_mpcof_${AIRFRAME}_${DIST}_s${seed}"
	# STAGE CHECKPOINTS (14/08/2026). Without this flag phased_ga falls back to a
	# SHARED /tmp/wnn-phased-ga-emergency dir whose filenames carry no tag, so each
	# run silently overwrites the previous run's dump — the lam0/s31337003 dump was
	# ~2.5 h from being destroyed by the next arm. It also left controller_arm_lib's
	# R1 "RESUME, don't restart" path dead: it globs $stagedir, which was empty, so
	# every watchdog kill silently re-earned hours from scratch.
	mkdir -p "$OUTDIR/ckpt/$tag"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"CALIB_AB\",\"calib\":\"${arm}\",\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"features\":\"pidmix\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
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
		$flag \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# INTERLEAVED: each seed flies BOTH arms before the next seed starts, so a crash
# at hour 6 still leaves one complete paired comparison.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (both calibration arms) ====="
	log "===== START synth seed=$seed ====="
	run_arm "synth" "$seed" "--no-calib-airframe"
	log "synth seed=$seed finished rc=$?"
	log "===== START afcal seed=$seed ====="
	run_arm "afcal" "$seed" "--calib-airframe"
	log "afcal seed=$seed finished rc=$?"
done

log "########## A/B COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
