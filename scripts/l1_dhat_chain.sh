#!/usr/bin/env bash
# L1 — does giving the student the mpcof teacher's DISTURBANCE OBSERVER (d̂) break
# the hold floor?
#
# WHY THIS RUN. The L4 teacher screen (docs/l4_teacher_screen_results.md) found five
# teachers spanning 0.69-1.78 deg of classical quality all produce students at
# 1.29-1.40 deg. The per-step decomposition (D1/D2, scripts/transient_decomposition.py)
# then localised it: every student is TEACHER-GRADE in recovery (0.88-1.21x) and hits
# an ABSOLUTE hold-attitude floor (steady 0.57-0.87 deg) regardless of whether its
# teacher holds at 0.00 deg (mpcof) or 0.65 deg (mpc). The natural control is mpc,
# whose own steady sits AT the floor — and whose student matches it at ratio ~1.0. So
# the floor is a DISTURBANCE-OBSERVABILITY limit: the teachers that beat it all carry
# integral action or an explicit observer, and the student carries neither.
#
# L1 gives it the observer. --obs-dhat runs the mpcof law inside WnnController.step()
#   dhat += l_gain * ((gyro - last_gyro)/dt - b*u_applied - dhat)
# from the student's OWN throttle accumulator, adding 3 thermometer features. b comes
# from ram_controller.calibrate_control_gains on --airframe (never re-derived in
# Python). Spec: docs/hold_floor_levers_spec.md.
#
# THE CONTROL ARM IS ALREADY FLOWN — it is the screen's mpcof arm, so this chain is
# only 2 runs. Every flag below is COPIED from scripts/l4_teacher_chain.sh's mpcof
# invocation, including L4_NEURONS_GENS=5 (the MPC-family cap): a budget difference
# would confound the feature under test with search depth.
#
#   control (screen mpcof)  s31337002  err 1.21 / stable 100.0 / steady 0.64
#                           s31337003  err 1.58 / stable 100.0 / steady 0.95
#
# SUCCESS (pre-registered, from the spec): steady drops below ~0.35 deg (the lqi
# teacher's classical hold) on BOTH seeds, i.e. clears the 0.57-0.87 floor band by
# more than the seed spread, AND the cruise ratio vs teacher falls from 3.3-4.3x
# toward <=2x. err below the 1.29 deg band is the headline if it follows.
# REFUTATION: steady stays inside the floor band on both seeds => the floor is not
# input observability, and attention moves to L3 (delta_leak/delta_max, never
# searched) / L4 (magnitude-weighted DAGGER conflicts).
#
# READ steady, NOT err, AS THE PRIMARY. err is ~80% recovery term, and recovery is
# already teacher-grade — a feature that fixes the hold can only move err by the
# ~20% the steady window carries. Judging L1 on err would under-read a real win.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/l1_dhat_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/l1dhat"
MARKDIR="experiments/l1dhat_markers"
AIRFRAME="${L1_AIRFRAME:-cf21_brushless}"
DIST="${L1_DIST:-L4C}"
SEEDS="${L1_SEEDS:-31337002 31337003}"
# MPC-family cap, copied from the control arm. NOT a free choice.
NEURONS_GENS="${L1_NEURONS_GENS:-5}"
LGAIN="${L1_LGAIN:-0.05}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# nf=15 pidmix + 3 d̂ = 18 features.
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[l1dhat] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — L1 obs_dhat airframe=$AIRFRAME dist=$DIST seeds=[$SEEDS] gens=$NEURONS_GENS l_gain=$LGAIN ##########"

# Never run alongside another controller: abort rather than contend.
if [ "$(controllers)" -gt 0 ]; then
	log "ABORT: controllers=$(controllers) already running — refusing to contend."
	exit 3
fi
log "box clear: controllers=0"

run_arm() {
	local seed="$1"
	run_controller_arm "L1D_mpcof_${AIRFRAME}_${DIST}_s${seed}" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"L1DHAT\",\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"features\":\"pidmix+dhat\",\"obs_dhat\":true,\"dhat_l_gain\":${LGAIN},\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--max-cells 180000 --max-cells-strict \
		--neurons-gens "$NEURONS_GENS" --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 \
		--fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-episodes 100 --holdout-pop-sample 8 \
		--grid-bits 24 30 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" \
		--disturbance "$DIST" \
		--teacher mpcof \
		$FEAT_PIDMIX \
		--obs-dhat --dhat-l-gain "$LGAIN" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

for seed in $SEEDS; do
	log "===== ROUND seed=$seed ====="
	run_arm "$seed"
	rc=$?
	log "seed=$seed finished rc=$rc"
done

log "########## L1 CHAIN DONE — markers in $MARKDIR ##########"
