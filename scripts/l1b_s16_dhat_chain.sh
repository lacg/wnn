#!/usr/bin/env bash
# L1b — does the HOLD floor move once the RANKING actually rewards hold?
#
# WHY THIS RUN. L1 (--obs-dhat under C10 weights) is REFUTED at n=2: giving the
# student the mpcof disturbance observer did not lower steady, it RAISED it.
#
#   arm            seed        MEMORY held-out  err / stable / steady
#   control (C10)  s31337002   1.21 / 100.0 / 0.64
#   C10 + dhat     s31337002   1.44+-0.20 / 99.8+-0.4 / 0.66+-0.17   (flat)
#   control (C10)  s31337003   1.58 / 100.0 / 0.95
#   C10 + dhat     s31337003   2.00+-0.27 / 99.8+-0.4 / 1.45+-0.32   (WORSE)
#
# But that verdict is CONFOUNDED and cannot stand alone. Both arms rank by C10
# (err^2 .40 / stable .30 / jerk .20 / mono .10), which carries NO steady term
# (--fit-weight-steady defaults 0.0, phased_ga.py:1925). So the search paid for
# +3 input features out of an unchanged grid budget and was never asked to keep
# hold-precision. "dhat is useless for hold" and "C10 never selected for hold"
# predict the same null — and even the same degradation.
#
# S16 (err .25 / steady .35 / stable .20 / jerk .15 / mono .05) is the winner of
# the 18-combo ABSOLUTE-substrate sweep (25/06/2026), where steady carries the
# LARGEST weight. It has never been flown on the DELTA substrate, and that sweep's
# own finding was that substrate dominates weights (+14.2 pp absolute-vs-delta
# against ~2.7 pp across weight sets) — so S16 does not transfer for free.
#
# THE 2x2 (spec: docs/hold_floor_levers_spec.md section L1b):
#
#            | no dhat                        | dhat
#   ---------|--------------------------------|-------------------------------
#   C10      | FLOWN (the L4 screen mpcof arm) | FLOWN (L1, refuted)
#   S16      | THIS CHAIN, 2 seeds             | THIS CHAIN, 2 seeds
#
# Only the two S16 cells are new => 4 runs. The `S16 + no dhat` cell is what makes
# this a 2x2 rather than a lone `S16 + dhat`: without it a gain cannot be
# attributed between the WEIGHTING and the FEATURE.
#
# ORDER IS INTERLEAVED BY DESIGN (feedback_sweeps_always_interleave): round 1 flies
# one of EACH combo at seed 31337002, round 2 repeats at 31337003. A chain that ran
# both seeds of one combo first would have to finish ~6 h before it could compare
# anything; interleaved, the first two runs already answer "did the weighting move
# steady at all" and a dead arm can be culled before the second seed is spent.
#
# READ STEADY AS THE PRIMARY, not err. err is ~80% recovery term and recovery is
# already teacher-grade (0.88-1.21x per D2), so a hold fix can move err by at most
# the ~20% the steady window carries.
#
# EXPECT S16 TO COST SOME err. Trading err for steady is exactly the bargain the
# 25/06 sweep accepted on the absolute substrate. Report the FULL TRIPLE
# (err/stable/steady) for every cell; do NOT declare a winner on one metric.
#
# WHAT EACH OUTCOME MEANS:
#   S16+dhat steady << S16-no-dhat steady  => dhat DOES help hold; C10 was hiding it
#                                             (L1's refutation was a ranking artifact)
#   both S16 cells drop vs their C10 twins => the WEIGHTING carries hold, not the feature
#   neither S16 cell moves steady          => the floor survives a ranking that explicitly
#                                             targets it => strongest evidence yet that it
#                                             is structural. Promotes L3 (delta_leak/
#                                             delta_max, never searched) and L4
#                                             (magnitude-weighted DAGGER conflicts).
#
# Every other flag is COPIED from scripts/l1_dhat_chain.sh, including the 5-gen
# NEURONS cap, so the only differences across the 2x2 are the fitness weights and
# the presence of --obs-dhat. A budget difference would confound the comparison.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/l1b_s16_dhat_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/l1bs16"
MARKDIR="experiments/l1bs16_markers"
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

log() { echo "[l1bs16] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — L1b S16x dhat 2x2 airframe=$AIRFRAME dist=$DIST seeds=[$SEEDS] gens=$NEURONS_GENS ##########"

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
	run_controller_arm "L1B_${variant}_mpcof_${AIRFRAME}_${DIST}_s${seed}" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"L1BS16\",\"weights\":\"S16\",\"variant\":\"${variant}\",\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",${dhat_json},\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
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

log "########## L1b CHAIN DONE — markers in $MARKDIR ##########"
