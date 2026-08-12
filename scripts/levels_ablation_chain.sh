#!/bin/bash
# OUTPUT-RESOLUTION ablation: does a finer decode alphabet break the ~1.2 deg floor?
#
# WHY. Three teachers spanning 0.69-0.93 deg of classical quality all produced students
# at 1.2-1.3 deg on cf21_brushless/L4C. A floor indifferent to teacher quality is a
# substrate property. scripts/decode_quantization_ceiling.py isolated it with NO learning
# involved: quantizing a 0.405 deg controller onto the BINARY antagonist alphabet
# (pwm = 0.5 + (SumE-SumI)/levels, levels=16 -> 17 values, step 0.0625) inflates its mean
# error to 1.941 deg — 4.8x — because one step at hover is ~1.23e-3 N.m of roll torque
# against a ~1.36e-3 N.m trim requirement. The smallest available action IS the whole
# correction, so the loop can only alternate one-step-under / one-step-over.
#
# levels was NEVER SEARCHED. The grid optimised state-neurons and bits while --levels 16
# sat fixed under every controller result to date. This measures what raising it buys.
#
# ARMS (mpcof only — the best classical teacher, so a floor here is not a teacher
# artefact; 2 seeds each for n=2). The existing L4 screen supplies the L=16 point:
#     L=16  (already measured)  MEMORY 1.21+-0.05 deg   64 output neurons
#     L=32  this driver          predicted ~0.94        128 output neurons
#     L=64  this driver          predicted ~0.57        256 output neurons
# Predictions are from the analytic harness, which is nearest-neighbour rounding = the
# BEST case for the alphabet (a real WNN must also LEARN which level to fire). Treat them
# as upper bounds on the gain, and expect the GA to fall short of them.
#
# ⚠️ THIS IS NOT A SINGLE-VARIABLE ABLATION, and the writeup must say so. Raising levels
# raises output neurons (4 x levels), which raises how many RAM cells the genome can hold.
# So an improvement is "finer output resolution AND proportionally more memory", not
# resolution alone. Resolution was already isolated analytically by the quantization
# script; this driver measures the ACHIEVABLE end-to-end gain, which is the decision-
# relevant number. The cell budget is scaled proportionally (below) precisely so the arm
# is not cell-starved for a reason unrelated to the question.
#
# RUN BUDGET. The L=16 screen ran --max-cells 180000 and mpcof peaked at mu143k, so the
# budget did not bind. At 4x the output neurons an unscaled 180k WOULD bind hard and cull
# genomes — cell starvation masquerading as "resolution does not help". So the budget
# scales with the neuron count (180000 * levels/16). If that proves to be an OOM risk on
# the L=64 arm, drop to the L=32 arm alone: it recovers 58% of the analytic loss at half
# the neurons and half the budget.
#
# ORDER: L=32 BEFORE L=64, deliberately. L=32 fits the existing --max-output-neurons 128
# cap, needs a smaller budget, and gives the 3-point trend (16/32/64) that makes the
# result interpretable — a monotonic trend is evidence, a single new point is an anecdote.
# If L=32 shows nothing, L=64 is unlikely to and can be cut.
#
# QUEUED BEHIND THE SCREEN: this waits for the L4 rerun driver AND any controller to exit
# before starting. One controller at a time is a hard project rule (the IDS worker runs
# alongside and must never be disturbed).

set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/levels_ablation.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/levels_abl"
MARKDIR="experiments/levels_abl_markers"
AIRFRAME="${LV_AIRFRAME:-cf21_brushless}"
DIST="${LV_DIST:-L4C}"
TEACHER="${LV_TEACHER:-mpcof}"
SEEDS="${LV_SEEDS:-31337002 31337003}"
LEVELS_LIST="${LV_LEVELS:-32 64}"
NEURONS_GENS="${LV_NEURONS_GENS:-5}"     # matched to the mpcof arm of the L4 screen
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
WAIT_MAX_S="${LV_WAIT_MAX_S:-43200}"     # 12h ceiling on waiting for the box

say() { echo "[levelsabl] $(date -u +%FT%TZ) $*" >> "$LOG"; }

controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }
screen_running() { pgrep -f "l4_rerun_fixedplant|l4_teacher_chain" >/dev/null 2>&1; }

mkdir -p "$OUTDIR" "$MARKDIR"
say "########## ARMED — teacher=$TEACHER airframe=$AIRFRAME dist=$DIST levels=[$LEVELS_LIST] seeds=[$SEEDS] ##########"

# ---- wait for the box, rather than aborting like the screen's own chain does ----
waited=0
while screen_running || [ "$(controllers)" -gt 0 ]; do
	if [ "$waited" -ge "$WAIT_MAX_S" ]; then
		say "ABORT: still waiting after ${WAIT_MAX_S}s (screen=$(screen_running && echo yes || echo no) controllers=$(controllers)) — refusing to wait longer."
		exit 3
	fi
	[ $((waited % 900)) -eq 0 ] && say "waiting for the box: screen_running=$(screen_running && echo yes || echo no) controllers=$(controllers) (${waited}s)"
	sleep 60
	waited=$((waited + 60))
done
say "box clear after ${waited}s: controllers=0, screen finished"

run_levels() {
	local levels="$1" seed="$2"
	local out_neurons=$((4 * levels))          # num_motors * levels_per_motor
	local budget=$((180000 * levels / 16))     # scaled with the neuron count, see header
	# levels AND the seed are in the tag: a marker that recorded them only internally
	# would be skipped by name on a re-run under a different resolution, silently
	# returning one configuration's result as another's.
	run_controller_arm "LVL${levels}_${TEACHER}_${AIRFRAME}_${DIST}_s${seed}" \
		"$MARKDIR" "$OUTDIR" "$VP" say \
		"\"arm\":\"LEVELSABL\",\"teacher\":\"${TEACHER}\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"features\":\"pidmix\",\"mode\":\"BINARY\",\"state_neurons\":0,\"levels\":${levels},\"out_neurons\":${out_neurons},\"max_cells\":${budget},\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
		-- \
		--levels "$levels" --skip-stages bits,connections --lamarckian \
		--max-cells "$budget" --max-cells-strict \
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
		--max-output-neurons "$out_neurons" \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" \
		--disturbance "$DIST" \
		--teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# Interleave by LEVELS inside each seed round (project rule: round 1 = one of each
# combo), so an early cull has a paired comparison to cull on rather than a single arm.
for seed in $SEEDS; do
	say "===== ROUND seed=$seed ====="
	for lv in $LEVELS_LIST; do
		if [ "$(controllers)" -gt 0 ]; then
			say "ABORT: controllers=$(controllers) appeared mid-chain — refusing to contend."
			exit 3
		fi
		run_levels "$lv" "$seed"
		rc=$?
		say "levels=$lv seed=$seed finished rc=$rc"
		if [ "$rc" -ne 0 ]; then
			say "ABORT: levels=$lv seed=$seed returned rc=$rc — stopping rather than stacking failures."
			exit "$rc"
		fi
	done
done

say "########## LEVELS ABLATION DONE — levels=[$LEVELS_LIST] x seeds=[$SEEDS] ##########"
say "Compare MEMORY multi-seed held-out against the L=16 point from the L4 screen"
say "(experiments/l4teach_markers/L4T_${TEACHER}_${AIRFRAME}_${DIST}_s*.json, 1.21+-0.05 deg)."
say "State in any writeup that levels co-varies with output neurons and cell budget."
