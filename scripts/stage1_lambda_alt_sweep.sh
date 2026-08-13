#!/usr/bin/env bash
# SCOPE C STAGE 1 — λ_alt REWARD-WEIGHT SWEEP (13/08/2026)
#
# WHY THIS IS THE FIRST STAGE-1 EXPERIMENT. The altitude reward term is
# −λ_alt·alt_err², and λ_alt also carries the metres↔radians conversion between
# the altitude and attitude terms. The spec is explicit that it comes from a
# SWEEP, "NOT guessed" — the same discipline that produced the C10 and S16
# weight sets. Nothing else in stage 1 can be interpreted until this is pinned:
# too small and the controller ignores altitude, too large and it trades away
# the attitude precision the whole programme is measured on.
#
# WHAT IS MEASURED. Per λ, the run's own held-out triple (stable%/err°/steady°)
# PLUS the altitude error the vertical channel is supposed to fix. The winner is
# the smallest λ that holds altitude while leaving the attitude triple inside
# the noise band of the λ=0 control — i.e. we buy altitude without selling
# attitude. λ=0 IS an arm here, and it is the control: with translation ON but
# λ=0 the vehicle still falls, so the arm also tells us what "no altitude
# reward" actually costs.
#
# INTERLEAVED (feedback_sweeps_always_interleave): round 1 flies ONE run of each
# λ before any λ gets its second seed, so an early cull is possible and a
# crash at hour 3 still leaves a complete round-1 comparison.
#
# PRE-REGISTERED READ. Rank by held-out ALTITUDE error first (does the channel
# work at all?), then require the attitude triple to be within ~1 SD of the λ=0
# control's. Report the whole table — one row per λ per seed — never just the
# winner (Rule 5/7: show the data).
#
# ~2h45m/run, one controller at a time. 5 λ values × 2 seeds = 10 runs ≈ 28 h,
# so round 1 (5 runs, ~14 h) is the overnight unit.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/stage1_lambda_sweep.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/stage1lambda"
MARKDIR="experiments/stage1lambda_markers"
AIRFRAME="${S1_AIRFRAME:-cf21_brushless}"
DIST="${S1_DIST:-L4C}"
SEEDS="${S1_SEEDS:-31337002 31337003}"
NEURONS_GENS="${S1_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
# λ=0 is the CONTROL arm (translation on, no altitude reward), not a filler.
LAMBDAS="${S1_LAMBDAS:-0 1 4 16 64}"

# Stage-1 feature set = today's pidmix (15) + the three vertical features (18).
FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"

# S16 weights — the banked attitude set, UNCHANGED so the attitude side of the
# comparison stays commensurable with every arm flown to date.
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[s1lambda] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — STAGE 1 lambda_alt sweep airframe=$AIRFRAME dist=$DIST lambdas=[$LAMBDAS] seeds=[$SEEDS] ##########"

# Never run alongside another controller (feedback_controller_one_at_a_time).
WAITED=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${WAITED}s)"
	sleep 60
	WAITED=$((WAITED + 60))
done

run_lambda() {
	local lam="$1" seed="$2"
	local tag="S1L_lam${lam}_mpcof_${AIRFRAME}_${DIST}_s${seed}"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"STAGE1_LAMBDA\",\"lambda_alt\":${lam},\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"features\":\"pidmix+vertical\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
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
		$FEAT_STAGE1 \
		--translation --fit-weight-alt "$lam" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# ROUND 1: one run of EVERY lambda before any lambda gets a second seed.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (all lambdas) ====="
	for lam in $LAMBDAS; do
		log "===== START lambda=$lam seed=$seed ====="
		run_lambda "$lam" "$seed"
		log "lambda=$lam seed=$seed finished rc=$?"
	done
done

log "########## SWEEP COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
