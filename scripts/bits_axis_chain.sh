#!/usr/bin/env bash
# BITS-AXIS ROUND 1 (14/08/2026, Luiz) — does the address-bits curve keep rising
# past b=30, or turn over?
#
# THE SIGNAL. Every stage-1 run's grid picked b=30 — the TOP of the
# `--grid-bits 24 30` axis. A search that selects its boundary is evidence the
# axis should extend. b is GEOMETRIC (each bit doubles the address space), so
# round 1 probes the upper boundary {32, 34, 36}; the interior refinement
# (Luiz's 2-by-2: 26, 28) is round 2, flown ONLY if the curve turns over
# between 30 and 36.
#
# REUSE (Luiz's idea): the banked lambda-sweep runs already measured 24/30 under
# this exact recipe and these base seeds, so the 5-point curve {24,30,32,34,36}
# costs 3 new points. Each run here fixes ONE grid point, so the grid stage IS
# the bits measurement, and the run then continues through NEURONS/MEMORY to
# show what the extra bits buy post-GA.
#
# CAVEATS carried from the design discussion:
#   * lambda must match the banked rows. Default 0 (the control arm — banked
#     grid 82.0/4.31/3.61 on s31337002). Re-run at the adopted lambda after the
#     sweep verdict = round 1b (BITS_LAMBDA env).
#   * Z-7020: b=30 measured 52-84%% BRAM; wider keys grow it. If b>30 cannot
#     fit, its rung is an accuracy datum, not a hardware claim — check before
#     any FPGA statement.
#   * The calib A/B may adopt the airframe ladder (lineage break) — then the
#     banked 24/30 rows die and this cohort needs re-flying on the new ladder.
#     That is WHY this chain is gated BEHIND the A/B + scope-cost.
#
# 3 points x 2 seeds = 6 runs, ~3h each, one controller at a time.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/bits_axis_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/bitsaxis"
MARKDIR="experiments/bitsaxis_markers"
AIRFRAME="${BITS_AIRFRAME:-cf21_brushless}"
DIST="${BITS_DIST:-L4C}"
SEEDS="${BITS_SEEDS:-31337002 31337003}"
BITS_POINTS="${BITS_POINTS:-32 34 36}"
BITS_LAMBDA="${BITS_LAMBDA:-0}"
NEURONS_GENS="${BITS_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# Gate: behind the scope-cost arm (which is itself behind the calib A/B).
WAIT_MARKERS="${BITS_WAIT_MARKERS:-$ROOT/experiments/scopecost_markers}"
WAIT_COUNT="${BITS_WAIT_COUNT:-2}"

# Stage-1 recipe, byte-identical to the lambda sweep except the grid axis —
# that identity is what makes the banked 24/30 rows reusable.
FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[bitsaxis] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — bits-axis round 1 points=[$BITS_POINTS] lambda=$BITS_LAMBDA, waiting for $WAIT_COUNT markers in $(basename "$WAIT_MARKERS") ##########"

# ---- gate on the scope-cost arm ---------------------------------------------
# Ceiling 96 h: ~14 h sweep + ~12 h A/B + ~6 h scope-cost upstream, with slack;
# past that an upstream chain died and waiting forever would hide it.
WAITED=0
while true; do
	N=$(ls "$WAIT_MARKERS" 2>/dev/null | wc -l | tr -d ' ')
	C=$(controllers)
	if [ "$N" -ge "$WAIT_COUNT" ] && [ "$C" -eq 0 ]; then
		log "gate open: markers=$N controllers=0 (waited ${WAITED}s)"
		break
	fi
	if [ "$WAITED" -ge 345600 ]; then
		log "ABORT: markers=$N controllers=$C after 96 h — an upstream chain likely died. Not starting."
		exit 1
	fi
	sleep 300
	WAITED=$((WAITED + 300))
done

run_point() {
	local b="$1" seed="$2"
	local tag="BITS_b${b}_lam${BITS_LAMBDA}_mpcof_${AIRFRAME}_${DIST}_s${seed}"
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
		"\"arm\":\"BITS_AXIS\",\"bits\":${b},\"lambda_alt\":${BITS_LAMBDA},\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"features\":\"pidmix+vertical\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
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
		--grid-bits "$b" \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" \
		--disturbance "$DIST" \
		--teacher mpcof \
		$FEAT_STAGE1 \
		--translation --fit-weight-alt "$BITS_LAMBDA" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# INTERLEAVED (sweep discipline): every point once per seed-round, so an early
# read exists for all three points before any gets its second seed.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (all bits points) ====="
	for b in $BITS_POINTS; do
		log "===== START b=$b seed=$seed ====="
		run_point "$b" "$seed"
		log "b=$b seed=$seed finished rc=$?"
	done
done

log "########## BITS-AXIS ROUND 1 COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
