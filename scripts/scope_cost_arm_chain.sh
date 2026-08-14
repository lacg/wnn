#!/usr/bin/env bash
# SCOPE-COST ARM (14/08/2026, Luiz) — translation ON, vertical features OFF.
#
# THE QUESTION. The stage-1 lambda=0 control (seed 31337002) scored held-out
# headline steady 1.89° where the attitude-only L1R plain arm on the SAME seed
# scored 0.82°. That ~1.1° gap is the COST OF SCOPE, but it conflates three
# causes: (a) +3 features eating address coverage (15→18 over the same b=30),
# (b) the harder plant (vertical DOF: altitude offsets, init vz, mass/collective
# jitter — the vehicle can now fall), (c) the teacher's altitude-PD cascade.
#
# THIS ARM ISOLATES (a) FROM (b)+(c): translation ON (same harder plant, same
# cascade teacher, same anchored collective) but the controller keeps the
# 15-feature pidmix layout — blind to altitude, riding the commanded collective.
#
# PRE-REGISTERED READ, per seed, on held-out headline STEADY:
#   attitude-only (L1R plain)  vs  THIS arm      = plant+teacher cost (b+c)
#   THIS arm                   vs  lambda=0 ctrl = feature cost (a)
# Report the whole 3-row decomposition per seed; n=1 per cell, so language stays
# "measurement", never "verdict".
#
# Chain position: AFTER the calib A/B (gate: 4 calibab markers + 0 controllers).
# ~3h/run, 2 runs, one controller at a time.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/scope_cost_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/scopecost"
MARKDIR="experiments/scopecost_markers"
AIRFRAME="${SCA_AIRFRAME:-cf21_brushless}"
DIST="${SCA_DIST:-L4C}"
SEEDS="${SCA_SEEDS:-31337002 31337003}"
NEURONS_GENS="${SCA_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# Gate: the calib A/B must have finished (its 4 markers) and the box be idle.
WAIT_MARKERS="${SCA_WAIT_MARKERS:-$ROOT/experiments/calibab_markers}"
WAIT_COUNT="${SCA_WAIT_COUNT:-4}"

# 15-feature pidmix — the attitude-only layout, DELIBERATELY without
# --obs-collective-cmd/--obs-alt-err/--obs-vz. That omission IS the experiment.
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"
S16_WEIGHTS="--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05"

log() { echo "[scopecost] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — scope-cost arm (translation ON, vertical features OFF), waiting for $WAIT_COUNT markers in $(basename "$WAIT_MARKERS") ##########"

# ---- gate on the calib A/B ---------------------------------------------------
# Ceiling 60 h: ~17 h of lambda sweep remain, then ~12 h of A/B; anything past
# 60 h means an upstream chain died and waiting forever would hide it.
WAITED=0
while true; do
	N=$(ls "$WAIT_MARKERS" 2>/dev/null | wc -l | tr -d ' ')
	C=$(controllers)
	if [ "$N" -ge "$WAIT_COUNT" ] && [ "$C" -eq 0 ]; then
		log "gate open: markers=$N controllers=0 (waited ${WAITED}s)"
		break
	fi
	if [ "$WAITED" -ge 216000 ]; then
		log "ABORT: markers=$N controllers=$C after 60 h — an upstream chain likely died. Not starting."
		exit 1
	fi
	sleep 300
	WAITED=$((WAITED + 300))
done

run_arm() {
	local seed="$1"
	local tag="SCOPE0_tonly_mpcof_${AIRFRAME}_${DIST}_s${seed}"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"SCOPE_COST\",\"variant\":\"translation_on_features_off\",\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"features\":\"pidmix\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
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
		--translation --fit-weight-alt 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

for seed in $SEEDS; do
	log "===== START scope-cost seed=$seed ====="
	run_arm "$seed"
	log "seed=$seed finished rc=$?"
done

log "########## SCOPE-COST ARM COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
