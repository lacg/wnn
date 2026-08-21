#!/usr/bin/env bash
# GATED WEIGHT SWEEP — the first controller runs under zscore + viability gate
# (21/08/2026, Luiz: "run a sweep with the zscore, deb's thing, gate on, the
# whole nine yards and the two control runs (c10/s16) with n=5").
#
# Every historical controller weight result (wsweep -> C10, ABS -> S16) was
# decided under rank-WHM with NO gate — the regime where a tumbling genome
# ranked 2/4 (commit 2aa768a3). This chain re-asks the weight question under
# the validated regime: --fit-aggregation zscore + --gate-stable 0.70
# --gate-err 8.0, flight config byte-identical to the fitness A/B (C10 flight,
# 128n, grid-bits 24+30, mpcof, cf21_brushless, L4C, lambda_alt=0).
#
# SIX ARMS (approved 21/08; wsweep+ABS full re-run held in reserve):
#   C10       err .40  stable .30  jerk .20  mono .10   CONTROL 1 (wsweep winner)
#   S16       err .25  steady .35  stable .20 jerk .15 mono .05  CONTROL 2 (ABS winner)
#   C10noJM   err .57  stable .43                        matched pair: C10 minus jerk/mono
#   S16noJM   err .3125 stable .25 steady .4375          matched pair: S16 minus jerk/mono
#   E50S50    err .50  stable .50                        mission-only, balanced
#   STEADY40  err .30  stable .30  steady .40            the steady axis isolated
#
# PRE-REGISTERED READ: paired per-seed on the held-out full row; primary =
# stable% and err°; winner = paired majority across the 5 seeds, NEVER
# best-of-N. The matched-pair prediction: with viability gated, removing
# jerk/mono weights is at worst neutral (noJM >= its control on the primaries).
#
# SEED-MAJOR interleave (feedback_sweeps_always_interleave): round 1 = one seed
# of every arm. Idempotent per marker. ~3.3 h/run x 30 runs ~ 99 h (~4.1 days).
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/gated_wsweep.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/gated_wsweep"
MARKDIR="experiments/gatedwsweep_markers"
AIRFRAME="${GW_AIRFRAME:-cf21_brushless}"
DIST="${GW_DIST:-L4C}"
SEEDS="${GW_SEEDS:-31337002 31337003 31337004 31337005 31337006}"
ARMS="${GW_ARMS:-C10 S16 C10noJM S16noJM E50S50 STEADY40}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
GATE="--gate-stable 0.70 --gate-err 8.0"

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"

# arm -> weight flags. ONE lookup so a typo dies at arm time, not mid-chain.
weights_for() {
	case "$1" in
		C10)      echo "--fit-weight-err-sq 0.40 --fit-weight-stable 0.30 --fit-weight-jerk 0.20 --fit-weight-mono 0.10" ;;
		S16)      echo "--fit-weight-err-sq 0.25 --fit-weight-steady 0.35 --fit-weight-stable 0.20 --fit-weight-jerk 0.15 --fit-weight-mono 0.05" ;;
		C10noJM)  echo "--fit-weight-err-sq 0.57 --fit-weight-stable 0.43" ;;
		S16noJM)  echo "--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375" ;;
		E50S50)   echo "--fit-weight-err-sq 0.50 --fit-weight-stable 0.50" ;;
		STEADY40) echo "--fit-weight-err-sq 0.30 --fit-weight-stable 0.30 --fit-weight-steady 0.40" ;;
		*) return 1 ;;
	esac
}

log() { echo "[gated-wsweep] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"

# Refuse to arm on a bad arm list BEFORE waiting hours for the box.
for arm in $ARMS; do
	weights_for "$arm" >/dev/null || { log "FATAL: unknown arm '$arm'"; exit 1; }
done

log "########## ARMED — GATED WEIGHT SWEEP [$ARMS] x seeds [$SEEDS], zscore + gate(0.70, 8.0°) ##########"

WAITED=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${WAITED}s)"
	sleep 60
	WAITED=$((WAITED + 60))
done

run_arm() {
	local arm="$1" seed="$2"
	local wf; wf="$(weights_for "$arm")"
	local tag="GWS_${arm}_${AIRFRAME}_${DIST}_s${seed}"
	mkdir -p "$OUTDIR/ckpt/$tag"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"GATED_WSWEEP\",\"weights\":\"${arm}\",\"aggregation\":\"zscore\",\"gate_stable\":0.70,\"gate_err\":8.0,\"reward_lambda_alt\":0,\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
		--max-cells 180000 --max-cells-strict \
		--neurons-gens 5 --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		$wf \
		--fit-aggregation zscore --zrank-clamp 3.0 \
		$GATE \
		--report-episodes 100 --holdout-pop-sample 8 \
		--grid-bits 24 30 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" \
		--disturbance "$DIST" \
		--teacher mpcof \
		$FEAT_STAGE1 \
		--translation --reward-lambda-alt 0 \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
}

# Seed-major: round 1 = one seed of EVERY arm (the standing interleave rule).
for seed in $SEEDS; do
	log "===== ROUND seed=$seed ====="
	for arm in $ARMS; do
		run_arm "$arm" "$seed"
	done
done

log "########## GATED WEIGHT SWEEP COMPLETE — $(ls "$MARKDIR" | wc -l | tr -d ' ') markers ##########"
