#!/bin/bash
# ALT-WEIGHT SWEEP — pick the fitness weights now that altitude is a RANK dimension.
#
# WHY (18/08/2026). The altitude channel used to reach the search ONLY through the
# reward, as -lambda_alt*alt_err². A lambda multiplies metres² against radians², so it
# carries a unit conversion and its tuned value is bound to the CAPACITY it was swept
# at. lambda_alt=16 WAS swept properly (stage1_lambda_alt_sweep.sh, {0,1,4,16,64},
# 16 won at 97.8%/1.76°/1.34°) — but at 128n/b30, where every lambda flies. The bits
# ladder runs at 32n and sweeps b, i.e. it varies precisely the axis that sweep held
# fixed. At 32n the altitude term measures ~9,866x the attitude term, and since
# --fit-weight-err-sq ranks on REWARD, the fitness ranked almost purely on altitude:
# at b=18 it headlined a genome hovering level while tumbling at 52 deg over one
# flying at 11 deg.
#
# A RANK is scale-free — metres never compete numerically with radians, which is how
# err/stable/jerk/mono/steady already coexist across five different units. So altitude
# is now its own rank weight (--fit-weight-alt) and the reward term is off
# (--reward-lambda-alt 0): altitude counts ONCE, where units cannot distort it.
#
# THE ARMS. Luiz's two banked winners x four altitude weights, alt=0.00 being the
# control (= today's C10 / S16 exactly), plus ONE reference arm flying the OLD setting.
# The reference matters because this sweep uses a QUICK recipe: without it we would be
# comparing quick-recipe numbers against the full-recipe ladder and could not tell a
# fitness improvement from a recipe difference. It holds the recipe fixed and changes
# only the fitness.
#
# PRE-REGISTERED READ. Rank by held-out STEADY first (the programme's primary metric),
# then require altitude error not to have regressed against the alt=0.00 control of the
# same base. We are buying altitude WITHOUT selling attitude; an arm that wins steady by
# abandoning altitude has not won. Report every arm — never just the winner (Rule 5/7).
#
# ~1.5 h/arm x 9 = ~14 h, one controller at a time, detached. Idempotent per arm:
# an arm whose marker exists is skipped, so a kill + re-arm resumes where it stopped.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/alt_weight_sweep.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/alt_weight_sweep"
MARKDIR="experiments/altweight_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
SEED="${AW_SEED:-31337002}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# The operating point: the ladder's own regime, at the width where the pathology was
# starkest (b=18 had the sweep's BEST block, 62.6%/8.60°/11.02°, and its WORST headline).
BITS="${AW_BITS:-18}"
NEURONS="${AW_NEURONS:-32}"

FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"

mkdir -p "$OUTDIR" "$MARKDIR"

log()
{
	echo "[altsweep] $(date -u +%FT%TZ) $*" >> "$LOG"
}

# arm <tag> <err_sq> <steady> <stable> <jerk> <mono> <alt_rank> <reward_lambda_alt>
arm()
{
	local tag="$1" e="$2" st="$3" sb="$4" j="$5" m="$6" alt="$7" lam="$8"
	mkdir -p "$OUTDIR/ckpt/$tag"
	log "===== $tag: err_sq=$e steady=$st stable=$sb jerk=$j mono=$m ALT_RANK=$alt lambda_alt=$lam ====="
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"sweep\":\"alt_weight\",\"bits\":${BITS},\"neurons\":${NEURONS},\"alt_rank\":${alt},\"reward_lambda_alt\":${lam},\"seed\":${SEED}" \
		-- \
		--levels 16 --lamarckian \
		--skip-stages neurons,bits \
		--max-cells 180000 --max-cells-strict \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
		--conns-gens 5 --conns-patience 3 \
		--memory-gens 40 --memory-patience 2 \
		--pop 30 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 40 --memory-eval-episodes 80 \
		--steps 2000 --tilt 5.0 \
		--fit-weight-err-sq "$e" --fit-weight-steady "$st" --fit-weight-stable "$sb" \
		--fit-weight-jerk "$j" --fit-weight-mono "$m" --fit-weight-alt "$alt" \
		--report-episodes 50 --holdout-pop-sample 8 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
		$FEAT_STAGE1 \
		--translation --reward-lambda-alt "$lam" \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--grid-bits "$BITS" --grid-output-neurons "$NEURONS" \
		--max-output-neurons "$NEURONS" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$SEED"
	log "$tag finished rc=$?"
}

log "########## ARMED — ALT-WEIGHT SWEEP b=${BITS} n=${NEURONS} seed=${SEED} (9 arms) ##########"

# --- C10 base (the DELTA winner): err .40 / stable .30 / jerk .20 / mono .10 --------
arm "AW_C10_alt000_b${BITS}n${NEURONS}_s${SEED}" 0.40 0.00 0.30 0.20 0.10 0.00 0
arm "AW_C10_alt010_b${BITS}n${NEURONS}_s${SEED}" 0.40 0.00 0.30 0.20 0.10 0.10 0
arm "AW_C10_alt020_b${BITS}n${NEURONS}_s${SEED}" 0.40 0.00 0.30 0.20 0.10 0.20 0
arm "AW_C10_alt035_b${BITS}n${NEURONS}_s${SEED}" 0.40 0.00 0.30 0.20 0.10 0.35 0

# --- S16 base (the ABSOLUTE winner, what the ladder flew): .25/.35/.20/.15/.05 ------
arm "AW_S16_alt000_b${BITS}n${NEURONS}_s${SEED}" 0.25 0.35 0.20 0.15 0.05 0.00 0
arm "AW_S16_alt010_b${BITS}n${NEURONS}_s${SEED}" 0.25 0.35 0.20 0.15 0.05 0.10 0
arm "AW_S16_alt020_b${BITS}n${NEURONS}_s${SEED}" 0.25 0.35 0.20 0.15 0.05 0.20 0
arm "AW_S16_alt035_b${BITS}n${NEURONS}_s${SEED}" 0.25 0.35 0.20 0.15 0.05 0.35 0

# --- REFERENCE: today's production setting — altitude in the REWARD, none in the rank.
# The control that makes the other eight interpretable under this quick recipe.
arm "AW_REF_lam16_b${BITS}n${NEURONS}_s${SEED}" 0.25 0.35 0.20 0.15 0.05 0.00 16

log "########## ALT-WEIGHT SWEEP COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
