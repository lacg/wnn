#!/usr/bin/env bash
# FITNESS AGGREGATION A/B — harmonic (the banked WHM) vs zscore (winsorized
# robust z), 19/08/2026, Luiz's reprioritization: "the first thing actually is
# let's do a sweep with n=5 with the 128n 30b c10 with this new fitness vs the
# old way we calculate fitness."
#
# WHY. The WHM is dominated by a genome's BEST weighted rank and nearly
# indifferent to its worst — arm 9 of the alt-weight sweep headlined a genome
# that lost four of five metrics because it held rank 1 on the heaviest. And
# ranks are magnitude-blind: 1st by 13° counts the same as 1st by 0.1°. The
# zscore combine (ram_core::fitness, winsorized robust z, clamp ±3) fixes both.
# Whether it finds BETTER GENOMES is an empirical question — this chain asks it.
#
# DESIGN. 2 aggregations × 5 seeds at the banked λ-sweep operating point
# (128n / grid-bits 24 30, pop 50, mpcof, L4C, translation, λ_alt=0), C10
# weights (err .40 / stable .30 / jerk .20 / mono .10 — the DELTA-substrate
# winner; C10 is MAIN, Luiz 19/08). --fit-aggregation makes each run coherent
# END-TO-END: grid ranking, GA elitism/incumbent, stage-select, val winner.
# Everything else byte-identical between the two arms of a pair.
#
# INTERLEAVED (feedback_sweeps_always_interleave): both aggregations at seed k
# before any run at seed k+1, so every completed pair is already a paired
# comparison and a crash at hour 10 still leaves whole pairs.
#
# PRE-REGISTERED READ. Paired per-seed on the HELD-OUT full row
# (stable/err/steady/jerk/mono/alt/pos — the 352035f4 row). Primary = stable%
# and err° (C10's top-weighted intents); steady°/alt always quoted. Winner =
# better on the paired majority across 5 seeds, NEVER best-of-N. Caveat to
# carry into the report: fitness changes the search trajectory, so this
# measures "does zscore FIND better genomes", not a bit-level A/B.
#
# ~3.1-3.5 h/run (measured, S1L markers) × 10 ≈ 33 h. One controller at a
# time. Idempotent per arm: an existing marker skips the run on re-arm.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/fitness_ab.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/fitness_ab"
MARKDIR="experiments/fitnessab_markers"
AIRFRAME="${FAB_AIRFRAME:-cf21_brushless}"
DIST="${FAB_DIST:-L4C}"
SEEDS="${FAB_SEEDS:-31337002 31337003 31337004 31337005 31337006}"
AGGREGATIONS="${FAB_AGGREGATIONS:-harmonic zscore}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

# Stage-1 feature set + recipe: byte-identical to stage1_lambda_alt_sweep.sh
# (the banked 128n/b30 operating point) except the WEIGHTS (C10, not S16 —
# C10 is MAIN) and λ_alt=0 (the reward stays clean; this A/B is about the
# COMBINE, and a λ term would corrupt err² exactly as documented on arm 9).
FEAT_STAGE1="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz"
C10_WEIGHTS="--fit-weight-err-sq 0.40 --fit-weight-stable 0.30 --fit-weight-jerk 0.20 --fit-weight-mono 0.10"

log() { echo "[fitness-ab] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null | grep "wnn.control.phased_ga" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — FITNESS AGGREGATION A/B [$AGGREGATIONS] × seeds [$SEEDS], C10, 128n/b{24,30}, λ=0 ##########"

# Never run alongside another controller (feedback_controller_one_at_a_time).
WAITED=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${WAITED}s)"
	sleep 60
	WAITED=$((WAITED + 60))
done

run_arm() {
	local agg="$1" seed="$2"
	local tag="FAB_${agg}_c10_${AIRFRAME}_${DIST}_s${seed}"
	mkdir -p "$OUTDIR/ckpt/$tag"
	# shellcheck disable=SC2086
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"FITNESS_AB\",\"aggregation\":\"${agg}\",\"weights\":\"C10\",\"reward_lambda_alt\":0,\"teacher\":\"mpcof\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"seed\":${seed}" \
		-- \
		--levels 16 --skip-stages bits,connections --lamarckian \
		--save-stage-checkpoints "$OUTDIR/ckpt/$tag" \
		--max-cells 180000 --max-cells-strict \
		--neurons-gens 5 --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		$C10_WEIGHTS \
		--fit-aggregation "$agg" --zrank-clamp 3.0 \
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

# Seed-major, aggregation-inner: whole pairs land together.
for seed in $SEEDS; do
	log "===== PAIR seed=$seed ====="
	for agg in $AGGREGATIONS; do
		log "===== START agg=$agg seed=$seed ====="
		run_arm "$agg" "$seed"
		log "agg=$agg seed=$seed finished rc=$?"
	done
done

log "########## A/B COMPLETE — $(ls "$MARKDIR" | wc -l) markers ##########"
