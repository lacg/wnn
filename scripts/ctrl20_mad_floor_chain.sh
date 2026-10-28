#!/usr/bin/env bash
# CTRL-20 A/B — the zscore MAD floor (--zrank-mad-floor) on the _op31 anchor recipe, x5 (s2-s6), paired vs _op31.
#
# WHY (Luiz 28/10/2026: "A/B for CTRL-20 after the probe"): the zscore combine scales each
# column by 1.4826*MAD across the candidates; when they cluster, the denominator collapses
# and noise-sized differences buy rank (_op30 s2 stage-select: a 0.4 pp stable edge = +1.35
# sigma). The floor bounds the denominator at the MEASURED within-frame comparison noise.
#
# ARM: identical to _op31 (same recipe, same S16noJM weights, alt 0.0) + --zrank-mad-floor.
# Floors (FitnessCalculatorControllerHarmonic.MEASURED_SCALE_FLOORS, CTRL-21 probe, 5ff8af70):
# stable 1.93 pp, steady 0.330 deg, err-sq column 1.88 reward units. Applies to the GA search
# AND the headline stage-select.
#
# PRE-REGISTERED READ (Plane CTRL-20, paired per seed vs _op31, REPORT headline):
#   adopt = mean paired hd better (lower) AND err not worse than +0.10 deg
#           AND stable not worse than -0.5 pp.
#
# ERA: controls flew wheel 800ae8f4 (ABI 31); the arm flies the current ABI-31 build (adds the
# weak-CAS slot fix 4384e68d + the optional floor kwarg). seed_arm_chain's era_caveat prints it.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/ctrl20_mad_floor.log"
log() { echo "[ctrl20] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

[ -f experiments/HOLD_CONTROLLER ] && { log "ABORT — HOLD_CONTROLLER present. Box left idle."; exit 1; }
log "launching CTRL-20 A/B _mf31 x5"

RECIPE="--obs-pwm --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
ARM_SUFFIX="_mf31" ARM_LABEL="ctrl20-mad-floor-obs-pwm-abi31" \
	ARM_SEEDS="31337002 31337003 31337004 31337005 31337006" \
	ARM_EXTRA_ARGS="${RECIPE} --zrank-mad-floor" \
	ARM_REQUIRE_FLAGS="--obs-pwm --skip-stages --max-output-neurons --zrank-mad-floor" \
	ARM_CTRL_SUFFIX="_op31" \
	ARM_MARKER_JSON=",\"ctrl20\":true,\"zrank_mad_floor\":true,\"mad_floors\":{\"stable\":0.0193,\"steady_deg\":0.330,\"err_sq_reward\":1.88},\"anchor\":\"_op31\",\"obs_pwm\":true" \
	ARM_LOG="/private/tmp/seed_arm_mf31.log" bash scripts/seed_arm_chain.sh
log "CTRL-20 chain exited rc=$?"
