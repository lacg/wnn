#!/usr/bin/env bash
# CTRL-19 rung 2 — altitude rank weight 0.10 on the _op31 anchor recipe, x5 (s2-s6), paired vs _op31.
#
# WHY (Luiz 26/10/2026): the 0.05 rung (_alt31, 4/5 banked) reads as a toss-up — attitude
# held (err -0.04 deg, steady -0.03 deg, stable -0.30 pp mean) but the altitude gain
# (-0.035 m) came from s2/s3 only, s4/s5 flat. Rather than more 0.05 seeds, one dose step:
# a clear altitude gain at 0.10 = dose-response; attitude starting to sell = the boundary.
#
# ARM: identical to _alt31 (scripts/ctrl19_alt_weight_chain.sh) except the weights:
# S16noJM renormalised x0.90 + --fit-weight-alt 0.10 (sum stays 1).
#
# PRE-REGISTERED READ (same rule as rung 1, paired per seed vs _op31, report-seed HEADLINE):
#   success = alt better AND err not worse than +0.10 deg AND stable not worse than -0.5 pp.
#
# STARTS ONLY AFTER rung 1: waits for the _alt31 seed_arm_chain (pid given as $1) to exit,
# then requires all 5 _alt31 markers. Fails closed otherwise — a human is needed.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/ctrl19_alt10.log"
WAIT_PID="${1:?usage: ctrl19_alt10_chain.sh <pid of the _alt31 seed_arm_chain>}"
MARK="experiments/sweepladder_markers"
log() { echo "[ctrl19b] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "ARMED — waiting for the _alt31 chain (pid ${WAIT_PID}) to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
n=0
for s in 31337002 31337003 31337004 31337005 31337006; do
	[ -f "${MARK}/SL_C_b24n256_cf21_brushless_L4C_g10_s${s}_alt31.json" ] && n=$((n+1))
done
[ "$n" = "5" ] || { log "ABORT — _alt31 chain exited with ${n}/5 markers. Box left idle."; exit 1; }
[ -f experiments/HOLD_CONTROLLER ] && { log "ABORT — HOLD_CONTROLLER present. Box left idle."; exit 1; }
sleep 120   # let the leaderboard/verdict tail of the previous chain settle
log "_alt31 5/5 banked — launching CTRL-19 rung 2 _alt10x31 x5"

RECIPE="--obs-pwm --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
WEIGHTS="--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-alt 0.10"
ARM_SUFFIX="_alt10x31" ARM_LABEL="ctrl19-alt-weight-0.10-obs-pwm-abi31" \
	ARM_SEEDS="31337002 31337003 31337004 31337005 31337006" \
	ARM_EXTRA_ARGS="${RECIPE} ${WEIGHTS}" \
	ARM_REQUIRE_FLAGS="--obs-pwm --skip-stages --max-output-neurons --fit-weight-alt" \
	ARM_CTRL_SUFFIX="_op31" \
	ARM_MARKER_JSON=",\"ctrl19\":true,\"ctrl19_rung\":2,\"fit_weight_alt\":0.10,\"weights_renormalised\":0.90,\"anchor\":\"_op31\",\"obs_pwm\":true" \
	ARM_LOG="/private/tmp/seed_arm_alt10x31.log" bash scripts/seed_arm_chain.sh
log "CTRL-19 rung 2 chain exited rc=$?"
