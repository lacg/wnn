#!/usr/bin/env bash
# CTRL-19 — altitude rank weight 0.05 on the _op31 anchor recipe, x5 (s2-s6), paired vs _op31.
#
# WHY (Luiz 21/10/2026): the fitness and stage-select have flown alt weight 0.0 since the
# 25/08 gated weight sweep. The bet was that 100% stable + low err/steady would carry
# altitude with it. Across the archive it does (227 altitude rows: Spearman err~alt +0.73,
# steady~alt +0.74), but it fades at the top: among the 17 rows already >=99% stable with
# err <= 1.35 deg, altitude still spans 0.14-0.43 m (3x), and the fitness cannot see it.
# n384 s6 (_Lop31) placed 3rd on the old hd holding 0.395 m vs its anchor's 0.174 m.
#
# ARM: the anchor recipe (_op31 = --obs-pwm + grid->neurons->memory, n256, 8->30) with the
# S16noJM weights renormalised x0.95 and --fit-weight-alt 0.05 (sum stays 1). Every other
# flag is the ladder's own, so the ONLY change vs the banked _op31 control is the weight.
# Luiz 21/10: 0.05 only; a 0.10 dose arm only if this one needs it.
#
# PRE-REGISTERED READ (paired per seed vs _op31, report-seed HEADLINE + MEMORY rows):
#   success = alt better AND err not worse than +0.10 deg AND stable not worse than -0.5 pp.
#   Verdict is the mean paired delta + 95% CI (paired_power.py), never a win tally.
#
# STARTS ONLY AFTER LEVELS: waits for queue_2009c to log "QUEUE COMPLETE" and exit. If the
# queue aborts (exits without that line) this chain fails closed — a human is needed.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/ctrl19_alt_weight.log"
QLOG="/private/tmp/queue_2009.log"
log() { echo "[ctrl19] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
queue_alive() { pgrep -f "scripts/queue_2009c_obs_pwm_recip[e].sh" >/dev/null; }

log "ARMED — waiting for queue_2009c (LEVELS) to complete"
while queue_alive; do sleep 300; done
grep -q "QUEUE COMPLETE" "$QLOG" || { log "ABORT — queue_2009c exited WITHOUT 'QUEUE COMPLETE' (LEVELS aborted?). Box left idle."; exit 1; }
log "LEVELS complete — launching CTRL-19 _alt31 x5"

RECIPE="--obs-pwm --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
WEIGHTS="--fit-weight-err-sq 0.296875 --fit-weight-stable 0.2375 --fit-weight-steady 0.415625 --fit-weight-alt 0.05"
ARM_SUFFIX="_alt31" ARM_LABEL="ctrl19-alt-weight-0.05-obs-pwm-abi31" \
	ARM_SEEDS="31337002 31337003 31337004 31337005 31337006" \
	ARM_EXTRA_ARGS="${RECIPE} ${WEIGHTS}" \
	ARM_REQUIRE_FLAGS="--obs-pwm --skip-stages --max-output-neurons --fit-weight-alt" \
	ARM_CTRL_SUFFIX="_op31" \
	ARM_MARKER_JSON=",\"ctrl19\":true,\"fit_weight_alt\":0.05,\"weights_renormalised\":0.95,\"anchor\":\"_op31\",\"obs_pwm\":true" \
	ARM_LOG="/private/tmp/seed_arm_alt31.log" bash scripts/seed_arm_chain.sh
log "CTRL-19 chain exited rc=$?"
