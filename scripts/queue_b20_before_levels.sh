#!/usr/bin/env bash
# queue_b20_before_levels — fly the b20 + obs-pwm arm (experiments/ctrl_b20_rule.json) in queue_2009c's
# STEP 6c tilt PARK, i.e. after _tc31 x4 and BEFORE LEVELS. queue_2009c is a running bash script and must
# never be edited in place, so this follow-on supervisor waits for its PARK line instead.
#
# LEVELS starts the moment experiments/tilt_decision exists — DO NOT write it until this logs
# "b20 DONE 4/4" (the tick STATE carries the same instruction). Honours HOLD_CONTROLLER via the chain.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG="/private/tmp/queue_b20.log"
QLOG="/private/tmp/queue_2009.log"
MARK="experiments/sweepladder_markers"
BASE="SL_C_b20n256_cf21_brushless_L4C_g10"
SEEDS="31337002 31337003 31337004 31337005"
PIPE_FLAGS="--skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000"
log() { echo "[qb20] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
banked() { local n=0; for s in $SEEDS; do [ -f "${MARK}/${BASE}_s${s}_op31.json" ] && n=$((n + 1)); done; echo $n; }
parked() { tail -n 5 "$QLOG" 2>/dev/null | grep -q "PARKED — waiting for the tilt decision"; }

log "########## ARMED — b20 + obs-pwm (_op31 recipe at --grid-bits 20) x4, slot = 2009c tilt PARK, before LEVELS ##########"
beat=0
until parked; do
	[ -f experiments/tilt_decision ] && { log "ABORT — tilt_decision already written; LEVELS may be running. b20 needs a human slot."; exit 1; }
	[ $((beat % 30)) = 0 ] && log "waiting for queue_2009c's tilt PARK (b20 ${BASE} $(banked)/4)"
	sleep 60; beat=$((beat + 1))
done
log "tilt PARK reached — flying b20 x4 (do NOT write experiments/tilt_decision until 'b20 DONE')"
ARM_SUFFIX="_op31" ARM_LABEL="b20-obs-pwm-abi31" ARM_SEEDS="$SEEDS" ARM_BITS=20 ARM_NEURONS=256 \
	ARM_EXTRA_ARGS="--obs-pwm ${PIPE_FLAGS}" ARM_REQUIRE_FLAGS="--obs-pwm --skip-stages --max-output-neurons" \
	ARM_NO_CONTROL=1 ARM_MARKER_JSON=",\"recipe\":\"grid-neurons-memory+obs-pwm\",\"comparator\":\"SL_C_b24n256_..._op31\",\"rule\":\"experiments/ctrl_b20_rule.json\"" \
	ARM_LOG="/private/tmp/seed_arm_b20_op31.log" bash scripts/seed_arm_chain.sh
[ "$(banked)" -ge 4 ] || { log "ABORT — b20 $(banked)/4 after the chain exited. A run needs a human."; exit 1; }
log "b20 DONE 4/4 — apply experiments/ctrl_b20_rule.json, THEN write experiments/tilt_decision (and the LEVELS recipe if b20 is adopted)"
