#!/usr/bin/env bash
# CTRL-8 RE-SMOKE on the ABI-32 plant (Luiz 31/10/2026): specific-force IMU + Mahony attitude
# features + rotor drag (default 0.3 1/s, placeholder coefficient) + attitude error vs the position
# loop's tilt reference. One seed (s31337002), _op31 argv, pop 6, all stages incl. MEMORY, xy 0.5 m,
# 2 s episodes, w_pos 0.10 (S16noJM x0.90). Arm vs position-blind twin, rotor drag ON then OFF.
#   arm_drag | twin_drag | arm_nodrag | twin_nodrag
# Read: (d) arm pos < twin pos and < start; attitude vs twin; drag on vs off.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ctrl8_resmoke_abi32
mkdir -p "$OUT"
LOG=$OUT/sweep.log
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
log() { echo "[ctrl8b] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt \
	| sed -E 's/--save-stage-checkpoints [^ ]+//; s/--save-winner [^ ]+//; s/--fit-weight-err-sq [^ ]+//; s/--fit-weight-stable [^ ]+//; s/--fit-weight-steady [^ ]+//')
W="--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1"
[ -n "$(pgrep -f 'MacOS/Python -u -m wnn.control.phased_g[a]')" ] && { log "ABORT — a controller run is already flying"; exit 1; }
[ -f experiments/HOLD_CONTROLLER ] && { log "ABORT — HOLD_CONTROLLER present"; exit 1; }
run() {
	local tag="$1" extra="$2"
	[ -f experiments/HOLD_CONTROLLER ] && { log "HOLD — stopping before $tag"; exit 0; }
	log "START $tag ($extra)"
	PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $BASE $W \
		--pop 6 --skip-stages connections,bits --xy-offset 0.5 --att-err-ref tilt $extra \
		--save-winner $OUT/${tag}_winner.yaml.gz > $OUT/${tag}.out 2>&1
	log "END $tag rc=$? | $(grep -a '\[stage-select\] HEADLINE held-out' $OUT/${tag}.out | tail -1 | cut -c1-300)"
}
F="--obs-pos-err-xy --obs-vel-xy"
run arm_drag    "$F"
run twin_drag   ""
run arm_nodrag  "$F --no-rotor-drag"
run twin_nodrag "--no-rotor-drag"
log "RE-SMOKE COMPLETE"
