#!/usr/bin/env bash
# CTRL-8 rotor-drag WNN sweep (Luiz 31/10/2026: "sweep 0.25, 0.5, 0.78 and use 0.78 if supported").
# Waits for ctrl8_resmoke_abi32.sh to finish, then flies the SAME re-smoke recipe (s31337002, pop 6,
# all stages, xy 0.5, 2 s, w_pos 0.10, --att-err-ref tilt) at k_d 0.78 (arm + twin), 0.5 (arm), 0.25 (arm).
# Sources: Folk 2026 CF2.1-Brushless free flight -> ~0.78 at 39.3 g; Förster 2015 rotor-only ~0.25.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ctrl8_drag_sweep_abi32
mkdir -p "$OUT"
LOG=$OUT/sweep.log
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
log() { echo "[ctrl8d] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "armed — waiting for ctrl8_resmoke_abi32.sh to finish"
while pgrep -f "ctrl8_resmoke_abi3[2]" >/dev/null || pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 60; done
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt \
	| sed -E 's/--save-stage-checkpoints [^ ]+//; s/--save-winner [^ ]+//; s/--fit-weight-err-sq [^ ]+//; s/--fit-weight-stable [^ ]+//; s/--fit-weight-steady [^ ]+//')
W="--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1"
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
run arm_kd078  "$F --rotor-drag 0.78"
run twin_kd078 "--rotor-drag 0.78"
run arm_kd050  "$F --rotor-drag 0.5"
run arm_kd025  "$F --rotor-drag 0.25"
log "DRAG SWEEP COMPLETE"
