#!/usr/bin/env bash
# CTRL-8 rotor-drag sweep at FULL population (Luiz 31/10/2026: "if we are deciding against drag, or the
# 0.25-0.78, we should use the full 50 pop. 6 is too small"). Smoke runs (pop 6) are plumbing checks only.
# Arming: HOLD_CONTROLLER was touched so ctrl8_resmoke_abi32.sh stops after twin_drag (its run() checks HOLD
# before each run); this script waits for that chain to exit, removes HOLD, then flies one seed (s31337002)
# of the anchor argv (pop 50) + the stage-2 bundle + w_pos 0.10 + --att-err-ref tilt, NEURONS 15 gens /
# patience 3 (Luiz 31/10: "yes" to 15 + patience 3), MEMORY unchanged, at k_d 0.78 -> 0.5 -> 0.25 (arm).
# Sources: Folk 2026 CF2.1-Brushless free flight ~0.78 at 39.3 g; Förster 2015 rotor-only ~0.25.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ctrl8_drag_sweep_pop50
mkdir -p "$OUT"
LOG=$OUT/sweep.log
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
log() { echo "[ctrl8p50] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "armed — waiting for ctrl8_resmoke_abi32.sh (stops after twin_drag on HOLD)"
while pgrep -f "ctrl8_resmoke_abi3[2]" >/dev/null || pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 60; done
rm -f experiments/HOLD_CONTROLLER
log "re-smoke exited; HOLD removed"
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt \
	| sed -E 's/--save-stage-checkpoints [^ ]+//; s/--save-winner [^ ]+//; s/--fit-weight-err-sq [^ ]+//; s/--fit-weight-stable [^ ]+//; s/--fit-weight-steady [^ ]+//')
W="--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1"
S2="--xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt --neurons-gens 15 --neurons-patience 3"
run() {
	local tag="$1" extra="$2"
	[ -f experiments/HOLD_CONTROLLER ] && { log "HOLD — stopping before $tag"; exit 0; }
	log "START $tag ($extra)"
	PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $BASE $W $S2 $extra \
		--save-stage-checkpoints $OUT/ckpt/$tag --save-winner $OUT/${tag}_winner.yaml.gz > $OUT/${tag}.out 2>&1
	log "END $tag rc=$? | $(grep -a '\[stage-select\] HEADLINE held-out' $OUT/${tag}.out | tail -1 | cut -c1-300)"
}
run arm_kd078 "--rotor-drag 0.78"
run arm_kd050 "--rotor-drag 0.5"
run arm_kd025 "--rotor-drag 0.25"
log "POP-50 DRAG SWEEP COMPLETE"
