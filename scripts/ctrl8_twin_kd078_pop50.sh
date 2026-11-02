#!/usr/bin/env bash
# CTRL-8 pop-50 POSITION-BLIND TWIN at k_d 0.78 (Luiz 02/11/2026: "yes" to one twin after the no-drag
# control). Same recipe as arm_kd078 (ctrl8_drag_sweep_pop50.sh) on s31337002, ABI 33, MINUS the two
# position inputs (--obs-pos-err-xy --obs-vel-xy); fitness keeps w_pos 0.10 like the re-smoke twins.
# Read: arm_kd078 vs twin_kd078 headline (stable/err/steady/alt/pos/dx/dy), same seed = n=1 direction.
# Waits for ctrl8_drag_off_pop50.sh (arm_kd000) and any controller run to exit.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ctrl8_drag_sweep_pop50
LOG=$OUT/sweep.log
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
log() { echo "[ctrl8p50-twin] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "armed — waiting for ctrl8_drag_sweep_pop50.sh + ctrl8_drag_off_pop50.sh to finish"
while pgrep -f "ctrl8_drag_sweep_pop5[0]" >/dev/null || pgrep -f "ctrl8_drag_off_pop5[0]" >/dev/null \
	|| pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 60; done
[ -f experiments/HOLD_CONTROLLER ] && { log "HOLD — not starting twin_kd078"; exit 0; }
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt \
	| sed -E 's/--save-stage-checkpoints [^ ]+//; s/--save-winner [^ ]+//; s/--fit-weight-err-sq [^ ]+//; s/--fit-weight-stable [^ ]+//; s/--fit-weight-steady [^ ]+//')
W="--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1"
S2="--xy-offset 0.5 --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto"
tag=twin_kd078
log "START $tag (--rotor-drag 0.78, position-blind)"
PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $BASE $W $S2 --rotor-drag 0.78 \
	--save-stage-checkpoints $OUT/ckpt/$tag --save-winner $OUT/${tag}_winner.yaml.gz > $OUT/${tag}.out 2>&1
log "END $tag rc=$? | $(grep -a '\[stage-select\] HEADLINE held-out' $OUT/${tag}.out | tail -1 | cut -c1-300)"
