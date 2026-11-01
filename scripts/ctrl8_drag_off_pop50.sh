#!/usr/bin/env bash
# CTRL-8 pop-50 NO-DRAG control (Luiz 01/11/2026: the drag on/off question was only flown at pop 6 —
# a smoke — so settle it at pop 50). Waits for ctrl8_drag_sweep_pop50.sh (k_d 0.78/0.5/0.25), then
# flies the SAME arm recipe with --no-rotor-drag (k_d 0) on s31337002.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ctrl8_drag_sweep_pop50
LOG=$OUT/sweep.log
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
log() { echo "[ctrl8p50-off] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "armed — waiting for ctrl8_drag_sweep_pop50.sh to finish"
while pgrep -f "ctrl8_drag_sweep_pop5[0]" >/dev/null || pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 60; done
[ -f experiments/HOLD_CONTROLLER ] && { log "HOLD — not starting arm_kd000"; exit 0; }
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt \
	| sed -E 's/--save-stage-checkpoints [^ ]+//; s/--save-winner [^ ]+//; s/--fit-weight-err-sq [^ ]+//; s/--fit-weight-stable [^ ]+//; s/--fit-weight-steady [^ ]+//')
W="--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1"
S2="--xy-offset 0.5 --obs-pos-err-xy --obs-vel-xy --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto"
tag=arm_kd000
log "START $tag (--no-rotor-drag)"
PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $BASE $W $S2 --no-rotor-drag \
	--save-stage-checkpoints $OUT/ckpt/$tag --save-winner $OUT/${tag}_winner.yaml.gz > $OUT/${tag}.out 2>&1
log "END $tag rc=$? | $(grep -a '\[stage-select\] HEADLINE held-out' $OUT/${tag}.out | tail -1 | cut -c1-300)"
