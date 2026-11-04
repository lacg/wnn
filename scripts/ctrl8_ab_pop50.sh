#!/usr/bin/env bash
# CTRL-8 stage-2 A/B at pop 50 (Luiz 04/11/2026: "yes, s3-s6 for arm vs twin now"). s2 = arm_kd078 /
# twin_kd078 (same recipe). Seed-major: arm s3, twin s3, arm s4, ... twin s6 (8 runs, ~9 h each).
# Pre-registered rule: experiments/ctrl8_ab_rule.json (n=5, REPORT headline, paired per seed).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ctrl8_ab_pop50
mkdir -p "$OUT"
LOG=$OUT/ab.log
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
log() { echo "[ctrl8ab] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
log "armed — waiting for the box to be controller-idle"
while pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]" >/dev/null; do sleep 60; done
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt \
	| sed -E 's/--save-stage-checkpoints [^ ]+//; s/--save-winner [^ ]+//; s/--fit-weight-err-sq [^ ]+//; s/--fit-weight-stable [^ ]+//; s/--fit-weight-steady [^ ]+//; s/--base-seed [0-9]+//')
W="--fit-weight-err-sq 0.28125 --fit-weight-stable 0.225 --fit-weight-steady 0.39375 --fit-weight-pos 0.1"
S2="--xy-offset 0.5 --att-err-ref tilt --neurons-gens 15 --neurons-patience 3 --eval-mem-budget auto --rotor-drag 0.78"
POS="--obs-pos-err-xy --obs-vel-xy"
run() {
	local tag="$1" seed="$2" extra="$3"
	[ -f experiments/HOLD_CONTROLLER ] && { log "HOLD — stopping before $tag"; exit 0; }
	[ -f "$OUT/${tag}_winner.yaml.gz" ] && { log "SKIP $tag (winner exists)"; return; }
	log "START $tag (seed $seed${extra:+, position inputs})"
	PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $BASE --base-seed "$seed" $W $S2 $extra \
		--save-stage-checkpoints $OUT/ckpt/$tag --save-winner $OUT/${tag}_winner.yaml.gz > $OUT/${tag}.out 2>&1
	log "END $tag rc=$? | $(grep -a '\[stage-select\] HEADLINE held-out' $OUT/${tag}.out | tail -1 | cut -c1-300)"
}
for s in 3 4 5 6; do
	run arm_s$s 3133700$s "$POS"
	run twin_s$s 3133700$s ""
done
log "CTRL-8 A/B COMPLETE (s3-s6)"
