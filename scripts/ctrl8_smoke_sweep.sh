#!/usr/bin/env bash
# CTRL-8 stage-2 SMOKE + position-weight SWEEP (Luiz 30/10/2026: "let's add [the position term] and do a sweep";
# "ok, let's do this smoke"). One seed (s31337002), the _op31 anchor argv, pop 6, grid -> neurons, MEMORY skipped.
# Stage-2 bundle: --translation (already in the anchor) --xy-offset 0.5 + position features.
# Fitness = the decided S16noJM zscore, weights scaled x(1-w_pos), + --fit-weight-pos w_pos (3-D position metric
# for now; D2 horizontal-only metric NOT yet approved). Gate λ_pos derived (flag unset).
# Runs, in order (the key pair first):
#   arm  w_pos 0.10  |  twin (NO position features) w_pos 0.10  |  arm w_pos 0.00  |  arm w_pos 0.05  |  arm w_pos 0.20
# Smoke pass (d): arm held-out position error < start mean (~0.385 m) AND < twin.
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=logs/controller/ctrl8_smoke
LOG=$OUT/sweep.log
VP=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
log() { echo "[ctrl8] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
BASE=$(grep -v '^#' experiments/crn_noise_probe/ARGS_SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_op31.txt \
	| sed -E 's/--save-stage-checkpoints [^ ]+//; s/--save-winner [^ ]+//; s/--fit-weight-err-sq [^ ]+//; s/--fit-weight-stable [^ ]+//; s/--fit-weight-steady [^ ]+//')
[ -n "$(pgrep -f 'MacOS/Python -u -m wnn.control.phased_g[a]')" ] && { log "ABORT — a controller run is already flying"; exit 1; }
[ -f experiments/HOLD_CONTROLLER ] && { log "ABORT — HOLD_CONTROLLER present"; exit 1; }
run() {
	local tag="$1" wpos="$2" feats="$3"
	local s=$($VP -c "w=$wpos; print(f'--fit-weight-err-sq {0.3125*(1-w):.6f} --fit-weight-stable {0.25*(1-w):.6f} --fit-weight-steady {0.4375*(1-w):.6f} --fit-weight-pos {w}')")
	log "START $tag (w_pos $wpos; features: ${feats:-none})"
	PYTHONPATH=src/wnn nice -n 5 $VP -u -m wnn.control.phased_ga $BASE $s \
		--pop 6 --skip-stages connections,bits,memory --xy-offset 0.5 $feats \
		--save-winner $OUT/${tag}_winner.yaml.gz > $OUT/${tag}.out 2>&1
	log "END $tag rc=$? | $(grep -a 'NEURONS MULTI-SEED' $OUT/${tag}.out | tail -1 | cut -c1-260)"
}
F="--obs-pos-err-xy --obs-vel-xy"
run arm_wpos010  0.10 "$F"
run twin_wpos010 0.10 ""
run arm_wpos000  0.0  "$F"
run arm_wpos005  0.05 "$F"
run arm_wpos020  0.20 "$F"
log "SWEEP COMPLETE"
