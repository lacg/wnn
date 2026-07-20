#!/usr/bin/env bash
# Single-layer ("reflex") WNN controller recipe — TEMPLATE (19/07/2026 promotion).
# The original RAMLayer architecture as a first-class controller citizen:
# thermometer encode → ONE RAM layer → QSR decode, trained by direct supervised
# writes (the sn=0 fast path in bptt_train_window skips every QSR solve).
#
# Knobs (env overrides):
#   MODE=BINARY|TERNARY|QUAD_WEIGHTED|QSR|PLN   memory mode        (default BINARY)
#   DIST=OFF|L1|L2|L3|L2D|L3D                   disturbance level  (default L2)
#   TEACHER=pid|lqr|mpc|lqi|mpcof               DAGGER/BC expert   (default lqr)
#   STRATEGY=ga|ts                              per-stage optimizer (default ga)
#   YAW=1                                       adds --obs-yaw-err (default off)
#   BC=1                                        pure behavior cloning: --expert-drives
#                                               + rounds=1 (default off = DAGGER)
#   BASE=<seed>                                 base seed (default 31337002)
#
# Chain: grid(sn=0 × bits) → CONNECTIONS GA/TS (the IJCNN-2004 heritage —
# connectivity discovery IS the single-layer architecture search) → MEMORY.
# NEURONS/BITS are skipped: sn is pinned 0 and output neurons are fixed by
# (motors × levels); bits stay a grid dimension.
#
# NOT launched by default anywhere — experiments are planned separately.
#   bash scripts/run_reflex_layer.sh    # or via detach_launch.py
set -u
ROOT="/Users/lacg/wnn"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
cd "$ROOT" || exit 1
MODE="${MODE:-BINARY}"
DIST="${DIST:-L2}"
TEACHER="${TEACHER:-lqr}"
STRATEGY="${STRATEGY:-ga}"
BASE="${BASE:-31337002}"
YAWFLAG=""; [ "${YAW:-0}" = "1" ] && YAWFLAG="--obs-yaw-err"
BCFLAGS=""; [ "${BC:-0}" = "1" ] && BCFLAGS="--expert-drives --rg-rounds 1"
tag="reflex_$(echo "$MODE" | tr '[:upper:]' '[:lower:]')_${DIST}_${TEACHER}_${STRATEGY}$([ -n "$YAWFLAG" ] && echo _yaw)$([ -n "$BCFLAGS" ] && echo _bc)_b${BASE}"
out="logs/controller/${tag}.out"
winner="logs/controller/${tag}_winner.yaml.gz"
echo "[reflex] START $tag -> $out $(date -u +%FT%TZ)"

# Recipe mirrors run_yawab_L2.sh (C10 weights, folds 5, held-out report) with the
# single-layer deltas: sn grid = 0, --max-state-neurons 0 (pins the box to one
# layer), CONNECTIONS stage ACTIVE (skip neurons,bits instead), no split traffic.
/usr/bin/time -l "$VP" -u -m wnn.control.phased_ga \
	--grid-state-neurons 0 --grid-bits 24 30 --levels 16 \
	--skip-stages neurons,bits --lamarckian \
	--conns-gens 60 --conns-patience 3 --memory-gens 120 --memory-patience 2 \
	--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 \
	--max-state-neurons 0 --max-output-neurons 128 --tilt 5.0 \
	--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
	--report-seed 99990101 --report-episodes 100 --holdout-pop-sample 8 \
	--base-seed "$BASE" --runs 1 --teacher "$TEACHER" \
	--disturbance "$DIST" --strategy "$STRATEGY" $YAWFLAG $BCFLAGS \
	--memory-mode "$MODE" --save-winner "$winner" > "$out" 2>&1
rc=$?
if [ -f "$winner" ]; then
	"$VP" -u scripts/gran_fpga_count.py "$winner" >> "$out" 2>&1
fi
echo "[reflex] END $tag rc=$rc $(date -u +%FT%TZ)"
