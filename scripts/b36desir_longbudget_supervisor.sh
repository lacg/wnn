#!/usr/bin/env bash
# LONG-BUDGET b=36, DESIR ARM (31/08/2026) — SPEC ONLY, NOT ARMED.
#
# ⚠️ Nothing launches this. It is written and committed so the run is one command
# away if Luiz wants it; see the recommendation in the session notes before arming.
#
# WHAT IT TESTS. The probe's b=36 DESIR run was the ONE point in the whole bits
# sweep still improving when the generation cap hit it:
#     Gen 1/5  best=1.4427 (=)        Gen 2/5  (=)
#     Gen 3/5  best=1.1837 (-0.2590)  Gen 4/5  (=)
#     Gen 5/5  best=1.1576 (-0.0262)  <- improved ON THE LAST GENERATION
# Its GATE twin, by contrast, made its last move at gen 3 and sat flat for two.
# So if any point in the sweep was truncated by budget rather than converged,
# this is it, and this run is the only way to find out.
#
# WHAT IT CANNOT SETTLE. b=36 DESIR lost the held-out to b=36 GATE by 29 points
# (37.4% vs 66.6% stable), and during-search motion has been ANTI-PREDICTIVE all
# the way through this sweep — b=36 GATE was WORST of four widths at gen 1 and
# best held out. "Still growing in-search" is not evidence it would have caught
# GATE. A win here reopens the aggregation A/B; it does not by itself overturn it.
#
# THE ONLY DELTA vs scripts/sweep_ladder_probe_wide.sh's DESIR arm at b=36:
#     probe:     --conns-gens 5  --conns-patience 3
#     this run:  --conns-gens 20 --conns-patience 5
# Everything else is byte-identical — same S16noJM weights, same desirability
# aggregation with NO gate flags (the calculator REFUSES gate+desirability), same
# features, folds 5, same report seeds, same base seed. Anything else changing
# would reintroduce the confound this run exists to kill.
#
# COST at the probe's measured 2066.7s/gen: ~11.5h if it runs the full 20 gens,
# ~6-8h if magnitude-aware patience stops it around gen 10-14. Grid adds ~0.5h.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/b36desir_longbudget.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/sweep_ladder"
MARKDIR="experiments/sweepladder_markers"
AIRFRAME="cf21_brushless"; DIST="L4C"; SEED=31337002; B=36; SWEEP_N=32
TAG="SL_A_b${B}n${SWEEP_N}_${AIRFRAME}_${DIST}_desir_g20_s${SEED}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"

log() { echo "[b36desir-long] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controller_pids() { pgrep -f -- "-m wnn.control.phased_ga" 2>/dev/null || true; }
chain_pids() { pgrep -f "scripts/sweep_ladder_ab_chain.sh" 2>/dev/null || true; }

mkdir -p "$OUTDIR/ckpt/$TAG" "$MARKDIR"
log "########## ARMED — long-budget b=36 DESIR ##########"

# ---- claim the box. ONE controller at a time, checked twice. PURE WAIT: whatever
# is flying is a legitimate run and finishes on its own terms. Never preempt.
for attempt in 1 2; do
	while [ -n "$(controller_pids)" ] || [ -n "$(chain_pids)" ]; do
		log "waiting for box: controllers=[$(controller_pids | tr '\n' ' ')] chain=[$(chain_pids | tr '\n' ' ')]"
		sleep 20
	done
	sleep 2
done
log "box is clear — claiming it."

log "===== START $TAG (conns-gens 20, conns-patience 5; probe used 5/3) ====="
run_controller_arm "$TAG" "$MARKDIR" "$OUTDIR" "$VP" log \
	"\"stage\":\"A\",\"sweep\":\"bits-budget\",\"arm\":\"desir\",\"budget\":\"conns20p5\",\"bits\":${B},\"neurons\":${SWEEP_N},\"input_window_k\":1,\"seed\":${SEED}" \
	-- \
	--levels 16 --lamarckian \
	--skip-stages neurons,bits \
	--max-cells 180000 --max-cells-strict \
	--save-stage-checkpoints "$OUTDIR/ckpt/$TAG" \
	--neurons-gens 5 --neurons-patience 3 \
	--conns-gens 20 --conns-patience 5 \
	--memory-gens 120 --memory-patience 2 \
	--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
	--eval-episodes 100 --memory-eval-episodes 200 \
	--steps 2000 --tilt 5.0 \
	--fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375 \
	--fit-aggregation desirability \
	--grid-bits "$B" --grid-output-neurons "$SWEEP_N" --max-output-neurons "$SWEEP_N" \
	--report-episodes 100 --holdout-pop-sample 8 \
	--runs 1 --memory-mode BINARY \
	--airframe "$AIRFRAME" --disturbance "$DIST" --teacher mpcof \
	--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i \
	--obs-collective-cmd --obs-alt-err --obs-vz \
	--translation --reward-lambda-alt 0 \
	--grid-state-neurons 0 --max-state-neurons 0 \
	--report-seeds $REPORT_SEEDS \
	--base-seed "$SEED"
log "$TAG finished rc=$? ########## LONG-BUDGET b=36 DESIR COMPLETE ##########"
log "READ IT AS: compare headline_holdout vs the probe's b36 DESIR (37.4%/10.29/11.55) and vs b36 GATE (66.6%/5.94/6.53)."
