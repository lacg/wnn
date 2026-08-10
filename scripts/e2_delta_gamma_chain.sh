#!/usr/bin/env bash
# E2 — delta-gamma: the L64 alphabet question re-asked at 1/3 the footprint.
#
# WHY (10/08/2026). Two quantizers sit between the error and the correction and they
# are MULTIPLICATIVE — however finely you can act, you cannot correct an error the
# encoder cannot resolve:
#
#   PERCEPTION  the thermometer ladder. Found mis-calibrated on 10/08 (wrong regime,
#               wrong plant, wrong policy) and addressed by E1.
#   ACTION      the delta alphabet. At levels=16 / delta_max=0.1 the smallest nonzero
#               per-step correction is 0.0125 PWM. The alphabet probe refined it
#               UNIFORMLY (L32/L64) and was REFUTED at its bar: L64 halved hold on
#               s31337002 (0.53 -> 0.31) and lost on s31337003.
#
# --delta-gamma shapes the decode's normalized offset by |t|^gamma before scaling to
# +/-delta_max. Same range, same level count, SAME CELL COUNT — resolution moved to
# where the hold window lives instead of spread uniformly. gamma=2 makes the finest
# step ~8x finer at levels=16, where levels=64 cost 3x cells (Sigma 36M vs 11M) for a
# gain that held on one seed and not the other.
#
# WHY IT FLIES AFTER E1, NOT BESIDE IT. L64's verdict is CONFOUNDED by the encoder
# defect: it was measured through a thermometer fitted on a distribution the
# controller never visits. Action resolution can only be read cleanly once perception
# is settled. Flying gamma beside E1 would measure it on an encoder we then abandon.
#
# THE ARM. gamma in {1.0 control, 2.0} x seeds {002, 003, 004} = 6 runs, ~3h.
#   3 SEEDS for the same reason E1 uses 3: the committee closure measured a 0.91 deg
#   between-seed range against 0.67-1.04 within-seed spreads, and the alphabet probe
#   itself died on "one seed each way". Two seeds cannot settle this question — that
#   is the specific failure being corrected for.
#
# THE CONTROL IS RE-FLOWN, not borrowed from E1. E1's cells ARE gamma=1.0 at this
# configuration, so reusing them looks free. It is not: a chain launches each cell
# from SOURCE at cell start, and on 10/08 a mid-chain edit confounded an entire sweep
# (the calib 2.5-vs-5.0 cells, retracted). Re-flying the control costs 3 runs and
# makes the arm internally uniform under whatever code it actually runs.
#
# BAR (pre-registered): gamma=2.0 must beat the gamma=1.0 control's HEADLINE steady on
# ALL THREE seeds without losing stable. Headline-to-headline, never stage-to-stage.
# REFUTATION: it does not => action resolution is not a hold-floor lever even with
# perception settled, and BOTH quantizer routes are closed. What remains is structural
# (sn>0 / state neurons, reassessment section 5).
#
# GRID IS NOT THE READ-OUT — same rule as E1. In the outer-q arm GRID moved opposite
# to the trained stages and swung from 96.8% to 31.0% stable across cells.
#
# ARMING. The base configuration is RESOLVED from the E1 and outer-q markers after the
# gate clears, so this chain can be armed before E1 has finished:
#   E2_WAIT_PID=<e1 chain pid> nohup scripts/e2_delta_gamma_chain.sh &
# (macOS has no setsid — nohup, then verify PPID=1.)
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/e2_delta_gamma.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/e2_delta_gamma"
MARKDIR="experiments/e2_gamma_markers"
E1_MARKDIR="${E2_E1_MARKDIR:-experiments/e1_coverage_markers}"
OQ_MARKDIR="${E2_OQ_MARKDIR:-experiments/outerq_sweep_markers}"
AIRFRAME="cf21_brushless"
DIST="L4C"
TEACHER="lqi"
NEURONS_GENS=5
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
REFIT_EPISODES="${E2_REFIT_EPISODES:-10}"

SEEDS="${E2_SEEDS:-31337002 31337003 31337004}"
GAMMAS="${E2_GAMMAS:-1.0 2.0}"
WAIT_PID="${E2_WAIT_PID:-}"
WAIT_CEIL="${E2_WAIT_CEIL:-259200}"

FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[e2] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — E2 DELTA-GAMMA gammas=[$GAMMAS] seeds=[$SEEDS] wait_pid=${WAIT_PID:-none} ##########"

if [ -n "$WAIT_PID" ]; then
	waited_pid=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited_pid % 1800)) -eq 0 ] && log "waiting for gate PID $WAIT_PID (${waited_pid}s)"
		sleep 60; waited_pid=$((waited_pid + 60))
		[ "$waited_pid" -ge "$WAIT_CEIL" ] && { log "ABORT: gate alive after ${waited_pid}s"; exit 3; }
	done
	log "gate PID $WAIT_PID exited after ${waited_pid}s"
fi

waited=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${waited}s)"
	sleep 60; waited=$((waited + 60))
	[ "$waited" -ge "$WAIT_CEIL" ] && { log "ABORT: box busy after ${waited}s"; exit 3; }
done
log "box clear: controllers=0"

# RESOLVE THE BASE CONFIGURATION from the two upstream arms, using the same
# pre-registered rule (beat the control on EVERY seed, no stable loss). A "no winner"
# on either factor means that factor's CONTROL level is the base — never the least-bad
# candidate. gamma is then the ONLY variable this arm changes.
log "resolving the base encoder from $OQ_MARKDIR"
BASE_Q="$(PYTHONPATH=src/wnn "$VP" scripts/pick_best_arm_config.py \
	"$OQ_MARKDIR" outer_q none 2>>"$LOG" | tr -d '[:space:]')"
log "resolving the base refit level from $E1_MARKDIR"
BASE_REFIT="$(PYTHONPATH=src/wnn "$VP" scripts/pick_best_arm_config.py \
	"$E1_MARKDIR" refit off 2>>"$LOG" | tr -d '[:space:]')"

if [ -n "$BASE_Q" ]; then
	ENCFLAGS="--threshold-calib-tilt 5.0 --threshold-outer-quantile $BASE_Q"
	ENCTAG="q${BASE_Q//./}"
else
	ENCFLAGS="--threshold-calib-tilt 30"
	ENCTAG="c30"
fi
if [ "$BASE_REFIT" = "on" ]; then
	REFITFLAGS="--threshold-refit-from-student --threshold-refit-episodes $REFIT_EPISODES"
	REFITTAG="refiton"
else
	REFITFLAGS=""
	REFITTAG="refitoff"
fi
log "base configuration RESOLVED: enc=$ENCTAG refit=$REFITTAG — gamma is the only variable"

# PRE-FLIGHT (the rule three dead cohorts bought on 09-10/08): one tiny run in this
# arm's exact flag shape before committing the box. gamma is in the pre-flight shape
# deliberately — this is --delta-gamma's FIRST flight, and an unflown code path is
# exactly what a 60s pop-6 launch catches and a green unit suite does not.
log "pre-flight: tiny --delta-gamma launch (enc=$ENCTAG refit=$REFITTAG)"
if ! PYTHONPATH=src/wnn "$VP" -u -m wnn.control.phased_ga \
		--levels 16 --delta-gamma 2.0 $ENCFLAGS $REFITFLAGS \
		--skip-stages bits,connections \
		--neurons-gens 1 --neurons-patience 1 --memory-gens 1 --memory-patience 1 \
		--pop 6 --num-eval-folds 1 --eval-episodes 2 --memory-eval-episodes 2 \
		--steps 200 --tilt 5.0 --report-episodes 2 --holdout-pop-sample 2 \
		--grid-bits 24 --grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 --runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--report-seeds 99990101 --base-seed 31337002 > "$OUTDIR/PREFLIGHT.out" 2>&1; then
	log "ABORT: pre-flight FAILED — see $OUTDIR/PREFLIGHT.out. Arming nothing."
	exit 4
fi
log "pre-flight OK"

# Interleaved: both gammas on the first seed, then the next.
CELLS=""
for s in $SEEDS; do for g in $GAMMAS; do CELLS="$CELLS $g:$s"; done; done
log "cells (interleaved): [$CELLS]"

for cell in $CELLS; do
	gamma="${cell%%:*}"; seed="${cell##*:}"
	gtag="g${gamma//./}"
	tag="E2_${TEACHER}_${gtag}_${ENCTAG}_${REFITTAG}_${AIRFRAME}_${DIST}_s${seed}"
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"E2GAMMA\",\"teacher\":\"${TEACHER}\",\"delta_gamma\":\"${gamma}\",\"enc\":\"${ENCTAG}\",\"refit\":\"${BASE_REFIT:-off}\",\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed},\"code\":\"post-5f3d113c\"" \
		-- \
		--levels 16 --delta-gamma "$gamma" $ENCFLAGS $REFITFLAGS \
		--skip-stages bits,connections --lamarckian \
		--max-cells 180000 --max-cells-strict \
		--neurons-gens "$NEURONS_GENS" --neurons-patience 3 \
		--memory-gens 120 --memory-patience 2 \
		--pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience \
		--eval-episodes 100 --memory-eval-episodes 200 \
		--steps 2000 --tilt 5.0 \
		--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 \
		--fit-weight-jerk 0.2 --fit-weight-mono 0.1 \
		--report-episodes 100 --holdout-pop-sample 8 \
		--grid-bits 24 30 \
		--grid-state-neurons 0 --max-state-neurons 0 \
		--max-output-neurons 128 \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" --disturbance "$DIST" --teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
	log "cell $cell finished rc=$?"
done

log "########## E2 DELTA-GAMMA DONE — markers in $MARKDIR ##########"
log "NEXT: compare gamma=2.0 vs the gamma=1.0 control per seed on HEADLINE steady. Bar: beat it on ALL THREE without losing stable. If it fails, BOTH quantizer routes (perception and action) are closed and the structural route (sn>0 / state neurons) is what remains."
