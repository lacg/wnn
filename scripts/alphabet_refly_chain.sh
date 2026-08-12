#!/usr/bin/env bash
# ALPHABET RE-FLY (v2 = post-fix code) — make the probe internally consistent.
#
# WHY. The 09/08/2026 fixes (shared tie-aware compute_ranks 8b839a30 + top-3
# stage selection d3f64e03) change GA SEARCH DYNAMICS — controller populations
# tie massively on stable_rate=100%, so the ranking rule is part of the
# optimizer. The alphabet probe's arms 1-2 (L32/L64 s31337002) flew on OLD code;
# arms 3-4 (s31337003) fly on NEW code; every levels=16 control (the committee
# cohort) is OLD code. Cross-code comparisons confound the levels question with
# the optimizer change. This chain re-flies the old-code cells under NEW code,
# tag prefix ALP2, so the whole probe reads on ONE optimizer:
#
#   ALP2_lqi_L16_s31337002   ALP2_lqi_L16_s31337003    <- fresh controls, and a
#     free old-vs-new A/B against CMT_lqi (same recipe, old code) = the first
#     measurement of what the tie fix does to controller search outcomes
#   ALP2_lqi_L32_s31337002   ALP2_lqi_L64_s31337002    <- arms 1-2 redone
#
# (s31337003's L32/L64 already run on new code as ALP arms 3-4 — not repeated.)
# ORDER: both L16s first (cheap, ~25 min each) so the tie-fix A/B lands early.
#
# L16 runs use --max-output-neurons 128 to be BYTE-IDENTICAL to the committee
# recipe (its cap; non-binding in practice — on=64 throughout the cohort), i.e.
# cap = max(4*levels, 128). Authorized by Luiz 09/08/2026 ("arm both").
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/alphabet_refly_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/alphabet_probe"
MARKDIR="experiments/alphabet_probe_markers"
AIRFRAME="cf21_brushless"
DIST="L4C"
TEACHER="lqi"
NEURONS_GENS=5
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
# seed:levels cells to fly, in order (L16 pair first for the early A/B read).
RUNS="${ALP2_RUNS:-${ALP2_CELLS:-31337002:16 31337003:16 31337002:32 31337002:64}}"
WAIT_PID="${ALP2_WAIT_PID:-}"
WAIT_CEIL="${ALP2_WAIT_CEIL:-172800}"

FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[alp2] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }
max_out() { local m=$((4 * $1)); [ "$m" -lt 128 ] && m=128; echo "$m"; }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — ALPHABET RE-FLY v2 cells=[$RUNS] wait_pid=${WAIT_PID:-none} ##########"

if [ -n "$WAIT_PID" ]; then
	waited_pid=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited_pid % 1800)) -eq 0 ] && log "waiting for gate PID $WAIT_PID to exit (${waited_pid}s)"
		sleep 60
		waited_pid=$((waited_pid + 60))
		if [ "$waited_pid" -ge "$WAIT_CEIL" ]; then
			log "ABORT: gate PID $WAIT_PID still alive after ${waited_pid}s"; exit 3
		fi
	done
	log "gate PID $WAIT_PID exited after ${waited_pid}s"
fi

waited=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${waited}s)"
	sleep 60
	waited=$((waited + 60))
	if [ "$waited" -ge "$WAIT_CEIL" ]; then log "ABORT: box busy after ${waited}s"; exit 3; fi
done
log "box clear: controllers=0"

for run_id in $RUNS; do
	seed="${run_id%%:*}"; levels="${run_id##*:}"
	tag="ALP2_${TEACHER}_L${levels}_${AIRFRAME}_${DIST}_s${seed}"
	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"ALPHABET_V2\",\"teacher\":\"${TEACHER}\",\"levels\":${levels},\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"obs_dhat\":false,\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed},\"code\":\"post-8b839a30\"" \
		-- \
		--levels "$levels" --skip-stages bits,connections --lamarckian \
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
		--max-output-neurons "$(max_out "$levels")" \
		--runs 1 --memory-mode BINARY \
		--airframe "$AIRFRAME" \
		--disturbance "$DIST" \
		--teacher "$TEACHER" \
		$FEAT_PIDMIX \
		--save-stage-checkpoints "$OUTDIR/${tag}_stages" \
		--report-seeds $REPORT_SEEDS \
		--base-seed "$seed"
	log "cell $run_id finished rc=$?"
done

log "########## ALPHABET RE-FLY v2 DONE ##########"
log "NEXT: (1) tie-fix A/B — ALP2_lqi_L16 vs CMT_lqi headline triples per seed (same recipe, old vs new optimizer). (2) probe read on new code only: ALP2_L{16,32,64}_s31337002 + ALP2_L16/ALP_L{32,64}_s31337003."
