#!/usr/bin/env bash
# ALPHABET PROBE — does increment resolution move the hold floor?
#
# THE HYPOTHESIS (pre-registered 09/08/2026, docs/controller_reassessment_2026_08.md).
# Delta control already carries integral action at the actuator (u += delta), which is
# why the info-route levers all refuted (L1 d-hat input, integral-INPUT): the loop
# exists, the STEP SIZE is the suspect. The antagonist decode quantizes the increment:
# decoded = 0.5 + (SigmaE - SigmaI)/levels, then decoded_to_delta() maps [0,1] to
# [-delta_max, +delta_max] piecewise-linear about neutral. At levels=16 / delta_max=0.1
# the smallest nonzero per-step correction is delta_max/(levels/2) = 0.0125 PWM — to
# hold 0.00 deg steady against the L4C bias the controller must orbit its equilibrium
# in a limit cycle whose amplitude is set by that quantum. mpcof posts 0.00+-0.00 with
# a CONTINUOUS correction. Doubling levels halves the quantum with authority
# (delta_max) held fixed — the clean resolution-only axis.
#
# THE DESIGN. 2 levels x 2 base seeds = 4 runs, teacher lqi (the cheapest member,
# ~20-25 min/run, and the best solo on s31337002), IDENTICAL to the committee control
# shape except --levels (and the output-neuron ceiling it implies):
#   L32: --levels 32  --max-output-neurons 128   (quantum 0.00625)
#   L64: --levels 64  --max-output-neurons 256   (quantum 0.003125)
# CONTROLS (already flown, levels=16): CMT_lqi_..._s31337002 = 99.8/1.11/0.53 and
# CMT_lqi_..._s31337003 = 100.0/1.58/0.81 (stable%/err deg/steady deg, headline
# held-out mean over report seeds 99990101-05).
#
# SUCCESS (pre-registered): an arm beats the SAME-SEED levels=16 control's steady on
# BOTH base seeds without losing stable. Beating it on one seed only = suggestive,
# re-fly before believing (base-seed variance is ~0.3 deg control-to-control).
# REFUTATION: neither arm beats either control => increment quantization is NOT the
# binding floor at this operating point => the observer gap needs structure, and the
# sn>0/state programme (same doc, part 2) is promoted on merit.
#
# COST. 4 x ~25 min ≈ 2h at lqi prices (levels=64 somewhat more: 4x output cells).
# Gated behind the 31337004 scoring chain PID, then box-idle — one controller at a
# time throughout, per the standing rule.
set -u

ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
# shellcheck source=controller_arm_lib.sh
. "$ROOT/scripts/controller_arm_lib.sh"

LOG="/private/tmp/alphabet_probe_chain.log"
VP="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
OUTDIR="logs/controller/alphabet_probe"
MARKDIR="experiments/alphabet_probe_markers"
AIRFRAME="${ALP_AIRFRAME:-cf21_brushless}"
DIST="${ALP_DIST:-L4C}"
SEEDS="${ALP_SEEDS:-31337002 31337003}"
TEACHER="${ALP_TEACHER:-lqi}"
LEVELS_ARMS="${ALP_LEVELS:-32 64}"
NEURONS_GENS="${ALP_NEURONS_GENS:-5}"
REPORT_SEEDS="99990101 99990102 99990103 99990104 99990105"
WAIT_PID="${ALP_WAIT_PID:-}"
WAIT_CEIL="${ALP_WAIT_CEIL:-172800}"

# nf=15 pidmix, no d-hat, default delta pair, NO L4 write flags — the committee
# cohort's control shape, byte-for-byte except --levels / --max-output-neurons.
FEAT_PIDMIX="--obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i"

log() { echo "[alphabet] $(date -u +%FT%TZ) $*" >> "$LOG"; }
controllers() { ps -axo pid,command 2>/dev/null \
	| grep -E "wnn.control.phased_ga|e5_residual_proof" \
	| grep -v "/usr/bin/time" | grep -v grep | grep -c python; }

# Output-neuron ceiling per arm: output_neurons = num_motors * levels, and the GA must
# not be allowed to grow past the arm's own resolution (that would turn a resolution
# probe into a neuron-count probe). Ceiling = exactly num_motors * levels.
max_out() { echo $((4 * $1)); }

mkdir -p "$OUTDIR" "$MARKDIR"
log "########## ARMED — ALPHABET PROBE teacher=$TEACHER levels=[$LEVELS_ARMS] seeds=[$SEEDS] airframe=$AIRFRAME dist=$DIST wait_pid=${WAIT_PID:-none} ##########"

if [ -n "$WAIT_PID" ]; then
	waited_pid=0
	while kill -0 "$WAIT_PID" 2>/dev/null; do
		[ $((waited_pid % 1800)) -eq 0 ] && log "waiting for gate PID $WAIT_PID to exit (${waited_pid}s)"
		sleep 60
		waited_pid=$((waited_pid + 60))
		if [ "$waited_pid" -ge "$WAIT_CEIL" ]; then
			log "ABORT: gate PID $WAIT_PID still alive after ${waited_pid}s — refusing to contend."
			exit 3
		fi
	done
	log "gate PID $WAIT_PID has exited after ${waited_pid}s"
fi

waited=0
while [ "$(controllers)" -gt 0 ]; do
	log "waiting for the box: controllers=$(controllers) (${waited}s)"
	sleep 60
	waited=$((waited + 60))
	if [ "$waited" -ge "$WAIT_CEIL" ]; then
		log "ABORT: box still busy after ${waited}s — refusing to contend."
		exit 3
	fi
done
log "box clear: controllers=0"

# run_arm <seed> <levels>
run_arm() {
	local seed="$1" levels="$2"
	local tag="ALP_${TEACHER}_L${levels}_${AIRFRAME}_${DIST}_s${seed}"

	run_controller_arm "$tag" \
		"$MARKDIR" "$OUTDIR" "$VP" log \
		"\"arm\":\"ALPHABET\",\"teacher\":\"${TEACHER}\",\"levels\":${levels},\"airframe\":\"${AIRFRAME}\",\"disturbance\":\"${DIST}\",\"obs_dhat\":false,\"mode\":\"BINARY\",\"state_neurons\":0,\"neurons_gens\":${NEURONS_GENS},\"seed\":${seed}" \
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
}

# INTERLEAVED by seed (the standing sweep rule): both arms fly on seed 1 before
# seed 2, so the first ~50 min already yield an L32-vs-L64-vs-control comparison.
for seed in $SEEDS; do
	log "===== ROUND seed=$seed (levels: $LEVELS_ARMS) ====="
	for levels in $LEVELS_ARMS; do
		run_arm "$seed" "$levels"
		log "seed=$seed levels=$levels finished rc=$?"
	done
done

log "########## ALPHABET PROBE DONE — markers in $MARKDIR ##########"
log "NEXT: compare each arm's HEADLINE held-out steady vs the SAME-SEED CMT_lqi levels=16 control (0.53 on s31337002, 0.81 on s31337003). Bar: beat BOTH without losing stable."
