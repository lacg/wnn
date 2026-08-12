#!/usr/bin/env bash
# #12 follow-up: does the split trainer's output error-gate cut cells without
# costing quality?
#
# Background: split_retrain_output writes a run at EVERY (record, output-neuron)
# it visits — an EMPTY cell always differs from the nudged target, so first visit
# always writes. Measured at production settings that is 17.3M cells/genome mean
# (50.4M max) against 3k on the non-split BPTT path, which trains only sampled
# bptt_window chunks. The 5800x gap costs ~63GB peak and ~1.28h/generation, and
# produces a memory far too large for the FPGA target.
#
# WNN_SPLIT_OUTPUT_ERR_GATE=1 skips a motor's neurons when its currently decoded
# pwm already agrees with the teacher within WNN_SPLIT_OUTPUT_ERR_TOL.
#
# Runs three arms to the SAME small budget (grid + 1 NEURONS generation), same
# seed, reporting the Gen 01 line — which carries BOTH cells/genome and quality,
# so the trade-off is read off one line per arm:
#   A  split OFF          (the 3k-cell reference)
#   B  split ON, gate OFF (today's 17M-cell behaviour)
#   C  split ON, gate ON  (the candidate)
#
# Deliberately smaller than production (pop 12, 30 episodes, 600 steps) so all
# three finish in ~1h TOTAL rather than ~4h/generation — the question here is the
# RATIO between arms, which does not need production scale. One controller at a
# time; the IDS worker keeps priority.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 RAYON_NUM_THREADS=8 WNN_CONTROLLER_GPU_EVAL=0
PY="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
[ -x "$PY" ] || PY="python"

ARGS="--levels 16 --skip-stages bits,connections --lamarckian \
--saturation-grow-gain 1.0 --neurons-gens 1 --neurons-patience 1 \
--memory-gens 1 --memory-patience 1 --pop 12 --num-eval-folds 5 \
--check-interval 2 --magnitude-aware-patience --eval-episodes 30 \
--memory-eval-episodes 30 --steps 600 --tilt 5.0 \
--fit-weight-err-sq 0.4 --fit-weight-stable 0.3 --fit-weight-jerk 0.2 \
--fit-weight-mono 0.1 --report-seed 99990101 --report-episodes 30 \
--holdout-pop-sample 4 --base-seed 31337002 --runs 1 --teacher lqr \
--disturbance L2 --memory-mode BINARY --grid-state-neurons 8 12 \
--grid-bits 24 --max-state-neurons 16 --max-output-neurons 64"

run_arm() {
	tag="$1"; split="$2"; gate="$3"
	out="/private/tmp/errgate_${tag}.out"
	echo "[errgate] START $tag (SPLIT=$split GATE=$gate) $(date -u +%FT%TZ)"
	WNN_STATE_SPLIT="$split" WNN_SPLIT_OUTPUT_ERR_GATE="$gate" \
		$PY -u -m wnn.control.phased_ga $ARGS > "$out" 2>&1
	echo "[errgate] END $tag rc=$? $(date -u +%FT%TZ)"
}

run_arm A_nosplit      0 0
run_arm B_split_nogate 1 0
run_arm C_split_gate   1 1

{
	echo "=== error-gate probe: cells vs quality, same seed 31337002 ==="
	echo "(pop 12, 30 episodes, 600 steps — ratios, not production magnitudes)"
	for a in A_nosplit B_split_nogate C_split_gate; do
		echo "--- $a ---"
		grep -E "Gen 01/1:|Gen 01/60:|Gen 01" "/private/tmp/errgate_${a}.out" | head -1
		grep -E "RESULT — during-search winner" "/private/tmp/errgate_${a}.out" | head -1
	done
} > /private/tmp/errgate_summary.txt
echo done > /tmp/wnn_errgate_done.marker
