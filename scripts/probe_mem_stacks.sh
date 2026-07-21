#!/usr/bin/env bash
# Memory-attribution probe for the controller optimizer.
#
# WHY THIS EXISTS: the phase-2 RSS sampler reported ~11 GB while the process
# was actually at 44 GB phys_footprint (peak 55 GB) — RSS excludes the ~21 GB
# macOS had pushed into the compressor. Six instruments have now been wrong
# about this number. This one does not measure a proxy: it runs under
# MallocStackLogging so `malloc_history -allBySize` attributes every LIVE
# allocation to the stack that made it.
#
# The architecture flags (pop / grid bits / neuron caps / folds / memory mode)
# are IDENTICAL to production, because those are what size the allocations.
# Only the episode/step/generation counts are cut, because those drive wall
# clock rather than peak footprint.
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1
export PYTHONPATH="${ROOT}/src/wnn:${PYTHONPATH:-}"
export WNN_RUST_DAGGER=1 RAYON_NUM_THREADS=10 WNN_CONTROLLER_GPU_EVAL=0
export WNN_STATE_SPLIT=1
export MallocStackLogging=1

OUT="${OUT:-/private/tmp/wnn_memprobe.out}"
PY="/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python"
[ -x "$PY" ] || PY="python"

# Production architecture (verbatim from run_prod_reflex_then_dfa.sh PHASE 2).
ARCH="--pop 50 --num-eval-folds 5 --memory-mode BINARY \
--grid-state-neurons 8 12 16 --grid-bits 24 30 \
--max-state-neurons 24 --max-output-neurons 128 \
--levels 16 --skip-stages bits,connections --lamarckian \
--saturation-grow-gain 1.0 --magnitude-aware-patience --teacher lqr --disturbance L2"

# Cut ONLY the GENERATION count. Episodes/steps/folds stay at production values:
# cells-per-genome is a function of TRAINING VOLUME (episodes x steps x folds),
# so shrinking those shrinks the very allocation under investigation. A 10-episode
# probe peaked at 895 MB while production hit 44 GB — that gap IS this axis.
SMALL="--neurons-gens 1 --neurons-patience 1 --memory-gens 1 --memory-patience 1 \
--eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 \
--report-episodes 100 --report-seed 99990101 --holdout-pop-sample 8 \
--base-seed 31337002 --runs 1"

echo "[memprobe] starting $(date -u +%FT%TZ)" > "$OUT"
exec "$PY" -u -m wnn.control.phased_ga $ARCH $SMALL >> "$OUT" 2>&1
