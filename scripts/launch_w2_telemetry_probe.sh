#!/bin/bash
# Telemetry probe (22/06): pin the integral-counter bottleneck (Hyp1/2/3) via the
# new [split-pressure] log. FULL difficulty (d=1.0, NO curriculum) drives straight
# to the hard regime where the d~0.7 plateau lives, so saturation/wish_bits at the
# converged state are maximally visible. W2 + bpf=8 (cheap; bpf is irrelevant to
# state saturation). Modest gens — we just need converged saturation/wish, not a
# champion. base-seed 20260609. RAYON=2 (XDS priority).
set -u
cd /Users/lacg/wnn
export PYTHONPATH=/Users/lacg/wnn/src/wnn
export WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
DIR=logs/controller/W2_telemetry_probe_20260622/seed0_base20260609
mkdir -p "$DIR"
echo "[w2-telemetry] $(date '+%Y-%m-%d %H:%M:%S') START full-difficulty split-pressure probe"
$PY -u -m wnn.control.phased_ga \
  --grid-state-neurons 8 12 16 --grid-bits 24 30 --levels 16 \
  --bits-per-feature 8 \
  --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 \
  --magnitude-aware-patience \
  --neurons-gens 30 --neurons-patience 6 --check-interval 5 \
  --memory-gens 20 --memory-patience 8 \
  --pop 30 --num-eval-folds 3 \
  --eval-episodes 100 --steps 500 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
  --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
  --fit-weight-err-sq 0.40 --fit-weight-stable 0.50 --fit-weight-jerk 0.05 --fit-weight-mono 0.05 \
  --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
  --base-seed 20260609 --runs 1 \
  --save-winner "$DIR/winner.yaml.gz" --save-stage-checkpoints "$DIR" \
  > "$DIR/run.out" 2>&1
echo "[w2-telemetry] $(date '+%Y-%m-%d %H:%M:%S') DONE (exit $?)"
echo "{\"telemetry_probe_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_telemetry_probe_done.json
