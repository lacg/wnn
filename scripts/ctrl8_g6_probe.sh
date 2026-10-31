#!/bin/bash
# CTRL-8 prep, G6 label-content probe (30/10/2026): the anchor _op31 argv, training-rollout label trace
# (WNN_LABEL_TRACE), xy 0 vs 0.5 m. Each run is cut after 6 min with SIGKILL (phased_ga treats SIGTERM as a graceful pause and keeps going to the next boundary) — only the first training rollouts are needed.
cd /Users/lacg/wnn
for XY in 0.0 0.5; do
  EXTRA=""; [ "$XY" = "0.5" ] && EXTRA="--obs-pos-err-xy --obs-vel-xy"
  rm -f logs/controller/ctrl8_prep/labels_xy$XY.csv
  WNN_LABEL_TRACE=logs/controller/ctrl8_prep/labels_xy$XY.csv PYTHONPATH=src/wnn timeout -s KILL 360 nice -n 5 \
    /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -u -m wnn.control.phased_ga --levels 16 --lamarckian --skip-stages neurons,bits --max-cells 180000 --max-cells-strict  --neurons-gens 5 --neurons-patience 3 --conns-gens 5 --conns-patience 3 --memory-gens 120 --memory-patience 2 --pop 50 --num-eval-folds 5 --check-interval 2 --magnitude-aware-patience --eval-episodes 100 --memory-eval-episodes 200 --steps 2000 --tilt 5.0 --fit-weight-err-sq 0.3125 --fit-weight-stable 0.25 --fit-weight-steady 0.4375 --fit-aggregation zscore --zrank-clamp 3.0 --gate-stable 0.70 --gate-err 8.0 --delta-gamma 1.0 --grid-bits 24 --grid-output-neurons 256 --max-output-neurons 256 --report-episodes 100 --holdout-pop-sample 8 --runs 1 --memory-mode BINARY --airframe cf21_brushless --disturbance L4C --teacher mpcof --obs-peraxis-p --obs-peraxis-i --no-obs-peraxis-yaw --obs-yaw-err --obs-yaw-err-i --obs-collective-cmd --obs-alt-err --obs-vz --translation --reward-lambda-alt 0 --grid-state-neurons 0 --max-state-neurons 0 --report-seeds 99990101 99990102 99990103 99990104 99990105 --base-seed 31337002 --obs-pwm --skip-stages connections,bits --max-output-neurons 512 --max-cells 1000000000 --teacher-hover derived  --skip-stages neurons,connections,bits,memory --xy-offset $XY $EXTRA \
    > logs/controller/ctrl8_prep/g6_probe_xy$XY.out 2>&1
  echo "xy $XY rc=$? rows=$(wc -l < logs/controller/ctrl8_prep/labels_xy$XY.csv 2>/dev/null)"
done
