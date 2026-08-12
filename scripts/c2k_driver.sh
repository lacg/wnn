#!/bin/bash
# C2K — the all-2000-trained committee pool (02/07/2026).
# Motivation (horizon-drift discovery, .claude/plans/controller_break_90_v2.md +
# memory project_controller_break_90): controllers trained at 500 steps never learn
# to HOLD (84->66->28% as eval horizon grows; steady 29deg @5000) while the E2 LONG
# cell (trained @2000) extrapolates cleanly (88.5+-1.8 @5000). Committees of @500
# members cancel each other's drift (93.8@2000) and adding LONG helps (94.8@2000).
# C2K trains a DIVERSE pool of @2000 members — family diversity is what makes the
# vote work (uncorrelated drift/failures). pidmix family excluded (brittle on every
# fresh-seed test). ANCH2K attacks BOTH diagnosed gaps (yaw observability + drift).
# Pool = these 8 runs + the two FREE s16@2000 members (E2 LONG_s09 done, LONG_s10
# in the E2 sweep). Assembly: fresh-seed rescore each member @2000 (truth serum),
# then mean-PWM committees of the best 6-8 on the horizon triplet (500/2000/5000);
# ALSO audition the existing @500-trained ANCH_s09 (different information set —
# solo-brittle members can still add committee value).
# Chained: waits for the low-edge seed10 rescue to finish (ONE controller at a time).
set -u
cd /Users/lacg/wnn
LOG=logs/controller/C2K_20260702.log
exec >>"$LOG" 2>&1

echo "[c2k] $(date '+%Y-%m-%d %H:%M:%S') WAITING for rescue (/tmp/wnn_rescue_done.json)"
while [ ! -f /tmp/wnn_rescue_done.json ]; do sleep 60; done
echo "[c2k] $(date '+%Y-%m-%d %H:%M:%S') rescue done — starting C2K pool training"

export PYTHONPATH=/Users/lacg/wnn/src/wnn WNN_RUST_DAGGER=1 WNN_STATE_SPLIT=1 RAYON_NUM_THREADS=2
PY=/Users/lacg/wnn-venv/bin/python
ERR=0.25 STEADY=0.35 STABLE=0.20 JERK=0.15 MONO=0.05
ROOT=logs/controller/C2K_20260702
SEEDS="20260609 20260610"                   # seed-outer
# arm: name | grid-state-neurons | grid-bits | extra flags (all cells: steps 2000 + immigrants,
# matching the proven LONG recipe; later argparse occurrence wins for overrides)
ARMS=(
  "PWM2K|8 12 16|24|--obs-pwm"
  "TILT2K|8 12 16|24|--obs-tilt-p --obs-tilt-i"
  "LEAN2K|8|12|"
  "ANCH2K|8 12 16|24|--obs-yaw-err --neurons-gens 30"
)

run_one() {
  local name="$1" gsn="$2" gbits="$3" extra="$4" seed="$5"
  local dir="$ROOT/${name}_seed${seed}"; mkdir -p "$dir"
  if [ -f "$dir/done.json" ]; then echo "[c2k] $(date '+%H:%M:%S') SKIP ${name} seed=${seed}"; return; fi
  echo "[c2k] $(date '+%Y-%m-%d %H:%M:%S') START ${name} seed=${seed} gsn=[${gsn}] gbits=${gbits} extra=[${extra}]"
  $PY -u -m wnn.control.phased_ga \
    --grid-state-neurons $gsn --grid-bits "$gbits" --levels 16 --bits-per-feature 8 \
    --no-delta-control --integral-leak 0.99 --integral-scale 1.0 \
    --skip-stages bits,connections --lamarckian --saturation-grow-gain 1.0 --magnitude-aware-patience \
    --neurons-gens 15 --neurons-patience 6 --check-interval 5 --memory-gens 15 --memory-patience 8 \
    --pop 24 --num-eval-folds 5 \
    --eval-episodes 100 --steps 2000 --tilt 5.0 --body-rate 0.5 --yaw-rate 0.3 \
    --rg-rounds 3 --rg-episodes-per-round 8 --universe-episodes 5 \
    --immigrants 0.15 \
    --fit-weight-err-sq "$ERR" --fit-weight-steady "$STEADY" --fit-weight-stable "$STABLE" \
    --fit-weight-jerk "$JERK" --fit-weight-mono "$MONO" \
    --report-episodes 100 --report-seeds 99990001 99990101 12345 67890 --holdout-pop-sample 8 \
    --base-seed "$seed" --runs 1 \
    $extra \
    --save-winner "$dir/winner.yaml.gz" --save-stage-checkpoints "$dir" > "$dir/run.out" 2>&1
  if [ $? -ne 0 ]; then echo "[c2k] $(date '+%H:%M:%S') FAIL ${name} seed=${seed} (continuing)"
  else echo "{\"arm\":\"${name}\",\"seed\":${seed},\"extra\":\"${extra}\"}" > "$dir/done.json"
       echo "[c2k] $(date '+%Y-%m-%d %H:%M:%S') COMPLETE ${name} seed=${seed}"; fi
}

for seed in $SEEDS; do
  for a in "${ARMS[@]}"; do
    IFS='|' read -r name gsn gbits extra <<< "$a"
    run_one "$name" "$gsn" "$gbits" "$extra" "$seed"
  done
done
echo "{\"c2k_done\":true,\"ts\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}" > /tmp/wnn_c2k_done.json
echo "[c2k] $(date '+%Y-%m-%d %H:%M:%S') ALL COMPLETE — assemble the committee (e4 fresh rescore @2000/@5000)"
