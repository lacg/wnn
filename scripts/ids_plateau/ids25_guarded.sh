#!/bin/bash
# IDS-25 guarded runner: one process per flow, RAYON_NUM_THREADS=4, wait for >=16 GiB available
# before each flow, kill ONLY our own process if available drops below 12 GiB.
# Usage: scripts/ids_plateau/ids25_guarded.sh <outdir> <flow> [<flow> ...]   (env LIMIT=N to cap genomes)
cd /Users/lacg/wnn
export PYTHONPATH=/Users/lacg/wnn/src/wnn:/Users/lacg/wnn/src RAYON_NUM_THREADS=4
PY=/Users/lacg/wnn-venv/bin/python; OUT=$1; shift; mkdir -p "$OUT"
avail() { vm_stat | awk '/free|inactive|speculative|purgeable/{gsub("\\.","",$NF); s+=$NF} END{printf "%d", s*16384/1073741824}'; }
for f in "$@"; do
  [ -s "$OUT/flow_$f.json" ] && { echo "$(date +%T) flow $f already done"; continue; }
  while [ "$(avail)" -lt 16 ]; do sleep 30; done
  $PY scripts/ids_plateau/ids25_rescore_grid.py --flow $f --out "$OUT/flow_$f.json" ${LIMIT:+--limit $LIMIT} >> "$OUT/run.log" 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    if [ "$(avail)" -lt 12 ]; then echo "$(date +%T) avail<12GiB - killing our flow $f re-score"; kill $P; sleep 5; kill -9 $P 2>/dev/null; break; fi
    sleep 10
  done
  wait $P; echo "$(date +%T) flow $f rc $?"
done
