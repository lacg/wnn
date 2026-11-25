#!/bin/bash
# CTRL-73: CPU re-score of every N0 / G / T / G+T winner on the 35 TEST units (production scorer, nice 10).
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
cd /Users/lacg/wnn/logs/controller/ctrl73
while read arm s tag; do
  [ -f /tmp/ctrl73/${arm}_${s}.npz ] && continue
  nice -n 10 $PY wnn_units.py "$tag" /tmp/ctrl73/${arm}_${s}.npz
done < /tmp/ctrl73/tags.txt
echo ALLDONE
