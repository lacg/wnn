#!/bin/bash
# Block 9 Amendment 2 (Luiz 14/11): after 'BLOCK9 DONE', re-pick dc_ms01 s1-s3 under rule P so all 20 controllers share rule P.
cd /Users/lacg/wnn; L=logs/controller/ctrl48; O=experiments/ctrl_select_tail
until grep -qE "BLOCK9 (DONE|ABORT)" $L/runs.log; do sleep 60; done
SKIP_RESULT=1 bash $L/select_tail_recalc.sh $O/runs_dcms01_s123.txt P
echo "$(date -u +%FT%TZ) BLOCK9 REPICK DONE (dc_ms01 s1-s3 under P; decision on s1-s10 due)" >> $L/runs.log
