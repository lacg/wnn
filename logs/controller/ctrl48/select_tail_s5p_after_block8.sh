#!/bin/bash
# CTRL-48 S5p substitute test (prereg docs/ctrl_select_tail_prereg.md Amendment 2 + Amendment 3 [14 block-8 runs], Luiz 13/11).
# Waits for 'BLOCK8 (DONE|ABORT)', then: S5p re-picks on the 44 runs (30 block-4/5 + 14 block-8), then B re-picks on the
# 14 block-8 runs (descriptive). SKIP_RESULT=1 so the committed P-vs-B readout is never overwritten. Ends 'S5P DONE'.
cd /Users/lacg/wnn
L=logs/controller/ctrl48; O=experiments/ctrl_select_tail
until grep -qE "BLOCK8 (DONE|ABORT)" $L/runs.log; do sleep 60; done
grep -q "BLOCK8 ABORT" $L/runs.log && { echo "$(date -u +%FT%TZ) S5P ABORT: block 8 aborted" >> $L/runs.log; exit 1; }
SKIP_RESULT=1 bash $L/select_tail_recalc.sh $O/runs_s5p_44.txt S5p
SKIP_RESULT=1 bash $L/select_tail_recalc.sh $O/runs_block8_14.txt B
echo "$(date -u +%FT%TZ) S5P DONE (S5p x44 + B x14 on block 8; readout pending)" >> $L/runs.log
