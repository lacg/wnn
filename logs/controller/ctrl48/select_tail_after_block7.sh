#!/bin/bash
# Starts the pre-registered select-tail test (amendment 1: 30 runs) once block 7 has finished, so the two drivers
# never race for the idle gate between block-7 runs. Luiz 12/11/2026, option (a).
cd /Users/lacg/wnn
L=logs/controller/ctrl48
until grep -qE "BLOCK7 (DONE|ABORT)" $L/runs.log 2>/dev/null; do sleep 60; done
exec bash $L/select_tail_recalc.sh experiments/ctrl_select_tail/runs_blocks45_a49_amended.txt
