#!/bin/bash
# Luiz 19/11 ~05:30 EST: "apply F to all D and Y and see what impact" — after the base-D FF block (orchestrator PID $1) ends,
# fly arm F on base Y (M + --delta-max-yaw 0.5), s1-s10, ABI 59. B rows = dmy Y s1-s5 / n10 Y s6-s10 (argv-identical, gated).
cd /Users/lacg/wnn; L=logs/controller/ctrl48
echo "$(date -u +%FT%TZ) FFY: armed (pid $$): waits for the base-D FF orchestrator (pid $1) to end, then F on base Y s1-s10 (ABI 59)" >> $L/runs.log
while kill -0 $1 2>/dev/null; do sleep 60; done
ABI=59 FF_BASE=Y FF_ARMS=F FF_SEEDS="1 2 3 4 5 6 7 8 9 10" bash $L/ab_yawff.sh
echo "$(date -u +%FT%TZ) FFY: driver exited rc=$?" >> $L/runs.log
