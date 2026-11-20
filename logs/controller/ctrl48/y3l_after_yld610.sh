#!/bin/bash
# Y3+L (Plane CTRL-65, Luiz pre-approved 20/11 conditional on the per-axis split = transient): after Y+F+L s6-s10 (waiter PID $1) ends.
cd /Users/lacg/wnn; L=logs/controller/ctrl48
echo "$(date -u +%FT%TZ) Y3L: armed (pid $$): waits for the Y+F+L s6-s10 waiter (pid $1) to end, then Y3+L s1-s5 (ABI 59)" >> $L/runs.log
while kill -0 $1 2>/dev/null; do sleep 60; done
ABI=59 Y3L_SEEDS="1 2 3 4 5" bash $L/ab_y3l.sh
echo "$(date -u +%FT%TZ) Y3L: driver exited rc=$?" >> $L/runs.log
