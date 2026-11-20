#!/bin/bash
# Luiz 20/11 ~07:15 EST: Y+F+L (adopted base) s6-s10 BEFORE A2, right after the Y3 block (waiter PID $1) ends.
cd /Users/lacg/wnn; L=logs/controller/ctrl48
echo "$(date -u +%FT%TZ) YLD610: armed (pid $$): waits for the Y3 waiter (pid $1) to end, then Y+F+L s6-s10 (ABI 59)" >> $L/runs.log
while kill -0 $1 2>/dev/null; do sleep 60; done
ABI=59 YLD_SEEDS="6 7 8 9 10" bash $L/ab_yawladder.sh
echo "$(date -u +%FT%TZ) YLD610: driver exited rc=$?" >> $L/runs.log
