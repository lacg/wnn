#!/bin/bash
# Luiz 19/11 ~14:15 EST: dense yaw ladder A/B on the Y+F base, right after the F-on-Y block (driver PID $1) ends.
cd /Users/lacg/wnn; L=logs/controller/ctrl48
echo "$(date -u +%FT%TZ) YLD: armed (pid $$): waits for the F-on-Y driver (pid $1) to end, then arm L (Y+F + dense yaw ladder) s1-s5 (ABI 59)" >> $L/runs.log
while kill -0 $1 2>/dev/null; do sleep 60; done
ABI=59 YLD_SEEDS="1 2 3 4 5" bash $L/ab_yawladder.sh
echo "$(date -u +%FT%TZ) YLD: driver exited rc=$?" >> $L/runs.log
