#!/bin/bash
# Luiz 19/11 ~18:40 EST: Y3 (--delta-max-yaw 0.3, no F) right after the dense yaw ladder block (waiter PID $1) ends.
cd /Users/lacg/wnn; L=logs/controller/ctrl48
echo "$(date -u +%FT%TZ) DY3: armed (pid $$): waits for the dense-ladder waiter (pid $1) to end, then arm Y3 (--delta-max-yaw 0.3, no F) s1-s5 (ABI 59)" >> $L/runs.log
while kill -0 $1 2>/dev/null; do sleep 60; done
ABI=59 DY3_SEEDS="1 2 3 4 5" bash $L/ab_dmaxyaw03.sh
echo "$(date -u +%FT%TZ) DY3: driver exited rc=$?" >> $L/runs.log
