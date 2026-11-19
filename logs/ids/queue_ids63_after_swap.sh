#!/bin/bash
# Luiz 18/11: queue IDS63 (fixed 96b, n=3) AFTER the ram_core worker swap, so the swap waiter's "0 queued" gate is not blocked.
# Gate: logs/worker_swap_ctrl58.log has "[swap] DONE" AND a worker process is up. Never touches paused flows.
LOG=/Users/lacg/wnn/logs/ids/queue_ids63_after_swap.log
SWAPLOG=/Users/lacg/wnn/logs/worker_swap_ctrl58.log
cd /Users/lacg/wnn
echo "$(date -u +%FT%TZ) armed: waiting for '[swap] DONE' in $SWAPLOG + worker up" >> $LOG
while :; do
  if grep -q "\[swap\] DONE" "$SWAPLOG" 2>/dev/null && pgrep -f "wnn.ram.experiments.worke[r]" >/dev/null; then break; fi
  sleep 120
done
sleep 60
echo "$(date -u +%FT%TZ) swap done, worker up — queueing IDS63" >> $LOG
unset CONDA_PREFIX; source wnn/bin/activate
python scripts/queue_ids_unswr_fixed_width.py --ids 63 --width 96 >> $LOG 2>&1
echo "$(date -u +%FT%TZ) queue rc=$?" >> $LOG
