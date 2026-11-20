#!/bin/bash
# Plane IDS-70 (Luiz 20/11): once flow 7590 is terminal and 0 flows running/queued → merge origin/ids-cut-method,
# restart the worker (worker_swap, no wheel; Python-only change, ABI stays 29), wait for the new worker, then queue
# the 6 IDS-70 flows via logs/ids/queue_ids70.sh (which itself refuses unless the running worker has the new code).
DB="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"; LOG=/Users/lacg/wnn/logs/ids/deploy_ids70_waiter.log
cd /Users/lacg/wnn
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "armed (pid $$): waiting for 7590 terminal and 0 running/queued flows"
while :; do
  s=$(sqlite3 "$DB" "select status from flows where id=7590;"); n=$(sqlite3 "$DB" "select count(*) from flows where status in ('running','queued');")
  if [[ "$s" =~ ^(completed|failed|cancelled)$ ]] && [ "$n" = 0 ]; then break; fi
  sleep 120
done
say "7590=$s, worker idle — merging origin/ids-cut-method"
git fetch -q origin || { say "ABORT: fetch failed"; exit 1; }
git merge --no-edit origin/ids-cut-method >> $LOG 2>&1 || { git merge --abort 2>/dev/null; say "ABORT: merge failed (no swap, no queue)"; exit 1; }
git push -q origin HEAD >> $LOG 2>&1; say "merged at $(git log --oneline -1)"
old=$(pgrep -f "wnn.ram.experiments.worke[r]" | head -1)
source wnn/bin/activate 2>/dev/null
python scripts/worker_swap.py --auto-detect-running --marker logs/ids/worker_swap_ids70.done >> logs/ids/worker_swap_ids70.log 2>&1
say "worker_swap rc=$? (old worker pid $old)"
for i in $(seq 1 60); do new=$(pgrep -f "wnn.ram.experiments.worke[r]" | head -1); [ -n "$new" ] && [ "$new" != "$old" ] && break; sleep 10; done
[ -n "$new" ] && [ "$new" != "$old" ] || { say "ABORT: no new worker after swap (old $old) — NOT queuing"; exit 1; }
say "new worker pid $new — queuing IDS-70"
bash logs/ids/queue_ids70.sh >> $LOG 2>&1; say "queue_ids70 rc=$?"
