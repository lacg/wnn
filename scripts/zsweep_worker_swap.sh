#!/usr/bin/env bash
# One-shot: wait for the worker to go idle, restart it onto the Z_RANK code
# (commit 111e3f06), then queue the IDSZ zscore weight sweep.
#
# WHY A WAITER: the running worker holds the OLD wnn.ram.fitness in memory, so
# the IDSZ flows must NOT be queued until it has been restarted — an old worker
# would ingest fitness_aggregation, not understand it, and silently rank
# harmonically. That is the failure this whole ablation exists to avoid.
#
# The flows are created as `pending`, which the worker never admits
# (_get_next_queued_flow selects status='queued' only), so they are inert until
# this script queues them. Nothing else is touched.
set -u
LOG=/private/tmp/zsweep_swap.log
DB="file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
PY=/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python
IDS_FILE=/Users/lacg/wnn/experiments/idsz_ids.txt
log(){ echo "[zswap] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

log "########## ARMED — waiting for worker idle, then Z_RANK restart + IDSZ queue ##########"

# 1. wait for no running flow (bounded: 3 h)
for _ in $(seq 1 540); do
  R=$(sqlite3 "$DB" "select count(*) from flows where status='running';" 2>/dev/null || echo 1)
  [ "$R" = "0" ] && break
  sleep 20
done
if [ "${R:-1}" != "0" ]; then log "FATAL: still a running flow after 3 h — not touching the worker."; exit 1; fi
log "worker idle (0 running)"

# 2. stop the worker
WPID=$(pgrep -f "wnn.ram.experiments.worker" | head -1)
if [ -n "$WPID" ]; then kill "$WPID"; log "SIGTERM -> worker $WPID"; sleep 8; fi
if pgrep -f "wnn.ram.experiments.worker" >/dev/null; then
  log "worker did not exit on SIGTERM; escalating to SIGKILL"; pkill -9 -f "wnn.ram.experiments.worker"; sleep 3
fi
log "worker stopped"

# 3. relaunch. detach_launch.py takes <logfile> <cwd> BEFORE the -- separator;
#    omitting them is what left the worker down for four hours on 20/08.
"$PY" /Users/lacg/wnn/scripts/detach_launch.py /tmp/wnn_worker.log /Users/lacg/wnn \
  -- "$PY" -u -B -m wnn.ram.experiments.worker \
     --url https://localhost:3000 --no-ssl-verify >> "$LOG" 2>&1
sleep 15
NEW=$(pgrep -f "wnn.ram.experiments.worker" | head -1)
if [ -z "$NEW" ]; then log "FATAL: worker did not come back — IDSZ left pending, nothing queued."; exit 1; fi
log "worker relaunched pid=$NEW"

# 4. prove the new code is the one loaded, on the path a run takes
"$PY" - >> "$LOG" 2>&1 <<'PYCHK'
from wnn.ram.strategies.connectivity.framework.configs import GAConfig
from wnn.ram.fitness import FitnessCalculatorType
c = GAConfig(fitness_calculator_type=FitnessCalculatorType.HARMONIC_RANK,
             fitness_weight_ce=.1, fitness_weight_acc=.2, fitness_weight_f1=.35,
             fitness_weight_fpr=.35, fitness_aggregation="zscore", zrank_clamp=3.0)
calc = c.create_fitness_calculator()
assert calc.aggregation == "zscore", calc.aggregation
print(f"[zswap] verify: {calc.name}  aggregation={calc.aggregation}")
PYCHK
if ! grep -q "aggregation=zscore" "$LOG"; then
  log "FATAL: zscore did not reach the calculator on the run path — IDSZ NOT queued."; exit 1
fi
log "zscore verified on the StrategyConfig path"

# 5. queue the sweep, lowest id first (worker admits min(id))
OK=0; FAIL=0
while read -r i; do
  [ -z "$i" ] && continue
  C=$(curl -sk -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" \
      -d '{"from_beginning":true}' "https://localhost:3000/api/flows/$i/restart")
  if [ "$C" = "200" ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); log "queue FAIL $i -> $C"; fi
done < "$IDS_FILE"
log "IDSZ queued ok=$OK fail=$FAIL"
echo "{\"queued\":$OK,\"failed\":$FAIL,\"worker_pid\":$NEW}" > /tmp/wnn_zsweep_done.json
log "########## DONE ##########"
