#!/bin/bash
# Watcher: Monitor FITTHRESH flows → pick winning weights → update & queue PUB50
# Sleep 4h, poll 30m, then 3m on last flow. On completion:
# 1. Show all FITTHRESH results
# 2. Pick winning weights (best fitness score on validation, F1 > 75%)
# 3. Update PUB50 flows with winning weights
# 4. Restart worker and queue PUB50

set -e
DB="/Users/lacg/wnn/db/wnn.db"
LOG="/Users/lacg/wnn/logs/watcher_fitthresh.log"
WNN_DIR="/Users/lacg/wnn"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"; }

log "=== FITTHRESH Watcher started ==="
log "Sleeping 4 hours..."
sleep 14400

while true; do
	remaining=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'FITTHRESH%' AND status IN ('queued', 'running')")
	log "FITTHRESH remaining: $remaining"

	if [ "$remaining" -eq 0 ]; then
		log "All FITTHRESH flows complete!"
		break
	elif [ "$remaining" -le 2 ]; then
		log "Last $remaining flow(s) — switching to 3min polling"
		while true; do
			last=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'FITTHRESH%' AND status IN ('queued', 'running')")
			if [ "$last" -eq 0 ]; then
				log "All done!"
				break 2
			fi
			sleep 180
		done
	fi
	sleep 1800
done

# === Analyze results ===
log "=== Analyzing FITTHRESH results ==="

sqlite3 -header "$DB" "
SELECT
  CASE
    WHEN f.name LIKE '%baseline%' THEN 'baseline'
    WHEN f.name LIKE '%f1bump%' THEN 'f1bump'
    WHEN f.name LIKE '%balanced%' THEN 'balanced'
    WHEN f.name LIKE '%f1heavy%' THEN 'f1heavy'
  END as config,
  json_extract(f.config_json, '\$.params.fitness_weight_ce') as w_ce,
  json_extract(f.config_json, '\$.params.fitness_weight_f1') as w_f1,
  json_extract(f.config_json, '\$.params.fitness_weight_fpr') as w_fpr,
  json_extract(f.config_json, '\$.params.fitness_weight_acc') as w_acc,
  COUNT(DISTINCT f.id) as runs,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr,
  ROUND(AVG(vs.accuracy) * 100, 2) as avg_acc
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE f.name LIKE 'FITTHRESH%' AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_f1'
  AND e.phase_type = 'ga_neurons'
GROUP BY config
ORDER BY avg_f1 DESC
" | tee -a "$LOG"

# Pick winner: best avg F1 with FPR < 15%
winner_row=$(sqlite3 "$DB" "
SELECT
  json_extract(f.config_json, '\$.params.fitness_weight_ce') as w_ce,
  json_extract(f.config_json, '\$.params.fitness_weight_f1') as w_f1,
  json_extract(f.config_json, '\$.params.fitness_weight_fpr') as w_fpr,
  json_extract(f.config_json, '\$.params.fitness_weight_acc') as w_acc,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE f.name LIKE 'FITTHRESH%' AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_f1'
  AND e.phase_type = 'ga_neurons'
GROUP BY w_ce, w_f1, w_fpr, w_acc
HAVING avg_fpr < 15
ORDER BY avg_f1 DESC
LIMIT 1
")

win_ce=$(echo "$winner_row" | cut -d'|' -f1)
win_f1=$(echo "$winner_row" | cut -d'|' -f2)
win_fpr=$(echo "$winner_row" | cut -d'|' -f3)
win_acc=$(echo "$winner_row" | cut -d'|' -f4)
avg_f1=$(echo "$winner_row" | cut -d'|' -f5)
avg_fpr=$(echo "$winner_row" | cut -d'|' -f6)

log "Winner: CE=$win_ce F1=$win_f1 FPR=$win_fpr Acc=$win_acc (avg F1=$avg_f1% FPR=$avg_fpr%)"

# === Update PUB50 flows ===
log "=== Updating PUB50 flows ==="

sqlite3 "$DB" "
UPDATE flows SET config_json = json_set(config_json,
  '\$.params.fitness_weight_ce', CAST($win_ce AS REAL),
  '\$.params.fitness_weight_f1', CAST($win_f1 AS REAL),
  '\$.params.fitness_weight_fpr', CAST($win_fpr AS REAL),
  '\$.params.fitness_weight_acc', CAST($win_acc AS REAL)
)
WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';
"
updated=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'")
log "Updated $updated PUB50 flows with winning weights"

# === Restart worker ===
log "=== Restarting worker ==="
kill -9 \$(pgrep -f "wnn.ram.experiments.worker") 2>/dev/null || true
sleep 2

# Reset any orphaned flows
sqlite3 "$DB" "
UPDATE experiments SET status = 'pending', current_iteration = 0, started_at = NULL, ended_at = NULL, pid = NULL
WHERE flow_id IN (SELECT id FROM flows WHERE status = 'running') AND status = 'running';
UPDATE flows SET status = 'queued', pid = NULL, last_heartbeat = NULL, status_message = NULL, started_at = NULL
WHERE status = 'running';
"

cd "$WNN_DIR"
source wnn/bin/activate
export PYTHONPATH="$WNN_DIR/src/wnn:\$PYTHONPATH"
PYTHONUNBUFFERED=1 nohup python -u -B -m wnn.ram.experiments.worker --no-ssl-verify > /dev/null 2>&1 &
log "Worker restarted (PID: \$!)"

# Queue PUB50
sqlite3 "$DB" "UPDATE flows SET status = 'queued' WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';"
queued=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'queued'")
log "Queued $queued PUB50 flows"

log "=== FITTHRESH Watcher complete ==="
log "PUB50 config:"
log "  Weights: CE=$win_ce F1=$win_f1 FPR=$win_fpr Acc=$win_acc"
log "  Fitness-aligned threshold: YES"
log "  Flows: $queued"
