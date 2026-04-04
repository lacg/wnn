#!/bin/bash
# Watcher 2: Monitor FEAT flows → pick winner → update & queue thermometer/PUB50 flows
# Sleep 8h, poll 30m, then 3m on last flow. On completion:
# 1. Pick winning feature_selection + fitness config from FEAT results
# 2. Update PUB50 pending flows with winning config
# 3. Verify random_3way (80/10/10)
# 4. Queue PUB50 flows

set -e
DB="/Users/lacg/wnn/db/wnn.db"
LOG="/Users/lacg/wnn/logs/watcher_feat.log"
WNN_DIR="/Users/lacg/wnn"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"; }

log "=== FEAT Watcher started ==="
log "Sleeping 8 hours..."
sleep 28800

while true; do
	remaining=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'FEAT-ciciot%' AND status IN ('queued', 'running')")
	log "FEAT remaining: $remaining"

	if [ "$remaining" -eq 0 ]; then
		log "All FEAT flows complete!"
		break
	elif [ "$remaining" -eq 1 ]; then
		log "Last flow running — switching to 3min polling"
		while true; do
			last=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'FEAT-ciciot%' AND status IN ('queued', 'running')")
			if [ "$last" -eq 0 ]; then
				log "Last FEAT flow complete!"
				break 2
			fi
			sleep 180
		done
	fi
	sleep 1800
done

# === Analyze FEAT results ===
log "=== Analyzing FEAT results ==="

# Show all results
sqlite3 -header "$DB" "
SELECT f.name,
  json_extract(f.config_json, '\$.params.ids_feature_selection') as features,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr,
  ROUND(AVG(vs.accuracy) * 100, 2) as avg_acc
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE f.name LIKE 'FEAT-ciciot%' AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_fitness'
  AND e.phase_type = 'ga_neurons'
GROUP BY features
ORDER BY avg_fpr ASC
" | tee -a "$LOG"

# Pick winning feature_selection: best avg FPR with F1 > 80%
winner_row=$(sqlite3 "$DB" "
SELECT
  json_extract(f.config_json, '\$.params.ids_feature_selection') as features,
  json_extract(f.config_json, '\$.params.fitness_weight_ce') as w_ce,
  json_extract(f.config_json, '\$.params.fitness_weight_f1') as w_f1,
  json_extract(f.config_json, '\$.params.fitness_weight_fpr') as w_fpr,
  json_extract(f.config_json, '\$.params.fitness_weight_acc') as w_acc,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE f.name LIKE 'FEAT-ciciot%' AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_fitness'
  AND e.phase_type = 'ga_neurons'
GROUP BY features
HAVING avg_f1 > 80
ORDER BY avg_fpr ASC
LIMIT 1
")

win_features=$(echo "$winner_row" | cut -d'|' -f1)
win_ce=$(echo "$winner_row" | cut -d'|' -f2)
win_f1=$(echo "$winner_row" | cut -d'|' -f3)
win_fpr=$(echo "$winner_row" | cut -d'|' -f4)
win_acc=$(echo "$winner_row" | cut -d'|' -f5)
win_avg_f1=$(echo "$winner_row" | cut -d'|' -f6)
win_avg_fpr=$(echo "$winner_row" | cut -d'|' -f7)

log "Winner: features=$win_features fitness=CE$win_ce/F1$win_f1/FPR$win_fpr/Acc$win_acc — avg F1=$win_avg_f1% FPR=$win_avg_fpr%"

# === Update PUB50 flows with winning config ===
log "=== Updating PUB50 flows ==="

# Count pending PUB50 flows
pending=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'")
log "Found $pending pending PUB50 flows"

# Update fitness weights
sqlite3 "$DB" "
UPDATE flows SET config_json = json_set(config_json,
  '\$.params.fitness_weight_ce', CAST($win_ce AS REAL),
  '\$.params.fitness_weight_f1', CAST($win_f1 AS REAL),
  '\$.params.fitness_weight_fpr', CAST($win_fpr AS REAL),
  '\$.params.fitness_weight_acc', CAST($win_acc AS REAL),
  '\$.params.ids_feature_selection', '$win_features'
)
WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';
"
log "Updated fitness weights and feature selection"

# Verify random_3way
r3w_count=$(sqlite3 "$DB" "
SELECT COUNT(*) FROM flows
WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'
  AND json_extract(config_json, '\$.params.ids_split') = 'random_3way'
")
total_pending=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'")

if [ "$r3w_count" -eq "$total_pending" ]; then
	log "Verified: all $r3w_count PUB50 flows use random_3way (80/10/10) ✓"
else
	log "WARNING: Only $r3w_count/$total_pending use random_3way — fixing..."
	sqlite3 "$DB" "
	UPDATE flows SET config_json = json_set(config_json, '\$.params.ids_split', 'random_3way')
	WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'
	  AND json_extract(config_json, '\$.params.ids_split') != 'random_3way';
	"
	log "Fixed all PUB50 flows to random_3way"
fi

# Queue PUB50 flows
sqlite3 "$DB" "UPDATE flows SET status = 'queued' WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';"
queued=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'queued'")
log "Queued $queued PUB50 flows"

log "=== FEAT Watcher complete ==="
log "PUB50 batch is now running with:"
log "  Feature selection: $win_features"
log "  Fitness: CE=$win_ce F1=$win_f1 FPR=$win_fpr Acc=$win_acc"
log "  Split: random_3way (80/10/10)"
log "  Flows: $queued"
