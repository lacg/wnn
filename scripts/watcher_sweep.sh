#!/bin/bash
# Watcher: Monitor thermometer SWEEP flows → pick winner → update & queue PUB50
# Sleep 4h, poll 30m, then 3m on last flow. On completion:
# 1. Show results table (F1, FPR, Acc by thermometer bits)
# 2. Pick winning thermometer encoding
# 3. Update PUB50 with winning ids_n_bits
# 4. Queue PUB50 flows

set -e
DB="/Users/lacg/wnn/db/wnn.db"
LOG="/Users/lacg/wnn/logs/watcher_sweep.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"; }

log "=== Thermometer SWEEP Watcher started ==="
log "Sleeping 4 hours..."
sleep 14400

while true; do
	remaining=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'SWEEP-ciciot%' AND status IN ('queued', 'running')")
	log "SWEEP remaining: $remaining"

	if [ "$remaining" -eq 0 ]; then
		log "All SWEEP flows complete!"
		break
	elif [ "$remaining" -le 2 ]; then
		log "Last $remaining flow(s) running — switching to 3min polling"
		while true; do
			last=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'SWEEP-ciciot%' AND status IN ('queued', 'running')")
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
log "=== Analyzing thermometer SWEEP results ==="

sqlite3 -header "$DB" "
SELECT
  json_extract(f.config_json, '\$.params.ids_n_bits') as therm_bits,
  COUNT(DISTINCT f.id) as runs,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr,
  ROUND(AVG(vs.accuracy) * 100, 2) as avg_acc,
  ROUND(AVG((julianday(f.completed_at) - julianday(f.started_at)) * 24 * 60), 0) as avg_min
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE f.name LIKE 'SWEEP-ciciot%' AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_fitness'
  AND e.phase_type = 'ga_neurons'
GROUP BY therm_bits
ORDER BY avg_fpr ASC
" | tee -a "$LOG"

# Pick winner: best avg FPR with F1 > 80%
win_bits=$(sqlite3 "$DB" "
SELECT json_extract(f.config_json, '\$.params.ids_n_bits') as therm_bits
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE f.name LIKE 'SWEEP-ciciot%' AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_fitness'
  AND e.phase_type = 'ga_neurons'
GROUP BY therm_bits
HAVING ROUND(AVG(vs.f1_macro) * 100, 2) > 80
ORDER BY AVG(vs.fpr) ASC
LIMIT 1
")

log "Winner: thermometer bits = $win_bits"

# === Update PUB50 flows ===
log "=== Updating PUB50 flows ==="

sqlite3 "$DB" "
UPDATE flows SET config_json = json_set(config_json,
  '\$.params.ids_n_bits', CAST($win_bits AS INTEGER)
)
WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';
"
updated=$(sqlite3 "$DB" "SELECT changes()")
log "Updated $updated PUB50 flows to ids_n_bits=$win_bits"

# Verify full config
sqlite3 "$DB" "
SELECT
  json_extract(config_json, '\$.params.ids_n_bits') as therm,
  json_extract(config_json, '\$.params.max_bits') as max_bits,
  json_extract(config_json, '\$.params.ids_split') as split,
  json_extract(config_json, '\$.params.ids_feature_selection') as features,
  json_extract(config_json, '\$.params.fitness_weight_ce') || '/' ||
  json_extract(config_json, '\$.params.fitness_weight_f1') || '/' ||
  json_extract(config_json, '\$.params.fitness_weight_fpr') || '/' ||
  json_extract(config_json, '\$.params.fitness_weight_acc') as weights,
  COUNT(*) as cnt
FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'
GROUP BY therm, max_bits, split, features, weights
" | tee -a "$LOG"

# Queue PUB50
sqlite3 "$DB" "UPDATE flows SET status = 'queued' WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';"
queued=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'queued'")
log "Queued $queued PUB50 flows"

log "=== Thermometer SWEEP Watcher complete ==="
log "PUB50 batch config:"
log "  thermometer: ${win_bits}b per feature"
log "  max_bits: 34"
log "  fitness: baseline (0.2/0.3/0.4/0.1)"
log "  features: top20"
log "  split: random_3way (80/10/10)"
log "  flows: $queued"
