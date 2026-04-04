#!/bin/bash
# Watcher: Monitor BITS + USAMP bit-sweep flows → pick winner → update & queue PUB50
# Sleep 2h, poll 30m, then 3m on last flow. On completion:
# 1. Pick winning max_bits + data strategy (regular vs USAMP)
# 2. Update PUB50 pending flows with winning config
# 3. Verify random_3way, baseline weights, top20 features
# 4. Queue PUB50 flows

set -e
DB="/Users/lacg/wnn/db/wnn.db"
LOG="/Users/lacg/wnn/logs/watcher_bits.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"; }

log "=== BITS/USAMP Watcher started ==="
log "Sleeping 2 hours..."
sleep 7200

while true; do
	remaining=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE (name LIKE 'BITS-ciciot%' OR name LIKE 'USAMP-ciciot-8b-f1dom%' OR name LIKE 'USAMP-ciciot-8b-baseline%') AND status IN ('queued', 'running')")
	log "BITS/USAMP remaining: $remaining"

	if [ "$remaining" -eq 0 ]; then
		log "All BITS/USAMP flows complete!"
		break
	elif [ "$remaining" -le 2 ]; then
		log "Last $remaining flow(s) running — switching to 3min polling"
		while true; do
			last=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE (name LIKE 'BITS-ciciot%' OR name LIKE 'USAMP-ciciot-8b-f1dom%' OR name LIKE 'USAMP-ciciot-8b-baseline%') AND status IN ('queued', 'running')")
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
log "=== Analyzing BITS/USAMP results ==="

# Show all results grouped by max_bits and data strategy
sqlite3 -header "$DB" "
SELECT
  CASE WHEN json_extract(f.config_json, '\$.params.undersample_majority') = 1 THEN 'USAMP' ELSE 'Regular' END as strategy,
  json_extract(f.config_json, '\$.params.max_bits') as max_bits,
  COUNT(DISTINCT f.id) as runs,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr,
  ROUND(AVG(vs.accuracy) * 100, 2) as avg_acc
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE (f.name LIKE 'BITS-ciciot%' OR f.name LIKE 'USAMP-ciciot-8b-f1dom%' OR f.name LIKE 'USAMP-ciciot-8b-baseline%')
  AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_fitness'
  AND e.phase_type = 'ga_neurons'
GROUP BY strategy, max_bits
ORDER BY avg_fpr ASC
" | tee -a "$LOG"

# Pick winner: best avg FPR with F1 > 80%
winner_row=$(sqlite3 "$DB" "
SELECT
  CASE WHEN json_extract(f.config_json, '\$.params.undersample_majority') = 1 THEN 'true' ELSE 'false' END as usamp,
  json_extract(f.config_json, '\$.params.max_bits') as max_bits,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE (f.name LIKE 'BITS-ciciot%' OR f.name LIKE 'USAMP-ciciot-8b-f1dom%' OR f.name LIKE 'USAMP-ciciot-8b-baseline%')
  AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_fitness'
  AND e.phase_type = 'ga_neurons'
GROUP BY usamp, max_bits
HAVING avg_f1 > 80
ORDER BY avg_fpr ASC
LIMIT 1
")

win_usamp=$(echo "$winner_row" | cut -d'|' -f1)
win_max_bits=$(echo "$winner_row" | cut -d'|' -f2)
win_avg_f1=$(echo "$winner_row" | cut -d'|' -f3)
win_avg_fpr=$(echo "$winner_row" | cut -d'|' -f4)

log "Winner: max_bits=$win_max_bits, usamp=$win_usamp — avg F1=$win_avg_f1% FPR=$win_avg_fpr%"

# === Update PUB50 flows ===
log "=== Updating PUB50 flows ==="

pending=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'")
log "Found $pending pending PUB50 flows"

# Update max_bits, undersample, and ensure baseline weights + top20 + random_3way
sqlite3 "$DB" "
UPDATE flows SET config_json = json_set(config_json,
  '\$.params.max_bits', CAST($win_max_bits AS INTEGER),
  '\$.params.undersample_majority', CASE WHEN '$win_usamp' = 'true' THEN json('true') ELSE json('false') END,
  '\$.params.fitness_weight_ce', 0.2,
  '\$.params.fitness_weight_f1', 0.3,
  '\$.params.fitness_weight_fpr', 0.4,
  '\$.params.fitness_weight_acc', 0.1,
  '\$.params.ids_feature_selection', 'top20',
  '\$.params.ids_split', 'random_3way'
)
WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';
"
log "Updated PUB50: max_bits=$win_max_bits, usamp=$win_usamp, baseline weights, top20, random_3way"

# Verify
r3w=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending' AND json_extract(config_json, '\$.params.ids_split') = 'random_3way'")
total=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending'")
log "Verified: $r3w/$total PUB50 flows on random_3way"

# Queue PUB50
sqlite3 "$DB" "UPDATE flows SET status = 'queued' WHERE name LIKE 'PUB50-ciciot%' AND status = 'pending';"
queued=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot%' AND status = 'queued'")
log "Queued $queued PUB50 flows"

log "=== BITS/USAMP Watcher complete ==="
log "PUB50 batch config:"
log "  max_bits: $win_max_bits"
log "  undersample: $win_usamp"
log "  fitness: baseline (0.2/0.3/0.4/0.1)"
log "  features: top20"
log "  split: random_3way (80/10/10)"
log "  flows: $queued"
