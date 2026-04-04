#!/bin/bash
# Watcher 1: Monitor VAL3W flows → rebuild worker → create FEAT flows
# Sleep 3h, poll 30m, then 3m on last flow. On completion:
# 1. Pick winning fitness config from VAL3W results
# 2. Kill worker, rebuild with new code, restart
# 3. Create 10 FEAT flows (5 feature_selections × 2 seeds) with winner
# 4. Queue them all

set -e
DB="/Users/lacg/wnn/db/wnn.db"
LOG="/Users/lacg/wnn/logs/watcher_val3w.log"
WNN_DIR="/Users/lacg/wnn"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG"; }

log "=== VAL3W Watcher started ==="
log "Sleeping 3 hours..."
sleep 10800

while true; do
	remaining=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'VAL3W%' AND status IN ('queued', 'running')")
	log "VAL3W remaining: $remaining"

	if [ "$remaining" -eq 0 ]; then
		log "All VAL3W flows complete!"
		break
	elif [ "$remaining" -eq 1 ]; then
		log "Last flow running — switching to 3min polling"
		while true; do
			last=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'VAL3W%' AND status IN ('queued', 'running')")
			if [ "$last" -eq 0 ]; then
				log "Last VAL3W flow complete!"
				break 2
			fi
			sleep 180
		done
	fi
	sleep 1800
done

# === Pick winning fitness config ===
log "=== Analyzing VAL3W results ==="

# Get the best config by averaging FPR across seeds, requiring F1 > 80%
winner=$(sqlite3 "$DB" "
SELECT
  CASE
    WHEN f.name LIKE '%f1only%' THEN 'f1only'
    WHEN f.name LIKE '%f1dom%' THEN 'f1dom'
    WHEN f.name LIKE '%baseline%' THEN 'baseline'
  END as config,
  json_extract(f.config_json, '\$.params.fitness_weight_ce') as w_ce,
  json_extract(f.config_json, '\$.params.fitness_weight_f1') as w_f1,
  json_extract(f.config_json, '\$.params.fitness_weight_fpr') as w_fpr,
  json_extract(f.config_json, '\$.params.fitness_weight_acc') as w_acc,
  ROUND(AVG(vs.f1_macro) * 100, 2) as avg_f1,
  ROUND(AVG(vs.fpr) * 100, 2) as avg_fpr,
  ROUND(AVG(vs.accuracy) * 100, 2) as avg_acc
FROM validation_summaries vs
JOIN experiments e ON vs.experiment_id = e.id
JOIN flows f ON e.flow_id = f.id
WHERE f.name LIKE 'VAL3W%' AND f.status = 'completed'
  AND vs.validation_point = 'final'
  AND vs.genome_type = 'best_fitness'
  AND e.phase_type = 'ga_neurons'
GROUP BY config
HAVING avg_f1 > 80
ORDER BY avg_fpr ASC
LIMIT 1
")

config_name=$(echo "$winner" | cut -d'|' -f1)
w_ce=$(echo "$winner" | cut -d'|' -f2)
w_f1=$(echo "$winner" | cut -d'|' -f3)
w_fpr=$(echo "$winner" | cut -d'|' -f4)
w_acc=$(echo "$winner" | cut -d'|' -f5)
avg_f1=$(echo "$winner" | cut -d'|' -f6)
avg_fpr=$(echo "$winner" | cut -d'|' -f7)

log "Winner: $config_name (CE=$w_ce F1=$w_f1 FPR=$w_fpr Acc=$w_acc) — avg F1=$avg_f1% FPR=$avg_fpr%"

# === Kill worker, rebuild, restart ===
log "=== Rebuilding worker ==="
kill -9 $(pgrep -f "wnn.ram.experiments.worker") 2>/dev/null || true
sleep 2

# Reset any orphaned running flow
sqlite3 "$DB" "
UPDATE experiments SET status = 'pending', current_iteration = 0, started_at = NULL, ended_at = NULL, pid = NULL
WHERE flow_id IN (SELECT id FROM flows WHERE status = 'running') AND status = 'running';
UPDATE flows SET status = 'queued', pid = NULL, last_heartbeat = NULL, status_message = NULL, started_at = NULL
WHERE status = 'running';
"
log "Reset orphaned flows"

cd "$WNN_DIR"
unset CONDA_PREFIX
source wnn/bin/activate
cd src/wnn/ram/strategies/accelerator
maturin develop --release 2>&1 | tail -3
log "Rust accelerator rebuilt"

cd "$WNN_DIR"
source wnn/bin/activate
export PYTHONPATH="$WNN_DIR/src/wnn:$PYTHONPATH"
PYTHONUNBUFFERED=1 nohup python -u -B -m wnn.ram.experiments.worker --no-ssl-verify > /dev/null 2>&1 &
log "Worker restarted (PID: $!)"
sleep 5

# === Create 10 FEAT flows ===
log "=== Creating FEAT flows ==="

feature_selections=("top15" "top20" "top25" "top20_split" "all")
seeds=(97 96)

for fs in "${feature_selections[@]}"; do
	for seed in "${seeds[@]}"; do
		name="FEAT-ciciot-8b-${config_name}-${fs}-r${seed}"

		result=$(curl -sk -X POST https://localhost:3000/api/flows \
			-H "Content-Type: application/json" \
			-d "{
				\"name\": \"$name\",
				\"description\": \"Feature selection sweep: $fs with $config_name fitness (CE=$w_ce F1=$w_f1 FPR=$w_fpr Acc=$w_acc), seed $seed, 80/10/10\",
				\"config\": {
					\"template\": \"ids-binary-2-phase\",
					\"params\": {
						\"ids_dataset\": \"ciciot2023\",
						\"ids_n_bits\": 8,
						\"ids_split\": \"random_3way\",
						\"ids_feature_selection\": \"$fs\",
						\"ids_classification\": \"binary\",
						\"ids_single_cluster\": true,
						\"ids_val_fraction\": 0.25,
						\"ids_num_parts\": 5,
						\"ids_k_folds\": 5,
						\"ids_kfold_per_gen\": 5,
						\"seed\": $seed,
						\"population_size\": 50,
						\"ga_generations\": 250,
						\"patience\": 5,
						\"min_bits\": 4,
						\"max_bits\": 34,
						\"min_neurons\": 5,
						\"max_neurons\": 500,
						\"neighbors_per_iter\": 50,
						\"phase_order\": \"neurons_first\",
						\"fitness_calculator\": \"harmonic_rank\",
						\"fitness_weight_ce\": $w_ce,
						\"fitness_weight_f1\": $w_f1,
						\"fitness_weight_fpr\": $w_fpr,
						\"fitness_weight_acc\": $w_acc,
						\"fitness_percentile\": 0.75,
						\"neuron_sample_rate\": 0.25,
						\"balance_classes\": true,
						\"architecture_type\": \"ids\",
						\"threshold_start\": 0,
						\"threshold_step\": 1,
						\"min_accuracy_floor\": 0,
						\"assortative_mating_ratio\": 0.85,
						\"cluster_crossover_ratio\": 0.5,
						\"pool_shuffle_ratio\": 0.8,
						\"adaptation_iterations\": 50
					}
				},
				\"experiments\": [
					{\"name\": \"Grid Search (neurons x bits)\", \"phase_type\": \"grid_search\", \"experiment_type\": \"grid_search\"},
					{\"name\": \"GA Neurons\", \"phase_type\": \"ga_neurons\", \"experiment_type\": \"ga\"}
				],
				\"status\": \"pending\"
			}" 2>/dev/null)

		flow_id=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)
		log "Created $name (flow $flow_id)"
	done
done

# Queue all FEAT flows
sqlite3 "$DB" "UPDATE flows SET status = 'queued' WHERE name LIKE 'FEAT-ciciot%' AND status = 'pending';"
queued=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'FEAT-ciciot%' AND status = 'queued'")
log "Queued $queued FEAT flows"
log "=== VAL3W Watcher complete ==="
