#!/bin/bash
# Watch CIC-IoT fitness sweep. When done, analyze top-3 configs and
# create feature selection flows (top-15, top-20, top-25, all-46) for each.

DB="/Users/lacg/wnn/db/wnn.db"
LOG="/tmp/watch_fitness.log"

echo "$(date): Starting fitness sweep watcher" >> "$LOG"

while true; do
    REMAINING=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'FIT-ciciot%' AND status IN ('queued','running');")
    COMPLETED=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'FIT-ciciot%' AND status = 'completed';")

    echo "$(date): FIT-ciciot: $COMPLETED completed, $REMAINING remaining" >> "$LOG"

    if [ "$REMAINING" -eq 0 ]; then
        echo "$(date): All fitness sweep flows done! Analyzing results..." >> "$LOG"

        # Get top-3 configs by best_fitness val_cal F1 (averaged across 2 seeds)
        TOP3=$(sqlite3 "$DB" "
            SELECT label, ROUND(AVG(f1) * 100, 2) as avg_f1, ROUND(AVG(fpr) * 100, 2) as avg_fpr,
                   ROUND(AVG(acc) * 100, 2) as avg_acc,
                   AVG(ce_w) as ce, AVG(f1_w) as f1w, AVG(fpr_w) as fprw, AVG(acc_w) as accw
            FROM (
                SELECT
                    REPLACE(REPLACE(REPLACE(f.name, 'FIT-ciciot-8b-', ''), '-r99', ''), '-r98', '') as label,
                    json_extract(vs.threshold_metadata, '$.val_cal.f1') as f1,
                    json_extract(vs.threshold_metadata, '$.val_cal.fpr') as fpr,
                    json_extract(vs.threshold_metadata, '$.val_cal.acc') as acc,
                    json_extract(f.config_json, '$.params.fitness_weight_ce') as ce_w,
                    json_extract(f.config_json, '$.params.fitness_weight_f1') as f1_w,
                    json_extract(f.config_json, '$.params.fitness_weight_fpr') as fpr_w,
                    json_extract(f.config_json, '$.params.fitness_weight_acc') as acc_w
                FROM flows f
                JOIN experiments e ON e.flow_id = f.id AND e.name = 'GA Neurons'
                JOIN validation_summaries vs ON vs.experiment_id = e.id
                    AND vs.validation_point = 'final' AND vs.genome_type = 'best_fitness'
                WHERE f.name LIKE 'FIT-ciciot%' AND f.status = 'completed'
            )
            GROUP BY label
            ORDER BY avg_f1 DESC
            LIMIT 3;
        ")

        echo "$(date): Top-3 fitness configs:" >> "$LOG"
        echo "$TOP3" >> "$LOG"

        # Also check best_acc genomes (the F1045 99.24% genome was best_acc)
        BEST_ACC=$(sqlite3 "$DB" "
            SELECT label, ROUND(AVG(f1) * 100, 2) as avg_f1, ROUND(AVG(fpr) * 100, 2) as avg_fpr
            FROM (
                SELECT
                    REPLACE(REPLACE(REPLACE(f.name, 'FIT-ciciot-8b-', ''), '-r99', ''), '-r98', '') as label,
                    json_extract(vs.threshold_metadata, '$.val_cal.f1') as f1,
                    json_extract(vs.threshold_metadata, '$.val_cal.fpr') as fpr
                FROM flows f
                JOIN experiments e ON e.flow_id = f.id AND e.name = 'GA Neurons'
                JOIN validation_summaries vs ON vs.experiment_id = e.id
                    AND vs.validation_point = 'final' AND vs.genome_type = 'best_acc'
                WHERE f.name LIKE 'FIT-ciciot%' AND f.status = 'completed'
            )
            GROUP BY label
            ORDER BY avg_f1 DESC
            LIMIT 3;
        ")

        echo "$(date): Top-3 by best_acc genome:" >> "$LOG"
        echo "$BEST_ACC" >> "$LOG"

        # Create feature selection flows for each top-3 config
        # Read top-3 labels and their weights
        echo "$TOP3" | while IFS='|' read -r label avg_f1 avg_fpr avg_acc ce f1w fprw accw; do
            echo "$(date): Creating feature flows for config: $label (CE=$ce, F1=$f1w, FPR=$fprw, Acc=$accw, avg_f1=$avg_f1%)" >> "$LOG"

            for feat_sel in "top15" "top20" "top25" "all"; do
                for seed in 99 98; do
                    FLOW_NAME="FEAT-ciciot-8b-${label}-${feat_sel}-r${seed}"

                    curl -sk -X POST "https://localhost:3000/api/flows" \
                        -H "Content-Type: application/json" \
                        -d "{
                            \"name\": \"${FLOW_NAME}\",
                            \"description\": \"CIC-IoT feature sweep: ${label} + ${feat_sel}, 8b therm, seed ${seed}\",
                            \"status\": \"queued\",
                            \"config\": {
                                \"template\": \"ids-binary-2-phase\",
                                \"params\": {
                                    \"ids_classification\": \"binary\",
                                    \"ids_val_fraction\": 0.25,
                                    \"threshold_step\": 1,
                                    \"adaptation_iterations\": 50,
                                    \"ids_num_parts\": 5,
                                    \"ids_k_folds\": 5,
                                    \"cluster_crossover_ratio\": 0.5,
                                    \"context_size\": 4,
                                    \"fitness_percentile\": 0.75,
                                    \"fitness_weight_acc\": ${accw},
                                    \"assortative_mating_ratio\": 0.85,
                                    \"ga_generations\": 250,
                                    \"patience\": 5,
                                    \"max_neurons\": 500,
                                    \"max_bits\": 34,
                                    \"phase_order\": \"neurons_first\",
                                    \"neuron_sample_rate\": 0.25,
                                    \"min_neurons\": 5,
                                    \"fitness_calculator\": \"harmonic_rank\",
                                    \"threshold_start\": 0,
                                    \"ids_n_bits\": 8,
                                    \"fitness_weight_ce\": ${ce},
                                    \"fitness_weight_f1\": ${f1w},
                                    \"ids_kfold_per_gen\": 5,
                                    \"ids_split\": \"random\",
                                    \"fitness_weight_fpr\": ${fprw},
                                    \"min_accuracy_floor\": 0,
                                    \"balance_classes\": true,
                                    \"min_bits\": 4,
                                    \"neighbors_per_iter\": 50,
                                    \"population_size\": 50,
                                    \"ids_dataset\": \"ciciot2023\",
                                    \"ids_feature_selection\": \"${feat_sel}\",
                                    \"architecture_type\": \"ids\",
                                    \"pool_shuffle_ratio\": 0.8,
                                    \"ids_single_cluster\": true,
                                    \"seed\": ${seed}
                                }
                            },
                            \"experiments\": [
                                {\"name\": \"Grid Search (neurons x bits)\", \"phase_type\": \"grid_search\", \"experiment_type\": \"grid_search\"},
                                {\"name\": \"GA Neurons\", \"phase_type\": \"ga_neurons\", \"experiment_type\": \"ga\"}
                            ]
                        }" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Created F{d[\"id\"]}: {d[\"name\"]}')" >> "$LOG" 2>&1
                done
            done
        done

        CREATED=$(grep -c "Created F" "$LOG")
        echo "$(date): Created $CREATED feature selection flows. Done!" >> "$LOG"

        break
    fi

    sleep 1800  # Check every 30 minutes
done
