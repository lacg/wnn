#!/bin/bash
# Watcher for the 46M cross-validation flow (EVAL46M-46M-200n4b-s42-CV-pipeline).
#
# Polls the database every 30s. When flow 1165 reaches status='completed':
#   1. Reads its validation_summaries (final metrics across all 7 thresholds)
#   2. Compares against the headline 200n×4b on 46M result:
#        F1=82.33%, FPR=6.67%, Acc=97.27% (fitness-aligned threshold)
#   3. Writes a verdict report to logs/46M_cv_verdict.md
#   4. Sends a Mac desktop notification with the headline numbers
#   5. Exits.
#
# After this, the worker will naturally pick up PUB50 next (highest queued ID).
#
# Run from project root:
#     nohup bash scripts/watcher_46M_cv.sh > /tmp/watcher_46M_cv.out 2>&1 &

set -e

cd "$(dirname "$0")/.."
DB="db/wnn.db"
TARGET_FLOW=1165
REPORT="logs/46M_cv_verdict.md"
LOG="/tmp/watcher_46M_cv.log"

mkdir -p "$(dirname "$REPORT")"

log() {
	echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

notify() {
	osascript -e "display notification \"$1\" with title \"46M Cross-Validation\""
}

log "==== Watcher started: monitoring flow $TARGET_FLOW ===="
log "Reference headline: 200n × 4b on 46M = F1 82.33% / FPR 6.67% / Acc 97.27%"

# Verify the target flow exists
exists=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE id=$TARGET_FLOW;")
if [ "$exists" != "1" ]; then
	log "ERROR: flow $TARGET_FLOW not found in database. Exiting."
	exit 1
fi

# Poll loop
start_time=$(date +%s)
last_status=""
while true; do
	status=$(sqlite3 "$DB" "SELECT status FROM flows WHERE id=$TARGET_FLOW;")

	if [ "$status" != "$last_status" ]; then
		log "  Flow $TARGET_FLOW status: $status"
		last_status="$status"
	fi

	if [ "$status" = "completed" ]; then
		log "==== Flow $TARGET_FLOW COMPLETED ===="
		break
	elif [ "$status" = "failed" ] || [ "$status" = "cancelled" ]; then
		log "ERROR: flow $TARGET_FLOW ended with status='$status'. Exiting."
		notify "Cross-validation FAILED (status=$status). Check logs."
		exit 1
	fi

	sleep 30
done

elapsed=$(($(date +%s) - start_time))
log "Watcher waited ${elapsed}s for flow to complete."

log "Generating verdict report..."

source wnn/bin/activate 2>/dev/null

python3 - <<'PYEOF' > "$REPORT"
import sqlite3
import json

conn = sqlite3.connect("db/wnn.db")
cur = conn.cursor()

TARGET = 1165
HEADLINE = {
    "F1":  82.33,
    "FPR": 6.67,
    "Acc": 97.27,
}

cur.execute("""
    SELECT vs.genome_type, vs.f1_macro, vs.fpr, vs.accuracy, vs.threshold_metadata
    FROM validation_summaries vs
    JOIN experiments e ON vs.experiment_id = e.id
    WHERE e.flow_id = ? AND vs.validation_point = 'final'
    ORDER BY vs.genome_type
""", (TARGET,))
rows = cur.fetchall()

print("# 46M Cross-Validation Verdict")
print()
print(f"**Flow:** {TARGET} -- EVAL46M-46M-200n4b-s42-CV-pipeline")
print(f"**Architecture:** 200 neurons x 4 bits (800 bytes)")
print(f"**Dataset:** CIC-IoT-2023 full 46M (random 80/20)")
print()
print("## Reference headline (from project_200n4b_46M_result.md, 2026-04-05)")
print()
print("| Metric | Headline |")
print("|---|---|")
for k, v in HEADLINE.items():
    print(f"| {k} | {v}% |")
print()
print("## Cross-validation results (best_* genomes)")
print()
print("| Genome | F1 | FPR | Acc | delta F1 vs headline |")
print("|---|---:|---:|---:|---:|")

best_results = {}
for gtype, f1, fpr, acc, meta in rows:
    f1_pct = (f1 or 0) * 100
    fpr_pct = (fpr or 0) * 100
    acc_pct = (acc or 0) * 100
    delta_f1 = f1_pct - HEADLINE["F1"]
    sign = "+" if delta_f1 >= 0 else ""
    print(f"| {gtype} | {f1_pct:.2f}% | {fpr_pct:.2f}% | {acc_pct:.2f}% | {sign}{delta_f1:.2f}pp |")
    best_results[gtype] = (f1_pct, fpr_pct, acc_pct)

print()
print("## Threshold mode breakdown (best_fitness genome)")
print()

cur.execute("""
    SELECT vs.threshold_metadata
    FROM validation_summaries vs
    JOIN experiments e ON vs.experiment_id = e.id
    WHERE e.flow_id = ? AND vs.validation_point = 'final' AND vs.genome_type = 'best_fitness'
""", (TARGET,))
row = cur.fetchone()
if row and row[0]:
    meta = json.loads(row[0])
    THRESHOLD_MODES = ['train_cal', 'fixed_05', 'platt', 'beta', 'empirical', 'empirical_cumulative', 'val_cal']
    print("| Threshold | F1 | FPR | Acc |")
    print("|---|---:|---:|---:|")
    for mode in THRESHOLD_MODES:
        if mode in meta:
            m = meta[mode]
            f1_pct = m.get("f1", 0) * 100
            fpr_pct = m.get("fpr", 0) * 100
            acc_pct = m.get("acc", 0) * 100
            print(f"| {mode} | {f1_pct:.2f}% | {fpr_pct:.2f}% | {acc_pct:.2f}% |")
    print()

print("## Verdict")
print()
if "best_fitness" in best_results:
    f1, fpr, acc = best_results["best_fitness"]
    delta_f1 = abs(f1 - HEADLINE["F1"])
    if delta_f1 < 1.0:
        verdict = "MATCH"
        commentary = f"best_fitness F1 ({f1:.2f}%) is within 1pp of the headline ({HEADLINE['F1']}%). Pipeline cross-validates successfully."
    elif delta_f1 < 3.0:
        verdict = "CLOSE"
        commentary = f"best_fitness F1 ({f1:.2f}%) is within 3pp of the headline ({HEADLINE['F1']}%). Likely OK but worth investigating."
    else:
        verdict = "MISMATCH"
        commentary = f"best_fitness F1 ({f1:.2f}%) differs by {delta_f1:.2f}pp from the headline ({HEADLINE['F1']}%). Pipeline needs investigation."
    print(f"**{verdict}**")
    print()
    print(commentary)
else:
    print("WARNING: No best_fitness summary found -- incomplete results.")

conn.close()
PYEOF

cat "$REPORT" >> "$LOG"
log "Report written to $REPORT"

verdict_line=$(grep -E "^\*\*(MATCH|CLOSE|MISMATCH)" "$REPORT" | head -1 | tr -d '*')
notify "Flow $TARGET_FLOW done. $verdict_line"

log "==== Watcher complete ===="
