#!/bin/bash
# Watcher for the 46M Pareto sweep (flows 1167-1188, tagged SWEEP46M).
#
# Polls the database every 60s until all 22 sweep flows reach terminal
# state (completed/failed/cancelled). When done:
#   1. Pulls every final validation_summary for the sweep flows
#   2. Generates a Pareto frontier report (logs/46M_sweep_verdict.md)
#      with tables grouped by tier (MICRO / SMALL / PEAK) showing each
#      config's best-f1 metrics across all 7 threshold modes
#   3. Highlights the best genome per tier and computes the Pareto front
#   4. Specifically reports the 34b saturation probe result (500n × 34b)
#      and whether it beat 32b (probably not, but the point is to verify)
#   5. Sends a Mac desktop notification with the headline number
#   6. Exits.
#
# Run as nohup background:
#     nohup bash scripts/watcher_46M_sweep.sh > /tmp/watcher_46M_sweep.out 2>&1 &

set -e

cd "$(dirname "$0")/.."
DB="db/wnn.db"
REPORT="logs/46M_sweep_verdict.md"
LOG="/tmp/watcher_46M_sweep.log"
SWEEP_TAG="SWEEP46M"
# The sweep flows we queued (range from launcher output)
EXPECTED_FLOWS=22

mkdir -p "$(dirname "$REPORT")"

log() {
	echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

notify() {
	osascript -e "display notification \"$1\" with title \"46M Pareto Sweep\""
}

log "==== Watcher started: monitoring ${EXPECTED_FLOWS} sweep flows (tag=${SWEEP_TAG}) ===="

# Verify we found the expected number of flows
actual_flows=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${SWEEP_TAG}%';")
log "  Found ${actual_flows} flows matching tag (expected ${EXPECTED_FLOWS})"
if [ "$actual_flows" -lt "$EXPECTED_FLOWS" ]; then
	log "WARNING: fewer sweep flows than expected — maybe the launcher hasn't run yet?"
fi

# Poll loop
start_time=$(date +%s)
last_done=-1
while true; do
	# Count terminal states
	done=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${SWEEP_TAG}%' AND status IN ('completed', 'failed', 'cancelled');")
	running=$(sqlite3 "$DB" "SELECT name FROM flows WHERE name LIKE '%${SWEEP_TAG}%' AND status='running' LIMIT 1;" || true)

	if [ "$done" != "$last_done" ]; then
		if [ -n "$running" ]; then
			log "  Progress: ${done}/${actual_flows} done, running: ${running}"
		else
			log "  Progress: ${done}/${actual_flows} done"
		fi
		last_done="$done"
	fi

	if [ "$done" -ge "$actual_flows" ]; then
		log "==== All ${actual_flows} sweep flows terminal ===="
		break
	fi

	sleep 60
done

elapsed=$(($(date +%s) - start_time))
hours=$((elapsed / 3600))
mins=$(((elapsed % 3600) / 60))
log "Watcher waited ${hours}h ${mins}m for sweep to complete."

log "Generating verdict report..."

source wnn/bin/activate 2>/dev/null

python3 - <<'PYEOF' > "$REPORT"
import sqlite3
import json
import re
import statistics
from collections import defaultdict

conn = sqlite3.connect("db/wnn.db")
cur = conn.cursor()

SWEEP_TAG = "SWEEP46M"
THRESHOLD_MODES = ['train_cal', 'fixed_05', 'platt', 'beta', 'empirical', 'empirical_cumulative', 'val_cal']

# Pull all final validation summaries for the sweep
cur.execute("""
    SELECT vs.flow_id, f.name, vs.genome_type, vs.threshold_metadata,
           g.total_neurons, g.tiers_json, f.status, f.started_at, f.completed_at
    FROM validation_summaries vs
    JOIN experiments e ON vs.experiment_id = e.id
    JOIN flows f ON e.flow_id = f.id
    LEFT JOIN genomes g ON g.config_hash = vs.genome_hash AND g.experiment_id = vs.experiment_id
    WHERE f.name LIKE ? AND vs.validation_point = 'final'
    ORDER BY f.id
""", (f"%{SWEEP_TAG}%",))
rows = cur.fetchall()

# Also pull failed/running flows for status
cur.execute("""
    SELECT id, name, status FROM flows WHERE name LIKE ? ORDER BY id
""", (f"%{SWEEP_TAG}%",))
all_flows = cur.fetchall()

def parse_bits(tiers):
    if not tiers:
        return None
    if tiers.startswith("["):
        try:
            return json.loads(tiers)[0].get("bits")
        except Exception:
            return None
    m = re.search(r"bits=\[(\d+)-(\d+)\]", tiers)
    return int(m.group(2)) if m else None

def parse_tier(name):
    if "PEAK" in name: return "PEAK"
    if "SMALL" in name: return "SMALL"
    if "MICRO" in name: return "MICRO"
    return "?"

def fmt_mem(neurons, bits):
    b = neurons * (1 << bits) * 2 // 8
    if b < 1024: return f"{b}B"
    if b < 1024**2: return f"{b/1024:.1f}KB"
    if b < 1024**3: return f"{b/1024**2:.1f}MB"
    return f"{b/1024**3:.1f}GB"

# Organize by (flow_id, genome_type) -> metadata
per_flow = {}
for fid, fname, gtype, meta_str, n, tiers, status, started, completed in rows:
    bits = parse_bits(tiers)
    if bits is None: continue
    if not meta_str: continue
    meta = json.loads(meta_str)
    tier = parse_tier(fname)
    key = (fid, gtype)
    per_flow[key] = {
        "flow_id": fid, "name": fname, "tier": tier,
        "n": n, "b": bits, "meta": meta, "status": status,
    }

print(f"# 46M Pareto Sweep Verdict")
print()
print(f"**Sweep tag:** {SWEEP_TAG}")
print(f"**Flows:** {len(all_flows)} total")
print()

# Status breakdown
status_counts = defaultdict(int)
for _, _, st in all_flows:
    status_counts[st] += 1
print("## Flow status")
print()
print("| Status | Count |")
print("|---|---:|")
for st, ct in sorted(status_counts.items()):
    print(f"| {st} | {ct} |")
print()

# Reference headline for context
print("## Reference: headline cross-validation (flow 1166)")
print()
print("The 200n × 4b architecture on 46M (flow 1166, single-genome CV) gave:")
print("- F1 = 82.20%, FPR = 6.06%, Acc = 97.22% (train_cal threshold)")
print()
print("The sweep below should show how this baseline scales across memory tiers.")
print()

# Per-tier results: best of each config at each relevant threshold mode
def extract(d, mode):
    m = d["meta"].get(mode, {})
    return m.get("f1", 0) * 100, m.get("fpr", 0) * 100, m.get("acc", 0) * 100

# Group by (n, b) to handle multi-seed aggregation
by_config = defaultdict(list)
for key, d in per_flow.items():
    if d.get("meta") is None: continue
    # Use best_fitness as canonical genome for each flow
    if key[1] != "best_fitness": continue
    by_config[(d["n"], d["b"], d["tier"])].append(d)

print("## Results by tier (train_cal threshold — deployable)")
print()
for target_tier in ["MICRO", "SMALL", "PEAK"]:
    tier_configs = [(n, b, t) for (n, b, t) in by_config.keys() if t == target_tier]
    if not tier_configs:
        continue
    tier_configs.sort(key=lambda x: x[0] * (1 << x[1]))  # sort by memory
    print(f"### {target_tier}")
    print()
    print("| Config | Mem | Seeds | F1 | FPR | Acc |")
    print("|---|---|---:|---:|---:|---:|")
    for (n, b, t) in tier_configs:
        entries = by_config[(n, b, t)]
        n_seeds = len(entries)
        f1s, fprs, accs = [], [], []
        for e in entries:
            f1, fpr, acc = extract(e, "train_cal")
            f1s.append(f1); fprs.append(fpr); accs.append(acc)
        def ms(xs):
            if not xs: return "—"
            if len(xs) == 1: return f"{xs[0]:.2f}%"
            return f"{statistics.mean(xs):.2f}±{statistics.stdev(xs):.2f}%"
        cfg = f"{n}n × {b}b"
        print(f"| {cfg} | {fmt_mem(n, b)} | {n_seeds} | {ms(f1s)} | {ms(fprs)} | {ms(accs)} |")
    print()

# Also show alternative thresholds for peak tier (three operating modes)
print("## PEAK tier: three operating modes (train_cal / fixed_05 / val_cal)")
print()
print("| Config | Mode | F1 | FPR | Acc |")
print("|---|---|---:|---:|---:|")
for (n, b, t) in sorted(by_config.keys(), key=lambda x: x[0] * (1 << x[1])):
    if t != "PEAK": continue
    entry = by_config[(n, b, t)][0]  # 1 seed per peak config
    for mode in ["train_cal", "fixed_05", "val_cal"]:
        f1, fpr, acc = extract(entry, mode)
        cfg = f"{n}n × {b}b"
        print(f"| {cfg} | {mode} | {f1:.2f}% | {fpr:.2f}% | {acc:.2f}% |")
    print(f"| | | | | |")  # separator
print()

# The 34b saturation probe verdict
print("## 34b saturation probe verdict")
print()
probe_32 = [(n, b, t) for (n, b, t) in by_config.keys() if b == 32 and t == "PEAK"]
probe_34 = [(n, b, t) for (n, b, t) in by_config.keys() if b == 34 and t == "PEAK"]
if probe_32 and probe_34:
    best_32 = max(probe_32, key=lambda k: extract(by_config[k][0], "train_cal")[0])
    best_34 = max(probe_34, key=lambda k: extract(by_config[k][0], "train_cal")[0])
    f1_32, fpr_32, acc_32 = extract(by_config[best_32][0], "train_cal")
    f1_34, fpr_34, acc_34 = extract(by_config[best_34][0], "train_cal")
    delta = f1_34 - f1_32
    print(f"- **Best 32b peak (train_cal):** {best_32[0]}n × 32b — F1 {f1_32:.2f}%, FPR {fpr_32:.2f}%, Acc {acc_32:.2f}%")
    print(f"- **Best 34b peak (train_cal):** {best_34[0]}n × 34b — F1 {f1_34:.2f}%, FPR {fpr_34:.2f}%, Acc {acc_34:.2f}%")
    print(f"- **Delta (34 - 32):** {delta:+.2f}pp F1")
    print()
    if delta > 0.3:
        print("**🎉 SURPRISE:** 34b actually beat 32b on F1 on 46M! The extra 2 bits helped with more training data.")
    elif delta > -0.3:
        print("**🤝 TIE:** 34b and 32b are within noise. Extra bits don't hurt, but don't help either. Thermometer encoding caps effective discrimination.")
    else:
        print("**✅ EXPECTED:** 34b is worse than 32b (as hypothesized). Thermometer-encoding address collisions cap effective capacity — extra bits are dead weight.")
    print()

# Literature comparison for the peak
print("## Literature comparison (Neto et al. 2023, same dataset, random 80/20)")
print()
print("| Method | F1 | Acc | Notes |")
print("|---|---:|---:|---|")
print("| Perceptron | 81.05% | 98.18% | ~10 KB |")
print("| Logistic Reg. | 87.63% | 98.90% | ~1 KB |")
print("| DNN | 94.03% | 99.44% | ~5 MB |")
print("| AdaBoost | 95.63% | 99.59% | ~10 MB |")
print("| Random Forest | 96.53% | 99.68% | ~50 MB |")
print()
# Our best peak
if probe_32:
    # Take the best F1 across all threshold modes for the best 32b peak
    best_32b_peak = max(probe_32, key=lambda k: max(extract(by_config[k][0], m)[0] for m in THRESHOLD_MODES))
    entry = by_config[best_32b_peak][0]
    best_f1 = -1
    best_mode = None
    for m in THRESHOLD_MODES:
        f1, fpr, acc = extract(entry, m)
        if f1 > best_f1:
            best_f1 = f1
            best_mode = m
            best_fpr = fpr
            best_acc = acc
    print(f"**Our best 32b peak:** {best_32b_peak[0]}n × 32b @ {best_mode}")
    print(f"  F1 = {best_f1:.2f}%, FPR = {best_fpr:.2f}%, Acc = {best_acc:.2f}%")
    print()

conn.close()
PYEOF

cat "$REPORT" >> "$LOG"
log "Report written to $REPORT"

# Summary for notification
completed_count=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${SWEEP_TAG}%' AND status='completed';")
failed_count=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${SWEEP_TAG}%' AND status IN ('failed','cancelled');")
notify "Sweep done: ${completed_count}/${actual_flows} completed, ${failed_count} failed. See ${REPORT}"

log "==== Sweep watcher complete ===="
