#!/bin/bash
# Watcher for the 46M thermometer encoding sweep (12 flows tagged THERMO46M).
#
# Polls every 60s until all THERMO46M flows reach terminal state, then
# generates a verdict report (logs/46M_thermometer_verdict.md) with:
#   1. Flow status breakdown
#   2. Per-architecture × per-thermometer-width F1/FPR/Acc table
#      (combined with yesterday's 8-bit baselines from flows 1185 and 1188
#       for a clean 4-point series 8/16/32/64)
#   3. Verdict on three hypotheses:
#      - Both architectures lift → encoding was the ceiling
#      - Both stay flat        → address-tap saturation confirmed
#      - Only one lifts        → architecture-dependent saturation
#   4. Mac desktop notification with the verdict
#
# Run as nohup background:
#     nohup bash scripts/watcher_46M_thermometer.sh > /tmp/watcher_46M_thermometer.out 2>&1 &

set -e

cd "$(dirname "$0")/.."
DB="db/wnn.db"
REPORT="logs/46M_thermometer_verdict.md"
LOG="/tmp/watcher_46M_thermometer.log"
SWEEP_TAG="THERMO46M"
EXPECTED_FLOWS=12

mkdir -p "$(dirname "$REPORT")"

log() {
	echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

notify() {
	osascript -e "display notification \"$1\" with title \"46M Thermometer Sweep\""
}

log "==== Watcher started: monitoring ${EXPECTED_FLOWS} ${SWEEP_TAG} flows ===="

actual_flows=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${SWEEP_TAG}%';")
log "  Found ${actual_flows} flows matching tag (expected ${EXPECTED_FLOWS})"

start_time=$(date +%s)
last_done=-1
while true; do
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
		log "==== All ${actual_flows} ${SWEEP_TAG} flows terminal ===="
		break
	fi

	sleep 60
done

elapsed=$(($(date +%s) - start_time))
hours=$((elapsed / 3600))
mins=$(((elapsed % 3600) / 60))
log "Watcher waited ${hours}h ${mins}m for thermometer sweep."

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

# Pull thermometer sweep results
cur.execute("""
    SELECT vs.flow_id, f.name, vs.threshold_metadata,
           json_extract(f.config_json, '$.params.ids_n_bits') as thermo_bits,
           g.total_neurons, g.tiers_json
    FROM validation_summaries vs
    JOIN experiments e ON vs.experiment_id = e.id
    JOIN flows f ON e.flow_id = f.id
    LEFT JOIN genomes g ON g.config_hash = vs.genome_hash AND g.experiment_id = vs.experiment_id
    WHERE f.name LIKE '%THERMO46M%'
      AND vs.validation_point = 'final'
      AND vs.genome_type = 'best_fitness'
    ORDER BY thermo_bits, g.total_neurons
""")
thermo_rows = cur.fetchall()

# Also pull yesterday's 8-bit baselines for the same architectures (flows 1185, 1188)
cur.execute("""
    SELECT vs.flow_id, f.name, vs.threshold_metadata,
           json_extract(f.config_json, '$.params.ids_n_bits') as thermo_bits,
           g.total_neurons, g.tiers_json
    FROM validation_summaries vs
    JOIN experiments e ON vs.experiment_id = e.id
    JOIN flows f ON e.flow_id = f.id
    LEFT JOIN genomes g ON g.config_hash = vs.genome_hash AND g.experiment_id = vs.experiment_id
    WHERE f.id IN (1185, 1188)
      AND vs.validation_point = 'final'
      AND vs.genome_type = 'best_fitness'
""")
baseline_rows = cur.fetchall()

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

# arch -> thermo -> [(f1, fpr, acc), ...]
results = defaultdict(lambda: defaultdict(list))

def add_row(row):
    fid, fname, meta_str, thermo_bits, n, tiers = row
    if not meta_str:
        return
    meta = json.loads(meta_str)
    bits = parse_bits(tiers)
    if bits is None or thermo_bits is None:
        return
    arch_key = (n, bits)
    tc = meta.get("train_cal", {})
    if not tc:
        return
    f1 = tc.get("f1", 0) * 100
    fpr = tc.get("fpr", 0) * 100
    acc = tc.get("acc", 0) * 100
    results[arch_key][int(thermo_bits)].append((f1, fpr, acc, fid))

for r in baseline_rows + thermo_rows:
    add_row(r)

print("# 46M Thermometer Encoding Sweep Verdict")
print()
print("**Sweep tag:** THERMO46M")
print("**Hypothesis tested:** Is the 32-bit address-tap saturation observed in")
print("the Pareto sweep bounded by the 8-bit thermometer encoding rather than")
print("the address-tap width itself?")
print()
print("**Method:** Two architectures × 4 thermometer widths (8/16/32/64). The")
print("8-bit baselines come from yesterday's Pareto sweep (flows 1185 and 1188);")
print("the 16/32/64-bit runs are from today's THERMO46M sweep (flows 1189-1200).")
print()

# Per-architecture results
print("## Results by architecture")
print()
for arch_key in sorted(results.keys()):
    n, b = arch_key
    print(f"### {n}n × {b}b")
    print()
    print("| Thermometer | Seeds | F1 | FPR | Acc |")
    print("|---|---:|---:|---:|---:|")
    for thermo in sorted(results[arch_key].keys()):
        entries = results[arch_key][thermo]
        f1s = [e[0] for e in entries]
        fprs = [e[1] for e in entries]
        accs = [e[2] for e in entries]
        def m(xs):
            if len(xs) == 1:
                return f"{xs[0]:.2f}%"
            return f"{statistics.mean(xs):.2f}±{statistics.stdev(xs):.2f}%"
        print(f"| {thermo}-bit | {len(entries)} | {m(f1s)} | {m(fprs)} | {m(accs)} |")
    print()

# Verdict logic: compare 16+ to 8 baseline for each arch
print("## Verdict")
print()
verdicts = {}
for arch_key in sorted(results.keys()):
    n, b = arch_key
    thermos = results[arch_key]
    if 8 not in thermos:
        verdicts[arch_key] = "NO BASELINE"
        continue
    f1_8 = statistics.mean([e[0] for e in thermos[8]])
    deltas = {}
    for t in (16, 32, 64):
        if t in thermos:
            f1_t = statistics.mean([e[0] for e in thermos[t]])
            deltas[t] = f1_t - f1_8

    max_lift = max(deltas.values()) if deltas else 0
    min_delta = min(deltas.values()) if deltas else 0
    if max_lift > 1.0:
        verdicts[arch_key] = f"LIFT (best +{max_lift:.2f}pp F1 vs 8b)"
    elif min_delta < -1.0:
        verdicts[arch_key] = f"DROP (worst {min_delta:.2f}pp F1 vs 8b)"
    else:
        verdicts[arch_key] = f"FLAT (within ±1pp of 8b baseline)"

for arch_key, v in verdicts.items():
    n, b = arch_key
    print(f"- **{n}n × {b}b:** {v}")
print()

# Final hypothesis call
all_lift = all("LIFT" in v for v in verdicts.values())
all_flat = all("FLAT" in v for v in verdicts.values())
all_drop = all("DROP" in v for v in verdicts.values())

if all_lift:
    print("### 🎉 ENCODING WAS THE CEILING")
    print()
    print("Both architectures show F1 lift on 16+ bit thermometer encoding.")
    print("The 32-bit address-tap saturation we observed yesterday was actually")
    print("a thermometer-encoding-collision ceiling, not an address-space ceiling.")
    print("This is a publishable finding — wider encoding unlocks more discrimination.")
elif all_flat:
    print("### ✅ ADDRESS-TAP SATURATION CONFIRMED")
    print()
    print("Neither architecture shows meaningful F1 lift across thermometer widths.")
    print("The 32-bit address-tap width is genuinely the discrimination ceiling on")
    print("this dataset, regardless of input encoding richness. The thermometer")
    print("encoding caps effective discrimination at the input layer, and increasing")
    print("its width doesn't help downstream discrimination.")
elif all_drop:
    print("### ⚠️  WIDER THERMOMETER HURTS")
    print()
    print("Both architectures show F1 drops with wider thermometer encoding.")
    print("This confirms the dilution hypothesis from the existing 1.3M sweep:")
    print("more thermometer bits dilute the per-neuron information, hurting")
    print("discrimination. The 8-bit thermometer is the sweet spot.")
else:
    print("### 🔬 ARCHITECTURE-DEPENDENT")
    print()
    print("The two architectures behave differently across thermometer widths.")
    print("This is a more nuanced finding — the saturation interacts with the")
    print("address-tap-to-input ratio. Worth investigating further.")
print()

conn.close()
PYEOF

cat "$REPORT" >> "$LOG"
log "Report written to $REPORT"

# Notification
verdict_line=$(grep -E "^### " "$REPORT" | head -1 | sed 's/^### //')
notify "Thermometer sweep done: ${verdict_line}"

log "==== Thermometer watcher complete ===="
