#!/bin/bash
# Watcher for the full 46M thermometer sweep result (combining 8b baseline,
# 16/32/64b high sweep, and 2/4b low sweep). Polls until all 8 LOW flows
# reach terminal state, then writes a unified verdict.
#
# Run as nohup background:
#     nohup bash scripts/watcher_46M_thermometer_full.sh > /tmp/watcher_thermo_full.out 2>&1 &

set -e

cd "$(dirname "$0")/.."
DB="db/wnn.db"
REPORT="logs/46M_thermometer_full_verdict.md"
LOG="/tmp/watcher_thermo_full.log"
LOW_TAG="THERMO46MLOW"
EXPECTED_LOW=8

mkdir -p "$(dirname "$REPORT")"

log() {
	echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

notify() {
	osascript -e "display notification \"$1\" with title \"46M Thermometer Full Sweep\""
}

log "==== Watcher started: waiting for ${EXPECTED_LOW} ${LOW_TAG} flows ===="

start_time=$(date +%s)
last_done=-1
while true; do
	done=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${LOW_TAG}%' AND status IN ('completed','failed','cancelled');")
	running=$(sqlite3 "$DB" "SELECT name FROM flows WHERE name LIKE '%${LOW_TAG}%' AND status='running' LIMIT 1;" || true)

	if [ "$done" != "$last_done" ]; then
		if [ -n "$running" ]; then
			log "  Progress: ${done}/${EXPECTED_LOW} done, running: ${running}"
		else
			log "  Progress: ${done}/${EXPECTED_LOW} done"
		fi
		last_done="$done"
	fi

	if [ "$done" -ge "$EXPECTED_LOW" ]; then
		log "==== All ${EXPECTED_LOW} ${LOW_TAG} flows terminal ===="
		break
	fi

	sleep 60
done

elapsed=$(($(date +%s) - start_time))
log "Watcher waited $((elapsed/60))m $((elapsed%60))s for LOW sweep."
log "Generating full thermometer verdict report..."

source wnn/bin/activate 2>/dev/null

python3 - <<'PYEOF' > "$REPORT"
import sqlite3
import json
import re
import statistics
from collections import defaultdict

conn = sqlite3.connect("db/wnn.db")
cur = conn.cursor()

# Pull results from three sources:
# 1. 8-bit baselines: Pareto sweep flows 1185 (96n × 32b) and 400n × 8b from 1184
# 2. HIGH sweep: THERMO46M tag (16/32/64-bit)
# 3. LOW sweep: THERMO46MLOW tag (2/4-bit)

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

# results[arch_key][thermo_bits] = list of (f1, fpr, acc)  (train_cal)
results = defaultdict(lambda: defaultdict(list))

# Pull all relevant flows
cur.execute("""
    SELECT f.id, f.name, vs.threshold_metadata, g.total_neurons, g.tiers_json,
           json_extract(f.config_json, '$.params.ids_n_bits') as thermo_bits
    FROM flows f
    JOIN experiments e ON e.flow_id = f.id
    JOIN validation_summaries vs ON vs.experiment_id = e.id
    LEFT JOIN genomes g ON g.config_hash = vs.genome_hash AND g.experiment_id = vs.experiment_id
    WHERE (f.name LIKE '%THERMO46M%' OR f.id IN (1184, 1185, 1188))
      AND vs.validation_point = 'final'
      AND vs.genome_type = 'best_fitness'
""")

for fid, fname, meta_str, n, tiers, thermo in cur.fetchall():
    if not meta_str or thermo is None: continue
    bits = parse_bits(tiers)
    if bits is None: continue
    meta = json.loads(meta_str)
    tc = meta.get("train_cal", {})
    if not tc: continue
    f1 = tc.get("f1", 0) * 100
    fpr = tc.get("fpr", 0) * 100
    acc = tc.get("acc", 0) * 100
    arch_key = (n, bits)
    results[arch_key][int(thermo)].append((f1, fpr, acc))

def ms(xs):
    if not xs: return "—"
    if len(xs) == 1: return f"{xs[0]:.2f}%"
    return f"{statistics.mean(xs):.2f}±{statistics.stdev(xs):.2f}%"

print("# 46M Full Thermometer Encoding Sweep Verdict")
print()
print("Combined 6-point thermometer-width curve (2/4/8/16/32/64-bit) on the")
print("full CIC-IoT-2023 46M dataset, across two architectures.")
print()
print("**Data sources:**")
print("- 8-bit baselines from the Pareto sweep (flows 1184, 1185, 1188)")
print("- 16/32/64-bit from THERMO46M (flows 1189-1200)")
print("- 2/4-bit from THERMO46MLOW (flows 1201-1208)")
print()

# For each architecture, print a row
archs_of_interest = sorted(results.keys())
for arch_key in archs_of_interest:
    n, b = arch_key
    if not results[arch_key]:
        continue
    print(f"## {n}n × {b}b")
    print()
    widths_present = sorted(results[arch_key].keys())
    print("| Thermometer | Seeds | F1 | FPR | Acc |")
    print("|---|---:|---:|---:|---:|")
    for t in widths_present:
        entries = results[arch_key][t]
        f1s = [e[0] for e in entries]
        fprs = [e[1] for e in entries]
        accs = [e[2] for e in entries]
        print(f"| {t}-bit | {len(entries)} | {ms(f1s)} | {ms(fprs)} | {ms(accs)} |")
    # Compute range
    all_f1s = [e[0] for t in widths_present for e in results[arch_key][t]]
    if len(all_f1s) > 1:
        f1_range = max(all_f1s) - min(all_f1s)
        print(f"\nF1 range across widths: **{f1_range:.2f}pp**")
    print()

# Verdict
print("## Verdict")
print()
for arch_key in archs_of_interest:
    n, b = arch_key
    widths_present = sorted(results[arch_key].keys())
    if len(widths_present) < 2:
        continue
    means_per_width = {t: statistics.mean([e[0] for e in results[arch_key][t]]) for t in widths_present}
    baseline_f1 = means_per_width.get(8)
    if baseline_f1 is None:
        continue
    best_width = max(means_per_width.items(), key=lambda x: x[1])
    worst_width = min(means_per_width.items(), key=lambda x: x[1])
    total_range = best_width[1] - worst_width[1]
    best_delta = best_width[1] - baseline_f1
    lines = []
    if abs(best_delta) > 1.0:
        lines.append(f"**{n}n × {b}b:** best thermometer is **{best_width[0]}-bit** with F1 {best_width[1]:.2f}% "
                     f"({best_delta:+.2f}pp vs 8-bit baseline).")
    else:
        lines.append(f"**{n}n × {b}b:** FLAT — best thermometer {best_width[0]}b at {best_width[1]:.2f}% F1, "
                     f"range {total_range:.2f}pp across all widths.")
    for ln in lines:
        print(f"- {ln}")
print()

conn.close()
PYEOF

cat "$REPORT" >> "$LOG"
log "Report written to $REPORT"
notify "Full thermometer sweep verdict ready: $REPORT"
log "==== Watcher complete ===="
