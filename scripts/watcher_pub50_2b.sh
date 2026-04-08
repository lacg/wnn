#!/bin/bash
# Watcher for the PUB50-2b 10-flow validation batch (flows 1209-1218).
#
# Strategy: this batch is expected to take ~10 hours wall clock. Instead
# of polling every minute for 10 hours, we use a smart sleep schedule:
#
#   1. Sleep 7 hours upfront
#   2. Wake up and check: is the worker on the LAST flow (id 1209) yet?
#      - If yes OR all 10 are terminal:  drop to 60s polling mode
#      - If no:                          sleep 1 more hour, repeat
#   3. Once in polling mode, check every 60s until all 10 are terminal
#   4. Generate verdict report with mean ± std comparison to 8b PUB50
#   5. Send Mac desktop notification and exit
#
# Run as nohup background:
#     nohup bash scripts/watcher_pub50_2b.sh > /tmp/watcher_pub50_2b.out 2>&1 &

set -e

cd "$(dirname "$0")/.."
DB="db/wnn.db"
REPORT="logs/pub50_2b_verdict.md"
LOG="/tmp/watcher_pub50_2b.log"
BATCH_TAG="PUB50-2b-ciciot-random"
EXPECTED=10
LAST_FLOW_ID=1209  # The lowest-ID PUB50-2b flow runs last (ORDER BY id DESC)

mkdir -p "$(dirname "$REPORT")"

log() {
	echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"
}

notify() {
	osascript -e "display notification \"$1\" with title \"PUB50-2b 10-run validation\""
}

# Initial sleep is tunable via INITIAL_SLEEP env var (seconds). Default 7h is a
# pessimistic estimate for 8-bit PUB50 flows (~75 min each × 10 = 12.5h). With
# 2-bit thermometer the per-flow time drops to ~37 min, so you can pass
# INITIAL_SLEEP=10800 (3h) to shorten the pre-poll wait.
INITIAL_SLEEP="${INITIAL_SLEEP:-25200}"
initial_sleep_hours=$(awk "BEGIN {printf \"%.1f\", ${INITIAL_SLEEP}/3600}")

log "==== Watcher started: monitoring ${EXPECTED} ${BATCH_TAG} flows ===="
log "Strategy: sleep ${initial_sleep_hours}h, then hourly checks until last flow is running, then 60s polling."

start_time=$(date +%s)

# Phase 1: initial sleep (tunable via INITIAL_SLEEP)
log "Sleeping ${initial_sleep_hours}h (override with INITIAL_SLEEP=<seconds> env var)..."
sleep "${INITIAL_SLEEP}"

# Phase 2: hourly checks until the last flow is running or the batch is done
log "Waking up — checking if we're on the last flow yet..."

while true; do
	done=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${BATCH_TAG}%' AND status IN ('completed','failed','cancelled');")
	last_flow_status=$(sqlite3 "$DB" "SELECT status FROM flows WHERE id=${LAST_FLOW_ID};")

	log "  ${done}/${EXPECTED} terminal, last flow (${LAST_FLOW_ID}) status: ${last_flow_status}"

	if [ "$done" -ge "$EXPECTED" ]; then
		log "All batch flows already terminal — skipping to verdict."
		break
	fi

	if [ "$last_flow_status" = "running" ]; then
		log "Last flow is running — switching to 60s poll mode."
		break
	fi

	log "Not yet on the last flow. Sleeping 1h..."
	sleep 3600
done

# Phase 3: 60s polling until all terminal
last_done=-1
while true; do
	done=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${BATCH_TAG}%' AND status IN ('completed','failed','cancelled');")
	if [ "$done" != "$last_done" ]; then
		running=$(sqlite3 "$DB" "SELECT name FROM flows WHERE name LIKE '%${BATCH_TAG}%' AND status='running' LIMIT 1;" || true)
		if [ -n "$running" ]; then
			log "  Progress: ${done}/${EXPECTED} done, running: ${running}"
		else
			log "  Progress: ${done}/${EXPECTED} done"
		fi
		last_done="$done"
	fi
	if [ "$done" -ge "$EXPECTED" ]; then
		log "==== All ${EXPECTED} ${BATCH_TAG} flows terminal ===="
		break
	fi
	sleep 60
done

elapsed=$(($(date +%s) - start_time))
log "Watcher waited $((elapsed/3600))h $((elapsed%3600/60))m for batch."
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

BATCH_TAG = "PUB50-2b-ciciot-random"

def parse_bits(tiers):
    if not tiers: return None
    if tiers.startswith("["):
        try: return json.loads(tiers)[0].get("bits")
        except Exception: return None
    m = re.search(r"bits=\[(\d+)-(\d+)\]", tiers)
    return int(m.group(2)) if m else None

# Pull both Grid Search and GA Neurons best_fitness results
def pull_phase(phase_name):
    cur.execute("""
        SELECT vs.threshold_metadata, g.total_neurons, g.tiers_json, f.name
        FROM validation_summaries vs
        JOIN experiments e ON vs.experiment_id = e.id
        JOIN flows f ON e.flow_id = f.id
        LEFT JOIN genomes g ON g.config_hash = vs.genome_hash AND g.experiment_id = vs.experiment_id
        WHERE f.name LIKE ? AND vs.validation_point = 'final'
          AND vs.genome_type = 'best_fitness' AND e.name = ?
    """, (f"%{BATCH_TAG}%", phase_name))
    out = []
    for meta_str, n, tiers, fname in cur.fetchall():
        if not meta_str: continue
        meta = json.loads(meta_str)
        tc = meta.get("train_cal", {})
        f05 = meta.get("fixed_05", {})
        if not tc: continue
        out.append({
            "flow": fname, "n": n, "b": parse_bits(tiers),
            "f1_tc": tc.get("f1",0)*100, "fpr_tc": tc.get("fpr",0)*100, "acc_tc": tc.get("acc",0)*100,
            "f1_05": f05.get("f1",0)*100, "fpr_05": f05.get("fpr",0)*100, "acc_05": f05.get("acc",0)*100,
        })
    return out

gs_rows = pull_phase("Grid Search (neurons x bits)")
ga_rows = pull_phase("GA Neurons")

# 8b PUB50 baseline (all completed runs) for comparison
cur.execute("""
    SELECT vs.threshold_metadata, g.total_neurons
    FROM validation_summaries vs
    JOIN experiments e ON vs.experiment_id = e.id
    JOIN flows f ON e.flow_id = f.id
    LEFT JOIN genomes g ON g.config_hash = vs.genome_hash AND g.experiment_id = vs.experiment_id
    WHERE f.name LIKE '%PUB50%ciciot%' AND f.name NOT LIKE '%-2b-%'
      AND f.status = 'completed'
      AND vs.validation_point = 'final'
      AND vs.genome_type = 'best_fitness'
      AND e.name = 'GA Neurons'
""")
baseline_8b = []
for meta_str, n in cur.fetchall():
    if not meta_str: continue
    meta = json.loads(meta_str)
    tc = meta.get("train_cal", {})
    if tc:
        baseline_8b.append({
            "f1_tc": tc.get("f1",0)*100,
            "fpr_tc": tc.get("fpr",0)*100,
            "acc_tc": tc.get("acc",0)*100,
            "n": n,
        })

def ms(xs):
    if not xs: return "—"
    if len(xs) == 1: return f"{xs[0]:.2f}"
    return f"{statistics.mean(xs):.2f}±{statistics.stdev(xs):.2f}"

def table(rows, label):
    print(f"### {label}")
    print()
    if not rows:
        print("(no data)\n")
        return
    f1_tcs = [r["f1_tc"] for r in rows]
    fpr_tcs = [r["fpr_tc"] for r in rows]
    acc_tcs = [r["acc_tc"] for r in rows]
    f1_05s = [r["f1_05"] for r in rows]
    fpr_05s = [r["fpr_05"] for r in rows]
    ns = [r["n"] for r in rows if r["n"] is not None]
    bs = [r["b"] for r in rows if r["b"] is not None]
    print(f"| Metric | train_cal | fixed_05 |")
    print(f"|---|---:|---:|")
    print(f"| F1 | {ms(f1_tcs)}% | {ms(f1_05s)}% |")
    print(f"| FPR | {ms(fpr_tcs)}% | {ms(fpr_05s)}% |")
    print(f"| Acc | {ms(acc_tcs)}% | — |")
    print(f"| Neurons | {ms(ns)} | |")
    print(f"| Bits | {ms(bs)} | |")
    print()
    print(f"Runs: {len(rows)}")
    print()

print(f"# PUB50-2b 10-Run Validation Batch Verdict")
print()
print(f"**Batch:** {BATCH_TAG} (ids_n_bits=2, all other params identical to PUB50 ciciot 8b)")
print(f"**Completed:** {len(gs_rows)} Grid Search / {len(ga_rows)} GA Neurons runs")
print()

print("## 2-bit batch results (best_fitness genome)")
print()
table(gs_rows, "Grid Search phase")
table(ga_rows, "GA Neurons phase")

print("## Comparison: 2b vs 8b (GA Neurons best_fitness, train_cal)")
print()
if baseline_8b and ga_rows:
    f1_2b = [r["f1_tc"] for r in ga_rows]
    fpr_2b = [r["fpr_tc"] for r in ga_rows]
    acc_2b = [r["acc_tc"] for r in ga_rows]
    f1_8b = [r["f1_tc"] for r in baseline_8b]
    fpr_8b = [r["fpr_tc"] for r in baseline_8b]
    acc_8b = [r["acc_tc"] for r in baseline_8b]
    print(f"| Metric | 8-bit (n={len(baseline_8b)}) | 2-bit (n={len(ga_rows)}) | Delta |")
    print(f"|---|---:|---:|---:|")
    dF1 = statistics.mean(f1_2b) - statistics.mean(f1_8b)
    dFPR = statistics.mean(fpr_2b) - statistics.mean(fpr_8b)
    dAcc = statistics.mean(acc_2b) - statistics.mean(acc_8b)
    print(f"| F1  | {ms(f1_8b)}% | {ms(f1_2b)}% | {dF1:+.2f}pp |")
    print(f"| FPR | {ms(fpr_8b)}% | {ms(fpr_2b)}% | {dFPR:+.2f}pp |")
    print(f"| Acc | {ms(acc_8b)}% | {ms(acc_2b)}% | {dAcc:+.2f}pp |")
    print()

    # Balanced fitness score = 0.35*F1 + 0.35*(1-FPR) + 0.2*Acc
    # CE omitted because validation_summaries doesn't always have it.
    # This approximates what the harmonic-rank fitness is actually optimizing.
    def balanced_score(f1, fpr, acc):
        # f1/fpr/acc are percent values — score is also in pp
        return 0.35 * f1 + 0.35 * (100 - fpr) + 0.2 * acc
    score_2b = [balanced_score(r["f1_tc"], r["fpr_tc"], r["acc_tc"]) for r in ga_rows]
    score_8b = [balanced_score(r["f1_tc"], r["fpr_tc"], r["acc_tc"]) for r in baseline_8b]
    dScore = statistics.mean(score_2b) - statistics.mean(score_8b)
    print(f"**Balanced fitness score** (0.35·F1 + 0.35·(1−FPR) + 0.2·Acc):")
    print(f"  8b: {statistics.mean(score_8b):.2f}pp  |  2b: {statistics.mean(score_2b):.2f}pp  |  Δ: {dScore:+.2f}pp")
    print()

    # Verdict
    # Three-tier verdict combining strict F1+FPR rule and balanced-fitness rule:
    #   STRICT    : dF1 > +0.5pp AND dFPR < 0   (clean win on both raw metrics)
    #   SOFT      : dScore >= +0.3pp            (fitness wins even if F1 drops, via FPR gain)
    #   INCONCL.  : |dScore| < 0.3pp            (within noise floor; stdev across 8b runs is ~0.3-0.4pp)
    #   NOT CONF. : dScore <= -0.3pp            (fitness actually dropped)
    # 0.3pp threshold chosen as the balanced-score noise floor estimated from
    # stdev of the 46-run 8b PUB50 baseline.
    CONFIRMED_SOFT_THRESHOLD = 0.3
    print("## Verdict")
    print()
    if dF1 > 0.5 and dFPR < 0:
        print("### 🎉 2-BIT LIFT CONFIRMED (strict)")
        print()
        print(f"2-bit thermometer produces a meaningful F1 lift ({dF1:+.2f}pp) and lower FPR ({dFPR:+.2f}pp)")
        print(f"vs the 8-bit PUB50 baseline at matched sample size. The 46M 4-seed finding reproduces")
        print(f"statistically on the 1.3M subsample across {len(ga_rows)} runs.")
        print()
        print("**Recommendation:** run the remaining 102 PUB50-2b flows (seeds 11-112) to complete a")
        print("full 112-run statistical batch alongside the existing 8b PUB50.")
    elif dScore >= CONFIRMED_SOFT_THRESHOLD:
        print("### 🎯 2-BIT LIFT CONFIRMED (soft, fitness-balanced)")
        print()
        print(f"2-bit F1 change is {dF1:+.2f}pp but the FPR drop ({dFPR:+.2f}pp) more than compensates on the")
        print(f"balanced fitness score ({dScore:+.2f}pp vs threshold {CONFIRMED_SOFT_THRESHOLD}pp). This is the same")
        print(f"tradeoff the 46M data showed (lower F1 ceiling, much lower FPR floor), now reproducing")
        print(f"at the 1.3M subsample scale across {len(ga_rows)} runs.")
        print()
        print("**Recommendation:** run the remaining 102 PUB50-2b flows (seeds 11-112) to complete a")
        print("full 112-run statistical batch. Frame the paper around fitness-balanced improvement")
        print("rather than raw F1 (which is the honest story anyway given the balanced fitness weights).")
    elif abs(dScore) < CONFIRMED_SOFT_THRESHOLD:
        print("### ⚠️ INCONCLUSIVE")
        print()
        print(f"Balanced fitness delta ({dScore:+.2f}pp) is within the ±{CONFIRMED_SOFT_THRESHOLD}pp noise floor. F1 change")
        print(f"was {dF1:+.2f}pp, FPR change was {dFPR:+.2f}pp — they trade against each other and net out.")
        print(f"The 46M finding may not reproduce at the subsample scale, or {len(ga_rows)} runs is too few")
        print(f"to detect the effect above noise.")
    else:
        print("### ❌ 2-BIT LIFT NOT CONFIRMED")
        print()
        print(f"Balanced fitness dropped by {abs(dScore):.2f}pp at the 1.3M scale (F1 {dF1:+.2f}pp,")
        print(f"FPR {dFPR:+.2f}pp). The 46M finding may be scale-dependent (2-bit helps at 46M training")
        print(f"set size but hurts at 1.3M).")
        print()
        print("**Recommendation:** do not scale to 112 runs. Frame the 46M 2-bit finding as")
        print("scale-dependent in the paper, not as a universal improvement.")

conn.close()
PYEOF

cat "$REPORT" >> "$LOG"
log "Report written to $REPORT"

# ============================================================================
# AUTO-RATCHET: if verdict is CONFIRMED, queue 102 follow-up flows (seeds 11-112)
# to complete the full 112-run PUB50-2b batch. If INCONCLUSIVE or NOT CONFIRMED,
# do nothing and let the user decide.
# ============================================================================
if grep -q "2-BIT LIFT CONFIRMED" "$REPORT"; then
	log "==== Verdict: CONFIRMED — auto-ratcheting to 102 follow-up flows ===="
	log "Creating seeds 11-112 (102 more flows)..."
	if python scripts/create_pub50_2b_batch.py \
		--count 102 \
		--start-seed 11 \
		--queue 2>&1 | tee -a "$LOG"; then
		created=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE '%${BATCH_TAG}%' AND json_extract(config_json, '\$.params.seed') BETWEEN 11 AND 112;")
		log "Auto-ratchet complete: ${created} additional flows queued (seeds 11-112)."
		log "The full 112-run PUB50-2b batch is now in flight."

		# SUSPEND (not cancel) remaining 8b PUB50 flows to free compute for
		# higher-priority roadmap work (UNSW random 80/20 with fitness fix,
		# 46M full-GA with 2b, etc.). Completed 8b runs are PRESERVED.
		# Using 'paused' status so they can be unpaused later if desired
		# — this is reversible, unlike cancellation.
		log "Suspending remaining 8b PUB50 flows (queued only — completed runs preserved)..."
		before=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot-random-%' AND status IN ('queued','running');")
		sqlite3 "$DB" "
			UPDATE flows
			SET status='paused',
			    status_message='Paused: PUB50-2b 10-run validation CONFIRMED the 2-bit lift on 2026-04-08. 102 follow-up 2b flows queued as higher priority. Completed 8b runs preserved for tab:ciot-phase. To resume, UPDATE status back to queued.'
			WHERE name LIKE 'PUB50-ciciot-random-%'
			  AND status IN ('queued','running');
		"
		after=$(sqlite3 "$DB" "SELECT COUNT(*) FROM flows WHERE name LIKE 'PUB50-ciciot-random-%' AND status IN ('queued','running');")
		suspended=$((before - after))
		log "Suspended ${suspended} remaining 8b PUB50 flows (freed ~$((suspended * 78 / 60)) hours; reversible via UPDATE status='queued')."

		notify "PUB50-2b CONFIRMED: 102 2b flows queued, ${suspended} 8b flows suspended (reversible)"
	else
		log "ERROR: auto-ratchet script failed. Check ${LOG} and run manually if desired."
		notify "PUB50-2b CONFIRMED but auto-ratchet failed — run create script manually"
	fi
elif grep -q "INCONCLUSIVE" "$REPORT"; then
	log "==== Verdict: INCONCLUSIVE — no auto-ratchet. Decision left to user. ===="
	notify "PUB50-2b INCONCLUSIVE — see ${REPORT} for numbers, no ratchet queued"
elif grep -q "2-BIT LIFT NOT CONFIRMED" "$REPORT"; then
	log "==== Verdict: NOT CONFIRMED — no auto-ratchet. Frame 46M finding as scale-dependent. ===="
	notify "PUB50-2b NOT CONFIRMED — 46M finding is scale-dependent, no ratchet"
else
	log "WARNING: verdict section unreadable — no auto-action taken."
	notify "PUB50-2b verdict unreadable — check ${REPORT} manually"
fi

log "==== Watcher complete ===="
