#!/usr/bin/env bash
# =============================================================
# Overnight WS0-WS3 Experiment Runner
# Waits for the current experiment (PID 3563) to finish,
# then runs WS0 → WS1 → WS2 → WS3 sequentially.
# =============================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

LOG_DIR="$SCRIPT_DIR/experiments"
mkdir -p "$LOG_DIR"

LOGFILE="$LOG_DIR/overnight_ws_$(date +%Y%m%d_%H%M%S).log"

log() {
	echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOGFILE"
}

# ---- Phase 0: Wait for current experiment to finish ----
WORKER_PID=3563

log "=== Overnight WS Runner Started ==="
log "Monitoring worker PID $WORKER_PID for idle state..."

# Phase 1: Wait until 3:30 AM (user estimated 2-3h from midnight)
EARLIEST_START="03:30"
log "Waiting until $EARLIEST_START before checking worker state..."

while true; do
	CURRENT_TIME=$(date +%H%M)
	TARGET_TIME=$(echo "$EARLIEST_START" | tr -d ':')
	if [ "$CURRENT_TIME" -ge "$TARGET_TIME" ]; then
		log "Reached $EARLIEST_START. Checking if worker is done..."
		break
	fi
	# Log every 30 min
	MINS=$(date +%M)
	if [ "$MINS" = "00" ] || [ "$MINS" = "30" ]; then
		CPU=$(ps -p "$WORKER_PID" -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
		log "Waiting... (worker at ${CPU}% CPU, target time: $EARLIEST_START)"
	fi
	sleep 60
done

# Phase 2: Verify worker is actually idle (CPU < 50% = not computing)
# When running experiments: 60-100% CPU. When just polling: < 20% CPU.
log "Verifying worker is idle..."
while true; do
	if ! kill -0 "$WORKER_PID" 2>/dev/null; then
		log "Worker PID $WORKER_PID has exited. Proceeding."
		break
	fi

	# Sample CPU 3 times over 60s for stability
	MAX_CPU=0
	for i in 1 2 3; do
		CPU=$(ps -p "$WORKER_PID" -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
		CPU_INT=$(echo "$CPU" | cut -d. -f1)
		if [ "${CPU_INT:-0}" -gt "$MAX_CPU" ]; then
			MAX_CPU="${CPU_INT:-0}"
		fi
		sleep 20
	done

	if [ "$MAX_CPU" -lt 50 ]; then
		log "Worker idle (peak ${MAX_CPU}% CPU over 60s). Experiment is done."
		break
	else
		log "Worker still computing (peak ${MAX_CPU}% CPU). Waiting 5 min..."
		sleep 300
	fi
done

log ""
log "============================================================"
log "Starting WS Experiments"
log "============================================================"
log ""

# ---- WS0: Confidence Measurement ----
log "=== WS0: Confidence Measurement ==="
WS0_START=$(date +%s)

python -u tests/test_confidence.py \
	--output "$LOG_DIR/ws0_confidence_results.json" \
	--batch-size 5000 \
	2>&1 | tee -a "$LOGFILE"

WS0_END=$(date +%s)
log "WS0 completed in $((WS0_END - WS0_START))s"
log ""

# ---- WS1: Content-Dependent Routing ----
log "=== WS1: Content-Dependent Routing ==="
WS1_START=$(date +%s)

python -u tests/test_routing.py \
	--output "$LOG_DIR/ws1_routing_results.json" \
	--num-routes 8 \
	--top-k 2 \
	--neurons-per-route 3 \
	--bits-per-neuron 10 \
	--context-size 4 \
	2>&1 | tee -a "$LOGFILE"

WS1_END=$(date +%s)
log "WS1 completed in $((WS1_END - WS1_START))s"
log ""

# ---- WS2: Hybrid RAM+Transformer ----
log "=== WS2: Hybrid RAM+Transformer ==="
WS2_START=$(date +%s)

# Use tiny transformer for fair comparison (no HF dependency needed)
python -u tests/test_hybrid_lm.py \
	--transformer tiny \
	--epochs 3 \
	--output "$LOG_DIR/ws2_hybrid_results.json" \
	--batch-size 50 \
	2>&1 | tee -a "$LOGFILE"

WS2_END=$(date +%s)
log "WS2 completed in $((WS2_END - WS2_START))s"
log ""

# ---- WS3: RAM as Embedding Layer ----
log "=== WS3: RAM as Embedding Layer ==="
WS3_START=$(date +%s)

python -u tests/test_ram_transformer.py \
	--epochs 3 \
	--output "$LOG_DIR/ws3_ram_transformer_results.json" \
	--feature-dim 256 \
	--seq-len 32 \
	--batch-size 16 \
	2>&1 | tee -a "$LOGFILE"

WS3_END=$(date +%s)
log "WS3 completed in $((WS3_END - WS3_START))s"
log ""

# ---- Summary ----
TOTAL_TIME=$((WS3_END - WS0_START))
log "============================================================"
log "ALL EXPERIMENTS COMPLETE"
log "============================================================"
log "WS0 (confidence):   $((WS0_END - WS0_START))s"
log "WS1 (routing):      $((WS1_END - WS1_START))s"
log "WS2 (hybrid):       $((WS2_END - WS2_START))s"
log "WS3 (embedding):    $((WS3_END - WS3_START))s"
log "Total:              ${TOTAL_TIME}s ($(echo "scale=1; $TOTAL_TIME / 3600" | bc)h)"
log ""
log "Results:"
log "  $LOG_DIR/ws0_confidence_results.json"
log "  $LOG_DIR/ws1_routing_results.json"
log "  $LOG_DIR/ws2_hybrid_results.json"
log "  $LOG_DIR/ws3_ram_transformer_results.json"
log "  $LOGFILE"
