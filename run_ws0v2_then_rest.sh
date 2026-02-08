#!/bin/bash
set -eo pipefail

# Navigate to project root
cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

LOG_DIR="experiments"
LOGFILE="$LOG_DIR/ws0v2_run.log"
PIPELINE_PID=22434

log() { echo "$(date "+%Y-%m-%d %H:%M:%S") | $*" | tee -a "$LOGFILE"; }

log "=== WS0v2 + WS2 + WS3 Pipeline ==="

# Step 1: Wait for WS1 to finish by monitoring the routing test process
log "Waiting for WS1 (test_routing.py) to finish..."
while ps -p 24422 > /dev/null 2>&1; do
	sleep 30
done
log "WS1 process finished"

# Step 2: Kill the parent pipeline before it starts WS2
# Give it a moment to log WS1 completion
sleep 5
if ps -p $PIPELINE_PID > /dev/null 2>&1; then
	log "Killing pipeline ($PIPELINE_PID) to insert WS0v2..."
	kill $PIPELINE_PID 2>/dev/null || true
	sleep 2
fi

# Step 3: Run WS0v2 with all four confidence metrics
log "=== WS0v2: Confidence Measurement (4 metrics) ==="
WS0_START=$(date +%s)
python -u tests/test_confidence.py \
	--output "$LOG_DIR/ws0v2_confidence_results.json" \
	--batch-size 5000 2>&1 | tee -a "$LOGFILE"
WS0_END=$(date +%s)
log "WS0v2 completed in $((WS0_END - WS0_START))s"

# Step 4: Run WS2
log "=== WS2: Hybrid RAM+Transformer ==="
WS2_START=$(date +%s)
python -u tests/test_hybrid_lm.py \
	--transformer tiny --epochs 3 \
	--output "$LOG_DIR/ws2_hybrid_results.json" \
	--batch-size 50 2>&1 | tee -a "$LOGFILE"
WS2_END=$(date +%s)
log "WS2 completed in $((WS2_END - WS2_START))s"

# Step 5: Run WS3
log "=== WS3: RAM as Embedding Layer ==="
WS3_START=$(date +%s)
python -u tests/test_ram_transformer.py \
	--epochs 3 \
	--output "$LOG_DIR/ws3_ram_transformer_results.json" \
	--feature-dim 256 --seq-len 32 --batch-size 16 2>&1 | tee -a "$LOGFILE"
WS3_END=$(date +%s)
log "WS3 completed in $((WS3_END - WS3_START))s"

TOTAL=$((WS3_END - WS0_START))
log "ALL COMPLETE - Total: ${TOTAL}s"
