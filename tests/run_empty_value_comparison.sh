#!/bin/bash
# Run EMPTY=0.0 experiment after the baseline (EMPTY=0.5) overnight run completes
#
# This script monitors the overnight run and starts the comparison experiment
# once it finishes.

set -e

cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

BASELINE_LOG="/tmp/five_tier_fulldata_overnight.log"
BASELINE_PID=89804
EMPTY0_LOG="/tmp/five_tier_empty0.log"

echo "=============================================="
echo "EMPTY Value Comparison Experiment"
echo "=============================================="
echo ""
echo "Baseline (EMPTY=0.5) running at PID $BASELINE_PID"
echo "Log: $BASELINE_LOG"
echo ""
echo "Waiting for baseline to complete..."

# Wait for baseline to complete
while kill -0 $BASELINE_PID 2>/dev/null; do
    # Show progress every 5 minutes
    sleep 300
    echo "$(date '+%H:%M:%S') - Baseline still running..."
    tail -3 "$BASELINE_LOG" 2>/dev/null | head -1 || true
done

echo ""
echo "=============================================="
echo "Baseline completed! Starting EMPTY=0.0 experiment..."
echo "=============================================="
echo ""

# Set EMPTY=0.0 (already default, but explicit)
python -c "import ram_accelerator; ram_accelerator.set_empty_value(0.0); print(f'EMPTY value set to: {ram_accelerator.get_empty_value()}')"

# Run the same 5-tier experiment with EMPTY=0.0
# Note: We run WITHOUT optimization first to see the raw improvement
echo "Running 5-tier simple config with EMPTY=0.0 (NO OPTIMIZATION)..."
echo "Config: 50,15,20;50,13,18;400,9,10;20000,7,9;rest,5,8"
echo "Log: $EMPTY0_LOG"
echo ""

PYTHONUNBUFFERED=1 python -u tests/ramlm_full_benchmark.py \
  --full-data \
  --tiered "50,15,20;50,13,18;400,9,10;20000,7,9;rest,5,8" \
  --per-tier \
  2>&1 | tee "$EMPTY0_LOG"

echo ""
echo "=============================================="
echo "EMPTY=0.0 experiment completed!"
echo "=============================================="
echo ""
echo "Results summary:"
echo "  Baseline (EMPTY=0.5): $BASELINE_LOG"
echo "  EMPTY=0.0: $EMPTY0_LOG"
echo ""
echo "Compare the 'Initial Evaluation' results to see the PPL impact."
