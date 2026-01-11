#!/bin/bash
# Monitor current benchmark and start standard sweep when done

LOG_FILE="/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/logs/2026/01/09/ramlm_benchmark_20260109_224155.log"
WNN_DIR="/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"

echo "========================================"
echo "Monitoring current benchmark..."
echo "Log: $LOG_FILE"
echo "========================================"

# Wait for SESSION COMPLETE or check if process is still running
while true; do
    if grep -q "SESSION COMPLETE" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "========================================"
        echo "Current benchmark completed!"
        echo "========================================"
        break
    fi
    
    # Also check if the log file hasn't been updated in 5 minutes (stale)
    if [[ -f "$LOG_FILE" ]]; then
        last_mod=$(stat -f %m "$LOG_FILE" 2>/dev/null || stat -c %Y "$LOG_FILE" 2>/dev/null)
        now=$(date +%s)
        age=$((now - last_mod))
        if [[ $age -gt 300 ]]; then
            echo ""
            echo "WARNING: Log file hasn't been updated in $age seconds"
        fi
    fi
    
    # Show latest line
    tail -1 "$LOG_FILE" 2>/dev/null | head -c 100
    echo -ne "\r"
    sleep 30
done

# Show final results from current run
echo ""
echo "Final results from current run:"
grep -E "(Test PPL|Test Acc|Improvement)" "$LOG_FILE" | tail -5

# Start standard sweep
echo ""
echo "========================================"
echo "Starting standard sweep..."
echo "========================================"

cd "$WNN_DIR"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

# Run standard sweep in background with nohup
PYTHONUNBUFFERED=1 nohup python -u tests/ramlm_full_benchmark.py \
    --sweep --set standard \
    --output sweep_standard_$(date +%Y%m%d_%H%M%S).json \
    > nohup_sweep.out 2>&1 &

SWEEP_PID=$!
echo "Standard sweep started with PID: $SWEEP_PID"
echo "Monitor with: tail -f $WNN_DIR/nohup_sweep.out"
echo "========================================"
