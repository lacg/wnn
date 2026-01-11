#!/bin/bash
WNN_DIR="/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
SWEEP_OUT="$WNN_DIR/nohup_sweep.out"

echo "========================================"
echo "Monitoring sweep for completion..."
echo "========================================"

# Wait for sweep to complete
while true; do
    if grep -q "Best PPL:" "$SWEEP_OUT" 2>/dev/null || grep -q "SWEEP RESULTS" "$SWEEP_OUT" 2>/dev/null; then
        echo ""
        echo "========================================"
        echo "Sweep completed! Starting analysis..."
        echo "========================================"
        break
    fi
    
    # Show progress
    tail -1 "$SWEEP_OUT" 2>/dev/null | head -c 80
    echo -ne "\r"
    sleep 60
done

# Activate venv and run analysis
cd "$WNN_DIR"
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

# Run analysis (will prompt for confirmation before running hybrid)
python analyze_and_run_best.py
