#!/bin/bash
# Run a sequence of RAM LM v2 experiments with different configurations
# Each experiment is separated by 10s to avoid timestamp collisions

cd "$(dirname "$0")/.."

# Activate virtual environment
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

cd tests

echo "========================================"
echo "RAM LM v2 EXPERIMENT SEQUENCE"
echo "========================================"
echo "1. FULL 128×6 GA→TS"
echo "2. OVERNIGHT 128×6 GA→TS"
echo "3. OVERNIGHT 128×8 GA→TS"
echo "4. OVERNIGHT 128×10 GA→TS"
echo "5. OVERNIGHT 128×12 GA→TS"
echo "(all with validation holdout + pre/post test comparison)"
echo "========================================"
echo ""

# Experiment 1: FULL 128×6 GA→TS
echo "[$(date '+%H:%M:%S')] Starting: FULL 128×6 GA→TS"
python ram_lm_v2.py --full --neurons 128 --bits 6 --strategy GA,TS --tokenizer gpt2_bpe
echo "[$(date '+%H:%M:%S')] Completed: FULL 128×6 GA→TS"
echo ""

sleep 10

# Experiment 2: OVERNIGHT 128×6 GA→TS
echo "[$(date '+%H:%M:%S')] Starting: OVERNIGHT 128×6 GA→TS"
python ram_lm_v2.py --overnight --neurons 128 --bits 6 --strategy GA,TS --tokenizer gpt2_bpe
echo "[$(date '+%H:%M:%S')] Completed: OVERNIGHT 128×6 GA→TS"
echo ""

sleep 10

# Experiment 3: OVERNIGHT 128×8 GA→TS
echo "[$(date '+%H:%M:%S')] Starting: OVERNIGHT 128×8 GA→TS"
python ram_lm_v2.py --overnight --neurons 128 --bits 8 --strategy GA,TS --tokenizer gpt2_bpe
echo "[$(date '+%H:%M:%S')] Completed: OVERNIGHT 128×8 GA→TS"
echo ""

sleep 10

# Experiment 4: OVERNIGHT 128×10 GA→TS
echo "[$(date '+%H:%M:%S')] Starting: OVERNIGHT 128×10 GA→TS"
python ram_lm_v2.py --overnight --neurons 128 --bits 10 --strategy GA,TS --tokenizer gpt2_bpe
echo "[$(date '+%H:%M:%S')] Completed: OVERNIGHT 128×10 GA→TS"
echo ""

sleep 10

# Experiment 5: OVERNIGHT 128×12 GA→TS
echo "[$(date '+%H:%M:%S')] Starting: OVERNIGHT 128×12 GA→TS"
python ram_lm_v2.py --overnight --neurons 128 --bits 12 --strategy GA,TS --tokenizer gpt2_bpe
echo "[$(date '+%H:%M:%S')] Completed: OVERNIGHT 128×12 GA→TS"
echo ""

echo "========================================"
echo "[$(date '+%H:%M:%S')] ALL EXPERIMENTS COMPLETED"
echo "========================================"
