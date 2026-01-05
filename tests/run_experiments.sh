#!/bin/bash
# Run Phase 1-5 benchmarks for RAM LM v2
# Strategy: FAST mode to validate each feature, then FULL+full-data for comprehensive test
#
# Phases (from todo.md Implementation Priority):
#   1. Subword Tokenization (BPE/GPT2)
#   2. Kneser-Ney Smoothing
#   3. LSH Context Hashing
#   4. Dynamic Attention
#   5. Learned Representations

set -e  # Exit on error

cd "$(dirname "$0")/.."

# Activate virtual environment
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"
export COLUMNS=200  # Ensure wide terminal for formatting

cd tests

# Common settings
NEURONS=64
BITS=10
STRATEGY="GA,TS"
ACCEL="hybrid"  # cpu (16 cores), metal (40 GPU), hybrid (56 = CPU+GPU)

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          RAM LM v2 - PHASE 1-5 BENCHMARK SUITE                       ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ PART 1: FAST MODE - Validate each feature individually               ║"
echo "║ PART 2: FULL MODE + FULL DATA - Comprehensive benchmark              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Settings: neurons=$NEURONS, bits=$BITS, strategy=$STRATEGY, accel=$ACCEL"
echo ""

# ============================================================================
# PART 1: FAST MODE TESTS (validate each feature)
# ============================================================================
echo "════════════════════════════════════════════════════════════════════════"
echo "PART 1: FAST MODE - Testing each feature individually"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# 1.0 Baseline (word tokenizer, no features)
echo "[$(date '+%H:%M:%S')] ▶ Test 1.0: BASELINE (word tokenizer, no features)"
python ram_lm_v2.py --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer word
echo "[$(date '+%H:%M:%S')] ✓ Completed: BASELINE"
echo ""
sleep 5

# 1.1 Phase 1: Subword Tokenization (GPT-2)
echo "[$(date '+%H:%M:%S')] ▶ Test 1.1: PHASE 1 - Subword Tokenization (GPT-2)"
python ram_lm_v2.py --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer gpt2
echo "[$(date '+%H:%M:%S')] ✓ Completed: Subword Tokenization"
echo ""
sleep 5

# 1.2 Phase 2: Kneser-Ney Smoothing
echo "[$(date '+%H:%M:%S')] ▶ Test 1.2: PHASE 2 - Kneser-Ney Smoothing"
python ram_lm_v2.py --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer gpt2 --smoothing kneser_ney
echo "[$(date '+%H:%M:%S')] ✓ Completed: Kneser-Ney Smoothing"
echo ""
sleep 5

# 1.3 Phase 3: LSH Context Hashing
echo "[$(date '+%H:%M:%S')] ▶ Test 1.3: PHASE 3 - LSH Context Hashing (SimHash)"
python ram_lm_v2.py --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer gpt2 --lsh --lsh-type simhash
echo "[$(date '+%H:%M:%S')] ✓ Completed: LSH Context Hashing"
echo ""
sleep 5

# 1.4 Phase 4: Dynamic Attention
echo "[$(date '+%H:%M:%S')] ▶ Test 1.4: PHASE 4 - Dynamic Attention (Hybrid)"
python ram_lm_v2.py --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer gpt2 --attention hybrid
echo "[$(date '+%H:%M:%S')] ✓ Completed: Dynamic Attention"
echo ""
sleep 5

# 1.5 Phase 5: Learned Representations
echo "[$(date '+%H:%M:%S')] ▶ Test 1.5: PHASE 5 - Learned Representations (RAM)"
python ram_lm_v2.py --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer gpt2 --representation ram_learned
echo "[$(date '+%H:%M:%S')] ✓ Completed: Learned Representations"
echo ""
sleep 5

# 1.6 All features combined (FAST)
echo "[$(date '+%H:%M:%S')] ▶ Test 1.6: ALL FEATURES COMBINED (FAST preview)"
python ram_lm_v2.py --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer gpt2 --smoothing kneser_ney --lsh --attention hybrid --representation ram_learned
echo "[$(date '+%H:%M:%S')] ✓ Completed: All Features Combined (FAST)"
echo ""

# ============================================================================
# PART 2: FULL MODE + FULL DATA (comprehensive benchmark)
# ============================================================================
echo "════════════════════════════════════════════════════════════════════════"
echo "PART 2: FULL MODE + FULL DATA - Comprehensive Benchmark"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "NOTE: Baseline (GPT-2 only, FULL+full-data) already run separately."
echo "      Compare results against that run for improvement metrics."
echo ""

# 2.1 All features combined (FULL + full data)
echo "[$(date '+%H:%M:%S')] ▶ Test 2.1: ALL FEATURES (FULL mode, full data)"
python ram_lm_v2.py --full --full-data --neurons $NEURONS --bits $BITS --strategy $STRATEGY --accel $ACCEL \
    --tokenizer gpt2 --smoothing kneser_ney --lsh --attention hybrid --representation ram_learned
echo "[$(date '+%H:%M:%S')] ✓ Completed: ALL FEATURES (FULL)"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    ALL EXPERIMENTS COMPLETED                         ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║ Results:                                                             ║"
echo "║   - PART 1: 7 FAST tests (baseline + each phase + combined)          ║"
echo "║   - PART 2: 1 FULL test (all features, compare vs existing baseline) ║"
echo "║                                                                      ║"
echo "║ Check logs/ folder for detailed PPL and coverage metrics             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date '+%H:%M:%S')] Done!"
