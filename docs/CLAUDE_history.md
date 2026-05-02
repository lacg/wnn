# CLAUDE.md — Archived Sections

Sections moved out of `CLAUDE.md` on 2026-05-01 to keep the live instructions
file under the recommended size threshold. These are LM-era contexts that
predate the IDS work — kept here for reference only.

---

## 🔬 KEY INSIGHT: Asymmetric Tiered Architecture
**Discovery (2026-01-11):** Asymmetric bit allocation dramatically outperforms uniform configurations for tiered language models.
### The Finding
| Configuration | Tier 0 | Tier 1 | Tier 2 | Test PPL |
|---------------|--------|--------|--------|----------|
| **Asymmetric (best)** | 20 bits | 12 bits | 8 bits | **36,853** |
| Uniform 20-bit | 20 bits | 20 bits | 20 bits | 49,675 |
The asymmetric config achieves **35% better PPL** than uniform.
### Why This Works
The key is **training data density per address space**:
| Tier | Tokens | Data % | Examples/Token | Can Fill |
|------|--------|--------|----------------|----------|
| Tier 0 | 100 frequent | 46% | ~11,000 | 2^20 addresses ✓ |
| Tier 1 | 400 medium | 13% | ~800 | 2^12 addresses ✓ |
| Tier 2 | 50K rare | 40% | ~20 | 2^8 addresses ✗ |
### Design Principle
**Match address space size to training data density:**
### Best Configuration So Far
```
tier0_20bit: 100,15,20;400,10,12;rest,5,8 (context=4)
```
See `experiments/overnight_sweep.md` for full rankings and per-tier breakdowns.
## 🔄 Phased Search: Configurable Optimization Order
The phased architecture search now supports flexible configuration for tiered architectures and optimization order.
### Configuration Options
| Option | CLI Flag | Description |
|--------|----------|-------------|
| **Tiered config** | `--tier-config` | Different bits/neurons per tier |
| **Phase order** | `--phase-order` | `neurons_first` or `bits_first` |
| **Tier0-only** | `--tier0-only` | Only mutate frequent tokens |
### Tier Config Format
```bash
## 🎯 Context Length: Why Transformers Scale, RAM WNNs Don't
### The Fundamental Difference
| Aspect | Transformer | RAM WNN |
|--------|-------------|---------|
| **Context handling** | Selective attention | All bits → address |
| **Context size** | 128K+ tokens | ~4-8 tokens optimal |
| **Scaling** | O(n²) memory, linear utility | Exponential address space |
| **Irrelevant tokens** | Ignored via low attention | ALL contribute to address |
### Why Transformers Scale with Context
Transformers use **selective attention** - they can:
1. Compute relevance scores between all token pairs
2. Attend strongly to relevant tokens, weakly to irrelevant ones
3. Dynamically focus on different parts of context for different queries
This means longer context = more opportunities to find relevant information, without penalty for irrelevant tokens.
### Why RAM WNNs Don't Scale with Context
RAM WNNs use **address-based lookup** - they:
1. Concatenate ALL context bits into an address
2. Look up that exact address in memory
3. Cannot ignore any bits - all contribute to the address
**The exponential problem:**
With limited training data, longer context = sparser address space = more EMPTY cells = worse predictions.
### Experimental Evidence
From overnight sweeps:
```
context=4:  Best PPL 36,853 ✓
context=8:  Higher PPL (worse)
context=16: Even higher PPL (even worse)
```
### Possible Paths for Longer Context
1. **Hierarchical compression**: Compress old context into summary bits
2. **Recurrent state**: Carry information forward in hidden state
3. **Sparse addressing**: Use LSH to select subset of context bits
4. **Multi-scale neurons**: Different neurons attend to different time scales
5. **Learned bit masking**: Gate which bits contribute to address
### Connection to Connectivity Optimization
The GA/TS connectivity optimization is actually implementing a form of **static attention**:
## 🚨 Fundamental Limitation & Future Direction
**Key finding (2026-01-21):** Pure RAM WNNs cannot match transformer LM performance due to mathematical barriers (address space explosion, no selective attention). State layers don't solve this—sequential lookups eliminate the speed advantage.
**Future direction:** Hybrid architecture using RAM for fast pattern caching + transformers for long-range dependencies.
📄 **Full analysis:** [`docs/RESEARCH_INSIGHTS.md`](docs/RESEARCH_INSIGHTS.md)
## Language Model Benchmarks
```bash
# Run Phase 1-5 benchmark suite
./tests/run_experiments.sh
# Manual runs with specific options
cd tests
# Basic run (FAST mode, sampled data)
python ram_lm_v2.py --tokenizer gpt2
# FULL mode with complete dataset
python ram_lm_v2.py --full --full-data --tokenizer gpt2
# With all LLM features (Phase 1-5)
python ram_lm_v2.py --tokenizer gpt2 --smoothing kneser_ney --lsh --attention hybrid --representation ram_learned
```
**Available Flags:**
| Flag | Options | Description |
|------|---------|-------------|
| `--tokenizer` | word, bpe, gpt2, char | Tokenization strategy |
| `--smoothing` | none, kneser_ney, backoff, add_k | N-gram smoothing |
| `--lsh` | (flag) | Enable LSH context hashing |
| `--lsh-type` | simhash, random_projection | LSH algorithm |
| `--attention` | none, position, content, hybrid, sparse | Dynamic attention |
| `--representation` | cooccurrence, mutual_info, ram_learned | Binary encoding |
| `--accel` | cpu, metal, hybrid | Hardware acceleration |
**Hardware Acceleration (M4 Max):**
| Mode | Cores | Description |
|------|-------|-------------|
| `--accel cpu` | 16 | Rust + rayon CPU parallelism |
| `--accel metal` | 40 | Metal GPU compute shaders |
| `--accel hybrid` | 56 | Both CPU + GPU in parallel |
```bash
# Use Metal GPU (40 cores)
python ram_lm_v2.py --accel metal --tokenizer gpt2
# Use Hybrid CPU+GPU (56 cores)
python ram_lm_v2.py --accel hybrid --full --full-data --tokenizer gpt2
```
## Running Overnight Sweeps
**⚠️ IMPORTANT:** Always use `wnn/` venv (NOT `.venv/`) with unbuffered output for background experiments.
```bash
# Activate the CORRECT venv and run overnight sweep in background
cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate  # ← MUST be wnn/, not .venv/
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"
# Quick sweep (4 experiments, ~4-6 hours)
PYTHONUNBUFFERED=1 nohup python -u tests/ramlm_full_benchmark.py \
  --sweep --set quick --full-data \
  --output experiments/sweep_results.json > nohup.out 2>&1 &
# Run specific experiments with weekend mode (1000 gens/iters, patience 5)
PYTHONUNBUFFERED=1 nohup python -u tests/ramlm_full_benchmark.py \
  --sweep --experiments asymmetric_extreme_t0,asymmetric_expanded_t0,two_tier_simple \
  --full-data --ga-gens 1000 --ts-iters 1000 --patience 5 \
  --output experiments/sweep_asymmetric.json > nohup.out 2>&1 &
# Monitor progress
tail -f nohup.out
# Check running experiments
ps aux | grep ramlm | grep -v grep
```
**Sweep Options:**
| Flag | Description |
|------|-------------|
| `--sweep` | Enable sweep mode (run multiple experiments) |
| `--set quick/standard/extended` | Experiment set (4/6/10/13 experiments by priority) |
| `--experiments name1,name2` | Run specific experiments by name |
| `--full-data` | Use full WikiText-2 dataset |
| `--ga-gens N` | GA generations (default: 50, weekend: 1000) |
| `--ts-iters N` | TS iterations (default: 100, weekend: 1000) |
| `--patience N` | Early stop patience (default: 1, weekend: 5+) |
| `--output FILE.json` | Output file for results |
| `--force-rerun` | Re-run completed experiments |
| `--no-optimize` | Disable GA+TS optimization |
**Experiment Priorities:**
### Running Coarse-Fine Search
**⚠️ IMPORTANT: This is the PRIMARY overnight experiment runner. Use `run_coarse_fine_search.py`, NOT `run_phased_search.py`.**
The current main experiment runner is `run_coarse_fine_search.py` in the project root:
```bash
# From project root with correct venv
cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate
# OVERNIGHT RUN: Tier0-only with asymmetric architecture (RECOMMENDED)
# - Uses best tier config: 100 tokens @ 20 bits, 400 @ 12 bits, rest @ 8 bits
# - Only optimizes tier0 (100 most frequent tokens) for faster convergence
# - Population/neighbors 50 for good diversity
# - Fitness percentile 0.75 keeps top 75% offspring by fitness (filters noisy low-performers)
# - ALWAYS use --checkpoint-dir to save progress and enable recovery from crashes
PYTHONUNBUFFERED=1 nohup python -u run_coarse_fine_search.py \
  --ga-gens 1000 \
  --ts-iters 1000 \
  --patience 10 \
  --population 50 \
  --neighbors 50 \
  --tier-config "100,15,20;400,10,12;rest,5,8" \
  --tier0-only \
  --fitness-percentile 0.75 \
  --checkpoint-dir checkpoints \
  > nohup.out 2>&1 &
# Monitor progress
tail -f nohup.out
# Check log file for detailed genome progress
tail -f logs/2026/01/*/coarse_fine_pass1_*.log
# Resume from checkpoint after crash or restart
PYTHONUNBUFFERED=1 nohup python -u run_coarse_fine_search.py \
  --resume checkpoints/latest_checkpoint.pkl \
  > nohup.out 2>&1 &
```
**⚠️ ALWAYS include `--checkpoint-dir` for overnight runs!** This saves progress every generation and allows resuming from the last checkpoint if the run crashes or is interrupted. Without checkpointing, hours of optimization can be lost.
