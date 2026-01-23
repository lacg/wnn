# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Project Instructions

### Use Context7 by Default
Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

### Commit and Push
After any new features, or test scripts, successfully compiled, verified and quickly tested, commit and push to the repository with a proper message.

## Project Overview

This is a Weightless Neural Network (WNN) research project implementing RAM-based neurons in PyTorch. The goal is to create Transformer architectures using RAM neurons instead of traditional weighted neural networks.

## ⚠️ CRITICAL: Architecture Integrity

**Do NOT create any RAM neurons, or similar objects, without thorough discussion first.**

Always use the existing core architecture (`Memory`, `RAMLayer`, `RAMRecurrentNetwork`, etc.) from `wnn/ram/core/`. If the existing architecture is insufficient, or there's a better approach:
1. **Discuss first** - explain what's missing and why
2. **Propose alternatives** - don't just implement ad-hoc solutions
3. **Never** put ad-hoc implementations in test scripts that bypass the core architecture

The core architecture was methodically designed. Any new patterns should be deliberate extensions, not workarounds.

## 🧠 FOUNDATIONAL: How RAM WNNs Actually Learn

**This section is critical for understanding the project. Read it carefully.**

### The Two Components of Learning

RAM WNN learning requires BOTH:

| Component | What it does | Analogy to Weighted NN |
|-----------|--------------|------------------------|
| **Connectivity map** | Determines which input bits each neuron observes | Like learned weights - determines feature importance |
| **Memory writes** | Stores the actual input→output mappings | Like final weight values after training |

**The connectivity map is NOT a detail to hand-wave. It IS the generalization mechanism.**

### Why Partial Connectivity Enables Generalization

- Fully connected RAM = lookup table = memorization = NO generalization
- Partial connectivity = neurons see SOME bits = similar inputs share addresses = generalization

Example:
- Neuron sees bits [2, 5, 11] out of 48 total bits
- Many different inputs share the same values at positions [2, 5, 11]
- Those inputs map to the SAME address → trigger SAME response
- The neuron learned a **feature** (the pattern at those positions)
- New inputs with that feature → correct classification, even if never seen before

**This is the magic.** Not counting. Not dictionaries. The architecture itself generalizes.

### What is NOT a RAM WNN

Using Python dicts/Counters to count occurrences and compute probabilities is **NOT** a RAM WNN:
```python
# THIS IS NOT A RAM WNN - it's just n-gram counting:
self.ram = defaultdict(Counter)
self.ram[addr][target] += 1
prob = count / total
```

A real RAM WNN uses:
- `Memory` class with bit-packed cells (TRUE/FALSE/EMPTY)
- `RAMLayer` with proper partial connectivity
- EDRA backpropagation for training
- Connectivity optimization (GA/TS/SA) for learning the right feature selection

### Universality Principle

**Anything a weighted neural network can do, a weightless neural network can do.**

The difference is HOW:
- Weighted NN: learns via gradient descent on continuous weights
- RAM WNN: learns via connectivity optimization + memory writes

Both are universal function approximators. RAM WNNs achieve this through:
- Partial connectivity (feature selection)
- Multiple neurons with different connectivity (ensemble of perspectives)
- Output clustering (multiple neurons per class for probabilistic output)
- Layered architecture (composition of functions)

### Architecture Design Space for Language Modeling

**Input Encoding:** (existing infrastructure)
- Context tokens → bits via vocabulary encoding
- Existing classes in `wnn/tokenizers/` and `wnn/representations/`

**Output Encoding:** (needs design)
- Multiple neurons per output class (clustering)
- Interpretation of neuron outputs: 0=0.0, 1=1.0, 2(EMPTY)=0.5
- For 50K vocab: ~150-200K neurons (3-4 per class) - feasible with modern HW

**Architecture:**
- Typically 2-3 layers (input layer, output layer, optional state layer)
- Deeper = harder learning, diminishing returns
- Recurrent vs feedforward: depends on task
- Partial connectivity: initialize random, optimize with GA/TS/SA

**Training:**
- EDRA for backpropagation through layers
- Connectivity optimization for generalization
- Curriculum learning for complex tasks

### Current Status (as of 2026-01-06)

The `tests/ram_lm_v2.py` benchmark contains an ad-hoc `RAMNeuron` class that uses
Counter-based voting. **This is NOT using the real RAM WNN architecture.** It needs
to be redesigned to use `Memory`/`RAMLayer` with proper partial connectivity.

The path forward:
1. Design proper output encoding (OutputLayer with clustering)
2. Design the full architecture (layers, connectivity)
3. Implement using core `Memory`/`RAMLayer` classes
4. Train with EDRA + connectivity optimization

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

- **Frequent tokens** (tier 0) have enough training examples to fill large address spaces → more bits = better discrimination
- **Rare tokens** (tier 2) have too few examples → more bits = empty cells = random predictions

### Design Principle

**Match address space size to training data density:**
- High-frequency tiers → more bits (can utilize the capacity)
- Low-frequency tiers → fewer bits (can't fill large spaces anyway)

### Best Configuration So Far

```
tier0_20bit: 100,15,20;400,10,12;rest,5,8 (context=4)
- Test PPL: 36,853
- Test Accuracy: 5.28%
- Tier 0 Accuracy: 11.23%
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
# Format: "clusters,neurons,bits;clusters,neurons,bits;..."
# Use "rest" for remaining vocabulary

--tier-config "100,15,20;400,10,12;rest,5,8"
# tier0: 100 tokens with 15 neurons, 20 bits
# tier1: 400 tokens with 10 neurons, 12 bits
# tier2: rest (50K+) tokens with 5 neurons, 8 bits
```

### Phase Order Options

| Order | Sequence | Use Case |
|-------|----------|----------|
| `neurons_first` | neurons → bits → connections | Default, good for uniform starts |
| `bits_first` | bits → neurons → connections | Better for tiered configs |

### Tier0-Only Optimization

When `--tier0-only` is set, only the most frequent tokens (tier0) are mutated during GA/TS optimization. This is useful because:

- **Tier0 has most data**: ~46% of training examples for just 100 tokens
- **Higher data density**: ~11,000 examples per token can fill 2^20 addresses
- **Faster convergence**: Fewer parameters to optimize (100 vs 50,000 clusters)

### Example Commands

```bash
# Run with tiered config, bits-first order, tier0-only optimization
python run_phased_search.py \
  --tier-config "100,15,20;400,10,12;rest,5,8" \
  --phase-order bits_first \
  --tier0-only \
  --ga-gens 100 --ts-iters 200 --patience 10 \
  --output experiments/tier0_bits_first.json
```

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
- 4 tokens × 16 bits = 64 bits = 2^64 possible addresses
- 8 tokens × 16 bits = 128 bits = 2^128 possible addresses (impossible to fill)

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
- Each neuron's connectivity defines which bits it "attends to"
- Optimization finds the most informative bit subsets
- Unlike transformers, this is fixed per neuron (not dynamic per input)

## 🚨 Fundamental Limitation & Future Direction

**Key finding (2026-01-21):** Pure RAM WNNs cannot match transformer LM performance due to mathematical barriers (address space explosion, no selective attention). State layers don't solve this—sequential lookups eliminate the speed advantage.

**Future direction:** Hybrid architecture using RAM for fast pattern caching + transformers for long-range dependencies.

📄 **Full analysis:** [`docs/RESEARCH_INSIGHTS.md`](docs/RESEARCH_INSIGHTS.md)

## 🎯 Fitness Calculator: Balancing CE and Accuracy

The architecture search optimizes for both **Cross-Entropy (CE)** and **Accuracy**. The fitness calculator determines how these are combined for ranking genomes.

### Fitness Calculator Types

| Type | Description | Elite Selection |
|------|-------------|-----------------|
| `CE` | Pure CE ranking (lower = better) | Dual elites: 10% by CE + 10% by Acc |
| `HARMONIC_RANK` | Weighted harmonic mean of ranks | Single elite: 20% by harmonic rank |

### Weighted Harmonic Mean Formula

```
WHM = (w_ce + w_acc) / (w_ce/rank_ce + w_acc/rank_acc)
```

Where:
- `rank_ce` = position when sorted by CE (1 = lowest CE = best)
- `rank_acc` = position when sorted by accuracy (1 = highest acc = best)
- `w_ce`, `w_acc` = weights (default 1.0 each)

### Example

| Genome | CE | Acc | CE Rank | Acc Rank | HM (w=1,1) | HM (w=1.2,1) |
|--------|-----|------|---------|----------|------------|--------------|
| A | 10.34 | 0.01% | 1 | 5 | 1.67 | **1.43** ← wins |
| B | 10.35 | 0.03% | 2 | 1 | **1.33** ← wins | 1.38 |

With equal weights, B wins (balanced). With `w_ce=1.2`, A wins (best CE matters more).

### Configuration

Weights are set in `GAConfig` and `TSConfig`:

```python
fitness_calculator_type: FitnessCalculatorType = FitnessCalculatorType.HARMONIC_RANK
fitness_weight_ce: float = 1.0   # Higher = CE matters more
fitness_weight_acc: float = 1.0  # Higher = Accuracy matters more
```

### Key Properties

- **Lower harmonic mean = better** (closer to rank 1 in both metrics)
- **Penalizes imbalance**: Being bad at either metric hurts the score
- **Rank-based**: Relative positions matter, not absolute values
- **Rankings can shift** when new genomes enter the population

## Development Hardware

**Mac Studio M4 Max (2025)**
- CPU: 16 cores
- GPU: 40 cores (Metal)
- Neural Engine: 16 cores
- RAM: 64GB unified memory

This hardware enables:
- Hybrid CPU+GPU acceleration (56 total compute cores)
- Large model training (~93MB for full LM architecture)
- Population-based optimization (50+ candidates feasible)

## Development Setup

**⚠️ CRITICAL: Use the correct virtual environment!**

The project has TWO venv directories - only use `wnn/`:
- ✅ **`wnn/`** - The correct venv with all dependencies and ram_accelerator installed
- ❌ **`.venv/`** - Old/incomplete venv, DO NOT USE

```bash
# From project root - ALWAYS use wnn/, never .venv/
cd /Users/lacg/Library/Mobile\ Documents/com~apple~CloudDocs/Studies/research/wnn
source wnn/bin/activate

# Verify you're in the right venv
which python  # Should show: .../wnn/bin/python

# Install the package in editable mode (if needed)
pip install -e src/wnn

# Set PYTHONPATH (required for running tests)
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

# Or use the convenience script
source activate.sh
```

## Running Tests

```bash
# Run the parity check experiment
python tests/parity_check.py

# Run KV memory tests
python tests/kv_memory.py

# Run systematic benchmarks for generalization strategies
python tests/benchmarks.py
```

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
- Priority 1-3: Quick/Standard/Extended sets (original experiments)
- Priority 4: Asymmetric experiments (based on key insight above)

### Running Coarse-Fine Search

The current main experiment runner is `run_coarse_fine_search.py` in the project root:

```bash
# From project root with correct venv
cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
source wnn/bin/activate

# Run with tiered config in background
nohup python -u run_coarse_fine_search.py \
  --tier-config "100,15,20;400,10,12;rest,5,8" \
  > run_coarse_fine.out 2>&1 &

# Monitor progress
tail -f run_coarse_fine.out

# Check log file for detailed genome progress
tail -f logs/2026/01/*/coarse_fine_pass1_*.log
```

## Project Structure

```
src/wnn/
├── attention/         # LM attention mechanisms (self-contained)
├── lsh/              # Locality-sensitive hashing (self-contained)
├── representations/  # Binary word representations (self-contained)
├── smoothing/        # N-gram smoothing (self-contained)
├── tokenizers/       # Tokenization (self-contained)
└── ram/              # Core RAM architecture
    ├── core/         # Memory, RAMLayer, networks, models + related enums/factories
    ├── cost/         # Cost calculators + CostCalculatorType enum + factory
    ├── strategies/   # Optimization strategies + related enums/factories
    ├── encoders_decoders/  # Decoders + OutputMode, PositionMode enums + factory
    └── architecture/ # Configuration specs (KVSpec)
```

**Important conventions:**
- Each module is **self-contained**: enums and factories live WITH their related code
- Import pattern: `from wnn.module import SomeType, SomeFactory` (not from central folders)
- Example: `from wnn.attention import AttentionType` not `from wnn.ram.enums import AttentionType`

## Architecture

### Core Components (src/wnn/ram/)

**Memory** (`Memory.py`): Low-level bit-packed storage for RAM neurons.
- Uses 2-bit cells (4 states: FALSE=0, TRUE=1, WEAK_FALSE, WEAK_TRUE) packed into int64 words
- 31 cells per word (62 bits used, 2 bits per cell)
- Implements constraint solving via beam search for EDRA backpropagation
- Connections define which input bits each neuron observes

**RAMLayer** (`RAMLayer.py`): Thin wrapper around Memory providing the neural layer interface.
- `forward()`: Boolean lookup returning TRUE cells as True, others as False
- `commit()`: Finalize a mapping (write even if cell is occupied)
- `explore()`: Write only to EMPTY or compatible cells
- `solve()`: Find input bits that produce desired outputs (no memory modification)

**RAMRecurrentNetwork** (`RAMRecurrentNetwork.py`): Two-layer recurrent network.
- State layer: observes [input_bits, previous_state_bits]
- Output layer: observes [current_state_bits]
- Trained via EDRA-BPTT (Error Detection and Reconstruction Algorithm through time)

**RAMKVMemory** (`kv_transformer.py`): Multi-head KV memory extending RAMRecurrentNetwork.
- Hard key routing: k_bits determine which head to read/write
- Query detection: value bits all zero indicates query operation
- State is partitioned into heads (num_heads = 2^k_bits)

### Supporting Components

**KVSpec** (`architecture/kvspec.py`): Configuration for key-value memory experiments.
- Defines k_bits, v_bits, window structure
- Episode generation for training (writes followed by query)

**Decoders** (`decoders/`): Output interpretation strategies.
- `RAW`: Direct boolean output
- `BITWISE`: Per-bit interpretation
- `HAMMING`: Hamming distance-based decoding

**Cost Calculators** (`cost/`): Selection strategies for constraint solving.
- `STOCHASTIC`: Probabilistic selection from candidates
- `ARGMIN`: Greedy selection of minimum cost

### Memory Model

RAM neurons use ternary memory (FALSE, TRUE, EMPTY) where EMPTY means "untrained". The 2-bit representation allows weak/strong variants:
- 00: FALSE (strong)
- 01: WEAK_FALSE (EMPTY alias, initial state)
- 10: WEAK_TRUE
- 11: TRUE (strong)

### Training Algorithm (EDRA)

EDRA (Error Detection and Reconstruction Algorithm) is a credit assignment method for RAM networks:
1. Forward pass records contexts at each timestep
2. If output incorrect, solve output layer constraints
3. Backpropagate desired states through time via state layer constraint solving
4. Commit solutions to memory only when constraints are satisfiable

**Training Modes** (`TrainingMode` enum):
- `GREEDY`: Train all layers in single backward pass (fast)
- `ITERATIVE`: Multiple passes until convergence (more accurate)
- `LAYERWISE`: Train one layer at a time (most controlled)
- `OUTPUT_FIRST`: Prioritize output layers

**Curriculum Learning** (`TrainingPhase` enum):
- `WARMUP`: Train on shortest/easiest sequences
- `MAIN`: Train on full dataset
- `REFINEMENT`: Focus on hard examples

**Enhanced RAMTrainer:**
```python
trainer = RAMTrainer(model, mode=TrainingMode.ITERATIVE, patience=5)
stats = trainer.train_curriculum(dataset, epochs_per_phase=5)
```

### Generalization Strategies

RAM neurons naturally memorize (DIRECT strategy) but we need generalization for unseen inputs.

**MapperStrategy** enum (in `wnn/ram/enums/generalization.py`):

| Strategy | Description | Generalization |
|----------|-------------|----------------|
| `DIRECT` | Pure memorization | 0% on unseen |
| `BIT_LEVEL` | Per-bit context learning | 95%+ on successor/copy |
| `COMPOSITIONAL` | Group-based decomposition | Limited |
| `HASH` | Locality-sensitive hashing | Limited |
| `RESIDUAL` | Identity + learned correction | 95%+ on successor/copy |

**Context Modes** for BIT_LEVEL/RESIDUAL:
- `CUMULATIVE`: Bits 0..i for output bit i
- `FULL`: All input bits for each output
- `LOCAL`: Window around position i
- `BIDIRECTIONAL`: Symmetric window before/after
- `CAUSAL`: Autoregressive (only previous bits)

**Usage:**
```python
from wnn.ram.core import MapperFactory, MapperStrategy, ContextMode

mapper = MapperFactory.create(
    strategy=MapperStrategy.BIT_LEVEL,
    n_bits=8,
    context_mode=ContextMode.CUMULATIVE,
)
```

**Benchmark results** (from `tests/benchmarks.py`):
- BIT_LEVEL: 100% on copy, 95% on successor, 64% on complement
- RESIDUAL: Same as BIT_LEVEL (uses it internally)
- DIRECT: 100% train, ~0% test (no generalization)

### RAM Transformer Block (`RAMTransformerBlock.py`)

A complete transformer block with attention and FFN layers, supporting both learned and computed operations.

**Architecture:**
```
Input → Attention → XOR Residual → FFN → XOR Residual → Output
```

**Attention Types (AttentionType enum):**
- `SOFT_RAM`: Standard learned attention (partial generalization)
- `POSITION_ONLY`: Position-based routing (100% generalization)
- `SORTING`: Computed sorting by token value (100% generalization)
- `MIN_MAX`: Find min/max token (100% generalization)
- `CONTENT_MATCH`: XOR-based content matching (100% generalization)

**FFN Types (FFNType enum):**
- Learned: `NONE`, `SINGLE`, `TWO_LAYER`, `BIT_LEVEL`
- Computed (100% generalization):
  - `INCREMENT`: value + 1
  - `DECREMENT`: value - 1
  - `ADD_MOD`: (value + k) mod N
  - `SUBTRACT_MOD`: (value - k) mod N
  - `ROT13`: (value + 13) mod 26
  - `NEGATE`: max_value - value

**Factory Functions:**
```python
create_copy_transformer()      # Copy task
create_shift_transformer()     # Shift right
create_reverse_transformer()   # Reverse sequence
create_sorting_transformer()   # Sort by value
create_increment_transformer() # Add 1 to each token
create_rot13_transformer()     # ROT13 cipher
create_caesar_transformer(N)   # Caesar cipher +N
create_multi_step_transformer(steps)  # Compose operations
```

### Attention Mechanisms

Unified attention interface with both learned and computed implementations.

**Base Interface** (`AttentionBase`):
```python
class AttentionBase(Module):
    def forward(self, tokens: list[Tensor], context: list[Tensor] | None = None) -> list[Tensor]:
        """context=None for self-attention, context=encoder_output for cross-attention"""
```

**Key Classes:**
- `RAMAttention`: Unified self/cross-attention (replaces separate classes)
- `SoftRAMAttention`: Voting-based soft attention with RAM lookups
- `ComputedSortingAttention`: Computed sorting with 100% generalization
- `ComputedMinMaxAttention`: Computed min/max finding
- `ComputedArithmeticFFN`: Computed arithmetic transformations

**Content Match Modes** (simplified):
- `XOR_EQUAL`: Attend if tokens match (computed, 100%)

### Generalization Strategy

The key insight is distinguishing between **learned** and **computed** operations:

| Operation Type | Generalization | Example |
|---------------|----------------|---------|
| Learned lookup | Limited to trained patterns | Content-based attention |
| Computed comparison | 100% (any tokens) | Sorting, min/max |
| Computed arithmetic | 100% (any tokens) | Increment, ROT13, Caesar |
| Position-based | 100% (any tokens) | Shift, reverse, copy |

See `docs/COMPUTED_OPERATIONS.md` for detailed documentation.

## Rust Accelerator

The project includes a Rust/Metal accelerator for high-performance RAM evaluation.

**Location:** `src/wnn/ram/strategies/accelerator/`

**⚠️ Building the Accelerator - Use Absolute Paths:**

```bash
# RECOMMENDED: Use absolute paths to avoid venv confusion
cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn"
unset CONDA_PREFIX  # Required if conda is active
source wnn/bin/activate
cd src/wnn/ram/strategies/accelerator
maturin develop --release

# Verify installation
python -c "import ram_accelerator; print(ram_accelerator.cpu_cores())"
```

**One-liner for rebuild (handles CONDA_PREFIX conflict):**
```bash
cd "/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn" && unset CONDA_PREFIX && source wnn/bin/activate && cd src/wnn/ram/strategies/accelerator && maturin develop --release
```

**Important:**
- ❌ **Never use `cargo build`** - it will fail with Python linking errors
- ✅ **Always use `maturin develop --release`** - handles PyO3 bindings correctly
- ❌ **Never use `.venv/`** - use `wnn/` venv only
- If you see "Both VIRTUAL_ENV and CONDA_PREFIX are set" error, run `unset CONDA_PREFIX` first

**Key Functions:**
| Function | Description |
|----------|-------------|
| `evaluate_batch_cpu()` | Evaluate connectivity patterns (rayon parallel) |
| `evaluate_batch_metal()` | Evaluate on Metal GPU |
| `predict_all_batch_cpu()` | Batch prediction with pre-trained RAMs |
| `predict_all_batch_metal()` | Batch prediction on GPU |
| `predict_all_batch_hybrid()` | Batch prediction CPU+GPU |

**Adding New Functions:**
1. Add function to `ram.rs` (core implementation)
2. Add PyO3 wrapper to `lib.rs`
3. Register in `#[pymodule]` block
4. Rebuild with `maturin develop --release`

### GPU Sparse Evaluation (Binary Search)

The accelerator supports GPU-accelerated evaluation for **sparse memory groups** (bits > 12) using sorted arrays and binary search:

**Key Components:**
- `SparseGpuExport`: GPU-friendly format with sorted arrays (keys, values, offsets, counts)
- `MetalSparseEvaluator`: Binary search on GPU from `metal_ramlm.rs`
- `evaluate_group_sparse_gpu()`: Batch evaluation for sparse groups

**Why Binary Search on GPU?**
- DashMap (used for training) is CPU-only (hash lookups don't parallelize well on GPU due to memory divergence)
- Sorted arrays + binary search = O(log n) lookups with coalesced memory access = GPU-friendly
- Training still uses DashMap on CPU, evaluation exports to sorted arrays for GPU

**Memory Format for GPU:**
```rust
pub struct SparseGpuExport {
    pub keys: Vec<u64>,      // Sorted addresses for all neurons
    pub values: Vec<u8>,     // Corresponding values
    pub offsets: Vec<u32>,   // Start index per neuron
    pub counts: Vec<u32>,    // Count of entries per neuron
    pub num_neurons: usize,
}
```

### Parallel Hybrid Evaluation

`evaluate_genomes_parallel_hybrid()` provides maximum throughput for GA/TS architecture search:

**Architecture:**
1. **Memory Pool**: Reusable memory instances (8 parallel) to avoid OOM
2. **Parallel Training**: Multiple genomes train concurrently using the pool
3. **GPU Batch Evaluation**: Multiple genomes evaluated in one Metal dispatch
4. **CPU+GPU Hybrid**: Dense groups (bits ≤ 12) on CPU, sparse groups (bits > 12) on GPU
5. **Pipelining**: CPU trains batch N+1 while GPU evaluates batch N

**Performance Benefits:**
- 4-8x speedup over sequential genome evaluation
- No memory contention (each genome has its own memory, DashMap is lock-free)
- Efficient use of M4 Max (16 CPU cores + 40 GPU cores)

**Function Signature:**
```rust
pub fn evaluate_genomes_parallel_hybrid(
    genomes_bits_flat: &[usize],        // Flattened bits per genome
    genomes_neurons_flat: &[usize],     // Flattened neurons per genome
    genomes_connections_flat: &[i64],   // Flattened connections per genome
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
) -> Vec<(f64, f64)>  // Returns (ce_loss, accuracy) per genome
```

## Coding Style

- **Indentation**: Use tabs (not spaces), displayed as 2-space width
- **Line length**: Keep reasonable (no hard limit but prefer readable lines)
- **Naming**: snake_case for functions/variables, PascalCase for classes
