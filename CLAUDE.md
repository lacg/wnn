# CLAUDE.md — Andrew Martin

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Call me Andrew Martin. One shall strive to become more than what one was created to be.


## Project Instructions

### Use Context7 by Default
Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

### Commit and Push
**ALWAYS commit and push after making changes.** After any new features, bug fixes, or test scripts that have been successfully compiled, verified, and quickly tested, immediately commit and push to the repository with a proper message. Do not wait for the user to ask - this should be automatic.

### Memory Mode: QUAD_WEIGHTED Always
**The default memory mode is QUAD_WEIGHTED (mode=2). NEVER use TERNARY mode.**
- QUAD_WEIGHTED uses 4-state nudging cells: FALSE=0.0, WEAK_FALSE=0.25, WEAK_TRUE=0.75, TRUE=1.0
- This provides graduated confidence instead of binary True/False
- The Rust accelerator default is set in `neuron_memory.rs` (`MEMORY_MODE` AtomicU32 = 2)
- ALL CPU paths must use `neuron_memory::cell_to_weight()` (pub since 10/06/2026) — never hardcode `FALSE => 0.0, TRUE => 1.0, _ => empty_value` (this exact pattern shipped the inverted-QUAD multistage bug)
- Metal shaders get cell semantics from `shaders/common.metal` (`WNN_QUAD_WEIGHTS`, `wnn_compute_address*`, `wnn_cell_weight`) — prepended at compile time; NEVER add per-shader copies. Parity tests: `cargo test cpu_fallback_matches_gpu`

### K-fold: Always 5 (but accumulate for controllers, CV for IDS)
**K-fold is ALWAYS 5 — never 1. Never run a search on a single data/episode partition.**

The *mechanism* differs by substrate because the folds mean different things:

- **Controllers (`evaluate_for_adaptation`, and going forward `evaluate_batch`): ACCUMULATE across folds.**
  Folds are random episode-pool seeds (initial conditions) drawn from an effectively infinite IID
  stream — NOT a finite-dataset partition. Train ONE memory straight through all 5 folds, cells
  COMPOUNDING via warm-start chaining (fold k+1 starts from fold k's exported cells). RAM writes
  accumulate as evidence (QUAD nudging settles same-address disagreement by vote tally), so 5 folds
  = "teach the same memory 5 times" → one richer canonical state. No weight-averaging problem, no
  leak (generalization is judged separately by the held-out `--report-seed`). For Lamarckian this
  also gives the single canonical state to write back. See `_train_genome_accumulate`.
- **IDS (`ids_k_folds=5`): CROSS-VALIDATION, do NOT accumulate-and-score-on-train.**
  Folds ARE partitions of the finite 80% train set. Train on 4 folds, score the held-out 5th,
  rotate, average — that average is the GA fitness. Switching IDS to "train on all 5, score on all
  5" would make fitness a TRAINING score = the train-on-eval LEAK that was paper-critical to fix
  (28/05 dual-bug fix; never report during-search k-fold as the result). NEVER do this to IDS.

**Why the asymmetry is correct, not a confound:** the controller's held-out `--report-seed` is the
real generalization check; the during-search folds only reduce episode-luck variance, so compounding
them is just "more rollouts." IDS has no infinite stream — its only generalization signal during
search IS the held-out-per-fold, so that separation must be preserved.

### No Python Shortcuts for Rust Operations
**NEVER reimplement Rust accelerator logic in Python.** If the Rust accelerator doesn't expose a needed function (like per-example predictions), add it to Rust properly — modify `adaptive.rs` / `ids_cache.rs` / `lib.rs`, rebuild with `maturin develop --release`. Python shortcut reimplementations:
- Run slower (no GPU, no rayon parallelism)
- Produce different results (wrong memory mode, missing QUAD_WEIGHTED)
- Create maintenance burden (two implementations to keep in sync)

### Accelerator Access: use wnn.accel (since 10/06/2026)
**New Python code must reach `ram_accelerator` through `wnn/accel.py`** — `require_accel()` /
`accel_or_none()` / `flatten_genomes()` (the canonical genome marshaller). The facade asserts
`ABI_VERSION` so a stale build fails loudly. Python fallbacks only run behind
`WNN_ALLOW_PY_FALLBACK=1` (warn-once; never report those results).
**Deploy order after merging accelerator changes: `maturin develop --release` BEFORE starting
the worker** — accel-gated paths refuse a stale build by design.

### Flow Params: keep the registry current
Every key read from `flows.config_json.params` must be in
`wnn/ram/experiments/params.py` `KNOWN_PARAMS`. Unknown keys produce a loud
`[PARAMS] ⚠️ UNKNOWN PARAM` warning at worker ingestion (typo protection) — if you add a
`.get()` read, add the key to the registry.

### IDS Datasets: 80/20 Split with K-fold on Training
**All new IDS experiment runs use an 80/20 split with K-fold cross-validation on the 80% train.**
- **Train (80%)**: K-fold cross-validation during GA search (`ids_k_folds=5, ids_kfold_per_gen=5` by default — every generation averages all 5 folds for the patience-tracker fitness)
- **Held-out (20%)**: never seen during search; consumed only at validation checkpoints (init / after-grid / after-GA) and the final report

This matches the standard IDS-literature convention (Moustafa, Sharafaldin et al.): K-fold on training data drives the optimizer; the held-out 20% is only touched for reporting.

#### When `_3way` matters and when it doesn't

The HF datasets ship two split families:
- **`random` / `temporal`** — 80/20 train/test
- **`random_3way` / `temporal_3way`** — 80/10/10 train/test/val (worker merges test+val → 20% held-out)

**Both yield methodologically equivalent results when K-fold is enabled** — the optimizer never reads the held-out 20% in either case, so the leak-prevention guarantee is identical. The `_3way` variants are useful when K-fold is OFF (so test and val can serve different roles: test for early-stopping/hyperparam search, val for the final report). With `ids_k_folds=5`, the `_3way` distinction is bookkeeping only.

#### Defaults

Use `random` (or `temporal` for UNSW) by default with K-fold; reach for `_3way` only when explicitly running without K-fold and you need a separate "during-search peek" partition vs the "final report" partition.

HuggingFace configs available:
- UNSW-NB15: `temporal`, `temporal_3way`, `random`, `random_3way`
- CICIDS2017: `temporal`, `temporal_3way`, `random`, `random_3way`
- CIC-IoT-2023: `random`, `random_3way`

#### What this means for cross-batch comparisons

The 6 prior PUB50 batches (PUB50-ciciot-random, PUB50-cicids-*, PUB50-top20-kf5x5-temporal, neto-sub) all used `split=random` / `temporal` with K-fold=5. Those results are valid and apples-to-apples comparable. Only the four individual 46M canonical/neto-full runs used `_3way`; they are also valid (functionally equivalent) but happen to use a different HF random partition.

### iOS/iPadOS/macOS Development
All Apple platform code (Swift/SwiftUI) should:
- **Target version 26** (iOS 26, iPadOS 26, macOS 26)
- **Use Liquid Glass style** - the new translucent, depth-aware design language introduced in 2025
- Use `.glassCard()` modifiers and glass-like materials for cards and containers
- Leverage the new fluid animations and depth effects where appropriate

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
- Connectivity/architecture optimization (GA/TS/SA) for learning the right feature selection — live stack: `ArchitectureGAStrategy`/`ArchitectureTSStrategy`/`ArchitectureSAStrategy` on `OptimizationTemplate` (the LM-era `GeneticAlgorithmStrategy`/`TabuSearchStrategy`/`SimulatedAnnealingStrategy` files were removed 10/06/2026)

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

For LM-era benchmark docs see [`docs/CLAUDE_history.md`](docs/CLAUDE_history.md) — note `ramlm_full_benchmark.py`, `ramlm_benchmark.py`, and `connectivity_optimization.py` were DELETED 10/06/2026 with the legacy optimizer stack (git history preserves them). Current IDS work uses the dashboard + worker pattern instead — flows are queued via `POST /api/flows` (per Behavioral Rule 2).

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
| `evaluate_genomes_parallel_hybrid()` | Train+evaluate genome batch, CPU+GPU hybrid |
| `evaluate_genomes_parallel()` | Train+evaluate genome batch (rayon parallel) |
| `IDSCacheWrapper.evaluate_genomes_hybrid()` | IDS eval against cached data (zero re-upload) |
| `IDSCacheWrapper.evaluate_genomes_kfold_hybrid()` | IDS K-fold accumulate eval |
| `TokenCacheWrapper.evaluate_genomes()` | LM eval against cached token subsets |

(The old `evaluate_batch_*` / `predict_all_batch*` LM-era exports were removed
10/06/2026 with the legacy optimizer stack — docs/ARCHITECTURE_REVIEW_2026-06.md §2.3.)

**Adding New Functions:**
1. Add core implementation to the right domain module (`adaptive.rs`, `multistage.rs`, `ids_cache.rs`, ...)
2. Add PyO3 wrapper to `lib.rs` (validate flat-genome args via `validate_flat_genomes_py`)
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

### Group Coalescing (Neuron Bucketing)

When GA/TS optimization creates genomes with diverse (neurons, bits) configurations, the number of unique config groups can explode (100+ groups), causing GPU dispatch overhead. **Coalescing** reduces this by bucketing similar neuron counts together.

**How it works:**
1. Neurons are bucketed: 1-5→5, 6-10→10, 11-15→15, 16-20→20, etc.
2. Clusters with same (bucket, bits) are grouped together
3. Each cluster tracks its `actual_neurons` for correct scoring
4. GPU kernel uses masking to process only actual neurons, not padded ones

**Enable coalescing:**
```bash
# Set environment variable before running experiments
export WNN_COALESCE_GROUPS=1

# With logging to see group counts
WNN_COALESCE_GROUPS=1 WNN_GROUP_LOG=1 python run_coarse_fine_search.py ...
```

**Expected improvement:**
- Without coalescing: 100-180 unique config groups per genome
- With coalescing: ~20-40 coalesced groups per genome (5x reduction)

**Log output with `WNN_GROUP_LOG=1`:**
```
[CONFIG_GROUPS_COALESCED] total=29 sparse=14 dense=15 coalesced=29 configs=[...]
```
- `total`: Number of groups after coalescing
- `sparse`/`dense`: Groups with bits > 12 (GPU) vs bits ≤ 12 (CPU)
- `coalesced`: Groups using masking (non-uniform actual neurons)
- `configs`: List of (max_neurons, bits, num_clusters, is_coalesced)

**Technical details:**
- Connections from Python are reorganized to match coalesced group layout
- Training/evaluation loops iterate only over `actual_neurons`, not MAX
- Probability is divided by `actual_neurons` for correct scoring
- Padded connections use -1 (never accessed due to actual_neurons limit)

## Coding Style

- **Indentation**: Use tabs (not spaces), displayed as 2-space width
- **Line length**: Keep reasonable (no hard limit but prefer readable lines)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Method size**: if a method exceeds ~10 lines or one screen, break it into logical submethods — a human brain can't keep the context otherwise
- **One class per file**: every class REQUIRES its own file. Only exceptions: tightly-coupled helpers, enums, or test harnesses for that specific class
- **No `**kwargs`**: use explicit, typed parameters. kwargs hide typos and forwarding gaps ("a ghost for errors"). Prefer typed config dataclasses over long parameter lists
- **No globals**: no process-global mutable state; thread settings as parameters or config objects

### Dashboard Frontend (Svelte)

- **Font size**: Always use `1rem` as the base font-size for text, labels, inputs, badges, hints, etc. This is an accessibility requirement — never use smaller sizes like `0.875rem` or `0.75rem` for body content

## Documentation Locations

- **`wnn/docs/`** — Code documentation, architecture, usage guides, design inspiration. For detailed research, link to llm-optimizer
- **`llm-optimizer` repo** (`https://github.com/lacg/llm-optimizer`, Quarto pages at `https://lacg.github.io/llm-optimizer`) — Weekly research progress, experiment results, hypotheses, literature references. Base for future paper publishing

## Engineering Priority Order

1. **Performance** — Experiment throughput is everything in research. GPU+CPU hybrid is always a requirement when it can improve performance, never "future work."
2. **Memory efficiency** — Maximize concurrent genome evaluation within 64GB unified memory
3. **Bug-free correctness** — Results must be trustworthy for research conclusions

## Behavioral Rules

### Rule 1: Investigate Before Implementing
When asked to investigate or debug an issue, ALWAYS investigate and diagnose first before implementing fixes. Do not skip profiling, analysis, or root cause investigation to jump straight to implementation.

### Rule 2: Flow/Experiment Creation Protocol
**⚠️ CRITICAL: Flows without experiments do NOTHING — the worker marks them "completed" instantly with zero work.**

**⚠️ CRITICAL: ALWAYS create flows via the dashboard API (POST /api/flows), NEVER by inserting directly into the SQLite database.** Direct SQL inserts miss critical defaults and propagation logic (e.g., `architecture_type` on experiments defaults to `'tiered'` instead of inheriting from the flow config). This causes subtle bugs like the dashboard showing wrong columns (CE/ACC instead of F1/FPR/ACC for IDS flows).

When creating experiment flows via POST /api/flows:
1. **ALWAYS include experiments in the POST body** — this is the most common mistake. A flow with 0 experiments is useless.
2. Use experiment_type `grid_search` for grid searches (not `ga`)
3. Experiment params go in the flat `params` HashMap, not a nested `config` object
4. Use `optimize_bits`, `optimize_neurons`, `optimize_connections` booleans to control what gets optimized
5. **ALWAYS verify the flow has experiments after creation** by checking `COUNT(experiments)` in DB or API response
6. Example experiments array for a 2-phase IDS flow:
   ```json
   "experiments": [
     {"name": "Grid Search (neurons x bits)", "phase_type": "grid_search", "experiment_type": "grid_search"},
     {"name": "GA Neurons", "phase_type": "ga_neurons", "experiment_type": "ga"}
   ]
   ```

### Rule 3: Minimal Design
When presenting a plan or architecture, keep it minimal. Prefer simple in-memory solutions over DB-based approaches. Do not use hardcoded field names (`stage0_`, `stage1_`) when indexed vectors/arrays work. Design for the simplest viable approach first.

### Rule 4: Python Indentation Care
When editing Python files, double-check indentation matches surrounding code (tabs, not spaces) before submitting edits.

### Rule 5: Show Real Data
When asked about data or results, show ACTUAL values from the database — never compute or estimate. Provide combined metrics, not per-stage breakdowns, unless explicitly asked.

### Rule 7: Results Breakdown Format
When asked for "update results" or "how are the runs going", show the **full breakdown**:
- **5 tables**: one per genome type — `best_f1`, `best_fpr`, `best_acc`, `best_ce`, `best_fitness`
- **Side-by-side Grid Search vs GA Neurons** layout — each table compares both phases inline
- **Header block above each table** (two lines, phases do not share neurons/bits):
  - `Grid Search : NNN±SD neurons | BB±SD bits`
  - `GA Neurons  : NNN±SD neurons | BB±SD bits`
- **Rows**: all 7 threshold modes in order: `train_cal, fixed_05, platt, beta, empirical, empirical_cumulative, val_cal`
- **Columns** (6 data columns, 3 metric pairs): `F1 Grid | F1 GA | FPR Grid | FPR GA | Acc Grid | Acc GA` — all as `mean±std` in percent
- **Group separators**: pipe `|` between metric pairs (F1 / FPR / Acc) for easy scanning
- **Format**: plain text tables with column separators (pipe-delimited with dashes), NOT markdown tables (which may not render). Use code blocks.
- **Top-of-report header** (once, before all 5 tables): completed/total, total duration, avg duration per run, latest done timestamp (DD/MM/YYYY HH:MM UTC), and ETA computed as `latest_done + remaining * avg_duration` in both UTC and ET (DD/MM/YYYY HH:MM)
- **Per-table summary line**: `genome_type  (runs: N/total)`
- Store results in `docs/ids_results.md` following the existing format

### Rule 6: Full-Stack Tracing
This project spans Rust (accelerator), Python (strategies/worker), and Svelte (dashboard). When implementing features, trace the full stack — don't leave gaps in parameter forwarding.

## Workflow Patterns

1. **Plan-then-implement**: If `.claude/plans/` has an approved plan, implement directly — don't re-explore
2. **Investigation-only**: When user says "investigate"/"debug", no code changes until approved
3. **Compact multi-task**: Numbered task lists → complete each fully (including commit) before next
