# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Weightless Neural Network (WNN) research project implementing RAM-based neurons in PyTorch. The goal is to create Transformer architectures using RAM neurons instead of traditional weighted neural networks.

## Development Setup

```bash
# Activate virtual environment
source wnn/bin/activate

# Install the package in editable mode
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

**Building the Accelerator:**
```bash
# Activate virtual environment first (required for Python linking)
source wnn/bin/activate

# Build and install with maturin
cd src/wnn/ram/strategies/accelerator
maturin develop --release

# Verify installation
python -c "import ram_accelerator; print(ram_accelerator.cpu_cores())"
```

**Important:** Never use `cargo build` directly - it will fail with Python linking errors. Always use `maturin develop --release` which handles Python bindings correctly.

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

## Coding Style

- **Indentation**: Use tabs (not spaces), displayed as 2-space width
- **Line length**: Keep reasonable (no hard limit but prefer readable lines)
- **Naming**: snake_case for functions/variables, PascalCase for classes
