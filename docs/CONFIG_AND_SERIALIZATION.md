# Phased Architecture Search

This document describes how to run phased architecture search experiments, including multi-pass optimization, checkpointing, and configuration.

## Running Experiments

### Basic Usage

```bash
# Activate environment
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Run a single pass with default settings
python run_coarse_fine_search.py --pass 1
```

### Phased Search Overview

Each pass runs through three optimization phases:

| Phase | Optimizes | Fixed | Description |
|-------|-----------|-------|-------------|
| **1a/1b** | Neurons per cluster | Bits | GA then TS to find optimal neuron counts |
| **2a/2b** | Bits per cluster | Neurons | GA then TS to find optimal bit widths |
| **3a/3b** | Connections | Architecture | GA then TS to optimize connectivity |

Each phase has two stages:
- **a** = Genetic Algorithm (exploration)
- **b** = Tabu Search (refinement)

### Multi-Pass Coarse-to-Fine Search

The recommended workflow uses multiple passes with increasing patience:

```bash
# Pass 1: Coarse exploration (low patience for quick results)
python run_coarse_fine_search.py --pass 1 --patience 2 \
    --checkpoint-dir checkpoints/run1 \
    --output results_pass1.json

# Pass 2: Refinement (higher patience, seed from pass 1)
python run_coarse_fine_search.py --pass 2 --patience 4 \
    --seed-from results_pass1.json \
    --checkpoint-dir checkpoints/run1 \
    --output results_pass2.json

# Pass 3: Final polish (lower patience to escape local minima)
python run_coarse_fine_search.py --pass 3 --patience 2 \
    --seed-from results_pass2.json \
    --checkpoint-dir checkpoints/run1 \
    --output results_pass3.json
```

### Seeding from Previous Results

Use `--seed-from` to start from a previous run's best genome:

```bash
# Seed from a results JSON file
python run_coarse_fine_search.py --seed-from results_pass1.json

# Seed from a checkpoint file
python run_coarse_fine_search.py --seed-from checkpoints/run1/phase_3b_ts_connections.json.gz
```

The loader automatically detects the file format and extracts the genome.

### Resuming from Checkpoints

If a run is interrupted, resume from any phase:

```bash
# Resume from phase 2a (bits GA)
python run_coarse_fine_search.py \
    --checkpoint-dir checkpoints/run1 \
    --resume-from 2a

# Valid resume points: 1a, 1b, 2a, 2b, 3a, 3b
```

Checkpoints are saved automatically after each phase completes.

### Running in Background

For long experiments, use nohup:

```bash
cd /path/to/wnn
source wnn/bin/activate
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

PYTHONUNBUFFERED=1 nohup python -u run_coarse_fine_search.py \
    --pass 1 --patience 4 \
    --ga-gens 100 --ts-iters 200 \
    --checkpoint-dir checkpoints/overnight \
    --output results_overnight.json > nohup.out 2>&1 &

# Monitor progress
tail -f nohup.out
```

### Key CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pass N` | Pass number (affects patience calculation) | 1 |
| `--patience N` | Early stopping patience | 2 × 2^(pass-1) |
| `--ga-gens N` | GA generations per phase | 100 |
| `--ts-iters N` | TS iterations per phase | 200 |
| `--population N` | GA population size | 50 |
| `--neighbors N` | TS neighbors per iteration | 50 |
| `--train-tokens N` | Training tokens | 200,000 |
| `--eval-tokens N` | Evaluation tokens | 50,000 |
| `--context N` | Context window size | 4 |
| `--ce-percentile F` | CE filter (0.75 = top 75%) | None |
| `--checkpoint-dir DIR` | Enable checkpointing | None |
| `--resume-from PHASE` | Resume from phase | None |
| `--seed-from FILE` | Seed genome from file | None |
| `--config FILE` | Load YAML config | None |

---

## Configuration and Serialization

### Format Strategy

We use a hybrid approach optimized for each use case:

| Use Case | Format | Why |
|----------|--------|-----|
| **Config files** | YAML | Human-readable, easy to edit |
| **Model saves/checkpoints** | JSON + gzip | Fast read/write, compact storage |

## Benchmark Results

Tested with realistic checkpoint data (40 genomes, 50,257 clusters each):

| Metric | JSON | YAML | Ratio |
|--------|------|------|-------|
| Write time | 2.3s | 223s | JSON **96x faster** |
| Read time | 2.3s | 283s | JSON **124x faster** |
| Raw size | 315 MB | 727 MB | YAML 2.3x larger |
| Gzipped | 81 MB | 90 MB | Similar |

**Key insight**: Both formats compress to similar sizes with gzip, but JSON is ~100x faster for serialization. For large model files, this difference is critical.

## YAML Configuration

### Using Config Files

```bash
# Run with YAML config
python run_coarse_fine_search.py --config configs/my_config.yaml

# CLI args override config file values
python run_coarse_fine_search.py --config configs/base.yaml --patience 8
```

### Example Config File

```yaml
# configs/example_search.yaml

# Data configuration
context_size: 4
token_parts: 3  # Number of subsets for rotation

# GA configuration
ga_generations: 100
population_size: 50

# TS configuration
ts_iterations: 200
neighbors_per_iter: 50

# Early stopping
patience: 4

# Architecture defaults
default_bits: 8
default_neurons: 5

# CE percentile filter (null = disabled)
ce_percentile: 0.75

# Random seed (null = time-based)
rotation_seed: null
```

### Programmatic Usage

```python
from wnn.ram.experiments import PhasedSearchConfig

# Create and save config
config = PhasedSearchConfig(
    context_size=4,
    patience=4,
    ce_percentile=0.75,
)
config.save_yaml("configs/my_config.yaml")

# Load config
config = PhasedSearchConfig.load_yaml("configs/my_config.yaml")

# Convert to/from YAML string
yaml_str = config.to_yaml()
config = PhasedSearchConfig.from_yaml(yaml_str)
```

## Compressed Model Saves

### Automatic Compression

`PhaseResult` and `ClusterGenome` automatically save as `.json.gz`:

```python
# Saves as checkpoint.json.gz (auto-adds .gz extension)
result.save("checkpoint.json")

# Loading works with both compressed and uncompressed
result, metadata = PhaseResult.load("checkpoint.json")      # tries .json.gz first
result, metadata = PhaseResult.load("checkpoint.json.gz")   # explicit
```

### Compression Ratio

Typical compression ratios for checkpoint files:

| Original Size | Compressed | Ratio |
|---------------|------------|-------|
| 786 MB | 70 MB | 91% reduction |
| 315 MB | 81 MB | 74% reduction |

### Backward Compatibility

The `load()` methods automatically detect format:
1. If path ends with `.gz` → decompress and load
2. If `.gz` version exists → load that
3. Otherwise → load as uncompressed JSON

This ensures old uncompressed files remain readable.

## File Extensions

| Extension | Description |
|-----------|-------------|
| `.yaml` | YAML config files |
| `.json` | Uncompressed JSON (legacy) |
| `.json.gz` | Gzip-compressed JSON (default for saves) |

## Checkpoints Directory

Checkpoints are saved to `checkpoints/` which is gitignored (files too large for GitHub). Structure:

```
checkpoints/
└── pass1_YYYYMMDD_HHMMSS/
    ├── phase_1a_ga_neurons.json.gz    # ~70 MB (was 786 MB)
    ├── phase_1b_ts_neurons.json.gz
    ├── phase_2a_ga_bits.json.gz
    └── ...
```

To compress existing uncompressed checkpoints:
```bash
gzip checkpoints/*/phase_*.json
```
