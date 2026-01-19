# Configuration and Serialization

This document describes the serialization formats used for configuration files and model checkpoints.

## Format Strategy

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
