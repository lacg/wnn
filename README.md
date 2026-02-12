# ram-wnn

**Weightless Neural Networks for Language Modeling** — RAM-based neurons in PyTorch.

[![PyPI](https://img.shields.io/pypi/v/ram-wnn)](https://pypi.org/project/ram-wnn/)
[![Python](https://img.shields.io/pypi/pyversions/ram-wnn)](https://pypi.org/project/ram-wnn/)
[![License](https://img.shields.io/pypi/l/ram-wnn)](https://github.com/lacg/wnn/blob/main/LICENSE)

This research explores whether **Weightless Neural Networks** (WNNs) — specifically RAM-based neurons — can serve as a foundation for language modeling, traditionally dominated by weighted transformer architectures.

RAM neurons use **lookup tables** instead of weighted connections. Partial connectivity is the generalization mechanism: each neuron observes a subset of input bits, so similar inputs map to the same address and trigger the same response.

## Installation

```bash
pip install ram-wnn
```

Requires Python 3.11+ and PyTorch 2.0+.

## Quick Start

```python
from wnn.ram.core.models.bitwise_ramlm import BitwiseRAMLM
from wnn.eval import WIKITEXT2_TEST

# Create a per-bit output language model (16 clusters for GPT-2's 16-bit vocab)
model = BitwiseRAMLM(
    vocab_size=50257,      # GPT-2 tokenizer
    context_size=4,        # 4-gram context
    neurons_per_cluster=200,
    bits_per_neuron=20,
)

# Load WikiText-2 test data (GPT-2 BPE tokenizer)
tokens = WIKITEXT2_TEST.load_tokens()

# Train
model.train_on_tokens(tokens[:200_000])

# Evaluate
stats = model.evaluate_fast(tokens[200_000:])
print(f"CE: {stats['cross_entropy']:.2f}, PPL: {stats['perplexity']:.0f}")
```

## Key Results

All models evaluated on WikiText-2 with GPT-2 tokenizer (50,257 vocab).

| Architecture | CE | PPL | Acc | Notes |
|---|---|---|---|---|
| Random baseline | 10.82 | 50,257 | 0.002% | Uniform prediction |
| Tiered RAMLM (50K clusters) | ~10.20 | ~27,000 | ~4.9% | 5-tier, EMPTY=0.0 |
| **BitwiseRAMLM (16 clusters)** | **~9.15** | **~9,400** | **~6.6%** | Per-bit prediction |
| | | | | |
| *Target: GPT-2 Small (124M)* | *3.38* | *29.41* | *--* | *Zero-shot* |

## Architecture

### BitwiseRAMLM

Instead of 50K output clusters (one per token), BitwiseRAMLM uses **16 clusters** (one per output bit). Each cluster predicts P(bit_i=1 | context). Token probabilities are reconstructed via log-product:

```
log P(token=t) = Σ_i [b_i(t)·log(P_i) + (1-b_i(t))·log(1-P_i)]
```

Key advantage: every neuron sees ALL training examples (not just ~20 for rare tokens).

### Tiered RAMLM

Frequency-based architecture where frequent tokens get more capacity:

| Tier | Tokens | Neurons | Bits | Data % |
|------|--------|---------|------|--------|
| 0 | 50 most frequent | 15 | 20 | 42% |
| 1 | Next 50 | 13 | 18 | 5% |
| 2 | Next 400 | 9 | 10 | 13% |
| 3 | Next 20K | 7 | 9 | 37% |
| 4 | Rest (~30K) | 5 | 8 | 3% |

## Reproducibility

Save and load checkpoints for reproducible results:

```python
from wnn.eval import Checkpoint, WIKITEXT2_TEST

# Save (connections only = ~381 KB)
Checkpoint.save(
    path="checkpoints/my_model",
    model=model,
    eval_task=WIKITEXT2_TEST,
    results={"ce": 9.15, "ppl": 9430, "accuracy": 0.066},
)

# Load and reconstruct
ckpt = Checkpoint.load("checkpoints/my_model")
model = ckpt.create_model()
print(ckpt.summary())
```

## Rust+Metal Accelerator (Optional)

For Apple Silicon Macs, a Rust accelerator provides 3-8x speedup using Metal GPU compute:

```bash
# Requires Rust toolchain
cd src/wnn/ram/strategies/accelerator
pip install maturin
maturin develop --release

# Verify
python -c "import ram_accelerator; print(ram_accelerator.cpu_cores())"
```

The accelerator is optional — all functionality works with pure PyTorch.

## Research Blog

Follow the research progress at [lacg.github.io/llm-optimizer](https://lacg.github.io/llm-optimizer/).

## Citation

```bibtex
@software{garcia2025ramwnn,
  author = {Garcia, Luiz Alberto Crispiniano},
  title = {RAM-WNN: Weightless Neural Networks for Language Modeling},
  year = {2025},
  url = {https://github.com/lacg/wnn}
}
```

## License

MIT
