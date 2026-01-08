# Overview

  WikiText-2 Dataset Structure:
  - train: ~2M tokens from Wikipedia articles (for training)
  - test: ~245k tokens from different Wikipedia articles (for final evaluation)
  - validation: ~214k tokens from different Wikipedia articles (for hyperparameter tuning)

  The key insight: these are completely different articles, not random splits of the same data. This explains why optimizing on train data doesn't generalize to test data!
  ─────────────────────────────────────────────────

# Next improvements

  ★ Insight ─────────────────────────────────────
  Why Modern LLMs Generalize Better

  | Feature                                           | RAM-based LM              | Modern LLMs (GPT/Claude)          |
  |---------------------------------------------------|---------------------------|-----------------------------------|
  | Tokenization                                      | Word-level (76k vocab)    | Subword BPE (~32k), handles OOV   |
  | Lookup                                            | Exact binary match        | Soft attention over embeddings    |
  | Context                                           | Fixed n-gram window (2-6) | Dynamic attention (any position)  |
  | Similarity                                        | None ("cat" ≠ "dog")      | Embedding space (cat ≈ dog)       |
  | Smoothing                                         | Cascade fallback          | Softmax + temperature + smoothing |
  | ───────────────────────────────────────────────── |                           |                                   |

  Key techniques we're missing:

  1. Subword tokenization (BPE): Modern LLMs break "unfamiliar" → "un" + "familiar". Your RAM model has 76k word vocabulary; unseen words = completely unknown. BPE has ~32k tokens and can compose ANY word.
  2. Embedding similarity: In LLMs, "cat" and "dog" are nearby in vector space. If model learned "cat ate fish", it generalizes to "dog ate fish". In RAMs, they're completely unrelated bit patterns.
  3. Attention mechanism: LLMs dynamically decide which context positions matter. Your RAM uses fixed 2-6 gram windows. LLMs can attend to position 1 AND position 100 if both are relevant.
  4. Softmax smoothing: LLMs output probability over ALL tokens. Even wrong predictions get partial credit. Your cascade gives 1/vocab for misses - harsh penalty.

  What could help your architecture:
  - Locality-sensitive hashing (LSH) for contexts → similar contexts hit same RAM address
  - Subword tokenization → smaller vocab, no OOV
  - Learned context embedding → compress context to bits that preserve similarity


# Limitations

  Why only 2-3k samples for optimization evaluation?

  The optimization loop evaluates every candidate connectivity (30 population members × 50 generations = 1500 evaluations). Each evaluation does a full forward pass through the model. Using the full 50k training tokens for each evaluation would be:
  - 50,000 tokens × 1,500 evaluations = 75 million token evaluations
  - That would take hours per optimization run

  The 2-3k sample is a trade-off for speed. But the real problem is that we're sampling from train data, not test/validation data.

# Quick tests

  Quick test results with WIKITEXT_WORD tokenizer:
  - Vocab: ~6,500 tokens (standard word-level)
  - Initial PPL: ~1482 (high - room for improvement)
  - Optimization running with perplexity as the goal

  Reference perplexity benchmarks on WikiText-2:
  | Model       | PPL      |
  |-------------|----------|
  | Trigram     | ~150-200 |
  | LSTM        | ~65-100  |
  | AWD-LSTM    | ~57      |
  | GPT-2 Small | ~29      |
  | GPT-2 Large | ~22      |


#  Ways to Lower PPL with RAM Architecture

  1. More Neurons Per Cluster (High Impact)

  Currently: 4 neurons → only 4 different "perspectives" per token

  neurons_per_cluster: 4 → 8, 16, or 32
  - More neurons = more patterns stored per vocab token
  - Each neuron sees different input bits (partial connectivity)
  - More coverage of the address space → less EMPTY cells

  2. Connectivity Optimization (What GA/TS is doing now)

  This is the "weight learning" for RAMs:
  - Determines which input bits each neuron observes
  - Better connectivity = neurons learn more discriminative features
  - This is why we're running GA/TS!

  3. Better Input Encoding

  Currently: Direct token ID → binary (16 bits per token)

  Could use:
  - LSH (SimHash): Group similar contexts together
  - Learned embeddings → binary: More semantic encoding
  - Position-weighted bits: Emphasize recent tokens

  4. Cascaded/Hierarchical Approach (like ram_lm_v2.py)

  if exact_4gram_match:     → use exact probability (high confidence)
  elif exact_3gram_match:   → use 3-gram fallback
  elif generalized_ram:     → use partial connectivity RAM
  else:                     → smoothing/backoff

  5. More Bits Per Neuron

  Currently: 10 bits = 1024 addresses

  Trade-off:
  - More bits → more specific patterns → better precision
  - But also sparser training → fewer examples per address

  6. Architecture Parameters Summary

  | Parameter           | Current | Can Increase | Trade-off               |
  |---------------------|---------|--------------|-------------------------|
  | neurons_per_cluster | 4       | 8-32         | Memory vs accuracy      |
  | bits_per_neuron     | 10      | 12-16        | Specificity vs sparsity |
  | context_size        | 4       | 6-8          | Information vs sparsity |
  | global_top_k        | 100     | 500-1000     | FALSE training coverage |
