# Next steps

# To remember

Name										Meaning												Shape
window_bits							raw input window							[1, input_bits]
input_layer_output			input layer output						[1, N_in]
state_bits							recurrent state								[1, N_state]
state_layer_input				[input_out(t), state(t-1)]		[1, N_in + N_state]
state_layer_output			state(t)											[1, N_state]
output_layer_input			[input_out(t), state(t)]			[1, N_in + N_state]
output_layer_output			final output									[1, N_out]

---

# Transformer Improvement Roadmap

## Completed
- [x] Scheduled Sampling for autoregressive training
- [x] Soft Attention via Voting (already existed)
- [x] Hard Example Mining (already existed)
- [x] Parity generalization → 100% with RecurrentParityMapper (PARITY strategy)
- [x] Shift-left generalization → 100% with SHIFTED context mode

## Generalization Results (all tasks at 100%)
| Task       | Best Strategy | Test Accuracy | Solution |
|------------|---------------|---------------|----------|
| parity     | PARITY        | 100%          | RecurrentParityMapper (1-bit XOR state) |
| shift_left | SHIFTED       | 100%          | SHIFTED context mode (offset routing) |
| complement | COMPOSITIONAL | 100%          | Group-based decomposition |
| copy       | BIT_LEVEL     | 100%          | Per-bit context learning |
| successor  | BIT_LEVEL     | 100%          | Per-bit context learning |
| sorting    | BIT_LEVEL     | 100%          | BitLevelComparator (70 patterns for 6-bit) |
| addition   | RECURRENT     | 100%          | LearnedFullAdder (8 binary / 200 decimal patterns) |
| multiply   | RECURRENT     | 100%          | Shift-and-add reuses addition (8 binary / 300 decimal) |
| division   | RECURRENT     | 100%          | Shift-and-subtract reuses subtraction (8 binary patterns) |

## Completed: Architectural Improvements
- [x] Learned Position Embeddings - LearnedPositionEncoder with RAMLayer
- [x] Cross-Attention - SoftRAMAttention supports key_bits and context parameter
- [x] Layer Normalization Equivalent - DiscreteNormalization with ENSEMBLE_VOTE/BIT_BALANCE
- [x] Sparse Attention Patterns - STRIDED, DILATED, LOCAL_GLOBAL in strategies/

## Completed: Training Enhancements
- [x] Curriculum Learning Integration - CurriculumTrainer with configurable difficulty metrics
- [x] Multi-Task Learning - MultiTaskTrainer with Task definitions and MixingStrategy
- [x] Contrastive Learning - ContrastiveTrainer with triplet training and hard negative mining

## Completed: New Task Domains
- [x] Sorting - BitLevelComparator decomposes comparison into bit-level ops (100% generalization)
- [x] Arithmetic (Addition) - LearnedFullAdder with carry propagation (100% generalization)
- [x] Arithmetic (Multiplication) - Shift-and-add reuses addition; binary needs 0 new patterns!
- [x] Arithmetic (Division) - Shift-and-subtract reuses subtraction; 8 patterns for 100%!

## Key Insight: The Decomposition Pattern
RAM networks generalize when complex operations are decomposed into small learnable primitives:

| Operation | Primitives | Composition | Patterns |
|-----------|-----------|-------------|----------|
| Parity | XOR (4) | Recurrent state | 4 |
| Comparison | bit-level <, == (8) | Cascade + prefix AND | 70 (6-bit) |
| Sorting | BitLevelComparator | Rank counting | 70 (6-bit) |
| Addition | Full adder (8/200) | Carry propagation | 8 (binary) / 200 (decimal) |
| Multiply | Full adder + digit-mult | Shift-and-add | 8 (binary) / 300 (decimal) |
| Division | Full subtractor (8) | Shift-and-subtract | 8 (binary) |
| Language | Type+Position features | Pattern abstraction | ~3,250 (vs 17,576) |

The pattern: **Decompose → Learn primitives → Compose/Recur → Generalize**

Note: Language decomposition improves from 39% → 72%, not 100% like arithmetic.
This is because language patterns are probabilistic, not deterministic.

## Explored: Sequence-to-Sequence (Cross-Attention)
- [x] RAMEncoderDecoder - Full encoder-decoder with cross-attention
- [x] Copy task - 100% (position alignment)
- [x] Reverse task - 100% (position routing)
- [x] Increment task - 100% (alignment + FFN transformation)
- [x] Character mapping - 100% (identity alignment)
- [x] Arithmetic evaluation - 100% with HYBRID approach!

**Key Finding**: Cross-attention excels at **alignment** (which source position to read) but cannot do **computation** (combining values mathematically).

**Solution: Hybrid Seq2Seq** = Cross-attention (alignment) + Decomposed Primitives (computation)
- Position-based operand extraction (alignment)
- LearnedFullAdder/Subtractor for arithmetic (computation)
- Result: 100% on test cases, 75%+ generalization

| Seq2Seq Task | Accuracy | Why |
|--------------|----------|-----|
| Copy | 100% | Decoder[i] attends to Encoder[i] |
| Reverse | 100% | Decoder[i] attends to Encoder[n-1-i] |
| Increment | 100% | Alignment + FFN transformation |
| Arithmetic | 0% | Requires computation, not just routing |

## Explored: Language Modeling (Improved with Pattern Abstraction + Frequency)
- [x] N-gram Language Model - Memorizes context→next (39% generalization)
- [x] Backoff N-gram - Falls back to shorter contexts (42%)
- [x] Pattern Abstraction - Decomposes chars into features (**74%** generalization)
- [x] Hierarchical N-gram - Ensemble voting (63%)
- [x] Combined (Exact+Pattern) - Exact match with pattern fallback (68%)
- [x] Multi-Scale Pattern - Multi-gram voting (47%)
- [x] **Frequency Aware** - Track all continuations, pick most common (**79%**!)

**Key Finding**: Language has inherent ambiguity that limits accuracy!

Unlike arithmetic (100% deterministic), language is stochastic:
- Same context "he " can lead to 'c' (cat), 'm' (mat), 'd' (dog), etc.
- Ambiguous contexts account for ~34% of predictions
- Frequency-based selection picks the most likely continuation

| Approach | Generalization | Key Idea |
|----------|---------------|----------|
| Basic N-gram | 39% | Exact context memorization |
| Backoff N-gram | 42% | Graceful degradation |
| Pattern Abstraction | 74% | Character feature decomposition |
| Hierarchical | 63% | Ensemble voting |
| Combined | 68% | Exact match + pattern fallback |
| Multi-Scale | 47% | Multi-gram voting |
| **Frequency Aware** | **79%** | **Track all continuations, pick most common** |
| **Word-level + Prob** | **92.9%** | **Confidence filtering on word n-grams** |

## Completed: Word-Level + Probabilistic Improvements
- [x] Word-level language model (reduces ambiguity at word boundaries)
- [x] Probabilistic outputs (return distribution, not just top-1)
- [x] Confidence filtering (filter by entropy for higher precision)

**Results:**
| Metric | Char-Level | Word-Level | Word + Confidence |
|--------|------------|------------|-------------------|
| Top-1 Accuracy | 79% | 64% | - |
| Top-3 Accuracy | 88% | 84% | - |
| Confident Accuracy | - | - | **92.9%** |
| Perplexity | - | 2.50 | - |

**Key Finding**: Probabilistic output with confidence filtering achieves **92.9%**!
- When entropy < 1 bit, the model is confident → higher accuracy
- Word-level top-3 captures "reasonable" predictions (100% on seen patterns)
- Trade-off: Fewer predictions but much higher precision
