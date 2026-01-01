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

## Future: Training Enhancements
- [ ] Curriculum Learning Integration - Start with short sequences, gradually increase
- [ ] Multi-Task Learning - Train on multiple tasks simultaneously
- [ ] Contrastive Learning - Learn representations that distinguish similar patterns

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

## Explored: Language Modeling (Improved with Pattern Abstraction)
- [x] N-gram Language Model - Memorizes context→next (39% generalization)
- [x] Backoff N-gram - Falls back to shorter contexts (39%)
- [x] Pattern Abstraction - Decomposes chars into features (**72%** generalization!)
- [x] Hierarchical N-gram - Ensemble voting (61%)

**Key Finding**: Pattern Abstraction applies decomposition to language!

Decompose characters into **features** (like bits for numbers):
- **Type**: vowel (V), consonant (C), space, punctuation (5 values)
- **Position**: 0-25 for a-z (26 values)

This reduces pattern space from O(26^n) to O(5^n × 26):
- Original 3-gram: 26³ = 17,576 patterns
- Pattern abstraction: 5³ × 26 = 3,250 patterns (5× reduction)

| Approach | Generalization | Key Idea |
|----------|---------------|----------|
| Basic N-gram | 39% | Exact context memorization |
| Backoff N-gram | 39% | Graceful degradation |
| **Pattern Abstraction** | **72%** | **Character feature decomposition** |
| Hierarchical | 61% | Ensemble voting |

Pattern Abstraction shows that **decomposition CAN improve language** - just need the right features!
