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

The pattern: **Decompose → Learn primitives → Compose/Recur → Generalize**

## Explored: Sequence-to-Sequence (Cross-Attention)
- [x] RAMEncoderDecoder - Full encoder-decoder with cross-attention
- [x] Copy task - 100% (position alignment)
- [x] Reverse task - 100% (position routing)
- [x] Increment task - 100% (alignment + FFN transformation)
- [x] Character mapping - 100% (identity alignment)
- [ ] Arithmetic evaluation - 0% (requires computation, not just routing)

**Key Finding**: Cross-attention excels at **alignment** (which source position to read) but cannot do **computation** (combining values mathematically). For arithmetic in seq2seq, would need to integrate decomposed primitives (LearnedFullAdder).

| Seq2Seq Task | Accuracy | Why |
|--------------|----------|-----|
| Copy | 100% | Decoder[i] attends to Encoder[i] |
| Reverse | 100% | Decoder[i] attends to Encoder[n-1-i] |
| Increment | 100% | Alignment + FFN transformation |
| Arithmetic | 0% | Requires computation, not just routing |

## Explored: Language Modeling (Limited Generalization Expected)
- [x] N-gram Language Model - Memorizes context→next mappings
- [x] Recurrent Language Model - State-based character prediction

**Key Finding**: Unlike arithmetic, language lacks universal primitives for decomposition:
- Train accuracy: 100% on seen patterns
- Test accuracy: ~57% on unseen combinations (21% gap)
- This is **expected** - language is fundamentally about statistical patterns, not composable operations

Language modeling shows where decomposition **doesn't** apply:
| Aspect | Arithmetic | Language |
|--------|------------|----------|
| Primitives | Universal (XOR, carry) | Context-dependent |
| Composition | Perfect (deterministic) | Probabilistic |
| Generalization | 100% from primitives | Requires similar training contexts |
