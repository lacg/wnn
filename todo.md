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

## Completed: Architectural Improvements
- [x] Learned Position Embeddings - LearnedPositionEncoder with RAMLayer
- [x] Cross-Attention - SoftRAMAttention supports key_bits and context parameter
- [x] Layer Normalization Equivalent - DiscreteNormalization with ENSEMBLE_VOTE/BIT_BALANCE
- [x] Sparse Attention Patterns - STRIDED, DILATED, LOCAL_GLOBAL in strategies/

## Future: Training Enhancements
- [ ] Curriculum Learning Integration - Start with short sequences, gradually increase
- [ ] Multi-Task Learning - Train on multiple tasks simultaneously
- [ ] Contrastive Learning - Learn representations that distinguish similar patterns

## Future: New Task Domains
- [ ] Arithmetic - Multi-digit addition, multiplication
- [ ] Sorting - Already have computed version; can we learn it?
- [ ] Language Modeling - Character-level text generation
