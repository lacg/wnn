#!/usr/bin/env python3
"""
RAM Attention Demonstration

Shows the key differences between:
1. Traditional Transformer attention (soft, continuous)
2. RAM-based attention (hard, discrete)
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMAttention import RAMAttention
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode
from torch import tensor, uint8

print("="*70)
print("RAM-Based Attention Demo")
print("="*70)

# Comparison
print("""
TRANSFORMER ATTENTION vs RAM ATTENTION
======================================

Transformer (Soft):
  scores[i,j] = softmax(Q[i] · K[j] / √d)   # Float in [0,1]
  output[i] = Σ scores[i,j] * V[j]           # Weighted sum

RAM (Hard):
  attend[i,j] = RAM_similarity(Q[i], K[j])   # Binary: 0 or 1
  output[i] = aggregate(V[j] where attend[i,j]=1)  # Vote/select

Key insight: RAM attention is DISCRETE - no gradients, no weighted sums.
But it CAN learn content-based routing patterns.
""")

# Create encoder for tokens
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

print(f"Token encoding: {bits_per_token} bits per token")
print()

# Create RAM attention layer
attention = RAMAttention(
	input_bits=bits_per_token,
	num_heads=4,
	neurons_per_head=8,
	use_positions=True,
	max_seq_len=16,
	causal=True,  # Can only look at past
	rng=42,
)

print(f"Model: {attention}")
print()

# Encode a sample sequence
sequence = "HELLO"
print(f"Input sequence: '{sequence}'")
print()

tokens = []
for char in sequence:
	bits = decoder.encode(char).squeeze()
	tokens.append(bits)
	bits_str = ''.join(str(int(b)) for b in bits)
	print(f"  '{char}' -> [{bits_str}]")

print()

# Visualize attention patterns for each head
print("="*70)
print("Attention Patterns (before training)")
print("="*70)
print("""
Legend:
  # = attends (RAM neuron outputs 1)
  . = doesn't attend (RAM neuron outputs 0)
  - = masked (causal: can't attend to future)
""")

for h in range(attention.num_heads):
	print(attention.visualize_attention(tokens, head_idx=h))
	print()

# Apply attention
print("="*70)
print("Forward Pass")
print("="*70)

outputs = attention.forward(tokens)

print("Output after attention:")
for i, (char, out) in enumerate(zip(sequence, outputs)):
	out_str = ''.join(str(int(b)) for b in out)
	decoded = decoder.decode(out.unsqueeze(0))
	print(f"  Position {i} ('{char}'): [{out_str}] -> '{decoded}'")

print()
print("="*70)
print("Architecture Comparison")
print("="*70)
print("""
                    TRANSFORMER              RAM ATTENTION
                    -----------              -------------
Attention weights:  Continuous [0,1]         Binary {0,1}
Combination:        Weighted sum             Vote/XOR/Select
Gradients:          Backprop through softmax EDRA (discrete)
Parallelism:        Full parallel            Full parallel
Position encoding:  Sinusoidal/Learned       Binary bits

WHAT THIS ENABLES:
- Content-based routing (query selects relevant keys)
- Learnable attention patterns (via RAM training)
- Multi-head specialization (different heads, different patterns)
- Causal masking (for autoregressive generation)

WHAT'S DIFFERENT:
- No fine-grained weighting (can't say "attend 0.7 to this, 0.3 to that")
- Selection is all-or-nothing per key
- Aggregation must be discrete (XOR, vote, first match)

TRAINING:
- Can train similarity heads to learn "when to attend"
- Target: correct next-token prediction
- Method: EDRA backprop through discrete neurons
""")

print("="*70)
print("Next Steps Toward RAM Transformer")
print("="*70)
print("""
1. Stack multiple RAM attention layers (depth)
2. Add feedforward layers between attention (like transformer blocks)
3. Train attention patterns on sequence prediction tasks
4. Add learned query/key projections (RAM-based)

The key question: Can discrete attention learn useful patterns?

Early transformers used hard attention too (Xu et al., 2015 - image captioning).
It works, but soft attention usually performs better because gradients flow.

RAM attention trades gradient flow for:
- Interpretable patterns (binary = easy to visualize)
- Memory efficiency (no float weights)
- Potential sparsity benefits
""")
