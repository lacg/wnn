#!/usr/bin/env python3
"""
RAM Seq2Seq Demo

Demonstrates the transformer-like architecture with stacked RAM attention layers.
This is the closest to a traditional Transformer using RAM neurons.

Tasks to demonstrate:
1. Copy task (identity mapping - simplest)
2. Reverse task (harder - needs to learn position relationships)
3. Next character prediction (autoregressive)
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMSeq2Seq import RAMSeq2Seq
from wnn.ram.encoders_decoders import (
	TransformerDecoderFactory, OutputMode, PositionMode
)
from torch import tensor, uint8, cat

print("=" * 70)
print("RAM Seq2Seq Demo")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

print(f"Token encoding: {bits_per_token} bits per token")
print()


def encode_sequence(text: str) -> list:
	"""Encode text to list of bit tensors."""
	return [decoder.encode(c).squeeze() for c in text]


def decode_sequence(tokens: list) -> str:
	"""Decode list of bit tensors to text."""
	return ''.join(decoder.decode(t.unsqueeze(0)) for t in tokens)


# ============================================================
# Task 1: Identity/Copy Task
# ============================================================
print("=" * 70)
print("Task 1: Identity/Copy")
print("=" * 70)
print("""
This tests whether the model can preserve information through
multiple attention layers. Input should equal output.
""")

# Create a shallow model
model = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	rng=42,
)

print(f"Model: {model}")
print()

# Test without training (random attention)
test_seq = "ABC"
print(f"Before training:")
tokens = encode_sequence(test_seq)
outputs = model.forward(tokens)
decoded = decode_sequence(outputs)
print(f"  Input:  '{test_seq}'")
print(f"  Output: '{decoded}'")
print()


# ============================================================
# Task 2: Architecture Visualization
# ============================================================
print("=" * 70)
print("Architecture Visualization")
print("=" * 70)

# Create deeper model
deep_model = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=3,
	num_heads=4,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	rng=123,
)

print(f"\nDeep Model: {deep_model}")
print()

# Show intermediate representations
test_seq = "HI"
print(f"Processing '{test_seq}' through 3 layers:")
tokens = encode_sequence(test_seq)
outputs, intermediates = deep_model.forward_with_intermediates(tokens)

for i, layer_state in enumerate(intermediates):
	layer_name = "Input" if i == 0 else f"Layer {i}"
	decoded = decode_sequence(layer_state)
	print(f"  {layer_name}: '{decoded}'")

print(f"  Output: '{decode_sequence(outputs)}'")
print()


# ============================================================
# Task 3: Autoregressive Generation
# ============================================================
print("=" * 70)
print("Autoregressive Generation (Untrained)")
print("=" * 70)
print("""
The model generates tokens one by one, using its own output
as input for the next step. Without training, output is random.
""")

# Create model for generation
gen_model = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=2,
	num_heads=4,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	rng=456,
)

prompt = "AB"
print(f"Prompt: '{prompt}'")
prompt_tokens = encode_sequence(prompt)

print("Generating 3 tokens:")
full_sequence = gen_model.generate(
	prompt_tokens,
	max_new_tokens=3,
	decoder=decoder
)

print(f"\nFull sequence: '{decode_sequence(full_sequence)}'")
print()


# ============================================================
# Model Comparison
# ============================================================
print("=" * 70)
print("Model Configurations")
print("=" * 70)
print("""
Different configurations for different use cases:
""")

configs = [
	{"num_layers": 1, "num_heads": 1, "desc": "Minimal (fast, low capacity)"},
	{"num_layers": 2, "num_heads": 4, "desc": "Standard (balanced)"},
	{"num_layers": 4, "num_heads": 4, "desc": "Deep (higher capacity)"},
	{"num_layers": 2, "num_heads": 8, "desc": "Wide (more attention patterns)"},
]

for config in configs:
	model = RAMSeq2Seq(
		input_bits=bits_per_token,
		num_layers=config["num_layers"],
		num_heads=config["num_heads"],
		position_mode=PositionMode.RELATIVE,
		max_seq_len=16,
		rng=0,
	)

	# Count approximate number of RAM neurons
	# (this is simplified - actual count depends on n_bits_per_neuron)
	sim_neurons = config["num_layers"] * config["num_heads"] * 1  # similarity heads
	val_neurons = config["num_layers"] * config["num_heads"] * bits_per_token  # value heads
	agg_neurons = config["num_layers"] * config["num_heads"] * bits_per_token  # aggregators
	out_neurons = config["num_layers"] * bits_per_token  # output layers

	print(f"  {config['desc']}")
	print(f"    Layers: {config['num_layers']}, Heads: {config['num_heads']}")
	print()


# ============================================================
# Summary
# ============================================================
print("=" * 70)
print("Summary: RAMSeq2Seq Architecture")
print("=" * 70)
print("""
RAMSeq2Seq is a transformer-like architecture using RAM neurons:

COMPONENTS:
  1. Stacked Attention Layers
     - Each layer has multi-head RAM attention
     - Learns content-based and positional patterns
     - Causal masking for autoregressive use

  2. Residual Connections (XOR-based)
     - Helps information flow through deep networks
     - XOR instead of addition (binary compatible)

  3. Optional Projections
     - Input projection: expand/compress token dimension
     - Output projection: map to output vocabulary

TRAINING (not shown here):
  - Forward pass records contexts at each layer
  - Error at output propagates back through layers
  - EDRA updates RAM neurons at each layer
  - Similar to backprop but for discrete neurons

LIMITATIONS vs TRANSFORMERS:
  - Binary attention (no fine-grained weights)
  - XOR aggregation (not weighted sum)
  - No continuous gradients (EDRA instead)

ADVANTAGES:
  - Interpretable attention patterns
  - Memory efficient (binary)
  - No floating point errors
  - Parallelizable like transformers
""")
print("=" * 70)
