#!/usr/bin/env python3
"""
Encoder-Decoder Architecture Tests

Tests the RAMCrossAttention and RAMEncoderDecoder:
1. Cross-attention mechanism
2. Full encoder-decoder forward pass
3. Simple translation task (reverse sequence)
4. Autoregressive generation
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMCrossAttention import RAMCrossAttention, CrossAttentionMode
from wnn.ram.RAMEncoderDecoder import RAMEncoderDecoder
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.RAMEmbedding import PositionEncoding
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import tensor, uint8, zeros, ones

print("=" * 70)
print("Encoder-Decoder Architecture Tests")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str):
	return decoder.encode(c).squeeze()

def decode_bits(bits):
	return decoder.decode(bits.unsqueeze(0))

def encode_sequence(text: str) -> list:
	return [encode_char(c) for c in text]

def decode_sequence(tokens: list) -> str:
	return ''.join(decode_bits(t) for t in tokens)


# =============================================================
# Test 1: Cross-Attention Basics
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Cross-Attention Mechanism")
print("=" * 60)

cross_attn = RAMCrossAttention(
	decoder_bits=bits_per_token,
	encoder_bits=bits_per_token,
	num_heads=2,
	position_mode=CrossAttentionMode.ENCODER_ONLY,
	max_encoder_len=16,
	max_decoder_len=16,
	rng=42,
)

print(f"\nCross-Attention: {cross_attn}")

# Test cross-attention: decoder tokens attend to encoder tokens
encoder_tokens = encode_sequence("HELLO")
decoder_tokens = encode_sequence("HI")

print(f"\nEncoder sequence: 'HELLO' ({len(encoder_tokens)} tokens)")
print(f"Decoder sequence: 'HI' ({len(decoder_tokens)} tokens)")

# Forward pass
outputs = cross_attn(decoder_tokens, encoder_tokens)
print(f"\nCross-attention output shapes:")
for i, out in enumerate(outputs):
	print(f"  Decoder pos {i}: {out.shape}")

# Visualize attention pattern
print("\n" + cross_attn.visualize_cross_attention(
	decoder_tokens, encoder_tokens,
	head_idx=0,
	decoder_labels=['H', 'I'],
	encoder_labels=['H', 'E', 'L', 'L', 'O']
))


# =============================================================
# Test 2: Encoder-Decoder Forward Pass
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Encoder-Decoder Forward Pass")
print("=" * 60)

enc_dec = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_encoder_len=16,
	max_decoder_len=16,
	use_residual=True,
	use_ffn=False,
	generalization=MapperStrategy.DIRECT,
	rng=42,
)

print(f"\nModel: {enc_dec}")

# Test forward pass
source = encode_sequence("ABC")
target = encode_sequence("XYZ")

print(f"\nSource: 'ABC'")
print(f"Target (input): 'XYZ'")

outputs = enc_dec.forward(source, target)
output_str = decode_sequence(outputs)
print(f"Output: '{output_str}'")


# =============================================================
# Test 3: Copy Task (Identity)
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Copy Task - Learn to Copy Input")
print("=" * 60)

copy_model = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_encoder_len=16,
	max_decoder_len=16,
	use_residual=True,
	use_ffn=False,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

print("\nTask: Given source 'ABC', output 'ABC' (copy)")

# Create dataset: (source, target_input, target_output)
# target_input is shifted right (e.g., start with special token, here we use source)
# target_output is what we want to predict
copy_dataset = [
	(encode_sequence("ABC"), encode_sequence("ABC"), encode_sequence("ABC")),
	(encode_sequence("DEF"), encode_sequence("DEF"), encode_sequence("DEF")),
	(encode_sequence("GHI"), encode_sequence("GHI"), encode_sequence("GHI")),
]

print("\nTraining on copy task...")
history = copy_model.train(copy_dataset, epochs=5)

# Test
print("\nTesting:")
for source_str in ["ABC", "JKL", "MNO"]:
	source = encode_sequence(source_str)
	outputs = copy_model.forward(source, source)  # Teacher forcing
	output_str = decode_sequence(outputs)
	status = "OK" if output_str == source_str else "X"
	print(f"  [{status}] '{source_str}' -> '{output_str}'")


# =============================================================
# Test 4: Reverse Task
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Reverse Task - Learn to Reverse Input")
print("=" * 60)

reverse_model = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_encoder_len=16,
	max_decoder_len=16,
	use_residual=True,
	use_ffn=True,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

print("\nTask: Given source 'ABC', output 'CBA' (reverse)")

# Create reverse dataset
reverse_dataset = [
	(encode_sequence("ABC"), encode_sequence("CBA"), encode_sequence("CBA")),
	(encode_sequence("DEF"), encode_sequence("FED"), encode_sequence("FED")),
	(encode_sequence("GHI"), encode_sequence("IHG"), encode_sequence("IHG")),
]

print("\nTraining on reverse task...")
history = reverse_model.train(reverse_dataset, epochs=10)

# Test on training data
print("\nTesting on training data:")
for source_str, _, _ in reverse_dataset:
	source_text = decode_sequence(source_str)
	target_text = source_text[::-1]
	target = encode_sequence(target_text)
	outputs = reverse_model.forward(source_str, target)
	output_str = decode_sequence(outputs)
	status = "OK" if output_str == target_text else "X"
	print(f"  [{status}] '{source_text}' -> '{output_str}' (expected '{target_text}')")

# Test generalization
print("\nTesting generalization on unseen sequences:")
for source_str in ["JKL", "MNO"]:
	source = encode_sequence(source_str)
	target_text = source_str[::-1]
	target = encode_sequence(target_text)
	outputs = reverse_model.forward(source, target)
	output_str = decode_sequence(outputs)
	status = "OK" if output_str == target_text else "X"
	print(f"  [{status}] '{source_str}' -> '{output_str}' (expected '{target_text}')")


# =============================================================
# Test 5: Cross-Attention Position Modes
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Cross-Attention Position Modes")
print("=" * 60)

for mode in [CrossAttentionMode.NONE, CrossAttentionMode.ENCODER_ONLY, CrossAttentionMode.BOTH]:
	print(f"\nMode: {mode.name}")

	cross_attn_mode = RAMCrossAttention(
		decoder_bits=bits_per_token,
		encoder_bits=bits_per_token,
		num_heads=2,
		position_mode=mode,
		max_encoder_len=8,
		max_decoder_len=8,
		rng=42,
	)

	enc = encode_sequence("ABC")
	dec = encode_sequence("XY")

	outputs = cross_attn_mode(dec, enc)
	print(f"  Input: decoder='XY', encoder='ABC'")
	print(f"  Output dims: {[o.shape for o in outputs]}")


# =============================================================
# Test 6: Full Pipeline with Embeddings
# =============================================================
print("\n" + "=" * 60)
print("Test 6: Full Pipeline with Embeddings")
print("=" * 60)

full_model = RAMEncoderDecoder(
	input_bits=bits_per_token,
	hidden_bits=bits_per_token * 2,  # Expand to 10 bits internally
	output_bits=bits_per_token,       # Back to 5 bits output
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_encoder_len=16,
	max_decoder_len=16,
	use_residual=True,
	use_ffn=True,
	use_embedding=True,
	embedding_position=PositionEncoding.BINARY,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

print(f"\nModel: {full_model}")
print(f"  Has encoder embedding: {full_model.encoder_embedding is not None}")
print(f"  Has decoder embedding: {full_model.decoder_embedding is not None}")

# Quick test
source = encode_sequence("TEST")
target = encode_sequence("TEST")
outputs = full_model.forward(source, target)
output_str = decode_sequence(outputs)
print(f"\nTest: 'TEST' -> '{output_str}'")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Encoder-Decoder Architecture")
print("=" * 60)

print("""
RAMEncoderDecoder Architecture:

  Source ─────▶ ENCODER ─────▶ Encoder Output (Memory)
                  │                     │
                  │ Self-Attention      │
                  │ (Bidirectional)     │
                  │                     │
  Target ─────▶ DECODER ◀───────────────┘
                  │           │
    Self-Attention│           │Cross-Attention
       (Causal)   │           │(to Encoder)
                  │           │
                  ▼           ▼
              Output Sequence

Key Components:
1. RAMCrossAttention
   - Query: from decoder (what am I looking for?)
   - Key/Value: from encoder (source context)
   - Non-causal: can attend to any encoder position

2. Position Modes:
   - NONE: Content-only attention
   - ENCODER_ONLY: Include encoder positions
   - BOTH: Include both decoder and encoder positions

3. Training:
   - Teacher forcing: provide correct previous tokens
   - EDRA backpropagation through all layers
   - BIT_LEVEL generalization for unseen sequences

Use Cases:
- Translation: source language → target language
- Summarization: document → summary
- Question Answering: question → answer
""")

print("=" * 60)
print("All encoder-decoder tests completed!")
print("=" * 60)
