#!/usr/bin/env python3
"""
Test position encoders.

Tests:
1. BinaryPositionEncoder - absolute position encoding
2. RelativePositionEncoder - relative distance encoding
3. PositionEncoderFactory - factory pattern
4. Integration with token encoding
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.encoders_decoders import (
	PositionMode,
	PositionEncoderFactory,
	BinaryPositionEncoder,
	RelativePositionEncoder,
	TransformerDecoderFactory,
	OutputMode,
)
from torch import cat

print("="*60)
print("Position Encoder Tests")
print("="*60)

# Test 1: BinaryPositionEncoder
print("\n" + "="*60)
print("Test 1: BinaryPositionEncoder")
print("="*60)

encoder = BinaryPositionEncoder(n_position_bits=4, max_seq_len=16)
print(f"Encoder: {encoder}")

print("\nPosition encodings (4 bits, max 16 positions):")
for pos in [0, 1, 2, 5, 7, 15]:
	bits = encoder.encode(pos)
	bits_str = ''.join(str(int(b)) for b in bits)
	decoded = encoder.decode(bits)
	check = "OK" if decoded == pos else "FAIL"
	print(f"  {pos:2d} -> [{bits_str}] -> {decoded:2d} {check}")

# Test round-trip
print("\nRound-trip test (all positions 0-15):")
all_ok = True
for pos in range(16):
	bits = encoder.encode(pos)
	decoded = encoder.decode(bits)
	if decoded != pos:
		print(f"  FAIL: {pos} -> {decoded}")
		all_ok = False
print(f"  Result: {'All OK' if all_ok else 'FAILED'}")

# Test 2: RelativePositionEncoder
print("\n" + "="*60)
print("Test 2: RelativePositionEncoder")
print("="*60)

rel_encoder = RelativePositionEncoder(n_distance_bits=4, max_distance=7)
print(f"Encoder: {rel_encoder}")

print("\nRelative distance encodings (4 bits, max dist 7):")
print("  Format: [sign | magnitude]")
for dist in [0, 1, 3, -1, -3, -7]:
	bits = rel_encoder.encode(dist)
	bits_str = ''.join(str(int(b)) for b in bits)
	sign = "+" if bits[0] == 0 else "-"
	decoded = rel_encoder.decode(bits)
	check = "OK" if decoded == dist else "FAIL"
	print(f"  {dist:+3d} -> [{bits_str}] ({sign}) -> {decoded:+3d} {check}")

print("\nQuery-Key relative encoding:")
for q_pos, k_pos in [(0, 2), (5, 3), (3, 3), (2, 7)]:
	bits = rel_encoder.encode_relative(q_pos, k_pos)
	bits_str = ''.join(str(int(b)) for b in bits)
	dist = k_pos - q_pos
	print(f"  Query@{q_pos}, Key@{k_pos} -> dist={dist:+2d} -> [{bits_str}]")

# Test 3: PositionEncoderFactory
print("\n" + "="*60)
print("Test 3: PositionEncoderFactory")
print("="*60)

# NONE mode
none_enc = PositionEncoderFactory.create(PositionMode.NONE)
print(f"PositionMode.NONE -> {none_enc}")

# BINARY mode with auto bits
binary_enc = PositionEncoderFactory.create(PositionMode.BINARY, max_seq_len=32)
print(f"PositionMode.BINARY (max_len=32) -> {binary_enc}")

# RELATIVE mode with auto bits
rel_enc = PositionEncoderFactory.create(PositionMode.RELATIVE, max_distance=15)
print(f"PositionMode.RELATIVE (max_dist=15) -> {rel_enc}")

# Test 4: Integration with token encoding
print("\n" + "="*60)
print("Test 4: Token + Position Encoding Integration")
print("="*60)

# Create token decoder
token_decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
pos_encoder = PositionEncoderFactory.create(PositionMode.BINARY, max_seq_len=16)

print(f"Token decoder: {token_decoder.bits_per_token} bits per token")
print(f"Position encoder: {pos_encoder.n_bits} bits per position")
print(f"Combined: {token_decoder.bits_per_token + pos_encoder.n_bits} bits per token+position")

print("\nEncoding 'HELLO' with positions:")
text = "HELLO"
for i, char in enumerate(text):
	token_bits = token_decoder.encode(char).squeeze(0)
	pos_bits = pos_encoder.encode(i)
	combined = cat([pos_bits, token_bits])

	token_str = ''.join(str(int(b)) for b in token_bits)
	pos_str = ''.join(str(int(b)) for b in pos_bits)
	combined_str = ''.join(str(int(b)) for b in combined)

	print(f"  '{char}' @ pos {i}:")
	print(f"    token:    [{token_str}]")
	print(f"    position: [{pos_str}]")
	print(f"    combined: [{combined_str}]")

# Demonstrate position-awareness
print("\nPosition-awareness demo:")
print("  Same token 'A' at different positions produces different combined encodings:")
for pos in [0, 3, 7]:
	token_bits = token_decoder.encode('A').squeeze(0)
	pos_bits = pos_encoder.encode(pos)
	combined = cat([pos_bits, token_bits])
	combined_str = ''.join(str(int(b)) for b in combined)
	print(f"    'A' @ pos {pos}: [{combined_str}]")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("Position encoding enables:")
print("  - 'A at pos 0' != 'A at pos 5' (different bit patterns)")
print("  - Position-dependent pattern learning")
print("  - Transformer-style positional awareness")
print("="*60)
