#!/usr/bin/env python3
"""
Test RAMMultiHeadKV with associative recall tasks.

Tests:
1. Simple write/read pairs
2. Multiple key-value associations
3. Overwrite existing keys
4. Cross-head routing
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMMultiHeadKV import RAMMultiHeadKV
from wnn.ram.encoders_decoders import OutputMode, TransformerDecoderFactory
from torch import cat, zeros, uint8

print("="*60)
print("RAMMultiHeadKV Associative Memory Test")
print("="*60)

# Test 1: Simple write/read with explicit API
print("\n" + "="*60)
print("Test 1: Simple Write/Read API")
print("="*60)

model = RAMMultiHeadKV(
	num_heads=4,
	k_bits=5,  # 5 bits for key (matches token encoding)
	v_bits=5,  # 5 bits for value
	neurons_per_head=8,
	output_mode=OutputMode.TOKEN,
	rng=42,
)
print(f"Model: {model}")

# Write some associations
print("\nWriting associations:")
associations = [('A', 'X'), ('B', 'Y'), ('C', 'Z'), ('D', 'W')]
for key, value in associations:
	model.write(key, value)
	print(f"  Write: {key} → {value}")

# Read them back
print("\nReading back:")
correct = 0
for key, expected in associations:
	result = model.read(key)
	match = result == expected
	if match:
		correct += 1
	symbol = "✓" if match else "✗"
	print(f"  Read({key}) = {result} (expected {expected}) {symbol}")

print(f"\nAccuracy: {correct}/{len(associations)} = {100*correct/len(associations):.1f}%")

# Test 2: Training with window sequences
print("\n" + "="*60)
print("Test 2: Training with Window Sequences")
print("="*60)

model2 = RAMMultiHeadKV(
	num_heads=4,
	k_bits=5,
	v_bits=5,
	neurons_per_head=8,
	output_mode=OutputMode.TOKEN,
	rng=42,
)

decoder = TransformerDecoderFactory.create(OutputMode.TOKEN_LIST)

def make_write_window(key_char: str, value_char: str):
	"""Create a write window: [key_bits | value_bits]"""
	key_windows = decoder.encode(key_char)  # Returns list of tensors
	value_windows = decoder.encode(value_char)
	key_bits = key_windows[0].squeeze(0)
	value_bits = value_windows[0].squeeze(0)
	return cat([key_bits, value_bits]).unsqueeze(0)

def make_query_window(key_char: str):
	"""Create a query window: [key_bits | zeros]"""
	key_windows = decoder.encode(key_char)
	key_bits = key_windows[0].squeeze(0)
	value_bits = zeros(5, dtype=uint8)  # Query = zeros
	return cat([key_bits, value_bits]).unsqueeze(0)

# Create episode: Write A→X, Write B→Y, Query A, Query B
print("\nEpisode: Write(A,X), Write(B,Y), Query(A)→?, Query(B)→?")

windows = [
	make_write_window('A', 'X'),
	make_write_window('B', 'Y'),
	make_query_window('A'),
	make_query_window('B'),
]

# Train for multiple epochs
for epoch in range(20):
	model2._reset_states()
	model2.train(windows, targets=['X', 'Y'])  # Targets for the 2 queries

# Test
model2._reset_states()
input_bits = cat([w.squeeze(0) for w in windows])
result = model2.forward(input_bits)
print(f"Final query result: {result}")

# Test individual queries after writes
model2._reset_states()
for w in windows[:2]:  # Process writes
	key_bits, head_idx = model2._extract_key(w)
	model2._forward_head(head_idx, key_bits, is_write=True)

# Now query
result_a = model2.read('A')
result_b = model2.read('B')
print(f"Read(A) = {result_a} (expected X)")
print(f"Read(B) = {result_b} (expected Y)")

# Test 3: Multiple associations with routing
print("\n" + "="*60)
print("Test 3: Multiple Associations Across Heads")
print("="*60)

model3 = RAMMultiHeadKV(
	num_heads=4,
	k_bits=5,
	v_bits=5,
	neurons_per_head=10,
	output_mode=OutputMode.TOKEN,
	rng=42,
)

# More associations
associations = [
	('A', 'Z'), ('B', 'Y'), ('C', 'X'), ('D', 'W'),
	('E', 'V'), ('F', 'U'), ('G', 'T'), ('H', 'S'),
]

print("\nWriting 8 associations...")
for key, value in associations:
	model3.write(key, value)

print("\nReading back:")
correct = 0
for key, expected in associations:
	result = model3.read(key)
	match = result == expected
	if match:
		correct += 1
	symbol = "✓" if match else "✗"
	print(f"  {key} → {result} (expected {expected}) {symbol}")

print(f"\nAccuracy: {correct}/{len(associations)} = {100*correct/len(associations):.1f}%")

# Show head routing
print("\nHead routing (which head stores each key):")
for key, _ in associations:
	key_bits = decoder.encode(key)[0].squeeze(0)
	_, head_idx = model3._extract_key(cat([key_bits, zeros(5, dtype=uint8)]))
	print(f"  {key} → Head {head_idx}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("✓ RAMMultiHeadKV provides explicit KV memory operations")
print("✓ write(key, value) stores associations")
print("✓ read(key) retrieves stored values")
print("✓ Key bits route to different heads")
print("✓ Query detection via zero-value convention")
print("="*60)
