#!/usr/bin/env python3
"""
Test position-aware KV memory.

Tests:
1. Basic position-aware write/read
2. Same key at different positions stores different values
3. Position-based retrieval
4. Comparison: with vs without position encoding
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMMultiHeadKV import RAMMultiHeadKV
from wnn.ram.encoders_decoders import OutputMode, PositionMode

print("="*60)
print("Position-Aware KV Memory Tests")
print("="*60)

# Test 1: Basic position-aware KV operations
print("\n" + "="*60)
print("Test 1: Basic Position-Aware Write/Read")
print("="*60)

model = RAMMultiHeadKV(
	num_heads=4,
	k_bits=5,
	v_bits=5,
	neurons_per_head=10,
	output_mode=OutputMode.TOKEN,
	position_mode=PositionMode.BINARY,
	max_seq_len=16,
	rng=42,
)
print(f"Model: {model}")
print(f"Position bits: {model.n_position_bits}")
print(f"Full key bits: {model.full_key_bits} (pos + content)")

# Write with explicit positions
print("\nWriting with explicit positions:")
model.write('A', 'X', position=0)
print("  Write: 'A' @ pos 0 -> 'X'")
model.write('B', 'Y', position=1)
print("  Write: 'B' @ pos 1 -> 'Y'")
model.write('C', 'Z', position=2)
print("  Write: 'C' @ pos 2 -> 'Z'")

# Read with same positions
print("\nReading with same positions:")
results = []
for key, pos, expected in [('A', 0, 'X'), ('B', 1, 'Y'), ('C', 2, 'Z')]:
	result = model.read(key, position=pos)
	match = result == expected
	results.append(match)
	symbol = "OK" if match else "FAIL"
	print(f"  Read('{key}', pos={pos}) = '{result}' (expected '{expected}') {symbol}")

accuracy = sum(results) / len(results)
print(f"\nAccuracy: {sum(results)}/{len(results)} = {accuracy:.0%}")

# Test 2: Same key, different positions -> different values
print("\n" + "="*60)
print("Test 2: Same Key at Different Positions")
print("="*60)

model2 = RAMMultiHeadKV(
	num_heads=4,
	k_bits=5,
	v_bits=5,
	neurons_per_head=12,
	output_mode=OutputMode.TOKEN,
	position_mode=PositionMode.BINARY,
	max_seq_len=16,
	rng=42,
)

# Same key 'A' with different values at different positions
print("\nWriting same key 'A' at different positions with different values:")
model2.write('A', 'P', position=0)
print("  Write: 'A' @ pos 0 -> 'P'")
model2.write('A', 'Q', position=1)
print("  Write: 'A' @ pos 1 -> 'Q'")
model2.write('A', 'R', position=2)
print("  Write: 'A' @ pos 2 -> 'R'")
model2.write('A', 'S', position=3)
print("  Write: 'A' @ pos 3 -> 'S'")

print("\nReading 'A' at different positions (should get different values):")
results = []
for pos, expected in [(0, 'P'), (1, 'Q'), (2, 'R'), (3, 'S')]:
	result = model2.read('A', position=pos)
	match = result == expected
	results.append(match)
	symbol = "OK" if match else "FAIL"
	print(f"  Read('A', pos={pos}) = '{result}' (expected '{expected}') {symbol}")

accuracy = sum(results) / len(results)
print(f"\nAccuracy: {sum(results)}/{len(results)} = {accuracy:.0%}")

if accuracy == 1.0:
	print("Position encoding works! Same key stores different values at different positions.")

# Test 3: Comparison - Without position encoding
print("\n" + "="*60)
print("Test 3: Comparison - Without Position Encoding")
print("="*60)

model_no_pos = RAMMultiHeadKV(
	num_heads=4,
	k_bits=5,
	v_bits=5,
	neurons_per_head=12,
	output_mode=OutputMode.TOKEN,
	position_mode=PositionMode.NONE,  # No position encoding
	rng=42,
)
print(f"Model (no pos): {model_no_pos}")

# Try same operation - same key with different values
print("\nWriting same key 'A' without position encoding:")
model_no_pos.write('A', 'P')  # No position
print("  Write: 'A' -> 'P'")
model_no_pos.write('A', 'Q')  # Overwrites!
print("  Write: 'A' -> 'Q' (overwrites previous)")
model_no_pos.write('A', 'R')  # Overwrites again!
print("  Write: 'A' -> 'R' (overwrites again)")

result = model_no_pos.read('A')
print(f"\nRead('A') = '{result}'")
print("Without positions, later writes overwrite earlier ones.")

# Test 4: Auto-incrementing positions
print("\n" + "="*60)
print("Test 4: Auto-Incrementing Positions")
print("="*60)

model3 = RAMMultiHeadKV(
	num_heads=4,
	k_bits=5,
	v_bits=5,
	neurons_per_head=10,
	output_mode=OutputMode.TOKEN,
	position_mode=PositionMode.BINARY,
	max_seq_len=16,
	rng=42,
)

print("\nWriting without explicit positions (auto-increment):")
model3.reset_position()
model3.write('A', 'W')  # pos 0
print("  Write: 'A' -> 'W' (auto pos 0)")
model3.write('B', 'X')  # pos 1
print("  Write: 'B' -> 'X' (auto pos 1)")
model3.write('C', 'Y')  # pos 2
print("  Write: 'C' -> 'Y' (auto pos 2)")
model3.write('D', 'Z')  # pos 3
print("  Write: 'D' -> 'Z' (auto pos 3)")

print("\nReading with explicit positions:")
for key, pos, expected in [('A', 0, 'W'), ('B', 1, 'X'), ('C', 2, 'Y'), ('D', 3, 'Z')]:
	result = model3.read(key, position=pos)
	symbol = "OK" if result == expected else "FAIL"
	print(f"  Read('{key}', pos={pos}) = '{result}' (expected '{expected}') {symbol}")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print("Position encoding enables:")
print("  - Same key can store different values at different positions")
print("  - 'A' @ pos 0 != 'A' @ pos 3 (different full keys)")
print("  - Transformer-style positional awareness")
print("  - Sequence order matters!")
print("="*60)
