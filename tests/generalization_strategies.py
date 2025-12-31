#!/usr/bin/env python3
"""
Generalization Strategies for RAM Networks

The problem: RAM neurons are lookup tables that only know
mappings they've explicitly seen.

Solution approaches:
1. Bit-level patterns - learn at bit level, not character level
2. Compositional decomposition - split into smaller problems
3. Hash overlap - similar inputs share addresses

This demo shows how bit-level learning enables generalization.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode
from torch import tensor, uint8, zeros

print("=" * 70)
print("Generalization Strategies for RAM Networks")
print("=" * 70)

decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str) -> tensor:
	return decoder.encode(c).squeeze()

def decode_bits(bits: tensor) -> str:
	return decoder.decode(bits.unsqueeze(0))


# ============================================================
# The Problem: Character-Level Learning Doesn't Generalize
# ============================================================
print("\n" + "=" * 70)
print("Problem: Character-Level Learning")
print("=" * 70)

# Train on subset of alphabet
train_chars = "ACEGIKMOQSUWY"  # Odd-indexed letters
test_chars = "BDFHJLNPRTVXZ"   # Even-indexed letters (unseen)

char_mapper = RAMLayer(
	total_input_bits=bits_per_token,
	num_neurons=bits_per_token,
	n_bits_per_neuron=bits_per_token,
	rng=42,
)

print(f"Training on: {train_chars}")
print(f"Testing on:  {test_chars} (unseen)")
print()

# Train: char -> next_char
for c in train_chars:
	if c < 'Z':
		inp = encode_char(c)
		out = encode_char(chr(ord(c) + 1))
		char_mapper.commit(inp.unsqueeze(0), out.unsqueeze(0))

# Test on training chars
print("Results on TRAINING chars:")
correct = 0
for c in train_chars[:5]:
	if c < 'Z':
		inp = encode_char(c)
		pred = char_mapper(inp.unsqueeze(0)).squeeze()
		pred_char = decode_bits(pred)
		expected = chr(ord(c) + 1)
		status = "ok" if pred_char == expected else "WRONG"
		if pred_char == expected:
			correct += 1
		print(f"  {c} -> {pred_char} (expected {expected}) [{status}]")

print(f"\nResults on UNSEEN chars:")
correct_unseen = 0
for c in test_chars[:5]:
	if c < 'Z':
		inp = encode_char(c)
		pred = char_mapper(inp.unsqueeze(0)).squeeze()
		pred_char = decode_bits(pred)
		expected = chr(ord(c) + 1)
		status = "ok" if pred_char == expected else "WRONG"
		if pred_char == expected:
			correct_unseen += 1
		print(f"  {c} -> {pred_char} (expected {expected}) [{status}]")

print(f"\nUnseen accuracy: {correct_unseen}/5 = {100*correct_unseen/5:.0f}%")
print("(Low because RAM doesn't know these mappings)")


# ============================================================
# Solution 1: Bit-Level Patterns
# ============================================================
print("\n" + "=" * 70)
print("Solution 1: Bit-Level Patterns")
print("=" * 70)
print("""
Key insight: "next character" = binary increment (+1)

Binary increment rules (per bit position):
  - Bit flips if all lower bits are 1
  - Otherwise bit stays the same

Example:
  C = 00010 -> D = 00011 (flip bit 0)
  D = 00011 -> E = 00100 (flip bits 0,1,2 due to carry)

If we train on enough examples to cover carry patterns,
the RAM learns the RULE, not just individual mappings.
""")

# Create per-bit increment learners
# For bit i, learn: should this bit flip given lower bits?
bit_flippers = []
for bit_pos in range(bits_per_token):
	# Input: all bits at positions <= bit_pos
	# Output: 1 if this bit should flip, 0 otherwise
	flipper = RAMLayer(
		total_input_bits=bit_pos + 1,  # Only need lower bits + self
		num_neurons=1,
		n_bits_per_neuron=bit_pos + 1,
		rng=100 + bit_pos,
	)
	bit_flippers.append(flipper)

# Train on a SPARSE subset that covers carry patterns
# Key: we need examples that exercise each carry pattern
training_examples = [
	# Pattern: no carry (LSB = 0)
	'A', 'C', 'E', 'G',  # xxxxx0 -> flip bit 0 only
	# Pattern: 1-bit carry (ends in 01)
	'B', 'F', 'J', 'N',  # xxxx01 -> flip bits 0,1
	# Pattern: 2-bit carry (ends in 011)
	'D', 'L', 'T',       # xxx011 -> flip bits 0,1,2
	# Pattern: 3-bit carry (ends in 0111)
	'H', 'X',            # xx0111 -> flip bits 0,1,2,3
	# Pattern: 4-bit carry (ends in 01111)
	'P',                 # x01111 -> flip bits 0,1,2,3,4
]

print(f"Training on {len(training_examples)} examples (sparse but covers patterns)")
print(f"Examples: {training_examples}")
print()

# Train bit flippers
for c in training_examples:
	if c >= 'Z':
		continue

	inp_bits = encode_char(c)
	next_char = chr(ord(c) + 1)
	out_bits = encode_char(next_char)

	for bit_pos in range(bits_per_token):
		# Input: bits 0..bit_pos of input
		flipper_input = inp_bits[bits_per_token - bit_pos - 1:].clone()
		# Reverse to get LSB-first order
		flipper_input = flipper_input.flip(0)

		# Target: did this bit flip?
		did_flip = (inp_bits[bits_per_token - bit_pos - 1] !=
					out_bits[bits_per_token - bit_pos - 1])
		target = tensor([[1 if did_flip else 0]], dtype=uint8)

		bit_flippers[bit_pos].commit(flipper_input.unsqueeze(0), target)

def increment_char_bitwise(c: str) -> str:
	"""Increment a character using learned bit-flip rules."""
	inp_bits = encode_char(c).clone()
	out_bits = inp_bits.clone()

	for bit_pos in range(bits_per_token):
		# Get relevant input bits
		flipper_input = inp_bits[bits_per_token - bit_pos - 1:].clone()
		flipper_input = flipper_input.flip(0)

		# Query: should this bit flip?
		should_flip = bit_flippers[bit_pos](flipper_input.unsqueeze(0)).item()

		if should_flip:
			# Flip the bit
			idx = bits_per_token - bit_pos - 1
			out_bits[idx] = 1 - out_bits[idx]

	return decode_bits(out_bits)

# Test on ALL characters (including unseen)
print("Testing on ALL characters (including many unseen):")
all_correct = 0
all_chars = "ABCDEFGHIJKLMNOPQRSTUVWXY"  # All except Z

for c in all_chars:
	pred = increment_char_bitwise(c)
	expected = chr(ord(c) + 1)
	if pred == expected:
		all_correct += 1

print(f"  Accuracy: {all_correct}/{len(all_chars)} = {100*all_correct/len(all_chars):.1f}%")
print()

# Show specific examples
print("Sample predictions:")
for c in "BDFHJLNPRTVX":  # These were UNSEEN
	pred = increment_char_bitwise(c)
	expected = chr(ord(c) + 1)
	seen = "seen" if c in training_examples else "UNSEEN"
	status = "ok" if pred == expected else "WRONG"
	print(f"  {c} -> {pred} (expected {expected}) [{seen}] [{status}]")


# ============================================================
# Solution 2: Compositional Decomposition
# ============================================================
print("\n" + "=" * 70)
print("Solution 2: Compositional Decomposition")
print("=" * 70)
print("""
Split the problem into smaller pieces:
  - High bits (coarse position in alphabet)
  - Low bits (fine position within group)

Train separately on each component.
Fewer combinations = better coverage.
""")

# Split 5 bits into 2 high + 3 low
high_bits = 2
low_bits = 3

# For increment: low bits cycle 0->1->2->...->7->0 with carry
# High bits only change when low bits wrap

low_mapper = RAMLayer(
	total_input_bits=low_bits,
	num_neurons=low_bits,
	n_bits_per_neuron=low_bits,
	rng=200,
)

carry_detector = RAMLayer(
	total_input_bits=low_bits,
	num_neurons=1,  # Output: is there a carry?
	n_bits_per_neuron=low_bits,
	rng=201,
)

high_mapper = RAMLayer(
	total_input_bits=high_bits + 1,  # high bits + carry
	num_neurons=high_bits,
	n_bits_per_neuron=high_bits + 1,
	rng=202,
)

# Train low-bits mapper (only 8 patterns: 000->001, 001->010, ..., 111->000)
print("Training low-bits mapper (only 8 patterns needed):")
for i in range(8):
	next_i = (i + 1) % 8
	inp = zeros(low_bits, dtype=uint8)
	out = zeros(low_bits, dtype=uint8)
	for b in range(low_bits):
		inp[low_bits - 1 - b] = (i >> b) & 1
		out[low_bits - 1 - b] = (next_i >> b) & 1
	low_mapper.commit(inp.unsqueeze(0), out.unsqueeze(0))

	# Carry if i == 7
	carry = tensor([[1 if i == 7 else 0]], dtype=uint8)
	carry_detector.commit(inp.unsqueeze(0), carry)

# Train high-bits mapper (only 4*2=8 patterns: 4 high values * 2 carry states)
print("Training high-bits mapper (only 8 patterns needed):")
for h in range(4):
	for carry in [0, 1]:
		next_h = (h + carry) % 4
		inp = zeros(high_bits + 1, dtype=uint8)
		out = zeros(high_bits, dtype=uint8)
		inp[0] = (h >> 1) & 1  # high bit of h
		inp[1] = h & 1         # low bit of h
		inp[2] = carry
		out[0] = (next_h >> 1) & 1
		out[1] = next_h & 1
		high_mapper.commit(inp.unsqueeze(0), out.unsqueeze(0))

def increment_compositional(c: str) -> str:
	"""Increment using compositional decomposition."""
	bits = encode_char(c)

	# Split into high and low
	high_in = bits[:high_bits].clone()
	low_in = bits[high_bits:].clone()

	# Process low bits
	low_out = low_mapper(low_in.unsqueeze(0)).squeeze()
	carry = carry_detector(low_in.unsqueeze(0)).squeeze()

	# Process high bits with carry
	high_input = zeros(high_bits + 1, dtype=uint8)
	high_input[:high_bits] = high_in
	high_input[high_bits] = carry
	high_out = high_mapper(high_input.unsqueeze(0)).squeeze()

	# Combine
	from torch import cat
	result = cat([high_out, low_out])
	return decode_bits(result)

print("\nTesting compositional approach on ALL chars:")
comp_correct = 0
for c in all_chars:
	pred = increment_compositional(c)
	expected = chr(ord(c) + 1)
	if pred == expected:
		comp_correct += 1

print(f"  Accuracy: {comp_correct}/{len(all_chars)} = {100*comp_correct/len(all_chars):.1f}%")

# Show some examples
print("\nSample predictions (ALL unseen as whole characters):")
for c in "BDFHJLNPRTVX":
	pred = increment_compositional(c)
	expected = chr(ord(c) + 1)
	status = "ok" if pred == expected else "WRONG"
	print(f"  {c} -> {pred} [{status}]")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("Summary: Generalization Strategies")
print("=" * 70)
print("""
RAM Generalization Problem:
  - RAMs are lookup tables: only know what they've seen
  - Training on A,C,E doesn't teach B,D,F

Solution 1: BIT-LEVEL PATTERNS
  - Learn at bit level, not symbol level
  - "Next char" = binary increment, follows bit-flip rules
  - Train on patterns that cover carry cases
  - Generalizes because unseen chars use same bit patterns

Solution 2: COMPOSITIONAL DECOMPOSITION
  - Split input into components (high/low bits)
  - Train small mappers for each component
  - 5-bit char = 2^5=32 patterns (need all)
  - Split 2+3 bits = 2^2 + 2^3 = 12 patterns (much fewer!)

Solution 3: HASH OVERLAP (not shown)
  - Use hashing so similar inputs share addresses
  - "B" might hash similarly to "A" and "C"
  - Implicit sharing through collisions

Key Insight:
  Generalization in RAMs comes from STRUCTURE:
  - Learning rules at a level that transfers
  - Breaking problems into reusable components
  - NOT from smooth function approximation
""")
print("=" * 70)
