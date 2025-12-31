#!/usr/bin/env python3
"""
Fix Autoregressive Generation

The problem: Model learns to copy decoder input, not to use encoder positions.

The fix: Train cross-attention to directly map encoder positions to outputs.
Each decoder position should attend to its corresponding encoder position
and apply the transformation.

Key insight: For position-aligned tasks (like Caesar cipher):
  - Decoder position 0 should attend to encoder position 0
  - Decoder position 1 should attend to encoder position 1
  - etc.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMEncoderDecoder import RAMEncoderDecoder
from wnn.ram.RAMCrossAttention import RAMCrossAttention, CrossAttentionMode
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import zeros, uint8, cat

print("=" * 70)
print("Fixing Autoregressive Generation")
print("=" * 70)

# Setup
decoder_factory = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder_factory.bits_per_token

def encode_char(c: str):
	return decoder_factory.encode(c).squeeze()

def decode_bits(bits):
	return decoder_factory.decode(bits.unsqueeze(0))

def encode_sequence(text: str) -> list:
	return [encode_char(c) for c in text]

def decode_sequence(tokens: list) -> str:
	return ''.join(decode_bits(t) for t in tokens)

def caesar_shift(text: str, shift: int = 1) -> str:
	result = []
	for c in text:
		if 'A' <= c <= 'Z':
			new_ord = ord('A') + (ord(c) - ord('A') + shift) % 26
			result.append(chr(new_ord))
		else:
			result.append(c)
	return ''.join(result)


# =============================================================
# Approach 1: Train cross-attention patterns explicitly
# =============================================================
print("\n" + "=" * 60)
print("Approach 1: Explicit Cross-Attention Training")
print("=" * 60)
print("Train: decoder[i] should attend to encoder[i]")

model1 = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	cross_attention_mode=CrossAttentionMode.BOTH,  # Use both positions!
	max_encoder_len=16,
	max_decoder_len=16,
	use_residual=True,
	use_ffn=True,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

# Train with diverse decoder inputs (not just correct ones)
train_words = ["CAT", "DOG", "BAT", "HAT", "PIG", "COW", "HEN", "FOX"]
dataset1 = []

print("\nTraining strategy: use various decoder inputs, same target")
for word in train_words:
	source = encode_sequence(word)
	target = encode_sequence(caesar_shift(word))

	# Standard training
	dataset1.append((source, target, target))

	# Also train with zeros as decoder input (forces reliance on encoder)
	zeros_input = [zeros(bits_per_token, dtype=uint8) for _ in range(len(word))]
	dataset1.append((source, zeros_input, target))

	# Train with partial correct + zeros
	for i in range(len(word)):
		partial = target[:i] + [zeros(bits_per_token, dtype=uint8)] * (len(word) - i)
		dataset1.append((source, partial, target))

print(f"Training on {len(dataset1)} examples...")
history = model1.train(dataset1, epochs=15, verbose=True)

# Test autoregressive
print("\n--- Autoregressive Test ---")
test_words = ["SUN", "RUN", "BEE"]

def autoregressive_generate(model, source, length):
	encoder_output = model.encode(source)
	generated = [zeros(bits_per_token, dtype=uint8)]

	for step in range(length):
		outputs = model.decode(generated, encoder_output)
		if step < length - 1:
			generated.append(outputs[step])
		# Replace last zero/prediction with actual prediction
		generated[step] = outputs[step]

	return generated[:length]

for word in test_words:
	source = encode_sequence(word)
	expected = caesar_shift(word)

	generated = autoregressive_generate(model1, source, len(word))
	predicted = decode_sequence(generated)

	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")


# =============================================================
# Approach 2: Direct position-to-output mapping
# =============================================================
print("\n" + "=" * 60)
print("Approach 2: Simpler Architecture - Direct Mapping")
print("=" * 60)
print("Idea: Each output position directly queries encoder position")

# For simple position-aligned tasks, we can use a simpler approach:
# The decoder at position i should directly use encoder[i]

model2 = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=4,  # More heads for better coverage
	position_mode=PositionMode.BINARY,  # Absolute positions
	cross_attention_mode=CrossAttentionMode.BOTH,
	max_encoder_len=8,
	max_decoder_len=8,
	use_residual=False,  # No residual - force learning
	use_ffn=True,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=789,
)

# Simpler training: just source -> target
dataset2 = []
for word in train_words:
	source = encode_sequence(word)
	target = encode_sequence(caesar_shift(word))
	dataset2.append((source, target, target))

	# Critical: also train with zeros input
	zeros_input = [zeros(bits_per_token, dtype=uint8) for _ in range(len(word))]
	dataset2.append((source, zeros_input, target))

print(f"\nTraining on {len(dataset2)} examples...")
history = model2.train(dataset2, epochs=20, verbose=True)

print("\n--- Autoregressive Test ---")
for word in test_words:
	source = encode_sequence(word)
	expected = caesar_shift(word)

	generated = autoregressive_generate(model2, source, len(word))
	predicted = decode_sequence(generated)

	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")


# =============================================================
# Approach 3: Position-indexed generation
# =============================================================
print("\n" + "=" * 60)
print("Approach 3: Position-Indexed Generation")
print("=" * 60)
print("Generate each position independently using encoder")

model3 = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	cross_attention_mode=CrossAttentionMode.BOTH,
	max_encoder_len=8,
	max_decoder_len=8,
	use_residual=True,
	use_ffn=True,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=999,
)

# Train focusing on single-position outputs
dataset3 = []
for word in train_words:
	source = encode_sequence(word)
	target = encode_sequence(caesar_shift(word))

	# Full sequence
	dataset3.append((source, target, target))

	# Single position training: teach each position independently
	for i in range(len(word)):
		# Input: zeros except position i might have something
		single_target = [target[i]]
		single_source_context = source  # Full source always available

		# Decoder input is just first i+1 positions
		partial_decoder = target[:i+1]
		partial_target = target[:i+1]
		dataset3.append((source, partial_decoder, partial_target))

print(f"\nTraining on {len(dataset3)} examples...")
history = model3.train(dataset3, epochs=15, verbose=True)


def position_indexed_generate(model, source, length):
	"""Generate by querying each position separately."""
	encoder_output = model.encode(source)
	generated = []

	for i in range(length):
		# Build decoder input: zeros up to position i
		if i == 0:
			dec_input = [zeros(bits_per_token, dtype=uint8)]
		else:
			dec_input = generated + [zeros(bits_per_token, dtype=uint8)]

		outputs = model.decode(dec_input, encoder_output)
		generated.append(outputs[i] if i < len(outputs) else outputs[-1])

	return generated


print("\n--- Position-Indexed Generation ---")
for word in test_words:
	source = encode_sequence(word)
	expected = caesar_shift(word)

	generated = position_indexed_generate(model3, source, len(word))
	predicted = decode_sequence(generated)

	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Fixing Autoregressive Generation")
print("=" * 60)

print("""
The Core Problem:
  Standard training teaches: "given correct prefix, predict next"
  But autoregressive needs: "given MY predictions, predict next"

Solutions Explored:

1. Diverse Decoder Inputs:
   - Train with zeros, partial correct, random inputs
   - Forces model to rely on encoder positions, not decoder content

2. No Residual Connection:
   - Removes shortcut that copies decoder input to output
   - Forces learning of encoder->output mapping

3. Position-Indexed Generation:
   - Generate each position using only previous generated tokens
   - More explicit position awareness

Key Insight:
  For position-aligned tasks (source[i] -> target[i]):
  Cross-attention should learn: "decoder position i attends to encoder position i"
  This requires training where decoder content is uninformative!
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
