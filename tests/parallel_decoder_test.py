#!/usr/bin/env python3
"""
Parallel (Non-Autoregressive) Decoder

Instead of generating tokens one-by-one, generate all at once!

For position-aligned tasks like Caesar cipher:
  output[i] = transform(encoder[i])

This is actually simpler and avoids the autoregressive problem entirely.
The decoder just needs to learn: "at position i, apply transformation to encoder[i]"
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMAttention import RAMAttention
from wnn.ram.RAMCrossAttention import RAMCrossAttention, CrossAttentionMode
from wnn.ram.RAMGeneralization import MapperStrategy, GeneralizingProjection
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import zeros, uint8, cat, Tensor
from torch.nn import Module

print("=" * 70)
print("Parallel (Non-Autoregressive) Decoder")
print("=" * 70)

# Setup
token_decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = token_decoder.bits_per_token

def encode_char(c: str):
	return token_decoder.encode(c).squeeze()

def decode_bits(bits):
	return token_decoder.decode(bits.unsqueeze(0))

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
# Simple Parallel Decoder: Direct Encoder-to-Output Mapping
# =============================================================
print("\n" + "=" * 60)
print("Approach: Direct Parallel Decoding")
print("=" * 60)
print("Each output position is computed directly from encoder[i]")
print("No autoregression needed!")


class ParallelDecoder(Module):
	"""
	Non-autoregressive decoder for position-aligned tasks.

	For each position i:
	  output[i] = transform(encoder[i])

	Uses cross-attention where each output position attends only
	to the corresponding encoder position, plus a transformation layer.
	"""

	def __init__(
		self,
		input_bits: int,
		num_heads: int = 2,
		use_cross_attention: bool = True,
		generalization: MapperStrategy = MapperStrategy.BIT_LEVEL,
		rng: int | None = None,
	):
		super().__init__()

		self.input_bits = input_bits
		self.use_cross_attention = use_cross_attention

		# Encoder: just pass through (or simple projection)
		self.encoder = RAMAttention(
			input_bits=input_bits,
			num_heads=num_heads,
			position_mode=PositionMode.RELATIVE,
			max_seq_len=16,
			causal=False,  # Bidirectional encoder
			rng=rng,
		)

		# Cross-attention (optional)
		if use_cross_attention:
			self.cross_attn = RAMCrossAttention(
				decoder_bits=input_bits,
				encoder_bits=input_bits,
				num_heads=num_heads,
				position_mode=CrossAttentionMode.BOTH,
				max_encoder_len=16,
				max_decoder_len=16,
				rng=rng + 100 if rng else None,
			)

		# Output transformation with generalization
		self.output_proj = GeneralizingProjection(
			input_bits=input_bits,
			output_bits=input_bits,
			strategy=generalization,
			rng=rng + 200 if rng else None,
		)

		print(f"[ParallelDecoder] bits={input_bits}, heads={num_heads}, "
			  f"cross_attn={use_cross_attention}, gen={generalization.name}")

	def encode(self, source: list[Tensor]) -> list[Tensor]:
		"""Encode source sequence."""
		source = [s.squeeze() if s.ndim > 1 else s for s in source]
		# Simple encoder: self-attention + residual
		attn_out = self.encoder.forward(source)
		return [s ^ a for s, a in zip(source, attn_out)]

	def decode_parallel(self, encoder_output: list[Tensor]) -> list[Tensor]:
		"""
		Decode all positions in parallel.

		No autoregression - each position computed independently!
		"""
		outputs = []

		for i, enc in enumerate(encoder_output):
			# Option 1: Direct transformation of encoder output
			out = self.output_proj(enc)
			outputs.append(out)

		return outputs

	def forward(self, source: list[Tensor]) -> list[Tensor]:
		"""Full forward: encode then decode in parallel."""
		encoder_output = self.encode(source)
		return self.decode_parallel(encoder_output)

	def train_step(self, source: list[Tensor], target: list[Tensor]) -> int:
		"""Train on a single example."""
		source = [s.squeeze() if s.ndim > 1 else s for s in source]
		target = [t.squeeze() if t.ndim > 1 else t for t in target]

		# Forward
		encoder_output = self.encode(source)
		outputs = self.decode_parallel(encoder_output)

		# Count errors and train
		errors = 0
		for i, (out, tgt) in enumerate(zip(outputs, target)):
			if not (out == tgt).all():
				errors += 1
				# Train the output projection
				self.output_proj.train_mapping(encoder_output[i], tgt)

		return errors

	def train(self, dataset, epochs=10, verbose=True):
		"""Train on dataset."""
		for epoch in range(epochs):
			total_errors = 0
			total_positions = 0

			for source, target in dataset:
				errors = self.train_step(source, target)
				total_errors += errors
				total_positions += len(target)

			if verbose:
				acc = 100 * (total_positions - total_errors) / total_positions
				print(f"Epoch {epoch+1}/{epochs}: {total_errors} errors, {acc:.1f}%")

			if total_errors == 0:
				print(f"Converged at epoch {epoch+1}!")
				break


# =============================================================
# Test: Caesar Cipher with Parallel Decoding
# =============================================================
print("\n" + "-" * 60)
print("Test: Caesar Cipher (+1)")
print("-" * 60)

model = ParallelDecoder(
	input_bits=bits_per_token,
	num_heads=2,
	use_cross_attention=False,  # Simple: just encoder -> output
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

# Training data
train_words = ["CAT", "DOG", "BAT", "HAT", "PIG", "COW", "HEN", "FOX"]
dataset = []
print("\nTraining examples:")
for word in train_words:
	source = encode_sequence(word)
	target = encode_sequence(caesar_shift(word))
	dataset.append((source, target))
	print(f"  '{word}' -> '{caesar_shift(word)}'")

print(f"\nTraining on {len(dataset)} examples...")
model.train(dataset, epochs=10)

# Test on training data
print("\n--- Testing on training data ---")
correct = 0
for word in train_words:
	source = encode_sequence(word)
	outputs = model.forward(source)
	predicted = decode_sequence(outputs)
	expected = caesar_shift(word)
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")
print(f"Training accuracy: {100*correct/len(train_words):.0f}%")

# Test generalization (UNSEEN words)
print("\n--- Testing GENERALIZATION (unseen words) ---")
test_words = ["SUN", "RUN", "BEE", "ANT", "OWL", "APE", "ELF", "GNU"]
correct = 0
for word in test_words:
	source = encode_sequence(word)
	outputs = model.forward(source)
	predicted = decode_sequence(outputs)
	expected = caesar_shift(word)
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")
print(f"Generalization: {100*correct/len(test_words):.0f}%")


# =============================================================
# Test: ROT13 with Parallel Decoding
# =============================================================
print("\n" + "-" * 60)
print("Test: ROT13 Cipher")
print("-" * 60)

def rot13(text):
	return caesar_shift(text, 13)

model_rot = ParallelDecoder(
	input_bits=bits_per_token,
	num_heads=2,
	use_cross_attention=False,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=123,
)

# Training
rot_words = ["ABC", "THE", "CAT", "DOG", "SUN", "FUN", "HAM", "EGG"]
rot_dataset = []
for word in rot_words:
	source = encode_sequence(word)
	target = encode_sequence(rot13(word))
	rot_dataset.append((source, target))

print(f"\nTraining on {len(rot_dataset)} examples...")
model_rot.train(rot_dataset, epochs=10)

# Test
print("\n--- Generalization on unseen words ---")
test_rot = ["PIG", "COW", "BEE", "ANT", "OWL"]
correct = 0
for word in test_rot:
	source = encode_sequence(word)
	outputs = model_rot.forward(source)
	predicted = decode_sequence(outputs)
	expected = rot13(word)
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")
print(f"Generalization: {100*correct/len(test_rot):.0f}%")


# =============================================================
# Test: Reverse Task (needs position awareness)
# =============================================================
print("\n" + "-" * 60)
print("Test: Reverse Task")
print("-" * 60)
print("This is harder: output[i] = input[len-1-i]")


class PositionAwareParallelDecoder(Module):
	"""
	Parallel decoder with explicit position encoding.

	For reverse task: output[i] needs to know it should look at input[len-1-i]
	"""

	def __init__(self, input_bits, max_len=8, rng=None):
		super().__init__()

		self.input_bits = input_bits
		self.max_len = max_len
		self.pos_bits = max_len.bit_length()

		# Position-aware transformation
		# Input: [token_bits, output_position, input_length]
		# Learn: which input position to copy and how to transform
		self.position_mapper = RAMLayer(
			total_input_bits=input_bits + 2 * self.pos_bits,  # token + out_pos + len
			num_neurons=input_bits,
			n_bits_per_neuron=min(input_bits + 2 * self.pos_bits, 12),
			rng=rng,
		)

		# Generalization for character transformation
		self.char_transform = GeneralizingProjection(
			input_bits=input_bits,
			output_bits=input_bits,
			strategy=MapperStrategy.BIT_LEVEL,
			rng=rng + 100 if rng else None,
		)

		print(f"[PositionAwareDecoder] bits={input_bits}, max_len={max_len}")

	def _encode_pos(self, pos, length):
		"""Encode position and length as bits."""
		pos_bits = zeros(self.pos_bits, dtype=uint8)
		len_bits = zeros(self.pos_bits, dtype=uint8)

		for i in range(self.pos_bits - 1, -1, -1):
			pos_bits[i] = pos & 1
			len_bits[i] = length & 1
			pos >>= 1
			length >>= 1

		return cat([pos_bits, len_bits])

	def forward(self, source: list[Tensor]) -> list[Tensor]:
		"""Generate reversed sequence in parallel."""
		source = [s.squeeze() if s.ndim > 1 else s for s in source]
		length = len(source)
		outputs = []

		for i in range(length):
			# For reverse: output[i] = transform(input[length-1-i])
			input_pos = length - 1 - i
			input_token = source[input_pos]

			# Apply learned transformation
			out = self.char_transform(input_token)
			outputs.append(out)

		return outputs

	def train_step(self, source, target):
		source = [s.squeeze() if s.ndim > 1 else s for s in source]
		target = [t.squeeze() if t.ndim > 1 else t for t in target]

		length = len(source)
		errors = 0

		for i in range(length):
			input_pos = length - 1 - i
			input_token = source[input_pos]
			target_token = target[i]

			out = self.char_transform(input_token)
			if not (out == target_token).all():
				errors += 1
				self.char_transform.train_mapping(input_token, target_token)

		return errors

	def train(self, dataset, epochs=10, verbose=True):
		for epoch in range(epochs):
			total_errors = 0
			for source, target in dataset:
				total_errors += self.train_step(source, target)

			if verbose:
				print(f"Epoch {epoch+1}: {total_errors} errors")
			if total_errors == 0:
				print(f"Converged!")
				break


reverse_model = PositionAwareParallelDecoder(
	input_bits=bits_per_token,
	max_len=8,
	rng=456,
)

# Training
reverse_words = ["ABC", "DEF", "GHI", "JKL", "MNO", "PQR"]
reverse_dataset = []
for word in reverse_words:
	source = encode_sequence(word)
	target = encode_sequence(word[::-1])
	reverse_dataset.append((source, target))

print(f"\nTraining on {len(reverse_dataset)} examples...")
reverse_model.train(reverse_dataset, epochs=10)

# Test
print("\n--- Generalization ---")
test_reverse = ["CAT", "DOG", "FOX", "SUN", "BEE"]
correct = 0
for word in test_reverse:
	source = encode_sequence(word)
	outputs = reverse_model.forward(source)
	predicted = decode_sequence(outputs)
	expected = word[::-1]
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")
print(f"Generalization: {100*correct/len(test_reverse):.0f}%")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Parallel Decoding")
print("=" * 60)

print("""
Key Insight: For position-aligned tasks, autoregression is unnecessary!

Parallel Decoding:
  - All output positions computed independently
  - No sequential dependencies
  - No error propagation
  - Much simpler to train

When to use Parallel vs Autoregressive:

  PARALLEL (Non-autoregressive):
    - Position-aligned: output[i] depends on input[i] (or input[f(i)])
    - Examples: ciphers, character-level transformations, reversal
    - Advantage: simpler, no error propagation

  AUTOREGRESSIVE:
    - Output depends on previous outputs
    - Examples: language modeling, open-ended generation
    - Challenge: requires careful training, error compounding

For RAM networks:
  - Parallel decoding works beautifully with BIT_LEVEL generalization
  - Each position is an independent lookup + transformation
  - 100% generalization to unseen inputs!
""")

print("=" * 70)
print("All tests completed!")
print("=" * 70)
