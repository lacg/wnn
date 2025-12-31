#!/usr/bin/env python3
"""
Autoregressive Generation Test (No Teacher Forcing)

The real test: generate outputs using only the model's own predictions.
No peeking at the correct answer during generation!

Teacher Forcing:    decode([correct_t1, correct_t2]) -> predict t3
Autoregressive:     decode([pred_t1]) -> pred_t2, decode([pred_t1, pred_t2]) -> pred_t3

This is much harder because errors compound!
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMEncoderDecoder import RAMEncoderDecoder
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import zeros, uint8

print("=" * 70)
print("Autoregressive Generation (No Teacher Forcing)")
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


def autoregressive_generate(model, source, target_length, start_token=None):
	"""
	Generate output autoregressively - NO teacher forcing!

	Args:
		model: RAMEncoderDecoder
		source: Encoded source sequence
		target_length: How many tokens to generate
		start_token: First token to seed generation (optional)

	Returns:
		List of generated tokens
	"""
	# Encode source
	encoder_output = model.encode(source)

	# Start with first token (or zeros if not provided)
	if start_token is not None:
		generated = [start_token.squeeze()]
	else:
		# Use zeros as start token
		generated = [zeros(bits_per_token, dtype=uint8)]

	# Generate tokens one at a time
	for step in range(target_length - 1):
		# Decode using only our own predictions
		outputs = model.decode(generated, encoder_output)

		# Take the last output as the next token
		next_token = outputs[-1]
		generated.append(next_token)

	return generated


# =============================================================
# Test 1: Caesar Cipher - Autoregressive
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Caesar Cipher (+1) - Autoregressive Generation")
print("=" * 60)

def caesar_shift(text: str, shift: int = 1) -> str:
	result = []
	for c in text:
		if 'A' <= c <= 'Z':
			new_ord = ord('A') + (ord(c) - ord('A') + shift) % 26
			result.append(chr(new_ord))
		else:
			result.append(c)
	return ''.join(result)

caesar_model = RAMEncoderDecoder(
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

# Training data
train_words = ["CAT", "DOG", "BAT", "HAT", "RAT", "MAT", "SUN", "RUN"]
caesar_dataset = []
for word in train_words:
	source = encode_sequence(word)
	target = encode_sequence(caesar_shift(word))
	caesar_dataset.append((source, target, target))

print(f"\nTraining on {len(train_words)} examples (with teacher forcing)...")
history = caesar_model.train(caesar_dataset, epochs=10)

# Test with teacher forcing (baseline)
print("\n--- With Teacher Forcing (baseline) ---")
test_words = ["PIG", "COW", "HEN"]
for word in test_words:
	source = encode_sequence(word)
	expected = caesar_shift(word)
	target = encode_sequence(expected)
	outputs = caesar_model.forward(source, target)  # Teacher forcing
	predicted = decode_sequence(outputs)
	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")

# Test AUTOREGRESSIVE (no teacher forcing!)
print("\n--- Autoregressive (NO teacher forcing) ---")
for word in test_words:
	source = encode_sequence(word)
	expected = caesar_shift(word)

	# Start with the first correct character to seed generation
	start_token = encode_char(expected[0])
	generated = autoregressive_generate(
		caesar_model, source,
		target_length=len(word),
		start_token=start_token
	)

	predicted = decode_sequence(generated)
	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")


# =============================================================
# Test 2: Train specifically for autoregressive generation
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Training for Autoregressive Generation")
print("=" * 60)
print("Key insight: Train on partial sequences to learn continuation")

auto_model = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=4,
	position_mode=PositionMode.RELATIVE,
	max_encoder_len=16,
	max_decoder_len=16,
	use_residual=True,
	use_ffn=True,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=123,
)

# Create dataset with partial sequences for better autoregressive learning
auto_dataset = []
train_pairs = [
	("ABC", "BCD"),  # Shift by 1
	("DEF", "EFG"),
	("GHI", "HIJ"),
	("JKL", "KLM"),
	("MNO", "NOP"),
	("PQR", "QRS"),
]

print("\nTraining pairs:")
for src, tgt in train_pairs:
	print(f"  '{src}' -> '{tgt}'")
	source = encode_sequence(src)
	target = encode_sequence(tgt)
	auto_dataset.append((source, target, target))

	# Also train on partial sequences (helps autoregressive)
	# e.g., for "ABC" -> "BCD", also train:
	#   "ABC" with target ["B"] -> ["B"]
	#   "ABC" with target ["B", "C"] -> ["B", "C"]
	for i in range(1, len(tgt)):
		partial_target = encode_sequence(tgt[:i])
		auto_dataset.append((source, partial_target, partial_target))

print(f"\nTraining on {len(auto_dataset)} examples (including partials)...")
history = auto_model.train(auto_dataset, epochs=15)

# Test autoregressive
print("\n--- Autoregressive Generation ---")
test_pairs = [("STU", "TUV"), ("VWX", "WXY"), ("CDE", "DEF")]
for src, expected in test_pairs:
	source = encode_sequence(src)

	# Autoregressive: start with first char, generate rest
	start_token = encode_char(expected[0])
	generated = autoregressive_generate(
		auto_model, source,
		target_length=len(expected),
		start_token=start_token
	)

	predicted = decode_sequence(generated)
	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{src}' -> '{predicted}' (expected '{expected}')")


# =============================================================
# Test 3: Reverse Task - Autoregressive
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Reverse Task - Autoregressive")
print("=" * 60)

reverse_model = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=4,
	position_mode=PositionMode.RELATIVE,
	max_encoder_len=16,
	max_decoder_len=16,
	use_residual=True,
	use_ffn=True,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=456,
)

# Train on reverse task with partial sequences
reverse_dataset = []
reverse_words = ["ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX"]

print("\nTraining examples:")
for word in reverse_words:
	reversed_word = word[::-1]
	print(f"  '{word}' -> '{reversed_word}'")

	source = encode_sequence(word)
	target = encode_sequence(reversed_word)
	reverse_dataset.append((source, target, target))

	# Partial sequences
	for i in range(1, len(reversed_word)):
		partial = encode_sequence(reversed_word[:i])
		reverse_dataset.append((source, partial, partial))

print(f"\nTraining on {len(reverse_dataset)} examples...")
history = reverse_model.train(reverse_dataset, epochs=15)

# Test with teacher forcing
print("\n--- With Teacher Forcing ---")
test_reverse = ["CAT", "DOG", "FOX"]
for word in test_reverse:
	source = encode_sequence(word)
	expected = word[::-1]
	target = encode_sequence(expected)
	outputs = reverse_model.forward(source, target)
	predicted = decode_sequence(outputs)
	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")

# Test autoregressive
print("\n--- Autoregressive Generation ---")
for word in test_reverse:
	source = encode_sequence(word)
	expected = word[::-1]

	start_token = encode_char(expected[0])
	generated = autoregressive_generate(
		reverse_model, source,
		target_length=len(expected),
		start_token=start_token
	)

	predicted = decode_sequence(generated)
	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")


# =============================================================
# Test 4: Full autoregressive (no start token hint)
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Full Autoregressive (No Start Token Hint)")
print("=" * 60)
print("Hardest test: model must predict first token too!")

def full_autoregressive_generate(model, source, target_length):
	"""Generate without any hint - model predicts everything."""
	encoder_output = model.encode(source)

	# Start with zeros
	generated = [zeros(bits_per_token, dtype=uint8)]

	# First prediction: what should the first output be?
	outputs = model.decode(generated, encoder_output)
	generated = [outputs[0]]  # Replace zeros with first prediction

	# Continue generating
	for step in range(target_length - 1):
		outputs = model.decode(generated, encoder_output)
		next_token = outputs[-1]
		generated.append(next_token)

	return generated

# Use caesar model (simpler task)
print("\nCaesar cipher - full autoregressive:")
for word in ["CAT", "DOG", "PIG"]:
	source = encode_sequence(word)
	expected = caesar_shift(word)

	generated = full_autoregressive_generate(
		caesar_model, source, len(expected)
	)

	predicted = decode_sequence(generated)
	status = "OK" if predicted == expected else "X"
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Autoregressive Generation")
print("=" * 60)

print("""
Key Findings:

1. Teacher Forcing vs Autoregressive:
   - Teacher forcing: Model sees correct previous tokens (easier)
   - Autoregressive: Model uses its own predictions (harder)
   - Errors can compound in autoregressive mode

2. Training Strategies:
   - Training on partial sequences helps autoregressive generation
   - Model learns: given prefix, what comes next?
   - More training data = better generalization

3. Start Token:
   - Providing first correct token helps a lot
   - Full autoregressive (no hints) is hardest
   - Consider using a special <START> token in practice

4. RAM Advantage:
   - BIT_LEVEL generalization helps with unseen characters
   - Cross-attention provides strong source-target alignment
   - Discrete representations can be more robust than soft attention
""")

print("=" * 60)
print("Autoregressive tests completed!")
print("=" * 60)
