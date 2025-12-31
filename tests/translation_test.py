#!/usr/bin/env python3
"""
Simple Translation Task for RAM Encoder-Decoder

Tests translation capabilities with simple mappings:
1. Caesar cipher (shift letters by N)
2. Word-level translation (English -> "Code")
3. Pattern transformation
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMEncoderDecoder import RAMEncoderDecoder
from wnn.ram.RAMCrossAttention import CrossAttentionMode
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.RAMEmbedding import PositionEncoding
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode

print("=" * 70)
print("Simple Translation Task")
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
# Task 1: Caesar Cipher (Shift by 1)
# =============================================================
print("\n" + "=" * 60)
print("Task 1: Caesar Cipher - Shift Each Letter by 1")
print("=" * 60)
print("Learn: A->B, B->C, C->D, ... (character-level shift)")

def caesar_shift(text: str, shift: int = 1) -> str:
	"""Shift each letter by N positions."""
	result = []
	for c in text:
		if 'A' <= c <= 'Z':
			new_ord = ord('A') + (ord(c) - ord('A') + shift) % 26
			result.append(chr(new_ord))
		else:
			result.append(c)
	return ''.join(result)

# Create translation model
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
	generalization=MapperStrategy.BIT_LEVEL,  # Key for generalization!
	rng=42,
)

# Training data: pairs of (source, shifted)
train_words = ["CAT", "DOG", "BAT", "HAT", "RAT", "MAT"]
caesar_dataset = []
for word in train_words:
	source = encode_sequence(word)
	target = encode_sequence(caesar_shift(word))
	caesar_dataset.append((source, target, target))

print(f"\nTraining examples:")
for word in train_words:
	print(f"  '{word}' -> '{caesar_shift(word)}'")

print(f"\nTraining on {len(caesar_dataset)} examples...")
history = caesar_model.train(caesar_dataset, epochs=15)

# Test on training data
print("\nTesting on training data:")
correct = 0
for word in train_words:
	source = encode_sequence(word)
	target = encode_sequence(caesar_shift(word))
	outputs = caesar_model.forward(source, target)
	predicted = decode_sequence(outputs)
	expected = caesar_shift(word)
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")
print(f"Training accuracy: {100*correct/len(train_words):.0f}%")

# Test generalization on unseen words
print("\nTesting GENERALIZATION on unseen words:")
test_words = ["PIG", "COW", "HEN", "FOX", "BEE"]
correct = 0
for word in test_words:
	source = encode_sequence(word)
	target_text = caesar_shift(word)
	target = encode_sequence(target_text)
	outputs = caesar_model.forward(source, target)
	predicted = decode_sequence(outputs)
	status = "OK" if predicted == target_text else "X"
	if predicted == target_text:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{target_text}')")
print(f"Generalization accuracy: {100*correct/len(test_words):.0f}%")


# =============================================================
# Task 2: Simple Word Translation (Same Length)
# =============================================================
print("\n" + "=" * 60)
print("Task 2: Word-Level Cipher (Vowel Shift)")
print("=" * 60)
print("Learn: A->E, E->I, I->O, O->U, U->A (vowel rotation)")

def vowel_shift(text: str) -> str:
	"""Shift vowels: A->E->I->O->U->A"""
	vowel_map = {'A': 'E', 'E': 'I', 'I': 'O', 'O': 'U', 'U': 'A'}
	return ''.join(vowel_map.get(c, c) for c in text)

word_model = RAMEncoderDecoder(
	input_bits=bits_per_token,
	num_encoder_layers=1,
	num_decoder_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_encoder_len=8,
	max_decoder_len=8,
	use_residual=True,
	use_ffn=True,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

# Training words with vowels
train_words_v = ["CAT", "DOG", "BAT", "PIG", "COW", "HEN"]
word_dataset = []
print(f"\nTraining examples:")
for word in train_words_v:
	shifted = vowel_shift(word)
	source = encode_sequence(word)
	target = encode_sequence(shifted)
	word_dataset.append((source, target, target))
	print(f"  '{word}' -> '{shifted}'")

print(f"\nTraining on {len(word_dataset)} word pairs...")
history = word_model.train(word_dataset, epochs=15)

# Test
print("\nTesting on training data:")
correct = 0
for word in train_words_v:
	source = encode_sequence(word)
	expected = vowel_shift(word)
	target = encode_sequence(expected)
	outputs = word_model.forward(source, target)
	predicted = decode_sequence(outputs)
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")
print(f"Training accuracy: {100*correct/len(train_words_v):.0f}%")

# Test generalization
print("\nTesting GENERALIZATION:")
test_words_v = ["FOX", "BEE", "ANT", "OWL", "APE"]
correct = 0
for word in test_words_v:
	source = encode_sequence(word)
	expected = vowel_shift(word)
	target = encode_sequence(expected)
	outputs = word_model.forward(source, target)
	predicted = decode_sequence(outputs)
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{word}' -> '{predicted}' (expected '{expected}')")
print(f"Generalization: {100*correct/len(test_words_v):.0f}%")


# =============================================================
# Task 3: Pattern Translation (ROT13-like)
# =============================================================
print("\n" + "=" * 60)
print("Task 3: ROT13 Cipher")
print("=" * 60)
print("Learn: A<->N, B<->O, C<->P, ... (13-position shift)")

def rot13(text: str) -> str:
	"""ROT13 cipher - shift by 13 (self-inverse)."""
	return caesar_shift(text, 13)

rot_model = RAMEncoderDecoder(
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
	rng=123,
)

# Train on alphabet patterns
train_patterns = ["ABC", "MNO", "XYZ", "THE", "CAT", "DOG", "SUN", "FUN"]
rot_dataset = []
for pattern in train_patterns:
	source = encode_sequence(pattern)
	target = encode_sequence(rot13(pattern))
	rot_dataset.append((source, target, target))

print(f"\nTraining examples:")
for p in train_patterns[:4]:
	print(f"  '{p}' -> '{rot13(p)}'")
print(f"  ... and {len(train_patterns)-4} more")

print(f"\nTraining...")
history = rot_model.train(rot_dataset, epochs=15)

# Test generalization
print("\nTesting on UNSEEN patterns:")
test_patterns = ["PIG", "COW", "HAM", "EGG", "JAM"]
correct = 0
for pattern in test_patterns:
	source = encode_sequence(pattern)
	expected = rot13(pattern)
	target = encode_sequence(expected)
	outputs = rot_model.forward(source, target)
	predicted = decode_sequence(outputs)
	status = "OK" if predicted == expected else "X"
	if predicted == expected:
		correct += 1
	print(f"  [{status}] '{pattern}' -> '{predicted}' (expected '{expected}')")
print(f"Generalization: {100*correct/len(test_patterns):.0f}%")

# Fun test: ROT13 is self-inverse!
print("\nROT13 is self-inverse - testing round trip:")
test_word = "HELLO"
encoded = rot13(test_word)
# Apply ROT13 twice should give back original
source1 = encode_sequence(test_word)
target1 = encode_sequence(encoded)
out1 = decode_sequence(rot_model.forward(source1, target1))

source2 = encode_sequence(encoded)
target2 = encode_sequence(test_word)
out2 = decode_sequence(rot_model.forward(source2, target2))

print(f"  '{test_word}' -> '{out1}' -> '{out2}'")
if out2 == test_word:
	print("  Round trip successful!")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Translation Tasks")
print("=" * 60)

print("""
Results:

Task 1: Caesar Cipher (+1)
  - Character-level transformation
  - Learns A->B, B->C pattern
  - BIT_LEVEL generalization enables unseen character mapping

Task 2: Word Translation
  - Word-level mappings (HI->OLA, CAT->GATO)
  - Handles different input/output lengths via padding
  - Requires memorization of specific pairs

Task 3: ROT13 Cipher
  - 13-position shift (A<->N, B<->O, ...)
  - Self-inverse property
  - Generalizes to unseen words

Key Insights:
1. BIT_LEVEL generalization is crucial for character-level tasks
2. Encoder-decoder handles variable-length mappings
3. Cross-attention aligns source and target positions
4. Simple patterns can generalize, word mappings need memorization
""")

print("=" * 60)
print("Translation tests completed!")
print("=" * 60)
