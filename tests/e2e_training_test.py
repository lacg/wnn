#!/usr/bin/env python3
"""
End-to-End Training Test for RAM Transformer

Tests the RAMTrainer on various sequence tasks:
1. Identity mapping (learn to copy)
2. Next character prediction
3. Simple sequence transformation
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMSeq2Seq import RAMSeq2Seq
from wnn.ram.RAMTrainer import RAMTrainer
from wnn.ram.RAMFeedForward import FFNMode
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import tensor, uint8

print("=" * 70)
print("End-to-End Training Test")
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
# Test 1: Identity Mapping (Learn to Copy)
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Identity Mapping (Copy Task)")
print("=" * 60)
print("Task: Input 'ABC' -> Output 'ABC'")

model_copy = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_ffn=False,
	generalization=MapperStrategy.DIRECT,
	rng=42,
)

trainer_copy = RAMTrainer(model_copy, verbose=True)

# Create dataset: sequences that should map to themselves
copy_dataset = [
	(encode_sequence("ABC"), encode_sequence("ABC")),
	(encode_sequence("DEF"), encode_sequence("DEF")),
	(encode_sequence("GHI"), encode_sequence("GHI")),
]

print("\nTraining on copy task...")
history = trainer_copy.train(copy_dataset, epochs=5)

print("\nEvaluation:")
eval_result = trainer_copy.evaluate(copy_dataset, decoder=decode_bits)
print(f"  Accuracy: {eval_result['accuracy']:.1f}%")
for ex in eval_result['examples'][:3]:
	status = "✓" if ex['correct'] else "✗"
	print(f"  {status} '{ex['output']}' (target: '{ex['target']}')")


# =============================================================
# Test 2: Next Character (with BIT_LEVEL generalization)
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Next Character Prediction")
print("=" * 60)
print("Task: 'A' -> 'B', 'B' -> 'C', etc.")

model_next = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,  # Crucial! Allows input passthrough when attention is zeros
	use_ffn=False,
	generalization=MapperStrategy.BIT_LEVEL,  # Key for generalization!
	rng=42,
)

trainer_next = RAMTrainer(model_next, verbose=True)

# Create dataset: char -> next_char
next_char_dataset = []
train_chars = "ABCDEFGH"  # Train on first 8
for c in train_chars:
	if c < 'Z':
		inp = [encode_char(c)]
		tgt = [encode_char(chr(ord(c) + 1))]
		next_char_dataset.append((inp, tgt))

print(f"\nTraining on {len(next_char_dataset)} examples: {train_chars}")
history = trainer_next.train(next_char_dataset, epochs=10)

# Test on unseen characters
print("\nTesting on UNSEEN characters (I-N):")
test_chars = "IJKLMN"
correct = 0
for c in test_chars:
	inp = [encode_char(c)]
	out = model_next.forward(inp)
	pred = decode_bits(out[0])
	expected = chr(ord(c) + 1)
	status = "✓" if pred == expected else "✗"
	if pred == expected:
		correct += 1
	print(f"  {status} {c} -> {pred} (expected {expected})")

print(f"\nGeneralization accuracy: {100*correct/len(test_chars):.1f}%")


# =============================================================
# Test 3: Full Transformer with FFN
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Full Transformer (Attention + FFN)")
print("=" * 60)
print("Task: Sequence shift - 'ABC' -> 'BCD'")

model_full = RAMSeq2Seq(
	input_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_ffn=True,
	ffn_expansion=4,
	ffn_mode=FFNMode.STANDARD,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

trainer_full = RAMTrainer(model_full, verbose=True)

# Create dataset: shift each character forward
shift_dataset = [
	(encode_sequence("ABC"), encode_sequence("BCD")),
	(encode_sequence("DEF"), encode_sequence("EFG")),
	(encode_sequence("GHI"), encode_sequence("HIJ")),
]

print("\nTraining on shift task...")
history = trainer_full.train(shift_dataset, epochs=10)

print("\nEvaluation on training data:")
eval_result = trainer_full.evaluate(shift_dataset, decoder=decode_bits)
print(f"  Accuracy: {eval_result['accuracy']:.1f}%")

# Test generalization
print("\nTesting generalization on unseen sequences:")
test_sequences = [
	(encode_sequence("JKL"), "KLM"),
	(encode_sequence("MNO"), "NOP"),
]

for inp, expected in test_sequences:
	out = model_full.forward(inp)
	pred = decode_sequence(out)
	status = "✓" if pred == expected else "✗"
	print(f"  {status} '{decode_sequence(inp)}' -> '{pred}' (expected '{expected}')")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: End-to-End Training")
print("=" * 60)

print("""
End-to-End EDRA Training Pipeline:

1. Forward Pass
   └─ Record all intermediate states (attention, FFN, projections)

2. Error Detection
   └─ Compare output to target, count bit errors

3. Backward Target Computation
   └─ For each layer, compute: "what SHOULD you have produced?"
   └─ XOR residual: layer_output = input ⊕ target_output

4. Layer Training
   └─ Train each layer on its input -> desired_output
   └─ FFN: commit expansion and projection
   └─ Token mapper: commit transformation patterns

Key Insights:
- BIT_LEVEL generalization enables learning with sparse examples
- Residual connections allow target backpropagation
- No gradients needed - pure constraint solving
""")

print("=" * 60)
print("All tests completed!")
print("=" * 60)
