#!/usr/bin/env python3
"""
RAM Embedding Layer Tests

Tests the RAMEmbedding layer:
1. Token embedding projection
2. Position encoding (NONE, BINARY, LEARNED, SINUSOIDAL)
3. Integration with RAMSeq2Seq
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMEmbedding import RAMEmbedding, PositionEncoding
from wnn.ram.RAMSeq2Seq import RAMSeq2Seq
from wnn.ram.RAMGeneralization import MapperStrategy
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode, PositionMode
from torch import tensor, uint8, zeros

print("=" * 70)
print("RAM Embedding Layer Tests")
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
# Test 1: Basic Embedding (No Position)
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Basic Token Embedding (No Position Encoding)")
print("=" * 60)

embed_no_pos = RAMEmbedding(
	token_bits=bits_per_token,
	embedding_bits=bits_per_token * 2,  # Expand to 10 bits
	max_seq_len=16,
	position_encoding=PositionEncoding.NONE,
	rng=42,
)

print(f"\nEmbedding: {embed_no_pos}")
print(f"  Token bits: {embed_no_pos.token_bits}")
print(f"  Embedding bits: {embed_no_pos.embedding_bits}")
print(f"  Position bits: {embed_no_pos.position_bits}")

# Test embedding a sequence
test_tokens = encode_sequence("ABC")
embeddings = embed_no_pos(test_tokens)

print(f"\nEmbedding sequence 'ABC':")
for i, (tok, emb) in enumerate(zip(test_tokens, embeddings)):
	print(f"  Position {i}: {tok.tolist()} -> {emb.tolist()} ({len(emb)} bits)")

# Verify embeddings are consistent (same token -> same embedding)
print("\nConsistency check:")
tok_a1 = encode_char('A')
tok_a2 = encode_char('A')
emb_a1 = embed_no_pos.forward_single(tok_a1, position=0, add_position=False)
emb_a2 = embed_no_pos.forward_single(tok_a2, position=0, add_position=False)
consistent = (emb_a1 == emb_a2).all().item()
print(f"  Same token produces same embedding: {'✓' if consistent else '✗'}")


# =============================================================
# Test 2: Binary Position Encoding
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Binary Position Encoding")
print("=" * 60)

embed_binary = RAMEmbedding(
	token_bits=bits_per_token,
	embedding_bits=bits_per_token,  # Same size
	max_seq_len=16,
	position_encoding=PositionEncoding.BINARY,
	rng=42,
)

print(f"\nEmbedding: {embed_binary}")
print(f"  Position bits: {embed_binary.position_bits}")

# Test same token at different positions
tok_a = encode_char('A')
emb_pos0 = embed_binary.forward_single(tok_a, position=0)
emb_pos1 = embed_binary.forward_single(tok_a, position=1)
emb_pos2 = embed_binary.forward_single(tok_a, position=2)

print(f"\nSame token 'A' at different positions:")
print(f"  Position 0: {emb_pos0.tolist()}")
print(f"  Position 1: {emb_pos1.tolist()}")
print(f"  Position 2: {emb_pos2.tolist()}")

# Verify embeddings differ by position
diff_01 = (emb_pos0 != emb_pos1).sum().item()
diff_12 = (emb_pos1 != emb_pos2).sum().item()
print(f"\n  Bits different (pos 0 vs 1): {diff_01}")
print(f"  Bits different (pos 1 vs 2): {diff_12}")


# =============================================================
# Test 3: Sinusoidal Position Encoding
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Sinusoidal Position Encoding")
print("=" * 60)

embed_sin = RAMEmbedding(
	token_bits=bits_per_token,
	embedding_bits=bits_per_token,
	max_seq_len=32,
	position_encoding=PositionEncoding.SINUSOIDAL,
	rng=42,
)

print(f"\nEmbedding: {embed_sin}")

# Show position patterns
tok_a = encode_char('A')
print("\nPosition encoding patterns for token 'A':")
for pos in range(6):
	emb = embed_sin.forward_single(tok_a, position=pos)
	print(f"  Position {pos}: {emb.tolist()}")


# =============================================================
# Test 4: Training Embeddings
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Training Token Embeddings")
print("=" * 60)

embed_train = RAMEmbedding(
	token_bits=bits_per_token,
	embedding_bits=bits_per_token,
	max_seq_len=16,
	position_encoding=PositionEncoding.NONE,
	rng=42,
)

# Train specific embeddings
print("\nTraining custom embeddings:")
tok_a = encode_char('A')
target_embed = zeros(bits_per_token, dtype=uint8)
target_embed[0] = 1  # Custom pattern: 10000

print(f"  Training 'A' -> {target_embed.tolist()}")
updated = embed_train.train_embedding(tok_a, target_embed)
print(f"  Updated: {updated}")

# Verify training worked
result = embed_train.forward_single(tok_a, add_position=False)
matches = (result == target_embed).all().item()
print(f"  Embedding matches target: {'✓' if matches else '✗'}")


# =============================================================
# Test 5: RAMSeq2Seq with Embeddings
# =============================================================
print("\n" + "=" * 60)
print("Test 5: RAMSeq2Seq with Learned Embeddings")
print("=" * 60)

model_with_embed = RAMSeq2Seq(
	input_bits=bits_per_token,
	hidden_bits=bits_per_token * 2,  # Embedding expands to 10 bits
	output_bits=bits_per_token,      # Output back to 5 bits
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_embedding=True,
	embedding_position=PositionEncoding.BINARY,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

print(f"\nModel: {model_with_embed}")
print(f"  Has embedding: {model_with_embed.embedding is not None}")

# Test forward pass
test_seq = encode_sequence("ABC")
outputs = model_with_embed.forward(test_seq)
output_str = decode_sequence(outputs)

print(f"\nForward pass test:")
print(f"  Input: 'ABC'")
print(f"  Output: '{output_str}'")


# =============================================================
# Test 6: Comparing with/without Embeddings
# =============================================================
print("\n" + "=" * 60)
print("Test 6: Comparing Models With/Without Embeddings")
print("=" * 60)

# Model without embedding (direct projection)
model_no_embed = RAMSeq2Seq(
	input_bits=bits_per_token,
	hidden_bits=bits_per_token * 2,
	output_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_embedding=False,  # No embedding
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

# Model with embedding
model_embed = RAMSeq2Seq(
	input_bits=bits_per_token,
	hidden_bits=bits_per_token * 2,
	output_bits=bits_per_token,
	num_layers=1,
	num_heads=2,
	position_mode=PositionMode.RELATIVE,
	max_seq_len=16,
	use_residual=True,
	use_embedding=True,
	embedding_position=PositionEncoding.BINARY,
	generalization=MapperStrategy.BIT_LEVEL,
	rng=42,
)

print(f"\nModel without embedding: {model_no_embed}")
print(f"Model with embedding: {model_embed}")

# Test on same sequence
test_seq = encode_sequence("HELLO")
out_no_embed = decode_sequence(model_no_embed.forward(test_seq))
out_embed = decode_sequence(model_embed.forward(test_seq))

print(f"\nInput: 'HELLO'")
print(f"  Without embedding: '{out_no_embed}'")
print(f"  With embedding: '{out_embed}'")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: RAM Embeddings")
print("=" * 60)

print("""
RAMEmbedding Architecture:

  Token bits ──────▶│ Embedding Projection │──────▶│ Position Encoding │──▶ Embedding
                    │  RAMLayer lookup     │       │ XOR with pos bits │

Position Encoding Options:
  NONE       - No position information (useful for order-invariant tasks)
  BINARY     - XOR with binary position representation
  LEARNED    - RAMLayer that maps position to embedding
  SINUSOIDAL - Discrete approximation of transformer sinusoidal encoding

Key Insights:
- RAM embeddings use discrete lookups instead of continuous matrices
- Position encoding via XOR is reversible (can recover token embedding)
- Learned embeddings can capture task-specific token representations
- Embedding expansion (token_bits < embedding_bits) increases capacity
""")

print("=" * 60)
print("All embedding tests completed!")
print("=" * 60)
