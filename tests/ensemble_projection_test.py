#!/usr/bin/env python3
"""
Ensemble with Diverse Projections Test

Hypothesis: Multiple RAM neurons with different random bit projections
can generalize better than a single RAM, because an unseen input might
be "familiar" to some projections even if new to others.

Key insight: Different projections create different "views" of the input.
Majority voting across views can recover the correct answer.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode
from torch import zeros, uint8, Tensor, randperm, manual_seed
from torch.nn import Module, ModuleList
import random

print("=" * 70)
print("Ensemble with Diverse Projections")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str) -> Tensor:
    return decoder.encode(c).squeeze()

def decode_bits(bits: Tensor) -> str:
    return decoder.decode(bits.unsqueeze(0))


# =============================================================
# Ensemble RAM with Diverse Projections
# =============================================================
class EnsembleRAM(Module):
    """
    Ensemble of RAM neurons with different random projections.

    Each RAM sees a different subset of input bits.
    Voting across RAMs provides probabilistic generalization.
    """

    def __init__(
        self,
        input_bits: int,
        output_bits: int,
        num_rams: int = 8,
        bits_per_ram: int | None = None,
        rng: int | None = None,
    ):
        super().__init__()

        self.input_bits = input_bits
        self.output_bits = output_bits
        self.num_rams = num_rams

        # Default: each RAM sees half the input bits
        if bits_per_ram is None:
            bits_per_ram = max(4, input_bits // 2)
        self.bits_per_ram = min(bits_per_ram, input_bits)

        # Generate diverse random projections (which bits each RAM sees)
        if rng is not None:
            manual_seed(rng)
            random.seed(rng)

        self.projections = []
        for i in range(num_rams):
            # Random subset of input bits for this RAM
            perm = randperm(input_bits)[:self.bits_per_ram].sort().values
            self.projections.append(perm)

        # Create RAM layers - each sees its projected subset
        self.rams = ModuleList([
            RAMLayer(
                total_input_bits=self.bits_per_ram,
                num_neurons=output_bits,
                n_bits_per_neuron=min(self.bits_per_ram, 8),
                rng=rng + i if rng else None,
            )
            for i in range(num_rams)
        ])

        print(f"[EnsembleRAM] {num_rams} RAMs, each sees {self.bits_per_ram}/{input_bits} bits")
        for i, proj in enumerate(self.projections[:3]):
            print(f"  RAM{i} sees bits: {proj.tolist()}")
        if num_rams > 3:
            print(f"  ... ({num_rams - 3} more)")

    def _project(self, x: Tensor, projection: Tensor) -> Tensor:
        """Extract bits specified by projection."""
        x = x.squeeze()
        return x[projection]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with majority voting.

        Each RAM produces an output, then we vote per bit.
        """
        x = x.squeeze()

        # Collect outputs from all RAMs
        outputs = []
        for ram, proj in zip(self.rams, self.projections):
            projected = self._project(x, proj).unsqueeze(0)
            out = ram(projected).squeeze()
            outputs.append(out)

        # Majority voting per output bit
        result = zeros(self.output_bits, dtype=uint8)
        for bit in range(self.output_bits):
            if self.output_bits == 1:
                # Handle 0-dim tensor case
                ones = sum(out.item() for out in outputs)
            else:
                ones = sum(out[bit].item() for out in outputs)
            result[bit] = 1 if ones > self.num_rams // 2 else 0

        return result

    def train_mapping(self, x: Tensor, y: Tensor) -> int:
        """Train all RAMs on the same (input, output) pair."""
        x = x.squeeze()
        y = y.squeeze()
        corrections = 0

        for ram, proj in zip(self.rams, self.projections):
            projected = self._project(x, proj).unsqueeze(0)
            current = ram(projected).squeeze()

            if not (current == y).all():
                corrections += 1
                ram.commit(projected, y.unsqueeze(0))

        return corrections


# =============================================================
# Single RAM baseline (for comparison)
# =============================================================
class SingleRAM(Module):
    """Single RAM for comparison."""

    def __init__(
        self,
        input_bits: int,
        output_bits: int,
        rng: int | None = None,
    ):
        super().__init__()

        self.ram = RAMLayer(
            total_input_bits=input_bits,
            num_neurons=output_bits,
            n_bits_per_neuron=min(input_bits, 8),
            rng=rng,
        )

        print(f"[SingleRAM] sees all {input_bits} bits")

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze().unsqueeze(0)
        return self.ram(x).squeeze()

    def train_mapping(self, x: Tensor, y: Tensor) -> int:
        x = x.squeeze().unsqueeze(0)
        y = y.squeeze().unsqueeze(0)
        current = self.ram(x).squeeze()
        if not (current == y.squeeze()).all():
            self.ram.commit(x, y)
            return 1
        return 0


# =============================================================
# Test 1: Identity Mapping with Partial Training
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Identity Mapping (train on A-L, test on M-Z)")
print("=" * 60)

# Train on first half of alphabet, test on second half
train_chars = "ABCDEFGHIJKL"
test_chars = "MNOPQRSTUVWXYZ"

for num_rams in [1, 4, 8, 16]:
    if num_rams == 1:
        model = SingleRAM(
            input_bits=bits_per_token,
            output_bits=bits_per_token,
            rng=42,
        )
        name = "Single RAM"
    else:
        model = EnsembleRAM(
            input_bits=bits_per_token,
            output_bits=bits_per_token,
            num_rams=num_rams,
            bits_per_ram=3,  # Each RAM sees 3 of 5 bits
            rng=42,
        )
        name = f"Ensemble ({num_rams} RAMs)"

    # Train
    for c in train_chars:
        token = encode_char(c)
        model.train_mapping(token, token)  # Identity: output = input

    # Test on trained data
    train_correct = 0
    for c in train_chars:
        token = encode_char(c)
        output = model.forward(token)
        if decode_bits(output) == c:
            train_correct += 1

    # Test on unseen data
    test_correct = 0
    for c in test_chars:
        token = encode_char(c)
        output = model.forward(token)
        if decode_bits(output) == c:
            test_correct += 1

    train_pct = 100 * train_correct / len(train_chars)
    test_pct = 100 * test_correct / len(test_chars)

    print(f"\n{name}:")
    print(f"  Trained chars (A-L): {train_pct:.0f}%")
    print(f"  Unseen chars (M-Z):  {test_pct:.0f}%")


# =============================================================
# Test 2: Increment Mapping
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Increment Mapping (A→B, B→C, ...)")
print("=" * 60)

train_pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("E", "F"), ("G", "H")]
test_pairs = [("D", "E"), ("F", "G"), ("H", "I"), ("M", "N"), ("X", "Y")]

for num_rams in [1, 8, 16]:
    if num_rams == 1:
        model = SingleRAM(
            input_bits=bits_per_token,
            output_bits=bits_per_token,
            rng=42,
        )
        name = "Single RAM"
    else:
        model = EnsembleRAM(
            input_bits=bits_per_token,
            output_bits=bits_per_token,
            num_rams=num_rams,
            bits_per_ram=3,
            rng=42,
        )
        name = f"Ensemble ({num_rams} RAMs)"

    # Train
    for inp, out in train_pairs:
        model.train_mapping(encode_char(inp), encode_char(out))

    # Test
    print(f"\n{name}:")
    for inp, expected in test_pairs:
        output = model.forward(encode_char(inp))
        result = decode_bits(output)
        status = "✓" if result == expected else "✗"
        trained = "(trained)" if (inp, expected) in train_pairs else "(unseen)"
        print(f"  {status} {inp} → {result} (want {expected}) {trained}")


# =============================================================
# Test 3: Varying Projection Size
# =============================================================
print("\n" + "=" * 60)
print("Test 3: Effect of Bits per RAM")
print("=" * 60)
print("More bits = more specific, fewer collisions but less coverage")

train_chars = "ABCDEFGH"
test_chars = "IJKLMNOP"

for bits_per_ram in [2, 3, 4, 5]:
    model = EnsembleRAM(
        input_bits=bits_per_token,
        output_bits=bits_per_token,
        num_rams=8,
        bits_per_ram=bits_per_ram,
        rng=42,
    )

    # Train
    for c in train_chars:
        token = encode_char(c)
        model.train_mapping(token, token)

    # Test
    test_correct = 0
    for c in test_chars:
        token = encode_char(c)
        output = model.forward(token)
        if decode_bits(output) == c:
            test_correct += 1

    test_pct = 100 * test_correct / len(test_chars)
    print(f"  {bits_per_ram} bits/RAM: {test_pct:.0f}% on unseen")


# =============================================================
# Test 4: Coverage Analysis
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Coverage Analysis")
print("=" * 60)
print("How many RAMs 'recognize' each input pattern?")

model = EnsembleRAM(
    input_bits=bits_per_token,
    output_bits=bits_per_token,
    num_rams=8,
    bits_per_ram=3,
    rng=42,
)

# Train on A-D
for c in "ABCD":
    token = encode_char(c)
    model.train_mapping(token, token)

print("\nAfter training on A, B, C, D:")
print("Testing which RAMs 'recognize' each pattern...")

for c in "ABCDEFGH":
    token = encode_char(c)

    # Count how many RAMs have seen this projected pattern
    recognizing = 0
    for ram, proj in zip(model.rams, model.projections):
        projected = model._project(token, proj).unsqueeze(0)
        # Check if output matches expected (trained on identity)
        output = ram(projected).squeeze()
        if (output == token).all():
            recognizing += 1

    trained = "trained" if c in "ABCD" else "unseen"
    print(f"  {c} ({trained}): {recognizing}/8 RAMs recognize it")


# =============================================================
# Test 5: Attention-like Task
# =============================================================
print("\n" + "=" * 60)
print("Test 5: Simulated Attention (position-based)")
print("=" * 60)
print("Task: Given (query_pos, key_pos), output 1 if should attend")

# Create attention pattern: attend if key_pos <= query_pos (causal)
# Input: 6 bits (3 for query_pos, 3 for key_pos)

def pos_to_bits(pos: int, n_bits: int = 3) -> Tensor:
    bits = zeros(n_bits, dtype=uint8)
    for i in range(n_bits):
        bits[n_bits - 1 - i] = (pos >> i) & 1
    return bits

def make_attention_input(q_pos: int, k_pos: int) -> Tensor:
    return torch.cat([pos_to_bits(q_pos), pos_to_bits(k_pos)])

import torch

# Train on some (q, k) pairs
train_examples = [
    (0, 0, 1),  # q=0, k=0: attend (self)
    (1, 0, 1),  # q=1, k=0: attend (causal)
    (1, 1, 1),  # q=1, k=1: attend (self)
    (2, 0, 1),  # q=2, k=0: attend
    (2, 1, 1),  # q=2, k=1: attend
    (2, 2, 1),  # q=2, k=2: attend
    (0, 1, 0),  # q=0, k=1: don't attend (future)
    (1, 2, 0),  # q=1, k=2: don't attend
]

for num_rams in [1, 8, 16]:
    if num_rams == 1:
        model = SingleRAM(input_bits=6, output_bits=1, rng=42)
        name = "Single"
    else:
        model = EnsembleRAM(input_bits=6, output_bits=1, num_rams=num_rams, bits_per_ram=4, rng=42)
        name = f"Ensemble({num_rams})"

    # Train
    for q, k, attend in train_examples:
        inp = make_attention_input(q, k)
        out = torch.tensor([attend], dtype=torch.uint8)
        model.train_mapping(inp, out)

    # Test on all positions 0-3
    correct = 0
    total = 0

    for q in range(4):
        for k in range(4):
            inp = make_attention_input(q, k)
            out = model.forward(inp)
            expected = 1 if k <= q else 0

            if out.item() == expected:
                correct += 1
            total += 1

    pct = 100 * correct / total
    print(f"  {name}: {correct}/{total} ({pct:.0f}%) causal attention patterns correct")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Ensemble with Diverse Projections")
print("=" * 60)

print("""
KEY FINDINGS:

1. ENSEMBLE MECHANISM:
   - Each RAM sees different subset of input bits
   - Majority voting combines their outputs
   - Provides probabilistic coverage of unseen patterns

2. TRADE-OFFS:
   - More RAMs = better coverage, more memory
   - Fewer bits/RAM = more collisions, better generalization
   - More bits/RAM = fewer collisions, less generalization

3. WHEN IT HELPS:
   - When unseen patterns share bit-subsets with trained patterns
   - Depends on data distribution and projection diversity

4. LIMITATIONS:
   - Still not true interpolation
   - Collisions can cause wrong outputs
   - Works best when input space has structure

5. COMPARISON TO NEURAL NETWORKS:
   - NN: Learns smooth functions, interpolates
   - Ensemble RAM: Probabilistic lookup, relies on coverage
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
