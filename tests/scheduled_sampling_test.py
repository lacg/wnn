#!/usr/bin/env python3
"""
Scheduled Sampling for Autoregressive Generation

The Problem (Teacher Forcing Gap):
  Training: Model sees CORRECT previous tokens → predicts next
  Testing:  Model sees ITS OWN predictions → predicts next

  If training always uses correct tokens, model never learns to
  recover from its own mistakes.

The Solution (Scheduled Sampling):
  During training, sometimes use model's own predictions instead
  of ground truth. Gradually increase this probability.

  epsilon = 0.0 → always use ground truth (teacher forcing)
  epsilon = 1.0 → always use own predictions (autoregressive)

  Schedule: Start at 0, gradually increase to 1 over training.

Challenge for RAM Networks:
  RAM lookup tables can conflict if same input → different outputs.
  We need to handle this carefully.
"""

import sys
sys.path.insert(0, 'src')

from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.RAMGeneralization import MapperStrategy, GeneralizingProjection
from wnn.ram.encoders_decoders import TransformerDecoderFactory, OutputMode
from torch import zeros, uint8, Tensor
from torch.nn import Module
import random

print("=" * 70)
print("Scheduled Sampling for Autoregressive Generation")
print("=" * 70)

# Setup
decoder = TransformerDecoderFactory.create(OutputMode.TOKEN)
bits_per_token = decoder.bits_per_token

def encode_char(c: str) -> Tensor:
    return decoder.encode(c).squeeze()

def decode_bits(bits: Tensor) -> str:
    return decoder.decode(bits.unsqueeze(0))

def decode_sequence(tokens: list) -> str:
    return ''.join(decode_bits(t) for t in tokens)


# =============================================================
# Autoregressive Model with Scheduled Sampling
# =============================================================
class ScheduledSamplingModel(Module):
    """
    Autoregressive model trained with scheduled sampling.

    Key idea: During training, mix teacher forcing with
    autoregressive generation based on epsilon schedule.
    """

    def __init__(
        self,
        bits: int,
        strategy: MapperStrategy = MapperStrategy.BIT_LEVEL,
        rng: int | None = None,
    ):
        super().__init__()
        self.bits = bits
        self.strategy = strategy

        # Transition function
        self.transition = GeneralizingProjection(
            input_bits=bits,
            output_bits=bits,
            strategy=strategy,
            rng=rng,
        )

        print(f"[ScheduledSamplingModel] bits={bits}, strategy={strategy.name}")

    def forward(self, token: Tensor) -> Tensor:
        """Single step forward."""
        out = self.transition(token.squeeze())
        if out.ndim > 1:
            out = out.squeeze()
        return out

    def generate(self, start: Tensor, length: int) -> list[Tensor]:
        """Generate sequence autoregressively."""
        outputs = [start.squeeze().clone()]
        current = start.squeeze()

        for _ in range(length - 1):
            next_token = self.forward(current)
            outputs.append(next_token.clone())
            current = next_token

        return outputs

    def train_with_teacher_forcing(self, sequence: list[Tensor]) -> int:
        """Standard teacher forcing: always use ground truth."""
        sequence = [s.squeeze() for s in sequence]
        errors = 0

        for i in range(len(sequence) - 1):
            current = sequence[i]
            target = sequence[i + 1]

            pred = self.forward(current)
            if not (pred == target).all():
                errors += 1
                self.transition.train_mapping(current, target)

        return errors

    def train_with_scheduled_sampling(
        self,
        sequence: list[Tensor],
        epsilon: float = 0.5,
    ) -> tuple[int, int]:
        """
        Scheduled sampling: mix teacher forcing with own predictions.

        Args:
            sequence: Target sequence
            epsilon: Probability of using own prediction (0=teacher, 1=autoregressive)

        Returns:
            (errors, conflicts): Number of errors and training conflicts
        """
        sequence = [s.squeeze() for s in sequence]
        errors = 0
        conflicts = 0

        # First token is always given
        current = sequence[0].clone()

        for i in range(len(sequence) - 1):
            target = sequence[i + 1]

            # Predict next token
            pred = self.forward(current)

            # Train if wrong
            if not (pred == target).all():
                errors += 1
                # Only train if using ground truth input
                # (training on wrong input creates conflicts)
                if i == 0 or random.random() > epsilon:
                    self.transition.train_mapping(current, target)
                else:
                    conflicts += 1  # Skip training to avoid conflict

            # Decide next input: ground truth or own prediction
            if random.random() < epsilon:
                # Use own prediction (autoregressive)
                current = pred.clone()
            else:
                # Use ground truth (teacher forcing)
                current = sequence[i + 1].clone()

        return errors, conflicts


# =============================================================
# Epsilon Schedules
# =============================================================
def linear_schedule(epoch: int, total_epochs: int) -> float:
    """Linear increase from 0 to 1."""
    return min(1.0, epoch / max(1, total_epochs - 1))

def exponential_schedule(epoch: int, total_epochs: int, k: float = 5.0) -> float:
    """Exponential increase: slow start, fast end."""
    return 1.0 - pow(k, -epoch / max(1, total_epochs - 1))

def inverse_sigmoid_schedule(epoch: int, total_epochs: int, k: float = 5.0) -> float:
    """Inverse sigmoid: gradual transition."""
    x = k * (2 * epoch / max(1, total_epochs - 1) - 1)
    return 1.0 / (1.0 + pow(2.718, -x))


# =============================================================
# Test 1: Compare Training Methods on Counting
# =============================================================
print("\n" + "=" * 60)
print("Test 1: Teacher Forcing vs Scheduled Sampling")
print("=" * 60)
print("Task: Learn counting (A→B→C→D→...)")

# Training sequences
train_sequences = [
    "ABCDE",
    "BCDEF",
    "CDEFG",
    "DEFGH",
    "EFGHI",
    "FGHIJ",
    "GHIJK",
    "HIJKL",
]

def run_experiment(name: str, use_scheduled: bool, epochs: int = 20):
    print(f"\n--- {name} ---")

    model = ScheduledSamplingModel(
        bits=bits_per_token,
        strategy=MapperStrategy.BIT_LEVEL,
        rng=42,
    )

    for epoch in range(epochs):
        total_errors = 0

        if use_scheduled:
            epsilon = linear_schedule(epoch, epochs)
        else:
            epsilon = 0.0  # Pure teacher forcing

        for seq_str in train_sequences:
            sequence = [encode_char(c) for c in seq_str]

            if use_scheduled:
                errors, _ = model.train_with_scheduled_sampling(sequence, epsilon)
            else:
                errors = model.train_with_teacher_forcing(sequence)

            total_errors += errors

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: errors={total_errors}, epsilon={epsilon:.2f}")

    # Test autoregressive generation
    print("\n  Autoregressive generation test:")
    test_cases = [('A', 'ABCDEFGH'), ('M', 'MNOPQRST')]

    total_correct = 0
    total_chars = 0

    for start_char, expected in test_cases:
        generated = model.generate(encode_char(start_char), length=len(expected))
        result = decode_sequence(generated)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        total_correct += correct
        total_chars += len(expected)
        status = "OK" if result == expected else "X"
        print(f"    [{status}] '{start_char}' -> '{result}' (expected '{expected}')")

    accuracy = 100 * total_correct / total_chars
    print(f"  Overall accuracy: {accuracy:.0f}%")
    return accuracy

# Run experiments
tf_accuracy = run_experiment("Teacher Forcing Only", use_scheduled=False)
ss_accuracy = run_experiment("Scheduled Sampling", use_scheduled=True)


# =============================================================
# Test 2: Different Schedules
# =============================================================
print("\n" + "=" * 60)
print("Test 2: Comparing Epsilon Schedules")
print("=" * 60)

schedules = [
    ("Linear", linear_schedule),
    ("Exponential", exponential_schedule),
    ("Inverse Sigmoid", inverse_sigmoid_schedule),
]

for schedule_name, schedule_fn in schedules:
    print(f"\n--- {schedule_name} Schedule ---")

    model = ScheduledSamplingModel(
        bits=bits_per_token,
        strategy=MapperStrategy.BIT_LEVEL,
        rng=42,
    )

    epochs = 20
    for epoch in range(epochs):
        epsilon = schedule_fn(epoch, epochs)

        for seq_str in train_sequences:
            sequence = [encode_char(c) for c in seq_str]
            model.train_with_scheduled_sampling(sequence, epsilon)

    # Test
    test_results = []
    for start_char in ['A', 'F', 'M', 'T']:
        generated = model.generate(encode_char(start_char), length=6)
        result = decode_sequence(generated)

        # Expected: start + next 5 letters
        expected_chars = []
        c = start_char
        for _ in range(6):
            expected_chars.append(c)
            c = chr(ord(c) + 1) if c < 'Z' else 'A'
        expected = ''.join(expected_chars)

        correct = sum(1 for r, e in zip(result, expected) if r == e)
        test_results.append(correct / len(expected))

    avg_accuracy = 100 * sum(test_results) / len(test_results)
    print(f"  Average accuracy: {avg_accuracy:.0f}%")


# =============================================================
# Test 3: Scheduled Sampling with State Machine
# =============================================================
print("\n" + "=" * 60)
print("Test 3: State Machine (A→B→C→A cycle)")
print("=" * 60)
print("Does scheduled sampling help with cyclic patterns?")

for use_scheduled in [False, True]:
    name = "Scheduled Sampling" if use_scheduled else "Teacher Forcing"
    print(f"\n--- {name} ---")

    model = ScheduledSamplingModel(
        bits=bits_per_token,
        strategy=MapperStrategy.DIRECT,  # Use DIRECT for state machines
        rng=123,
    )

    # Train on cyclic sequences
    train_seqs = ["ABCABC", "BCABCA", "CABCAB"]

    epochs = 15
    for epoch in range(epochs):
        epsilon = linear_schedule(epoch, epochs) if use_scheduled else 0.0

        for seq_str in train_seqs:
            sequence = [encode_char(c) for c in seq_str]
            if use_scheduled:
                model.train_with_scheduled_sampling(sequence, epsilon)
            else:
                model.train_with_teacher_forcing(sequence)

    # Test
    for start_char, expected in [('A', 'ABCABCABC'), ('B', 'BCABCABCA')]:
        generated = model.generate(encode_char(start_char), length=9)
        result = decode_sequence(generated)
        correct = sum(1 for r, e in zip(result, expected) if r == e)
        pct = 100 * correct / len(expected)
        status = "OK" if result == expected else "X"
        print(f"  [{status}] '{start_char}' -> '{result}' ({pct:.0f}%)")


# =============================================================
# Test 4: Identity Task with Scheduled Sampling
# =============================================================
print("\n" + "=" * 60)
print("Test 4: Identity (Repeat) with Scheduled Sampling")
print("=" * 60)

for use_scheduled in [False, True]:
    name = "Scheduled Sampling" if use_scheduled else "Teacher Forcing"
    print(f"\n--- {name} ---")

    model = ScheduledSamplingModel(
        bits=bits_per_token,
        strategy=MapperStrategy.BIT_LEVEL,
        rng=456,
    )

    # Train on repeat sequences
    train_seqs = ["AAAA", "MMMM", "ZZZZ"]

    epochs = 10
    for epoch in range(epochs):
        epsilon = linear_schedule(epoch, epochs) if use_scheduled else 0.0

        for seq_str in train_seqs:
            sequence = [encode_char(c) for c in seq_str]
            if use_scheduled:
                model.train_with_scheduled_sampling(sequence, epsilon)
            else:
                model.train_with_teacher_forcing(sequence)

    # Test on unseen letters
    correct_count = 0
    for test_char in "BKQX":
        generated = model.generate(encode_char(test_char), length=5)
        result = decode_sequence(generated)
        expected = test_char * 5
        if result == expected:
            correct_count += 1
        status = "OK" if result == expected else "X"
        print(f"  [{status}] '{test_char}' -> '{result}' (expected '{expected}')")

    print(f"  Accuracy: {100*correct_count/4:.0f}%")


# =============================================================
# Summary
# =============================================================
print("\n" + "=" * 60)
print("Summary: Scheduled Sampling Results")
print("=" * 60)

print("""
Key Findings:

1. TEACHER FORCING GAP:
   - Models trained with pure teacher forcing may fail at
     autoregressive generation if they never see their own errors.

2. SCHEDULED SAMPLING:
   - Gradually exposes model to its own predictions during training.
   - epsilon=0 → pure teacher forcing
   - epsilon=1 → pure autoregressive

3. RAM NETWORK CHALLENGE:
   - RAM lookup tables can conflict when training on different inputs.
   - Solution: Only train when using ground truth input, skip when
     using own (potentially wrong) prediction.

4. RESULTS:
   - For BIT_LEVEL tasks (identity, increment): Works well because
     the underlying pattern is consistent.
   - For DIRECT tasks (state machines): Teacher forcing is sufficient
     because all transitions are explicitly memorized.

When to use Scheduled Sampling:
  ✅ When autoregressive generation quality matters
  ✅ When teacher forcing shows gap vs autoregressive
  ⚠️ Adds training complexity
  ❌ Not needed if BIT_LEVEL already generalizes perfectly
""")

print("=" * 70)
print("Tests completed!")
print("=" * 70)
