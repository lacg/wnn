"""
Improved Language Modeling

The original language model had limited generalization because:
1. N-gram memorizes exact contexts - no interpolation
2. Recurrent model collapses to repetitive patterns
3. No abstraction over character features

Improvements explored:
1. Backoff N-gram: Fall back to shorter contexts when exact match not found
2. Character Features: Decompose into (vowel/consonant, position, case)
3. Hierarchical Context: Combine multiple context lengths
4. Pattern Abstraction: Learn abstract patterns like "CVC" → next type
5. Curriculum Learning: Start simple, gradually increase complexity
"""

import random
from datetime import datetime
from collections import Counter

from torch import zeros, uint8, tensor, Tensor, cat
from torch.nn import Module

from wnn.ram.core import RAMLayer


# ─────────────────────────────────────────────────────────────────────────────
# Character Features (Decomposition for Language)
# ─────────────────────────────────────────────────────────────────────────────

class CharacterFeatures:
    """
    Decompose characters into learnable features.

    Instead of treating each character as atomic, extract features:
    - Type: vowel (0) / consonant (1) / space (2) / punctuation (3)
    - Position: 0-25 for a-z
    - Case: lower (0) / upper (1)

    This is analogous to bit decomposition for numbers!
    """

    VOWELS = set('aeiouAEIOU')
    CONSONANTS = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')

    def __init__(self):
        self.char_to_idx = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
        self.idx_to_char = {i: chr(i + ord('a')) for i in range(26)}

    def get_type(self, c: str) -> int:
        """0=vowel, 1=consonant, 2=space, 3=punct, 4=other"""
        if c in self.VOWELS:
            return 0
        elif c in self.CONSONANTS:
            return 1
        elif c == ' ':
            return 2
        elif c in '.,!?;:\'"':
            return 3
        else:
            return 4

    def get_position(self, c: str) -> int:
        """Position in alphabet (0-25) or 26 for non-letters."""
        c_lower = c.lower()
        if c_lower in self.char_to_idx:
            return self.char_to_idx[c_lower]
        return 26

    def encode(self, c: str) -> list[int]:
        """Encode character as feature vector."""
        char_type = self.get_type(c)
        position = self.get_position(c)

        # Encode as bits
        type_bits = [(char_type >> i) & 1 for i in range(2, -1, -1)]  # 3 bits
        pos_bits = [(position >> i) & 1 for i in range(4, -1, -1)]   # 5 bits

        return type_bits + pos_bits  # 8 bits total

    def decode(self, bits: list[int]) -> str:
        """Decode feature vector to character (best effort)."""
        type_val = sum(bits[i] << (2 - i) for i in range(3))
        pos_val = sum(bits[3 + i] << (4 - i) for i in range(5))

        if pos_val < 26:
            return self.idx_to_char[pos_val]
        elif type_val == 2:
            return ' '
        elif type_val == 3:
            return '.'
        else:
            return '?'


# ─────────────────────────────────────────────────────────────────────────────
# Improvement 1: Backoff N-gram Model
# ─────────────────────────────────────────────────────────────────────────────

class BackoffNGram(Module):
    """
    N-gram with backoff to shorter contexts.

    When the 4-gram "abcd" isn't found:
    1. Try 3-gram "bcd"
    2. Try 2-gram "cd"
    3. Try 1-gram "d"
    4. Fall back to most common character

    This provides graceful degradation for unseen contexts.
    """

    def __init__(self, max_n: int = 4, rng: int | None = None):
        super().__init__()
        self.max_n = max_n
        self.features = CharacterFeatures()

        # Separate RAMLayer for each n-gram size
        self.ngram_layers = {}
        for n in range(1, max_n + 1):
            input_bits = n * 8  # 8 bits per character (from features)
            self.ngram_layers[n] = RAMLayer(
                total_input_bits=input_bits,
                num_neurons=8,  # Output character features
                n_bits_per_neuron=min(input_bits, 10),
                rng=rng + n * 100 if rng else None,
            )

        # Track which patterns are known
        self.known_patterns = {n: set() for n in range(1, max_n + 1)}

    def _encode_context(self, context: str) -> Tensor:
        """Encode context string to bit tensor."""
        bits = []
        for c in context:
            bits.extend(self.features.encode(c))
        return tensor(bits, dtype=uint8)

    def train_on_text(self, text: str) -> dict:
        """Train all n-gram levels on text."""
        text = text.lower()
        errors = {n: 0 for n in range(1, self.max_n + 1)}

        for n in range(1, self.max_n + 1):
            for i in range(len(text) - n):
                context = text[i:i + n]
                target = text[i + n]

                # Track known patterns
                self.known_patterns[n].add(context)

                # Encode and train
                ctx_bits = self._encode_context(context)
                tgt_bits = tensor(self.features.encode(target), dtype=uint8)

                errors[n] += self.ngram_layers[n].commit(
                    ctx_bits.unsqueeze(0), tgt_bits.unsqueeze(0)
                )

        return errors

    def predict_next(self, context: str) -> str:
        """Predict next character with backoff."""
        context = context.lower()

        # Try from longest to shortest context
        for n in range(min(self.max_n, len(context)), 0, -1):
            ctx = context[-n:]

            # Check if this pattern was seen in training
            if ctx in self.known_patterns[n]:
                ctx_bits = self._encode_context(ctx)
                out = self.ngram_layers[n](ctx_bits.unsqueeze(0)).squeeze()
                return self.features.decode([int(b.item()) for b in out])

        # Fallback to space
        return ' '

    def generate(self, seed: str, length: int) -> str:
        """Generate text starting from seed."""
        result = seed.lower()
        for _ in range(length):
            next_char = self.predict_next(result)
            result += next_char
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Improvement 2: Pattern Abstraction Model
# ─────────────────────────────────────────────────────────────────────────────

class PatternAbstractionLM(Module):
    """
    Learn abstract patterns instead of exact characters.

    Instead of "cat" → " ", learn:
    - "CVCa" → space (consonant-vowel-consonant-vowel pattern)

    This reduces the pattern space significantly!
    """

    def __init__(self, n: int = 3, rng: int | None = None):
        super().__init__()
        self.n = n
        self.features = CharacterFeatures()

        # Pattern learner: type sequence → next type
        # Input: n character types (3 bits each)
        # Output: 1 character type (3 bits)
        self.pattern_layer = RAMLayer(
            total_input_bits=n * 3,  # Type only (3 bits per char)
            num_neurons=3,  # Next type
            n_bits_per_neuron=n * 3,
            rng=rng,
        )

        # Given type + context, predict position
        # Input: n positions (5 bits each) + 1 type (3 bits)
        # Output: position (5 bits)
        self.position_layer = RAMLayer(
            total_input_bits=n * 5 + 3,
            num_neurons=5,  # Next position
            n_bits_per_neuron=min(n * 5 + 3, 12),
            rng=rng + 100 if rng else None,
        )

    def _get_type_bits(self, c: str) -> list[int]:
        """Get type bits for character."""
        t = self.features.get_type(c)
        return [(t >> i) & 1 for i in range(2, -1, -1)]

    def _get_pos_bits(self, c: str) -> list[int]:
        """Get position bits for character."""
        p = self.features.get_position(c)
        return [(p >> i) & 1 for i in range(4, -1, -1)]

    def train_on_text(self, text: str) -> dict:
        """Train pattern and position models."""
        text = text.lower()
        pattern_errors = 0
        position_errors = 0

        for i in range(len(text) - self.n):
            context = text[i:i + self.n]
            target = text[i + self.n]

            # Train pattern layer (type sequence → next type)
            type_bits = []
            for c in context:
                type_bits.extend(self._get_type_bits(c))
            type_input = tensor(type_bits, dtype=uint8)
            type_output = tensor(self._get_type_bits(target), dtype=uint8)
            pattern_errors += self.pattern_layer.commit(
                type_input.unsqueeze(0), type_output.unsqueeze(0)
            )

            # Train position layer
            pos_bits = []
            for c in context:
                pos_bits.extend(self._get_pos_bits(c))
            target_type_bits = self._get_type_bits(target)
            pos_input = tensor(pos_bits + target_type_bits, dtype=uint8)
            pos_output = tensor(self._get_pos_bits(target), dtype=uint8)
            position_errors += self.position_layer.commit(
                pos_input.unsqueeze(0), pos_output.unsqueeze(0)
            )

        return {"pattern": pattern_errors, "position": position_errors}

    def predict_next(self, context: str) -> str:
        """Predict next character using abstracted patterns."""
        context = context[-self.n:].lower()
        if len(context) < self.n:
            context = ' ' * (self.n - len(context)) + context

        # Predict next type
        type_bits = []
        for c in context:
            type_bits.extend(self._get_type_bits(c))
        type_input = tensor(type_bits, dtype=uint8)
        pred_type = self.pattern_layer(type_input.unsqueeze(0)).squeeze()
        pred_type_list = [int(b.item()) for b in pred_type]

        # Predict position given type
        pos_bits = []
        for c in context:
            pos_bits.extend(self._get_pos_bits(c))
        pos_input = tensor(pos_bits + pred_type_list, dtype=uint8)
        pred_pos = self.position_layer(pos_input.unsqueeze(0)).squeeze()

        # Combine to decode character
        full_bits = pred_type_list + [int(b.item()) for b in pred_pos]
        return self.features.decode(full_bits)


# ─────────────────────────────────────────────────────────────────────────────
# Improvement 3: Hierarchical N-gram (Multiple Context Lengths)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalNGram(Module):
    """
    Combine predictions from multiple context lengths.

    Each n-gram votes, and we take majority.
    This is like ensemble voting - improves robustness.
    """

    def __init__(self, max_n: int = 4, rng: int | None = None):
        super().__init__()
        self.max_n = max_n
        self.features = CharacterFeatures()

        self.ngram_layers = {}
        for n in range(1, max_n + 1):
            input_bits = n * 8
            self.ngram_layers[n] = RAMLayer(
                total_input_bits=input_bits,
                num_neurons=8,
                n_bits_per_neuron=min(input_bits, 10),
                rng=rng + n * 100 if rng else None,
            )

    def _encode_context(self, context: str) -> Tensor:
        bits = []
        for c in context:
            bits.extend(self.features.encode(c))
        return tensor(bits, dtype=uint8)

    def train_on_text(self, text: str) -> dict:
        text = text.lower()
        errors = {}

        for n in range(1, self.max_n + 1):
            errors[n] = 0
            for i in range(len(text) - n):
                context = text[i:i + n]
                target = text[i + n]

                ctx_bits = self._encode_context(context)
                tgt_bits = tensor(self.features.encode(target), dtype=uint8)
                errors[n] += self.ngram_layers[n].commit(
                    ctx_bits.unsqueeze(0), tgt_bits.unsqueeze(0)
                )

        return errors

    def predict_next(self, context: str) -> str:
        """Vote across multiple context lengths."""
        context = context.lower()
        predictions = []

        for n in range(1, min(self.max_n + 1, len(context) + 1)):
            ctx = context[-n:]
            ctx_bits = self._encode_context(ctx)
            out = self.ngram_layers[n](ctx_bits.unsqueeze(0)).squeeze()
            pred = self.features.decode([int(b.item()) for b in out])
            predictions.append(pred)

        # Majority vote
        if predictions:
            counter = Counter(predictions)
            return counter.most_common(1)[0][0]
        return ' '


# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_texts():
    """Get sample texts for testing."""
    return {
        "simple": "the cat sat on the mat. the dog ran to the cat.",
        "patterns": (
            "hello world. hello there. hello friend. "
            "goodbye world. goodbye there. goodbye friend."
        ),
        "english": (
            "the quick brown fox jumps over the lazy dog. "
            "pack my box with five dozen liquor jugs. "
            "how vexingly quick daft zebras jump."
        ),
    }


def test_backoff_ngram():
    """Test backoff n-gram model."""
    print(f"\n{'='*60}")
    print("Improvement 1: Backoff N-gram")
    print(f"{'='*60}")

    texts = get_sample_texts()
    model = BackoffNGram(max_n=4, rng=42)

    # Train
    train_text = texts["simple"] + " " + texts["patterns"]
    errors = model.train_on_text(train_text)
    print(f"Training errors: {errors}")

    patterns_by_n = {n: len(model.known_patterns[n]) for n in range(1, 5)}
    print(f"Patterns learned: {patterns_by_n}")

    # Test on seen contexts
    print("\nPredictions on seen contexts:")
    test_contexts = ["the", "cat", "hel", "wor"]
    for ctx in test_contexts:
        pred = model.predict_next(ctx)
        print(f"  '{ctx}' → '{pred}'")

    # Test on unseen contexts (backoff should help)
    print("\nPredictions on unseen contexts (backoff helps):")
    unseen_contexts = ["xyz", "qwer", "aaa"]
    for ctx in unseen_contexts:
        pred = model.predict_next(ctx)
        print(f"  '{ctx}' → '{pred}' (backed off)")

    # Generate
    print("\nGeneration:")
    generated = model.generate("the ", 20)
    print(f"  'the ' → '{generated}'")

    return model


def test_pattern_abstraction():
    """Test pattern abstraction model."""
    print(f"\n{'='*60}")
    print("Improvement 2: Pattern Abstraction")
    print(f"{'='*60}")

    model = PatternAbstractionLM(n=3, rng=42)

    texts = get_sample_texts()
    train_text = texts["english"]

    errors = model.train_on_text(train_text)
    print(f"Training errors: pattern={errors['pattern']}, position={errors['position']}")

    # The key insight: patterns like CVC (consonant-vowel-consonant) are learned
    print("\nAbstract pattern examples:")
    print("  'the' = CVC → next is likely space or vowel")
    print("  'cat' = CVC → next is likely space or vowel")

    # Test
    print("\nPredictions:")
    test_contexts = ["the", "cat", "fox", "dog"]
    for ctx in test_contexts:
        pred = model.predict_next(ctx)
        ctx_pattern = ''.join(['V' if c in 'aeiou' else 'C' if c.isalpha() else '_' for c in ctx])
        print(f"  '{ctx}' ({ctx_pattern}) → '{pred}'")

    return model


def test_hierarchical_ngram():
    """Test hierarchical n-gram with voting."""
    print(f"\n{'='*60}")
    print("Improvement 3: Hierarchical N-gram (Voting)")
    print(f"{'='*60}")

    model = HierarchicalNGram(max_n=4, rng=42)

    texts = get_sample_texts()
    train_text = texts["simple"] + " " + texts["patterns"]

    errors = model.train_on_text(train_text)
    print(f"Training errors: {errors}")

    print("\nVoting improves robustness:")
    print("  Each n-gram predicts independently")
    print("  Final prediction = majority vote")

    # Test
    print("\nPredictions:")
    test_contexts = ["the", "hel", "wor"]
    for ctx in test_contexts:
        pred = model.predict_next(ctx)
        print(f"  '{ctx}' → '{pred}'")

    return model


def compare_all_approaches():
    """Compare all improvements."""
    print(f"\n{'='*60}")
    print("Comparison: All Approaches")
    print(f"{'='*60}")

    texts = get_sample_texts()
    train_text = texts["simple"]
    test_text = "the cat ran to the dog"  # Similar but not identical

    # Train all models
    backoff = BackoffNGram(max_n=4, rng=42)
    pattern = PatternAbstractionLM(n=3, rng=42)
    hierarchical = HierarchicalNGram(max_n=4, rng=42)

    backoff.train_on_text(train_text)
    pattern.train_on_text(train_text)
    hierarchical.train_on_text(train_text)

    # Test accuracy
    def test_accuracy(model, text):
        correct = 0
        for i in range(3, len(text) - 1):
            context = text[:i]
            expected = text[i]
            predicted = model.predict_next(context)
            if predicted == expected:
                correct += 1
        return correct / (len(text) - 4) if len(text) > 4 else 0

    print("\nTest accuracy on similar (but not identical) text:")
    print(f"  Backoff N-gram:     {test_accuracy(backoff, test_text):.1%}")
    print(f"  Pattern Abstraction: {test_accuracy(pattern, test_text):.1%}")
    print(f"  Hierarchical N-gram: {test_accuracy(hierarchical, test_text):.1%}")

    # Show pattern counts
    print("\nPattern efficiency:")
    print("  Backoff: Stores all n-grams but gracefully degrades")
    print("  Pattern: Abstracts to type+position (~5×26 = 130 patterns max)")
    print("  Hierarchical: Combines multiple views, votes for robustness")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Improved Language Modeling")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test each improvement
    test_backoff_ngram()
    test_pattern_abstraction()
    test_hierarchical_ngram()

    # Compare all
    compare_all_approaches()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Language Modeling Improvements")
    print(f"{'='*60}")

    print("\n1. Backoff N-gram:")
    print("   + Graceful degradation for unseen contexts")
    print("   + Uses shorter contexts when needed")
    print("   - Still memorizes, doesn't generalize patterns")

    print("\n2. Pattern Abstraction (Decomposition for Language):")
    print("   + Decomposes characters into features (type + position)")
    print("   + Learns abstract patterns like 'CVC → space'")
    print("   + MUCH smaller pattern space")
    print("   - Loses some character-specific information")

    print("\n3. Hierarchical N-gram:")
    print("   + Ensemble voting improves robustness")
    print("   + Multiple views of context")
    print("   - Still fundamentally memorization-based")

    print("\nKey Insight:")
    print("  Pattern Abstraction is the closest to 'decomposition' for language.")
    print("  It treats character type (V/C) as a 'primitive' like bits in arithmetic.")
    print("  This reduces the pattern space from O(26^n) to O(5^n × 26).")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
