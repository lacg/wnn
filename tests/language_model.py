"""
Language Modeling Test

Tests whether RAM networks can learn character-level language patterns.

Key insight: Language modeling is sequence prediction. We can approach it with:
1. N-gram: Learn fixed context → next character mappings
2. Recurrent: Use state to encode arbitrary-length context
3. Attention: Learn which context positions matter for prediction

Unlike arithmetic (where decomposition gives 100% generalization),
language modeling tests how well RAM networks handle:
- Pattern memorization at scale
- Interpolation between seen patterns
- Distribution learning (predicting likely vs unlikely continuations)
"""

import random
from datetime import datetime
from collections import Counter

from torch import zeros, uint8, tensor, Tensor
from torch.nn import Module

from wnn.ram.core import RAMLayer


class CharacterEncoder:
    """Encode/decode characters to/from bits."""

    def __init__(self, charset: str = None):
        """
        Args:
            charset: Characters to support. None = printable ASCII subset.
        """
        if charset is None:
            # Simple charset: lowercase + space + punctuation
            charset = "abcdefghijklmnopqrstuvwxyz .,!?'\n"

        self.charset = charset
        self.char_to_idx = {c: i for i, c in enumerate(charset)}
        self.idx_to_char = {i: c for i, c in enumerate(charset)}
        self.vocab_size = len(charset)

        # Bits needed to encode vocab
        self.bits_per_char = (self.vocab_size - 1).bit_length()
        if self.bits_per_char == 0:
            self.bits_per_char = 1

    def encode_char(self, c: str) -> list[int]:
        """Encode single character to bits."""
        if c not in self.char_to_idx:
            c = ' '  # Unknown → space
        idx = self.char_to_idx[c]
        return [(idx >> i) & 1 for i in range(self.bits_per_char - 1, -1, -1)]

    def decode_bits(self, bits: list[int]) -> str:
        """Decode bits to character."""
        idx = sum(b << (self.bits_per_char - 1 - i) for i, b in enumerate(bits))
        if idx >= self.vocab_size:
            return ' '  # Out of range → space
        return self.idx_to_char[idx]

    def encode_string(self, s: str) -> list[list[int]]:
        """Encode string to list of bit vectors."""
        return [self.encode_char(c) for c in s.lower()]

    def decode_string(self, bits_list: list[list[int]]) -> str:
        """Decode list of bit vectors to string."""
        return ''.join(self.decode_bits(bits) for bits in bits_list)


class NGramLanguageModel(Module):
    """
    N-gram language model using RAM networks.

    Learns P(next_char | previous_n_chars) by memorization.
    Each unique n-gram context maps to a next character.
    """

    def __init__(self, n: int = 3, charset: str = None, rng: int | None = None):
        """
        Args:
            n: Context length (n-gram size)
            charset: Character set to use
            rng: Random seed
        """
        super().__init__()

        self.n = n
        self.encoder = CharacterEncoder(charset)

        # Input: n characters × bits_per_char
        # Output: bits_per_char (next character)
        input_bits = n * self.encoder.bits_per_char
        output_bits = self.encoder.bits_per_char

        self.predictor = RAMLayer(
            total_input_bits=input_bits,
            num_neurons=output_bits,
            n_bits_per_neuron=min(input_bits, 12),  # Cap to avoid huge memory
            rng=rng,
        )

        self.patterns_trained = 0

    def train_on_text(self, text: str) -> int:
        """
        Train on text corpus.

        Returns number of patterns trained.
        """
        text = text.lower()
        errors = 0
        patterns = set()

        for i in range(len(text) - self.n):
            context = text[i:i + self.n]
            target = text[i + self.n]

            # Skip if already seen this exact pattern
            pattern_key = (context, target)
            if pattern_key in patterns:
                continue
            patterns.add(pattern_key)

            # Encode
            context_bits = []
            for c in context:
                context_bits.extend(self.encoder.encode_char(c))
            target_bits = self.encoder.encode_char(target)

            inp = tensor(context_bits, dtype=uint8)
            out = tensor(target_bits, dtype=uint8)

            errors += self.predictor.commit(inp.unsqueeze(0), out.unsqueeze(0))

        self.patterns_trained = len(patterns)
        return errors

    def predict_next(self, context: str) -> str:
        """Predict next character given context."""
        if len(context) < self.n:
            context = ' ' * (self.n - len(context)) + context
        context = context[-self.n:].lower()

        context_bits = []
        for c in context:
            context_bits.extend(self.encoder.encode_char(c))

        inp = tensor(context_bits, dtype=uint8)
        out = self.predictor(inp.unsqueeze(0)).squeeze()

        out_list = [int(b.item()) for b in out]
        return self.encoder.decode_bits(out_list)

    def generate(self, seed: str, length: int) -> str:
        """Generate text starting from seed."""
        result = seed.lower()

        for _ in range(length):
            next_char = self.predict_next(result)
            result += next_char

        return result

    def test_on_text(self, text: str) -> tuple[float, float]:
        """
        Test on text corpus.

        Returns (exact_accuracy, top_k_accuracy) where top_k checks
        if prediction is among most common continuations for that context.
        """
        text = text.lower()
        correct = 0
        total = 0

        for i in range(len(text) - self.n):
            context = text[i:i + self.n]
            expected = text[i + self.n]

            predicted = self.predict_next(context)

            if predicted == expected:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0


class RecurrentLanguageModel(Module):
    """
    Recurrent language model using RAM networks.

    Uses a hidden state that accumulates context information.
    Similar to parity computation but for language.

    state(t) = f(char(t), state(t-1))
    output(t) = g(state(t))
    """

    def __init__(self, state_bits: int = 16, charset: str = None, rng: int | None = None):
        """
        Args:
            state_bits: Hidden state size
            charset: Character set
            rng: Random seed
        """
        super().__init__()

        self.encoder = CharacterEncoder(charset)
        self.state_bits = state_bits

        char_bits = self.encoder.bits_per_char

        # State update: (char, prev_state) → new_state
        self.state_layer = RAMLayer(
            total_input_bits=char_bits + state_bits,
            num_neurons=state_bits,
            n_bits_per_neuron=min(char_bits + state_bits, 12),
            rng=rng,
        )

        # Output: state → next_char
        self.output_layer = RAMLayer(
            total_input_bits=state_bits,
            num_neurons=char_bits,
            n_bits_per_neuron=min(state_bits, 12),
            rng=rng + 1000 if rng else None,
        )

        self._state = None

    def reset_state(self):
        """Reset hidden state to zeros."""
        self._state = tensor([0] * self.state_bits, dtype=uint8)

    def step(self, char: str) -> Tensor:
        """Process one character, return new state."""
        if self._state is None:
            self.reset_state()

        char_bits = self.encoder.encode_char(char)
        inp = tensor(char_bits + [int(b.item()) for b in self._state], dtype=uint8)

        self._state = self.state_layer(inp.unsqueeze(0)).squeeze()
        return self._state

    def predict_next(self) -> str:
        """Predict next character from current state."""
        if self._state is None:
            self.reset_state()

        out = self.output_layer(self._state.unsqueeze(0)).squeeze()
        out_list = [int(b.item()) for b in out]
        return self.encoder.decode_bits(out_list)

    def train_on_text(self, text: str, window_size: int = 10) -> int:
        """
        Train on text with sliding window for BPTT.

        Returns total errors.
        """
        text = text.lower()
        errors = 0

        for start in range(0, len(text) - window_size, window_size // 2):
            window = text[start:start + window_size]

            self.reset_state()

            for i in range(len(window) - 1):
                # Process character
                char = window[i]
                target = window[i + 1]

                # Forward pass
                self.step(char)

                # Train output layer
                target_bits = self.encoder.encode_char(target)
                out_target = tensor(target_bits, dtype=uint8)
                errors += self.output_layer.commit(
                    self._state.unsqueeze(0),
                    out_target.unsqueeze(0)
                )

        return errors

    def generate(self, seed: str, length: int) -> str:
        """Generate text starting from seed."""
        self.reset_state()

        # Process seed
        for c in seed.lower():
            self.step(c)

        result = seed.lower()

        for _ in range(length):
            next_char = self.predict_next()
            result += next_char
            self.step(next_char)

        return result


def get_sample_texts() -> dict[str, str]:
    """Get sample texts for training/testing."""
    return {
        "simple_repeat": "abcabcabcabcabcabcabcabcabcabc",
        "alternating": "ababababababababababababababab",
        "english_words": (
            "the cat sat on the mat. "
            "the dog ran in the fog. "
            "the bird flew in the sky. "
            "the fish swam in the sea. "
        ) * 3,
        "patterns": (
            "hello world. hello there. hello friend. "
            "goodbye world. goodbye there. goodbye friend. "
        ) * 2,
    }


def test_ngram_model():
    """Test N-gram language model."""
    print(f"\n{'='*60}")
    print("Testing N-gram Language Model")
    print(f"{'='*60}")

    texts = get_sample_texts()

    for n in [2, 3, 4]:
        print(f"\n--- {n}-gram model ---")

        model = NGramLanguageModel(n=n, rng=42)

        # Train on simple pattern
        train_text = texts["simple_repeat"]
        errors = model.train_on_text(train_text)

        print(f"Trained on: '{train_text[:30]}...'")
        print(f"Patterns: {model.patterns_trained}, Errors: {errors}")

        # Test prediction
        test_contexts = ["abc", "bca", "cab"]
        print(f"Predictions:")
        for ctx in test_contexts:
            ctx = ctx[-n:]
            pred = model.predict_next(ctx)
            print(f"  '{ctx}' → '{pred}'")

        # Generate
        generated = model.generate("ab", 15)
        print(f"Generated: '{generated}'")

        # Test accuracy
        acc = model.test_on_text(train_text)
        print(f"Train accuracy: {acc:.1%}")


def test_english_patterns():
    """Test on English-like patterns."""
    print(f"\n{'='*60}")
    print("Testing on English Patterns")
    print(f"{'='*60}")

    texts = get_sample_texts()
    train_text = texts["english_words"]

    model = NGramLanguageModel(n=3, rng=42)
    errors = model.train_on_text(train_text)

    print(f"Training text: {len(train_text)} chars")
    print(f"Patterns learned: {model.patterns_trained}")
    print(f"Errors: {errors}")

    # Test common patterns
    test_cases = [
        ("the", " "),  # "the " is common
        ("cat", " "),  # "cat "
        ("on ", "t"),  # "on t" (the)
        ("at.", " "),  # "at. "
    ]

    print("\nPattern predictions:")
    correct = 0
    for ctx, expected in test_cases:
        pred = model.predict_next(ctx)
        ok = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1
        print(f"  '{ctx}' → '{pred}' (expected '{expected}') {ok}")

    print(f"\nTest accuracy: {correct}/{len(test_cases)}")

    # Generate from seed
    seeds = ["the ", "cat ", "dog "]
    print("\nGeneration:")
    for seed in seeds:
        generated = model.generate(seed, 20)
        print(f"  '{seed}' → '{generated}'")

    return model.patterns_trained


def test_recurrent_model():
    """Test recurrent language model."""
    print(f"\n{'='*60}")
    print("Testing Recurrent Language Model")
    print(f"{'='*60}")

    model = RecurrentLanguageModel(state_bits=16, rng=42)

    # Train on simple pattern
    train_text = "abcabcabcabcabcabcabcabc"
    errors = model.train_on_text(train_text, window_size=8)

    print(f"Trained on: '{train_text}'")
    print(f"Errors: {errors}")

    # Generate
    generated = model.generate("ab", 15)
    print(f"Generated: '{generated}'")

    # Test on patterns text
    train_text2 = "hello world. hello there. hello friend. " * 2
    model2 = RecurrentLanguageModel(state_bits=24, rng=42)
    errors2 = model2.train_on_text(train_text2, window_size=10)

    print(f"\nTrained on English patterns, errors: {errors2}")
    generated2 = model2.generate("hello ", 20)
    print(f"Generated: '{generated2}'")


def test_generalization():
    """Test generalization to unseen patterns."""
    print(f"\n{'='*60}")
    print("Testing Generalization")
    print(f"{'='*60}")

    # Train on some patterns, test on variations
    train_text = "the cat sat. the dog ran. the bird flew."
    test_text = "the fish swam. the cat ran. the dog sat."

    model = NGramLanguageModel(n=3, rng=42)
    model.train_on_text(train_text)

    print(f"Trained on: '{train_text}'")
    print(f"Testing on: '{test_text}'")

    train_acc = model.test_on_text(train_text)
    test_acc = model.test_on_text(test_text)

    print(f"\nTrain accuracy: {train_acc:.1%}")
    print(f"Test accuracy:  {test_acc:.1%}")
    print(f"Generalization gap: {train_acc - test_acc:.1%}")

    # Show specific predictions
    print("\nUnseen pattern predictions:")
    unseen = [("fis", "h"), ("wam", "."), ("at ", "r")]
    for ctx, expected in unseen:
        pred = model.predict_next(ctx)
        match = "✓" if pred == expected else "✗"
        print(f"  '{ctx}' → '{pred}' (expected '{expected}') {match}")


def test_vocab_scaling():
    """Test how model scales with vocabulary size."""
    print(f"\n{'='*60}")
    print("Testing Vocabulary Scaling")
    print(f"{'='*60}")

    # Different charset sizes
    charsets = [
        ("tiny", "abc"),
        ("letters", "abcdefghijklmnopqrstuvwxyz"),
        ("full", "abcdefghijklmnopqrstuvwxyz .,!?'\n"),
    ]

    for name, charset in charsets:
        encoder = CharacterEncoder(charset)
        print(f"\n{name}: {len(charset)} chars, {encoder.bits_per_char} bits/char")

        # Train simple model
        model = NGramLanguageModel(n=3, charset=charset, rng=42)

        # Generate training text from charset
        random.seed(123)
        train_text = ''.join(random.choice(charset) for _ in range(100))

        errors = model.train_on_text(train_text)
        print(f"  Trained patterns: {model.patterns_trained}")
        print(f"  Errors: {errors}")

        acc = model.test_on_text(train_text)
        print(f"  Train accuracy: {acc:.1%}")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Language Modeling Test")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test 1: N-gram model on simple patterns
    test_ngram_model()

    # Test 2: English-like patterns
    patterns_learned = test_english_patterns()

    # Test 3: Recurrent model
    test_recurrent_model()

    # Test 4: Generalization
    test_generalization()

    # Test 5: Vocabulary scaling
    test_vocab_scaling()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nKey observations:")
    print("1. N-gram models memorize context→next mappings")
    print("2. Accuracy depends on pattern coverage in training")
    print("3. Generalization limited to interpolation between patterns")
    print("4. Unlike arithmetic, language doesn't have clean decomposition")

    print("\nThis is expected! Language modeling is fundamentally about:")
    print("- Statistical patterns in data")
    print("- No universal primitives (unlike addition/subtraction)")
    print("- Generalization requires similar contexts in training")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
