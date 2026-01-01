"""
Language Model v2: Deeper Analysis and Improvements

Goal: Improve beyond 72% by understanding failure patterns and combining approaches.

Key insight: Pattern Abstraction works by decomposing characters into features.
But we can go further:
1. Combine exact match (when available) with pattern abstraction (fallback)
2. Add more features: bigram patterns, word boundaries
3. Weight recent context more heavily
4. Use multiple abstraction levels together
"""

import random
from datetime import datetime
from collections import Counter, defaultdict

from torch import zeros, uint8, tensor, Tensor, cat
from torch.nn import Module

from wnn.ram.core import RAMLayer


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Character Features
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedCharacterFeatures:
    """
    Extended character features for better pattern learning.

    Features:
    - Type: vowel/consonant/space/punct (3 bits)
    - Position: a-z position (5 bits)
    - Frequency class: common/medium/rare (2 bits)
    - Word position: start/middle/end (2 bits) based on spaces
    """

    VOWELS = set('aeiou')
    CONSONANTS = set('bcdfghjklmnpqrstvwxyz')
    COMMON = set('etaoinshrdlu')  # Most common English letters
    RARE = set('zqxjkvbp')

    def __init__(self):
        self.char_to_idx = {chr(i): i - ord('a') for i in range(ord('a'), ord('z') + 1)}
        self.idx_to_char = {i: chr(i + ord('a')) for i in range(26)}

    def get_type(self, c: str) -> int:
        c = c.lower()
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
        c = c.lower()
        if c in self.char_to_idx:
            return self.char_to_idx[c]
        return 26

    def get_frequency_class(self, c: str) -> int:
        c = c.lower()
        if c in self.COMMON:
            return 0
        elif c in self.RARE:
            return 2
        else:
            return 1

    def encode_basic(self, c: str) -> list[int]:
        """Basic encoding: type (3 bits) + position (5 bits) = 8 bits."""
        char_type = self.get_type(c)
        position = self.get_position(c)

        type_bits = [(char_type >> i) & 1 for i in range(2, -1, -1)]
        pos_bits = [(position >> i) & 1 for i in range(4, -1, -1)]

        return type_bits + pos_bits

    def encode_extended(self, c: str, prev_c: str = None, next_c: str = None) -> list[int]:
        """Extended encoding with context awareness."""
        basic = self.encode_basic(c)

        # Add frequency class (2 bits)
        freq = self.get_frequency_class(c)
        freq_bits = [(freq >> i) & 1 for i in range(1, -1, -1)]

        # Add word position indicator (2 bits)
        # 0 = word start, 1 = word middle, 2 = word end, 3 = space/punct
        if c == ' ' or c in '.,!?':
            word_pos = 3
        elif prev_c is None or prev_c == ' ' or prev_c in '.,!?':
            word_pos = 0  # Word start
        elif next_c is None or next_c == ' ' or next_c in '.,!?':
            word_pos = 2  # Word end
        else:
            word_pos = 1  # Word middle

        word_bits = [(word_pos >> i) & 1 for i in range(1, -1, -1)]

        return basic + freq_bits + word_bits  # 12 bits total

    def decode(self, bits: list[int]) -> str:
        """Decode from basic encoding."""
        if len(bits) < 8:
            return '?'
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
# Combined Model: Exact + Pattern Abstraction
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLanguageModel(Module):
    """
    Combines exact n-gram matching with pattern abstraction.

    Strategy:
    1. Try exact n-gram match first (100% accurate when found)
    2. Fall back to pattern abstraction (good generalization)
    3. Use confidence weighting between approaches

    This gets the best of both worlds:
    - High accuracy on seen patterns
    - Good generalization on unseen patterns
    """

    def __init__(self, n: int = 3, rng: int | None = None):
        super().__init__()
        self.n = n
        self.features = EnhancedCharacterFeatures()

        # Exact match layer (full character encoding)
        self.exact_layer = RAMLayer(
            total_input_bits=n * 8,  # 8 bits per char
            num_neurons=8,
            n_bits_per_neuron=min(n * 8, 12),
            rng=rng,
        )

        # Pattern abstraction layer (type only)
        self.pattern_layer = RAMLayer(
            total_input_bits=n * 3,  # 3 bits (type) per char
            num_neurons=3,  # Predict next type
            n_bits_per_neuron=n * 3,
            rng=rng + 100 if rng else None,
        )

        # Position prediction given type
        self.position_layer = RAMLayer(
            total_input_bits=n * 3 + 3,  # Context types + target type
            num_neurons=5,  # Position (0-25)
            n_bits_per_neuron=min(n * 3 + 3, 12),
            rng=rng + 200 if rng else None,
        )

        # Track which exact patterns are known
        self.known_exact = set()

    def train_on_text(self, text: str) -> dict:
        """Train both exact and pattern models."""
        text = text.lower()
        exact_errors = 0
        pattern_errors = 0
        position_errors = 0

        for i in range(len(text) - self.n):
            context = text[i:i + self.n]
            target = text[i + self.n]

            # Train exact layer
            self.known_exact.add(context)
            ctx_bits = []
            for c in context:
                ctx_bits.extend(self.features.encode_basic(c))
            tgt_bits = self.features.encode_basic(target)

            ctx_tensor = tensor(ctx_bits, dtype=uint8)
            tgt_tensor = tensor(tgt_bits, dtype=uint8)
            exact_errors += self.exact_layer.commit(
                ctx_tensor.unsqueeze(0), tgt_tensor.unsqueeze(0)
            )

            # Train pattern layer (type sequence → next type)
            type_bits = []
            for c in context:
                t = self.features.get_type(c)
                type_bits.extend([(t >> j) & 1 for j in range(2, -1, -1)])

            target_type = self.features.get_type(target)
            target_type_bits = [(target_type >> j) & 1 for j in range(2, -1, -1)]

            type_tensor = tensor(type_bits, dtype=uint8)
            tgt_type_tensor = tensor(target_type_bits, dtype=uint8)
            pattern_errors += self.pattern_layer.commit(
                type_tensor.unsqueeze(0), tgt_type_tensor.unsqueeze(0)
            )

            # Train position layer
            target_pos = self.features.get_position(target)
            target_pos_bits = [(target_pos >> j) & 1 for j in range(4, -1, -1)]

            pos_input = tensor(type_bits + target_type_bits, dtype=uint8)
            pos_output = tensor(target_pos_bits, dtype=uint8)
            position_errors += self.position_layer.commit(
                pos_input.unsqueeze(0), pos_output.unsqueeze(0)
            )

        return {
            "exact": exact_errors,
            "pattern": pattern_errors,
            "position": position_errors,
            "known_patterns": len(self.known_exact),
        }

    def predict_next(self, context: str) -> str:
        """Predict using combined approach."""
        context = context[-self.n:].lower()
        if len(context) < self.n:
            context = ' ' * (self.n - len(context)) + context

        # Strategy 1: Try exact match first
        if context in self.known_exact:
            ctx_bits = []
            for c in context:
                ctx_bits.extend(self.features.encode_basic(c))
            ctx_tensor = tensor(ctx_bits, dtype=uint8)
            out = self.exact_layer(ctx_tensor.unsqueeze(0)).squeeze()
            return self.features.decode([int(b.item()) for b in out])

        # Strategy 2: Pattern abstraction fallback
        # Get type sequence
        type_bits = []
        for c in context:
            t = self.features.get_type(c)
            type_bits.extend([(t >> j) & 1 for j in range(2, -1, -1)])

        # Predict next type
        type_tensor = tensor(type_bits, dtype=uint8)
        pred_type = self.pattern_layer(type_tensor.unsqueeze(0)).squeeze()
        pred_type_bits = [int(b.item()) for b in pred_type]

        # Predict position given type
        pos_input = tensor(type_bits + pred_type_bits, dtype=uint8)
        pred_pos = self.position_layer(pos_input.unsqueeze(0)).squeeze()

        # Decode
        full_bits = pred_type_bits + [int(b.item()) for b in pred_pos]
        return self.features.decode(full_bits)

    def test_on_text(self, text: str) -> tuple[float, dict]:
        """Test and return accuracy plus breakdown."""
        text = text.lower()
        correct = 0
        exact_hits = 0
        pattern_hits = 0
        total = 0

        for i in range(self.n, len(text)):
            context = text[i - self.n:i]
            expected = text[i]

            # Track which method was used
            used_exact = context in self.known_exact

            predicted = self.predict_next(context)

            if predicted == expected:
                correct += 1
                if used_exact:
                    exact_hits += 1
                else:
                    pattern_hits += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy, {
            "exact_hits": exact_hits,
            "pattern_hits": pattern_hits,
            "total": total,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Scale Pattern Model
# ─────────────────────────────────────────────────────────────────────────────

class MultiScalePatternModel(Module):
    """
    Learn patterns at multiple scales simultaneously.

    - 1-gram: Single character type → next type
    - 2-gram: Pair of types → next type
    - 3-gram: Triple of types → next type

    Combine predictions using voting or confidence weighting.
    """

    def __init__(self, max_n: int = 4, rng: int | None = None):
        super().__init__()
        self.max_n = max_n
        self.features = EnhancedCharacterFeatures()

        # Pattern layers for each scale
        self.pattern_layers = {}
        self.position_layers = {}

        for n in range(1, max_n + 1):
            # Type sequence → next type
            self.pattern_layers[n] = RAMLayer(
                total_input_bits=n * 3,
                num_neurons=3,
                n_bits_per_neuron=n * 3,
                rng=rng + n * 100 if rng else None,
            )

            # Type sequence + next type → position
            self.position_layers[n] = RAMLayer(
                total_input_bits=n * 3 + 3,
                num_neurons=5,
                n_bits_per_neuron=min(n * 3 + 3, 12),
                rng=rng + n * 100 + 50 if rng else None,
            )

    def _get_type_bits(self, c: str) -> list[int]:
        t = self.features.get_type(c)
        return [(t >> i) & 1 for i in range(2, -1, -1)]

    def train_on_text(self, text: str) -> dict:
        text = text.lower()
        errors = {n: {"pattern": 0, "position": 0} for n in range(1, self.max_n + 1)}

        for n in range(1, self.max_n + 1):
            for i in range(len(text) - n):
                context = text[i:i + n]
                target = text[i + n]

                # Get type bits
                type_bits = []
                for c in context:
                    type_bits.extend(self._get_type_bits(c))

                target_type_bits = self._get_type_bits(target)
                target_pos = self.features.get_position(target)
                target_pos_bits = [(target_pos >> j) & 1 for j in range(4, -1, -1)]

                # Train pattern layer
                type_tensor = tensor(type_bits, dtype=uint8)
                tgt_type_tensor = tensor(target_type_bits, dtype=uint8)
                errors[n]["pattern"] += self.pattern_layers[n].commit(
                    type_tensor.unsqueeze(0), tgt_type_tensor.unsqueeze(0)
                )

                # Train position layer
                pos_input = tensor(type_bits + target_type_bits, dtype=uint8)
                pos_output = tensor(target_pos_bits, dtype=uint8)
                errors[n]["position"] += self.position_layers[n].commit(
                    pos_input.unsqueeze(0), pos_output.unsqueeze(0)
                )

        return errors

    def predict_next(self, context: str) -> str:
        """Predict using multi-scale voting."""
        context = context.lower()
        predictions = []

        for n in range(1, min(self.max_n + 1, len(context) + 1)):
            ctx = context[-n:]

            type_bits = []
            for c in ctx:
                type_bits.extend(self._get_type_bits(c))

            # Predict type
            type_tensor = tensor(type_bits, dtype=uint8)
            pred_type = self.pattern_layers[n](type_tensor.unsqueeze(0)).squeeze()
            pred_type_bits = [int(b.item()) for b in pred_type]

            # Predict position
            pos_input = tensor(type_bits + pred_type_bits, dtype=uint8)
            pred_pos = self.position_layers[n](pos_input.unsqueeze(0)).squeeze()

            # Decode
            full_bits = pred_type_bits + [int(b.item()) for b in pred_pos]
            predictions.append(self.features.decode(full_bits))

        # Majority vote
        if predictions:
            counter = Counter(predictions)
            return counter.most_common(1)[0][0]
        return ' '


# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_texts():
    train = "the cat sat on the mat. the dog ran to the cat."
    # Test on similar but not identical text
    test = "the cat ran to the dog"
    return {
        "simple": train,
        "train": train,
        "test": test,
    }


def test_combined_model():
    """Test combined exact + pattern model."""
    print(f"\n{'='*60}")
    print("Combined Model: Exact + Pattern Abstraction")
    print(f"{'='*60}")

    texts = get_sample_texts()
    model = CombinedLanguageModel(n=3, rng=42)

    # Train
    errors = model.train_on_text(texts["train"])
    print(f"Training errors: {errors}")

    # Test on training data
    train_acc, train_stats = model.test_on_text(texts["train"])
    print(f"\nTrain accuracy: {train_acc:.1%}")
    print(f"  Exact matches: {train_stats['exact_hits']}")
    print(f"  Pattern matches: {train_stats['pattern_hits']}")

    # Test on unseen data
    test_acc, test_stats = model.test_on_text(texts["test"])
    print(f"\nTest accuracy: {test_acc:.1%}")
    print(f"  Exact matches: {test_stats['exact_hits']}")
    print(f"  Pattern matches: {test_stats['pattern_hits']}")

    return test_acc


def test_multiscale_model():
    """Test multi-scale pattern model."""
    print(f"\n{'='*60}")
    print("Multi-Scale Pattern Model")
    print(f"{'='*60}")

    texts = get_sample_texts()
    model = MultiScalePatternModel(max_n=4, rng=42)

    # Train
    errors = model.train_on_text(texts["train"])
    print(f"Training errors by scale:")
    for n, e in errors.items():
        print(f"  {n}-gram: pattern={e['pattern']}, position={e['position']}")

    # Test
    def test_accuracy(model, text):
        text = text.lower()
        correct = 0
        for i in range(3, len(text)):
            context = text[:i]
            expected = text[i]
            predicted = model.predict_next(context)
            if predicted == expected:
                correct += 1
        return correct / (len(text) - 3) if len(text) > 3 else 0

    train_acc = test_accuracy(model, texts["train"])
    test_acc = test_accuracy(model, texts["test"])

    print(f"\nTrain accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")

    return test_acc


def compare_all_models():
    """Compare all language model approaches."""
    print(f"\n{'='*60}")
    print("Comparison: All Approaches")
    print(f"{'='*60}")

    texts = get_sample_texts()

    # Import original models
    import sys
    sys.path.insert(0, '/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn/tests')
    from language_model_improved import BackoffNGram, PatternAbstractionLM, HierarchicalNGram

    # Train all models
    models = {
        "Backoff N-gram": BackoffNGram(max_n=4, rng=42),
        "Pattern Abstraction (v1)": PatternAbstractionLM(n=3, rng=42),
        "Hierarchical N-gram": HierarchicalNGram(max_n=4, rng=42),
        "Combined (Exact+Pattern)": CombinedLanguageModel(n=3, rng=42),
        "Multi-Scale Pattern": MultiScalePatternModel(max_n=4, rng=42),
        "Frequency Aware": FrequencyAwareModel(n=3),
    }

    for name, model in models.items():
        model.train_on_text(texts["train"])

    # Test accuracy
    def test_accuracy(model, text):
        text = text.lower()
        correct = 0
        for i in range(3, len(text)):
            context = text[:i]
            expected = text[i]
            predicted = model.predict_next(context)
            if predicted == expected:
                correct += 1
        return correct / (len(text) - 3) if len(text) > 3 else 0

    print("\nResults on test set:")
    results = {}
    for name, model in models.items():
        acc = test_accuracy(model, texts["test"])
        results[name] = acc
        print(f"  {name}: {acc:.1%}")

    # Find best
    best = max(results, key=results.get)
    print(f"\nBest: {best} ({results[best]:.1%})")

    return results


class FrequencyAwareModel(Module):
    """
    Language model that tracks pattern frequencies.

    Key insight: Same context can have multiple valid continuations.
    Instead of memorizing one answer, track all and pick most frequent.
    """

    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n
        self.features = EnhancedCharacterFeatures()

        # Store counts: context → {next_char: count}
        self.context_counts = defaultdict(Counter)

        # Pattern abstraction fallback
        self.pattern_counts = defaultdict(Counter)  # type_pattern → {next_type: count}

    def train_on_text(self, text: str):
        """Count all context → next_char occurrences."""
        text = text.lower()

        for i in range(len(text) - self.n):
            context = text[i:i + self.n]
            target = text[i + self.n]

            # Exact context
            self.context_counts[context][target] += 1

            # Pattern abstraction
            pattern = tuple(self.features.get_type(c) for c in context)
            target_type = self.features.get_type(target)
            self.pattern_counts[pattern][target_type] += 1

        return {
            "contexts": len(self.context_counts),
            "patterns": len(self.pattern_counts),
        }

    def predict_next(self, context: str) -> str:
        """Predict most frequent continuation."""
        context = context[-self.n:].lower()
        if len(context) < self.n:
            context = ' ' * (self.n - len(context)) + context

        # Try exact context first
        if context in self.context_counts:
            return self.context_counts[context].most_common(1)[0][0]

        # Fall back to pattern
        pattern = tuple(self.features.get_type(c) for c in context)
        if pattern in self.pattern_counts:
            # Get most common next type
            next_type = self.pattern_counts[pattern].most_common(1)[0][0]

            # Find most common character of that type
            type_chars = {
                0: 'e',  # vowel - 'e' is most common
                1: 't',  # consonant - 't' is most common
                2: ' ',  # space
                3: '.',  # punctuation
            }
            return type_chars.get(next_type, ' ')

        return ' '


def test_frequency_aware_model():
    """Test frequency-aware model that handles ambiguous contexts."""
    print(f"\n{'='*60}")
    print("Frequency-Aware Model: Handling Ambiguous Contexts")
    print(f"{'='*60}")

    texts = get_sample_texts()
    model = FrequencyAwareModel(n=3)

    # Train
    stats = model.train_on_text(texts["train"])
    print(f"Training stats: {stats}")

    # Show ambiguous patterns (same context, different continuations)
    print("\nAmbiguous patterns found:")
    ambiguous = []
    for ctx, counts in model.context_counts.items():
        if len(counts) > 1:
            ambiguous.append((ctx, counts))
            print(f"  '{ctx}' → {dict(counts)}")

    print(f"\nTotal unique contexts: {len(model.context_counts)}")
    print(f"Ambiguous contexts: {len(ambiguous)}")

    # Test accuracy
    def test_accuracy(model, text):
        text = text.lower()
        correct = 0
        for i in range(3, len(text)):
            context = text[i - 3:i]
            expected = text[i]
            predicted = model.predict_next(context)
            if predicted == expected:
                correct += 1
        return correct / (len(text) - 3) if len(text) > 3 else 0

    train_acc = test_accuracy(model, texts["train"])
    test_acc = test_accuracy(model, texts["test"])

    print(f"\nTrain accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")

    return test_acc


def analyze_ambiguity_impact():
    """Analyze how context ambiguity affects language model performance."""
    print(f"\n{'='*60}")
    print("Ambiguity Impact Analysis")
    print(f"{'='*60}")

    # Use larger corpus to see more patterns
    train_corpus = """
    the cat sat on the mat. the dog ran to the cat.
    a cat is a pet. a dog is a pet too.
    the bird flew over the tree. the cat watched the bird.
    cats and dogs are common pets. some prefer cats.
    the quick brown fox jumps over the lazy dog.
    """

    test_text = "the cat ran to the dog and sat on the mat"

    model = FrequencyAwareModel(n=3)
    model.train_on_text(train_corpus)

    # Count how many test predictions involve ambiguous contexts
    text = test_text.lower()
    ambiguous_predictions = 0
    unambiguous_predictions = 0
    unseen_contexts = 0

    for i in range(3, len(text)):
        context = text[i - 3:i]
        expected = text[i]

        if context in model.context_counts:
            counts = model.context_counts[context]
            if len(counts) > 1:
                ambiguous_predictions += 1
                predicted = counts.most_common(1)[0][0]
                is_correct = predicted == expected
                print(f"  Ambiguous: '{context}' → predict '{predicted}' "
                      f"(expected '{expected}', counts: {dict(counts)}) "
                      f"{'✓' if is_correct else '✗'}")
            else:
                unambiguous_predictions += 1
        else:
            unseen_contexts += 1

    total = len(text) - 3
    print(f"\nPrediction breakdown:")
    print(f"  Unambiguous contexts: {unambiguous_predictions} ({100*unambiguous_predictions/total:.0f}%)")
    print(f"  Ambiguous contexts: {ambiguous_predictions} ({100*ambiguous_predictions/total:.0f}%)")
    print(f"  Unseen contexts: {unseen_contexts} ({100*unseen_contexts/total:.0f}%)")

    # Key insight
    print("\n" + "="*40)
    print("KEY INSIGHT")
    print("="*40)
    print("Language prediction has an upper bound due to inherent ambiguity.")
    print("Same context can legitimately lead to different continuations.")
    print("Unlike arithmetic (100% deterministic), language is stochastic.")
    print("\nImprovement strategies:")
    print("1. Longer context (reduces ambiguity)")
    print("2. Word-level instead of character-level")
    print("3. Accept probabilistic outputs instead of single predictions")


def analyze_failure_patterns():
    """Analyze what patterns the models fail on."""
    print(f"\n{'='*60}")
    print("Failure Pattern Analysis")
    print(f"{'='*60}")

    texts = get_sample_texts()
    model = CombinedLanguageModel(n=3, rng=42)
    model.train_on_text(texts["train"])

    text = texts["test"].lower()
    failures = []
    successes = []

    for i in range(3, len(text)):
        context = text[i - 3:i]
        expected = text[i]
        predicted = model.predict_next(context)

        if predicted != expected:
            failures.append({
                "context": context,
                "expected": expected,
                "predicted": predicted,
                "used_exact": context in model.known_exact,
            })
        else:
            successes.append({
                "context": context,
                "expected": expected,
                "used_exact": context in model.known_exact,
            })

    print(f"\nSuccesses: {len(successes)}/{len(failures) + len(successes)}")
    print(f"Failures: {len(failures)}")

    if failures:
        print("\nFailure examples:")
        for f in failures[:5]:
            exact_str = "(exact)" if f["used_exact"] else "(pattern)"
            print(f"  '{f['context']}' → '{f['predicted']}' "
                  f"(expected '{f['expected']}') {exact_str}")

    # Analyze failure patterns
    print("\nFailure pattern analysis:")
    failure_types = Counter()
    for f in failures:
        ctx_pattern = ''.join(['V' if c in 'aeiou' else 'C' if c.isalpha() else '_' for c in f['context']])
        exp_type = 'V' if f['expected'] in 'aeiou' else 'C' if f['expected'].isalpha() else '_'
        failure_types[f"{ctx_pattern} → {exp_type}"] += 1

    print("  Most common failure patterns:")
    for pattern, count in failure_types.most_common(5):
        print(f"    {pattern}: {count}")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("Language Model v2: Analysis and Improvements")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}")

    # Test combined model
    combined_acc = test_combined_model()

    # Test multi-scale model
    multiscale_acc = test_multiscale_model()

    # Test frequency-aware model
    freq_acc = test_frequency_aware_model()

    # Analyze ambiguity impact
    analyze_ambiguity_impact()

    # Compare all approaches
    results = compare_all_models()

    # Analyze failures
    analyze_failure_patterns()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nImprovement results:")
    print(f"  Original Pattern Abstraction: 72%")
    print(f"  Combined (Exact+Pattern):     {results.get('Combined (Exact+Pattern)', 0):.0%}")
    print(f"  Multi-Scale Pattern:          {results.get('Multi-Scale Pattern', 0):.0%}")
    print(f"  Frequency Aware:              {results.get('Frequency Aware', 0):.0%}")

    best = max(results, key=results.get)
    best_acc = results[best]

    print(f"\nBest approach: {best}")
    print(f"Best accuracy: {best_acc:.0%}")

    if best_acc > 0.72:
        print(f"\nImprovement: {best_acc - 0.72:.0%} over original 72%")
    else:
        print(f"\nNo improvement over 72% baseline.")
        print("This is expected - language has inherent ambiguity that limits accuracy.")

    print(f"\n{'='*60}")
    print(f"Finished at: {datetime.now()}")
    print(f"{'='*60}")
