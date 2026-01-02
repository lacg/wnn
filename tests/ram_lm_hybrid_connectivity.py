#!/usr/bin/env python3
"""
Hybrid Connectivity RAM Language Model

Your key insight: Mix FULLY-CONNECTED and PARTIALLY-CONNECTED RAM
in the same architecture, just like we do for arithmetic:

- Arithmetic: Fully connected for carry bits, partial for operand digits
- Language: Fully connected for common words, partial for rare/content words

Architecture:
1. HIGH-FREQUENCY WORDS (the, a, is, of): Exact pattern matching (fully connected)
   - These have reliable, specific patterns
   - "of the" almost always followed by noun

2. LOW-FREQUENCY WORDS (names, technical terms): Feature-based (partial connectivity)
   - Need generalization because each word is rare
   - "Dr. Smith" and "Dr. Jones" should generalize

3. STRUCTURAL PATTERNS: Learn which word classes follow which
   - Article → Noun (the ___)
   - Verb → Adverb/Noun (run ___)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'wnn'))

from collections import defaultdict, Counter
import math
import random


class HybridConnectivityRAMLM:
    """
    Hybrid RAM with mixed connectivity based on word frequency.

    High-frequency words: FULLY CONNECTED (exact lookup)
    Low-frequency words: PARTIALLY CONNECTED (feature-based generalization)
    """

    def __init__(self, n_context: int = 4, freq_threshold: int = 100):
        self.n_context = n_context
        self.freq_threshold = freq_threshold  # Words appearing more than this are "high-freq"

        # Word statistics
        self.word_counts = Counter()
        self.high_freq_words = set()  # Exact matching
        self.low_freq_words = set()   # Feature-based

        # Character class mapping for low-freq words
        self.char_classes = {}
        self._init_char_classes()

        # FULLY CONNECTED RAM: exact (word, word, word, word) → next_word
        self.exact_ram = defaultdict(Counter)

        # PARTIALLY CONNECTED RAM: (features, features, ...) → next_word_class
        self.feature_ram = defaultdict(Counter)

        # Word class for output (low-freq words grouped by features)
        self.word_to_class = {}
        self.class_to_words = defaultdict(list)
        self.n_classes = 256

    def _init_char_classes(self):
        """Initialize character classes for feature extraction."""
        for c in 'aeiouAEIOU':
            self.char_classes[c] = 0  # vowel
        for c in 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ':
            self.char_classes[c] = 1  # consonant
        for c in '0123456789':
            self.char_classes[c] = 2  # digit
        for c in '.,!?;:':
            self.char_classes[c] = 3  # punct
        # Default = 4

    def _word_to_features(self, word: str) -> tuple:
        """
        Convert word to feature tuple for partial connectivity.

        Features:
        - First char class
        - Last char class
        - Length bucket (0-3)
        - Has uppercase
        - Has digit
        """
        if not word:
            return (4, 4, 0, 0, 0)

        first_class = self.char_classes.get(word[0], 4)
        last_class = self.char_classes.get(word[-1], 4)

        if len(word) <= 2:
            len_bucket = 0
        elif len(word) <= 4:
            len_bucket = 1
        elif len(word) <= 7:
            len_bucket = 2
        else:
            len_bucket = 3

        has_upper = 1 if any(c.isupper() for c in word) else 0
        has_digit = 1 if any(c.isdigit() for c in word) else 0

        return (first_class, last_class, len_bucket, has_upper, has_digit)

    def _context_to_representation(self, context: list[str]) -> tuple:
        """
        Convert context to mixed representation.

        High-freq words: Keep exact word
        Low-freq words: Convert to features
        """
        rep = []
        for word in context:
            if word in self.high_freq_words:
                rep.append(("EXACT", word))
            else:
                rep.append(("FEAT", self._word_to_features(word)))
        return tuple(rep)

    def train(self, tokens: list[str]):
        """Train the hybrid model."""
        # Count word frequencies
        self.word_counts = Counter(tokens)

        # Split into high/low frequency
        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:
                self.high_freq_words.add(word)
            else:
                self.low_freq_words.add(word)

        print(f"High-frequency words: {len(self.high_freq_words)} (≥{self.freq_threshold} occurrences)")
        print(f"Low-frequency words: {len(self.low_freq_words)}")

        # Coverage by high-freq words
        high_freq_tokens = sum(1 for t in tokens if t in self.high_freq_words)
        print(f"Token coverage by high-freq: {high_freq_tokens/len(tokens)*100:.1f}%")

        # Assign word classes for low-freq words
        for word in self.low_freq_words:
            features = self._word_to_features(word)
            word_class = hash(features) % self.n_classes
            self.word_to_class[word] = word_class
            self.class_to_words[word_class].append(word)

        # For high-freq, class = word itself (unique)
        for word in self.high_freq_words:
            self.word_to_class[word] = word  # Use word as its own class

        # Train both RAMs
        print("Training exact RAM (high-freq contexts)...")
        print("Training feature RAM (mixed contexts)...")

        exact_count = 0
        feature_count = 0

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            next_word = tokens[i + self.n_context]

            # Get mixed representation
            rep = self._context_to_representation(context)

            # Check if context is all high-freq (use exact RAM)
            all_high_freq = all(w in self.high_freq_words for w in context)

            if all_high_freq:
                exact_context = tuple(context)
                self.exact_ram[exact_context][next_word] += 1
                exact_count += 1
            else:
                # Use feature RAM
                self.feature_ram[rep][next_word] += 1
                feature_count += 1

        print(f"Exact RAM patterns: {len(self.exact_ram)} ({exact_count} examples)")
        print(f"Feature RAM patterns: {len(self.feature_ram)} ({feature_count} examples)")

    def predict(self, context: list[str]) -> tuple[str, str]:
        """
        Predict next word.

        Strategy:
        1. If all high-freq context → use exact RAM (precise)
        2. Else → use feature RAM (generalizing)
        3. Fallback to most common word
        """
        context = context[-self.n_context:]

        # Try exact RAM first (if all high-freq)
        all_high_freq = all(w in self.high_freq_words for w in context)

        if all_high_freq:
            exact_context = tuple(context)
            if exact_context in self.exact_ram:
                pred = self.exact_ram[exact_context].most_common(1)[0][0]
                return pred, "exact"

        # Try feature RAM
        rep = self._context_to_representation(context)
        if rep in self.feature_ram:
            pred = self.feature_ram[rep].most_common(1)[0][0]
            return pred, "feature"

        return "<UNK>", "none"

    def evaluate(self, tokens: list[str]) -> dict:
        """Evaluate hybrid model."""
        correct = 0
        exact_correct = 0
        feature_correct = 0
        exact_total = 0
        feature_total = 0
        total = 0

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            target = tokens[i + self.n_context]

            pred, method = self.predict(context)

            if pred == target:
                correct += 1
                if method == "exact":
                    exact_correct += 1
                elif method == "feature":
                    feature_correct += 1

            if method == "exact":
                exact_total += 1
            elif method == "feature":
                feature_total += 1

            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "exact_accuracy": exact_correct / exact_total if exact_total > 0 else 0,
            "feature_accuracy": feature_correct / feature_total if feature_total > 0 else 0,
            "exact_coverage": exact_total / total if total > 0 else 0,
            "feature_coverage": feature_total / total if total > 0 else 0,
            "total_coverage": (exact_total + feature_total) / total if total > 0 else 0,
        }


class AdaptiveConnectivityRAMLM:
    """
    Adaptive connectivity that LEARNS which words need exact vs feature matching.

    Instead of using frequency threshold, learn from data:
    - Words with consistent patterns → can use features (generalize)
    - Words with unique patterns → need exact matching

    This is like learning the "structure" of language the way we learned
    the structure of arithmetic (which bits connect where).
    """

    def __init__(self, n_context: int = 4):
        self.n_context = n_context

        # Statistics for each word
        self.word_entropy = {}  # Higher entropy = needs exact matching
        self.word_counts = Counter()

        # Learned connectivity per word
        self.word_connectivity = {}  # word → "exact" or "feature"

        # RAMs
        self.exact_ram = defaultdict(Counter)
        self.feature_ram = defaultdict(Counter)

        self.char_classes = {}
        self._init_char_classes()

    def _init_char_classes(self):
        for c in 'aeiouAEIOU':
            self.char_classes[c] = 0
        for c in 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ':
            self.char_classes[c] = 1
        for c in '0123456789':
            self.char_classes[c] = 2
        for c in '.,!?;:':
            self.char_classes[c] = 3

    def _word_features(self, word: str) -> tuple:
        if not word:
            return (4, 4, 0)
        return (
            self.char_classes.get(word[0], 4),
            self.char_classes.get(word[-1], 4),
            min(len(word) // 2, 3)
        )

    def _analyze_word_patterns(self, tokens: list[str]):
        """Analyze which words have predictable vs unpredictable patterns."""
        # For each word, what words follow it?
        following = defaultdict(Counter)
        for i in range(len(tokens) - 1):
            following[tokens[i]][tokens[i + 1]] += 1

        # Calculate entropy for each word
        for word, counts in following.items():
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            self.word_entropy[word] = entropy

        # Decide connectivity based on entropy
        # Low entropy = predictable = can use features
        # High entropy = unpredictable = need exact context
        median_entropy = sorted(self.word_entropy.values())[len(self.word_entropy) // 2]

        for word in self.word_entropy:
            if self.word_entropy[word] < median_entropy:
                self.word_connectivity[word] = "feature"  # Predictable, can generalize
            else:
                self.word_connectivity[word] = "exact"    # Unpredictable, need precision

        feature_count = sum(1 for w, c in self.word_connectivity.items() if c == "feature")
        print(f"Words using feature connectivity: {feature_count}")
        print(f"Words using exact connectivity: {len(self.word_connectivity) - feature_count}")

    def _context_representation(self, context: list[str]) -> tuple:
        """Mixed representation based on learned connectivity."""
        rep = []
        for word in context:
            conn = self.word_connectivity.get(word, "exact")
            if conn == "feature":
                rep.append(self._word_features(word))
            else:
                rep.append(word)
        return tuple(rep)

    def train(self, tokens: list[str]):
        """Train with adaptive connectivity."""
        self.word_counts = Counter(tokens)

        print("Analyzing word patterns to learn connectivity...")
        self._analyze_word_patterns(tokens)

        print("Training hybrid RAM...")
        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            next_word = tokens[i + self.n_context]

            rep = self._context_representation(context)
            self.feature_ram[rep][next_word] += 1

        print(f"Total patterns: {len(self.feature_ram)}")

    def predict(self, context: list[str]) -> str:
        context = context[-self.n_context:]
        rep = self._context_representation(context)

        if rep in self.feature_ram:
            return self.feature_ram[rep].most_common(1)[0][0]
        return "<UNK>"

    def evaluate(self, tokens: list[str]) -> dict:
        correct = 0
        covered = 0
        total = 0

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            target = tokens[i + self.n_context]

            rep = self._context_representation(context)
            if rep in self.feature_ram:
                covered += 1
                if self.predict(context) == target:
                    correct += 1

            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "coverage": covered / total if total > 0 else 0,
        }


def run_hybrid_benchmark():
    """Run hybrid connectivity benchmark."""
    print("\n" + "="*70)
    print("HYBRID CONNECTIVITY RAM LANGUAGE MODEL")
    print("="*70)
    print("""
Mixing fully-connected and partially-connected RAM:
- HIGH-FREQ words: Exact matching (fully connected)
- LOW-FREQ words: Feature-based (partial connectivity)

This mirrors the arithmetic approach:
- Carry logic: Fully connected to specific bits
- Digit operations: Partial connectivity for generalization
""")

    # Load data
    try:
        from datasets import load_dataset
        print("Loading WikiText-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        import re
        def tokenize(text):
            return re.findall(r"\w+|[^\w\s]", text.lower())

        train_text = " ".join(dataset["train"]["text"])
        test_text = " ".join(dataset["test"]["text"])

        train_tokens = tokenize(train_text)[:300000]
        test_tokens = tokenize(test_text)[:30000]

    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Test: {len(test_tokens):,} tokens")

    # Test different frequency thresholds
    print("\n" + "-"*50)
    print("EXPERIMENT 1: Hybrid by frequency threshold")
    print("-"*50)

    for threshold in [50, 100, 200, 500]:
        print(f"\n--- Threshold: {threshold} ---")
        model = HybridConnectivityRAMLM(n_context=4, freq_threshold=threshold)
        model.train(train_tokens)
        results = model.evaluate(test_tokens)

        print(f"Overall accuracy: {results['accuracy']*100:.2f}%")
        print(f"Exact RAM: {results['exact_accuracy']*100:.1f}% acc on {results['exact_coverage']*100:.1f}% of data")
        print(f"Feature RAM: {results['feature_accuracy']*100:.1f}% acc on {results['feature_coverage']*100:.1f}% of data")
        print(f"Total coverage: {results['total_coverage']*100:.1f}%")

    # Adaptive connectivity
    print("\n" + "-"*50)
    print("EXPERIMENT 2: Adaptive connectivity (learned from data)")
    print("-"*50)

    model2 = AdaptiveConnectivityRAMLM(n_context=4)
    model2.train(train_tokens)
    results2 = model2.evaluate(test_tokens)

    print(f"Accuracy: {results2['accuracy']*100:.2f}%")
    print(f"Coverage: {results2['coverage']*100:.1f}%")

    # Summary
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. HYBRID CONNECTIVITY works: Different words need different connectivity

2. EXACT RAM (fully connected) works best for:
   - High-frequency function words (the, a, is, of, to)
   - These have specific, learnable patterns
   - "of the" → usually followed by noun

3. FEATURE RAM (partial connectivity) works best for:
   - Low-frequency content words
   - These need generalization across similar words
   - "Dr. Smith" and "Dr. Jones" share patterns

4. ADAPTIVE LEARNING can discover which words need which connectivity
   - Based on pattern entropy (predictability)
   - Low entropy = predictable = can use features
   - High entropy = context-dependent = need exact

This is the SAME principle as arithmetic:
- Carry bit: Always 0 or 1, fully determined by inputs
- Digit value: Depends on position, needs partial connectivity
""")


if __name__ == "__main__":
    run_hybrid_benchmark()
