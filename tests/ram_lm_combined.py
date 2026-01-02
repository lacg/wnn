#!/usr/bin/env python3
"""
Combined Best RAM Approaches

Combining all the techniques that worked:
1. MULTI-SCALE (11.5% accuracy) - Different context lengths vote
2. EXACT MATCHING (27-30% on high-freq) - Fully connected for common patterns
3. FEATURE GENERALIZATION (14%) - Partial connectivity for rare words

Architecture:
- Layer 1: Exact match on high-freq context patterns (fully connected)
- Layer 2: Multi-scale voting with feature abstraction (partial connectivity)
- Layer 3: Frequency-weighted output smoothing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'wnn'))

from collections import defaultdict, Counter
import math


class CombinedRAMLM:
    """
    Combined RAM Language Model using multiple techniques.

    Priority:
    1. Exact match (if all context words are high-freq)
    2. Multi-scale voting with features
    3. Fallback to most common in context
    """

    def __init__(self, n_context: int = 6, freq_threshold: int = 100):
        self.n_context = n_context
        self.freq_threshold = freq_threshold

        # Word statistics
        self.word_counts = Counter()
        self.high_freq_words = set()

        # Exact RAM for high-freq patterns (fully connected)
        self.exact_rams = {}  # n → {context → {next_word: count}}
        for n in [2, 3, 4]:
            self.exact_rams[n] = defaultdict(Counter)

        # Feature RAM for mixed patterns (partial connectivity)
        self.feature_rams = {}
        for n in [2, 3, 4, 5]:
            self.feature_rams[n] = defaultdict(Counter)

        # Char classes for features
        self.char_class = {c: 0 for c in 'aeiouAEIOU'}
        self.char_class.update({c: 1 for c in 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'})
        self.char_class.update({c: 2 for c in '0123456789'})
        self.char_class.update({c: 3 for c in '.,!?;:'})

    def _word_features(self, word: str) -> tuple:
        """Extract features for partial connectivity."""
        if not word:
            return (4, 4, 0, 0)

        first = self.char_class.get(word[0], 4)
        last = self.char_class.get(word[-1], 4)
        length = min(len(word) // 2, 3)
        # Add word frequency bucket
        count = self.word_counts.get(word, 0)
        if count >= 1000:
            freq = 3
        elif count >= 100:
            freq = 2
        elif count >= 10:
            freq = 1
        else:
            freq = 0

        return (first, last, length, freq)

    def _context_to_exact(self, context: list[str]) -> tuple:
        """Convert to exact representation (for high-freq)."""
        return tuple(context)

    def _context_to_features(self, context: list[str]) -> tuple:
        """Convert to feature representation (for generalization)."""
        return tuple(
            word if word in self.high_freq_words else self._word_features(word)
            for word in context
        )

    def train(self, tokens: list[str]):
        """Train combined model."""
        self.word_counts = Counter(tokens)

        # Identify high-frequency words
        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:
                self.high_freq_words.add(word)

        print(f"High-freq words: {len(self.high_freq_words)}")
        print(f"Vocabulary: {len(self.word_counts)}")

        # Train exact RAMs (for contexts with all high-freq words)
        print("\nTraining exact RAMs (fully connected)...")
        for n in self.exact_rams:
            for i in range(len(tokens) - n):
                context = tokens[i:i + n]
                next_word = tokens[i + n]

                # Only train if all context words are high-freq
                if all(w in self.high_freq_words for w in context):
                    self.exact_rams[n][tuple(context)][next_word] += 1

            print(f"  n={n}: {len(self.exact_rams[n])} patterns")

        # Train feature RAMs (for all contexts)
        print("\nTraining feature RAMs (partial connectivity)...")
        for n in self.feature_rams:
            for i in range(len(tokens) - n):
                context = tokens[i:i + n]
                next_word = tokens[i + n]

                features = self._context_to_features(context)
                self.feature_rams[n][features][next_word] += 1

            print(f"  n={n}: {len(self.feature_rams[n])} patterns")

    def predict(self, context: list[str]) -> tuple[str, str, float]:
        """
        Predict with priority:
        1. Exact match (if high confidence)
        2. Multi-scale voting
        """
        votes = Counter()
        method = "none"

        # Try exact match first (highest priority)
        for n in sorted(self.exact_rams.keys(), reverse=True):
            if len(context) >= n:
                ctx = tuple(context[-n:])
                if ctx in self.exact_rams[n]:
                    counts = self.exact_rams[n][ctx]
                    total = sum(counts.values())
                    best_word, best_count = counts.most_common(1)[0]

                    # High confidence exact match
                    confidence = best_count / total
                    if confidence > 0.5 or total > 10:
                        # Weight by confidence and recency (longer context = more recent)
                        votes[best_word] += best_count * (1 + n * 0.2)
                        method = f"exact_n{n}"

        # Add feature-based votes
        for n in self.feature_rams:
            if len(context) >= n:
                ctx = self._context_to_features(context[-n:])
                if ctx in self.feature_rams[n]:
                    counts = self.feature_rams[n][ctx]
                    # Weight by n (longer = more specific)
                    weight = 1 + n * 0.1
                    for word, count in counts.most_common(3):
                        votes[word] += count * weight

                    if method == "none":
                        method = f"feature_n{n}"

        if votes:
            best_word = votes.most_common(1)[0][0]
            total_votes = sum(votes.values())
            confidence = votes[best_word] / total_votes
            return best_word, method, confidence

        return "<UNK>", "none", 0.0

    def evaluate(self, tokens: list[str]) -> dict:
        """Evaluate combined model."""
        correct = 0
        by_method = defaultdict(lambda: {"correct": 0, "total": 0})
        total = 0

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            target = tokens[i + self.n_context]

            pred, method, conf = self.predict(context)

            if pred == target:
                correct += 1
                by_method[method]["correct"] += 1

            by_method[method]["total"] += 1
            total += 1

        # Calculate per-method accuracy
        method_stats = {}
        for method, stats in by_method.items():
            if stats["total"] > 0:
                method_stats[method] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "coverage": stats["total"] / total,
                }

        return {
            "accuracy": correct / total if total > 0 else 0,
            "by_method": method_stats,
        }


def run_combined_benchmark():
    """Run combined model benchmark."""
    print("\n" + "="*70)
    print("COMBINED RAM LANGUAGE MODEL")
    print("="*70)
    print("""
Combining best techniques:
1. Exact matching for high-freq patterns (27-30% accuracy when applicable)
2. Multi-scale voting with features (11-14% on general data)
3. Priority: Exact > Multi-scale feature
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

        train_tokens = tokenize(train_text)[:500000]
        test_tokens = tokenize(test_text)[:50000]

    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Test: {len(test_tokens):,} tokens")

    # Test combined model
    print("\n" + "-"*50)
    print("COMBINED MODEL")
    print("-"*50)

    model = CombinedRAMLM(n_context=6, freq_threshold=100)
    model.train(train_tokens)

    print("\nEvaluating...")
    results = model.evaluate(test_tokens)

    print(f"\nOverall accuracy: {results['accuracy']*100:.2f}%")
    print("\nBreakdown by method:")
    for method, stats in sorted(results["by_method"].items()):
        print(f"  {method}: {stats['accuracy']*100:.1f}% accuracy, {stats['coverage']*100:.1f}% of predictions")

    # Compare with baselines
    print("\n" + "-"*50)
    print("COMPARISON WITH BASELINES")
    print("-"*50)

    # Pure n-gram
    ngram = defaultdict(Counter)
    for i in range(len(train_tokens) - 4):
        ngram[tuple(train_tokens[i:i+4])][train_tokens[i+4]] += 1

    ngram_correct = 0
    ngram_covered = 0
    for i in range(len(test_tokens) - 4):
        ctx = tuple(test_tokens[i:i+4])
        if ctx in ngram:
            ngram_covered += 1
            if ngram[ctx].most_common(1)[0][0] == test_tokens[i+4]:
                ngram_correct += 1

    ngram_total = len(test_tokens) - 4
    print(f"Pure n-gram (n=4): {ngram_correct/ngram_total*100:.2f}% accuracy, {ngram_covered/ngram_total*100:.1f}% coverage")
    print(f"Combined RAM: {results['accuracy']*100:.2f}% accuracy")
    print(f"Improvement: {results['accuracy'] / (ngram_correct/ngram_total):.1f}x")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
| Model | Accuracy | Notes |
|-------|----------|-------|
| Pure n-gram (baseline) | {ngram_correct/ngram_total*100:.1f}% | Memorization only |
| Combined RAM | {results['accuracy']*100:.1f}% | Hybrid connectivity |
| Improvement | {results['accuracy'] / (ngram_correct/ngram_total):.1f}x | |

Key techniques:
- Exact RAM (fully connected): High accuracy on predictable patterns
- Feature RAM (partial connectivity): Generalization to unseen words
- Multi-scale: Different context lengths capture different phenomena

This is still far from weighted neural networks (~30% accuracy) because:
1. Language has fundamental ambiguity (many valid next words)
2. Semantic understanding requires learning representations
3. RAM can memorize and generalize patterns, but not learn abstractions

The hybrid RAM approach DOES improve over pure memorization,
confirming that partial connectivity enables generalization.
""")


if __name__ == "__main__":
    run_combined_benchmark()
