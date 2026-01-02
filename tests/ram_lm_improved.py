#!/usr/bin/env python3
"""
Improved Pure RAM Language Models

Exploring techniques to improve RAM language modeling WITHOUT weighted networks:

1. HIERARCHICAL RAM - Like arithmetic decomposition:
   - Char → char_class (learns "a,e,i,o,u are vowels")
   - Word → word_class (learns "run, walk, jump are verbs")
   - Context → prediction using classes

2. CO-OCCURRENCE CONNECTIVITY - Learn which tokens predict which:
   - Tokens appearing in similar contexts → similar connectivity
   - This is distributional semantics via RAM!

3. MULTI-SCALE ENSEMBLE - Different context windows vote:
   - Short-range: Recent 2 tokens (local syntax)
   - Medium-range: 4-6 tokens (phrase structure)
   - Long-range: 8+ tokens (topic/theme)

4. SELECTIVE CONNECTIVITY - Learn which positions matter:
   - Position 1 might be more predictive than position 4
   - Learn this from data, not hand-engineered

Key insight: We need to discover structure in language the same way
we discovered structure in arithmetic (carry chains, bit positions).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'wnn'))

from collections import defaultdict, Counter
import math
import random
from typing import Optional
import time


# =============================================================================
# 1. HIERARCHICAL RAM (like arithmetic decomposition)
# =============================================================================

class HierarchicalRAMLM:
    """
    Hierarchical RAM language model.

    Level 1: Character patterns → character class (8 classes)
    Level 2: Word (as char classes) → word class (64 classes)
    Level 3: Context (as word classes) → output word class

    This is the same decomposition pattern as arithmetic:
    - Arithmetic: bit patterns → carry → result
    - Language: char patterns → word class → prediction
    """

    def __init__(self, n_context: int = 4):
        self.n_context = n_context

        # Level 1: Char → char_class (vowel, consonant, punct, digit, etc.)
        self.char_classes = 8
        self.char_to_class = {}

        # Level 2: Word → word_class
        self.word_classes = 64
        self.word_to_class = {}
        self.class_to_words = defaultdict(list)

        # Level 3: Context classes → output class
        self.context_ram = defaultdict(Counter)  # (class_tuple) → {out_class: count}

        # Vocabulary
        self.vocab = set()

    def _init_char_classes(self):
        """Initialize character classes (hand-crafted linguistic knowledge)."""
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        digits = set('0123456789')
        punct = set('.,!?;:\'"()-[]{}')
        space = set(' \t\n')

        for c in vowels:
            self.char_to_class[c] = 0
        for c in consonants:
            self.char_to_class[c] = 1
        for c in digits:
            self.char_to_class[c] = 2
        for c in punct:
            self.char_to_class[c] = 3
        for c in space:
            self.char_to_class[c] = 4
        # Default for unknown
        self.default_char_class = 5

    def _word_to_char_pattern(self, word: str) -> tuple:
        """Convert word to character class pattern."""
        if len(word) == 0:
            return (5,)  # Unknown

        # Use first 4 chars + last char + length bucket
        chars = word[:4] + (word[-1] if len(word) > 4 else '')
        pattern = tuple(self.char_to_class.get(c, self.default_char_class) for c in chars)

        # Add length class
        if len(word) <= 2:
            len_class = 0
        elif len(word) <= 4:
            len_class = 1
        elif len(word) <= 7:
            len_class = 2
        else:
            len_class = 3

        return pattern + (len_class,)

    def _learn_word_classes(self, tokens: list[str]):
        """Learn word classes from co-occurrence patterns."""
        # Group words by their character patterns
        char_pattern_to_words = defaultdict(list)
        for word in set(tokens):
            pattern = self._word_to_char_pattern(word)
            char_pattern_to_words[pattern].append(word)

        # Assign word classes based on patterns
        # Words with same char pattern → same class (generalization!)
        pattern_to_class = {}
        class_id = 0
        for pattern in char_pattern_to_words:
            pattern_to_class[pattern] = class_id % self.word_classes
            class_id += 1

        for word in set(tokens):
            pattern = self._word_to_char_pattern(word)
            word_class = pattern_to_class[pattern]
            self.word_to_class[word] = word_class
            self.class_to_words[word_class].append(word)

    def train(self, tokens: list[str]):
        """Train the hierarchical model."""
        self._init_char_classes()
        self.vocab = set(tokens)

        print("Learning word classes from character patterns...")
        self._learn_word_classes(tokens)

        unique_patterns = len(set(self._word_to_char_pattern(w) for w in self.vocab))
        print(f"Vocabulary: {len(self.vocab)} words → {unique_patterns} char patterns → {self.word_classes} word classes")

        # Train context → output mapping
        print("Training context RAM...")
        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            next_word = tokens[i + self.n_context]

            # Convert to classes
            context_classes = tuple(self.word_to_class.get(w, 0) for w in context)
            output_class = self.word_to_class.get(next_word, 0)

            self.context_ram[context_classes][output_class] += 1

        print(f"Context patterns: {len(self.context_ram)}")

    def predict(self, context: list[str]) -> tuple[str, int]:
        """Predict next word given context."""
        context_classes = tuple(self.word_to_class.get(w, 0) for w in context[-self.n_context:])

        if context_classes not in self.context_ram:
            return "<UNK>", 0

        # Get most likely output class
        output_class = self.context_ram[context_classes].most_common(1)[0][0]

        # Get most common word in that class
        if self.class_to_words[output_class]:
            # Pick the most frequent word in training for this class
            return self.class_to_words[output_class][0], output_class
        return "<UNK>", output_class

    def evaluate(self, tokens: list[str]) -> dict:
        """Evaluate on test data."""
        correct_word = 0
        correct_class = 0
        covered = 0
        total = 0

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            target = tokens[i + self.n_context]
            target_class = self.word_to_class.get(target, 0)

            context_classes = tuple(self.word_to_class.get(w, 0) for w in context)

            if context_classes in self.context_ram:
                covered += 1
                pred_word, pred_class = self.predict(context)

                if pred_word == target:
                    correct_word += 1
                if pred_class == target_class:
                    correct_class += 1

            total += 1

        return {
            "word_accuracy": correct_word / total if total > 0 else 0,
            "class_accuracy": correct_class / total if total > 0 else 0,
            "coverage": covered / total if total > 0 else 0,
        }


# =============================================================================
# 2. CO-OCCURRENCE LEARNED CONNECTIVITY
# =============================================================================

class CooccurrenceRAMLM:
    """
    Learn word representations from co-occurrence patterns.

    Key insight: Words appearing in similar contexts should have similar
    "addresses" in RAM. This is distributional semantics!

    Process:
    1. Build co-occurrence matrix: word × context_word
    2. Cluster words by co-occurrence patterns (similar context → similar cluster)
    3. Use cluster IDs as the "address bits" for RAM lookup
    """

    def __init__(self, n_context: int = 4, n_clusters: int = 128):
        self.n_context = n_context
        self.n_clusters = n_clusters

        # Co-occurrence statistics
        self.cooccurrence = defaultdict(Counter)  # word → {context_word: count}

        # Learned word clusters
        self.word_to_cluster = {}
        self.cluster_to_words = defaultdict(list)

        # Prediction RAM: cluster_context → output_cluster
        self.context_ram = defaultdict(Counter)

    def _build_cooccurrence(self, tokens: list[str], window: int = 5):
        """Build co-occurrence matrix."""
        for i, word in enumerate(tokens):
            # Look at surrounding words
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)

            for j in range(start, end):
                if i != j:
                    context_word = tokens[j]
                    self.cooccurrence[word][context_word] += 1

    def _cluster_by_cooccurrence(self, tokens: list[str]):
        """Cluster words by their co-occurrence patterns."""
        vocab = list(set(tokens))

        # Get top-k context words for each word (signature)
        word_signatures = {}
        for word in vocab:
            if word in self.cooccurrence:
                top_contexts = self.cooccurrence[word].most_common(10)
                # Signature = set of top context words
                word_signatures[word] = frozenset(c for c, _ in top_contexts)
            else:
                word_signatures[word] = frozenset()

        # Group words by similar signatures
        signature_to_words = defaultdict(list)
        for word, sig in word_signatures.items():
            # Hash signature to cluster (simple approach)
            cluster = hash(sig) % self.n_clusters
            signature_to_words[cluster].append(word)

        # Assign clusters
        for cluster, words in signature_to_words.items():
            for word in words:
                self.word_to_cluster[word] = cluster
                self.cluster_to_words[cluster].append(word)

    def train(self, tokens: list[str]):
        """Train the co-occurrence model."""
        print("Building co-occurrence matrix...")
        self._build_cooccurrence(tokens)

        print("Clustering words by context similarity...")
        self._cluster_by_cooccurrence(tokens)

        print(f"Vocabulary: {len(set(tokens))} words → {self.n_clusters} clusters")

        # Count words per cluster
        cluster_sizes = [len(self.cluster_to_words[c]) for c in range(self.n_clusters)]
        print(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={sum(cluster_sizes)/len(cluster_sizes):.1f}")

        # Train context → output mapping
        print("Training context RAM...")
        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            next_word = tokens[i + self.n_context]

            context_clusters = tuple(self.word_to_cluster.get(w, 0) for w in context)
            output_cluster = self.word_to_cluster.get(next_word, 0)

            self.context_ram[context_clusters][output_cluster] += 1

        print(f"Context patterns: {len(self.context_ram)}")

    def predict(self, context: list[str]) -> tuple[str, int]:
        """Predict next word."""
        context_clusters = tuple(self.word_to_cluster.get(w, 0) for w in context[-self.n_context:])

        if context_clusters not in self.context_ram:
            return "<UNK>", 0

        output_cluster = self.context_ram[context_clusters].most_common(1)[0][0]

        if self.cluster_to_words[output_cluster]:
            return self.cluster_to_words[output_cluster][0], output_cluster
        return "<UNK>", output_cluster

    def evaluate(self, tokens: list[str]) -> dict:
        """Evaluate on test data."""
        correct_word = 0
        correct_cluster = 0
        covered = 0
        total = 0

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            target = tokens[i + self.n_context]
            target_cluster = self.word_to_cluster.get(target, 0)

            context_clusters = tuple(self.word_to_cluster.get(w, 0) for w in context)

            if context_clusters in self.context_ram:
                covered += 1
                pred_word, pred_cluster = self.predict(context)

                if pred_word == target:
                    correct_word += 1
                if pred_cluster == target_cluster:
                    correct_cluster += 1

            total += 1

        return {
            "word_accuracy": correct_word / total if total > 0 else 0,
            "cluster_accuracy": correct_cluster / total if total > 0 else 0,
            "coverage": covered / total if total > 0 else 0,
        }


# =============================================================================
# 3. MULTI-SCALE ENSEMBLE
# =============================================================================

class MultiScaleRAMLM:
    """
    Ensemble of RAMs at different scales.

    - Short-range RAM (n=2): Local syntax patterns
    - Medium-range RAM (n=4): Phrase structure
    - Long-range RAM (n=8): Topic/theme

    Vote across scales for final prediction.
    Different scales capture different linguistic phenomena.
    """

    def __init__(self, scales: list[int] = [2, 4, 6, 8]):
        self.scales = scales
        self.scale_rams = {}  # scale → {context_tuple → Counter(next_word)}
        self.scale_weights = {}  # Learned weights for each scale

        self.vocab = set()
        self.word_to_id = {}

    def train(self, tokens: list[str]):
        """Train RAMs at each scale."""
        self.vocab = set(tokens)
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}

        for n in self.scales:
            print(f"Training scale n={n}...")
            self.scale_rams[n] = defaultdict(Counter)

            for i in range(len(tokens) - n):
                context = tuple(tokens[i:i + n])
                next_word = tokens[i + n]
                self.scale_rams[n][context][next_word] += 1

            print(f"  Patterns: {len(self.scale_rams[n])}")

            # Weight by average context entropy (more certain = higher weight)
            total_entropy = 0
            for context, counts in self.scale_rams[n].items():
                total = sum(counts.values())
                probs = [c / total for c in counts.values()]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                total_entropy += entropy

            avg_entropy = total_entropy / len(self.scale_rams[n]) if self.scale_rams[n] else 10
            # Lower entropy = more confident = higher weight
            self.scale_weights[n] = 1.0 / (avg_entropy + 0.1)

        # Normalize weights
        total_weight = sum(self.scale_weights.values())
        for n in self.scales:
            self.scale_weights[n] /= total_weight
            print(f"Scale n={n}: weight={self.scale_weights[n]:.3f}")

    def predict(self, context: list[str]) -> tuple[str, dict]:
        """Predict with voting across scales."""
        votes = Counter()

        for n in self.scales:
            if len(context) >= n:
                ctx = tuple(context[-n:])
                if ctx in self.scale_rams[n]:
                    counts = self.scale_rams[n][ctx]
                    weight = self.scale_weights[n]

                    for word, count in counts.most_common(3):
                        votes[word] += count * weight

        if votes:
            best = votes.most_common(1)[0][0]
            return best, dict(votes.most_common(5))
        return "<UNK>", {}

    def evaluate(self, tokens: list[str]) -> dict:
        """Evaluate with multi-scale voting."""
        correct = 0
        covered = 0
        total = 0

        max_n = max(self.scales)

        for i in range(len(tokens) - max_n):
            context = tokens[i:i + max_n]
            target = tokens[i + max_n]

            pred, votes = self.predict(context)

            if votes:  # At least one scale had a prediction
                covered += 1
                if pred == target:
                    correct += 1

            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "coverage": covered / total if total > 0 else 0,
        }


# =============================================================================
# 4. SELECTIVE CONNECTIVITY (learn which positions matter)
# =============================================================================

class SelectiveConnectivityRAMLM:
    """
    Learn which context positions are most predictive.

    Instead of using all n positions, learn which subset matters:
    - Some positions may be noise
    - Different output classes may use different positions

    Process:
    1. Train full RAM
    2. Measure predictive power of each position
    3. Select top-k positions per output class
    4. Retrain with selective connectivity
    """

    def __init__(self, n_context: int = 6, n_select: int = 3):
        self.n_context = n_context
        self.n_select = n_select  # How many positions to actually use

        # Full RAM for analysis
        self.full_ram = defaultdict(Counter)

        # Position importance per output word
        self.position_importance = defaultdict(lambda: [0.0] * n_context)

        # Selective RAM: uses only important positions
        self.selective_ram = defaultdict(Counter)
        self.selected_positions = {}  # output_class → list of position indices

        self.vocab = set()
        self.word_to_class = {}
        self.n_classes = 64

    def _analyze_position_importance(self, tokens: list[str]):
        """Analyze which positions are most predictive for each output."""
        # For each output word, measure how much each position reduces uncertainty
        position_counts = defaultdict(lambda: [Counter() for _ in range(self.n_context)])

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            next_word = tokens[i + self.n_context]
            output_class = self.word_to_class.get(next_word, 0)

            for pos, word in enumerate(context):
                position_counts[output_class][pos][word] += 1

        # Calculate importance as entropy reduction
        for output_class in position_counts:
            for pos in range(self.n_context):
                counts = position_counts[output_class][pos]
                total = sum(counts.values())
                if total > 0:
                    # High concentration = high importance
                    top_count = counts.most_common(1)[0][1] if counts else 0
                    concentration = top_count / total
                    self.position_importance[output_class][pos] = concentration

    def train(self, tokens: list[str]):
        """Train with selective connectivity."""
        self.vocab = set(tokens)

        # Assign word classes
        for i, word in enumerate(self.vocab):
            self.word_to_class[word] = hash(word) % self.n_classes

        print("Analyzing position importance...")
        self._analyze_position_importance(tokens)

        # Select top positions per output class
        for output_class in range(self.n_classes):
            importance = self.position_importance[output_class]
            # Get indices of top-k positions
            sorted_positions = sorted(range(self.n_context),
                                      key=lambda p: importance[p],
                                      reverse=True)
            self.selected_positions[output_class] = sorted_positions[:self.n_select]

        # Show position selection stats
        position_usage = Counter()
        for positions in self.selected_positions.values():
            for p in positions:
                position_usage[p] += 1
        print(f"Position usage across classes: {dict(position_usage)}")

        # Train selective RAM
        print("Training selective RAM...")
        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            next_word = tokens[i + self.n_context]
            output_class = self.word_to_class.get(next_word, 0)

            # Use only selected positions for this output class
            positions = self.selected_positions[output_class]
            selective_context = tuple(context[p] for p in positions)

            self.selective_ram[selective_context][next_word] += 1

        print(f"Selective patterns: {len(self.selective_ram)}")

        # Compare with full RAM
        full_patterns = len(set(tuple(tokens[i:i+self.n_context])
                               for i in range(len(tokens) - self.n_context)))
        compression = full_patterns / len(self.selective_ram) if self.selective_ram else 0
        print(f"Full patterns: {full_patterns}, Compression: {compression:.1f}x")

    def predict(self, context: list[str]) -> str:
        """Predict using selective connectivity."""
        # Try each output class's selected positions
        best_word = "<UNK>"
        best_count = 0

        for output_class in range(self.n_classes):
            positions = self.selected_positions[output_class]
            if len(context) >= self.n_context:
                ctx = context[-self.n_context:]
                selective_context = tuple(ctx[p] for p in positions)

                if selective_context in self.selective_ram:
                    counts = self.selective_ram[selective_context]
                    top_word, count = counts.most_common(1)[0]
                    if count > best_count:
                        best_count = count
                        best_word = top_word

        return best_word

    def evaluate(self, tokens: list[str]) -> dict:
        """Evaluate selective connectivity model."""
        correct = 0
        total = 0

        for i in range(len(tokens) - self.n_context):
            context = tokens[i:i + self.n_context]
            target = tokens[i + self.n_context]

            pred = self.predict(context)
            if pred == target:
                correct += 1
            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
        }


# =============================================================================
# BENCHMARK
# =============================================================================

def run_improved_benchmark():
    """Compare all improved RAM approaches."""
    print("\n" + "="*70)
    print("IMPROVED PURE RAM LANGUAGE MODELS")
    print("="*70)
    print("""
Testing four approaches to improve RAM language modeling:
1. Hierarchical RAM (char → word → prediction)
2. Co-occurrence clustering (distributional semantics)
3. Multi-scale ensemble (different context lengths vote)
4. Selective connectivity (learn which positions matter)
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

        train_tokens = tokenize(train_text)[:200000]  # Subset for speed
        test_tokens = tokenize(test_text)[:20000]

    except Exception as e:
        print(f"Error: {e}")
        train_tokens = "the cat sat on the mat and the dog sat on the log".split() * 5000
        test_tokens = "the cat sat on the log and the dog sat on the mat".split() * 500

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Test: {len(test_tokens):,} tokens")

    results = {}

    # 1. Hierarchical RAM
    print("\n" + "-"*50)
    print("1. HIERARCHICAL RAM (char patterns → word classes)")
    print("-"*50)
    model1 = HierarchicalRAMLM(n_context=4)
    model1.train(train_tokens)
    r1 = model1.evaluate(test_tokens)
    print(f"Word accuracy: {r1['word_accuracy']*100:.2f}%")
    print(f"Class accuracy: {r1['class_accuracy']*100:.2f}%")
    print(f"Coverage: {r1['coverage']*100:.1f}%")
    results["hierarchical"] = r1

    # 2. Co-occurrence clustering
    print("\n" + "-"*50)
    print("2. CO-OCCURRENCE CLUSTERING (distributional semantics)")
    print("-"*50)
    model2 = CooccurrenceRAMLM(n_context=4, n_clusters=128)
    model2.train(train_tokens)
    r2 = model2.evaluate(test_tokens)
    print(f"Word accuracy: {r2['word_accuracy']*100:.2f}%")
    print(f"Cluster accuracy: {r2['cluster_accuracy']*100:.2f}%")
    print(f"Coverage: {r2['coverage']*100:.1f}%")
    results["cooccurrence"] = r2

    # 3. Multi-scale ensemble
    print("\n" + "-"*50)
    print("3. MULTI-SCALE ENSEMBLE (vote across context lengths)")
    print("-"*50)
    model3 = MultiScaleRAMLM(scales=[2, 3, 4, 5, 6])
    model3.train(train_tokens)
    r3 = model3.evaluate(test_tokens)
    print(f"Accuracy: {r3['accuracy']*100:.2f}%")
    print(f"Coverage: {r3['coverage']*100:.1f}%")
    results["multiscale"] = r3

    # 4. Selective connectivity
    print("\n" + "-"*50)
    print("4. SELECTIVE CONNECTIVITY (learn important positions)")
    print("-"*50)
    model4 = SelectiveConnectivityRAMLM(n_context=6, n_select=3)
    model4.train(train_tokens)
    r4 = model4.evaluate(test_tokens)
    print(f"Accuracy: {r4['accuracy']*100:.2f}%")
    results["selective"] = r4

    # Baseline: Pure n-gram
    print("\n" + "-"*50)
    print("BASELINE: Pure N-gram (n=4)")
    print("-"*50)
    ngram = defaultdict(Counter)
    for i in range(len(train_tokens) - 4):
        ctx = tuple(train_tokens[i:i+4])
        ngram[ctx][train_tokens[i+4]] += 1

    correct = covered = 0
    for i in range(len(test_tokens) - 4):
        ctx = tuple(test_tokens[i:i+4])
        target = test_tokens[i+4]
        if ctx in ngram:
            covered += 1
            if ngram[ctx].most_common(1)[0][0] == target:
                correct += 1

    total = len(test_tokens) - 4
    print(f"Accuracy: {correct/total*100:.2f}%")
    print(f"Coverage: {covered/total*100:.1f}%")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
| Approach | Coverage | Class/Word Acc | Key Insight |
|----------|----------|----------------|-------------|""")

    print(f"| Baseline n-gram | {covered/total*100:.1f}% | {correct/total*100:.1f}% | Pure memorization |")
    print(f"| Hierarchical | {r1['coverage']*100:.1f}% | {r1['class_accuracy']*100:.1f}% / {r1['word_accuracy']*100:.1f}% | Char→word decomposition |")
    print(f"| Co-occurrence | {r2['coverage']*100:.1f}% | {r2['cluster_accuracy']*100:.1f}% / {r2['word_accuracy']*100:.1f}% | Distributional semantics |")
    print(f"| Multi-scale | {r3['coverage']*100:.1f}% | {r3['accuracy']*100:.1f}% | Different ranges vote |")
    print(f"| Selective | - | {r4['accuracy']*100:.1f}% | Learn which positions matter |")

    return results


if __name__ == "__main__":
    run_improved_benchmark()
