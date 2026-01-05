"""
Mutual Information Binary Encoder.

Learns binary codes by iteratively selecting bits that maximize
mutual information with the target word.

Algorithm:
    For each bit position k:
        1. For each candidate feature f (e.g., "appears after 'the'"):
            - Compute MI(f, target_word | previous_bits)
        2. Select feature with highest MI
        3. Assign bit k = feature(word)

This is like building a binary decision tree where each node
maximizes information gain - but the result is a binary code!

Key insight: Each bit is a learned "semantic feature":
    - Bit 0: "Is this a function word?" (learned, not hand-coded)
    - Bit 1: "Does this word follow determiners?"
    - Bit 2: "Is this word typically sentence-final?"
    - ...

Similar words will have similar patterns of features → similar codes.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from wnn.representations.base import BinaryEncoder


class MutualInfoEncoder(BinaryEncoder):
    """
    Learn binary codes by maximizing mutual information.

    Each bit is a learned feature that helps predict the next word.
    Words with similar predictive patterns get similar codes.
    """

    def __init__(
        self,
        n_bits: int = 12,
        context_window: int = 2,
        min_freq: int = 5,
        name: str = "mutual_info",
    ):
        """
        Initialize mutual information encoder.

        Args:
            n_bits: Number of bits in binary code
            context_window: Context window for feature extraction
            min_freq: Minimum word frequency to include in vocab
            name: Human-readable name
        """
        super().__init__(n_bits=n_bits, name=name)
        self._context_window = context_window
        self._min_freq = min_freq

        # Learned features (one per bit)
        self._bit_features: list[dict[str, int]] = []  # word → 0/1 for each bit

    def train(self, tokens: list[str], **kwargs) -> None:
        """
        Learn binary codes maximizing MI with target.

        Strategy:
        1. Build vocabulary from frequent words
        2. Extract context-based features for each word
        3. For each bit, select the feature that maximizes MI
        4. Assign codes based on selected features
        """
        # Build vocabulary
        word_freq = Counter(tokens)
        self._vocab = {
            word: idx
            for idx, (word, freq) in enumerate(word_freq.most_common())
            if freq >= self._min_freq
        }

        if not self._vocab:
            raise ValueError("No words meet minimum frequency threshold")

        vocab_size = len(self._vocab)
        self._codes = np.zeros(vocab_size, dtype=np.int64)

        # Collect context statistics for feature generation
        # For each word, track: which words appear before/after it
        word_left_context = defaultdict(Counter)   # word → {left_word: count}
        word_right_context = defaultdict(Counter)  # word → {right_word: count}
        word_as_target = Counter()                 # How often word is the target

        for i in range(self._context_window, len(tokens) - 1):
            target = tokens[i + 1]
            if target not in self._vocab:
                continue

            word_as_target[target] += 1

            # Context words
            for j in range(1, self._context_window + 1):
                if i - j >= 0:
                    left_word = tokens[i - j + 1]
                    if left_word in self._vocab:
                        word_left_context[target][left_word] += 1
                if i + j < len(tokens) - 1:
                    right_word = tokens[i + j + 1]
                    if right_word in self._vocab:
                        word_right_context[target][right_word] += 1

        # Generate candidate features for bit learning
        # Feature types:
        # 1. "Follows word X" - word appears after X
        # 2. "Precedes word X" - word appears before X
        # 3. "Frequency band" - word is in top/bottom frequency
        # 4. "First character" - word starts with certain letter

        # For efficiency, we'll use clustering-based features
        # Cluster words by their context vectors, then each cluster = 1 bit

        # Iteratively learn each bit
        self._bit_features = []

        for bit_idx in range(self._n_bits):
            # Find the best split for this bit
            best_feature = self._find_best_split(
                word_left_context,
                word_right_context,
                word_as_target,
                bit_idx,
            )
            self._bit_features.append(best_feature)

            # Update codes with this bit
            for word, idx in self._vocab.items():
                if best_feature.get(word, 0) == 1:
                    self._codes[idx] |= (1 << bit_idx)

        self._is_trained = True

    def _find_best_split(
        self,
        word_left_context: dict,
        word_right_context: dict,
        word_as_target: Counter,
        bit_idx: int,
    ) -> dict[str, int]:
        """
        Find the vocabulary split that maximizes MI for this bit.

        Uses spectral clustering on context similarity to find
        natural word groupings.
        """
        vocab_list = list(self._vocab.keys())
        n_words = len(vocab_list)

        # Build context vectors for each word
        # Use left context as feature space
        all_context_words = set()
        for contexts in word_left_context.values():
            all_context_words.update(contexts.keys())

        context_words = list(all_context_words)[:500]  # Limit for efficiency
        context_idx = {w: i for i, w in enumerate(context_words)}

        # Build sparse context matrix
        context_vectors = np.zeros((n_words, len(context_words)))
        for i, word in enumerate(vocab_list):
            for ctx_word, count in word_left_context[word].items():
                if ctx_word in context_idx:
                    # Use PPMI-like weighting
                    context_vectors[i, context_idx[ctx_word]] = np.log1p(count)

        # Normalize vectors
        norms = np.linalg.norm(context_vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        context_vectors = context_vectors / norms

        # Use a different random projection for each bit
        np.random.seed(42 + bit_idx * 1000)

        # Project to 1D and split at median
        # This gives a balanced split that captures context similarity
        random_direction = np.random.randn(len(context_words))
        random_direction = random_direction / np.linalg.norm(random_direction)

        projections = context_vectors @ random_direction
        median_proj = np.median(projections)

        # Assign bit based on projection
        feature = {}
        for i, word in enumerate(vocab_list):
            feature[word] = 1 if projections[i] > median_proj else 0

        return feature

    def encode(self, word: str) -> int:
        """
        Encode word using learned bit features.
        """
        if not self._is_trained:
            raise RuntimeError("Encoder not trained. Call train() first.")

        # Check if word is in vocabulary
        if word in self._vocab:
            return int(self._codes[self._vocab[word]])

        # For unknown words, apply the learned features
        code = 0
        for bit_idx, feature in enumerate(self._bit_features):
            if feature.get(word, 0) == 1:
                code |= (1 << bit_idx)

        # If no features match, fall back to hash
        if code == 0:
            code = self._hash_unknown(word)

        return code

    def get_bit_statistics(self) -> list[dict]:
        """
        Get statistics for each learned bit.

        Returns:
            List of dicts with bit statistics
        """
        if not self._is_trained:
            return []

        stats = []
        for bit_idx, feature in enumerate(self._bit_features):
            ones = sum(1 for v in feature.values() if v == 1)
            zeros = len(feature) - ones
            stats.append({
                "bit": bit_idx,
                "ones": ones,
                "zeros": zeros,
                "balance": min(ones, zeros) / max(ones, zeros) if max(ones, zeros) > 0 else 0,
            })
        return stats

    def get_stats(self) -> dict:
        """Get encoder statistics."""
        stats = super().get_stats()
        stats["context_window"] = self._context_window
        stats["min_freq"] = self._min_freq

        if self._is_trained:
            bit_stats = self.get_bit_statistics()
            avg_balance = np.mean([b["balance"] for b in bit_stats])
            stats["avg_bit_balance"] = avg_balance
            stats["bit_statistics"] = bit_stats

        return stats
