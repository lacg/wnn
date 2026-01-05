"""
RAM-based Binary Encoder.

Uses RAM networks to learn binary codes where each bit is a
learned classifier trained on context patterns.

Key insight: Each bit is a RAM that learns:
    context_pattern → 0/1

The RAM stores which context patterns should activate this bit.
Words appearing in similar contexts will have similar bits activated.

Architecture:
    Bit 0 RAM: context features → 0/1
    Bit 1 RAM: context features → 0/1
    ...
    Bit k RAM: context features → 0/1

Training:
    1. Extract context features for each word occurrence
    2. Cluster words by context similarity
    3. Assign cluster IDs as initial codes
    4. Train each bit's RAM to predict: context → bit value
    5. Refine codes based on prediction accuracy

This is the most "RAM-native" approach - using RAM to learn embeddings!
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from wnn.representations.base import BinaryEncoder


class RAMBinaryEncoder(BinaryEncoder):
    """
    RAM-based binary encoder.

    Uses RAM lookup tables to learn context → bit mappings.
    Each bit is essentially a RAM classifier.
    """

    def __init__(
        self,
        n_bits: int = 12,
        context_window: int = 2,
        min_freq: int = 5,
        n_context_features: int = 64,
        name: str = "ram_learned",
    ):
        """
        Initialize RAM binary encoder.

        Args:
            n_bits: Number of bits in binary code
            context_window: Context window for feature extraction
            min_freq: Minimum word frequency
            n_context_features: Number of context features per word
            name: Human-readable name
        """
        super().__init__(n_bits=n_bits, name=name)
        self._context_window = context_window
        self._min_freq = min_freq
        self._n_context_features = n_context_features

        # RAM for each bit: context_hash → Counter{0: count, 1: count}
        self._bit_rams: list[dict[int, Counter]] = []

        # Context feature extractors (learned word → feature mappings)
        self._context_word_to_feature: dict[str, int] = {}

        # Word context signatures for unknown word handling
        self._word_context_signature: dict[str, np.ndarray] = {}

    def train(self, tokens: list[str], **kwargs) -> None:
        """
        Learn binary codes using RAM networks.

        Algorithm:
        1. Build vocabulary and context features
        2. Extract context signatures for each word
        3. Cluster words by context similarity → initial codes
        4. Train RAM for each bit: context → bit value
        5. Refine codes based on RAM predictions
        """
        # Build vocabulary
        word_freq = Counter(tokens)
        vocab_words = [
            word for word, freq in word_freq.most_common()
            if freq >= self._min_freq
        ]
        self._vocab = {word: idx for idx, word in enumerate(vocab_words)}

        if len(self._vocab) < 2:
            raise ValueError("Vocabulary too small")

        vocab_size = len(self._vocab)

        # Assign context feature IDs to most common words
        # These become the "address bits" for context lookup
        top_context_words = [w for w, _ in word_freq.most_common(self._n_context_features)]
        self._context_word_to_feature = {w: i for i, w in enumerate(top_context_words)}

        # Collect context signatures for each word
        # signature[i] = how often word i appears in context
        word_context_counts = defaultdict(lambda: np.zeros(self._n_context_features))
        word_occurrence_count = Counter()

        for i, token in enumerate(tokens):
            if token not in self._vocab:
                continue

            word_occurrence_count[token] += 1

            # Collect context features
            for j in range(max(0, i - self._context_window), min(len(tokens), i + self._context_window + 1)):
                if i == j:
                    continue
                context_word = tokens[j]
                if context_word in self._context_word_to_feature:
                    feature_idx = self._context_word_to_feature[context_word]
                    word_context_counts[token][feature_idx] += 1

        # Normalize context signatures
        for word in self._vocab:
            count = word_occurrence_count[word]
            if count > 0:
                self._word_context_signature[word] = word_context_counts[word] / count
            else:
                self._word_context_signature[word] = np.zeros(self._n_context_features)

        # Initial codes from clustering
        # Use random projections on context signatures (like LSH)
        self._codes = np.zeros(vocab_size, dtype=np.int64)

        # Generate random projection vectors for each bit
        np.random.seed(42)
        projections = np.random.randn(self._n_bits, self._n_context_features)

        # Normalize projections
        for i in range(self._n_bits):
            projections[i] /= np.linalg.norm(projections[i])

        # Assign initial codes based on projections
        for word, idx in self._vocab.items():
            signature = self._word_context_signature[word]
            code = 0
            for bit in range(self._n_bits):
                if np.dot(projections[bit], signature) > 0:
                    code |= (1 << bit)
            self._codes[idx] = code

        # Train RAM for each bit
        # RAM learns: context_hash → predicted_bit
        self._bit_rams = [{} for _ in range(self._n_bits)]

        # Collect training data: (context_hash, target_word) pairs
        for i in range(self._context_window, len(tokens) - 1):
            target = tokens[i + 1]
            if target not in self._vocab:
                continue

            # Hash context to RAM address
            context_hash = self._hash_context(tokens, i)
            target_code = self._codes[self._vocab[target]]

            # Update each bit's RAM
            for bit in range(self._n_bits):
                bit_value = (target_code >> bit) & 1

                if context_hash not in self._bit_rams[bit]:
                    self._bit_rams[bit][context_hash] = Counter()
                self._bit_rams[bit][context_hash][bit_value] += 1

        # Refine codes based on RAM predictions
        # Words that consistently activate same RAMs should have same bits
        self._refine_codes(tokens)

        self._is_trained = True

    def _hash_context(self, tokens: list[str], position: int) -> int:
        """
        Hash context around position to RAM address.

        Uses feature indices of context words, combined with position.
        """
        h = 0
        for j in range(max(0, position - self._context_window), min(len(tokens), position + self._context_window + 1)):
            if j == position:
                continue
            word = tokens[j]
            if word in self._context_word_to_feature:
                feature = self._context_word_to_feature[word]
                # Combine with relative position
                rel_pos = j - position + self._context_window
                h ^= (feature * 31 + rel_pos) * (1 << ((j - position + 3) * 4))

        # Limit to reasonable address space
        return h % (1 << 16)

    def _refine_codes(self, tokens: list[str]) -> None:
        """
        Refine codes based on RAM prediction accuracy.

        For each word, predict its code from contexts and
        update if RAM predictions are more consistent.
        """
        word_bit_votes = defaultdict(lambda: [Counter() for _ in range(self._n_bits)])

        for i in range(self._context_window, len(tokens) - 1):
            target = tokens[i + 1]
            if target not in self._vocab:
                continue

            context_hash = self._hash_context(tokens, i)

            # Get RAM prediction for each bit
            for bit in range(self._n_bits):
                if context_hash in self._bit_rams[bit]:
                    votes = self._bit_rams[bit][context_hash]
                    if votes:
                        predicted_bit = votes.most_common(1)[0][0]
                        word_bit_votes[target][bit][predicted_bit] += 1

        # Update codes based on majority RAM predictions
        for word, idx in self._vocab.items():
            new_code = 0
            for bit in range(self._n_bits):
                votes = word_bit_votes[word][bit]
                if votes:
                    # Use RAM prediction if confident
                    total = sum(votes.values())
                    if total > 0:
                        predicted = votes.most_common(1)[0][0]
                        confidence = votes[predicted] / total
                        if confidence > 0.6:  # Only update if confident
                            if predicted == 1:
                                new_code |= (1 << bit)
                        else:
                            # Keep original bit
                            new_code |= (self._codes[idx] & (1 << bit))
                    else:
                        new_code |= (self._codes[idx] & (1 << bit))
                else:
                    # No RAM data, keep original
                    new_code |= (self._codes[idx] & (1 << bit))

            self._codes[idx] = new_code

    def encode(self, word: str) -> int:
        """
        Encode word using learned RAM codes.
        """
        if not self._is_trained:
            raise RuntimeError("Encoder not trained. Call train() first.")

        if word in self._vocab:
            return int(self._codes[self._vocab[word]])

        # For unknown words, use context signature if available
        if word in self._word_context_signature:
            signature = self._word_context_signature[word]
            # Find nearest known word by signature
            best_dist = float('inf')
            best_code = 0
            for known_word, idx in self._vocab.items():
                known_sig = self._word_context_signature.get(known_word)
                if known_sig is not None:
                    dist = np.linalg.norm(signature - known_sig)
                    if dist < best_dist:
                        best_dist = dist
                        best_code = int(self._codes[idx])
            return best_code

        # Fall back to hash
        return self._hash_unknown(word)

    def predict_from_context(self, context: list[str]) -> int:
        """
        Predict code from context using trained RAMs.

        Args:
            context: Context words

        Returns:
            Predicted binary code
        """
        if not self._is_trained:
            return 0

        # Create fake token list for hashing
        position = len(context) - 1
        context_hash = 0
        for j, word in enumerate(context):
            if word in self._context_word_to_feature:
                feature = self._context_word_to_feature[word]
                rel_pos = j - position + self._context_window
                context_hash ^= (feature * 31 + rel_pos) * (1 << ((j - position + 3) * 4))
        context_hash = context_hash % (1 << 16)

        # Get prediction from each bit's RAM
        code = 0
        for bit in range(self._n_bits):
            if context_hash in self._bit_rams[bit]:
                votes = self._bit_rams[bit][context_hash]
                if votes:
                    predicted = votes.most_common(1)[0][0]
                    if predicted == 1:
                        code |= (1 << bit)

        return code

    def get_ram_statistics(self) -> list[dict]:
        """
        Get statistics for each bit's RAM.
        """
        if not self._is_trained:
            return []

        stats = []
        for bit in range(self._n_bits):
            ram = self._bit_rams[bit]
            n_addresses = len(ram)

            # Count how many addresses predict 0 vs 1
            predicts_0 = sum(1 for v in ram.values() if v.most_common(1)[0][0] == 0)
            predicts_1 = n_addresses - predicts_0

            # Average confidence
            confidences = []
            for votes in ram.values():
                total = sum(votes.values())
                if total > 0:
                    conf = votes.most_common(1)[0][1] / total
                    confidences.append(conf)

            stats.append({
                "bit": bit,
                "n_addresses": n_addresses,
                "predicts_0": predicts_0,
                "predicts_1": predicts_1,
                "avg_confidence": np.mean(confidences) if confidences else 0,
            })

        return stats

    def get_stats(self) -> dict:
        """Get encoder statistics."""
        stats = super().get_stats()
        stats["context_window"] = self._context_window
        stats["n_context_features"] = self._n_context_features

        if self._is_trained:
            ram_stats = self.get_ram_statistics()
            total_addresses = sum(r["n_addresses"] for r in ram_stats)
            avg_confidence = np.mean([r["avg_confidence"] for r in ram_stats])
            stats["total_ram_addresses"] = total_addresses
            stats["avg_ram_confidence"] = avg_confidence
            stats["ram_statistics"] = ram_stats

        return stats
