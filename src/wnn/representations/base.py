"""
Base class for Binary Encoders.

Binary encoders learn to map words to fixed-size binary codes
that preserve semantic similarity. This enables RAM networks
to generalize between similar words.

The key property: words with similar meanings should have
similar binary codes (low Hamming distance).

Example:
    encoder.encode("cat") = 0b101101001100
    encoder.encode("dog") = 0b101101001101  # 1 bit different
    encoder.encode("car") = 0b010010110011  # Many bits different
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BinaryEncoder(ABC):
    """
    Abstract base class for learning binary word representations.

    Binary codes enable RAM-compatible word representations where
    similar words map to similar addresses.
    """

    def __init__(self, n_bits: int = 12, name: str = "base"):
        """
        Initialize binary encoder.

        Args:
            n_bits: Number of bits in the binary code
            name: Human-readable name
        """
        self._n_bits = n_bits
        self._name = name
        self._is_trained = False
        self._vocab: dict[str, int] = {}  # word → index
        self._codes: Optional[np.ndarray] = None  # index → binary code

    @property
    def n_bits(self) -> int:
        return self._n_bits

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @abstractmethod
    def train(self, tokens: list[str], **kwargs) -> None:
        """
        Learn binary codes from corpus.

        Args:
            tokens: Training corpus tokens
            **kwargs: Encoder-specific parameters
        """
        ...

    def encode(self, word: str) -> int:
        """
        Encode a word to its binary code.

        Args:
            word: Word to encode

        Returns:
            Binary code as integer
        """
        if not self._is_trained:
            raise RuntimeError("Encoder not trained. Call train() first.")

        idx = self._vocab.get(word)
        if idx is None:
            # Unknown word: return hash-based code
            return self._hash_unknown(word)

        return int(self._codes[idx])

    def encode_sequence(self, words: list[str]) -> list[int]:
        """
        Encode a sequence of words.

        Args:
            words: List of words

        Returns:
            List of binary codes
        """
        return [self.encode(word) for word in words]

    def _hash_unknown(self, word: str) -> int:
        """
        Generate a code for unknown words using hashing.

        Uses a simple hash that tends to place similar-looking
        words near each other (via character-level features).
        """
        # Simple hash based on character features
        h = 0
        for i, c in enumerate(word):
            h ^= (ord(c) * (i + 1)) % (1 << self._n_bits)
        return h % (1 << self._n_bits)

    def decode(self, code: int) -> Optional[str]:
        """
        Find the word with this exact code (if any).

        Args:
            code: Binary code

        Returns:
            Word with this code, or None if no exact match
        """
        if not self._is_trained or self._codes is None:
            return None

        matches = np.where(self._codes == code)[0]
        if len(matches) == 0:
            return None

        # Return first match (arbitrary if multiple)
        for word, idx in self._vocab.items():
            if idx == matches[0]:
                return word
        return None

    def hamming_distance(self, word1: str, word2: str) -> int:
        """
        Compute Hamming distance between two words' codes.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Number of differing bits
        """
        code1 = self.encode(word1)
        code2 = self.encode(word2)
        xor = code1 ^ code2
        return bin(xor).count('1')

    def similarity(self, word1: str, word2: str) -> float:
        """
        Compute similarity between two words (0 to 1).

        Args:
            word1: First word
            word2: Second word

        Returns:
            Similarity score (1 = identical codes, 0 = all bits differ)
        """
        dist = self.hamming_distance(word1, word2)
        return 1.0 - dist / self._n_bits

    def nearest_neighbors(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Find k nearest neighbors by Hamming distance.

        Args:
            word: Query word
            k: Number of neighbors

        Returns:
            List of (word, similarity) tuples
        """
        if not self._is_trained or self._codes is None:
            return []

        target_code = self.encode(word)

        # Compute distances to all words
        distances = []
        for w, idx in self._vocab.items():
            if w == word:
                continue
            code = int(self._codes[idx])
            dist = bin(target_code ^ code).count('1')
            distances.append((w, dist))

        # Sort by distance and return top k
        distances.sort(key=lambda x: x[1])
        return [(w, 1.0 - d / self._n_bits) for w, d in distances[:k]]

    def get_code_distribution(self) -> dict[int, int]:
        """
        Get distribution of words per code.

        Returns:
            Dict mapping code → count of words with that code
        """
        if not self._is_trained or self._codes is None:
            return {}

        from collections import Counter
        return dict(Counter(self._codes.tolist()))

    def get_stats(self) -> dict:
        """Get encoder statistics."""
        stats = {
            "trained": self._is_trained,
            "n_bits": self._n_bits,
            "vocab_size": self.vocab_size,
            "name": self._name,
        }

        if self._is_trained and self._codes is not None:
            # Code utilization
            unique_codes = len(set(self._codes.tolist()))
            max_codes = 1 << self._n_bits
            stats["unique_codes"] = unique_codes
            stats["code_utilization"] = unique_codes / max_codes
            stats["avg_words_per_code"] = self.vocab_size / unique_codes

        return stats
