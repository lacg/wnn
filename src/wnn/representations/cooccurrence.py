"""
Co-occurrence Binary Codes.

Baseline approach that uses SVD on the co-occurrence matrix
and binarizes the resulting vectors.

This is similar to GloVe (which also uses matrix factorization)
but outputs binary codes instead of continuous vectors.

Algorithm:
    1. Build word co-occurrence matrix C
    2. Apply PPMI (Positive Pointwise Mutual Information)
    3. Reduce dimensions with SVD to n_bits
    4. Binarize by taking sign of each dimension

This provides a strong baseline for comparison with the
RAM-learned representations.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Optional

from wnn.representations.base import BinaryEncoder


class CooccurrenceCodes(BinaryEncoder):
    """
    Binary codes from SVD on co-occurrence matrix.

    Like GloVe but with binary output. Provides a baseline
    for comparison with RAM-learned representations.
    """

    def __init__(
        self,
        n_bits: int = 12,
        context_window: int = 5,
        min_freq: int = 5,
        name: str = "cooccurrence",
    ):
        """
        Initialize co-occurrence encoder.

        Args:
            n_bits: Number of bits in binary code
            context_window: Context window for co-occurrence
            min_freq: Minimum word frequency
            name: Human-readable name
        """
        super().__init__(n_bits=n_bits, name=name)
        self._context_window = context_window
        self._min_freq = min_freq

        # SVD components for encoding unknown words
        self._svd_components: Optional[np.ndarray] = None
        self._word_vectors: Optional[np.ndarray] = None

    def train(self, tokens: list[str], **kwargs) -> None:
        """
        Learn binary codes from co-occurrence statistics.

        Steps:
        1. Build co-occurrence matrix
        2. Apply PPMI weighting
        3. SVD to reduce to n_bits dimensions
        4. Binarize by sign
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

        # Build co-occurrence matrix
        cooccur = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        for i, token in enumerate(tokens):
            if token not in self._vocab:
                continue
            token_idx = self._vocab[token]

            # Count co-occurrences within window
            for j in range(max(0, i - self._context_window), min(len(tokens), i + self._context_window + 1)):
                if i == j:
                    continue
                context_token = tokens[j]
                if context_token not in self._vocab:
                    continue
                context_idx = self._vocab[context_token]

                # Distance-weighted co-occurrence
                distance = abs(i - j)
                weight = 1.0 / distance
                cooccur[token_idx, context_idx] += weight

        # Apply PPMI (Positive Pointwise Mutual Information)
        ppmi = self._compute_ppmi(cooccur)

        # SVD to reduce dimensions
        # Use n_bits dimensions
        n_components = min(self._n_bits, vocab_size - 1)

        try:
            from scipy.sparse.linalg import svds
            from scipy.sparse import csr_matrix

            # Convert to sparse for efficiency
            sparse_ppmi = csr_matrix(ppmi)
            U, S, Vt = svds(sparse_ppmi, k=n_components)

            # Sort by singular values (svds returns in ascending order)
            idx = np.argsort(S)[::-1]
            U = U[:, idx]
            S = S[idx]

            # Weight by sqrt of singular values (like GloVe)
            self._word_vectors = U * np.sqrt(S)

        except ImportError:
            # Fallback to numpy SVD
            U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
            self._word_vectors = U[:, :n_components] * np.sqrt(S[:n_components])

        # Store components for unknown word handling
        self._svd_components = Vt[:n_components, :] if 'Vt' in dir() else None

        # Binarize by sign
        # Each dimension becomes one bit
        self._codes = np.zeros(vocab_size, dtype=np.int64)

        for i in range(vocab_size):
            code = 0
            for bit in range(min(n_components, self._n_bits)):
                if self._word_vectors[i, bit] > 0:
                    code |= (1 << bit)
            self._codes[i] = code

        self._is_trained = True

    def _compute_ppmi(self, cooccur: np.ndarray) -> np.ndarray:
        """
        Compute Positive Pointwise Mutual Information.

        PPMI(w, c) = max(0, log(P(w,c) / (P(w) * P(c))))
        """
        # Smoothing
        cooccur = cooccur + 1e-10

        # Total count
        total = cooccur.sum()

        # Marginal probabilities
        p_word = cooccur.sum(axis=1) / total
        p_context = cooccur.sum(axis=0) / total

        # Joint probability
        p_joint = cooccur / total

        # PMI = log(P(w,c) / (P(w) * P(c)))
        # Use outer product for P(w) * P(c)
        expected = np.outer(p_word, p_context)
        expected = np.maximum(expected, 1e-10)

        pmi = np.log(p_joint / expected)

        # Positive PMI (clip negative values)
        ppmi = np.maximum(pmi, 0)

        return ppmi

    def get_stats(self) -> dict:
        """Get encoder statistics."""
        stats = super().get_stats()
        stats["context_window"] = self._context_window
        stats["min_freq"] = self._min_freq

        if self._is_trained and self._word_vectors is not None:
            # Variance explained by each dimension
            variances = np.var(self._word_vectors, axis=0)
            total_var = variances.sum()
            stats["variance_by_bit"] = (variances / total_var).tolist()

        return stats
