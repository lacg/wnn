"""
XOR-based Cross-Attention

Cross-attention mechanism that uses XOR (Hamming distance) for query-key matching.
This provides content-addressable memory lookup with 100% generalization.

Key insight: XOR-based similarity is COMPUTED, not learned:
- similarity(q, k) = n_bits - popcount(q ⊕ k)
- High similarity means more matching bits
- Generalizes to ANY query/key values (no memorization needed)

Use cases:
- Content-addressable memory (find key most similar to query)
- Associative lookup (like dictionary lookup but fuzzy)
- Encoder-decoder attention where content matching matters
"""

from enum import Enum, auto
from torch import Tensor, zeros, uint8, float32, cat
from torch.nn import Module, ModuleList

from wnn.ram.core import RAMLayer
from wnn.ram.core.transformers.attention_base import ComputedAttention
from wnn.ram.core.transformers.computed_arithmetic import bits_to_int


class TopKAggregation(Enum):
    """Aggregation strategies for combining top-k matched values."""
    FIRST = auto()      # Take only the best match (k=1 behavior)
    XOR = auto()        # Bitwise XOR of all matched values
    MAJORITY = auto()   # Per-bit majority voting
    WEIGHTED = auto()   # Similarity-weighted majority voting


class XORCrossAttention(ComputedAttention):
    """
    Cross-attention using XOR (Hamming distance) for query-key matching.

    For each query, finds the key(s) with highest bit-level similarity
    (lowest Hamming distance = fewest differing bits).

    This is a COMPUTED attention mechanism - the matching is done via
    XOR computation, not learned lookup tables. This means it generalizes
    100% to unseen query/key pairs.

    The value projection can optionally be learned or computed (identity).
    """

    def __init__(
        self,
        query_bits: int,
        key_bits: int | None = None,
        value_bits: int | None = None,
        top_k: int = 1,
        threshold: float | None = None,
        aggregation: TopKAggregation = TopKAggregation.FIRST,
        learn_value_projection: bool = False,
        rng: int | None = None,
    ):
        """
        Args:
            query_bits: Bits per query token
            key_bits: Bits per key token (default: same as query_bits)
            value_bits: Bits per value token (default: same as key_bits)
            top_k: Number of best matches to attend to (1 = argmax)
            threshold: Optional similarity threshold (0-1). If set, only attend
                      to keys with similarity >= threshold * max_possible
            aggregation: How to combine top-k matched values:
                - FIRST: Use only the best match (ignores k > 1)
                - XOR: Bitwise XOR of all k values
                - MAJORITY: Per-bit majority voting across k values
                - WEIGHTED: Similarity-weighted per-bit voting
            learn_value_projection: If True, learn a projection from value to output
            rng: Random seed
        """
        super().__init__()

        self.query_bits = query_bits
        self.key_bits = key_bits or query_bits
        self.value_bits = value_bits or self.key_bits
        self.top_k = top_k
        self.threshold = threshold
        self.aggregation = aggregation
        self.learn_value_projection = learn_value_projection

        # Optional learned value projection
        if learn_value_projection:
            self.value_projection = RAMLayer(
                total_input_bits=self.value_bits,
                num_neurons=query_bits,  # Project to query dimension
                n_bits_per_neuron=min(self.value_bits, 10),
                rng=rng,
            )
        else:
            self.value_projection = None

        agg_name = aggregation.name
        print(f"[XORCrossAttention] query={query_bits}b, key={self.key_bits}b, "
              f"value={self.value_bits}b, top_k={top_k}, agg={agg_name}, "
              f"learned_proj={learn_value_projection}")

    def _hamming_similarity(self, query: Tensor, key: Tensor) -> int:
        """
        Compute Hamming similarity (matching bits) between query and key.

        Returns: number of matching bits (higher = more similar)
        """
        query = query.squeeze()
        key = key.squeeze()

        # XOR gives 1 where bits differ, 0 where they match
        xor_result = query ^ key

        # Count matching bits = total bits - differing bits
        differing = int(xor_result.sum().item())
        matching = len(query) - differing

        return matching

    def _get_similarities(self, query: Tensor, keys: list[Tensor]) -> list[int]:
        """Get similarity scores for query against all keys."""
        return [self._hamming_similarity(query, k) for k in keys]

    def _aggregate_top_k(
        self,
        values: list[Tensor],
        similarities: list[int],
        indexed: list[tuple[int, int]],
    ) -> Tensor:
        """
        Aggregate top-k values according to the selected strategy.

        Args:
            values: All value tensors
            similarities: Similarity scores for each value
            indexed: Sorted list of (similarity, index) pairs (descending)

        Returns:
            Aggregated value tensor
        """
        # Get top-k values and their similarities
        k = min(self.top_k, len(indexed))

        if k == 0:
            return zeros(self.value_bits, dtype=uint8)

        top_k_pairs = indexed[:k]
        top_k_values = [values[idx] for _, idx in top_k_pairs]
        top_k_sims = [sim for sim, _ in top_k_pairs]

        # Apply aggregation strategy
        if self.aggregation == TopKAggregation.FIRST or k == 1:
            # Just use the best match
            return top_k_values[0].clone()

        elif self.aggregation == TopKAggregation.XOR:
            # Bitwise XOR of all k values
            result = top_k_values[0].clone()
            for i in range(1, k):
                result = result ^ top_k_values[i]
            return result

        elif self.aggregation == TopKAggregation.MAJORITY:
            # Per-bit majority voting (unweighted)
            n_bits = len(top_k_values[0])
            result = zeros(n_bits, dtype=uint8)
            threshold = k // 2

            for bit in range(n_bits):
                ones_count = sum(v[bit].item() for v in top_k_values)
                result[bit] = 1 if ones_count > threshold else 0

            return result

        elif self.aggregation == TopKAggregation.WEIGHTED:
            # Similarity-weighted per-bit voting
            n_bits = len(top_k_values[0])
            result = zeros(n_bits, dtype=uint8)
            total_sim = sum(top_k_sims)

            if total_sim == 0:
                return top_k_values[0].clone()

            for bit in range(n_bits):
                weighted_ones = sum(
                    sim for v, sim in zip(top_k_values, top_k_sims)
                    if v[bit].item() == 1
                )
                result[bit] = 1 if weighted_ones > total_sim / 2 else 0

            return result

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """
        Get attention weights based on XOR similarity.

        Args:
            tokens: Query tokens [query_len, query_bits]
            context: Key/value tokens [context_len, key_bits] (if None, self-attention)

        Returns:
            Tensor of shape [query_len, key_len] with attention weights
        """
        queries = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        keys = context if context is not None else tokens
        keys = [k.squeeze() if k.ndim > 1 else k for k in keys]

        n_queries = len(queries)
        n_keys = len(keys)

        weights = zeros(n_queries, n_keys, dtype=float32)

        for i, query in enumerate(queries):
            similarities = self._get_similarities(query, keys)

            if self.threshold is not None:
                # Use threshold-based selection
                max_sim = self.query_bits  # Maximum possible similarity
                min_required = int(self.threshold * max_sim)
                for j, sim in enumerate(similarities):
                    if sim >= min_required:
                        weights[i, j] = sim / max_sim
            else:
                # Use top-k selection
                # Find top-k indices
                indexed = [(sim, j) for j, sim in enumerate(similarities)]
                indexed.sort(key=lambda x: -x[0])  # Sort descending

                for rank in range(min(self.top_k, len(indexed))):
                    sim, j = indexed[rank]
                    # Normalize by max possible similarity
                    weights[i, j] = sim / self.query_bits

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """
        Apply XOR-based cross-attention.

        Args:
            tokens: Query tokens (decoder hidden states)
            context: Key/value tokens (encoder output). If None, self-attention.

        Returns:
            List of output tokens, one per query position
        """
        queries = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        keys = context if context is not None else tokens
        keys = [k.squeeze() if k.ndim > 1 else k for k in keys]
        values = keys  # In cross-attention, values usually same as keys

        outputs = []

        for query in queries:
            similarities = self._get_similarities(query, keys)

            # Find best match(es) - sort by similarity descending
            indexed = [(sim, j) for j, sim in enumerate(similarities)]
            indexed.sort(key=lambda x: -x[0])

            # Aggregate top-k values
            value = self._aggregate_top_k(values, similarities, indexed)

            # Apply value projection if learned
            if self.value_projection is not None:
                value = self.value_projection(value.unsqueeze(0)).squeeze()

            outputs.append(value.clone())

        return outputs

    def lookup(self, query: Tensor, keys: list[Tensor], values: list[Tensor]) -> Tensor:
        """
        Single query lookup (convenience method for dictionary-style access).

        Args:
            query: Single query token
            keys: List of key tokens
            values: List of value tokens (parallel to keys)

        Returns:
            Value corresponding to best matching key
        """
        query = query.squeeze()
        keys = [k.squeeze() if k.ndim > 1 else k for k in keys]
        values = [v.squeeze() if v.ndim > 1 else v for v in values]

        similarities = self._get_similarities(query, keys)
        best_idx = max(range(len(similarities)), key=lambda i: similarities[i])

        return values[best_idx].clone()


class XORContentAddressableMemory(Module):
    """
    Content-Addressable Memory using XOR matching.

    A simple key-value store where lookup is done by finding the key
    with highest Hamming similarity to the query.

    This is useful for:
    - Encoder-decoder attention (find most relevant encoder state)
    - Memory networks (retrieve relevant memories)
    - Associative lookup (fuzzy dictionary)
    """

    def __init__(
        self,
        key_bits: int,
        value_bits: int | None = None,
        capacity: int = 64,
    ):
        """
        Args:
            key_bits: Bits per key
            value_bits: Bits per value (default: same as key_bits)
            capacity: Maximum number of key-value pairs
        """
        super().__init__()

        self.key_bits = key_bits
        self.value_bits = value_bits or key_bits
        self.capacity = capacity

        # Storage
        self.keys: list[Tensor] = []
        self.values: list[Tensor] = []

        print(f"[XORContentAddressableMemory] key={key_bits}b, "
              f"value={self.value_bits}b, capacity={capacity}")

    def write(self, key: Tensor, value: Tensor) -> None:
        """Store a key-value pair."""
        key = key.squeeze()
        value = value.squeeze()

        if len(self.keys) >= self.capacity:
            # Simple FIFO eviction
            self.keys.pop(0)
            self.values.pop(0)

        self.keys.append(key.clone())
        self.values.append(value.clone())

    def read(self, query: Tensor) -> Tensor:
        """Read value for best matching key."""
        if not self.keys:
            return zeros(self.value_bits, dtype=uint8)

        query = query.squeeze()

        # Find best match
        best_idx = 0
        best_sim = -1

        for i, key in enumerate(self.keys):
            xor = query ^ key
            sim = self.key_bits - int(xor.sum().item())
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        return self.values[best_idx].clone()

    def read_with_similarity(self, query: Tensor) -> tuple[Tensor, float]:
        """Read value and return similarity score."""
        if not self.keys:
            return zeros(self.value_bits, dtype=uint8), 0.0

        query = query.squeeze()

        best_idx = 0
        best_sim = -1

        for i, key in enumerate(self.keys):
            xor = query ^ key
            sim = self.key_bits - int(xor.sum().item())
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        normalized_sim = best_sim / self.key_bits
        return self.values[best_idx].clone(), normalized_sim

    def clear(self) -> None:
        """Clear all stored key-value pairs."""
        self.keys.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.keys)
