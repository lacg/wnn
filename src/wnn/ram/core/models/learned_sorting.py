"""
Learned Sorting Attention

Attention mechanism that learns to sort tokens by learning pairwise comparison.
Unlike ComputedSortingAttention which uses explicit numeric comparison,
this version learns the comparison function from examples.

Key insight: Sorting = Comparison + Routing
- Learn a pairwise comparator: (a, b) → is a < b?
- Use comparator to count rank for each token
- Route tokens to positions based on rank
"""

from torch import Tensor, zeros, uint8, float32, cat, tensor
from torch.nn import Module

from wnn.ram.core import RAMLayer
from wnn.ram.core.models.attention_base import LearnableAttention
from wnn.ram.core.models.computed_arithmetic import bits_to_int, int_to_bits


class BitLevelComparator(Module):
    """
    Bit-level comparison: learns a < b by decomposing into bit decisions.

    Key insight: Comparison is a cascade:
    - Check MSB: if a[0] < b[0], then a < b
    - If a[0] == b[0], check next bit
    - Continue until difference found

    This learns:
    1. Per-bit less-than: a[i] < b[i] (4 patterns per bit position)
    2. Per-bit equality: a[i] == b[i] (4 patterns per bit position)
    3. Cumulative combination: "first difference determines result"

    By learning smaller patterns, this can generalize to unseen pairs.
    """

    def __init__(
        self,
        n_bits: int,
        rng: int | None = None,
    ):
        """
        Args:
            n_bits: Number of bits per value being compared
            rng: Random seed for initialization
        """
        super().__init__()

        self.n_bits = n_bits

        # For each bit position, learn:
        # Input: (a[i], b[i]) = 2 bits
        # Output 1: is a[i] < b[i]? (i.e., a[i]=0 AND b[i]=1)
        # Output 2: is a[i] == b[i]?

        # Per-bit less-than detector: 2 bits → 1 bit
        # Only 4 possible inputs: (0,0)→0, (0,1)→1, (1,0)→0, (1,1)→0
        self.less_at = RAMLayer(
            total_input_bits=2,
            num_neurons=1,
            n_bits_per_neuron=2,
            rng=rng,
        )

        # Per-bit equality detector: 2 bits → 1 bit
        # Only 4 possible inputs: (0,0)→1, (0,1)→0, (1,0)→0, (1,1)→1
        self.equal_at = RAMLayer(
            total_input_bits=2,
            num_neurons=1,
            n_bits_per_neuron=2,
            rng=rng + 100 if rng else None,
        )

        # Cumulative combiner: learns "are all bits 0..i-1 equal?"
        # Input: (equal[0], equal[1], ..., equal[i-1]) → all_equal_so_far
        # This uses cumulative context: each position sees previous results
        self.prefix_equal = []
        for i in range(n_bits):
            if i == 0:
                # No prefix to check
                self.prefix_equal.append(None)
            else:
                # Learn: given equal[0..i-1], are they all 1?
                layer = RAMLayer(
                    total_input_bits=i,
                    num_neurons=1,
                    n_bits_per_neuron=min(i, 8),  # Cap address bits
                    rng=rng + 200 + i if rng else None,
                )
                self.prefix_equal.append(layer)

        self._trained = False

    def _train_basic_ops(self) -> int:
        """Train the per-bit less-than and equality operations."""
        errors = 0

        # Train less_at: (a, b) → a < b for single bits
        # (0,0)→0, (0,1)→1, (1,0)→0, (1,1)→0
        patterns = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0)]
        for a, b, expected in patterns:
            inp = tensor([a, b], dtype=uint8)
            out = tensor([[expected]], dtype=uint8)
            errors += self.less_at.commit(inp.unsqueeze(0), out)

        # Train equal_at: (a, b) → a == b for single bits
        # (0,0)→1, (0,1)→0, (1,0)→0, (1,1)→1
        patterns = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        for a, b, expected in patterns:
            inp = tensor([a, b], dtype=uint8)
            out = tensor([[expected]], dtype=uint8)
            errors += self.equal_at.commit(inp.unsqueeze(0), out)

        return errors

    def _train_prefix_combiner(self) -> int:
        """Train the cumulative AND logic for prefix equality."""
        errors = 0

        for i, layer in enumerate(self.prefix_equal):
            if layer is None:
                continue

            # Train: input is i bits, output is 1 iff all inputs are 1
            for val in range(1 << i):
                bits = [(val >> j) & 1 for j in range(i)]
                inp = tensor(bits, dtype=uint8)
                all_ones = 1 if all(b == 1 for b in bits) else 0
                out = tensor([[all_ones]], dtype=uint8)
                errors += layer.commit(inp.unsqueeze(0), out)

        return errors

    def train_all(self) -> int:
        """Train all components."""
        errors = 0
        errors += self._train_basic_ops()
        errors += self._train_prefix_combiner()
        self._trained = True
        return errors

    def forward(self, a: Tensor, b: Tensor) -> int:
        """
        Compare two values: is a < b?

        Algorithm:
        1. Compute less_at[i] and equal_at[i] for each bit position
        2. For each position i (MSB to LSB):
           - If all bits before i are equal AND less_at[i], return 1
        3. If no position triggers, return 0

        Args:
            a: First value as bit tensor [n_bits]
            b: Second value as bit tensor [n_bits]

        Returns:
            1 if a < b, 0 otherwise
        """
        a = a.squeeze() if a.ndim > 1 else a
        b = b.squeeze() if b.ndim > 1 else b

        # Compute per-bit indicators
        less_bits = []
        equal_bits = []

        for i in range(self.n_bits):
            pair = tensor([a[i].item(), b[i].item()], dtype=uint8)
            less_i = self.less_at(pair.unsqueeze(0)).squeeze().item()
            equal_i = self.equal_at(pair.unsqueeze(0)).squeeze().item()
            less_bits.append(less_i)
            equal_bits.append(equal_i)

        # Check cascade: first position where we're less (with all previous equal)
        for i in range(self.n_bits):
            if i == 0:
                # No prefix to check
                prefix_all_equal = 1
            else:
                # Check if all previous bits were equal
                prefix = tensor(equal_bits[:i], dtype=uint8)
                layer = self.prefix_equal[i]
                if layer is not None:
                    prefix_all_equal = layer(prefix.unsqueeze(0)).squeeze().item()
                else:
                    prefix_all_equal = 1

            if prefix_all_equal and less_bits[i]:
                return 1

        return 0

    def test_accuracy(self, max_value: int | None = None) -> float:
        """Test comparison accuracy on all pairs."""
        if max_value is None:
            max_value = (1 << self.n_bits) - 1

        correct = 0
        total = 0

        for a_val in range(max_value + 1):
            for b_val in range(max_value + 1):
                a = int_to_bits(a_val, self.n_bits)
                b = int_to_bits(b_val, self.n_bits)
                expected = 1 if a_val < b_val else 0

                predicted = self.forward(a, b)
                if predicted == expected:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def __repr__(self):
        return f"BitLevelComparator(n_bits={self.n_bits}, trained={self._trained})"


class LearnedComparator(Module):
    """
    Learn pairwise comparison: is a < b?

    Uses a RAMLayer to learn the comparison function from examples.
    For n_bits input values, the comparator sees 2*n_bits (concatenated pair).

    Training requires showing (a, b, expected) triples where expected is
    1 if a < b, else 0.
    """

    def __init__(
        self,
        n_bits: int,
        rng: int | None = None,
    ):
        """
        Args:
            n_bits: Number of bits per value being compared
            rng: Random seed for initialization
        """
        super().__init__()

        self.n_bits = n_bits

        # RAMLayer: 2*n_bits input → 1 output (comparison result)
        # Use all input bits for addressing to capture full comparison logic
        self.comparator = RAMLayer(
            total_input_bits=2 * n_bits,
            num_neurons=1,
            n_bits_per_neuron=min(2 * n_bits, 12),  # Cap at 12 bits (4096 addresses)
            rng=rng,
        )

        self._trained_pairs = 0

    def forward(self, a: Tensor, b: Tensor) -> int:
        """
        Compare two values: is a < b?

        Args:
            a: First value as bit tensor [n_bits]
            b: Second value as bit tensor [n_bits]

        Returns:
            1 if a < b, 0 otherwise
        """
        a = a.squeeze() if a.ndim > 1 else a
        b = b.squeeze() if b.ndim > 1 else b

        # Concatenate: [a, b]
        pair = cat([a, b])

        # Lookup comparison result
        result = self.comparator(pair.unsqueeze(0)).squeeze()
        return int(result.item())

    def compare_batch(self, tokens: list[Tensor]) -> list[list[int]]:
        """
        Compare all pairs of tokens.

        Args:
            tokens: List of token tensors

        Returns:
            Matrix where result[i][j] = 1 if tokens[i] < tokens[j]
        """
        n = len(tokens)
        result = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    result[i][j] = self.forward(tokens[i], tokens[j])

        return result

    def train_pair(self, a: Tensor, b: Tensor, expected: int) -> int:
        """
        Train on a single comparison pair.

        Args:
            a: First value
            b: Second value
            expected: 1 if a < b, 0 otherwise

        Returns:
            Number of corrections made (0 or 1)
        """
        a = a.squeeze() if a.ndim > 1 else a
        b = b.squeeze() if b.ndim > 1 else b

        pair = cat([a, b])
        target = zeros(1, dtype=uint8)
        target[0] = expected

        errors = self.comparator.commit(pair.unsqueeze(0), target.unsqueeze(0))
        self._trained_pairs += 1

        return errors

    def train_all_pairs(self, max_value: int | None = None) -> tuple[int, int]:
        """
        Train on all possible comparison pairs.

        Args:
            max_value: Maximum value (default: 2^n_bits - 1)

        Returns:
            (total_pairs, total_errors)
        """
        if max_value is None:
            max_value = (1 << self.n_bits) - 1

        total_pairs = 0
        total_errors = 0

        for a_val in range(max_value + 1):
            for b_val in range(max_value + 1):
                a = int_to_bits(a_val, self.n_bits)
                b = int_to_bits(b_val, self.n_bits)
                expected = 1 if a_val < b_val else 0

                errors = self.train_pair(a, b, expected)
                total_pairs += 1
                total_errors += errors

        return total_pairs, total_errors

    def test_accuracy(self, max_value: int | None = None) -> float:
        """
        Test comparison accuracy on all pairs.

        Args:
            max_value: Maximum value (default: 2^n_bits - 1)

        Returns:
            Accuracy (0.0 to 1.0)
        """
        if max_value is None:
            max_value = (1 << self.n_bits) - 1

        correct = 0
        total = 0

        for a_val in range(max_value + 1):
            for b_val in range(max_value + 1):
                a = int_to_bits(a_val, self.n_bits)
                b = int_to_bits(b_val, self.n_bits)
                expected = 1 if a_val < b_val else 0

                predicted = self.forward(a, b)
                if predicted == expected:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def __repr__(self):
        return f"LearnedComparator(n_bits={self.n_bits}, trained_pairs={self._trained_pairs})"


class LearnedSortingAttention(LearnableAttention):
    """
    Sorting attention using learned pairwise comparison.

    For each token, counts how many other tokens are smaller (its rank),
    then routes tokens to output positions based on rank.

    Unlike ComputedSortingAttention, the comparison function is learned
    from examples rather than computed numerically.

    Two comparator modes:
    - "memorize": LearnedComparator - memorizes all pairs (no generalization)
    - "bit_level": BitLevelComparator - learns bit-level logic (100% generalization)
    """

    def __init__(
        self,
        input_bits: int,
        descending: bool = False,
        comparator_mode: str = "bit_level",
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            descending: If True, sort largest first
            comparator_mode: "memorize" or "bit_level" (default: "bit_level")
            rng: Random seed for initialization
        """
        super().__init__()

        self.input_bits = input_bits
        self.descending = descending
        self.comparator_mode = comparator_mode

        # Choose comparator based on mode
        if comparator_mode == "bit_level":
            self.comparator = BitLevelComparator(input_bits, rng)
            self._bit_level = True
        else:
            self.comparator = LearnedComparator(input_bits, rng)
            self._bit_level = False

        print(f"[LearnedSortingAttention] input={input_bits}b, "
              f"order={'descending' if descending else 'ascending'}, "
              f"comparator={comparator_mode}")

    def _get_rank(self, token: Tensor, all_tokens: list[Tensor], token_idx: int) -> int:
        """
        Get the rank of a token (how many tokens are smaller).

        For stable sorting, if values are equal, use original index as tiebreaker.

        Args:
            token: The token to rank
            all_tokens: All tokens in sequence
            token_idx: Original index of this token

        Returns:
            Rank (0 = smallest, n-1 = largest for ascending)
        """
        token = token.squeeze() if token.ndim > 1 else token
        rank = 0

        for i, other in enumerate(all_tokens):
            if i == token_idx:
                continue

            other = other.squeeze() if other.ndim > 1 else other

            if self.descending:
                # For descending: count how many are LARGER
                is_larger = self.comparator(token, other)  # token < other means other is larger
                if is_larger:
                    rank += 1
                elif self.comparator(other, token) == 0:
                    # Equal values: use index for stable sort
                    if i < token_idx:
                        rank += 1
            else:
                # For ascending: count how many are SMALLER
                is_smaller = self.comparator(other, token)  # other < token
                if is_smaller:
                    rank += 1
                elif self.comparator(token, other) == 0:
                    # Equal values: use index for stable sort
                    if i < token_idx:
                        rank += 1

        return rank

    def get_attention_weights(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> Tensor:
        """
        Get attention weights for sorting.

        Returns a [num_queries, num_keys] tensor where each row has exactly
        one 1.0 at the position of the token that belongs at that output position.
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Get rank for each token
        ranks = [self._get_rank(t, tokens, i) for i, t in enumerate(tokens)]

        # Build attention matrix: output[rank] attends to input[i]
        weights = zeros(n, n, dtype=float32)
        for i, rank in enumerate(ranks):
            if 0 <= rank < n:
                weights[rank, i] = 1.0

        return weights

    def forward(
        self,
        tokens: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """
        Sort the sequence using learned comparison.

        Args:
            tokens: Input sequence
            context: Ignored (sorting is self-attention)

        Returns:
            Sorted sequence
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        # Get rank for each token
        ranks = [self._get_rank(t, tokens, i) for i, t in enumerate(tokens)]

        # Route tokens to output positions based on rank
        outputs = [zeros(self.input_bits, dtype=uint8) for _ in range(n)]
        used_positions = set()

        for i, (token, rank) in enumerate(zip(tokens, ranks)):
            # Handle rank collisions (shouldn't happen with proper comparison)
            pos = rank
            while pos in used_positions and pos < n:
                pos += 1
            if pos >= n:
                # Find any unused position
                for p in range(n):
                    if p not in used_positions:
                        pos = p
                        break

            if pos < n:
                outputs[pos] = token.clone()
                used_positions.add(pos)

        return outputs

    def train_step(
        self,
        tokens: list[Tensor],
        targets: list[Tensor],
        context: list[Tensor] | None = None,
    ) -> int:
        """
        Single training step - extract comparisons from sorted targets.

        Args:
            tokens: Input tokens (unsorted)
            targets: Target tokens (sorted)
            context: Ignored

        Returns:
            Number of corrections made
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

        # Extract comparison relationships from the sorted order
        errors = 0
        target_values = [bits_to_int(t) for t in targets]

        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                # targets[i] should be <= targets[j]
                a, b = targets[i], targets[j]
                val_a, val_b = target_values[i], target_values[j]

                if val_a < val_b:
                    errors += self.comparator.train_pair(a, b, 1)  # a < b
                    errors += self.comparator.train_pair(b, a, 0)  # b >= a
                elif val_a > val_b:
                    # Shouldn't happen in sorted sequence
                    pass
                # Equal: no training needed

        return errors

    def train_comparator(self, max_value: int | None = None) -> tuple[int, int]:
        """
        Train the comparator.

        For bit_level mode: trains bit-level operations (few patterns, generalizes)
        For memorize mode: trains on all pairs (many patterns, no generalization)

        Args:
            max_value: Maximum value to train on (only used for memorize mode)

        Returns:
            (total_patterns, total_errors)
        """
        if self._bit_level:
            # BitLevelComparator: train bit-level operations
            errors = self.comparator.train_all()
            # Count patterns: 8 basic + sum(2^i for i in 1..n-1)
            n = self.input_bits
            patterns = 8 + sum(1 << i for i in range(1, n))
            return patterns, errors
        else:
            # LearnedComparator: train on all pairs
            return self.comparator.train_all_pairs(max_value)

    def train_on_examples(
        self,
        examples: list[tuple[list[int], list[int]]],
    ) -> int:
        """
        Train comparator from sorting examples.

        Extracts pairwise comparison relationships from (input, sorted) pairs.

        Args:
            examples: List of (input_sequence, sorted_sequence) pairs
                     where values are integers

        Returns:
            Total corrections made
        """
        total_errors = 0

        for input_seq, sorted_seq in examples:
            # Extract ordering relationships
            for i, val_i in enumerate(sorted_seq):
                for j, val_j in enumerate(sorted_seq):
                    if i < j:
                        # val_i should be <= val_j (appears earlier in sorted)
                        a = int_to_bits(val_i, self.input_bits)
                        b = int_to_bits(val_j, self.input_bits)

                        if val_i < val_j:
                            # a < b
                            total_errors += self.comparator.train_pair(a, b, 1)
                            total_errors += self.comparator.train_pair(b, a, 0)
                        elif val_i > val_j:
                            # This shouldn't happen in a sorted sequence
                            pass
                        # Equal values: no comparison training needed

        return total_errors

    def test_sorting_accuracy(
        self,
        sequences: list[list[int]],
    ) -> tuple[float, int, int]:
        """
        Test sorting accuracy on sequences.

        Args:
            sequences: List of integer sequences to sort

        Returns:
            (accuracy, correct_sequences, total_sequences)
        """
        correct = 0
        total = len(sequences)

        for seq in sequences:
            # Convert to bit tensors
            tokens = [int_to_bits(v, self.input_bits) for v in seq]

            # Sort using learned attention
            sorted_tokens = self.forward(tokens)
            sorted_values = [bits_to_int(t) for t in sorted_tokens]

            # Expected sorted sequence
            if self.descending:
                expected = sorted(seq, reverse=True)
            else:
                expected = sorted(seq)

            if sorted_values == expected:
                correct += 1

        return correct / total if total > 0 else 0.0, correct, total

    def visualize_comparison_matrix(self, tokens: list[Tensor]) -> str:
        """Visualize the learned comparison results."""
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]
        n = len(tokens)

        lines = ["Learned Comparison Matrix (1 = row < col):"]
        values = [bits_to_int(t) for t in tokens]

        # Header
        header = "     " + " ".join(f"{v:3d}" for v in values)
        lines.append(header)

        # Matrix
        for i, t_i in enumerate(tokens):
            row = f"{values[i]:3d}: "
            for j, t_j in enumerate(tokens):
                if i == j:
                    row += "  - "
                else:
                    cmp = self.comparator(t_i, t_j)
                    row += f"  {cmp} "
            lines.append(row)

        return "\n".join(lines)

    def __repr__(self):
        return (f"LearnedSortingAttention(input_bits={self.input_bits}, "
                f"descending={self.descending}, mode={self.comparator_mode})")
