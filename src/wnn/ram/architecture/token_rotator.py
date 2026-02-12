"""
Token Rotator for cycling through token subsets during optimization.

Divides tokens into N parts (halves, thirds, fourths, etc.) and rotates
through them to avoid bias in which data is used for which optimization phase.

Rotation pattern per cycle:
1. Randomly pick 1 part from all N parts
2. Randomly pick 1 part from remaining N-1 parts
3. Continue until all parts are used
4. Repeat cycle

This ensures all parts get equal usage but in randomized order within each cycle.

Usage:
    rotator = TokenRotator(all_tokens, num_parts=3, seed=42)

    # Get subset for Phase 1a
    train_subset = rotator.next()  # Returns ~1/3 of tokens

    # Get subset for Phase 1b
    train_subset = rotator.next()  # Returns different ~1/3

    # Get full tokens for final evaluation
    full_tokens = rotator.full()
"""

import random
from dataclasses import dataclass
from typing import Optional, Sequence, TypeVar

T = TypeVar('T')


@dataclass
class RotatorConfig:
	"""Configuration for TokenRotator."""
	num_parts: int = 3  # Default to thirds
	seed: Optional[int] = None  # Random seed for reproducibility


class TokenRotator:
	"""
	Rotates through token subsets to avoid training bias.

	Divides tokens into N equal parts and cycles through them in a
	randomized but balanced pattern. Each cycle uses all parts exactly once.

	Attributes:
		num_parts: Number of parts to divide tokens into (default: 3 for thirds)
		seed: Random seed for reproducible rotation order

	Example with thirds (A, B, C):
		Cycle 1: B, A, C (random order)
		Cycle 2: C, B, A (different random order)
		Cycle 3: A, C, B (different random order)
		...

	Each call to next() returns the next part in the current cycle.
	When a cycle completes, the next cycle starts with a new random order.
	"""

	def __init__(
		self,
		tokens: Sequence[int],
		num_parts: int = 3,
		seed: Optional[int] = None,
	):
		"""
		Initialize the token rotator.

		Args:
			tokens: Full sequence of tokens to divide
			num_parts: Number of parts to divide into (2=halves, 3=thirds, 4=fourths, etc.)
			seed: Random seed for reproducible rotation order
		"""
		if num_parts < 2:
			raise ValueError(f"num_parts must be >= 2, got {num_parts}")

		self._full_tokens = list(tokens)
		self._num_parts = num_parts
		self._seed = seed
		self._rng = random.Random(seed)

		# Divide tokens into parts
		self._parts = self._divide_tokens()

		# Current cycle state
		self._current_cycle: list[int] = []  # Indices of parts in current cycle order
		self._cycle_position: int = 0  # Position within current cycle
		self._total_calls: int = 0  # Total number of next() calls

		# Start first cycle
		self._start_new_cycle()

	def _divide_tokens(self) -> list[list[int]]:
		"""Divide tokens into num_parts equal(ish) parts."""
		n = len(self._full_tokens)
		part_size = n // self._num_parts
		remainder = n % self._num_parts

		parts = []
		start = 0
		for i in range(self._num_parts):
			# Distribute remainder across first few parts
			end = start + part_size + (1 if i < remainder else 0)
			parts.append(self._full_tokens[start:end])
			start = end

		return parts

	def _start_new_cycle(self) -> None:
		"""Start a new cycle with randomized order."""
		# Create list of part indices and shuffle
		self._current_cycle = list(range(self._num_parts))
		self._rng.shuffle(self._current_cycle)
		self._cycle_position = 0

	def next(self) -> list[int]:
		"""
		Get the next token subset in the rotation.

		Returns:
			List of tokens for this rotation step (~1/num_parts of total)
		"""
		# Get current part index
		part_idx = self._current_cycle[self._cycle_position]

		# Advance position
		self._cycle_position += 1
		self._total_calls += 1

		# Start new cycle if needed
		if self._cycle_position >= self._num_parts:
			self._start_new_cycle()

		return self._parts[part_idx]

	def peek(self) -> list[int]:
		"""
		Peek at the next token subset without advancing the rotation.

		Returns:
			List of tokens that would be returned by next()
		"""
		part_idx = self._current_cycle[self._cycle_position]
		return self._parts[part_idx]

	def full(self) -> list[int]:
		"""
		Get the full token sequence (for final evaluation).

		Returns:
			Complete list of all tokens
		"""
		return self._full_tokens

	def reset(self, seed: Optional[int] = None) -> None:
		"""
		Reset the rotator to the beginning.

		Args:
			seed: Optional new seed (uses original seed if None)
		"""
		if seed is not None:
			self._seed = seed
		self._rng = random.Random(self._seed)
		self._cycle_position = 0
		self._total_calls = 0
		self._start_new_cycle()

	@property
	def num_parts(self) -> int:
		"""Number of parts tokens are divided into."""
		return self._num_parts

	@property
	def part_size(self) -> int:
		"""Approximate size of each part."""
		return len(self._full_tokens) // self._num_parts

	@property
	def total_tokens(self) -> int:
		"""Total number of tokens."""
		return len(self._full_tokens)

	@property
	def current_cycle_order(self) -> list[int]:
		"""Current cycle's part order (for debugging)."""
		return list(self._current_cycle)

	@property
	def calls_made(self) -> int:
		"""Number of next() calls made."""
		return self._total_calls

	def __repr__(self) -> str:
		return (
			f"TokenRotator(tokens={self.total_tokens}, "
			f"parts={self._num_parts}, "
			f"part_size≈{self.part_size}, "
			f"calls={self._total_calls})"
		)


class DatasetRotator:
	"""
	Manages token rotation for train, eval, and test datasets.

	Provides synchronized rotation across multiple datasets,
	ensuring consistent subset selection.

	Usage:
		rotator = DatasetRotator(
			train_tokens=train_data,
			eval_tokens=eval_data,
			test_tokens=test_data,
			num_parts=3,
		)

		# Get subsets for current phase
		train_subset = rotator.train_next()
		eval_subset = rotator.eval_next()

		# Get full datasets for final evaluation
		train_full, eval_full, test_full = rotator.full()
	"""

	def __init__(
		self,
		train_tokens: Sequence[int],
		eval_tokens: Optional[Sequence[int]] = None,
		test_tokens: Optional[Sequence[int]] = None,
		num_parts: int = 3,
		seed: Optional[int] = None,
	):
		"""
		Initialize dataset rotator.

		Args:
			train_tokens: Training token sequence
			eval_tokens: Optional evaluation token sequence
			test_tokens: Optional test token sequence
			num_parts: Number of parts to divide into (default: 3 for thirds)
			seed: Random seed for reproducibility
		"""
		self._train_rotator = TokenRotator(train_tokens, num_parts, seed)
		self._eval_rotator = TokenRotator(eval_tokens, num_parts, seed) if eval_tokens else None
		self._test_rotator = TokenRotator(test_tokens, num_parts, seed) if test_tokens else None
		self._num_parts = num_parts

	def train_next(self) -> list[int]:
		"""Get next training token subset."""
		return self._train_rotator.next()

	def eval_next(self) -> list[int]:
		"""Get next eval token subset (or full if no eval rotator)."""
		if self._eval_rotator:
			return self._eval_rotator.next()
		return []

	def test_next(self) -> list[int]:
		"""Get next test token subset (or full if no test rotator)."""
		if self._test_rotator:
			return self._test_rotator.next()
		return []

	def advance_all(self) -> tuple[list[int], list[int], list[int]]:
		"""
		Advance all rotators and return subsets.

		Returns:
			Tuple of (train_subset, eval_subset, test_subset)
		"""
		train = self._train_rotator.next()
		eval_ = self._eval_rotator.next() if self._eval_rotator else []
		test = self._test_rotator.next() if self._test_rotator else []
		return train, eval_, test

	def full(self) -> tuple[list[int], list[int], list[int]]:
		"""
		Get full datasets (for final evaluation).

		Returns:
			Tuple of (train_full, eval_full, test_full)
		"""
		train = self._train_rotator.full()
		eval_ = self._eval_rotator.full() if self._eval_rotator else []
		test = self._test_rotator.full() if self._test_rotator else []
		return train, eval_, test

	def reset(self, seed: Optional[int] = None) -> None:
		"""Reset all rotators."""
		self._train_rotator.reset(seed)
		if self._eval_rotator:
			self._eval_rotator.reset(seed)
		if self._test_rotator:
			self._test_rotator.reset(seed)

	@property
	def train_rotator(self) -> TokenRotator:
		"""Access to the training token rotator."""
		return self._train_rotator

	@property
	def num_parts(self) -> int:
		"""Number of parts datasets are divided into."""
		return self._num_parts

	def __repr__(self) -> str:
		eval_str = f", eval={self._eval_rotator.total_tokens}" if self._eval_rotator else ""
		test_str = f", test={self._test_rotator.total_tokens}" if self._test_rotator else ""
		return (
			f"DatasetRotator(train={self._train_rotator.total_tokens}"
			f"{eval_str}{test_str}, parts={self._num_parts})"
		)
