"""Bitwise IDS classifier — binary-coded attack type classification.

Instead of 9 discriminators (one per class), this approach uses N independent
binary classifiers, each predicting one bit of an error-correcting codeword.
The predicted bits are combined into a codeword and decoded to the nearest
valid class via Hamming distance.

Architecture:
    Input (321 bits) → N binary evaluators → N-bit codeword → nearest class

Error correction:
    With 6-bit codewords and minimum Hamming distance 3, any single-bit
    prediction error is automatically corrected (nearest valid codeword
    is still the correct class).
"""

import numpy as np
from dataclasses import dataclass
from itertools import combinations

from .dataset import IDSDataset


@dataclass
class CodewordTable:
	"""Maps classes to binary codewords with error-correcting properties."""
	num_bits: int
	num_classes: int
	codewords: np.ndarray   # (num_classes, num_bits) bool
	class_names: list[str]
	min_distance: int        # minimum Hamming distance between any two codewords

	def encode(self, class_idx: int) -> np.ndarray:
		"""Get the codeword for a class index."""
		return self.codewords[class_idx]

	def decode(self, bits: np.ndarray) -> int:
		"""Decode a (possibly noisy) codeword to the nearest class."""
		# Hamming distance to each valid codeword
		distances = np.sum(self.codewords != bits, axis=1)
		return int(np.argmin(distances))

	def decode_batch(self, bits: np.ndarray) -> np.ndarray:
		"""Decode a batch of codewords. bits: (n_samples, num_bits)."""
		# (n_samples, 1, num_bits) vs (1, num_classes, num_bits) → (n_samples, num_classes)
		distances = np.sum(
			bits[:, np.newaxis, :] != self.codewords[np.newaxis, :, :],
			axis=2,
		)
		return np.argmin(distances, axis=1)

	def __repr__(self) -> str:
		lines = [f"CodewordTable(bits={self.num_bits}, classes={self.num_classes}, "
				 f"min_distance={self.min_distance})"]
		for i, name in enumerate(self.class_names):
			code = "".join(str(int(b)) for b in self.codewords[i])
			lines.append(f"  [{i}] {name:15s} → {code}")
		return "\n".join(lines)


def _hamming_7_4_generator() -> np.ndarray:
	"""Standard [7,4,3] Hamming code generator matrix.

	Maps 4 data bits → 7 coded bits. Systematic form: [I4 | P]
	where P is the parity sub-matrix.
	"""
	return np.array([
		# d0 d1 d2 d3 p0 p1 p2
		[1, 0, 0, 0, 1, 1, 0],
		[0, 1, 0, 0, 1, 0, 1],
		[0, 0, 1, 0, 0, 1, 1],
		[0, 0, 0, 1, 1, 1, 1],
	], dtype=np.uint8)


def select_codewords(num_classes: int, num_bits: int = 6) -> np.ndarray:
	"""Select codewords using a shortened [7,4,3] Hamming code.

	The [7,4,3] Hamming code has 16 codewords with minimum distance 3.
	Shortening to 6 bits (dropping one column) preserves d=3, giving
	a [6,4,3] code with 16 codewords. We pick num_classes of them,
	choosing the subset that maximizes the minimum pairwise distance.

	Args:
		num_classes: number of classes to encode (must be ≤ 16)
		num_bits: output bits (6 for shortened Hamming, 7 for full)

	Returns:
		np.ndarray of shape (num_classes, num_bits) with bool dtype
	"""
	if num_bits not in (6, 7):
		raise ValueError(f"Hamming codewords only support 6 or 7 bits, got {num_bits}")
	if num_classes > 16:
		raise ValueError(f"Hamming [7,4] code has 16 codewords, can't encode {num_classes} classes")

	G = _hamming_7_4_generator()

	# Generate all 16 codewords: every 4-bit message × G (mod 2)
	all_codewords = np.zeros((16, 7), dtype=np.uint8)
	for i in range(16):
		message = np.array([(i >> b) & 1 for b in range(4)], dtype=np.uint8)
		all_codewords[i] = message @ G % 2

	# Shorten to 6 bits by dropping the last parity column
	if num_bits == 6:
		all_codewords = all_codewords[:, :6]

	# Pick the best num_classes codewords: greedy farthest-first
	# Start with the all-zeros codeword (index 0), then greedily pick
	# the codeword that maximizes the minimum distance to all selected
	selected = [0]
	remaining = set(range(1, 16))

	while len(selected) < num_classes:
		best_idx = -1
		best_min_dist = -1
		for candidate in remaining:
			min_dist = min(
				int(np.sum(all_codewords[candidate] != all_codewords[s]))
				for s in selected
			)
			if min_dist > best_min_dist:
				best_min_dist = min_dist
				best_idx = candidate
		selected.append(best_idx)
		remaining.remove(best_idx)

	return all_codewords[selected].astype(bool)


def create_bitwise_codebook(
	class_names: list[str],
	num_bits: int = 7,
) -> CodewordTable:
	"""Create an error-correcting codebook for attack type classification."""
	num_classes = len(class_names)
	codewords = select_codewords(num_classes, num_bits)

	# Compute minimum Hamming distance
	min_dist = num_bits + 1
	for i, j in combinations(range(num_classes), 2):
		dist = int(np.sum(codewords[i] != codewords[j]))
		min_dist = min(min_dist, dist)

	table = CodewordTable(
		num_bits=num_bits,
		num_classes=num_classes,
		codewords=codewords,
		class_names=class_names,
		min_distance=min_dist,
	)

	print(table)
	print(f"  Error correction: can correct {(min_dist - 1) // 2} bit errors, "
		  f"detect {min_dist - 1} bit errors")

	return table


def create_per_bit_dataset(
	dataset: IDSDataset,
	codebook: CodewordTable,
	bit_index: int,
) -> IDSDataset:
	"""Create a binary dataset for one output bit.

	For each example, the binary label is the bit_index-th bit
	of the class's codeword. The features are unchanged.
	"""
	# Map multi-class labels to per-bit binary labels
	# y_train_multi contains class indices 0-8 (already remapped for attack-only)
	y_train_bits = codebook.codewords[dataset.y_train_multi, bit_index].astype(np.int32)
	y_test_bits = codebook.codewords[dataset.y_test_multi, bit_index].astype(np.int32)

	return IDSDataset(
		X_train=dataset.X_train,
		y_train_binary=y_train_bits,
		y_train_multi=dataset.y_train_multi,  # keep original for reference
		X_test=dataset.X_test,
		y_test_binary=y_test_bits,
		y_test_multi=dataset.y_test_multi,
		encoder=dataset.encoder,
		category_names=[f"bit{bit_index}=0", f"bit{bit_index}=1"],
		feature_names=dataset.feature_names,
	)
