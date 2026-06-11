"""Cell semantics — the Python mirror of the Rust single source of truth.

Mirrors `neuron_memory.rs` (Rust) and `shaders/common.metal` (GPU). The
three implementations MUST agree; the unit test in
tests/cell_semantics_parity.py asserts this table, and the accelerator
parity test compares live forward passes once the ABI-2 build is installed.

Background (the trap this closes): Python's Memory writes TERNARY-encoded
cells (FALSE=0, TRUE=1, EMPTY=2), but Rust training (e.g.
ramlm_train_batch_numpy) writes QUAD-encoded words back into the SAME
buffer (FALSE=0, WEAK_FALSE=1, WEAK_TRUE=2, TRUE=3). Reading Rust-trained
memory with ternary comparisons scores WEAK_FALSE as TRUE — the Python-side
twin of the inverted-QUAD multistage bug fixed on 10/06/2026.
"""

import torch
from torch import Tensor

# --- Quad-mode cell values (match neuron_memory.rs) ---
QUAD_FALSE = 0
QUAD_WEAK_FALSE = 1   # initial state for quad modes
QUAD_WEAK_TRUE = 2
QUAD_TRUE = 3

# --- Memory modes (match neuron_memory.rs MODE_*) ---
MODE_TERNARY = 0
MODE_QUAD_BINARY = 1
MODE_QUAD_WEIGHTED = 2

# --- QUAD_WEIGHTED forward weights (match neuron_memory.rs QUAD_WEIGHTS) ---
QUAD_WEIGHTS = (0.0, 0.25, 0.75, 1.0)


def cell_to_weight(cell: int, memory_mode: int, empty_value: float) -> float:
	"""Scalar twin of neuron_memory::cell_to_weight (see its doc-comment)."""
	if memory_mode in (MODE_QUAD_BINARY, MODE_QUAD_WEIGHTED):
		return QUAD_WEIGHTS[min(max(int(cell), 0), 3)]
	if cell == 0:
		return 0.0
	if cell == 1:
		return 1.0
	return empty_value


def cell_weight_table(memory_mode: int, empty_value: float) -> Tensor:
	"""4-entry lookup table for vectorized cell→weight conversion.

	Usage: `weights = cell_weight_table(mode, ev)[raw_cells.long()]`.
	"""
	return torch.tensor(
		[cell_to_weight(c, memory_mode, empty_value) for c in range(4)],
		dtype=torch.float32,
	)
