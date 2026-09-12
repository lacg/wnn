"""The cell-mode ladder — a typed view over the wire codes the Rust core uses.

Ordered by information per cell. `value` is the code every kernel and marker
carries (BINARY=3, TERNARY=0, PLN=5, QUAD_BINARY=1, QUAD_WEIGHTED=2, QSR=4);
`ram_core::cell_mode::CellMode` is the same table in Rust and the test suite
pins the two against each other.
"""

from enum import IntEnum


class CellMode(IntEnum):
	"""How a RAM cell is stored and read.

	BINARY        1 bit, FALSE/TRUE. TRUE fires. Classical WiSARD / n-tuple:
	              own-class visits set TRUE, other classes are ignored.
	TERNARY       FALSE / untrained / TRUE. Untrained reads `empty_value`.
	              Majority vote: own class +1, other classes -1.
	PLN           TERNARY cells, but untrained fires a fair coin (seeded).
	QUAD_BINARY   4-state nudging cells (FALSE, WEAK_FALSE, WEAK_TRUE, TRUE),
	              binary read: WEAK_TRUE/TRUE fire.            [ablation]
	QUAD_WEIGHTED 4-state nudging cells, graded read 0 / .25 / .75 / 1. DEFAULT.
	QSR           QUAD_WEIGHTED cells, but the read fires a coin whose
	              probability is the graded weight (seeded).
	"""

	TERNARY = 0
	QUAD_BINARY = 1
	QUAD_WEIGHTED = 2
	BINARY = 3
	QSR = 4
	PLN = 5

	@property
	def is_stochastic(self) -> bool:
		return self in (CellMode.QSR, CellMode.PLN)

	@property
	def uses_empty_value(self) -> bool:
		return self is CellMode.TERNARY

	@property
	def expected_read_mode(self) -> "CellMode":
		"""The deterministic mode whose read equals this mode's EXPECTED read."""
		if self is CellMode.QSR:
			return CellMode.QUAD_WEIGHTED
		if self is CellMode.PLN:
			return CellMode.TERNARY
		return self

	@property
	def num_states(self) -> int:
		if self is CellMode.BINARY:
			return 2
		if self in (CellMode.TERNARY, CellMode.PLN):
			return 3
		return 4


LADDER = (
	CellMode.BINARY,
	CellMode.TERNARY,
	CellMode.PLN,
	CellMode.QUAD_BINARY,
	CellMode.QUAD_WEIGHTED,
	CellMode.QSR,
)
"""Every mode in ladder order (information per cell, ascending)."""
