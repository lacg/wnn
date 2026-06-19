"""Output decoders for thermometer-encoded PWM — re-exported from Rust.

The implementations live in
`src/wnn/ram/strategies/accelerator/controller.rs`. This module exists
so Python callers can write `from wnn.control.decoders import
strategy_5_qsr_weighted` without reaching into the accelerator package
directly.

Exposed:
  - strategy_5_qsr_weighted(cells, levels_per_motor, num_motors)
      → list[int]: bucket per motor in [0, levels_per_motor-1].
      QSR-weighted sum: FALSE=0.0, WEAK_FALSE=0.25, WEAK_TRUE=0.75,
      TRUE=1.0; sum per motor, round. Paper #1's primary decoder.
  - strategy_1_count_true(cells, levels_per_motor, num_motors)
      → list[int]: same shape, decoded as count of cells with QSR
      value ≥ 2 (TRUE-or-WEAK_TRUE). Ablation baseline.
  - monotonicity_violations(cells, levels_per_motor, num_motors)
      → int: total count of thermometer-pattern violations across
      all motors (used in the soft monotonicity reward).

Per the design memory `project_drone_controller_paper1.md`.
"""

from __future__ import annotations

from wnn.control._accel import (
	strategy_1_count_true,
	strategy_5_qsr_weighted,
	monotonicity_violations,
)

__all__ = [
	"strategy_5_qsr_weighted",
	"strategy_1_count_true",
	"monotonicity_violations",
]
