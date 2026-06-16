"""Unit tests for magnitude-aware patience (controller fitness redesign (a),
16/06/2026 — docs/controller_fitness_patience_redesign.md).

The rank-WHM fitness is magnitude-blind: a real err°/stable% jump barely moves
it, so the patience tracker watching the WHM mis-early-stops. check_magnitude()
instead watches the controller's PHYSICAL metrics (err° down, stable% up) and
recovers patience PROPORTIONAL to the size of the real gain.

Synthetic trajectories (the design's three scenarios) — no GPU/training, runs in
milliseconds:
  1. flat  → drains to stop in `patience` checks
  2. jump  → a gen-23-style improvement recovers patience proportionally
  3. worse → both metrics regress → drains by the floor (+1)
"""

import pytest

from wnn.ram.strategies.connectivity.framework.early_stopping import (
	EarlyStoppingConfig,
	EarlyStoppingTracker,
)


def _tracker(patience=3, check_interval=1, delta=0.05, rho_cap=0.0):
	cfg = EarlyStoppingConfig(
		patience=patience,
		check_interval=check_interval,
		magnitude_aware=True,
		mag_eps_err=0.5,
		mag_stable_offset=0.05,
		mag_delta=delta,
		mag_rho_cap=rho_cap,
	)
	t = EarlyStoppingTracker(cfg, logger=lambda _m: None, method_name="Test")
	t.reset(initial_fitness=1.0)
	return t


def test_first_check_seeds_watermarks_no_drain():
	"""The first check has nothing to compare against — seed, return False, no drain."""
	t = _tracker()
	assert t.check_magnitude(0, best_err_deg=10.0, best_stable=0.2) is False
	assert t._patience_counter == 0
	assert t._best_err_deg == 10.0 and t._best_stable == 0.2


def test_flat_drains_to_stop_in_patience_checks():
	"""Flat err°/stable° → ρ≈1 → counter +1 each check → stop after `patience`."""
	t = _tracker(patience=3, check_interval=1)
	assert t.check_magnitude(0, 10.0, 0.2) is False  # seed
	assert t.check_magnitude(1, 10.0, 0.2) is False  # counter 1
	assert t.check_magnitude(2, 10.0, 0.2) is False  # counter 2
	assert t.check_magnitude(3, 10.0, 0.2) is True   # counter 3 ≥ patience → stop
	assert t._patience_counter >= 3


def test_jump_recovers_patience_proportionally():
	"""A stable 20%→70% jump (ρ≈3) recovers a built-up counter near to zero."""
	t = _tracker(patience=5, check_interval=1)
	t.check_magnitude(0, 10.0, 0.2)   # seed
	t.check_magnitude(1, 10.0, 0.2)   # flat → counter 1
	t.check_magnitude(2, 10.0, 0.2)   # flat → counter 2
	assert t._patience_counter == 2
	# Jump: stable 0.2→0.7 → ρ_stb = 0.75/0.25 = 3.0 → counter = max(0, 2-3) = 0
	stop = t.check_magnitude(3, 10.0, 0.7)
	assert stop is False
	assert t._patience_counter == pytest.approx(0.0)
	assert t._best_stable == 0.7  # watermark ratcheted up


def test_err_halving_recovers_about_two():
	"""err° halved (10→5) → ρ_err=2 → recover ~2 patience."""
	t = _tracker(patience=5, check_interval=1)
	t.check_magnitude(0, 10.0, 0.2)   # seed
	t.check_magnitude(1, 10.0, 0.2)   # counter 1
	t.check_magnitude(2, 10.0, 0.2)   # counter 2
	t.check_magnitude(3, 10.0, 0.2)   # counter 3
	assert t._patience_counter == 3
	# err 10→5 → ρ_err = 10/5 = 2 → counter = max(0, 3-2) = 1
	t.check_magnitude(4, 5.0, 0.2)
	assert t._patience_counter == pytest.approx(1.0)
	assert t._best_err_deg == 5.0


def test_worse_check_drains():
	"""Both metrics regress → ρ<1 → counter +1 (drain by the floor)."""
	t = _tracker(patience=5, check_interval=1)
	t.check_magnitude(0, 10.0, 0.2)   # seed
	# err 10→12 (worse), stable 0.2→0.1 (worse) → ρ<1 → +1
	t.check_magnitude(1, 12.0, 0.1)
	assert t._patience_counter == 1
	# watermarks must NOT regress (ratchet keeps the best)
	assert t._best_err_deg == 10.0 and t._best_stable == 0.2


def test_eps_err_floor_guards_div0_near_zero():
	"""err° at 0 must not div-by-zero; ρ_err is bounded by the cap."""
	t = _tracker(patience=4, check_interval=1)
	t.check_magnitude(0, 10.0, 0.2)   # seed
	t.check_magnitude(1, 10.0, 0.2)   # counter 1
	# err→0.0 → max(0, eps=0.5) → ρ_err = 10/0.5 = 20, capped at patience(4)
	stop = t.check_magnitude(2, 0.0, 0.2)
	assert stop is False
	assert t._patience_counter == pytest.approx(0.0)


def test_rho_cap_bounds_recovery():
	"""An explicit ρ_cap bounds a fluke ratio's recovery."""
	t = _tracker(patience=10, check_interval=1, rho_cap=2.0)
	t.check_magnitude(0, 10.0, 0.2)
	for it in range(1, 6):
		t.check_magnitude(it, 10.0, 0.2)   # drain to counter 5
	assert t._patience_counter == 5
	# Huge jump but cap=2.0 → recover only 2
	t.check_magnitude(6, 0.5, 0.99)
	assert t._patience_counter == pytest.approx(3.0)


def test_check_interval_gates_checks():
	"""With check_interval=3, only every 3rd (1-indexed) iteration evaluates."""
	t = _tracker(patience=3, check_interval=3)
	assert t.check_magnitude(0, 10.0, 0.2) is False  # (0+1)%3 != 0 → skip
	assert t.check_magnitude(1, 10.0, 0.2) is False  # skip
	assert t._best_err_deg is None                   # never seeded yet
	assert t.check_magnitude(2, 10.0, 0.2) is False  # (2+1)%3==0 → seed
	assert t._best_err_deg == 10.0


def test_magnitude_aware_off_by_default():
	"""The opt-in flag defaults off in both config layers (comparability guard)."""
	from wnn.ram.strategies.connectivity.framework.configs import GAConfig
	assert EarlyStoppingConfig().magnitude_aware is False
	assert GAConfig().magnitude_aware_patience is False
