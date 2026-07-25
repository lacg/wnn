"""R14b — regression tests for EarlyStoppingTracker._initial_fitness.

The bug this guards: the dashboard's "baseline CE" was None on every FRESH run.
Callers poked a `_best_fitness` attribute onto the tracker from the RESUME path
only, and the three read sites (generic_ga.py:598, generic_ts.py:609,
generic_sa.py:292) went through a hasattr/getattr guard that silently yielded
None when it was absent — so a fresh run reported no baseline at all and nobody
saw an error.

`_initial_fitness` is the run's STARTING fitness, owned by the tracker, set in
reset() and never mutated after. It exists precisely BECAUSE `_baseline` cannot
serve that role: reset_trend() reassigns `_baseline` to the rolling trend mean.

Run:  /Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python tests/test_early_stopper_initial_fitness.py
"""

from __future__ import annotations

from wnn.ram.strategies.connectivity.framework.early_stopping import (
	EarlyStoppingConfig, EarlyStoppingTracker,
)


def _mk(patience: int = 3, check_interval: int = 1) -> EarlyStoppingTracker:
	cfg = EarlyStoppingConfig(patience=patience, check_interval=check_interval,
	                          min_improvement_pct=0.02)
	return EarlyStoppingTracker(cfg, logger=lambda _m: None, method_name="Test")


def _read_as_consumer(tracker) -> float | None:
	"""Exactly how generic_ga/ts/sa read the baseline for the dashboard."""
	return tracker._initial_fitness


def test_absent_before_reset():
	"""Fresh tracker has no starting fitness yet — but the attribute must EXIST
	and be None, not be missing. A missing attribute is what let the old
	hasattr-guarded read fail silently."""
	t = _mk()
	assert hasattr(t, "_initial_fitness"), \
		"_initial_fitness must be declared in __init__, not poked on by callers"
	assert t._initial_fitness is None, "must start None until reset() supplies it"
	print("✓ absent_before_reset")


def test_reset_sets_both_baseline_and_initial():
	t = _mk()
	t.reset(initial_fitness=12.5)
	assert t._initial_fitness == 12.5, "reset must record the starting fitness"
	assert t._baseline == 12.5, "reset must also seed the working baseline"
	assert _read_as_consumer(t) == 12.5, "consumer read path must see it on a FRESH run"
	print("✓ reset_sets_both_baseline_and_initial")


def test_reset_trend_moves_baseline_but_not_initial():
	"""THE invariant. reset_trend() repurposes _baseline as the rolling trend
	mean — which is exactly why _baseline cannot be the stable reference and
	_initial_fitness has to exist separately."""
	t = _mk()
	t.reset(initial_fitness=12.5)
	t.reset_trend([4.0, 6.0, 8.0])          # mean 6.0
	assert t._baseline == 6.0, \
		f"reset_trend must retarget _baseline to the trend mean, got {t._baseline}"
	assert t._initial_fitness == 12.5, (
		f"_initial_fitness must survive reset_trend unchanged, got {t._initial_fitness} "
		f"— if this drifts, the dashboard baseline silently tracks the trend instead "
		f"of the run's start")
	print("✓ reset_trend_moves_baseline_but_not_initial")


def test_check_trend_iterations_do_not_touch_initial():
	"""Running the optimizer must never mutate the starting reference."""
	t = _mk(patience=10, check_interval=1)
	t.reset(initial_fitness=100.0)
	t.reset_trend([100.0, 100.0])
	for it in range(12):
		t.check_trend(it, [100.0 - it, 101.0 - it, 102.0 - it])
	assert t._initial_fitness == 100.0, (
		f"_initial_fitness mutated during the run: {t._initial_fitness}")
	print("✓ check_trend_iterations_do_not_touch_initial")


def test_check_iterations_do_not_touch_initial():
	"""Same for the non-trend single-best check path."""
	t = _mk(patience=10, check_interval=1)
	t.reset(initial_fitness=50.0)
	for it in range(12):
		t.check(it, 50.0 - it * 0.1)
	assert t._initial_fitness == 50.0, (
		f"_initial_fitness mutated during check(): {t._initial_fitness}")
	print("✓ check_iterations_do_not_touch_initial")


def test_restore_only_touches_patience():
	"""restore() is the explicit resume counterpart for patience — it must not
	disturb the starting fitness (callers used to poke _patience_counter
	directly, which is what it replaced)."""
	t = _mk(patience=5)
	t.reset(initial_fitness=7.25)
	t.restore(patience_counter=3)
	assert t._patience_counter == 3, "restore must set the patience counter"
	assert t._initial_fitness == 7.25, "restore must not disturb _initial_fitness"
	t.restore(patience_counter=-4)
	assert t._patience_counter == 0, "restore must clamp negatives to 0"
	assert t._initial_fitness == 7.25, "clamped restore must still leave initial alone"
	print("✓ restore_only_touches_patience")


def test_resume_path_can_rebaseline():
	"""The resume path (generic_ga.py:339) deliberately reassigns
	_initial_fitness to the restored population's best, so further improvement is
	measured against where the resumed run actually starts — not the original
	cold-start fitness. That write must stick."""
	t = _mk()
	t.reset(initial_fitness=12.5)
	t.restore(patience_counter=2)
	t._initial_fitness = 9.0                # what the resume branch does
	assert _read_as_consumer(t) == 9.0, "resume re-baseline must be visible to consumers"
	t.reset_trend([3.0, 5.0])
	assert t._initial_fitness == 9.0, "resume baseline must survive a later reset_trend"
	print("✓ resume_path_can_rebaseline")


def test_delta_baseline_is_computable_on_fresh_run():
	"""The end-to-end shape of the original bug: on a FRESH run the consumer
	computed `delta_baseline = best - baseline_ce` and got None because
	baseline_ce was None. It must now be a real number."""
	t = _mk()
	t.reset(initial_fitness=20.0)
	baseline_ce = _read_as_consumer(t)
	best_fitness = 14.0
	delta_baseline = (best_fitness - baseline_ce) if baseline_ce is not None else None
	assert delta_baseline is not None, \
		"fresh run must yield a numeric baseline delta (the original bug)"
	assert delta_baseline == -6.0, f"expected -6.0, got {delta_baseline}"
	print("✓ delta_baseline_is_computable_on_fresh_run")


if __name__ == "__main__":
	test_absent_before_reset()
	test_reset_sets_both_baseline_and_initial()
	test_reset_trend_moves_baseline_but_not_initial()
	test_check_trend_iterations_do_not_touch_initial()
	test_check_iterations_do_not_touch_initial()
	test_restore_only_touches_patience()
	test_resume_path_can_rebaseline()
	test_delta_baseline_is_computable_on_fresh_run()
	print("\nAll _initial_fitness regression tests passed.")
