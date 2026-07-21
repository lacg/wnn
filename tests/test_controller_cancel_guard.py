"""Unit tests for the controller evaluate_batch CANCEL-GUARD (01/06/2026).

Mirrors the IDS F1=0.49 fix: a cancel flag set during train/score leaves
controllers UNTRAINED. We must (a) honor a PROPER cancel (real SIGTERM →
sentinels so the GA unwinds + the curriculum dumps/resumes), and (b) treat a
SPURIOUS cancel (Rust flag set with no signal to this process) as recoverable —
reset + retry up to 3×, then raise loudly rather than ever return untrained
garbage.

These tests bypass the heavy ControllerEvaluator.__init__ and stub the
train/score methods, so they run in milliseconds with no GPU/training.
"""

import math
import types

import pytest

import ram_accelerator
from wnn.control import cancel_state
from wnn.control.evaluator import ControllerEvaluator
from wnn.ram.metrics import Metrics


def _make_evaluator(scored_reward=2.5, scored_stable=0.8, scored_err=3.0):
	"""A ControllerEvaluator with __init__ bypassed and only the attributes /
	methods the evaluate_batch guard path touches, stubbed to be cheap."""
	ev = ControllerEvaluator.__new__(ControllerEvaluator)
	ev.seed = 7
	ev.max_train_workers = 1
	# _evaluate_core attributes the guard path reads (kept current with the
	# production path — this stub going stale is exactly how these three tests
	# silently broke: fitness_seeds was subsumed by num_eval_folds, and the H4
	# axis mask added _generation / fixed_axes / axis_curriculum_gens reads).
	ev.num_eval_folds = 1
	ev._generation = -1
	ev.fixed_axes = None
	ev.axis_curriculum_gens = 0          # falsy -> full 3-axis mask
	ev._ensure_ga_ready = lambda: None
	ev._advance_fold = lambda: None
	ev._shape_key = lambda g: 0
	ev._eval_batch_size = lambda gs: max(1, len(gs))
	# Fallback (non-Rust-batch) train path: materialize -> _train_core per fold.
	ev._materialize = lambda g: (None, None, None)
	ev._train_core = lambda spec, sc, oc, init_s, init_o, seed: (object(), {})
	ev._score_grouped = lambda controllers, keys: [
		(scored_reward, {"stable_rate": scored_stable,
		                 "mean_attitude_error_deg": scored_err})
		for _ in controllers
	]
	return ev


@pytest.fixture(autouse=True)
def _clean_cancel_state():
	"""Each test starts from a clean slate and restores the real Rust fns.

	Also forces the PER-GENOME fallback train path: the Rust batched trainer is
	default-ON via env, and the stub deliberately provides only the fallback's
	surface (_materialize/_train_core) — the guard logic under test is shared
	by both paths."""
	import wnn.control.evaluator as _ev_mod
	real_rust = _ev_mod._rust_dagger_enabled
	_ev_mod._rust_dagger_enabled = lambda: False
	cancel_state.reset_sigterm()
	real_is = ram_accelerator.is_cancelled
	real_reset = ram_accelerator.reset_cancel_flag
	try:
		ram_accelerator.reset_cancel_flag()
	except Exception:
		pass
	yield
	_ev_mod._rust_dagger_enabled = real_rust
	ram_accelerator.is_cancelled = real_is
	ram_accelerator.reset_cancel_flag = real_reset
	cancel_state.reset_sigterm()
	try:
		ram_accelerator.reset_cancel_flag()
	except Exception:
		pass


def test_clean_eval_returns_real_metrics():
	ram_accelerator.is_cancelled = lambda: False
	ev = _make_evaluator(scored_reward=2.5, scored_stable=0.8, scored_err=3.0)
	out = ev.evaluate_batch([object(), object()])
	assert len(out) == 2
	for m in out:
		assert m.ce == pytest.approx(-2.5)       # ce = -reward
		assert m.fitness == pytest.approx(2.5)
		assert m.acc == pytest.approx(0.8)
		assert m.mean_attitude_error_deg == pytest.approx(3.0)


def test_spurious_cancel_recovers_after_retry():
	# Cancel flag reads True on the first poll, then False after the guard
	# resets it → one retry, then a clean result.
	state = {"cancelled": True}
	ram_accelerator.is_cancelled = lambda: state["cancelled"]
	ram_accelerator.reset_cancel_flag = lambda: state.__setitem__("cancelled", False)
	ev = _make_evaluator(scored_reward=1.0)
	out = ev.evaluate_batch([object()])
	assert out[0].ce == pytest.approx(-1.0)      # recovered, real metrics
	assert out[0].fitness != float("-inf")


def test_spurious_cancel_persists_raises_loudly():
	# Flag stays set despite resets, no SIGTERM → after 3 tries, raise.
	ram_accelerator.is_cancelled = lambda: True
	ram_accelerator.reset_cancel_flag = lambda: None
	ev = _make_evaluator()
	with pytest.raises(RuntimeError, match="SPURIOUS"):
		ev.evaluate_batch([object()])


def test_proper_cancel_returns_sentinels():
	# A real SIGTERM (marked in cancel_state) + flag set → sentinels, NOT a
	# retry and NOT a raise. The GA ranks ce=+inf last and unwinds.
	ram_accelerator.is_cancelled = lambda: True
	cancel_state.mark_sigterm(15)
	ev = _make_evaluator()
	out = ev.evaluate_batch([object(), object(), object()])
	assert len(out) == 3
	for m in out:
		assert m.ce == float("inf")
		assert m.fitness == float("-inf")
		assert m.mean_attitude_error_deg == pytest.approx(180.0)


def test_proper_cancel_beats_spurious_branch():
	# Even if reset would clear it, a PROPER cancel must short-circuit to
	# sentinels on the first detection (no retry attempts consumed).
	calls = {"reset": 0}
	ram_accelerator.is_cancelled = lambda: True
	ram_accelerator.reset_cancel_flag = lambda: calls.__setitem__("reset", calls["reset"] + 1)
	cancel_state.mark_sigterm(2)
	ev = _make_evaluator()
	out = ev.evaluate_batch([object()])
	assert out[0].ce == float("inf")
	assert calls["reset"] == 0   # never entered the spurious-retry path
