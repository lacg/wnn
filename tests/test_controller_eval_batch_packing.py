"""Sub-batch packing for the controller evaluator (25/09/2026).

The old width was budget / (800 B x the LARGEST genome), so one BITS-stage
outlier pinned the whole population to width 2. Packing now charges each
genome its OWN cells and keeps sub-batches CONTIGUOUS, which is what keeps
per-genome seeds — and therefore every metric — bit-identical to the
unbatched path. These tests pin both halves: the packing arithmetic, and the
seed/metric parity across any choice of boundaries.
"""

import pytest

import ram_accelerator
from wnn.control import cancel_state
from wnn.control.evaluator import ControllerEvaluator

CAP = ControllerEvaluator.EVAL_BUDGET_BYTES // ControllerEvaluator.EVAL_BYTES_PER_CELL


class _Cells:
	"""Stand-in for a GenomeCells handle: a count, and an empty warm-start."""

	def __init__(self, n):
		self.n = n

	def cell_count(self):
		return self.n

	def to_triples(self):
		return ([], [])


class _Controller:
	"""What training returns: its identity + seed, and an empty fold export."""

	def __init__(self, gid, seed):
		self.gid = gid
		self.seed = seed

	def export_cells(self):
		return ([], [])


class _Genome:
	def __init__(self, gid, cells=None):
		self.gid = gid
		self.cells = None if cells is None else _Cells(cells)


class _Spec:
	def __init__(self, mode):
		self.mode = mode
		self.state_neurons = 0

	def memory_mode_int(self):
		return self.mode


def _sizer(mode=3):
	"""An evaluator with only what the packing path reads (BINARY = mode 3)."""
	ev = ControllerEvaluator.__new__(ControllerEvaluator)
	ev.spec = _Spec(mode)
	return ev


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
	monkeypatch.delenv("WNN_CTRL_EVAL_BATCH", raising=False)
	monkeypatch.delenv("WNN_STATE_SPLIT", raising=False)


def _check_contiguous(bounds, n):
	assert bounds[0][0] == 0 and bounds[-1][1] == n
	for (_, e), (s, _) in zip(bounds, bounds[1:]):
		assert e == s


def test_one_outlier_no_longer_pins_the_population():
	"""The measured BITS case: 50 genomes of ~870k with one 3.26M outlier."""
	cells = [870_000] * 49 + [3_264_079]
	bounds = _sizer()._eval_batch_bounds([_Genome(i, c) for i, c in enumerate(cells)])
	_check_contiguous(bounds, 50)
	widths = [e - s for s, e in bounds]
	old_width = CAP // 3_264_079
	assert old_width == 2
	assert max(widths) >= 9, widths
	assert len(bounds) <= 7, widths


def test_every_batch_fits_the_budget_unless_alone():
	cells = [200_000, 5_000_000, 3_000_000, 900_000, 9_000_000, 100, 4_000_000, 4_000_000]
	gs = [_Genome(i, c) for i, c in enumerate(cells)]
	ev = _sizer()
	est = ev._genome_cell_estimates(gs)
	bounds = ev._eval_batch_bounds(gs)
	_check_contiguous(bounds, len(gs))
	for s, e in bounds:
		assert e - s == 1 or sum(est[s:e]) <= CAP


def test_genome_over_budget_runs_alone():
	gs = [_Genome(0, 500_000), _Genome(1, CAP * 2), _Genome(2, 500_000)]
	assert _sizer()._eval_batch_bounds(gs) == [(0, 1), (1, 2), (2, 3)]


def test_unmeasured_genome_is_charged_the_largest_measured():
	gs = [_Genome(0, 700_000), _Genome(1, None), _Genome(2, 0), _Genome(3, 2_500_000)]
	assert _sizer()._genome_cell_estimates(gs) == [700_000, 2_500_000, 2_500_000, 2_500_000]


def test_mode_floors():
	gs = [_Genome(0, 10), _Genome(1, None)]
	assert _sizer(mode=3)._genome_cell_estimates(gs) == [200_000, 200_000]
	assert _sizer(mode=0)._genome_cell_estimates(gs) == [7_000_000, 7_000_000]


def test_whole_population_in_one_batch_when_it_fits():
	"""Small genomes are charged the 200k light-mode floor: 40 fit, the 41st splits —
	exactly the width the old N x max rule gave for a uniform population."""
	gs = [_Genome(i, 100_000) for i in range(40)]
	assert _sizer()._eval_batch_bounds(gs) == [(0, 40)]
	gs.append(_Genome(40, 100_000))
	assert _sizer()._eval_batch_bounds(gs) == [(0, 40), (40, 41)]


def test_empty_population():
	assert len(_sizer()._eval_batch_bounds([])) <= 1


def test_env_override_is_a_fixed_width(monkeypatch):
	monkeypatch.setenv("WNN_CTRL_EVAL_BATCH", "4")
	gs = [_Genome(i, 9_000_000) for i in range(10)]
	assert _sizer()._eval_batch_bounds(gs) == [(0, 4), (4, 8), (8, 10)]


# ---- parity: any boundaries -> identical seeds and metrics -------------------

def _recording_evaluator(bounds_fn, crn, seen):
	"""Evaluator whose training records (genome, seed) and whose score is a pure
	function of that record — so a changed seed changes the metric."""
	ev = ControllerEvaluator.__new__(ControllerEvaluator)
	ev.seed = 11
	ev.score_crn = crn
	ev.num_eval_folds = 5
	ev._generation = -1
	ev.fixed_axes = None
	ev.axis_curriculum_gens = 0
	ev._ensure_ga_ready = lambda: None
	ev._advance_fold = lambda: None
	ev._shape_key = lambda g: 0
	ev._eval_batch_bounds = bounds_fn
	ev._materialize = lambda g: (g, None, None)

	def train(spec, sc, oc, init_s, init_o, seed):
		seen.append((spec.gid, seed))
		return (_Controller(spec.gid, seed), {})
	ev._train_core = train
	ev._score_fitness = lambda controllers, keys: [
		(c.gid * 1000.0 + c.seed, {"stable_rate": 0.5,
		                            "mean_attitude_error_deg": c.seed % 97})
		for c in controllers
	]
	return ev


@pytest.fixture
def _python_train_path():
	"""Force the per-genome Python train path the recording stub provides."""
	import wnn.control.evaluator as _ev_mod
	real = _ev_mod._rust_dagger_enabled
	real_cancelled = ram_accelerator.is_cancelled
	_ev_mod._rust_dagger_enabled = lambda: False
	ram_accelerator.is_cancelled = lambda: False
	cancel_state.reset_sigterm()
	yield
	_ev_mod._rust_dagger_enabled = real
	ram_accelerator.is_cancelled = real_cancelled


@pytest.mark.parametrize("crn", [False, True])
@pytest.mark.parametrize("cuts", [[], [3], [1, 2, 3, 4, 5, 6], [5], [2, 7]])
def test_any_boundaries_give_identical_seeds_and_metrics(_python_train_path, crn, cuts):
	"""Position seeds (legacy) are the hard case: genome gi must train on the SAME
	seed whether it runs in batch 0 or batch 3."""
	gs = [_Genome(i, 100_000) for i in range(8)]
	one = lambda sub: [(0, len(sub))]
	edges = [0] + cuts + [8]
	split = lambda sub: ([(a, b) for a, b in zip(edges, edges[1:])]
	                     if len(sub) == 8 else [(0, len(sub))])

	seen_whole, seen_split = [], []
	whole = _recording_evaluator(one, crn, seen_whole).evaluate_batch(gs)
	parts = _recording_evaluator(split, crn, seen_split).evaluate_batch(gs)

	assert sorted(seen_whole) == sorted(seen_split)
	assert [m.reward for m in whole] == [m.reward for m in parts]
	assert [m.mean_attitude_error_deg for m in whole] == [m.mean_attitude_error_deg for m in parts]
