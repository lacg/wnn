"""ids_vote_share — the IDS-56 MIN-SHARE vote (worker ABI 28, 09/11/2026).

`ids_write_rule="vote"` + `ids_vote_share=θ`: an own-class visit adds
(1−θ)·w, any other θ·w to the cell's tally; TRUE iff tally > 0
(docs/ids55_family_fix_decomposition.md, Lever B). θ = 0.5 / unset is the
plain vote (bit-identical to ABI 27); the param is invalid without "vote".
The Rust parity suites (hand cases, reference tally, CPU == Metal, the ABI-27
digest for θ = 0.5) live in cargo: vote_share_tests.rs.

Accelerator-backed tests skip unless the installed ram_accelerator is ABI >= 28.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "wnn"))
sys.path.insert(0, str(ROOT / "tests"))

from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.experiments.ids_param_resolution import resolve_vote_share
from wnn.ram.experiments.params import KNOWN_PARAMS
from wnn.ram.experiments.test_cache_key import build_test_cache_key


def _abi() -> int:
	try:
		import ram_accelerator
		return int(getattr(ram_accelerator, "ABI_VERSION", 0))
	except ImportError:
		return 0


needs_abi28 = pytest.mark.skipif(_abi() < 28, reason="needs ram_accelerator ABI >= 28")
VOTE = {"memory_mode": "BINARY", "ids_write_rule": "vote"}


# ---- params / resolution / cache key (pure Python) ---------------------------

def test_vote_share_is_a_known_param():
	assert "ids_vote_share" in KNOWN_PARAMS


def test_resolve_vote_share_defaults_and_grid():
	assert resolve_vote_share({}) is None
	assert resolve_vote_share(dict(VOTE)) is None
	assert resolve_vote_share(dict(VOTE, ids_vote_share=None)) is None
	assert resolve_vote_share(dict(VOTE, ids_vote_share=0.1)) == 0.1
	assert resolve_vote_share(dict(VOTE, ids_vote_share=0.5)) == 0.5
	assert resolve_vote_share(dict(VOTE, ids_vote_share=0.3000000001)) == 0.3
	for mode in ("TERNARY", "PLN"):
		assert resolve_vote_share(dict(VOTE, memory_mode=mode, ids_vote_share=0.25)) == 0.25


@pytest.mark.parametrize("theta", [0, 0.0, -0.1, 0.51, 1.0, 0.123, 0.005, float("nan"), float("inf"), "0.1", True])
def test_resolve_vote_share_rejects_out_of_range_off_grid_and_non_numbers(theta):
	with pytest.raises(ValueError):
		resolve_vote_share(dict(VOTE, ids_vote_share=theta))


def test_vote_share_requires_the_vote_writer():
	with pytest.raises(ValueError, match="only valid with ids_write_rule='vote'"):
		resolve_vote_share({"memory_mode": "BINARY", "ids_vote_share": 0.1})
	with pytest.raises(ValueError, match="only valid with ids_write_rule='vote'"):
		resolve_vote_share({"memory_mode": "BINARY", "ids_write_rule": "lattice", "ids_vote_share": 0.1})
	# vote on a QUAD-family mode is already refused by resolve_write_rule
	with pytest.raises(ValueError):
		resolve_vote_share({"memory_mode": "QUAD_WEIGHTED", "ids_write_rule": "vote", "ids_vote_share": 0.1})


def test_cache_key_unchanged_when_unset_or_half_and_split_when_set():
	base = {"ids_dataset": "unsw-nb15", "memory_mode": "BINARY"}
	lattice = build_test_cache_key(base, True)
	plain = build_test_cache_key(dict(base, ids_write_rule="vote"), True)
	half = build_test_cache_key(dict(base, ids_write_rule="vote", ids_vote_share=0.5), True)
	none = build_test_cache_key(dict(base, ids_write_rule="vote", ids_vote_share=None), True)
	tenth = build_test_cache_key(dict(base, ids_write_rule="vote", ids_vote_share=0.1), True)
	fifth = build_test_cache_key(dict(base, ids_write_rule="vote", ids_vote_share=0.2), True)
	assert "vs=" not in lattice and "vs=" not in plain
	assert half == plain == none  # θ = 0.5 trains the plain vote bit-identically
	assert "vs=0.1" in tenth and tenth != plain and tenth != lattice
	assert fifth != tenth
	with pytest.raises(ValueError):
		build_test_cache_key(dict(base, ids_vote_share=0.1), True)


def test_settings_carry_the_resolved_share():
	assert IdsEvaluatorSettings.from_params({}).vote_share is None
	s = IdsEvaluatorSettings.from_params(dict(VOTE, ids_vote_share=0.1))
	assert (s.write_rule, s.vote_share) == ("vote", 0.1)


# ---- accelerator-backed ------------------------------------------------------

BINARY, TERNARY, PLN, QUAD_WEIGHTED = 3, 0, 5, 2


def _evaluator(ds, single_cluster: bool, memory_mode: int, write_rule: str = "lattice", vote_share=None):
	from wnn.ram.architecture.ids_evaluator import IDSEvaluator
	return IDSEvaluator(
		dataset=ds, classification="binary", num_parts=1, empty_value=0.5,
		memory_mode=memory_mode, seed=3, run_seed=3, neuron_sample_rate=1.0,
		single_cluster=single_cluster, write_rule=write_rule, vote_share=vote_share,
	)


def _scores(ev, genome):
	e, t, v, _, _ = ev.evaluate_at_thresholds(genome, [-1.0, 0.5])
	return list(e), list(t), list(v)


@needs_abi28
@pytest.mark.parametrize("mode", [BINARY, TERNARY, PLN])
def test_end_to_end_half_is_plain_vote_and_unset_is_lattice(mode):
	"""Tiny synthetic IDS run through IDSEvaluator (CPU/Metal hybrid, the
	production entry point): θ = 0.5 reproduces "vote" exactly, θ unset
	reproduces "lattice"/"vote" exactly, θ = 0.1 trains something else."""
	from test_ids_write_rule_and_binary_k2 import _dataset, _genome
	ds = _dataset()
	g = _genome(1)
	lattice = _scores(_evaluator(ds, True, mode), g)
	vote = _scores(_evaluator(ds, True, mode, "vote"), g)
	assert _scores(_evaluator(ds, True, mode, "vote", 0.5), g) == vote
	assert _scores(_evaluator(ds, True, mode, "vote", None), g) == vote
	assert _scores(_evaluator(ds, True, mode, "lattice", None), g) == lattice
	assert _scores(_evaluator(ds, True, mode, "vote", 0.1), g) != vote
	# GA-search metrics (single- and two-cluster) follow the same identity.
	for sc, k in ((True, 1), (False, 2)):
		plain = _evaluator(ds, sc, mode, "vote").evaluate_batch_full([_genome(k)])[0]
		half = _evaluator(ds, sc, mode, "vote", 0.5).evaluate_batch_full([_genome(k)])[0]
		assert (half.ce, half.acc, half.f1, half.fpr) == (plain.ce, plain.acc, plain.f1, plain.fpr)


@needs_abi28
def test_min_share_raises_the_firing_rate_vs_plain_vote():
	"""θ = 0.1 commits TRUE wherever attacks are not a clear minority, so at
	the fixed 0.5 threshold it fires at least as often as the plain vote on
	the same data (FPR >= plain vote's, recall >= too)."""
	from test_ids_write_rule_and_binary_k2 import _dataset, _genome
	ds = _dataset()
	g = _genome(1)
	_, _, _, (_, fixed_vote), _ = _evaluator(ds, True, BINARY, "vote").evaluate_at_thresholds(g, [-1.0, 0.5])
	_, _, _, (_, fixed_ms), _ = _evaluator(ds, True, BINARY, "vote", 0.1).evaluate_at_thresholds(g, [-1.0, 0.5])
	assert fixed_ms.fpr >= fixed_vote.fpr


@needs_abi28
def test_set_write_rule_validates_the_share():
	from test_ids_write_rule_and_binary_k2 import _dataset
	ev = _evaluator(_dataset(), True, BINARY)
	cache = ev._cache
	cache.set_write_rule("vote", vote_share=0.1)
	cache.set_write_rule("vote", vote_share=0.5)
	cache.set_write_rule("vote")
	cache.set_write_rule("lattice")
	for bad in (0.0, 0.6, 0.123, float("nan")):
		with pytest.raises(ValueError):
			cache.set_write_rule("vote", vote_share=bad)
	with pytest.raises(ValueError, match="only valid"):
		cache.set_write_rule("lattice", vote_share=0.1)
	with pytest.raises(ValueError):
		_evaluator(_dataset(), True, QUAD_WEIGHTED, "vote", 0.1)
