"""ids_calib_sample (worker ABI 24, 02/11/2026): the single-cluster binary SEARCH
fitness calibrates each genome x fold threshold on a fixed stratified subsample of
the fold's train rows (drawn in Rust, adaptive/calib_sample.rs). Pinned here:
  - registry / resolution / validation-cache key (pure Python);
  - through the real wheel (ABI >= 24): absent == 0 == oversized is bit-identical
    on the k-fold search path, a set sample is deterministic, the validation path
    (evaluate_genomes_full_hybrid) never changes, and multi-cluster caches refuse.
The Rust twins: calib_sample_tests.rs + adaptive::calib_sample::tests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "wnn"))

from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
from wnn.ram.experiments.ids_param_resolution import resolve_calib_sample
from wnn.ram.experiments.params import KNOWN_PARAMS
from wnn.ram.experiments.validation_cache_key import build_validation_cache_key


def _abi() -> int:
	try:
		import ram_accelerator
		return int(getattr(ram_accelerator, "ABI_VERSION", 0))
	except ImportError:
		return 0


needs_abi24 = pytest.mark.skipif(_abi() < 24, reason="needs ram_accelerator ABI >= 24")
SINGLE = {"ids_classification": "binary", "ids_single_cluster": True}


# ---- params / resolution (pure Python) ---------------------------------------

def test_registered():
	assert "ids_calib_sample" in KNOWN_PARAMS


@pytest.mark.parametrize("params, want", [({}, 0), ({"ids_calib_sample": None}, 0), ({"ids_calib_sample": 0}, 0),
	({**SINGLE, "ids_calib_sample": 200_000}, 200_000)])
def test_resolution(params, want):
	assert resolve_calib_sample(params) == want
	assert IdsEvaluatorSettings.from_params(params).calib_sample == want


@pytest.mark.parametrize("value", [-1, 1.5, "200000", True])
def test_bad_values_raise(value):
	with pytest.raises(ValueError):
		resolve_calib_sample({**SINGLE, "ids_calib_sample": value})


@pytest.mark.parametrize("params", [
	{"ids_classification": "binary", "ids_single_cluster": False},
	{"ids_classification": "multi", "ids_single_cluster": False},
	{"ids_classification": "hierarchical", "ids_single_cluster": True},
	{},  # ids_single_cluster defaults to False
])
def test_non_single_cluster_flows_refuse_a_sample(params):
	with pytest.raises(ValueError, match="single-cluster"):
		resolve_calib_sample({**params, "ids_calib_sample": 1000})
	assert resolve_calib_sample({**params, "ids_calib_sample": 0}) == 0


def test_validation_cache_key_ignores_it():
	base = {**SINGLE, "ids_dataset": "cicids2017"}
	assert build_validation_cache_key(base, True) == build_validation_cache_key({**base, "ids_calib_sample": 200_000}, True)


# ---- through the wheel ---------------------------------------------------------

TIB = 24


def _cache(single_cluster: bool = True):
	import ram_accelerator
	rng = np.random.default_rng(7)
	def part(n):
		x = rng.integers(0, 2, size=(n, TIB), dtype=np.uint8)
		y = ((x[:, 0] & x[:, 1]) | (x[:, 2] & x[:, 3] & x[:, 4])).astype(np.int64)
		noise = rng.random(n) < 0.1
		y[noise] = rng.integers(0, 2, size=noise.sum())
		return np.ascontiguousarray(np.packbits(x, axis=1).ravel()), [int(v) for v in y]
	(xt, yt), (xe, ye) = part(3000), part(1000)
	cache = ram_accelerator.IDSCacheWrapper.new_from_numpy(
		train_features=xt, train_labels=yt, eval_features=xe, eval_labels=ye, num_classes=2,
		total_features=TIB, num_parts=5, num_negatives=1, seed=42, single_cluster=single_cluster,
	)
	cache.set_memory_mode(0, 42)  # TERNARY: deterministic on every route without OI
	strata = np.asarray([0 if v == 0 else 1 + (i % 3) for i, v in enumerate(yt)], dtype=np.int64)
	return cache, strata


def _genome():
	conns = [(n * 5 + k * 3) % TIB for n in range(6) for k in range(8)]
	return [8] * 6, [6], conns


def _search(cache):
	b, npc, c = _genome()
	return [cache.evaluate_genomes_kfold_mean_hybrid(b, npc, c, 1, [0, 1, 2, 3, 4], 0.5, 1.0, 0)[0][:5]] + [
		cache.evaluate_genomes_kfold_hybrid(b, npc, c, 1, f, 0.5, 1.0, 0)[0][:5] for f in range(5)]


def _full(cache):
	b, npc, c = _genome()
	return cache.evaluate_genomes_full_hybrid(b, npc, c, 1, 0.5, 1.0, 0)[0][:5]


@needs_abi24
def test_absent_zero_oversized_bit_identical():
	want = _search(_cache()[0])
	for sample in (0, 3000, 10**7):
		cache, strata = _cache()
		cache.set_calib_sample(sample, 42, strata)
		assert cache.calib_sample() == sample
		assert _search(cache) == want


@needs_abi24
def test_set_sample_is_deterministic_and_validation_untouched():
	a, strata = _cache()
	a.set_calib_sample(500, 42, strata)
	b, _ = _cache()
	b.set_calib_sample(500, 42, strata)
	assert _search(a) == _search(a) == _search(b)
	assert _full(a) == _full(_cache()[0])


@needs_abi24
def test_multi_cluster_and_bad_strata_refuse():
	two, _ = _cache(single_cluster=False)
	with pytest.raises(ValueError, match="single-cluster"):
		two.set_calib_sample(500, 42)
	two.set_calib_sample(0, 42)  # 0 is always fine
	one, strata = _cache()
	with pytest.raises(ValueError, match="strata"):
		one.set_calib_sample(500, 42, strata[:-1])
