"""IDS receptive fields (worker ABI 27): ids_receptive_fields = off | family.

Each neuron wires to one feature FAMILY (the family holding the majority of its
connections) with at most floor(ids_rf_cross x bits) connections elsewhere. Off
is the default and must leave every RNG stream and genome bit-identical. The
Rust operator tests (ram_core receptive_fields*.rs, neighbor_search_rf_tests.rs,
worker receptive_fields_genesis_tests.rs) cover the GA/TS/genesis operators;
these cover the family maps, the encoded-bit map (dedupe on/off), the param
plumbing and the Python creators.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ram_accelerator = pytest.importorskip("ram_accelerator")
if getattr(ram_accelerator, "ABI_VERSION", 0) < 27:
	pytest.skip("worker wheel predates ids_receptive_fields (ABI < 27)", allow_module_level=True)

from wnn.ids.feature_families import FAMILY_MAP_VERSION, bit_families, family_file, load_family_map  # noqa: E402
from wnn.ram.strategies.connectivity.receptive_fields import ReceptiveFields, project_if_on  # noqa: E402
from wnn.representations.thermometer import ThermometerEncoder  # noqa: E402

RANKINGS = ROOT / "src" / "wnn" / "ids" / "feature_rankings" / "train_only_v1"


# ---------------------------------------------------------------- family maps

def _every_list(dataset: str) -> list:
	"""Every feature list the map must cover: legacy constants + full train_only rankings."""
	from wnn.ids import cicids2017, ciciot2023
	from wnn.ids.dataset import TOP20_RF_FEATURES as unsw
	files = {"unsw-nb15": ["unsw-nb15_random_3way", "unsw-nb15_temporal_3way"],
		"cicids2017": ["cicids2017_random_3way"], "ciciot2023": ["ciciot2023_neto_subsample_random_3way"]}[dataset]
	legacy = {"unsw-nb15": [unsw], "cicids2017": [cicids2017.TOP20_RF_FEATURES],
		"ciciot2023": [ciciot2023.TOP20_RF_FEATURES, ciciot2023.TOP20_MI_8B_FEATURES, ciciot2023.TOP20_MI_96B_FEATURES]}[dataset]
	ranked = [[r["feature"] for r in json.loads((RANKINGS / f"{f}.json").read_text())["ranking"]] for f in files]
	return [f for lst in legacy + ranked for f in lst]


@pytest.mark.parametrize("dataset", ["unsw-nb15", "cicids2017", "ciciot2023"])
def test_every_listed_feature_maps_to_exactly_one_family(dataset):
	owner, families = load_family_map(dataset)
	missing = sorted(set(_every_list(dataset)) - set(owner))
	assert not missing, f"{dataset}: no family for {missing}"
	assert len(families) >= 5
	raw = json.loads(family_file(dataset).read_text())
	assert raw["version"] == "v2" and FAMILY_MAP_VERSION == "families_v2"
	assert all(spec["why"] for spec in raw["families"].values()), "every family documents its mechanism"


def test_a_feature_in_two_families_is_refused(tmp_path, monkeypatch):
	import wnn.ids.feature_families as ff
	bad = {"families": {"a": {"why": "x", "features": ["f"]}, "b": {"why": "y", "features": ["f"]}}}
	(tmp_path / "unsw-nb15.json").write_text(json.dumps(bad))
	monkeypatch.setattr(ff, "_DIR", tmp_path)
	with pytest.raises(ValueError, match="in both"):
		ff.load_family_map("unsw-nb15")


def test_ciciot_sizes_share_one_map():
	assert family_file("ciciot2023_neto_full") == family_file("ciciot2023_neto_subsample")


# ---------------------------------------------------------------- encoded-bit map

def _unsw_frame(n=3000, seed=0) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	sttl = rng.choice([0, 31, 62, 254], n).astype(float)
	return pd.DataFrame({
		"sttl": sttl, "dttl": rng.choice([0, 29, 252], n).astype(float),
		"state": rng.choice(["FIN", "INT", "CON", "REQ"], n),
		"tcprtt": np.where(rng.random(n) < 0.5, 0.0, rng.exponential(0.1, n)),
		"sbytes": rng.integers(0, 5000, n).astype(float), "dbytes": rng.integers(0, 9000, n).astype(float),
		"dur": rng.exponential(1.0, n), "Sintpkt": rng.exponential(5.0, n),
		"ct_srv_src": rng.integers(1, 30, n).astype(float),
		"proto": rng.choice(["tcp", "udp", "arp"], n), "service": rng.choice(["-", "http", "dns"], n),
		"ct_state_ttl": np.minimum(sttl, 2.0),   # a coarse copy-ish feature (dedupe shrinks it)
	})


@pytest.mark.parametrize("dedupe", [False, True])
def test_bit_map_covers_the_actual_encoded_layout(dedupe):
	df = _unsw_frame()
	enc = ThermometerEncoder(n_bits=16, dedupe=dedupe).fit(df)
	_, total = enc.transform(df)
	bit_family, names, _ = bit_families(enc, total, "unsw-nb15")
	assert len(bit_family) == total, "every encoded bit mapped"
	owner, _ = load_family_map("unsw-nb15")
	for feature, (start, end) in enc.feature_bit_ranges().items():
		assert set(bit_family[start:end]) <= {names.index(owner[feature])}
	assert set(names) == {owner[f] for f in enc.feature_names_}, "only families with encoded bits are indexed"
	if dedupe:
		assert total < 16 * 9 + 2 + 2 + 2, "dedupe dropped bits and the map follows them"


def test_flag_bit_belongs_to_its_feature():
	df = _unsw_frame()
	df.loc[::7, "dur"] = np.nan
	enc = ThermometerEncoder(n_bits=8, invalid_encoding="single_bit").fit(df)
	_, total = enc.transform(df)
	bit_family, names, _ = bit_families(enc, total, "unsw-nb15")
	start, end = enc.feature_bit_ranges()["dur"]
	assert {names[f] for f in bit_family[start:end]} == {"timing"}


def test_unmapped_feature_and_foreign_layout_are_refused():
	df = _unsw_frame().assign(mystery=1.0 * np.arange(3000))
	enc = ThermometerEncoder(n_bits=8).fit(df)
	_, total = enc.transform(df)
	with pytest.raises(ValueError, match="without a family"):
		bit_families(enc, total, "unsw-nb15")
	enc = ThermometerEncoder(n_bits=8).fit(_unsw_frame())
	with pytest.raises(ValueError, match="top20_split"):
		bit_families(enc, enc.total_bits + 5, "unsw-nb15")


# ---------------------------------------------------------------- projection

def _rf(cross=0.2) -> ReceptiveFields:
	bit_family = tuple([0] * 64 + [1] * 32 + [2] * 24 + [3] * 8)
	return ReceptiveFields(bit_family=bit_family, family_names=("a", "b", "c", "d"), cross=cross,
		version=FAMILY_MAP_VERSION, source="test")


def _genome(widths, seed=0):
	from wnn.ram.genome import ClusterGenome
	rng = np.random.default_rng(seed)
	conns = rng.integers(0, 128, size=sum(widths)).tolist()
	return ClusterGenome(bits_per_neuron=list(widths), neurons_per_cluster=[len(widths)], connections=conns)


def _assert_satisfied(rf, genome):
	"""Every neuron: exempt, or at most `budget` connections outside its held family."""
	fields = rf.neuron_fields(genome.bits_per_neuron, genome.connections)
	for w, (fam, in_family, budget, exempt) in zip(genome.bits_per_neuron, fields):
		assert exempt or (fam >= 0 and w - in_family <= budget), (w, fam, in_family, budget)


def test_projected_genome_satisfies_budget_and_is_a_fixed_point():
	rf = _rf()
	g = _genome([4 + i % 31 for i in range(300)])
	rf.project_genome(g, random.Random(1))
	_assert_satisfied(rf, g)
	before = list(g.connections)
	rf.project_genome(g, random.Random(2))
	assert g.connections == before


def test_off_draws_nothing_and_returns_the_same_genome():
	g = _genome([10] * 20)
	before = list(g.connections)
	rng = random.Random(5)
	state = rng.getstate()
	assert project_if_on(None, g, rng) is g
	assert g.connections == before and rng.getstate() == state


def test_a_changed_genome_drops_its_cached_metrics():
	from wnn.ram.metrics import IDSMetrics
	g = _genome([12] * 30)
	g.metrics = IDSMetrics(ce=0.1, acc=0.9, f1=0.9, fpr=0.1)
	_rf().project_genome(g, random.Random(3))
	assert g.metrics is None


def test_cross_must_leave_the_family_a_majority():
	with pytest.raises(ValueError):
		_rf(cross=0.5)


# ---------------------------------------------------------------- params / plumbing

def test_params_default_off_refuse_typos_and_inert_cross():
	from wnn.ram.experiments.ids_param_resolution import resolve_receptive_fields, resolve_rf_cross
	from wnn.ram.experiments.params import KNOWN_PARAMS
	assert {"ids_receptive_fields", "ids_rf_cross"} <= KNOWN_PARAMS
	assert resolve_receptive_fields({}) == "off" and resolve_rf_cross({}) == 0.2
	assert resolve_rf_cross({"ids_receptive_fields": "family", "ids_rf_cross": 0.3}) == 0.3
	with pytest.raises(ValueError):
		resolve_receptive_fields({"ids_receptive_fields": "families"})
	with pytest.raises(ValueError, match="no effect"):
		resolve_rf_cross({"ids_rf_cross": 0.3})
	with pytest.raises(ValueError):
		resolve_rf_cross({"ids_receptive_fields": "family", "ids_rf_cross": 0.5})


def test_validation_key_unchanged_when_off_and_versioned_when_on():
	from wnn.ram.experiments.test_cache_key import build_test_cache_key
	base = {"ids_dataset": "unsw-nb15", "ids_split": "random_3way", "memory_mode": "BINARY"}
	assert build_test_cache_key(base, True) == build_test_cache_key({**base, "ids_receptive_fields": "off"}, True)
	on = build_test_cache_key({**base, "ids_receptive_fields": "family"}, True)
	assert on != build_test_cache_key(base, True) and f"{FAMILY_MAP_VERSION}:0.2" in on


class _Dataset:
	def __init__(self, enc, total):
		self.encoder = enc
		self.X_train = type("X", (), {"total_bits": total})()


def test_factory_builds_rf_from_the_dataset_and_logs_provenance():
	from wnn.ram.experiments.IdsEvaluatorFactory import IdsEvaluatorFactory
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	df = _unsw_frame()
	enc = ThermometerEncoder(n_bits=8).fit(df)
	logs = []
	factory = IdsEvaluatorFactory(logs.append)
	off = IdsEvaluatorSettings.from_params({"ids_dataset": "unsw-nb15"})
	assert factory.receptive_fields(_Dataset(enc, enc.total_bits), off) is None and not logs
	on = IdsEvaluatorSettings.from_params({"ids_dataset": "unsw-nb15", "ids_receptive_fields": "family", "ids_rf_cross": 0.25})
	rf = factory.receptive_fields(_Dataset(enc, enc.total_bits), on)
	assert rf.cross == 0.25 and rf.total_input_bits == enc.total_bits
	assert logs and "[RF]" in logs[0] and FAMILY_MAP_VERSION in logs[0] and "unsw-nb15.json" in logs[0]


# ---------------------------------------------------------------- creators

def _arch_config(rf, tib=128):
	from wnn.ram.strategies.connectivity.architecture_config import ArchitectureConfig
	return ArchitectureConfig(num_clusters=2, min_bits=4, max_bits=30, min_neurons=5, max_neurons=40,
		optimize_bits=True, optimize_neurons=True, default_bits=8, default_neurons=10,
		total_input_bits=tib, receptive_fields=rf)


def _ga(rf, seed=11):
	from wnn.ram.strategies.connectivity.architecture_ga import ArchitectureGAStrategy
	from wnn.ram.strategies.connectivity.framework import GAConfig
	return ArchitectureGAStrategy(arch_config=_arch_config(rf), ga_config=GAConfig(population_size=4), seed=seed)


def test_ga_random_genomes_immigrants_and_mutants_are_projected():
	rf = _rf()
	ga = _ga(rf)
	pop = [ga.create_random_genome() for _ in range(4)]
	for g in pop + [ga.create_immigrant(pop), ga.mutate_genome(pop[0], 0.5), ga.crossover_genomes(pop[0], pop[1])]:
		_assert_satisfied(rf, g)


def test_ga_projects_seeds_before_any_rng_use():
	"""Flows 7452/7454 (08/11): a phased GA (train_subset_idx given) projects
	its seed population before anything seeded the RNG -> _rng was None."""
	rf = _rf()
	ga = _ga(rf)
	seed = _genome([20, 16, 12])
	assert ga._rng is None
	_assert_satisfied(rf, ga._project_rf(seed.clone()))


def test_ga_off_builds_the_same_genomes_as_before():
	"""Off: the config carries None, no projection seed is drawn -> the exact
	pre-ABI-27 draw sequence (compared with an RF strategy that consumed draws)."""
	a = [_ga(None, seed=3).create_random_genome() for _ in range(1)]
	b = [_ga(None, seed=3).create_random_genome() for _ in range(1)]
	assert a[0].connections == b[0].connections
	on = _ga(_rf(), seed=3)
	g_on = on.create_random_genome()
	assert g_on.bits_per_neuron == a[0].bits_per_neuron, "projection never changes the architecture"


def test_grid_genomes_are_projected_and_off_is_unchanged():
	from wnn.ram.strategies.connectivity.grid_search import GridSearchConfig, GridSearchStrategy
	rf = _rf()
	on = GridSearchStrategy(GridSearchConfig(num_clusters=2, total_input_bits=128, receptive_fields=rf), seed=9)
	_assert_satisfied(rf, on._create_genome(20, 16))
	off1 = GridSearchStrategy(GridSearchConfig(num_clusters=2, total_input_bits=128), seed=9)._create_genome(20, 16)
	from wnn.ram.genome import generate_connections
	expect = generate_connections([16] * 40, 128, random.Random(9).randint(0, 2**63))
	assert off1.connections == expect, "off grid genome = the original single draw"


# ---------------------------------------------------------------- merge_into (IDS-55 follow-up)

def _layout(dedupe: bool, max_bits: int = 34, cross: float = 0.2):
	from wnn.ids.feature_families import host_bits
	df = _unsw_frame()
	enc = ThermometerEncoder(n_bits=16, dedupe=dedupe).fit(df)
	_, total = enc.transform(df)
	return enc, total, bit_families(enc, total, "unsw-nb15", host_bits(max_bits, cross))


_V2_PARTNERS = {"ttl_state": "context", "handshake": "timing", "protocol": "context"}


def _merged_sizes(merges: list[str]) -> dict[str, tuple[int, str]]:
	""""family(Nb)->partner" -> {family: (N, partner)}."""
	out = {}
	for m in merges:
		head, partner = m.split("->")
		family, size = head.rstrip("b)").split("(")
		out[family] = (int(size), partner)
	return out


def test_small_families_merge_into_their_declared_partner_when_dedupe_shrinks_them():
	enc, total, (bit_family, names, merges) = _layout(dedupe=True)
	merged = _merged_sizes(merges)
	assert {f: p for f, (_, p) in merged.items()} == _V2_PARTNERS
	assert not set(_V2_PARTNERS) & set(names)
	owner, _ = load_family_map("unsw-nb15")
	for feature in enc.feature_names_:
		home = _V2_PARTNERS.get(owner[feature], owner[feature])
		start, end = enc.feature_bit_ranges()[feature]
		assert set(bit_family[start:end]) == {names.index(home)}, feature


def test_only_families_below_host_bits_merge_and_hosting_families_keep_their_bits():
	enc, total, (bit_family, names, merges) = _layout(dedupe=False)
	merged = _merged_sizes(merges)
	assert "ttl_state" in names and "ttl_state" not in merged
	assert all(size < 28 for size, _ in merged.values())
	base, base_names, _ = bit_families(enc, total, "unsw-nb15", 0)
	for i, fam in enumerate(base_names):
		if fam in merged or fam in {p for _, p in merged.values()}:
			continue
		assert [b == i for b in base] == [b == names.index(fam) for b in bit_family], fam


def test_host_bits_zero_never_merges():
	enc, total, _ = _layout(dedupe=True)
	assert bit_families(enc, total, "unsw-nb15", 0)[2] == []


def test_host_threshold_is_the_rust_eligibility_rule_at_max_bits():
	from wnn.ids.feature_families import host_bits
	assert host_bits(34, 0.2) == 28 and host_bits(24, 0.2) == 20 and host_bits(10, 0.0) == 10


def test_budget_invariant_holds_on_a_merged_map():
	_, _, (bit_family, names, merges) = _layout(dedupe=True)
	rf = ReceptiveFields(bit_family=tuple(bit_family), family_names=tuple(names), cross=0.2,
		version=FAMILY_MAP_VERSION, source="test", merges=tuple(merges), host_bits=28)
	widths = [4 + i % 31 for i in range(300)]
	from wnn.ram.genome import ClusterGenome
	rng = np.random.default_rng(1)
	g = ClusterGenome(bits_per_neuron=widths, neurons_per_cluster=[300],
		connections=rng.integers(0, len(bit_family), size=sum(widths)).tolist())
	rf.project_genome(g, random.Random(2))
	_assert_satisfied(rf, g)
	assert "merged (< 28 bits to host max_bits): ttl_state(" in rf.provenance()


def test_merge_declarations_are_validated(tmp_path, monkeypatch):
	import wnn.ids.feature_families as ff
	monkeypatch.setattr(ff, "_DIR", tmp_path)
	def write(families):
		(tmp_path / "unsw-nb15.json").write_text(json.dumps({"families": families}))
	write({"a": {"why": "x", "merge_into": "nope", "features": ["f"]}})
	with pytest.raises(ValueError, match="not another family"):
		ff.merge_targets("unsw-nb15")
	write({"a": {"why": "x", "merge_into": "b", "features": ["f"]}, "b": {"why": "y", "merge_into": "a", "features": ["g"]}})
	with pytest.raises(ValueError, match="cycle"):
		ff.merge_targets("unsw-nb15")


def test_a_merge_chain_follows_a_partner_that_is_still_too_small():
	from wnn.ids.feature_families import _merge_small_families
	owner = {"f1": "a", "f2": "b", "f3": "c"}
	ranges = {"f1": (0, 4), "f2": (4, 8), "f3": (8, 40)}
	merged, merges = _merge_small_families(owner, ranges, {"a": "b", "b": "c"}, host_bits=20)
	assert set(merged.values()) == {"c"} and merges == ["a(4b)->b", "b(8b)->c"]


def test_settings_carry_the_worker_ids_max_bits_default():
	from wnn.ram.experiments.IdsEvaluatorSettings import IdsEvaluatorSettings
	from wnn.ram.experiments.ids_param_resolution import IDS_DEFAULT_MAX_BITS
	assert IdsEvaluatorSettings.from_params({}).max_bits == IDS_DEFAULT_MAX_BITS == 24
	assert IdsEvaluatorSettings.from_params({"max_bits": 34}).max_bits == 34
