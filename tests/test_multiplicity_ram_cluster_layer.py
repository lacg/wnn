"""MultiplicityRAMClusterLayer == RAMClusterLayer, bit for bit (IDS-37 stage 6).

Same LOGICAL connectivity with planted duplicate neurons (repeats inside a
cluster; the same connections in ANOTHER cluster stay distinct), identical
training sequences through every training method of each backend, then every
forward path compared with torch.equal. Plus: storage = one row per distinct
neuron, factory selection, connectivity regrouping, save/load.

Needs the ABI-17 wheel (group_neurons, layer_footprint, sparse index_of).
"""
import pytest
import torch

from wnn.ram.core import (
	ClusterLayerConfig, ClusterLayerFactory, MemoryBackend, MultiplicityRAMClusterLayer, RAMClusterLayer,
)

TIB = 16
CLUSTERS = 3
NPC = 6
# Per cluster: base-row indices in logical order (repeats = duplicate neurons).
PLAN = [[0, 1, 0, 2, 0, 1], [3, 3, 4, 3, 0, 4], [5, 6, 7, 8, 9, 5]]


def _connections(bits: int, seed: int = 11) -> torch.Tensor:
	gen = torch.Generator().manual_seed(seed)
	rows = torch.stack([torch.randperm(TIB, generator=gen)[:bits] for _ in range(10)])
	return torch.cat([rows[plan] for plan in PLAN])


def _data(examples: int = 40, seed: int = 5):
	gen = torch.Generator().manual_seed(seed)
	inputs = torch.randint(0, 2, (examples, TIB), generator=gen).bool()
	trues = torch.randint(0, CLUSTERS, (examples,), generator=gen)
	falses = torch.stack([(trues + 1) % CLUSTERS, (trues + 2) % CLUSTERS], dim=1)
	return inputs, trues, falses


def _pair(bits: int, backend: MemoryBackend, gating: bool = False):
	conns = _connections(bits)
	config = ClusterLayerConfig(total_input_bits=TIB, num_clusters=CLUSTERS, neurons_per_cluster=NPC,
		bits_per_neuron=bits, connections=conns, backend=backend)
	plain = RAMClusterLayer(total_input_bits=TIB, num_clusters=CLUSTERS, neurons_per_cluster=NPC,
		bits_per_neuron=bits, connections=conns, backend=backend)
	mult = ClusterLayerFactory.create(config)
	assert isinstance(mult, MultiplicityRAMClusterLayer)
	if gating:
		plain.create_gating(neurons_per_gate=3, bits_per_neuron=4, rng=9)
		mult.create_gating(neurons_per_gate=3, bits_per_neuron=4, rng=9)
	return plain, mult


def _assert_forwards_equal(plain, mult, paths):
	inputs, _, _ = _data(25, seed=99)
	for path in paths:
		a, b = getattr(plain, path)(inputs), getattr(mult, path)(inputs)
		for x, y in zip(a if isinstance(a, tuple) else (a,), b if isinstance(b, tuple) else (b,)):
			assert torch.equal(x, y), f"{path} differs"


DENSE_FORWARDS = ["forward", "forward_counts", "forward_quad_scores", "forward_raw", "_forward_ungated",
	"forward_rust", "forward_hybrid", "_forward_metal_cached"]


def _dense_train(layer, method: str, inputs, trues, falses):
	match method:
		case "train_pattern":
			for x, t in zip(inputs, trues):
				layer.train_pattern(x, int(t))
		case "train_batch":
			layer.train_batch(inputs, trues, falses)
		case "train_multi_examples":
			layer.train_multi_examples(inputs, trues, falses)
		case "rust_numpy":
			layer.train_multi_examples_rust_numpy(inputs, trues, falses)


@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.parametrize("method", ["train_pattern", "train_batch", "train_multi_examples", "rust_numpy"])
def test_dense_parity(method, gating):
	plain, mult = _pair(6, MemoryBackend.DENSE, gating)
	inputs, trues, falses = _data()
	_dense_train(plain, method, inputs, trues, falses)
	_dense_train(mult, method, inputs, trues, falses)
	assert plain.forward_raw(inputs).ne(2).any(), "fixture must actually train"
	assert torch.equal(plain.memory.memory_words, mult.logical_memory_words())
	assert mult.memory.memory_words.shape[0] == mult._unit_grouping().num_groups < plain.total_neurons
	_assert_forwards_equal(plain, mult, DENSE_FORWARDS)
	from wnn.accel import metal_available
	if metal_available():
		_assert_forwards_equal(plain, mult, ["forward_metal"])


@pytest.mark.parametrize("gating", [False, True])
def test_sparse_parity(gating):
	plain, mult = _pair(12, MemoryBackend.SPARSE, gating)
	inputs, trues, falses = _data()
	plain.train_auto(inputs, trues, falses)
	mult.train_auto(inputs, trues, falses)
	assert mult._sparse_memory.num_neurons == mult._unit_grouping().num_groups
	assert 0 < mult._sparse_memory.total_cells() < plain._sparse_memory.total_cells()
	_assert_forwards_equal(plain, mult, ["forward", "forward_sparse"])


def test_lsh_parity():
	plain, mult = _pair(8, MemoryBackend.LSH)
	inputs, trues, falses = _data(12)
	plain.train_lsh(inputs, trues, falses)
	mult.train_lsh(inputs, trues, falses)
	assert 0 < mult._lsh_memory.total_cells() < plain._lsh_memory.total_cells()
	_assert_forwards_equal(plain, mult, ["forward", "forward_lsh"])


def test_footprint_logical_and_physical():
	plain, mult = _pair(12, MemoryBackend.SPARSE)
	inputs, trues, falses = _data()
	plain.train_auto(inputs, trues, falses)
	mult.train_auto(inputs, trues, falses)
	fp, fm = plain.footprint(), mult.footprint()
	groups = mult._unit_grouping().num_groups
	assert (fp.logical_neurons, fp.physical_neurons) == (CLUSTERS * NPC, CLUSTERS * NPC)
	assert (fm.logical_neurons, fm.physical_neurons) == (CLUSTERS * NPC, groups)
	assert fm.logical_cells == fp.logical_cells and fm.physical_cells < fp.physical_cells
	assert groups == sum(len(set(plan)) for plan in PLAN) == 11


def test_factory_selection():
	conns = _connections(6)
	def make(rate: float, connections):
		return ClusterLayerFactory.create(ClusterLayerConfig(total_input_bits=TIB, num_clusters=CLUSTERS,
			neurons_per_cluster=NPC, bits_per_neuron=6, connections=connections, neuron_sample_rate=rate))
	assert type(make(1.0, conns)) is MultiplicityRAMClusterLayer
	assert type(make(0.25, conns)) is RAMClusterLayer, "below rate 1.0 copies may train differently"
	assert type(make(1.0, None)) is RAMClusterLayer, "random connectivity"
	distinct = torch.stack([torch.roll(torch.arange(TIB), i)[:6] for i in range(CLUSTERS * NPC)])
	assert type(make(1.0, distinct)) is RAMClusterLayer, "nothing to collapse"


def test_connectivity_change_regroups_and_resets():
	_, mult = _pair(6, MemoryBackend.DENSE)
	inputs, trues, falses = _data()
	mult.train_multi_examples(inputs, trues, falses)
	before = mult._unit_grouping().num_groups
	new = mult.connections.clone()
	new[1] = (new[0] + 1) % TIB  # neuron 1 of cluster 0 becomes a new distinct neuron
	mult.connections = new
	assert torch.equal(mult.connections, new)
	assert mult._unit_grouping().num_groups == before + 1
	plain = RAMClusterLayer(total_input_bits=TIB, num_clusters=CLUSTERS, neurons_per_cluster=NPC,
		bits_per_neuron=6, connections=new, backend=MemoryBackend.DENSE)
	assert torch.equal(mult.logical_memory_words(), plain.memory.memory_words), "storage reset to EMPTY"


def test_save_load_round_trip(tmp_path):
	_, mult = _pair(6, MemoryBackend.DENSE)
	inputs, trues, falses = _data()
	mult.train_multi_examples(inputs, trues, falses)
	mult.save(str(tmp_path / "m.pt"))
	loaded = MultiplicityRAMClusterLayer.load(str(tmp_path / "m.pt"))
	assert type(loaded) is MultiplicityRAMClusterLayer
	assert torch.equal(loaded.forward(inputs), mult.forward(inputs))
