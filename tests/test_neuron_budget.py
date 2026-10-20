"""max_neurons_total: the Python helpers (over the Rust neuron_budget exports),
the flow-param validation and the grid exclusion rule."""

import pytest

ram_accelerator = pytest.importorskip("ram_accelerator")
if getattr(ram_accelerator, "ABI_VERSION", 0) < 21:
	pytest.skip("worker wheel predates max_neurons_total (ABI < 21)", allow_module_level=True)

from wnn.ram.experiments.worker import _max_neurons_total_param  # noqa: E402
from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome  # noqa: E402
from wnn.ram.strategies.connectivity.grid_search import GridSearchConfig, GridSearchStrategy  # noqa: E402
from wnn.ram.strategies.connectivity.neuron_budget import budget_counts, fit_genome, validate_budget  # noqa: E402


def test_budget_counts_none_and_within_are_unchanged():
	assert budget_counts([400, 400], None, 1) == [400, 400]
	assert budget_counts([200, 250], 500, 1) == [200, 250]


def test_budget_counts_shrinks_proportionally():
	assert budget_counts([500, 500], 500, 1) == [250, 250]
	assert budget_counts([600, 300], 500, 1) == [333, 167]


def test_fit_genome_returns_same_object_within_budget_and_truncates_over():
	g = ClusterGenome(bits_per_neuron=[2, 3, 2, 2, 1, 1, 1, 1], neurons_per_cluster=[4, 4], connections=list(range(13)))
	assert fit_genome(g, 8, 1) is g
	f = fit_genome(g, 4, 1)
	assert f.neurons_per_cluster == [2, 2]
	assert f.bits_per_neuron == [2, 3, 1, 1]
	assert f.connections == [0, 1, 2, 3, 4, 9, 10]


def test_validate_budget():
	validate_budget(None, 2, 5)
	validate_budget(10, 2, 5)
	with pytest.raises(ValueError):
		validate_budget(9, 2, 5)


@pytest.mark.parametrize("raw,ok", [(None, None), (500, 500), (0, ValueError), (-3, ValueError), (2.5, ValueError)])
def test_flow_param_validation(raw, ok):
	params = {} if raw is None else {"max_neurons_total": raw}
	if ok is ValueError:
		with pytest.raises(ValueError):
			_max_neurons_total_param(params)
	else:
		assert _max_neurons_total_param(params) == ok


def _grid(total, clusters=2, neurons=(100, 250, 300, 500)):
	cfg = GridSearchConfig(num_clusters=clusters, neurons_grid=list(neurons), bits_grid=[16, 32],
	                       total_input_bits=64, max_neurons_total=total)
	return GridSearchStrategy(cfg, seed=1), cfg


def test_grid_excludes_cells_over_the_total():
	s, cfg = _grid(500)
	cells = s._build_config_list(cfg, None)
	assert sorted({n for n, _, _ in cells}) == [100, 250]
	s, cfg = _grid(None)
	assert len(s._build_config_list(cfg, None)) == 8


def test_grid_refuses_an_empty_grid():
	s, cfg = _grid(100, neurons=(300, 500))
	with pytest.raises(RuntimeError):
		s._build_config_list(cfg, None)
