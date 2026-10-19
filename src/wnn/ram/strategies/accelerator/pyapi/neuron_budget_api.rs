//! ram_core::neuron_budget bridge (flow param `max_neurons_total`, ABI 21).
//!
//! Rust enforces the whole-genome budget on every GA child / TS neighbour inside
//! mutate_genome; the Python side builds the GRID genomes and IMMIGRANTS itself,
//! so it calls this same function for their per-cluster counts (one rule, one
//! implementation — never a Python copy).

use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn neuron_budget_counts(neurons: Vec<usize>, total: usize, min_neurons: usize) -> Vec<usize>
{
	ram_core::neuron_budget::budget_counts(&neurons, total, min_neurons)
}

/// Fit a whole genome (bits per neuron, neurons per cluster, connections) to the
/// budget, truncating each cluster from its end — for genomes Python already
/// holds (grid leaderboard seeds, GA initial population, TS start genome).
#[pyfunction]
pub(crate) fn neuron_budget_fit_genome(
	bits: Vec<usize>,
	neurons: Vec<usize>,
	connections: Vec<i64>,
	total: usize,
	min_neurons: usize,
) -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	ram_core::neuron_budget::fit_genome(&bits, &neurons, &connections, total, min_neurons)
}
