//! ram_core::receptive_fields bridge (flow param `ids_receptive_fields`, ABI 27).
//!
//! Rust enforces the per-neuron family constraint on every GA child / TS
//! neighbour inside mutate_genome and after every genesis pass; the Python side
//! builds the GRID genomes, the GA initial population and the IMMIGRANTS itself
//! (and runs the Python mutate/crossover fallbacks), so it calls this same
//! projection on them — one rule, one implementation, never a Python copy.

use pyo3::prelude::*;
use rand::SeedableRng;
use ram_core::receptive_fields::ReceptiveFields;
use std::sync::Arc;

/// The flow's receptive fields from the PyO3 args (None = off). Refuses a map
/// whose width differs from the cache's encoded input.
pub(crate) fn receptive_fields_arg(
	bit_family: Option<Vec<u8>>,
	rf_cross: f64,
	total_input_bits: usize,
) -> PyResult<Option<Arc<ReceptiveFields>>>
{
	let Some(map) = bit_family else { return Ok(None) };
	let rf = ReceptiveFields::new(map, rf_cross).map_err(pyo3::exceptions::PyValueError::new_err)?;
	if rf.total_input_bits() != total_input_bits
	{
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"receptive fields: family map covers {} bits, the cache encodes {total_input_bits}",
			rf.total_input_bits()
		)));
	}
	Ok(Some(Arc::new(rf)))
}

/// Project a genome's connections (laid out by bits_per_neuron) onto the family
/// constraint. Returns (connections, neurons changed). Deterministic in `seed`.
#[pyfunction]
pub(crate) fn rf_project_connections(
	bits_per_neuron: Vec<usize>,
	mut connections: Vec<i64>,
	bit_family: Vec<u8>,
	rf_cross: f64,
	seed: u64,
) -> PyResult<(Vec<i64>, usize)>
{
	let rf = ReceptiveFields::new(bit_family, rf_cross).map_err(pyo3::exceptions::PyValueError::new_err)?;
	let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
	let changed = rf.project(&bits_per_neuron, &mut connections, &mut rng).map_err(pyo3::exceptions::PyValueError::new_err)?;
	Ok((connections, changed))
}

/// Per neuron: (held family or -1, in-family connections, cross budget, exempt).
/// Read-only (no RNG) — provenance logs, smoke checks and tests.
#[pyfunction]
pub(crate) fn rf_neuron_fields(
	bits_per_neuron: Vec<usize>,
	connections: Vec<i64>,
	bit_family: Vec<u8>,
	rf_cross: f64,
) -> PyResult<Vec<(i64, usize, usize, bool)>>
{
	let rf = ReceptiveFields::new(bit_family, rf_cross).map_err(pyo3::exceptions::PyValueError::new_err)?;
	if connections.len() != bits_per_neuron.iter().sum::<usize>()
	{
		return Err(pyo3::exceptions::PyValueError::new_err("rf_neuron_fields: connections do not match bits_per_neuron"));
	}
	let mut start = 0;
	Ok(bits_per_neuron
		.iter()
		.map(|&w| {
			let f = rf.field(&connections[start..start + w]);
			start += w;
			(f.family.map_or(-1, i64::from), f.in_family, f.budget, f.exempt)
		})
		.collect())
}
