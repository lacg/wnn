//! `UnitGrouping` for Python (IDS-37 stage 6): the multiplicity layer's
//! logical -> physical map is derived HERE, in ram_core (`NeuronKey` +
//! `UnitGrouping::by_key`), never re-derived in Python
//! (`wnn.ram.core.UnitGroupingView` only holds what this returns).

use crate::pyapi::footprint_api::{footprint_to_py, FootprintTuple};
use pyo3::prelude::*;
use ram_core::footprint::Footprint;
use ram_core::neuron_key::NeuronKey;
use ram_core::unit_grouping::UnitGrouping;

/// `(index_of, representatives, member_offsets, members)` — the CSR parts.
pub(crate) type GroupingTuple = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>);

fn grouping_to_py(g: &UnitGrouping) -> GroupingTuple
{
	(
		g.index_map().to_vec(),
		g.representatives().to_vec(),
		g.member_offsets().to_vec(),
		g.all_members().to_vec(),
	)
}

/// Group a flat genome's identical neurons (same cluster, bits, connections).
/// `connections_flat` is unpadded: neuron i owns `bits_per_neuron[i]` entries.
#[pyfunction]
pub(crate) fn group_neurons(
	bits_per_neuron: Vec<usize>,
	neurons_per_cluster: Vec<usize>,
	connections_flat: Vec<i64>,
) -> PyResult<GroupingTuple>
{
	let neurons: usize = neurons_per_cluster.iter().sum();
	let width: usize = bits_per_neuron.iter().sum();
	if neurons != bits_per_neuron.len() || width != connections_flat.len()
	{
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"group_neurons: {} neurons per clusters, {} bits entries, {} connections for {} bits",
			neurons,
			bits_per_neuron.len(),
			connections_flat.len(),
			width
		)));
	}
	let keys = NeuronKey::genome_keys(&bits_per_neuron, &neurons_per_cluster, &connections_flat);
	Ok(grouping_to_py(&UnitGrouping::by_key(keys)))
}

/// Footprint of a layer whose logical neuron i reads unit `index_of[i]`,
/// unit u holding `unit_cells[u]` cells (`ram_core::Footprint`, the same
/// accounting as `GenomeExport::footprint`).
#[pyfunction]
pub(crate) fn layer_footprint(index_of: Vec<u32>, unit_cells: Vec<u64>)
	-> PyResult<FootprintTuple>
{
	let grouping =
		UnitGrouping::from_index_of(index_of).map_err(pyo3::exceptions::PyValueError::new_err)?;
	if grouping.num_groups() != unit_cells.len()
	{
		return Err(pyo3::exceptions::PyValueError::new_err(format!(
			"layer_footprint: {} units, {} unit_cells",
			grouping.num_groups(),
			unit_cells.len()
		)));
	}
	let mut fp = Footprint::default();
	for n in 0..grouping.num_units()
	{
		fp.add_neuron(
			unit_cells[grouping.index_of(n) as usize],
			grouping.is_representative(n),
		);
	}
	Ok(footprint_to_py(fp))
}
