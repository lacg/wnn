//! Python form of `ram_core::footprint::Footprint` (IDS-37 stage 3).
//!
//! A flat tuple in a FIXED order, mirrored by `wnn.ram.architecture.Footprint.from_tuple`:
//! (logical_neurons, physical_neurons, logical_cells, physical_cells).

/// `(logical_neurons, physical_neurons, logical_cells, physical_cells)`.
pub(crate) type FootprintTuple = (u64, u64, u64, u64);

pub(crate) fn footprint_to_py(fp: ram_core::footprint::Footprint) -> FootprintTuple
{
	(fp.logical_neurons, fp.physical_neurons, fp.logical_cells, fp.physical_cells)
}
