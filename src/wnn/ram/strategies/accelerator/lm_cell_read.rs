//! How the LM adaptation passes read a ClusterStorage cell as a vote (IDS-81).
//!
//! The bitwise LM trains its ClusterStorage under the flow's memory mode
//! (TERNARY / BINARY / PLN / the 4-state family), so the stats and
//! axonogenesis passes must read cells through that mode's semantics
//! (`cell_to_weight`). Until IDS-81 they read every mode through QUAD_WEIGHTS,
//! so a TERNARY / PLN / BINARY TRUE (cell 1) scored 0.25 and never fired.

use ram_core::neuron_memory::ClusterStorage;

/// The memory mode (+ TERNARY empty value) a ClusterStorage was trained under.
#[derive(Clone, Copy, Debug)]
pub struct LmCellRead
{
	pub memory_mode: u8,
	pub empty_value: f32,
}

impl LmCellRead
{
	/// The cell at (neuron, address) as a vote in [0, 1].
	pub fn vote(&self, storage: &ClusterStorage, neuron_idx: usize, address: u32) -> f32
	{
		let raw = storage.read_cell(neuron_idx, address as usize);
		ram_core::neuron_memory::cell_to_weight(raw, self.memory_mode, self.empty_value)
	}
}
