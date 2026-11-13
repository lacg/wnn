//! `NeuronOutputs`: per-row, per-neuron outputs of a trained genome (IDS-57 arm
//! E, Luiz 12/11/2026) — READ-ONLY. The cell each logical neuron reads for each
//! row, straight out of the scored `GenomeExport`, so a learned / weighted
//! readout can be fitted offline on exactly the outputs the reported score sums.
//!
//! A neuron's output is its EFFECTIVE cell: what the production scorer feeds
//! `cell_to_weight` — a sparse miss reads the mode's miss cell (0 when
//! coverage-aware), a coverage-aware dense hole reads 0 (weight 0.0 in every
//! mode, the scorer's "skip"). Combining a row's cells by the scorer's rule
//! (`cluster_scores`: mean cell weight per cluster) reproduces the production
//! per-cluster scores bit-exactly for the deterministic modes
//! (neuron_outputs_tests.rs).
//!
//! Layout: row-major; neurons in GENOME order (cluster-major, the order of the
//! flat genome's `bits`). Unpacked = one u8 per neuron. Packed = 2 bits per
//! neuron, 4 per byte, LSB-first (neuron j in byte j/4 at shift 2·(j%4)), each
//! row padded to a whole byte — lossless for every mode at a quarter the size.

use super::genome_export_physical::{cluster_starts, group_slots};
use super::*;
use ram_core::packed_bits::PackedBits;

/// One logical neuron's read site in the export.
struct NeuronSite
{
	group: usize,
	slot: usize,
	conn_start: usize,
}

pub struct NeuronOutputs
{
	pub num_rows: usize,
	/// Logical neurons per cluster (the genome's), cluster order.
	pub neurons_per_cluster: Vec<usize>,
	/// 2-bit packed rows (see module doc) vs one u8 per neuron.
	pub packed: bool,
	/// `num_rows × row_width` cells.
	pub cells: Vec<u8>,
	/// `num_rows × num_clusters` production-scorer scores.
	pub scores: Vec<f64>,
	/// Target class of each row.
	pub labels: Vec<i64>,
	/// Weight of each cell code (`cell_to_weight`); None for the stochastic
	/// modes (QSR / PLN read a per-example coin, not a fixed weight).
	pub cell_weights: Option<[f32; 4]>,
}

impl NeuronOutputs
{
	pub fn total_neurons(&self) -> usize
	{
		self.neurons_per_cluster.iter().sum()
	}

	/// Bytes per row for `total` neurons.
	pub fn row_width(total: usize, packed: bool) -> usize
	{
		if packed
		{
			total.div_ceil(4)
		}
		else
		{
			total
		}
	}

	/// Row `row`, neuron `neuron` (genome order).
	#[inline]
	pub fn cell(&self, row: usize, neuron: usize) -> u8
	{
		let base = row * Self::row_width(self.total_neurons(), self.packed);
		if self.packed
		{
			(self.cells[base + neuron / 4] >> (2 * (neuron % 4))) & 0x3
		}
		else
		{
			self.cells[base + neuron]
		}
	}

	/// The scorer's rule over the exported cells: per cluster, the f32 sum of
	/// the cell weights divided by the cluster's neuron count. `num_rows ×
	/// num_clusters`; None for a stochastic mode.
	pub fn cluster_scores(&self) -> Option<Vec<f64>>
	{
		let weights = self.cell_weights?;
		let starts = cluster_starts(&self.neurons_per_cluster);
		let mut out = Vec::with_capacity(self.num_rows * starts.len());
		for row in 0..self.num_rows
		{
			for (c, &n) in self.neurons_per_cluster.iter().enumerate()
			{
				let sum: f32 = (0..n).map(|j| weights[self.cell(row, starts[c] + j) as usize]).sum();
				out.push((sum / n as f32) as f64);
			}
		}
		Some(out)
	}
}

/// Weight per cell code, None for the stochastic modes.
pub fn deterministic_cell_weights(memory_mode: u8, empty_value: f32) -> Option<[f32; 4]>
{
	if matches!(memory_mode, ram_core::neuron_memory::QSR | ram_core::neuron_memory::PLN)
	{
		return None;
	}
	Some([0i64, 1, 2, 3].map(|cell| cell_to_weight(cell, memory_mode, empty_value)))
}

impl GenomeExport
{
	/// Where each logical neuron (genome order) reads: its group, slot and
	/// connection start. Errors unless every neuron has exactly one site.
	fn neuron_sites(&self, neurons_per_cluster: &[usize]) -> Result<Vec<NeuronSite>, String>
	{
		let total: usize = neurons_per_cluster.iter().sum();
		let starts = cluster_starts(neurons_per_cluster);
		let mut sites: Vec<Option<NeuronSite>> = (0..total).map(|_| None).collect();
		for (gi, group) in self.groups.iter().enumerate()
		{
			for (slot, l) in group_slots(group, &starts).into_iter().enumerate()
			{
				let Some(l) = l else { continue };
				let site = NeuronSite { group: gi, slot, conn_start: group.conn_offset + slot * group.bits };
				if l >= total || sites[l].replace(site).is_some()
				{
					return Err(format!("neuron outputs: logical neuron {l} has no unique slot in the export"));
				}
			}
		}
		sites
			.into_iter()
			.enumerate()
			.map(|(l, s)| s.ok_or_else(|| format!("neuron outputs: logical neuron {l} is not in the export")))
			.collect()
	}

	/// The effective cell `site` reads for one packed input row — the value
	/// the production scorer hands `cell_to_weight`.
	#[inline]
	fn effective_cell(&self, site: &NeuronSite, row: &[u8], miss_cell: u8, coverage_aware: bool) -> u8
	{
		let group = &self.groups[site.group];
		let (is_sparse, sub, _) = &self.group_info[site.group];
		let address = ram_core::neuron_memory::compute_address_packed_bytes(
			row,
			&self.connections[site.conn_start..],
			group.bits,
		);
		if *is_sparse
		{
			return self.sparse_exports[*sub].lookup(site.slot, address as u64, miss_cell);
		}
		let coverage = self.dense_coverage.get(*sub).map_or(&[][..], |c| c.as_slice());
		if coverage_aware && !dense_is_covered(coverage, site.slot, address, 1usize << group.bits)
		{
			return 0;
		}
		read_cell(&self.dense_exports[*sub], site.slot, address, group.words_per_neuron) as u8
	}

	/// Every logical neuron's effective cell for every row of `input_bits`
	/// (`num_rows × row_width`, module-doc layout). CPU, parallel over rows.
	pub fn neuron_cells(
		&self,
		input_bits: &PackedBits,
		num_rows: usize,
		neurons_per_cluster: &[usize],
		settings: ram_core::neuron_memory::EvalSettings,
		packed: bool,
	) -> Result<Vec<u8>, String>
	{
		let sites = self.neuron_sites(neurons_per_cluster)?;
		let miss_cell =
			ram_core::metal_sparse::default_cell_for_coverage(settings.memory_mode, settings.coverage_aware) as u8;
		let width = NeuronOutputs::row_width(sites.len(), packed).max(1);
		let mut cells = vec![0u8; num_rows * width];
		cells.par_chunks_mut(width).enumerate().for_each(|(r, out)| {
			let row = input_bits.packed_row(r);
			for (j, site) in sites.iter().enumerate()
			{
				let cell = self.effective_cell(site, row, miss_cell, settings.coverage_aware);
				if packed
				{
					out[j / 4] |= (cell & 0x3) << (2 * (j % 4));
				}
				else
				{
					out[j] = cell;
				}
			}
		});
		cells.truncate(num_rows * NeuronOutputs::row_width(sites.len(), packed));
		Ok(cells)
	}

	/// Cells + production scores (all K clusters, the scorer `evaluate_at_
	/// thresholds` uses — GPU when available) of `input_bits`' rows.
	#[allow(clippy::too_many_arguments)]
	pub fn neuron_outputs(
		&self,
		input_bits: &PackedBits,
		labels: Vec<i64>,
		neurons_per_cluster: &[usize],
		total_input_bits: usize,
		settings: ram_core::neuron_memory::EvalSettings,
		packed: bool,
	) -> Result<NeuronOutputs, String>
	{
		let num_rows = labels.len();
		let num_clusters = neurons_per_cluster.len();
		let cells = self.neuron_cells(input_bits, num_rows, neurons_per_cluster, settings, packed)?;
		let (metal, sparse_metal) = (get_metal_evaluator(), get_sparse_metal_evaluator());
		let scores = compute_per_example_scores(
			self, input_bits, num_rows, num_clusters, total_input_bits, settings.empty_value, settings.memory_mode,
			settings.run_seed, settings.coverage_aware, metal.as_deref(), sparse_metal.as_deref(),
		);
		Ok(NeuronOutputs {
			num_rows,
			neurons_per_cluster: neurons_per_cluster.to_vec(),
			packed,
			cells,
			scores: scores.into_iter().flatten().collect(),
			labels,
			cell_weights: deterministic_cell_weights(settings.memory_mode, settings.empty_value),
		})
	}
}
