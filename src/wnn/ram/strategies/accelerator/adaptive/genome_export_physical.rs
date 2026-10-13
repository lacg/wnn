//! `GenomeExport::from_physical`: the IDS multiplicity layer's logical view.
//!
//! Duplicate neurons are trained once as UNITS: one genome's physical genome
//! (`ram_core::physical_genome`, stage 2) or a whole batch's super-genome
//! (`ram_core::super_genome`, stages 4-5). This rebuilds a genome's LOGICAL
//! export every scorer consumes — the same groups, connections and neuron slots
//! as the plain export of the logical genome — with each logical slot reading
//! its unit's memory: sparse slots alias the unit's sorted range (stored once),
//! dense slots get the unit's row at the logical group's width. Scorers are
//! untouched (docs/ids_multiplicity_layer_design.md §3).
//!
//! A unit may sit in a WIDER physical group than its logical group (the
//! super-genome's cluster max bits is the batch's): its cells are addresses of
//! its own bits, so a dense logical slot re-packs a wider dense row or
//! densifies a sparse range — exactly what the plain export's
//! `densify_sparse_export` of the same keys produces.

use super::*;
use ram_core::neuron_memory::{empty_word_for_mode, BITS_PER_CELL, CELLS_PER_WORD, CELL_MASK};
use ram_core::super_genome::FlatGenome;
use ram_core::unit_grouping::UnitGrouping;

/// What a logical slot reads: a physical unit, or a coalescing pad (empty).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SlotSource
{
	Unit(usize),
	Pad(usize),
}

/// A trained export of units plus where each unit lives in it.
pub struct PhysicalUnits<'a>
{
	export: &'a GenomeExport,
	/// unit -> (group index, slot)
	locations: Vec<(usize, usize)>,
}

/// One unit's cells at a dense logical width: packed row + coverage bits
/// (empty when the physical export did not track coverage).
struct DenseRow
{
	words: Vec<i64>,
	coverage: Vec<u64>,
}

impl<'a> PhysicalUnits<'a>
{
	/// `neurons_per_cluster`: units per cluster of the genome `export` was trained from.
	pub fn new(export: &'a GenomeExport, neurons_per_cluster: &[usize]) -> Self
	{
		let starts = cluster_starts(neurons_per_cluster);
		let mut locations = vec![(usize::MAX, usize::MAX); neurons_per_cluster.iter().sum()];
		for (gi, group) in export.groups.iter().enumerate()
		{
			for (slot, unit) in group_slots(group, &starts).into_iter().enumerate()
			{
				if let Some(u) = unit
				{
					locations[u] = (gi, slot);
				}
			}
		}
		Self { export, locations }
	}

	fn is_sparse(&self, unit: usize) -> bool
	{
		self.export.group_info[self.locations[unit].0].0
	}

	fn bits(&self, unit: usize) -> usize
	{
		self.export.groups[self.locations[unit].0].bits
	}

	fn sparse_range(&self, unit: usize) -> (&'a [u64], &'a [u8])
	{
		let (pg, slot) = self.locations[unit];
		self.export.sparse_exports[self.export.group_info[pg].1].neuron_range(slot)
	}

	/// The unit's cells at `bits` (<= its group's bits) as a dense row.
	fn dense_row(&self, unit: usize, bits: usize, memory_mode: u8) -> DenseRow
	{
		if self.is_sparse(unit)
		{
			let (keys, values) = self.sparse_range(unit);
			let (words, coverage) =
				densify_sparse_export(keys, values, &[0], &[keys.len() as u32], 1, bits, memory_mode);
			return DenseRow { words, coverage };
		}
		self.repacked_row(unit, bits, memory_mode)
	}

	/// A dense unit's row re-packed to the first 2^bits addresses (verbatim
	/// when the widths match).
	fn repacked_row(&self, unit: usize, bits: usize, memory_mode: u8) -> DenseRow
	{
		let (pg, pslot) = self.locations[unit];
		let sub = self.export.group_info[pg].1;
		let (src_wpn, src_apn, apn) = (self.export.groups[pg].words_per_neuron, 1usize << self.bits(unit), 1usize << bits);
		let src = &self.export.dense_exports[sub][pslot * src_wpn..(pslot + 1) * src_wpn];
		let mut words = vec![empty_word_for_mode(memory_mode); apn.div_ceil(CELLS_PER_WORD)];
		for addr in 0..apn
		{
			let shift = (addr % CELLS_PER_WORD) * BITS_PER_CELL;
			let cell = (src[addr / CELLS_PER_WORD] >> shift) & CELL_MASK;
			let w = &mut words[addr / CELLS_PER_WORD];
			*w = (*w & !(CELL_MASK << shift)) | (cell << shift);
		}
		let tracked = self.export.dense_coverage.get(sub).map_or(&[][..], |c| c.as_slice());
		let mut coverage = if tracked.is_empty() { Vec::new() } else { vec![0u64; apn.div_ceil(64)] };
		copy_bits(tracked, pslot * src_apn, &mut coverage, 0, apn);
		DenseRow { words, coverage }
	}
}

impl GenomeExport
{
	/// Logical export of one genome over trained units. `unit_of[l]` is the
	/// unit logical neuron l reads; `grouping` (the genome's own logical ->
	/// distinct-neuron map) is stored for the footprint.
	pub fn from_physical(
		units: &PhysicalUnits,
		unit_of: &[u32],
		grouping: UnitGrouping,
		logical: FlatGenome,
		memory_mode: u8,
	) -> Result<GenomeExport, String>
	{
		let groups = build_groups(&per_cluster_max_bits(logical.bits, logical.neurons_per_cluster), logical.neurons_per_cluster);
		let mut out = GenomeExport::empty();
		out.connections = reorganize_connections_for_gpu(logical.connections, logical.bits, logical.neurons_per_cluster, &groups);
		let starts = cluster_starts(logical.neurons_per_cluster);
		for group in &groups
		{
			let sources = slot_sources(group, &starts, unit_of);
			push_logical_group(units, &mut out, group, &sources, memory_mode)?;
		}
		// A physical export that holds dense rows but tracks no coverage (CPU
		// classic path) keeps the same empty outer Vec in the logical view.
		if !units.export.dense_exports.is_empty() && units.export.dense_coverage.is_empty()
		{
			out.dense_coverage.clear();
		}
		out.groups = groups;
		out.grouping = grouping;
		Ok(out)
	}
}

/// Append one logical group (sparse or dense by its bits, as the trainer
/// export decides) whose slots read `sources`.
fn push_logical_group(
	units: &PhysicalUnits,
	out: &mut GenomeExport,
	group: &ConfigGroup,
	sources: &[SlotSource],
	memory_mode: u8,
) -> Result<(), String>
{
	let is_sparse = group.bits > SPARSE_THRESHOLD;
	check_kinds(units, group, sources, is_sparse)?;
	if is_sparse
	{
		out.group_info.push((true, out.sparse_exports.len(), group.cluster_ids.clone()));
		out.sparse_exports.push(logical_sparse(units, sources));
	}
	else
	{
		let (words, coverage) = logical_dense(units, group, sources, memory_mode);
		out.group_info.push((false, out.dense_exports.len(), group.cluster_ids.clone()));
		out.dense_exports.push(words);
		out.dense_coverage.push(coverage);
	}
	Ok(())
}

/// A unit's physical group is never narrower than its logical group, and a
/// sparse logical group reads sparse units (wider is sparse too) — both hold
/// by construction (a unit's group max covers every genome's max for its
/// cluster); checked because exactness rests on it.
fn check_kinds(units: &PhysicalUnits, group: &ConfigGroup, sources: &[SlotSource], is_sparse: bool) -> Result<(), String>
{
	for s in sources
	{
		if let SlotSource::Unit(u) = *s
		{
			if units.bits(u) < group.bits || (is_sparse && !units.is_sparse(u))
			{
				return Err(format!(
					"from_physical: unit {u} lives in a {}-bit {} group, logical group is {}-bit {}",
					units.bits(u),
					kind(units.is_sparse(u)),
					group.bits,
					kind(is_sparse)
				));
			}
		}
	}
	Ok(())
}

/// Sparse: each distinct source's range is stored once; slots alias it.
fn logical_sparse(units: &PhysicalUnits, sources: &[SlotSource]) -> SparseGpuExport
{
	let slot_grouping = UnitGrouping::by_key(sources.iter().copied());
	let compact = SparseGpuExport::from_ranges(slot_grouping.representatives().iter().map(|&r| match sources[r as usize]
	{
		SlotSource::Unit(u) => units.sparse_range(u),
		SlotSource::Pad(_) => (&[][..], &[][..]),
	}));
	compact.alias(&slot_grouping)
}

/// Dense: each slot gets its unit's row at the group's width (≤ 1 KB/neuron at b ≤ 12).
fn logical_dense(units: &PhysicalUnits, group: &ConfigGroup, sources: &[SlotSource], memory_mode: u8) -> (Vec<i64>, Vec<u64>)
{
	let (wpn, apn) = (group.words_per_neuron, 1usize << group.bits);
	let mut words = Vec::with_capacity(sources.len() * wpn);
	let mut coverage = vec![0u64; (sources.len() * apn).div_ceil(64)];
	let mut tracked = true;
	for (slot, s) in sources.iter().enumerate()
	{
		match *s
		{
			SlotSource::Unit(u) =>
			{
				let row = units.dense_row(u, group.bits, memory_mode);
				words.extend_from_slice(&row.words);
				tracked &= !row.coverage.is_empty();
				copy_bits(&row.coverage, 0, &mut coverage, slot * apn, apn);
			}
			SlotSource::Pad(_) => words.extend(std::iter::repeat(empty_word_for_mode(memory_mode)).take(wpn)),
		}
	}
	// An untracked physical bitmap stays untracked (empty) in the logical view.
	(words, if tracked { coverage } else { Vec::new() })
}

fn kind(is_sparse: bool) -> &'static str
{
	if is_sparse
	{
		"sparse"
	}
	else
	{
		"dense"
	}
}

/// First genome-order neuron of each cluster.
pub(super) fn cluster_starts(npc: &[usize]) -> Vec<usize>
{
	let mut starts = Vec::with_capacity(npc.len());
	let mut acc = 0usize;
	for &n in npc
	{
		starts.push(acc);
		acc += n;
	}
	starts
}

/// A group's slot layout (cluster li's neurons at li * group.neurons + n, the
/// layout every export and scorer uses): Some(genome-order neuron) or None for
/// a coalescing pad.
pub(super) fn group_slots(group: &ConfigGroup, starts: &[usize]) -> Vec<Option<usize>>
{
	let mut slots = Vec::with_capacity(group.total_neurons());
	for (li, &c) in group.cluster_ids.iter().enumerate()
	{
		let actual = group.actual_neurons.as_ref().map_or(group.neurons, |a| a[li] as usize);
		slots.extend((0..group.neurons).map(|n| (n < actual).then(|| starts[c] + n)));
	}
	slots
}

/// Logical slots of a group mapped to the physical units they read.
fn slot_sources(group: &ConfigGroup, starts: &[usize], unit_of: &[u32]) -> Vec<SlotSource>
{
	group_slots(group, starts)
		.into_iter()
		.enumerate()
		.map(|(slot, logical)| match logical
		{
			Some(l) => SlotSource::Unit(unit_of[l] as usize),
			None => SlotSource::Pad(slot),
		})
		.collect()
}

/// Copy `len` bits from `src` at `src_bit` into `dst` at `dst_bit` (dst bits
/// start clear). An empty `src` (coverage not tracked) copies nothing.
fn copy_bits(src: &[u64], src_bit: usize, dst: &mut [u64], dst_bit: usize, len: usize)
{
	if src.is_empty()
	{
		return;
	}
	for i in 0..len
	{
		let (s, d) = (src_bit + i, dst_bit + i);
		if (src[s / 64] >> (s % 64)) & 1 == 1
		{
			dst[d / 64] |= 1u64 << (d % 64);
		}
	}
}
