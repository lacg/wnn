//! `GenomeExport::from_physical`: the IDS multiplicity layer's logical view.
//!
//! A genome with duplicate neurons is trained as its PHYSICAL genome (one
//! neuron per `NeuronKey`, `ram_core::physical_genome`). This rebuilds the
//! LOGICAL export every scorer consumes — the same groups, connections and
//! neuron slots as the plain export of the logical genome — with each logical
//! slot reading its unit's memory: sparse slots alias the unit's sorted range
//! (stored once), dense slots copy the unit's packed row + coverage row.
//! Scorers are untouched (docs/ids_multiplicity_layer_design.md §3).

use super::*;
use ram_core::neuron_memory::empty_word_for_mode;
use ram_core::physical_genome::PhysicalGenome;
use ram_core::unit_grouping::UnitGrouping;

/// What a logical slot reads: a physical unit, or a coalescing pad (empty).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SlotSource
{
	Unit(usize),
	Pad(usize),
}

impl GenomeExport
{
	/// Logical export over a physically-trained one. `grouping` maps each
	/// logical neuron (cluster-major genome order) to its physical neuron in
	/// `physical`, which was exported from the collapsed genome.
	pub fn from_physical(
		physical: GenomeExport,
		grouping: UnitGrouping,
		logical_bits: &[usize],
		logical_npc: &[usize],
		logical_conns: &[i64],
		memory_mode: u8,
	) -> Result<GenomeExport, String>
	{
		let groups = build_groups(&per_cluster_max_bits(logical_bits, logical_npc), logical_npc);
		let physical_npc = PhysicalGenome::cluster_counts(&grouping, logical_npc);
		let locations = physical.unit_locations(&physical_npc);
		let mut out = GenomeExport::empty();
		out.connections = reorganize_connections_for_gpu(logical_conns, logical_bits, logical_npc, &groups);
		let starts = cluster_starts(logical_npc);
		for group in &groups
		{
			let sources = slot_sources(group, &starts, &grouping);
			physical.push_logical_group(&mut out, group, &sources, &locations, memory_mode)?;
		}
		// A physical export that tracks no dense coverage at all (CPU classic
		// path) keeps the same empty outer Vec in the logical view.
		if physical.dense_coverage.is_empty()
		{
			out.dense_coverage.clear();
		}
		out.groups = groups;
		out.grouping = grouping;
		Ok(out)
	}

	/// (group index, slot) of every physical neuron in this export.
	fn unit_locations(&self, physical_npc: &[usize]) -> Vec<(usize, usize)>
	{
		let starts = cluster_starts(physical_npc);
		let mut loc = vec![(usize::MAX, usize::MAX); physical_npc.iter().sum()];
		for (gi, group) in self.groups.iter().enumerate()
		{
			for (slot, unit) in group_slots(group, &starts).into_iter().enumerate()
			{
				if let Some(u) = unit
				{
					loc[u] = (gi, slot);
				}
			}
		}
		loc
	}

	/// Append one logical group (sparse or dense by its bits, as the trainer
	/// export decides) whose slots read `sources`.
	fn push_logical_group(
		&self,
		out: &mut GenomeExport,
		group: &ConfigGroup,
		sources: &[SlotSource],
		locations: &[(usize, usize)],
		memory_mode: u8,
	) -> Result<(), String>
	{
		let is_sparse = group.bits > SPARSE_THRESHOLD;
		self.check_kinds(group, sources, locations, is_sparse)?;
		if is_sparse
		{
			out.group_info.push((true, out.sparse_exports.len(), group.cluster_ids.clone()));
			out.sparse_exports.push(self.logical_sparse(sources, locations));
		}
		else
		{
			let (words, coverage) = self.logical_dense(group, sources, locations, memory_mode);
			out.group_info.push((false, out.dense_exports.len(), group.cluster_ids.clone()));
			out.dense_exports.push(words);
			out.dense_coverage.push(coverage);
		}
		Ok(())
	}

	/// Every unit a logical group reads must sit in a physical group of the same
	/// bits and storage kind — collapsing never removes a cluster's max-bits
	/// neuron, so this holds by construction; checked because exactness rests on it.
	fn check_kinds(
		&self,
		group: &ConfigGroup,
		sources: &[SlotSource],
		locations: &[(usize, usize)],
		is_sparse: bool,
	) -> Result<(), String>
	{
		for s in sources
		{
			if let SlotSource::Unit(u) = *s
			{
				let pg = locations[u].0;
				if self.groups[pg].bits != group.bits || self.group_info[pg].0 != is_sparse
				{
					return Err(format!(
						"from_physical: unit {u} lives in a {}-bit {} group, logical group is {}-bit {}",
						self.groups[pg].bits,
						kind(self.group_info[pg].0),
						group.bits,
						kind(is_sparse)
					));
				}
			}
		}
		Ok(())
	}

	/// Sparse: each distinct source's range is stored once; slots alias it.
	fn logical_sparse(&self, sources: &[SlotSource], locations: &[(usize, usize)]) -> SparseGpuExport
	{
		let slot_grouping = UnitGrouping::by_key(sources.iter().copied());
		let compact = SparseGpuExport::from_ranges(slot_grouping.representatives().iter().map(|&r| {
			match sources[r as usize]
			{
				SlotSource::Unit(u) =>
				{
					let (pg, slot) = locations[u];
					self.sparse_exports[self.group_info[pg].1].neuron_range(slot)
				}
				SlotSource::Pad(_) => (&[][..], &[][..]),
			}
		}));
		compact.alias(&slot_grouping)
	}

	/// Dense: copy each slot's packed row and coverage row (≤ 1 KB/neuron at b ≤ 12).
	fn logical_dense(
		&self,
		group: &ConfigGroup,
		sources: &[SlotSource],
		locations: &[(usize, usize)],
		memory_mode: u8,
	) -> (Vec<i64>, Vec<u64>)
	{
		let wpn = group.words_per_neuron;
		let apn = 1usize << group.bits;
		let mut words = Vec::with_capacity(sources.len() * wpn);
		let mut coverage = vec![0u64; (sources.len() * apn).div_ceil(64)];
		let mut tracked = true;
		for (slot, s) in sources.iter().enumerate()
		{
			match *s
			{
				SlotSource::Unit(u) =>
				{
					let (pg, pslot) = locations[u];
					let sub = self.group_info[pg].1;
					words.extend_from_slice(&self.dense_exports[sub][pslot * wpn..(pslot + 1) * wpn]);
					let src = &self.dense_coverage.get(sub).map_or(&[][..], |c| c.as_slice());
					tracked &= !src.is_empty();
					copy_bits(src, pslot * apn, &mut coverage, slot * apn, apn);
				}
				SlotSource::Pad(_) => words.extend(std::iter::repeat(empty_word_for_mode(memory_mode)).take(wpn)),
			}
		}
		// An untracked physical bitmap stays untracked (empty) in the logical view.
		(words, if tracked { coverage } else { Vec::new() })
	}
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
fn cluster_starts(npc: &[usize]) -> Vec<usize>
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
fn group_slots(group: &ConfigGroup, starts: &[usize]) -> Vec<Option<usize>>
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
fn slot_sources(group: &ConfigGroup, starts: &[usize], grouping: &UnitGrouping) -> Vec<SlotSource>
{
	group_slots(group, starts)
		.into_iter()
		.enumerate()
		.map(|(slot, logical)| match logical
		{
			Some(l) => SlotSource::Unit(grouping.index_of(l) as usize),
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
