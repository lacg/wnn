//! Duplicate-aware scoring (worker ABI 23): score each DISTINCT unit once per
//! row and count it by its multiplicity.
//!
//! The multiplicity layer (IDS-37) trains duplicate neurons once, but its
//! export is the LOGICAL layout — k copies of a unit are k slots that alias one
//! range — so every scorer still read every copy. Duplicates share their
//! connections (`NeuronKey`) and their memory, so for a DETERMINISTIC read the
//! k copies produce k identical weights: `sum_l w_l = sum_u m_u * w_u`, and the
//! cluster score `sum / N` (N = logical neurons) is unchanged.
//!
//! Bit-exactness, per memory mode (the gate is `DistinctScoring::for_export`):
//! - QUAD_WEIGHTED (0, .25, .75, 1), QUAD_BINARY / BINARY (0, 1) and TERNARY
//!   with a quarter `empty_value` (0, 1, ev): every weight is an integer number
//!   of QUARTERS (`cell_weight_quarters`). A partial sum of N weights is then a
//!   multiple of 0.25 below N * max_q / 4; while N * max_q <= 2^24 every partial
//!   sum and every product m_u * w_u is exactly representable in f32, so the
//!   sum is exact in ANY order and equals the logical one. The final division
//!   by N is the same f32 operation on the same operands. Coverage-aware
//!   scoring only adds weight 0 (skipped dense cell / cell-0 sparse miss).
//! - The GPU count kernels (TERNARY / BINARY / QUAD_BINARY) accumulate integer
//!   counts; the weighted counts are the same integers.
//! - QSR / PLN: NOT exact — their coin is keyed by the neuron index, so k
//!   copies draw k independent coins. Disabled (`cell_weight_quarters` = None).
//! - TERNARY with a non-quarter `empty_value` (e.g. 0.3): the f32 sum rounds,
//!   so m * w differs from w + ... + w. Disabled.
//!
//! Disabled cases (and every identity grouping: the plain layer, rate < 1.0)
//! score the logical export unchanged. The distinct view is verified against
//! the export's own content (connections and memory of every copy equal its
//! representative's) — a mismatch falls back to the logical path, loudly.

use super::genome_export_physical::{cluster_starts, copy_bits, group_slots};
use super::*;
use ram_core::neuron_memory::empty_word_for_mode;
use ram_core::slot_weights::SlotWeights;
use ram_core::sparse_gpu_export::SparseGpuExport;

/// Largest exactly-representable f32 integer magnitude: every partial sum,
/// counted in quarters, must stay at or below it.
const F32_EXACT_QUARTERS: u64 = 1 << 24;

/// One distinct-slot group's multiplicities (owned `SlotWeights`).
pub(crate) struct GroupWeights
{
	slot_mult: Vec<u32>,
	distinct: Vec<u32>,
	norm: Vec<u32>,
}

impl GroupWeights
{
	pub(crate) fn view(&self) -> SlotWeights<'_>
	{
		SlotWeights { slot_mult: &self.slot_mult, distinct: &self.distinct, norm: &self.norm }
	}
}

/// A logical export re-laid out as distinct units per cluster, plus the
/// multiplicities that make its weighted scores equal the logical ones.
pub(crate) struct DistinctScoring
{
	pub(crate) export: GenomeExport,
	pub(crate) weights: Vec<GroupWeights>,
}

/// Where a compact slot reads: the logical slot of its unit's representative.
struct CompactGroup
{
	group: ConfigGroup,
	/// Logical slot per compact slot (None = pad)
	sources: Vec<Option<usize>>,
	weights: GroupWeights,
}

impl DistinctScoring
{
	/// The distinct view when it is exact (module doc) and saves work; None
	/// = score the logical export.
	pub(crate) fn for_export(export: &GenomeExport, memory_mode: u8, empty_value: f32) -> Option<Self>
	{
		if export.grouping.is_identity()
		{
			return None;
		}
		let max_quarters = *cell_weight_quarters(memory_mode, empty_value)?.iter().max()? as u64;
		let compact = Self::compact_groups(export)?;
		let max_norm = compact.iter().flat_map(|c| c.weights.norm.iter()).copied().max().unwrap_or(0) as u64;
		if !sums_exact_in_f32(max_norm, max_quarters)
		{
			return None;
		}
		Some(Self::assemble(export, compact, memory_mode))
	}

	/// Per group: units in first-occurrence order per cluster, verified
	/// against the export's content. None when the grouping does not fit the
	/// export or a copy differs from its representative (warned once).
	fn compact_groups(export: &GenomeExport) -> Option<Vec<CompactGroup>>
	{
		let npc = export.logical_npc();
		if export.grouping.num_units() != npc.iter().sum::<usize>()
		{
			return warn_fallback("grouping size != logical neurons");
		}
		let starts = cluster_starts(&npc);
		let mut out = Vec::with_capacity(export.groups.len());
		for (gi, group) in export.groups.iter().enumerate()
		{
			let slots = group_slots(group, &starts);
			let compact = compact_group(export, group, &slots);
			if !copies_match(export, gi, &slots, &compact)
			{
				return warn_fallback("a copy's connections or memory differ from its representative");
			}
			out.push(compact);
		}
		Some(out)
	}

	/// Build the compact export (same group order, kinds and sub indices).
	fn assemble(export: &GenomeExport, compact: Vec<CompactGroup>, memory_mode: u8) -> Self
	{
		let mut out = GenomeExport::empty();
		out.group_info = export.group_info.clone();
		let mut sparse: Vec<Option<SparseGpuExport>> = vec![None; export.sparse_exports.len()];
		let mut dense: Vec<Option<(Vec<i64>, Vec<u64>)>> = vec![None; export.dense_exports.len()];
		let mut weights = Vec::with_capacity(compact.len());
		for (gi, mut c) in compact.into_iter().enumerate()
		{
			let (is_sparse, sub, _) = export.group_info[gi];
			c.group.conn_offset = out.connections.len();
			out.connections.extend(compact_connections(export, gi, &c.sources));
			if is_sparse
			{
				sparse[sub] = Some(compact_sparse(&export.sparse_exports[sub], &c.sources));
			}
			else
			{
				dense[sub] = Some(compact_dense(export, gi, &c.sources, memory_mode));
			}
			out.groups.push(c.group);
			weights.push(c.weights);
		}
		out.sparse_exports = sparse.into_iter().map(|s| s.expect("every sparse sub index filled")).collect();
		let (words, coverage): (Vec<_>, Vec<_>) = dense.into_iter().map(|d| d.expect("every dense sub index filled")).unzip();
		out.dense_exports = words;
		out.dense_coverage = if export.dense_coverage.is_empty() { Vec::new() } else { coverage };
		out.grouping = ram_core::unit_grouping::UnitGrouping::identity(out.groups.iter().map(|g| g.true_total_neurons()).sum());
		Self { export: out, weights }
	}
}

/// Every partial sum of `norm` weights of at most `max_quarters` quarters is
/// an f32-exact multiple of 0.25 (module doc).
fn sums_exact_in_f32(max_norm: u64, max_quarters: u64) -> bool
{
	max_norm.saturating_mul(max_quarters) <= F32_EXACT_QUARTERS
}

fn warn_fallback<T>(why: &str) -> Option<T>
{
	static WARNED: std::sync::Once = std::sync::Once::new();
	WARNED.call_once(|| eprintln!("[DISTINCT_SCORING] ⚠️ {why} — scoring the LOGICAL export (exact, no speedup)"));
	None
}

/// A group's distinct layout: cluster li's units in first-occurrence order at
/// a stride of the largest cluster's unit count.
fn compact_group(export: &GenomeExport, group: &ConfigGroup, slots: &[Option<usize>]) -> CompactGroup
{
	let per_cluster: Vec<Vec<(usize, u32)>> = (0..group.cluster_count())
		.map(|li| cluster_units(export, &slots[li * group.neurons..(li + 1) * group.neurons], li * group.neurons))
		.collect();
	let stride = per_cluster.iter().map(|u| u.len()).max().unwrap_or(0);
	let mut sources = vec![None; group.cluster_count() * stride];
	let mut slot_mult = vec![0u32; group.cluster_count() * stride];
	for (li, units) in per_cluster.iter().enumerate()
	{
		for (k, &(rep, m)) in units.iter().enumerate()
		{
			sources[li * stride + k] = Some(rep);
			slot_mult[li * stride + k] = m;
		}
	}
	let distinct: Vec<u32> = per_cluster.iter().map(|u| u.len() as u32).collect();
	let norm: Vec<u32> = per_cluster.iter().map(|u| u.iter().map(|&(_, m)| m).sum()).collect();
	let mut compact = group.clone();
	compact.neurons = stride;
	compact.actual_neurons = Some(distinct.clone());
	CompactGroup { group: compact, sources, weights: GroupWeights { slot_mult, distinct, norm } }
}

/// One cluster's units as (representative's group slot, multiplicity), in
/// first-occurrence order. `cluster_slots` is the cluster's slot run (pads
/// None), starting at group slot `first_slot`.
fn cluster_units(export: &GenomeExport, cluster_slots: &[Option<usize>], first_slot: usize) -> Vec<(usize, u32)>
{
	let mut index: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
	let mut units: Vec<(usize, u32)> = Vec::new();
	for (n, l) in cluster_slots.iter().enumerate()
	{
		let Some(l) = *l
		else
		{
			continue;
		};
		let k = *index.entry(export.grouping.index_of(l)).or_insert_with(|| {
			units.push((first_slot + n, 0));
			units.len() - 1
		});
		units[k].1 += 1;
	}
	units
}

/// Every logical slot of group `gi` reads exactly what its unit's
/// representative reads: same connections, same memory (sparse range or dense
/// row + coverage bits). Exactness of the weighted sum rests on this.
fn copies_match(export: &GenomeExport, gi: usize, slots: &[Option<usize>], compact: &CompactGroup) -> bool
{
	let rep_of_unit: std::collections::HashMap<u32, usize> = compact
		.sources
		.iter()
		.flatten()
		.map(|&slot| (export.grouping.index_of(slots[slot].expect("a representative is a real slot")), slot))
		.collect();
	slots.iter().enumerate().all(|(slot, l)| match l
	{
		None => true,
		Some(l) => rep_of_unit
			.get(&export.grouping.index_of(*l))
			.map_or(false, |&rep| slot_reads_equal(export, gi, slot, rep)),
	})
}

/// Slots `a` and `b` of group `gi` compute the same address and read the
/// same cell for every input.
fn slot_reads_equal(export: &GenomeExport, gi: usize, a: usize, b: usize) -> bool
{
	if a == b
	{
		return true;
	}
	let group = &export.groups[gi];
	let conns = |s: usize| &export.connections[group.conn_offset + s * group.bits..group.conn_offset + (s + 1) * group.bits];
	if conns(a) != conns(b)
	{
		return false;
	}
	let (is_sparse, sub, _) = &export.group_info[gi];
	if *is_sparse
	{
		let e = &export.sparse_exports[*sub];
		return e.neuron_range(a) == e.neuron_range(b);
	}
	let wpn = group.words_per_neuron;
	let words = &export.dense_exports[*sub];
	words[a * wpn..(a + 1) * wpn] == words[b * wpn..(b + 1) * wpn] && coverage_equal(export, *sub, group.bits, a, b)
}

fn coverage_equal(export: &GenomeExport, sub: usize, bits: usize, a: usize, b: usize) -> bool
{
	let cov = export.dense_coverage.get(sub).map_or(&[][..], |c| c.as_slice());
	let apn = 1usize << bits;
	cov.is_empty() || (0..apn).all(|addr| dense_is_covered(cov, a, addr, apn) == dense_is_covered(cov, b, addr, apn))
}

/// The compact group's connections: each slot's representative run (pads 0,
/// never read — the kernels stop at `distinct[c]`).
fn compact_connections(export: &GenomeExport, gi: usize, sources: &[Option<usize>]) -> Vec<i64>
{
	let group = &export.groups[gi];
	let mut out = Vec::with_capacity(sources.len() * group.bits);
	for s in sources
	{
		match *s
		{
			Some(slot) =>
			{
				let start = group.conn_offset + slot * group.bits;
				out.extend_from_slice(&export.connections[start..start + group.bits]);
			}
			None => out.extend(std::iter::repeat(0).take(group.bits)),
		}
	}
	out
}

fn compact_sparse(logical: &SparseGpuExport, sources: &[Option<usize>]) -> SparseGpuExport
{
	SparseGpuExport::from_ranges(sources.iter().map(|s| match *s
	{
		Some(slot) => logical.neuron_range(slot),
		None => (&[][..], &[][..]),
	}))
}

/// Dense rows + coverage bits of each compact slot (pads empty, uncovered).
fn compact_dense(export: &GenomeExport, gi: usize, sources: &[Option<usize>], memory_mode: u8) -> (Vec<i64>, Vec<u64>)
{
	let group = &export.groups[gi];
	let sub = export.group_info[gi].1;
	let (wpn, apn) = (group.words_per_neuron, 1usize << group.bits);
	let words_in = &export.dense_exports[sub];
	let cov_in = export.dense_coverage.get(sub).map_or(&[][..], |c| c.as_slice());
	let mut words = Vec::with_capacity(sources.len() * wpn);
	let mut coverage = if cov_in.is_empty() { Vec::new() } else { vec![0u64; (sources.len() * apn).div_ceil(64)] };
	for (k, s) in sources.iter().enumerate()
	{
		match *s
		{
			Some(slot) =>
			{
				words.extend_from_slice(&words_in[slot * wpn..(slot + 1) * wpn]);
				copy_bits(cov_in, slot * apn, &mut coverage, k * apn, apn);
			}
			None => words.extend(std::iter::repeat(empty_word_for_mode(memory_mode)).take(wpn)),
		}
	}
	(words, coverage)
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn exactness_bound_is_two_to_the_24_quarters()
	{
		// QUAD's largest weight is 4 quarters: 2^22 logical neurons is the edge.
		assert!(sums_exact_in_f32(1 << 22, 4));
		assert!(!sums_exact_in_f32((1 << 22) + 1, 4));
		assert!(sums_exact_in_f32(500, 4), "the IDS shapes are nowhere near it");
		// The edge really is where f32 stops being exact for quarter sums:
		// 2^24 quarters (= 2^22) and every quarter below it are representable,
		// the next quarter is not.
		let edge = (1u32 << 24) as f32 * 0.25;
		let below = edge - 0.25;
		assert_eq!(edge - below, 0.25);
		assert_eq!(below + 0.25, edge);
		assert_eq!(edge + 0.25, edge, "2^24 + 1 quarters must round");
	}
}
