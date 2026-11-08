//! Receptive fields through the IDS genesis path (worker ABI 27): synaptogenesis
//! grows connections and neurogenesis appends fresh neurons INSIDE the
//! train -> adapt -> retrain cycle (adaptive_eval.rs), so the projection runs
//! there, after the passes and before the retrain. Every adapted genome the
//! cycle returns must satisfy the per-neuron family budget.

use crate::adaptation::AdaptationConfig;
use crate::ids_cache::{evaluate_genomes_ids_cached_hybrid_adaptive, IDSCache};
use ram_core::neuron_memory::QUAD_WEIGHTED;
use ram_core::packed_bits::PackedBits;
use ram_core::receptive_fields::ReceptiveFields;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

const TIB: usize = 64;

/// Random rows; the label is a noisy rule over bits of two families.
fn split(n: usize, rng: &mut SmallRng) -> (PackedBits, Vec<i64>)
{
	let mut bools = Vec::with_capacity(n * TIB);
	let mut labels = Vec::with_capacity(n);
	for _ in 0..n
	{
		let row: Vec<bool> = (0..TIB).map(|_| rng.gen_bool(0.5)).collect();
		let signal = row[3] && row[5] || row[40] && row[41];
		labels.push(if rng.gen_bool(0.9) { signal as i64 } else { rng.gen_bool(0.3) as i64 });
		bools.extend(row);
	}
	(PackedBits::from_bool_slice(&bools, TIB), labels)
}

fn cache() -> IDSCache
{
	let mut rng = SmallRng::seed_from_u64(0x6E5E);
	let (tr, trl) = split(600, &mut rng);
	let (ev, evl) = split(200, &mut rng);
	let mut cache = IDSCache::new(tr, trl, ev, evl, None, None, 2, TIB, 5, 1, 42, false, false, false, 1.0);
	cache.memory_mode = QUAD_WEIGHTED;
	cache
}

/// 4 families x 16 contiguous bits.
fn rf() -> Arc<ReceptiveFields>
{
	Arc::new(ReceptiveFields::new((0..TIB).map(|b| (b / 16) as u8).collect(), 0.2).unwrap())
}

/// Two genomes, 2 clusters x 6 neurons x 8 bits, UNIFORM (unprojected) wiring.
fn genomes() -> (Vec<usize>, Vec<usize>, Vec<i64>)
{
	let mut rng = SmallRng::seed_from_u64(9);
	let bits = vec![8usize; 24];
	let conns = (0..24 * 8).map(|_| rng.gen_range(0..TIB as i64)).collect();
	(bits, vec![6usize; 4], conns)
}

fn adapt_config(rf: Option<Arc<ReceptiveFields>>) -> AdaptationConfig
{
	AdaptationConfig {
		synaptogenesis_enabled: true,
		neurogenesis_enabled: true,
		warmup_generations: 0,
		total_generations: 100,
		min_bits: 4,
		max_bits: 12,
		total_input_bits: TIB,
		stats_sample_size: 0,
		receptive_fields: rf,
		..Default::default()
	}
}

fn run(rf: Option<Arc<ReceptiveFields>>) -> Vec<crate::adaptive::AdaptiveGenomeResult>
{
	let (bits, neurons, conns) = genomes();
	evaluate_genomes_ids_cached_hybrid_adaptive(&cache(), &bits, &neurons, &conns, 2, 0, 0.5, 1.0, 0, &adapt_config(rf), 5)
}

#[test]
fn genesis_output_satisfies_the_family_budget()
{
	let rf = rf();
	for r in run(Some(rf.clone()))
	{
		assert_eq!(r.adapted_connections.len(), r.adapted_bits.iter().sum::<usize>());
		let mut start = 0;
		for &w in &r.adapted_bits
		{
			let slots = &r.adapted_connections[start..start + w];
			assert!(rf.satisfied(slots), "genesis neuron {slots:?} -> {:?}", rf.field(slots));
			start += w;
		}
	}
}

#[test]
fn genesis_with_receptive_fields_is_deterministic()
{
	let a: Vec<Vec<i64>> = run(Some(rf())).into_iter().map(|r| r.adapted_connections).collect();
	let b: Vec<Vec<i64>> = run(Some(rf())).into_iter().map(|r| r.adapted_connections).collect();
	assert_eq!(a, b);
}
