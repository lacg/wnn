//! Unit tests for the IDS-26 GA knobs: ga_dedupe_neurons and ga_joint_mutation.
//! (Bit-identity with the knobs OFF is neighbor_search_golden.rs.)

use super::golden::{fake_eval, ga_config, population_ids, population_multi, Genome};
use super::*;
use std::collections::HashSet;

/// Every cluster's neurons as (bits, connections) keys.
fn cluster_keys(child: &Child) -> Vec<Vec<(usize, Vec<i64>)>>
{
	let n_off = compute_neuron_offsets(&child.1);
	let c_off = compute_conn_offsets(&child.0);
	(0..child.1.len())
		.map(|c| (n_off[c]..n_off[c + 1]).map(|g| (child.0[g], child.2[c_off[g]..c_off[g + 1]].to_vec())).collect())
		.collect()
}

/// Layout consistent AND every cluster's neurons distinct.
fn assert_consistent_distinct(child: &Child)
{
	assert_eq!(child.0.len(), child.1.iter().sum::<usize>(), "bits vs neuron counts");
	assert_eq!(child.2.len(), child.0.iter().sum::<usize>(), "conns vs bits");
	for keys in cluster_keys(child)
	{
		let distinct: HashSet<_> = keys.iter().collect();
		assert_eq!(distinct.len(), keys.len(), "duplicate neuron in a cluster");
	}
}

/// Fraction of duplicated neurons a population carries (sanity: the fixture has some).
fn has_duplicates(pop: &[Genome]) -> bool
{
	pop.iter().any(|g| {
		let child = (g.0.clone(), g.1.clone(), g.2.clone());
		cluster_keys(&child).iter().any(|k| k.iter().collect::<HashSet<_>>().len() < k.len())
	})
}

#[test]
fn dedupe_crossover_children_are_distinct_with_full_counts()
{
	for (pop, tib) in [(population_multi(), 96usize), (population_ids(), 200usize)]
	{
		assert!(has_duplicates(&pop), "fixture must carry duplicate neurons");
		let xo = NeuronXo { dedupe: true, total_input_bits: tib };
		let mut rng = StdRng::seed_from_u64(99);
		for k in 0..pop.len() - 1
		{
			let (p1, p2) = (&pop[k], &pop[k + 1]);
			let (a, b) = crossover_ps_neurons(p1, p2, p1.1.len(), xo, &mut rng);
			assert_consistent_distinct(&a);
			assert_consistent_distinct(&b);
			let (a, b) = crossover2_neurons(p1, p2, p1.1.len(), xo, &mut rng);
			assert_consistent_distinct(&a);
			assert_consistent_distinct(&b);
			assert_eq!(a.1, p1.1, "uniform child1 keeps parent1's counts");
			assert_eq!(b.1, p2.1, "uniform child2 keeps parent2's counts");
		}
	}
}

#[test]
fn dedupe_fills_shortfall_at_mode_bits()
{
	// Both parents: 6 copies of ONE neuron. Pool set = 1 neuron; 5 must be fresh.
	let neuron: Vec<i64> = (0..10).collect();
	let p: Genome = (vec![10; 6], vec![6], neuron.repeat(6), 1.0);
	let xo = NeuronXo { dedupe: true, total_input_bits: 200 };
	let mut rng = StdRng::seed_from_u64(3);
	let (a, _) = crossover_ps_neurons(&p, &p.clone(), 1, xo, &mut rng);
	assert_consistent_distinct(&a);
	assert_eq!(a.1, vec![6]);
	assert!(a.0.iter().all(|&b| b == 10), "fresh neurons use the parents' mode bits");
	assert!(cluster_keys(&a)[0].contains(&(10, neuron)), "the inherited neuron survives");
}

#[test]
fn dedupe_off_still_duplicates()
{
	let neuron: Vec<i64> = (0..10).collect();
	let p: Genome = (vec![10; 6], vec![6], neuron.repeat(6), 1.0);
	let xo = NeuronXo { dedupe: false, total_input_bits: 200 };
	let (a, _) = crossover_ps_neurons(&p, &p.clone(), 1, xo, &mut StdRng::seed_from_u64(3));
	assert_eq!(cluster_keys(&a)[0].iter().collect::<HashSet<_>>().len(), 1);
}

#[test]
fn dedupe_search_offspring_all_distinct()
{
	let pop = population_ids();
	for ps_ratio in [0.0, 1.0]
	{
		let mut cfg = ga_config(&pop, PhaseType::Neurons, 200, (0.0, ps_ratio, 0.85));
		cfg.dedupe_neurons = true;
		cfg.crossover_rate = 1.0; // clones copy a parent verbatim — dedupe is a crossover knob
		let res = search_offspring(&pop, 20, 60, 0.0, &cfg, &fake_eval, 5, None, None, None, true, None);
		assert_eq!(res.candidates.len(), 20);
		for c in &res.candidates
		{
			assert_consistent_distinct(&(c.bits_per_neuron.clone(), c.neurons_per_cluster.clone(), c.connections.clone()));
		}
	}
}

/// Joint crossover IS the neurons crossover: same RNG → same children.
#[test]
fn joint_crossover_matches_neurons_crossover()
{
	let pop = population_multi();
	let xo = NeuronXo { dedupe: false, total_input_bits: 96 };
	for k in 0..pop.len() - 1
	{
		let run = |phase| {
			let mut rng = StdRng::seed_from_u64(k as u64);
			let ps = crossover_pool_shuffle(&pop[k], &pop[k + 1], 3, phase, xo, &mut rng);
			let u = crossover2(&pop[k], &pop[k + 1], 3, phase, xo, &mut rng);
			(ps, u)
		};
		assert!(run(PhaseType::Joint) == run(PhaseType::Neurons));
	}
}

/// Which axes changed between a parent and its Joint-mutated child (kept-prefix compare).
fn axes_changed(parent: &Genome, child: &Child) -> (bool, bool, bool)
{
	let neurons = child.1 != parent.1;
	let keep = parent.0.len().min(child.0.len());
	let bits = parent.0[..keep] != child.0[..keep];
	let (po, co) = (compute_conn_offsets(&parent.0), compute_conn_offsets(&child.0));
	let conns = (0..keep).any(|g| parent.0[g] == child.0[g] && parent.2[po[g]..po[g + 1]] != child.2[co[g]..co[g + 1]]);
	(neurons, bits, conns)
}

#[test]
fn joint_mutation_moves_all_three_axes()
{
	let pop = population_ids();
	let mut cfg = ga_config(&pop, PhaseType::Joint, 200, (0.0, 0.0, 0.0));
	cfg.neurons_mutation_rate = 1.0;
	cfg.bits_mutation_rate = 0.2;
	let mut seen = (false, false, false);
	for seed in 0..20
	{
		let mut rng = StdRng::seed_from_u64(seed);
		let child = mutate_ga(&pop[0].0, &pop[0].1, &pop[0].2, &cfg, &mut rng);
		assert_eq!(child.2.len(), child.0.iter().sum::<usize>(), "conns stay length-consistent");
		let (n, b, c) = axes_changed(&pop[0], &child);
		seen = (seen.0 || n, seen.1 || b, seen.2 || c);
	}
	assert_eq!(seen, (true, true, true), "neurons, bits and connections must all move");
}

#[test]
fn joint_mutation_honours_pinned_bits()
{
	let pop = population_ids();
	let mut cfg = ga_config(&pop, PhaseType::Joint, 200, (0.0, 0.0, 0.0));
	(cfg.min_bits, cfg.max_bits) = (10, 10);
	cfg.neurons_mutation_rate = 1.0;
	cfg.bits_mutation_rate = 1.0;
	for seed in 0..10
	{
		let child = mutate_ga(&pop[0].0, &pop[0].1, &pop[0].2, &cfg, &mut StdRng::seed_from_u64(seed));
		assert!(child.0.iter().all(|&b| b == 10), "min_bits == max_bits pins bits");
		assert_eq!(child.2.len(), child.0.iter().sum::<usize>());
	}
}

#[test]
fn joint_search_offspring_respects_bands()
{
	let pop = population_ids();
	let mut cfg = ga_config(&pop, PhaseType::Joint, 200, (0.0, 0.8, 0.85));
	(cfg.min_bits, cfg.max_bits, cfg.min_neurons, cfg.max_neurons) = (4, 34, 5, 500);
	(cfg.neurons_mutation_rate, cfg.bits_mutation_rate) = (0.1, 0.1);
	let res = search_offspring(&pop, 20, 60, 0.0, &cfg, &fake_eval, 9, None, None, None, true, None);
	for c in &res.candidates
	{
		assert!(c.bits_per_neuron.iter().all(|&b| (4..=34).contains(&b)));
		assert!(c.neurons_per_cluster.iter().all(|&n| (5..=500).contains(&n)));
		assert_eq!(c.connections.len(), c.bits_per_neuron.iter().sum::<usize>());
	}
}

// ---- mode-bits tie-break (11/10/2026) ----------------------------------------

/// The pre-11/10 rule, verbatim: HashMap order decides a tie.
fn legacy_mode_bits(bits: &[usize]) -> Option<usize>
{
	let mut counts = std::collections::HashMap::new();
	for &b in bits
	{
		*counts.entry(b).or_insert(0usize) += 1;
	}
	counts.iter().max_by_key(|&(_, &count)| count).map(|(&b, _)| b)
}

/// Values sharing the top count, ascending (a tie when more than one).
fn top_values(bits: &[usize]) -> Vec<usize>
{
	let count = |b: &usize| bits.iter().filter(|&x| x == b).count();
	let max = bits.iter().map(count).max().unwrap_or(0);
	let mut top: Vec<usize> = bits.iter().copied().filter(|b| count(b) == max).collect();
	top.sort_unstable();
	top.dedup();
	top
}

#[test]
fn mode_bits_changes_only_on_ties()
{
	let mut rng = StdRng::seed_from_u64(2026);
	let (mut ties, mut non_ties) = (0, 0);
	for _ in 0..5000
	{
		let len = rng.gen_range(1..30);
		let bits: Vec<usize> = (0..len).map(|_| rng.gen_range(4..9)).collect();
		let top = top_values(&bits);
		if top.len() == 1
		{
			non_ties += 1;
			assert_eq!(cluster_mode_bits(&bits), legacy_mode_bits(&bits), "non-tie must be unchanged: {bits:?}");
		}
		else
		{
			ties += 1;
			assert_eq!(cluster_mode_bits(&bits), Some(top[0]), "tie must go to the fewest bits: {bits:?}");
		}
	}
	assert!(ties > 100 && non_ties > 100, "both cases exercised ({ties} ties, {non_ties} non-ties)");
	assert_eq!(cluster_mode_bits(&[]), None);
}

#[test]
fn mode_bits_tie_is_stable_across_fresh_hashmaps()
{
	// 10, 12 and 34 tied; every call hashes with fresh keys.
	for _ in 0..200
	{
		assert_eq!(cluster_mode_bits(&[12, 10, 34, 12, 10, 34]), Some(10));
	}
}
