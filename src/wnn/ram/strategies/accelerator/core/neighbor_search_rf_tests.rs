//! Receptive fields through the GA/TS operators (flow param ids_receptive_fields,
//! worker ABI 27). Off-mode bit-identity is GOLDEN_SEARCH / GOLDEN_OPERATORS /
//! GOLDEN_DEDUPE_OPERATORS in neighbor_search_golden.rs (receptive_fields: None).

use super::golden::{fake_eval, ga_config, population_ids, population_multi, Genome};
use super::*;
use crate::crossover_bits::CrossoverBits;
use crate::receptive_fields::ReceptiveFields;
use std::sync::Arc;

/// Contiguous family runs covering `tib` bits: shares 40/25/20/10/5 %.
fn family_map(tib: usize) -> Vec<u8>
{
	let cuts = [0.40, 0.65, 0.85, 0.95, 1.0].map(|c| (c * tib as f64).round() as usize);
	(0..tib).map(|b| cuts.iter().position(|&c| b < c).unwrap() as u8).collect()
}

fn rf(tib: usize) -> Arc<ReceptiveFields>
{
	Arc::new(ReceptiveFields::new(family_map(tib), 0.2).unwrap())
}

/// Every neuron of a flat genome satisfies the constraint (or is exempt).
fn assert_satisfied(rf: &ReceptiveFields, bits: &[usize], conns: &[i64], what: &str)
{
	assert_eq!(conns.len(), bits.iter().sum::<usize>(), "{what}: layout");
	let mut start = 0;
	for &w in bits
	{
		let slots = &conns[start..start + w];
		assert!(rf.satisfied(slots), "{what}: neuron {slots:?} -> {:?}", rf.field(slots));
		start += w;
	}
}

/// The creators (grid / immigrants / random init) project in Python before the
/// GA ever sees a genome; mirror that here.
fn projected(pop: &[Genome], rf: &ReceptiveFields) -> Vec<Genome>
{
	let mut rng = StdRng::seed_from_u64(0xF00D);
	pop.iter()
		.map(|g| {
			let mut c = g.2.clone();
			rf.project(&g.0, &mut c, &mut rng).unwrap();
			(g.0.clone(), g.1.clone(), c, g.3)
		})
		.collect()
}

fn rf_config(pop: &[Genome], phase: PhaseType, tib: usize, ratios: (f64, f64, f64)) -> GAConfig
{
	GAConfig { receptive_fields: Some(rf(tib)), ..ga_config(pop, phase, tib, ratios) }
}

/// Every knob combination the IDS GA can fly with.
fn knob_variants(base: &GAConfig) -> Vec<GAConfig>
{
	let mut out = Vec::new();
	for xb in [CrossoverBits::Inherit, CrossoverBits::BlendNeuron, CrossoverBits::BlendCluster]
	{
		for (dedupe, fill) in [(false, true), (true, true), (true, false)]
		{
			for total in [None, Some(60)]
			{
				out.push(GAConfig { crossover_bits: xb, dedupe_neurons: dedupe, dedupe_fill: fill, max_neurons_total: total, ..base.clone() });
			}
		}
	}
	out
}

/// Breed `gens` generations (offspring become the next population) and check
/// every evaluated child.
fn breed(pop: Vec<Genome>, cfg: &GAConfig, gens: usize, what: &str)
{
	let rf = cfg.receptive_fields.clone().unwrap();
	let mut pop = pop;
	for g in 0..gens
	{
		let res = search_offspring(&pop, 12, 60, 0.0, cfg, &fake_eval, 31 + g as u64, None, Some(g), Some(gens), true, None);
		for c in &res.candidates
		{
			assert_satisfied(&rf, &c.bits_per_neuron, &c.connections, what);
		}
		pop = res.candidates.iter().map(|c| (c.bits_per_neuron.clone(), c.neurons_per_cluster.clone(), c.connections.clone(), c.cross_entropy)).collect();
	}
}

#[test]
fn every_offspring_satisfies_the_budget_over_generations()
{
	for (pop, tib) in [(population_multi(), 96usize), (population_ids(), 200usize)]
	{
		for phase in [PhaseType::Neurons, PhaseType::Bits, PhaseType::Connections, PhaseType::Cluster, PhaseType::Joint]
		{
			for ratios in [(0.0, 0.0, 0.0), (0.0, 0.8, 0.85), (0.5, 0.8, 0.85), (0.3, 1.0, 0.0)]
			{
				let base = rf_config(&pop, phase, tib, ratios);
				let start = projected(&pop, base.receptive_fields.as_ref().unwrap());
				for cfg in knob_variants(&base)
				{
					breed(start.clone(), &cfg, 6, &format!("{phase:?} {ratios:?} {:?}", cfg.crossover_bits));
				}
			}
		}
	}
}

#[test]
fn unprojected_parents_still_breed_satisfied_children()
{
	// Even a population that never met the creators is repaired on the way out.
	let pop = population_ids();
	let cfg = rf_config(&pop, PhaseType::Neurons, 200, (0.5, 0.8, 0.85));
	breed(pop, &cfg, 10, "raw parents");
}

#[test]
fn search_neighbors_satisfies_the_budget()
{
	let pop = population_ids();
	let rfs = rf(200);
	let start = projected(&pop, &rfs);
	for phase in [PhaseType::Neurons, PhaseType::Bits, PhaseType::Connections, PhaseType::Joint]
	{
		let cfg = GAConfig { receptive_fields: Some(rfs.clone()), ..ga_config(&start, phase, 200, (0.0, 0.0, 0.0)) };
		let mcfg = MutationConfig {
			num_clusters: 1,
			mutable_clusters: None,
			min_bits: cfg.min_bits,
			max_bits: cfg.max_bits,
			min_neurons: 1,
			max_neurons: 500,
			bits_mutation_rate: 0.5,
			neurons_mutation_rate: 0.5,
			total_input_bits: 200,
			phase_type: phase,
			max_neurons_total: None,
			max_bit_delta: 0,
			receptive_fields: Some(rfs.clone()),
		};
		let res = search_neighbors_best_n(&start[0].0, &start[0].1, &start[0].2, 20, 40, 0.0, &mcfg, &fake_eval, 5, None, None, None, None);
		res.iter().for_each(|c| assert_satisfied(&rfs, &c.bits_per_neuron, &c.connections, "TS neighbour"));
	}
}

/// Every raw operator's output is repaired by the ONE projection and is then a
/// fixed point (idempotent): crossover2 / pool-shuffle (all phases, dedupe on and
/// off), crossover_bits blends, dedupe_fill, and each mutation phase.
#[test]
fn projection_repairs_every_operator_and_is_idempotent()
{
	let pop = population_ids();
	let rfs = rf(200);
	let start = projected(&pop, &rfs);
	let mut rng = StdRng::seed_from_u64(77);
	let mut children: Vec<Child> = Vec::new();
	for xo in [NeuronXo { dedupe: false, total_input_bits: 200, fill: true, min_neurons: 0 },
		NeuronXo { dedupe: true, total_input_bits: 200, fill: true, min_neurons: 2 }]
	{
		for phase in [PhaseType::Neurons, PhaseType::Bits, PhaseType::Connections, PhaseType::Cluster]
		{
			for k in 0..start.len() - 1
			{
				let (a, b) = crossover2(&start[k], &start[k + 1], 1, phase, xo, &mut rng);
				let (c, d) = crossover_pool_shuffle(&start[k], &start[k + 1], 1, phase, xo, &mut rng);
				children.extend([a, b, c, d]);
			}
		}
	}
	for xb in [CrossoverBits::BlendNeuron, CrossoverBits::BlendCluster]
	{
		let cfg = GAConfig { crossover_bits: xb, ..ga_config(&start, PhaseType::Neurons, 200, (0.0, 0.0, 0.0)) };
		let blended: Vec<Child> = children.iter().take(20).map(|c| blend_bits(c.clone(), &start[0], &start[1], &cfg, &mut rng)).collect();
		children.extend(blended);
	}
	for phase in [PhaseType::Neurons, PhaseType::Bits, PhaseType::Connections, PhaseType::Joint]
	{
		let cfg = ga_config(&start, phase, 200, (0.0, 0.0, 0.0));
		children.extend(start.iter().map(|g| mutate_ga(&g.0, &g.1, &g.2, &cfg, &mut rng)));
	}
	let violating = children.iter().filter(|c| !all_satisfied(&rfs, c)).count();
	assert!(violating > 0, "the raw operators must produce violations for this test to mean anything");
	for child in children
	{
		let fixed = project_receptive_fields(child, Some(&rfs), &mut rng);
		assert_satisfied(&rfs, &fixed.0, &fixed.2, "projected operator output");
		let again = project_receptive_fields(fixed.clone(), Some(&rfs), &mut rng);
		assert_eq!(again, fixed, "projection must be idempotent");
	}
}

fn all_satisfied(rf: &ReceptiveFields, g: &Child) -> bool
{
	let mut start = 0;
	g.0.iter().all(|&w| {
		let ok = rf.satisfied(&g.2[start..start + w]);
		start += w;
		ok
	})
}

#[test]
fn off_draws_nothing_and_changes_nothing()
{
	let g: Child = (vec![3, 3], vec![2], vec![0, 50, 199, 1, 2, 3]);
	let mut a = StdRng::seed_from_u64(1);
	let b = a.clone();
	let out = project_receptive_fields(g.clone(), None, &mut a);
	assert_eq!(out, g);
	assert_eq!(a.gen::<u64>(), b.clone().gen::<u64>(), "off must not consume RNG");
}
