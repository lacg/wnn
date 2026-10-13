//! Golden bit-identity tests for the GA operators (IDS-26 follow-up, 11/10/2026).
//!
//! The digests below were captured from the operators BEFORE the
//! ga_dedupe_neurons / ga_joint_mutation work landed. With every new knob at
//! its default (off), the operators must reproduce them exactly: same
//! offspring, same RNG draw sequence. A digest change here means the default
//! GA path changed behaviour — never "update the golden" without that being
//! the intended outcome.
//!
//! Populations are built so every cluster has ONE bits value across all
//! genomes: `mutate_neurons_phase` picks the mode through a std HashMap, whose
//! tie-break is per-instance random, so a tie would make the digest flaky.

use super::*;

const SEED: u64 = 0x1D5_2026;

pub(super) type Genome = (Vec<usize>, Vec<usize>, Vec<i64>, f64);

/// FNV-1a over a stream of u64 words.
struct Digest(u64);

impl Digest
{
	fn new() -> Self
	{
		Digest(0xcbf2_9ce4_8422_2325)
	}

	fn word(&mut self, w: u64)
	{
		for b in w.to_le_bytes()
		{
			self.0 ^= b as u64;
			self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
		}
	}

	fn genome(&mut self, bits: &[usize], neurons: &[usize], conns: &[i64])
	{
		self.word(bits.len() as u64);
		bits.iter().for_each(|&b| self.word(b as u64));
		self.word(neurons.len() as u64);
		neurons.iter().for_each(|&n| self.word(n as u64));
		self.word(conns.len() as u64);
		conns.iter().for_each(|&c| self.word(c as u64));
	}
}

/// One genome: cluster c has `neurons[c]` neurons of `cluster_bits[c]` bits,
/// connections drawn from `rng` (some neurons copied to create duplicates).
fn make_genome(neurons: &[usize], cluster_bits: &[usize], tib: usize, fit: f64, rng: &mut StdRng) -> Genome
{
	let mut bits = Vec::new();
	let mut conns: Vec<i64> = Vec::new();
	for (c, &n) in neurons.iter().enumerate()
	{
		let b = cluster_bits[c];
		for local in 0..n
		{
			bits.push(b);
			let dup = local > 0 && local % 4 == 0;
			let fresh: Vec<i64> = (0..b).map(|_| rng.gen_range(0..tib as i64)).collect();
			let neuron = if dup { conns[conns.len() - b..].to_vec() } else { fresh };
			conns.extend(neuron);
		}
	}
	(bits, neurons.to_vec(), conns, fit)
}

/// 8-genome, 3-cluster population with per-cluster uniform bits.
pub(super) fn population_multi() -> Vec<Genome>
{
	let mut rng = StdRng::seed_from_u64(7);
	let cluster_bits = [6usize, 9, 12];
	(0..8)
		.map(|i| {
			let neurons = [3 + i % 3, 4 + (i * 2) % 5, 2 + i % 4];
			make_genome(&neurons, &cluster_bits, 96, 1.0 + i as f64 * 0.1, &mut rng)
		})
		.collect()
}

/// IDS-like: one cluster, 20..40 neurons of 10 bits, 200 input bits.
pub(super) fn population_ids() -> Vec<Genome>
{
	let mut rng = StdRng::seed_from_u64(11);
	(0..10)
		.map(|i| make_genome(&[20 + (i * 7) % 21], &[10], 200, 0.5 + i as f64 * 0.03, &mut rng))
		.collect()
}

pub(super) fn ga_config(pop: &[Genome], phase: PhaseType, tib: usize, ratios: (f64, f64, f64)) -> GAConfig
{
	GAConfig {
		num_clusters: pop[0].1.len(),
		mutable_clusters: None,
		min_bits: 4,
		max_bits: 34,
		min_neurons: 1,
		max_neurons: 500,
		bits_mutation_rate: 0.1,
		neurons_mutation_rate: 0.3,
		crossover_rate: 0.7,
		tournament_size: 3,
		total_input_bits: tib,
		phase_type: phase,
		cluster_crossover_ratio: ratios.0,
		pool_shuffle_ratio: ratios.1,
		assortative_mating_ratio: ratios.2,
		dedupe_neurons: false,
		dedupe_fill: true,
	}
}

/// The new knob at its default: original operators.
fn xo_off(total_input_bits: usize) -> NeuronXo
{
	NeuronXo { dedupe: false, total_input_bits, fill: true, min_neurons: 0 }
}

/// Deterministic stand-in for the IDS evaluator: metrics are a hash of the genome.
pub(super) fn fake_eval(bits: &[usize], neurons: &[usize], conns: &[i64], count: usize) -> Vec<(f64, f64, f64, f64)>
{
	let nc = neurons.len() / count.max(1);
	let mut out = Vec::with_capacity(count);
	let (mut bo, mut co) = (0usize, 0usize);
	for g in 0..count
	{
		let ns = &neurons[g * nc..(g + 1) * nc];
		let tn: usize = ns.iter().sum();
		let bs = &bits[bo..bo + tn];
		let tb: usize = bs.iter().sum();
		let mut d = Digest::new();
		d.genome(bs, ns, &conns[co..co + tb]);
		bo += tn;
		co += tb;
		let h = d.0;
		out.push(((h % 1000) as f64 / 1000.0, ((h >> 10) % 1000) as f64 / 1000.0, 0.5, 0.1));
	}
	out
}

fn search_digest(pop: &[Genome], cfg: &GAConfig, threshold: f64) -> u64
{
	let res = search_offspring(pop, 24, 120, threshold, cfg, &fake_eval, SEED, None, Some(3), Some(10), true, None);
	let mut d = Digest::new();
	d.word(res.evaluated as u64);
	d.word(res.viable as u64);
	for c in &res.candidates
	{
		d.genome(&c.bits_per_neuron, &c.neurons_per_cluster, &c.connections);
		d.word(c.cross_entropy.to_bits());
	}
	d.0
}

const PHASES: [PhaseType; 4] = [PhaseType::Neurons, PhaseType::Bits, PhaseType::Connections, PhaseType::Cluster];
const RATIOS: [(f64, f64, f64); 4] = [(0.0, 0.0, 0.0), (0.0, 0.8, 0.85), (0.3, 1.0, 0.0), (0.1, 0.5, 0.85)];

/// Digest of search_offspring over every phase × ratio combo, both populations.
fn all_search_digests() -> Vec<u64>
{
	let mut out = Vec::new();
	for (pop, tib) in [(population_multi(), 96usize), (population_ids(), 200usize)]
	{
		for phase in PHASES
		{
			for ratios in RATIOS
			{
				let cfg = ga_config(&pop, phase, tib, ratios);
				out.push(search_digest(&pop, &cfg, 0.0));
				out.push(search_digest(&pop, &cfg, 0.6));
			}
		}
	}
	out
}

/// Digest of the raw crossover + mutation operators on fixed parent pairs.
fn operator_digest() -> u64
{
	let mut d = Digest::new();
	for (pop, tib) in [(population_multi(), 96usize), (population_ids(), 200usize)]
	{
		let nc = pop[0].1.len();
		let mut rng = StdRng::seed_from_u64(SEED);
		for phase in PHASES
		{
			for k in 0..pop.len() - 1
			{
				let (a, b) = crossover_pool_shuffle(&pop[k], &pop[k + 1], nc, phase, xo_off(tib), &mut rng);
				d.genome(&a.0, &a.1, &a.2);
				d.genome(&b.0, &b.1, &b.2);
				let (a, b) = crossover2(&pop[k], &pop[k + 1], nc, phase, xo_off(tib), &mut rng);
				d.genome(&a.0, &a.1, &a.2);
				d.genome(&b.0, &b.1, &b.2);
				let cfg = ga_config(&pop, phase, tib, (0.0, 0.0, 0.0));
				let m = mutate_ga(&pop[k].0, &pop[k].1, &pop[k].2, &cfg, &mut rng);
				d.genome(&m.0, &m.1, &m.2);
			}
		}
	}
	d.0
}

/// Digest of the DEDUPE crossover path (ga_dedupe_neurons on, fill on and off):
/// the operators that call `dedupe_fill`. The defaults-off digests above never
/// reach it, so the 12/10/2026 UnitGrouping/NeuronKey refactor of
/// `push_distinct` is pinned here instead.
fn dedupe_operator_digest() -> u64
{
	let mut d = Digest::new();
	for (pop, tib) in [(population_multi(), 96usize), (population_ids(), 200usize)]
	{
		let nc = pop[0].1.len();
		let mut rng = StdRng::seed_from_u64(SEED ^ 0xD3D0);
		for fill in [true, false]
		{
			let xo = NeuronXo { dedupe: true, total_input_bits: tib, fill, min_neurons: 2 };
			for k in 0..pop.len() - 1
			{
				let (a, b) = crossover_ps_neurons(&pop[k], &pop[k + 1], nc, xo, &mut rng);
				d.genome(&a.0, &a.1, &a.2);
				d.genome(&b.0, &b.1, &b.2);
				let (a, b) = crossover2_neurons(&pop[k], &pop[k + 1], nc, xo, &mut rng);
				d.genome(&a.0, &a.1, &a.2);
				d.genome(&b.0, &b.1, &b.2);
			}
		}
	}
	d.0
}

/// Captured from c2b8bf97 (pre-refactor `push_distinct`, inline HashSet key).
const GOLDEN_DEDUPE_OPERATORS: u64 = 0xfca5_f895_40ce_c136;

#[test]
fn golden_dedupe_operators_unchanged()
{
	eprintln!("GOLDEN_DEDUPE_OPERATORS = {:#x}", dedupe_operator_digest());
	assert_eq!(dedupe_operator_digest(), GOLDEN_DEDUPE_OPERATORS);
}

/// Captured from commit 4cb4fc62 (pre-change operators).
const GOLDEN_SEARCH: [u64; 64] = [
	13130087642153563210, 15943725861416142908, 6987742805683153432, 6906628960249555962,
	9544838116225842168, 18224622485755037389, 10942774813316834292, 9224631863277957375,
	15901512414421893096, 3278380593733643641, 4953509642262260517, 4183560281427913968,
	8874650687420391469, 17600586105773256196, 10369741240044072346, 1475676748068845558,
	11567106797326243722, 3638408923334769113, 13526507301641529102, 8809165931173123776,
	6339302321027812075, 12177438704719946743, 16747455072026056932, 9196157784588898397,
	2621387483767040367, 1321070145475606268, 7286029156565553706, 3400672088133922836,
	14233530748331951863, 3645272273417455639, 16728804916432324925, 8351052953374296943,
	10742580088940275871, 13541739964806261289, 9272023364883686368, 6471945762765622281,
	172267523491293043, 4468630479488140470, 9718186518998889858, 1254502403907616700,
	17428726419095023640, 6884043247440184400, 17209901156269261627, 10757000452371566290,
	12176797264499403305, 2253244522069747092, 14015485188286715742, 4709618206436553706,
	12293966116007488860, 13675607081385938105, 3716341213506317614, 1787638108842838626,
	16013402658948415888, 4009351201690029725, 9666119744884112315, 16113515272449303668,
	6586341560205424606, 3920298506046487352, 1614834662338920092, 434380721649012505,
	13291588902284855060, 4185706927628993487, 572036171412239197, 10008416046545043935,
];
const GOLDEN_OPERATORS: u64 = 0xee4c_ed71_4e31_832b;

#[test]
fn golden_print()
{
	let s = all_search_digests();
	eprintln!("GOLDEN_SEARCH = {:?}", s);
	eprintln!("GOLDEN_OPERATORS = {:#x}", operator_digest());
	// Repeatable in-process (catches HashMap tie-break flakiness).
	assert_eq!(s, all_search_digests());
}

#[test]
fn golden_search_offspring_defaults_unchanged()
{
	assert_eq!(all_search_digests(), GOLDEN_SEARCH.to_vec());
}

#[test]
fn golden_operators_defaults_unchanged()
{
	assert_eq!(operator_digest(), GOLDEN_OPERATORS);
}

// ---- tie population (11/10/2026: deterministic mode-bits tie-break) -----------

/// IDS-like single cluster whose neurons alternate 10 / 12 bits, so the cluster
/// mode is TIED in every parent (even sizes) and in most crossover children.
/// Under the pre-11/10 HashMap tie-break this population was not reproducible.
fn population_ties() -> Vec<Genome>
{
	let mut rng = StdRng::seed_from_u64(13);
	(0..10)
		.map(|i| {
			let n = 20 + 2 * ((i * 7) % 11);
			let bits: Vec<usize> = (0..n).map(|k| if k % 2 == 0 { 10 } else { 12 }).collect();
			let total: usize = bits.iter().sum();
			let conns: Vec<i64> = (0..total).map(|_| rng.gen_range(0..200i64)).collect();
			(bits, vec![n], conns, 0.5 + i as f64 * 0.03)
		})
		.collect()
}

/// search_offspring digests on the tie population: the phases whose operators
/// read the mode (Neurons, Cluster, Joint) × ratio combos × dedupe off/on.
fn tie_digests() -> Vec<u64>
{
	let pop = population_ties();
	let mut out = Vec::new();
	for phase in [PhaseType::Neurons, PhaseType::Cluster, PhaseType::Joint]
	{
		for ratios in RATIOS
		{
			for dedupe in [false, true]
			{
				let mut cfg = ga_config(&pop, phase, 200, ratios);
				cfg.dedupe_neurons = dedupe;
				out.push(search_digest(&pop, &cfg, 0.0));
			}
		}
	}
	out
}

/// RECORDED 11/10/2026 at 9ed986ac, the first commit with the deterministic
/// (fewest-bits) tie-break. No pre-fix golden exists for this population: under
/// the old HashMap tie-break its digests were not reproducible at all.
const GOLDEN_TIES: [u64; 24] = [
	2670295294962613493, 2670295294962613493, 9142224010583880723, 2970185506096067796,
	9750160952991062454, 3174606918906747558, 9949403992593417485, 9949403992593417485,
	16030816903769521192, 16030816903769521192, 10894041436323278235, 10894041436323278235,
	11632464423398358903, 11632464423398358903, 11843267071221348819, 11843267071221348819,
	592215015617585973, 592215015617585973, 468223564014063847, 15000729928561170539,
	11138139907166796251, 2493226752126226561, 18040611543794862114, 11868948441558813033,
];

#[test]
fn golden_ties_reproducible_across_processes()
{
	assert_eq!(tie_digests(), GOLDEN_TIES.to_vec());
}

#[test]
fn golden_ties_print()
{
	let d = tie_digests();
	eprintln!("GOLDEN_TIES = {:?}", d);
	// Every call builds fresh HashMaps (fresh hash keys): must not move the result.
	for _ in 0..5
	{
		assert_eq!(tie_digests(), d);
	}
}
