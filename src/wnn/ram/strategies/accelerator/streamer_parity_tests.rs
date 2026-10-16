//! IDSGenomeStreamer == in-memory IDSCache, for every memory_mode × write rule
//! × single_cluster (16/10/2026, worker ABI 20).
//!
//! Until ABI 20 the streamer hardcoded QUAD_WEIGHTED: a TERNARY / QUAD_BINARY /
//! BINARY / QSR / PLN streaming flow trained and scored as QUAD under its own
//! label. It now takes the same `EvalSettings` the cache builds
//! (`IDSCache::eval_settings`). Pinned here, on one small dataset with a val
//! partition, for each (mode, rule, single_cluster, dense/sparse genome,
//! balance_classes, coverage_aware, neuron_sample_rate):
//!   - search metrics: `finalize_metrics` == `evaluate_genomes_ids_cached_full_hybrid`
//!     (the GA-fitness path; (ce, acc, f1, fpr, threshold));
//!   - validation scores: eval / train / val scores bit-identical to
//!     `evaluate_at_thresholds_ids_cached` (single-cluster; plus its metrics at
//!     a fixed and the oracle threshold) or to
//!     `evaluate_multiclass_at_thresholds_ids_cached` (K clusters — K = 2 binary
//!     two-cluster incl. margin_train_f1 / margin_val_f1, and K = 3; every
//!     decode mode compared).
//! The streamer trains on the CPU slot trainer in 3 uneven chunks and scores in
//! 2; the cache trains through the Metal marker trainer when present, so this is
//! also a CPU == GPU training parity sweep.
//!
//! The 4-state modes (QUAD_BINARY / QUAD_WEIGHTED / QSR) are only deterministic
//! under the OI trainer (production IDS: the worker sets
//! WNN_ORDER_INDEPENDENT_TRAIN=1); without it their legacy clamped nudge is
//! order-dependent on BOTH paths, so they are skipped (logged) and the suite is
//! run a second time with the env set:
//!   WNN_ORDER_INDEPENDENT_TRAIN=1 cargo test -p ram_accelerator --lib --no-default-features streamer_parity

use crate::ids_cache::{
	evaluate_at_thresholds_ids_cached, evaluate_genomes_ids_cached_full_hybrid,
	evaluate_multiclass_at_thresholds_ids_cached, IDSCache,
};
use crate::ids_streaming::IDSGenomeStreamer;
use ram_core::cell_mode::{CellMode, WriteRule};
use ram_core::neuron_memory::{EvalSettings, BINARY, PLN, QSR, QUAD_BINARY, QUAD_WEIGHTED, TERNARY};
use ram_core::packed_bits::PackedBits;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const TIB: usize = 24;
const RUN_SEED: u64 = 11;
const ALL_MODES: [u8; 6] = [TERNARY, QUAD_BINARY, QUAD_WEIGHTED, BINARY, QSR, PLN];

/// One labelled partition.
struct Split
{
	rows: PackedBits,
	labels: Vec<i64>,
}

impl Split
{
	/// `n` rows whose label is a noisy function of bits 0..3 (so the genome
	/// learns something and the metrics are not degenerate).
	fn new(n: usize, classes: usize, rng: &mut SmallRng) -> Self
	{
		let mut bools = Vec::with_capacity(n * TIB);
		let mut labels = Vec::with_capacity(n);
		for _ in 0..n
		{
			let row: Vec<bool> = (0..TIB).map(|_| rng.gen_bool(0.5)).collect();
			let signal = row[..3].iter().filter(|&&b| b).count() % classes;
			labels.push(if rng.gen_bool(0.85) { signal } else { rng.gen_range(0..classes) } as i64);
			bools.extend(row);
		}
		Self { rows: PackedBits::from_bool_slice(&bools, TIB), labels }
	}

	/// Rows `range` as their own PackedBits (a streamed chunk).
	fn chunk(&self, range: std::ops::Range<usize>) -> (PackedBits, &[i64])
	{
		let mut part = PackedBits::empty(TIB);
		range.clone().for_each(|r| part.extend_row_from(&self.rows, r));
		(part, &self.labels[range])
	}
}

struct Dataset
{
	classes: usize,
	train: Split,
	eval: Split,
	val: Split,
}

impl Dataset
{
	fn new(classes: usize) -> Self
	{
		let mut rng = SmallRng::seed_from_u64(0xD5 + classes as u64);
		Self {
			classes,
			train: Split::new(420, classes, &mut rng),
			eval: Split::new(210, classes, &mut rng),
			val: Split::new(150, classes, &mut rng),
		}
	}
}

/// One cell of the parity matrix.
#[derive(Clone, Copy, Debug)]
struct Case
{
	mode: u8,
	vote: bool,
	clusters: usize, // 1 = single-cluster binary; 2 = binary two-cluster; 3 = multiclass
	bits: usize,     // 8 = dense groups, 14 = sparse
	balance: bool,
	coverage_aware: bool,
	rate: f32,
}

impl Case
{
	fn single_cluster(&self) -> bool
	{
		self.clusters == 1
	}

	fn num_classes(&self) -> usize
	{
		self.clusters.max(2)
	}

	/// Genome: 5 neurons per cluster, two of them DUPLICATES (the multiplicity
	/// layer collapses them at rate 1.0 under an order-independent writer).
	fn genome(&self) -> (Vec<usize>, Vec<usize>, Vec<i64>)
	{
		let npc = vec![5usize; self.clusters];
		let mut conns = Vec::new();
		for c in 0..self.clusters
		{
			for n in [0usize, 1, 0, 2, 1]
			{
				conns.extend((0..self.bits).map(|k| ((c * 3 + n * 5 + k * 7) % TIB) as i64));
			}
		}
		(vec![self.bits; 5 * self.clusters], npc, conns)
	}

	fn settings(&self) -> EvalSettings
	{
		EvalSettings {
			empty_value: 0.5,
			memory_mode: self.mode,
			normal_class: 0,
			fitness_weights: None,
			run_seed: RUN_SEED,
			coverage_aware: self.coverage_aware,
			write_vote: self.vote,
		}
	}

	/// The in-memory cache, configured exactly as IDSEvaluator configures it.
	fn cache(&self, data: &Dataset) -> IDSCache
	{
		let mut cache = IDSCache::new(
			data.train.rows.clone(),
			data.train.labels.clone(),
			data.eval.rows.clone(),
			data.eval.labels.clone(),
			Some(data.val.rows.clone()),
			Some(data.val.labels.clone()),
			self.num_classes(),
			TIB,
			5,
			self.num_classes() - 1,
			42,
			self.balance,
			self.single_cluster(),
			false,
			1.0,
		);
		let s = self.settings();
		cache.memory_mode = s.memory_mode;
		cache.run_seed = s.run_seed;
		cache.coverage_aware = s.coverage_aware;
		cache.write_vote = s.write_vote;
		cache
	}

	/// A streamer trained on `data.train` in 3 uneven chunks and sealed.
	fn streamer(&self, data: &Dataset, cache: &IDSCache) -> IDSGenomeStreamer
	{
		let (bits, npc, conns) = self.genome();
		let mut s = IDSGenomeStreamer::new(
			bits,
			npc,
			conns,
			self.num_classes(),
			self.num_classes() - 1,
			self.single_cluster(),
			self.settings(),
			self.rate,
			0, // the in-memory paths' rng_seed (IDSEvaluator passes 0 to both)
			cache.class_weights().map(<[u32]>::to_vec),
		)
		.expect("valid streamer settings");
		for range in [0..150, 150..280, 280..420]
		{
			let (rows, labels) = data.train.chunk(range);
			s.train_chunk(&rows, labels);
		}
		s.seal_for_scoring();
		s
	}
}

/// Score `split` through the streamer in 2 uneven chunks (accumulated).
fn stream_scores_into(s: &mut IDSGenomeStreamer, split: &Split)
{
	let mid = split.labels.len() / 3;
	for range in [0..mid, mid..split.labels.len()]
	{
		let (rows, labels) = split.chunk(range);
		s.score_chunk(&rows, labels);
	}
}

/// Score `split` through the streamer; drain the scores.
fn stream_scores(s: &mut IDSGenomeStreamer, split: &Split) -> Vec<f64>
{
	stream_scores_into(s, split);
	s.take_scores()
}

fn bits_of(v: &[f64]) -> Vec<u64>
{
	v.iter().map(|x| x.to_bits()).collect()
}

/// Search metrics: the GA-fitness path on the full train/eval split.
fn assert_search_parity(case: &Case, data: &Dataset, cache: &IDSCache)
{
	let mut s = case.streamer(data, cache);
	if case.single_cluster()
	{
		// The in-memory default: threshold calibrated on the TRAIN scores.
		stream_scores_into(&mut s, &data.train);
		s.calibrate_threshold().expect("single-cluster calibration");
	}
	stream_scores_into(&mut s, &data.eval);
	let streamed = s.finalize_metrics(None).expect("finalize");
	let (bits, npc, conns) = case.genome();
	let (ce, acc, f1, fpr, thr, _ms) =
		evaluate_genomes_ids_cached_full_hybrid(cache, &bits, &npc, &conns, 1, 0.5, case.rate, 0, None)[0];
	assert_eq!(
		(streamed.1, streamed.2, streamed.3, streamed.4),
		(acc, f1, fpr, thr),
		"{case:?}: search (acc, f1, fpr, threshold) streaming != in-memory"
	);
	// CE is exact on the CPU paths; the in-memory K-cluster fast path may
	// reduce CE on the GPU (MetalSparseCEEvaluator), so allow f32 accumulation.
	assert!((streamed.0 - ce).abs() <= 1e-6 * ce.abs().max(1.0), "{case:?}: search CE {} != {ce}", streamed.0);
}

/// Validation: scores (and everything derived from them) on eval/train/val.
fn assert_validation_parity(case: &Case, data: &Dataset, cache: &IDSCache)
{
	let mut s = case.streamer(data, cache);
	let eval = stream_scores(&mut s, &data.eval);
	let train = stream_scores(&mut s, &data.train);
	let val = stream_scores(&mut s, &data.val);
	let (bits, npc, conns) = case.genome();
	if case.single_cluster()
	{
		let (m_eval, m_train, m_val, metrics, _fp) =
			evaluate_at_thresholds_ids_cached(cache, &bits, &npc, &conns, &[0.5, -1.0], 0.5, case.rate, 0);
		assert_eq!(bits_of(&eval), bits_of(&m_eval), "{case:?}: eval scores");
		assert_eq!(bits_of(&train), bits_of(&m_train), "{case:?}: train scores");
		assert_eq!(bits_of(&val), bits_of(&m_val.expect("cache holds val")), "{case:?}: val scores");
		for &(ce, acc, f1, fpr, t) in &metrics
		{
			let ours = crate::adaptive::compute_binary_metrics_at_threshold(&eval, &data.eval.labels, t, 0);
			assert_eq!(ours, (ce, acc, f1, fpr), "{case:?}: metrics at threshold {t}");
		}
		return;
	}
	let (k, modes, _fp) = evaluate_multiclass_at_thresholds_ids_cached(cache, &bits, &npc, &conns, 0.5, case.rate, 0);
	let ours = crate::multiclass_metrics::modes_from_scores(
		&eval,
		&data.eval.labels,
		&train,
		&data.train.labels,
		Some(&val),
		Some(&data.val.labels),
		k,
		0,
	);
	if k == 2
	{
		let names: Vec<&str> = ours.iter().map(|m| m.mode.as_str()).collect();
		assert!(names.contains(&"margin_train_f1") && names.contains(&"margin_val_f1"), "{case:?}: {names:?}");
	}
	assert_eq!(format!("{ours:?}"), format!("{modes:?}"), "{case:?}: K={k} decode modes");
}

/// Every case of the matrix that this process can run deterministically.
fn cases() -> Vec<Case>
{
	let oi = ram_core::neuron_memory::order_independent_training_enabled();
	let mut out = Vec::new();
	for mode in ALL_MODES
	{
		let four_state = CellMode::from_u8(mode).expect("mode").write_rule() == WriteRule::Nudge;
		if four_state && !oi
		{
			eprintln!("[streamer_parity] mode {mode}: needs WNN_ORDER_INDEPENDENT_TRAIN=1 (legacy nudge is order-dependent) — skipped");
			continue;
		}
		let votes: &[bool] = if four_state { &[false] } else { &[false, true] };
		for &vote in votes
		{
			for clusters in [1usize, 2, 3]
			{
				for bits in [8usize, 14]
				{
					for (balance, coverage_aware) in [(false, false), (true, true), (false, true)]
					{
						for rate in [1.0f32, 0.5]
						{
							out.push(Case { mode, vote, clusters, bits, balance, coverage_aware, rate });
						}
					}
				}
			}
		}
	}
	out
}

#[test]
fn streamer_parity_every_mode_rule_and_cluster_layout()
{
	let datasets = [Dataset::new(2), Dataset::new(3)];
	let cases = cases();
	assert!(!cases.is_empty());
	// Every case runs (a failure is collected, not fatal) so one run reports
	// the whole failing set.
	let failed: Vec<&Case> = cases
		.iter()
		.filter(|case| {
			let data = &datasets[case.num_classes() - 2];
			assert_eq!(data.classes, case.num_classes());
			let run = || {
				let cache = case.cache(data);
				assert_search_parity(case, data, &cache);
				assert_validation_parity(case, data, &cache);
			};
			std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err()
		})
		.collect();
	assert!(failed.is_empty(), "{} of {} streamer parity cases differ: {failed:#?}", failed.len(), cases.len());
	eprintln!("[streamer_parity] {} cases identical", cases.len());
}

/// The streamer refuses what the cache refuses: vote on a 4-state mode, and
/// an unknown mode code.
#[test]
fn streamer_refuses_vote_on_four_state_and_unknown_modes()
{
	let case = Case { mode: QUAD_WEIGHTED, vote: false, clusters: 1, bits: 8, balance: false, coverage_aware: false, rate: 1.0 };
	let (bits, npc, conns) = case.genome();
	let build = |settings: EvalSettings| {
		IDSGenomeStreamer::new(bits.clone(), npc.clone(), conns.clone(), 2, 1, true, settings, 1.0, 0, None).err()
	};
	for mode in [QUAD_BINARY, QUAD_WEIGHTED, QSR]
	{
		let err = build(EvalSettings { memory_mode: mode, write_vote: true, ..EvalSettings::default() });
		assert!(err.is_some_and(|e| e.contains("BINARY/TERNARY/PLN only")), "mode {mode}");
	}
	for mode in [BINARY, TERNARY, PLN]
	{
		assert!(build(EvalSettings { memory_mode: mode, write_vote: true, ..EvalSettings::default() }).is_none());
	}
	let err = build(EvalSettings { memory_mode: 99, ..EvalSettings::default() });
	assert!(err.is_some_and(|e| e.contains("unknown memory_mode")));
}

/// A non-QUAD streamer must not train as QUAD (the ABI-19 bug): BINARY's
/// one-shot Set gives different scores from QUAD_WEIGHTED's nudge.
#[test]
fn streamer_mode_reaches_the_cells()
{
	let data = Dataset::new(2);
	let score = |mode: u8| {
		let case = Case { mode, vote: false, clusters: 1, bits: 8, balance: false, coverage_aware: false, rate: 1.0 };
		let cache = case.cache(&data);
		let mut s = case.streamer(&data, &cache);
		bits_of(&stream_scores(&mut s, &data.eval))
	};
	assert_ne!(score(BINARY), score(QUAD_WEIGHTED));
	assert_ne!(score(TERNARY), score(BINARY));
}
