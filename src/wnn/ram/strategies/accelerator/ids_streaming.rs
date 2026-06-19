//! Streaming IDS genome evaluator (Option F).
//!
//! Unlike `IDSCache` (which holds the full train/eval matrices in memory),
//! this evaluator holds **only the genome state** — memory cells and
//! per-cluster config — and processes data in chunks. Peak memory is one
//! chunk regardless of dataset size, enabling evaluation on datasets that
//! don't fit in RAM or even on disk.
//!
//! ## Lifecycle (per genome)
//!
//! 1. `IDSGenomeStreamer::new(genome, config)` — initializes empty memory cells.
//! 2. Train phase: stream the training source chunk-by-chunk, calling
//!    `train_chunk(packed, labels)` each time. The chunk's per-row data
//!    drives memory cell writes via the existing `train_genome_in_slot`
//!    kernel; the chunk is dropped after the call.
//! 3. `seal_for_scoring()` — finalizes training. Builds the
//!    `GenomeExport` (memory cell snapshot in GPU-ready form) once.
//! 4. Score phase: stream the eval source chunk-by-chunk, calling
//!    `score_chunk(packed, labels)`. Per-row scores are accumulated in
//!    a small `eval_scores: Vec<f64>` buffer.
//! 5. `finalize_metrics() → (ce, acc, f1, fpr, threshold)` — runs the
//!    same threshold/metric logic as `evaluate_genome_hybrid`'s
//!    single-cluster path on the accumulated scores.
//!
//! ## What this doesn't (yet) support
//!
//! - **undersample_majority**: needs full-dataset class statistics AND
//!   row-level rejection of majority-class samples in the stream. v1
//!   omits this (callers pass `undersample_majority=False`). Adding it
//!   would mean: materialize per-class row indices, build a Bernoulli
//!   accept mask per class, gate train_chunk rows on that mask.
//!
//! Note: **balance_classes** IS supported via class_weights — the caller
//! computes weights from the materialized label array (already in RAM at
//! `IDSDataset.y_train_binary`) and passes them to `new()`. No streaming
//! pre-pass needed — labels are tiny relative to features and materialized
//! once during dataset construction.
//! - **GPU train-side address pre-computation**: `train_genome_in_slot`
//!   accepts an optional `gpu_addresses` slice computed up-front from
//!   the full packed dataset. For streaming, addresses are computed
//!   per-row on the CPU. Eval scoring still uses GPU dispatch via the
//!   already-existing per-chunk path in `compute_per_example_scores`.
//! - **Multi-cluster mode**: v1 supports single-cluster binary
//!   discriminator only. Multi-cluster requires accumulating per-cluster
//!   scores per chunk and computing argmax/CE differently — straightforward
//!   extension but out of v1 scope.

use crate::adaptive::{
    build_groups, build_neuron_metadata, compute_per_example_scores,
    compute_f1_fpr_with_normal_class, export_genome_for_gpu,
    find_optimal_threshold_auto, get_metal_evaluator, get_sparse_metal_evaluator,
    per_cluster_max_bits, reorganize_connections_for_gpu, train_genome_in_slot,
    ConfigGroup, GenomeExport, GroupMemory,
};
use ram_core::neuron_memory::pack_packed_to_u64;
use ram_core::packed_bits::PackedBits;

/// Streaming evaluator state for a single genome.
///
/// One instance per (genome, evaluation pass). The training and scoring
/// phases consume chunks sequentially; only the memory cells (genome state)
/// and the small `eval_scores` accumulator persist across chunks.
pub struct IDSGenomeStreamer {
    // ── Frozen config ────────────────────────────────────────────────────
    num_classes: usize,
    num_negatives: usize,
    num_genome_clusters: usize,
    normal_class: usize,
    /// (w_ce, w_f1, w_fpr, w_acc) — when Some, threshold sweep maximizes fitness.
    pub fitness_weights: Option<(f32, f32, f32, f32)>,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    memory_mode: u8,
    /// Per-class repetition weights for balance_classes=True. None = no balancing.
    /// Computed by the caller (Python) from materialized y_train labels.
    class_weights: Option<Vec<u32>>,

    // ── Genome ───────────────────────────────────────────────────────────
    bits_flat: Vec<usize>,
    neurons_flat: Vec<usize>,
    original_connections: Vec<i64>,

    // ── Derived once at construction ─────────────────────────────────────
    groups: Vec<ConfigGroup>,
    cluster_to_group: Vec<(usize, usize)>,
    cluster_neuron_starts: Vec<usize>,
    neuron_conn_offsets: Vec<usize>,

    // ── Training phase state ────────────────────────────────────────────
    memories: Vec<GroupMemory>,
    train_seen: usize, // count of rows seen during training (for stats)

    // ── Scoring phase state ─────────────────────────────────────────────
    export: Option<GenomeExport>,
    eval_scores: Vec<f64>,
    eval_labels: Vec<i64>,
}

impl IDSGenomeStreamer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bits_flat: Vec<usize>,
        neurons_flat: Vec<usize>,
        connections: Vec<i64>,
        num_classes: usize,
        num_negatives: usize,
        single_cluster: bool,
        normal_class: usize,
        empty_value: f32,
        neuron_sample_rate: f32,
        rng_seed: u64,
        class_weights: Option<Vec<u32>>,
    ) -> Self {
        // Streamer is QUAD-only (Option F shipped after the QUAD mandate).
        let memory_mode = ram_core::neuron_memory::MODE_QUAD_WEIGHTED;
        let bits_per_cluster = per_cluster_max_bits(&bits_flat, &neurons_flat);
        let groups = build_groups(&bits_per_cluster, &neurons_flat);
        let (cluster_neuron_starts, neuron_conn_offsets) =
            build_neuron_metadata(&bits_flat, &neurons_flat);

        let num_clusters = neurons_flat.len();
        let num_genome_clusters = if single_cluster { 1 } else { num_classes };
        let actual_negatives = if single_cluster { 0 } else { num_negatives };

        let mut cluster_to_group = vec![(0usize, 0usize); num_clusters];
        for (group_idx, group) in groups.iter().enumerate() {
            for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                cluster_to_group[cluster_id] = (group_idx, local_idx);
            }
        }

        let memories: Vec<GroupMemory> = groups
            .iter()
            .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
            .collect();

        Self {
            num_classes,
            num_negatives: actual_negatives,
            num_genome_clusters,
            normal_class,
            fitness_weights: None,
            empty_value,
            neuron_sample_rate,
            rng_seed,
            memory_mode,
            class_weights,
            bits_flat,
            neurons_flat,
            original_connections: connections,
            groups,
            cluster_to_group,
            cluster_neuron_starts,
            neuron_conn_offsets,
            memories,
            train_seen: 0,
            export: None,
            eval_scores: Vec::new(),
            eval_labels: Vec::new(),
        }
    }

    /// Train on a single chunk. Writes to memory cells in-place.
    ///
    /// `packed`: bit-packed features for this chunk.
    /// `labels`: class index per row (len == packed.num_rows()).
    pub fn train_chunk(&mut self, packed: &PackedBits, labels: &[i64]) {
        assert_eq!(
            packed.num_rows(),
            labels.len(),
            "train_chunk: feature rows ({}) != label count ({})",
            packed.num_rows(),
            labels.len()
        );
        if self.export.is_some() {
            panic!("train_chunk called after seal_for_scoring()");
        }

        let num_train = packed.num_rows();
        let total_input_bits = packed.total_bits();

        // Negatives per chunk: for each example, the K class indices that are
        // NOT the target. For single_cluster mode (num_negatives=0), this is
        // empty. For multi-class, exhaustive enumeration: all classes except target.
        let negatives = compute_chunk_negatives(labels, self.num_classes, self.num_negatives);

        train_genome_in_slot(
            &mut self.memories,
            &self.groups,
            &self.original_connections,
            &self.bits_flat,
            &self.cluster_neuron_starts,
            &self.neuron_conn_offsets,
            &self.cluster_to_group,
            packed,
            labels,
            &negatives,
            num_train,
            self.num_negatives,
            total_input_bits,
            None, // gpu_addresses: per-row CPU compute (streaming v1)
            self.neuron_sample_rate,
            self.rng_seed.wrapping_add(self.train_seen as u64),
            self.memory_mode,
            self.class_weights.as_deref(),
            true, // parallel
        );

        self.train_seen += num_train;
    }

    /// Finalize training. Builds the GPU-ready genome export so subsequent
    /// `score_chunk` calls can dispatch to Metal where dense.
    pub fn seal_for_scoring(&mut self) {
        if self.export.is_some() {
            panic!("seal_for_scoring called twice");
        }
        let gpu_connections = reorganize_connections_for_gpu(
            &self.original_connections,
            &self.bits_flat,
            &self.neurons_flat,
            &self.groups,
        );
        self.export = Some(export_genome_for_gpu(
            &self.memories,
            &self.groups,
            &gpu_connections,
        ));
    }

    /// Score a single eval chunk. Accumulates per-row scores into the
    /// `eval_scores` buffer (small — one f64 per eval row).
    pub fn score_chunk(&mut self, packed: &PackedBits, labels: &[i64]) {
        assert_eq!(
            packed.num_rows(),
            labels.len(),
            "score_chunk: feature rows ({}) != label count ({})",
            packed.num_rows(),
            labels.len()
        );
        let export = self.export.as_ref().expect(
            "score_chunk called before seal_for_scoring — finalize training first",
        );

        let num_eval = packed.num_rows();
        let total_input_bits = packed.total_bits();
        let (packed_eval_u64, words_per_example) = pack_packed_to_u64(packed);

        let metal_arc = get_metal_evaluator();
        let sparse_metal_arc = get_sparse_metal_evaluator();
        let metal = metal_arc.as_ref().map(|a| a.as_ref());
        let sparse_metal = sparse_metal_arc.as_ref().map(|a| a.as_ref());

        let chunk_scores = compute_per_example_scores(
            export,
            packed,
            &packed_eval_u64,
            words_per_example,
            num_eval,
            self.num_genome_clusters,
            total_input_bits,
            self.empty_value,
            self.memory_mode,
            metal,
            sparse_metal,
        );

        // v1: single_cluster binary path only — score[ex][0] is the
        // attack probability. Multi-cluster would accumulate per-cluster
        // scores differently.
        for scores in chunk_scores {
            self.eval_scores.push(scores[0]);
        }
        self.eval_labels.extend_from_slice(labels);
    }

    /// Compute final metrics from the accumulated eval scores.
    ///
    /// Returns (ce, acc, f1, fpr, threshold). Threshold is auto-selected
    /// on the eval data (matches `evaluate_genome_hybrid`'s single-cluster
    /// path when override_threshold=None).
    pub fn finalize_metrics(&self) -> (f64, f64, f64, f64, f64) {
        let epsilon = 1e-10f64;
        let num_eval = self.eval_scores.len();
        assert!(num_eval > 0, "finalize_metrics: no scores accumulated");

        // BCE loss (threshold-independent)
        let mut total_ce = 0.0f64;
        for (i, &s) in self.eval_scores.iter().enumerate() {
            let s = s.clamp(epsilon, 1.0 - epsilon);
            let y = self.eval_labels[i] as f64;
            total_ce += -(y * s.ln() + (1.0 - y) * (1.0 - s).ln());
        }
        let ce = total_ce / num_eval as f64;

        // Auto-find threshold + compute predictions/metrics at it
        let (threshold, _f1_at_thr, _fpr_at_thr) =
            find_optimal_threshold_auto(&self.eval_scores, &self.eval_labels, self.fitness_weights);

        let mut correct = 0u64;
        let mut predictions = Vec::with_capacity(num_eval);
        for (i, &s) in self.eval_scores.iter().enumerate() {
            let pred = if s >= threshold { 1u32 } else { 0u32 };
            predictions.push(pred);
            if pred as i64 == self.eval_labels[i] {
                correct += 1;
            }
        }
        let acc = correct as f64 / num_eval as f64;
        let (f1, fpr) = compute_f1_fpr_with_normal_class(
            &predictions,
            &self.eval_labels,
            2,
            self.normal_class,
        );

        (ce, acc, f1, fpr, threshold)
    }

    /// Number of training rows seen so far.
    pub fn train_seen(&self) -> usize {
        self.train_seen
    }

    /// Number of eval rows scored so far.
    pub fn eval_scored(&self) -> usize {
        self.eval_scores.len()
    }
}

/// For each label, fill in `num_negatives` class indices that are NOT the
/// target. For exhaustive enumeration (`num_negatives >= num_classes - 1`),
/// emits all non-target classes. For fewer, picks the first num_negatives.
/// For single_cluster mode (num_negatives=0), returns an empty Vec.
fn compute_chunk_negatives(labels: &[i64], num_classes: usize, num_negatives: usize) -> Vec<i64> {
    if num_negatives == 0 {
        return Vec::new();
    }
    let n = labels.len();
    let mut negatives = vec![0i64; n * num_negatives];
    for (ex, &target) in labels.iter().enumerate() {
        let mut k = 0;
        for c in 0..num_classes as i64 {
            if c != target && k < num_negatives {
                negatives[ex * num_negatives + k] = c;
                k += 1;
            }
        }
        // Fill remainder by cycling through non-target classes (matches
        // ids_cache::build_subset behavior when num_negatives > num_classes - 1).
        while k < num_negatives {
            for c in 0..num_classes as i64 {
                if c != target && k < num_negatives {
                    negatives[ex * num_negatives + k] = c;
                    k += 1;
                }
            }
        }
    }
    negatives
}
