//! Genome training + evaluation: parallel CPU, hybrid CPU+GPU, single-genome paths.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

use super::*;

/// Evaluate multiple genomes SEQUENTIALLY with METAL GPU ACCELERATION.
///
/// This is the KEY acceleration function for GA optimization.
/// Each genome is evaluated independently with its own memory.
///
/// Hybrid acceleration strategy:
/// - Training: CPU with rayon parallelism (random writes)
/// - Evaluation: Metal GPU for dense groups (40 cores on M4 Max)
///               CPU for sparse groups (hash lookups not GPU-friendly)
///
/// Performance: Metal accelerates evaluation by ~10-20x for dense groups,
/// which typically contain 80%+ of clusters (those with bits <= 12).
///
/// Args:
///   genomes_bits_flat: [num_genomes * num_clusters] bits per cluster for each genome
///   genomes_neurons_flat: [num_genomes * num_clusters] neurons per cluster for each genome
///   genomes_connections_flat: [num_genomes * total_connections] flattened connection indices, or empty for random
///   num_genomes: Number of genomes to evaluate
///   num_clusters: Number of clusters (vocab size)
///   train_input_bits: [num_train * total_input_bits] training contexts
///   train_targets: [num_train] target cluster for each training example
///   train_negatives: [num_train * num_negatives] negative clusters
///   num_train: Number of training examples
///   num_negatives: Number of negative samples per example
///   eval_input_bits: [num_eval * total_input_bits] evaluation contexts
///   eval_targets: [num_eval] target cluster for each eval example
///   num_eval: Number of evaluation examples
///   total_input_bits: Input bits per example
///   empty_value: Value for EMPTY cells (0.0 recommended)
///
/// Returns: [num_genomes] cross-entropy values (lower is better)
/// Evaluate multiple genomes SEQUENTIALLY, returning (CE, accuracy) for each.
///
/// Genomes are evaluated one at a time to:
/// 1. Prevent memory explosion (only 1 genome's memory allocated)
/// 2. Allow full CPU utilization for each genome's training/eval (16 cores)
/// 3. Avoid thread pool contention from nested parallelism
///
/// Each genome's training (200K examples) and evaluation (50K examples)
/// use full rayon parallelism internally.
///
/// IMPORTANT: Connections must be provided for proper evolutionary search.
/// If genomes_connections_flat is empty, random connections are generated.
///
/// Returns Vec of (cross_entropy, accuracy) tuples - one per genome.
///
/// Architectural unification (path2-lm-followup): this is now a thin wrapper
/// around `evaluate_genomes_parallel_hybrid`, which is the same training+eval
/// implementation used by the IDS path. Single training entry point, single
/// eval path, single Path 2 marker / dense routing.
///
/// Set `WNN_LM_USE_LEGACY_TRAIN=1` to fall back to the original
/// `_legacy` implementation (kept for parity testing and emergency rollback;
/// can be removed once LM workloads are validated on the unified path).
pub fn evaluate_genomes_parallel(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64, f64, f64)> {
    if std::env::var("WNN_LM_USE_LEGACY_TRAIN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return evaluate_genomes_parallel_legacy(
            genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
            num_genomes, num_clusters,
            train_input_bits, train_targets, train_negatives,
            num_train, num_negatives,
            eval_input_bits, eval_targets, num_eval,
            total_input_bits, settings, neuron_sample_rate, rng_seed,
        );
    }
    // Unified path: delegate to the IDS-shaped hybrid implementation which
    // already supports Path 2 marker training, dense fallback, OI, and
    // hybrid policy routing.
    let results = evaluate_genomes_parallel_hybrid(
        genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
        num_genomes, num_clusters,
        train_input_bits, train_targets, train_negatives,
        num_train, num_negatives,
        eval_input_bits, eval_targets, num_eval,
        total_input_bits, settings, neuron_sample_rate, rng_seed,
        None, // class_weights: LM doesn't use class balancing
    );
    // Drop the threshold and per_genome_ms fields (LM API contract: 4-tuple).
    results.into_iter().map(|(ce, acc, f1, fpr, _, _)| (ce, acc, f1, fpr)).collect()
}

/// Legacy LM training+eval path. Preserved as a fallback under
/// `WNN_LM_USE_LEGACY_TRAIN=1`. The unified path (above) is the default.
pub(crate) fn evaluate_genomes_parallel_legacy(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64, f64, f64)> {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;
    use rand::prelude::*;
    use rand::SeedableRng;

    // Check if connections are provided
    let use_provided_connections = !genomes_connections_flat.is_empty();

    // Pre-compute genome_bpn_offsets: genomes_bits_flat has total_neurons entries per genome
    // (per-neuron bits), NOT num_clusters entries.
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    debug_assert_eq!(
        genomes_bits_flat.len(),
        *genome_bpn_offsets.last().unwrap(),
        "genomes_bits_flat length ({}) != expected total neurons ({})",
        genomes_bits_flat.len(),
        genome_bpn_offsets.last().unwrap(),
    );

    // Pre-compute per-genome connection offsets: conn_size = sum of per-neuron bits
    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    // Pack input bits to u64 once (shared across all genomes for GPU address computation)
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_packed_to_u64(train_input_bits);

    // Check if progress logging is enabled via env var
    let progress_log = std::env::var("WNN_PROGRESS_LOG").map(|v| v == "1").unwrap_or(false);
    let log_path = std::env::var("WNN_LOG_PATH").ok();
    // Get generation info from env vars (set by Python before calling)
    let current_gen: usize = std::env::var("WNN_PROGRESS_GEN")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let total_gens: usize = std::env::var("WNN_PROGRESS_TOTAL_GENS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    // Log type: Init, New, Nbr, CE, Acc (default Init)
    let log_type = std::env::var("WNN_PROGRESS_TYPE").unwrap_or_else(|_| "Init".to_string());
    // Offset for batch position (e.g., batch starting at genome 11 in a 50-genome set)
    let batch_offset: usize = std::env::var("WNN_PROGRESS_OFFSET")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    // Total count (for showing X/50 instead of X/batch_size)
    let total_count: usize = std::env::var("WNN_PROGRESS_TOTAL")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(num_genomes);
    let _start_time = std::time::Instant::now();

    // SEQUENTIAL genome evaluation - each genome gets full thread pool for token parallelism
    // Parallel genome eval causes contention: 10 genomes × nested token parallelism = thrashing
    // Sequential is faster: ~6s/genome vs ~10s/genome with parallel outer loop
    let results: Vec<(f64, f64, f64, f64)> = (0..num_genomes).map(|genome_idx| {
        let genome_start = std::time::Instant::now();
        // Extract this genome's per-neuron bits and per-cluster neurons
        let genome_offset = genome_idx * num_clusters;
        let neurons_per_cluster: Vec<usize> = genomes_neurons_flat[genome_offset..genome_offset + num_clusters].to_vec();

        // Extract per-neuron bits for this genome
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let per_neuron_bits: Vec<usize> = genomes_bits_flat[bpn_start..bpn_end].to_vec();

        // Compute per-cluster max bits (for build_groups and GPU dispatch)
        let bits_per_cluster = per_cluster_max_bits(&per_neuron_bits, &neurons_per_cluster);

        // Build neuron metadata for per-neuron training and CPU eval
        let (cluster_neuron_starts, neuron_conn_offsets) =
            build_neuron_metadata(&per_neuron_bits, &neurons_per_cluster);

        // Build config groups for this genome (using per-cluster max bits)
        let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);

        // Create hybrid memory for each config group
        let mut group_memories: Vec<GroupMemory> = groups.iter()
            .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
            .collect();

        // Get original per-neuron connections for this genome
        let original_connections: Vec<i64> = if use_provided_connections {
            let conn_offset = conn_offsets[genome_idx];
            let conn_size = conn_sizes[genome_idx];
            genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec()
        } else {
            // Generate random per-neuron connections (legacy fallback)
            let total_conn: usize = per_neuron_bits.iter().sum();
            let mut rng = rand::rngs::SmallRng::from_entropy();
            let mut conns: Vec<i64> = Vec::with_capacity(total_conn);
            for _ in 0..total_conn {
                conns.push(rng.gen_range(0..total_input_bits as i64));
            }
            conns
        };

        // Build GPU-padded connections (per-neuron → group layout with padding)
        let gpu_connections = reorganize_connections_for_gpu(
            &original_connections,
            &per_neuron_bits,
            &neurons_per_cluster,
            &groups,
        );

        // Build cluster-to-group mapping
        let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
        for (group_idx, group) in groups.iter().enumerate() {
            for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                cluster_to_group[cluster_id] = (group_idx, local_idx);
            }
        }

        // Compute training addresses on GPU (falls back to CPU if unavailable)
        let gpu_addresses = try_gpu_addresses_adaptive(
            &packed_train_input,
            words_per_example,
            &per_neuron_bits,
            &neuron_conn_offsets,
            &original_connections,
            num_train,
        );

        // Train this genome using per-neuron bits (PARALLEL across examples)
        train_genome_in_slot(
            &mut group_memories,
            &groups,
            &original_connections,
            &per_neuron_bits,
            &cluster_neuron_starts,
            &neuron_conn_offsets,
            &cluster_to_group,
            train_input_bits,
            train_targets,
            train_negatives,
            num_train,
            num_negatives,
            total_input_bits,
            gpu_addresses.as_deref(),
            neuron_sample_rate,
            rng_seed.wrapping_add(genome_idx as u64),
            memory_mode,
            None, // class_weights: only used by IDS via evaluate_genomes_parallel_hybrid
            true, // parallel: safe here, not inside outer par_iter
        );

        // Evaluate this genome - HYBRID Metal/CPU acceleration
        // - Dense groups (bits <= 12): Metal GPU (all examples at once)
        // - Sparse groups (bits > 12): CPU (hash lookups not GPU-friendly)
        let epsilon = 1e-10f64;

        // Pre-compute scores for all examples × clusters
        // Shape: [num_eval][num_clusters]
        let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];

        // Get Metal evaluators (lazy init, thread-safe)
        // These are Arc<T> so we can clone and hold references across the loop
        let metal = get_metal_evaluator();
        let sparse_metal = get_sparse_metal_evaluator();

        // Pack eval input bits to u64 for GPU (pack once, reuse for all groups)
        let (packed_eval, words_per_example) = crate::neuron_memory::pack_packed_to_u64(eval_input_bits);

        // Process each group - Metal for dense, GPU sparse for sparse, CPU fallback
        for (group_idx, group) in groups.iter().enumerate() {
            let memory = &group_memories[group_idx];

            if let (Some(ref metal_eval), true) = (&metal, memory.is_dense()) {
                // Metal path: evaluate all examples at once for this dense group
                // GPU uses padded connections (group layout with max_bits per neuron)
                if let Some(memory_words) = memory.export_for_metal() {
                    match evaluate_group_metal(
                        metal_eval.as_ref(),
                        &packed_eval,
                        &gpu_connections,
                        &memory_words,
                        group,
                        num_eval,
                        words_per_example,
                        memory_mode,
                        empty_value,
                    ) {
                        Ok(group_scores) => {
                            for ex_idx in 0..num_eval {
                                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                    let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                    all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                                }
                            }
                            continue;
                        }
                        Err(_e) => {
                        }
                    }
                }
            }

            // GPU sparse path: evaluate sparse groups using binary search on GPU
            if let (Some(ref sparse_eval), true) = (&sparse_metal, memory.is_sparse()) {
                if let Some(export) = memory.export_for_gpu_sparse() {
                    match evaluate_group_sparse_gpu(
                        sparse_eval.as_ref(),
                        &packed_eval,
                        &gpu_connections,
                        &export,
                        group,
                        num_eval,
                        words_per_example,
                        memory_mode,
                        empty_value,
                    ) {
                        Ok(group_scores) => {
                            for ex_idx in 0..num_eval {
                                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                    let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                    all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                                }
                            }
                            continue;
                        }
                        Err(_e) => {
                        }
                    }
                }
            }

            // CPU path: evaluate examples in parallel using per-neuron bits
            all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                let input_bits = eval_input_bits.packed_row(ex_idx);

                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                    let actual_neurons = if let Some(ref an) = group.actual_neurons {
                        an[local_cluster] as usize
                    } else {
                        group.neurons
                    };

                    let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

                    let mut sum = 0.0f32;
                    for n in 0..actual_neurons {
                        let global_n = cluster_neuron_starts[cluster_id] + n;
                        let n_bits = per_neuron_bits[global_n];
                        let conn_start = neuron_conn_offsets[global_n];
                        let address = crate::neuron_memory::compute_address_packed_bytes(input_bits, &original_connections[conn_start..], n_bits);
                        let cell = memory.read(neuron_base + n, address);
                        sum += cell_to_weight(cell, memory_mode, empty_value);
                    }

                    scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                }
            });
        }

        // Extract predictions and compute CE/accuracy from pre-computed scores
        let predictions: Vec<u32> = all_scores.par_iter()
            .map(|scores| {
                scores.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            })
            .collect();

        let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
            let target_idx = eval_targets[ex_idx] as usize;
            let predicted = predictions[ex_idx] as usize;
            let correct: u64 = if predicted == target_idx { 1 } else { 0 };

            // Softmax and cross-entropy for this example
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            let target_prob = exp_scores[target_idx] / sum_exp;
            let ce = -(target_prob + epsilon).ln();

            (ce, correct)
        }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

        let avg_ce = total_ce / num_eval as f64;
        let accuracy = total_correct as f64 / num_eval as f64;
        let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, settings.normal_class);

        // Progress logging (to log file if WNN_LOG_PATH set, otherwise stderr)
        // Format matches Python's format_genome_log for consistency
        if progress_log {
            use std::io::Write;
            let genome_elapsed = genome_start.elapsed().as_secs_f64();
            let now = chrono::Local::now();

            // Calculate overall position using batch offset
            let overall_position = batch_offset + genome_idx + 1;

            // Calculate padding widths based on totals
            let gen_width = total_gens.to_string().len();
            let pos_width = total_count.to_string().len();

            // Pad type indicator to 4 chars (Init, New , Nbr , CE  , Acc )
            let type_padded = format!("{:<4}", &log_type[..log_type.len().min(4)]);

            // Format: [Gen 001/100] Genome 01/50 (Init): CE=10.6588, Acc=0.0100% (8.7s)
            let msg = format!(
                "{} | [Gen {:0gen_width$}/{:0gen_width$}] Genome {:0pos_width$}/{} ({}): CE={:.4}, Acc={:.4}% ({:.1}s)\n",
                now.format("%H:%M:%S"),
                current_gen, total_gens,
                overall_position, total_count,
                type_padded,
                avg_ce, accuracy * 100.0,
                genome_elapsed,
                gen_width = gen_width,
                pos_width = pos_width,
            );
            if let Some(ref path) = log_path {
                use fs2::FileExt;
                if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
                    // Lock file to prevent interleaved writes with Python
                    if file.lock_exclusive().is_ok() {
                        let _ = file.write_all(msg.as_bytes());
                        let _ = file.flush();
                        let _ = file.unlock();
                    }
                }
            } else {
                eprint!("{}", msg);
            }
        }

        (avg_ce, accuracy, f1, fpr)
    }).collect();

    results
}

// =============================================================================
// PARALLEL HYBRID CPU+GPU EVALUATION
// =============================================================================

/// Export data for a single genome (used in batched GPU evaluation)
#[derive(Clone)]
pub struct GenomeExport {
    /// Connections for all groups, flattened
    pub connections: Vec<i64>,
    /// For each group: (is_sparse, group_idx, cluster_ids)
    pub group_info: Vec<(bool, usize, Vec<usize>)>,
    /// Dense group exports: memory words
    pub dense_exports: Vec<Vec<i64>>,
    /// Sparse group exports: sorted arrays for binary search
    pub sparse_exports: Vec<SparseGpuExport>,
    /// Config groups for this genome
    pub groups: Vec<ConfigGroup>,
}

impl GenomeExport {
    /// Sentinel: an "empty" export used as a partial-result placeholder when
    /// a per-genome rayon worker bails on cancellation (added 31/05/2026).
    /// All fields are empty Vecs; downstream code treats this as "no cells
    /// trained for this genome" and the per-genome metrics come out as
    /// the empty-memory defaults (matching how default-initialised genomes
    /// behave on day 1).
    pub fn empty() -> Self {
        Self {
            connections:   Vec::new(),
            group_info:    Vec::new(),
            dense_exports: Vec::new(),
            sparse_exports: Vec::new(),
            groups:        Vec::new(),
        }
    }

    /// Path 2 abstraction: read a single trained-memory cell at
    /// (logical_group_idx, neuron_in_group, address). Returns the raw cell
    /// value as i64 (matches GroupMemory.read so cell_to_weight just works).
    ///
    /// Used by `compute_neuron_stats_adaptive` so the IDS adaptive research
    /// variant can consume a `GenomeExport` directly instead of needing the
    /// dense `Vec<GroupMemory>` representation (which forced the legacy
    /// `train_genome_in_slot` path).
    #[inline]
    pub fn read_cell_at(&self, group_idx: usize, neuron_in_group: usize, address: u64) -> i64 {
        let (is_sparse, sub_idx, _) = &self.group_info[group_idx];
        if *is_sparse {
            self.sparse_exports[*sub_idx].lookup(neuron_in_group, address) as i64
        } else {
            let words = &self.dense_exports[*sub_idx];
            let bits = self.groups[group_idx].bits;
            let cells_per_neuron = 1usize << bits;
            let words_per_neuron = (cells_per_neuron + crate::neuron_memory::CELLS_PER_WORD - 1)
                / crate::neuron_memory::CELLS_PER_WORD;
            let addr = address as usize;
            let word_idx = addr / crate::neuron_memory::CELLS_PER_WORD;
            let cell_idx = addr % crate::neuron_memory::CELLS_PER_WORD;
            let word = words[neuron_in_group * words_per_neuron + word_idx];
            (word >> (cell_idx * crate::neuron_memory::BITS_PER_CELL))
                & crate::neuron_memory::CELL_MASK
        }
    }

    /// Path 2 abstraction: fraction of a neuron's addresses that are non-EMPTY.
    /// `neuron_in_group` is the neuron's position WITHIN the group (NOT global).
    /// `bits` is the number of address bits this neuron uses.
    ///
    /// Mirrors `GroupMemory::neuron_fill_rate` so call-sites can be migrated
    /// without changing semantics.
    pub fn neuron_fill_rate(&self, group_idx: usize, neuron_in_group: usize, bits: usize) -> f32 {
        let total_cells = 1usize << bits;
        let (is_sparse, sub_idx, _) = &self.group_info[group_idx];
        if *is_sparse {
            let s = &self.sparse_exports[*sub_idx];
            if neuron_in_group >= s.counts.len() {
                return 0.0;
            }
            let count = s.counts[neuron_in_group] as usize;
            count.min(total_cells) as f32 / total_cells.max(1) as f32
        } else {
            let words = &self.dense_exports[*sub_idx];
            let words_per_neuron = (total_cells + crate::neuron_memory::CELLS_PER_WORD - 1)
                / crate::neuron_memory::CELLS_PER_WORD;
            let start = neuron_in_group * words_per_neuron;
            let mut filled = 0u32;
            for w in 0..words_per_neuron {
                if start + w >= words.len() {
                    break;
                }
                let word = words[start + w];
                for c in 0..crate::neuron_memory::CELLS_PER_WORD {
                    let cell = (word >> (c * crate::neuron_memory::BITS_PER_CELL))
                        & crate::neuron_memory::CELL_MASK;
                    if cell != crate::neuron_memory::EMPTY {
                        filled += 1;
                    }
                }
            }
            filled.min(total_cells as u32) as f32 / total_cells.max(1) as f32
        }
    }
}

/// Calculate optimal pool and batch sizes based on memory budget
pub(crate) fn calculate_pool_size(
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
    _num_clusters: usize,
    budget_gb: f64,
    cpu_cores: usize,
) -> (usize, usize) {
    // Estimate memory per genome (use same grouping strategy as actual training)
    let groups = build_groups(bits_per_cluster, neurons_per_cluster);
    let mut bytes_per_genome = 0usize;

    for group in &groups {
        if group.bits <= SPARSE_THRESHOLD {
            // Dense: 2 bits per cell, 2^bits cells per neuron
            let cells_per_neuron = 1 << group.bits;
            let words_per_neuron = (cells_per_neuron + 30) / 31; // 31 cells per word
            bytes_per_genome += group.total_neurons() * words_per_neuron * 8;
        } else {
            // Sparse: Based on measured data from actual training
            // With 100K training examples + 5 negatives = 600K writes, but many collide
            // Measured: ~1.2K unique entries per neuron on average (8.9M / 7500 neurons)
            // Use 3K as conservative estimate to leave headroom
            // Memory per entry: key(8) + value(1) + DashMap overhead (~24 bytes)
            bytes_per_genome += group.total_neurons() * 3_000 * 32;
        }
    }

    let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let max_pool_size = (budget_bytes / bytes_per_genome).max(1);

    // Pool size cap:
    //   - Default (baseline path): cap at `cpu_cores` (rayon-bounded per-genome
    //     parallelism — more batches than cores just queues serially).
    //   - GPU batched train: kernel dispatches all genomes in one Metal call
    //     (GPU has ~1280 SIMD lanes), so the cpu_cores cap is artificial.
    //     Use a higher cap (B9_GPU_BATCH_CAP) so we can absorb 50+ genomes
    //     per dispatch when memory allows.
    // WNN_BATCH_SIZE env var still overrides everything for testing.
    let gpu_batched = gpu_batched_train_enabled();
    const B9_GPU_BATCH_CAP: usize = 50;
    let effective_cap = if gpu_batched {
        cpu_cores.max(B9_GPU_BATCH_CAP)
    } else {
        cpu_cores
    };
    let pool_size = max_pool_size.min(effective_cap).max(1);

    // Batch size = pool size (process one batch at a time)
    let batch_size = pool_size;

    (pool_size, batch_size)
}

/// Get available memory in GB (macOS specific)
pub(crate) fn get_available_memory_gb() -> f64 {
    // Try to read from sysctl
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = mem_str.trim().parse::<u64>() {
                    return bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                }
            }
        }
    }
    // Fallback: assume 64GB (M4 Max typical)
    64.0
}

pub fn compute_class_weights_with_multiplier(labels: &[i64], num_classes: usize, multiplier: f32) -> Vec<u32> {
    let mut counts = vec![0u64; num_classes];
    for &label in labels {
        let c = label as usize;
        if c < num_classes {
            counts[c] += 1;
        }
    }
    let max_count = *counts.iter().max().unwrap_or(&1);
    counts.iter().map(|&c| {
        if c == 0 { 1 } else {
            let base = (max_count / c).max(1) as f32;
            (base * multiplier).max(1.0) as u32
        }
    }).collect()
}

/// Train a genome using the given memory slot.
/// When `gpu_addresses` is Some, uses pre-computed GPU addresses instead of CPU compute_address().
/// GPU address layout: addresses[global_neuron_idx * num_train + example_idx].
/// When `parallel` is true, uses rayon par_iter for example-level parallelism.
/// Set `parallel=false` when calling from within an outer par_iter to avoid nested parallelism deadlock.
pub(crate) fn train_genome_in_slot(
    memories: &mut [GroupMemory],
    groups: &[ConfigGroup],
    original_connections: &[i64],    // Per-neuron layout (NOT group layout)
    per_neuron_bits: &[usize],       // Bits per neuron
    cluster_neuron_starts: &[usize], // First neuron idx per cluster
    neuron_conn_offsets: &[usize],   // Conn offset per neuron
    cluster_to_group: &[(usize, usize)],
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    _total_input_bits: usize,
    gpu_addresses: Option<&[u32]>,
    neuron_sample_rate: f32,
    rng_seed: u64,
    memory_mode: u8,
    class_weights: Option<&[u32]>,
    parallel: bool,
) {
    // OI orchestration: init counter buffers (when enabled), train, then commit.
    let oi = crate::neuron_memory::order_independent_training_enabled()
        && memory_mode == crate::neuron_memory::MODE_QUAD_WEIGHTED;
    if oi {
        for m in memories.iter_mut() {
            m.init_oi_counters();
        }
    }
    // Thin wrapper: full-range training with stride == num_train (existing behavior).
    train_genome_in_slot_range(
        memories, groups, original_connections, per_neuron_bits,
        cluster_neuron_starts, neuron_conn_offsets, cluster_to_group,
        train_input_bits, train_targets, train_negatives,
        num_train, num_negatives, _total_input_bits,
        gpu_addresses, 0..num_train, num_train,
        neuron_sample_rate, rng_seed, memory_mode, class_weights, parallel,
    );
    if oi {
        for m in memories.iter_mut() {
            m.commit_oi();
        }
    }
}

/// Range-aware training: writes memory cells for examples in `example_range`.
///
/// `addr_stride` is the stride between neurons in `gpu_addresses`. For the
/// non-chunked path this equals `num_train` (passed by the wrapper); for chunked
/// GPU address compute it equals the chunk length. Address indexing:
/// `gpu_addresses[global_n * addr_stride + (ex_idx - example_range.start)]`.
pub(crate) fn train_genome_in_slot_range(
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    original_connections: &[i64],
    per_neuron_bits: &[usize],
    cluster_neuron_starts: &[usize],
    neuron_conn_offsets: &[usize],
    cluster_to_group: &[(usize, usize)],
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    _num_train: usize,
    num_negatives: usize,
    _total_input_bits: usize,
    gpu_addresses: Option<&[u32]>,
    example_range: std::ops::Range<usize>,
    addr_stride: usize,
    neuron_sample_rate: f32,
    rng_seed: u64,
    memory_mode: u8,
    class_weights: Option<&[u32]>,
    parallel: bool,
) {
    let use_sampling = neuron_sample_rate < 1.0;
    let use_nudge = memory_mode != crate::neuron_memory::MODE_TERNARY;
    // OI is only meaningful for QUAD_WEIGHTED (the only mode where the existing
    // clamped nudge has order-dependence to fix).
    let use_oi = crate::neuron_memory::order_independent_training_enabled()
        && memory_mode == crate::neuron_memory::MODE_QUAD_WEIGHTED;
    let chunk_start = example_range.start;

    let train_one_example = |ex_idx: usize| {
        let input_bits = train_input_bits.packed_row(ex_idx);

        let num_clusters = cluster_to_group.len();
        // Single-cluster mode: always target cluster 0, nudge direction = label
        // Multi-cluster mode: target cluster = label, always nudge TRUE
        let true_cluster = if num_clusters == 1 { 0 } else { train_targets[ex_idx] as usize };
        let nudge_direction = if num_clusters == 1 {
            train_targets[ex_idx] == 1  // Attack=TRUE, Normal=FALSE
        } else {
            true  // Always positive for target cluster in multi-cluster mode
        };

        // Train positive example
        {
            let (group_idx, local_cluster) = cluster_to_group[true_cluster];
            let group = &groups[group_idx];
            let memory = &memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

            for n in 0..actual_neurons {
                let global_n = cluster_neuron_starts[true_cluster] + n;

                // Per-(neuron, example) deterministic sampling
                // Uses hash of (rng_seed, neuron_idx, example_idx) for parallel-safe decisions
                if use_sampling {
                    let mut rng = (rng_seed as u32)
                        .wrapping_add(global_n as u32 * 1000003)
                        .wrapping_add(ex_idx as u32 * 2654435761);
                    if rng == 0 { rng = 1; }
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    if (rng >> 8) as f32 / 16777216.0 >= neuron_sample_rate {
                        continue;
                    }
                }

                let address = if let Some(addrs) = gpu_addresses {
                    addrs[global_n * addr_stride + (ex_idx - chunk_start)] as usize
                } else {
                    let n_bits = per_neuron_bits[global_n];
                    let conn_start = neuron_conn_offsets[global_n];
                    crate::neuron_memory::compute_address_packed_bytes(input_bits, &original_connections[conn_start..], n_bits)
                };
                // Weight by original label for class balancing
                let weight_idx = train_targets[ex_idx] as usize;
                let repeats = class_weights.map_or(1u32, |w| w[weight_idx]);
                if use_oi {
                    // OI: one accumulating call per example with weight = class_weight.
                    // Semantically counts this as a single observation (obs += 1)
                    // regardless of weight, while the net moves by ±weight.
                    memory.nudge_oi(neuron_base + n, address, nudge_direction, repeats);
                } else if use_nudge {
                    for _ in 0..repeats {
                        memory.nudge(neuron_base + n, address, nudge_direction);
                    }
                } else {
                    let value = if nudge_direction { TRUE } else { FALSE };
                    memory.write(neuron_base + n, address, value, false);
                }
            }
        }

        // Train negative examples.
        //
        // Single-cluster (binary IDS) encodes the FALSE direction via
        // train_targets[ex_idx] == 0 + nudge_direction in the positive branch
        // above; the negative loop is multi-class only (K > 1 with explicit
        // per-example negative cluster IDs in train_negatives). Skip when
        // there's only one cluster — defense against callers that mis-form
        // the train_negatives buffer (e.g., the FPGA export wrapper bug at
        // lib.rs:7662-7667 that pre-dated this guard, where row-indices got
        // mis-interpreted as cluster-ids and panicked at the indexing below).
        if cluster_to_group.len() == 1 {
            // Inside a rayon closure (per-example), `return` exits this
            // closure invocation cleanly — equivalent to `continue` in a
            // regular for-loop.
            return;
        }
        let neg_start = ex_idx * num_negatives;
        for k in 0..num_negatives {
            let false_cluster = train_negatives[neg_start + k] as usize;
            if false_cluster == true_cluster {
                continue;
            }

            let (group_idx, local_cluster) = cluster_to_group[false_cluster];
            let group = &groups[group_idx];
            let memory = &memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

            for n in 0..actual_neurons {
                let global_n = cluster_neuron_starts[false_cluster] + n;

                // Same per-(neuron, example) sampling for negative examples
                if use_sampling {
                    let mut rng = (rng_seed as u32)
                        .wrapping_add(global_n as u32 * 1000003)
                        .wrapping_add(ex_idx as u32 * 2654435761);
                    if rng == 0 { rng = 1; }
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    if (rng >> 8) as f32 / 16777216.0 >= neuron_sample_rate {
                        continue;
                    }
                }

                let address = if let Some(addrs) = gpu_addresses {
                    addrs[global_n * addr_stride + (ex_idx - chunk_start)] as usize
                } else {
                    let n_bits = per_neuron_bits[global_n];
                    let conn_start = neuron_conn_offsets[global_n];
                    crate::neuron_memory::compute_address_packed_bytes(input_bits, &original_connections[conn_start..], n_bits)
                };
                // For negative nudges, weight by the TRUE class of the example
                // (the example "belongs to" true_cluster, so its weight applies)
                let repeats = class_weights.map_or(1u32, |w| w[true_cluster]);
                if use_oi {
                    memory.nudge_oi(neuron_base + n, address, false, repeats);
                } else if use_nudge {
                    for _ in 0..repeats {
                        memory.nudge(neuron_base + n, address, false);
                    }
                } else {
                    memory.write(neuron_base + n, address, FALSE, false);
                }
            }
        }
    };

    let range_len = example_range.end - example_range.start;
    if parallel {
        let chunk_size = 10_000.max(range_len / 20);
        example_range.clone().into_par_iter()
            .with_min_len(chunk_size)
            .for_each(|ex_idx| train_one_example(ex_idx));
    } else {
        for ex_idx in example_range.clone() {
            train_one_example(ex_idx);
        }
    }
}

/// Export trained memory to GPU-compatible format
pub(crate) fn export_genome_for_gpu(
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    connections_flat: &[i64],
) -> GenomeExport {
    let mut dense_exports = Vec::new();
    let mut sparse_exports = Vec::new();
    let mut group_info = Vec::new();

    for (group_idx, (group, memory)) in groups.iter().zip(memories.iter()).enumerate() {
        let is_sparse = memory.is_sparse();
        group_info.push((is_sparse, group_idx, group.cluster_ids.clone()));

        if is_sparse {
            if let Some(export) = memory.export_for_gpu_sparse() {
                sparse_exports.push(export);
            } else {
                // Fallback: empty export
                sparse_exports.push(SparseGpuExport {
                    keys: vec![],
                    values: vec![],
                    offsets: vec![0; group.total_neurons()],
                    counts: vec![0; group.total_neurons()],
                    num_neurons: group.total_neurons(),
                });
            }
        } else {
            if let Some(words) = memory.export_for_metal() {
                dense_exports.push(words);
            } else {
                dense_exports.push(vec![]);
            }
        }
    }

    GenomeExport {
        connections: connections_flat.to_vec(),
        group_info,
        dense_exports,
        sparse_exports,
        groups: groups.to_vec(),
    }
}

// Thread-local cache for GPU buffers to avoid expensive 10GB buffer allocation per evaluation
// The scores buffer is ~10GB (50K examples × 50K clusters × 4 bytes), so reusing it is critical.
// The cache includes the reset generation to invalidate on Metal reset.
#[cfg(target_os = "macos")]
thread_local! {
    // (reset_gen, num_eval, num_clusters, buffer)
    static CACHED_SCORES_BUFFER: std::cell::RefCell<Option<(u64, usize, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
    // (reset_gen, size, buffer)
    static CACHED_INPUT_BUFFER: std::cell::RefCell<Option<(u64, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
}

/// Evaluate a genome export using CPU+GPU hybrid
/// Returns (cross_entropy, accuracy)
/// Compute per-example, per-cluster scores from a trained genome export.
///
/// Shared by `evaluate_genome_hybrid` (CE/accuracy) and `predict_genome_hybrid` (argmax).
/// Tries GPU evaluation (sparse + dense) for each group, falling back to CPU binary search.
pub(crate) fn compute_per_example_scores(
    export: &GenomeExport,
    eval_input_bits: &crate::packed_bits::PackedBits,
    packed_eval: &[u64],
    words_per_example: usize,
    num_eval: usize,
    num_clusters: usize,
    _total_input_bits: usize,
    empty_value: f32,
    memory_mode: u8,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&crate::metal_ramlm::MetalSparseEvaluator>,
) -> Vec<Vec<f64>> {
    let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];

    let mut dense_idx = 0usize;
    let mut sparse_idx = 0usize;

    for (is_sparse, group_idx, cluster_ids) in &export.group_info {
        let group = &export.groups[*group_idx];

        if *is_sparse {
            let sparse_export = &export.sparse_exports[sparse_idx];
            sparse_idx += 1;

            let gpu_success = if let Some(sparse_eval) = sparse_metal {
                match evaluate_group_sparse_gpu(
                    sparse_eval,
                    packed_eval,
                    &export.connections,
                    sparse_export,
                    group,
                    num_eval,
                    words_per_example,
                    memory_mode,
                    empty_value,
                ) {
                    Ok(group_scores) => {
                        let num_group_clusters = group.cluster_count();
                        all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                            for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * num_group_clusters + local_cluster;
                                scores[cluster_id] = group_scores[score_idx] as f64;
                            }
                        });
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };

            if !gpu_success {
                // CPU fallback using binary search
                all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                    let input_bits = eval_input_bits.packed_row(ex_idx);

                    for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                        let actual_neurons = if let Some(ref an) = group.actual_neurons {
                            an[local_cluster] as usize
                        } else {
                            group.neurons
                        };

                        let neuron_base = local_cluster * group.neurons;
                        let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                        let mut sum = 0.0f32;
                        for n in 0..actual_neurons {
                            let conn_start = conn_base + n * group.bits;
                            let address = crate::neuron_memory::compute_address_packed_bytes(input_bits, &export.connections[conn_start..], group.bits);
                            let cell = sparse_export.lookup(neuron_base + n, address as u64);
                            sum += cell_to_weight(cell as i64, memory_mode, empty_value);
                        }

                        scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                    }
                });
            }
        } else {
            let dense_words = &export.dense_exports[dense_idx];
            dense_idx += 1;

            let gpu_success = if let Some(metal_eval) = metal {
                match evaluate_group_metal(
                    metal_eval,
                    packed_eval,
                    &export.connections,
                    dense_words,
                    group,
                    num_eval,
                    words_per_example,
                    memory_mode,
                    empty_value,
                ) {
                    Ok(group_scores) => {
                        let num_group_clusters = group.cluster_count();
                        all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                            for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * num_group_clusters + local_cluster;
                                scores[cluster_id] = group_scores[score_idx] as f64;
                            }
                        });
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };

            if !gpu_success {
                // CPU fallback for dense groups
                all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                    let input_bits = eval_input_bits.packed_row(ex_idx);

                    for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                        let actual_neurons = if let Some(ref an) = group.actual_neurons {
                            an[local_cluster] as usize
                        } else {
                            group.neurons
                        };

                        let neuron_base = local_cluster * group.neurons;
                        let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                        let mut sum = 0.0f32;
                        for n in 0..actual_neurons {
                            let conn_start = conn_base + n * group.bits;
                            let address = crate::neuron_memory::compute_address_packed_bytes(input_bits, &export.connections[conn_start..], group.bits);
                            let cell = read_cell(dense_words, neuron_base + n, address, group.words_per_neuron);
                            sum += cell_to_weight(cell, memory_mode, empty_value);
                        }

                        scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                    }
                });
            }
        }
    }

    all_scores
}

pub fn evaluate_genome_hybrid(
    export: &GenomeExport,
    eval_input_bits: &crate::packed_bits::PackedBits,
    eval_targets: &[i64],
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&crate::metal_ramlm::MetalSparseEvaluator>,
    override_threshold: Option<f64>,
) -> (f64, f64, f64, f64, f64) {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;
    let epsilon = 1e-10f64;

    // Detailed timing (enabled via WNN_GROUP_TIMING env var)
    let timing_enabled = std::env::var("WNN_GROUP_TIMING").is_ok();
    let eval_start = std::time::Instant::now();

    // Pack eval input bits to u64 for GPU (pack once, reuse for all GPU paths)
    let (packed_eval, words_per_example) = crate::neuron_memory::pack_packed_to_u64(eval_input_bits);
    let _ = total_input_bits; // kept for ABI; eval_input_bits.total_bits() is authoritative

    // SINGLE-CLUSTER BINARY DISCRIMINATOR: use override or find threshold, binary cross-entropy
    if num_clusters == 1 {
        let all_scores = compute_per_example_scores(
            export, eval_input_bits, &packed_eval, words_per_example,
            num_eval, num_clusters, total_input_bits, empty_value,
            memory_mode, metal, sparse_metal,
        );

        // BCE loss (threshold-independent)
        let mut total_ce = 0.0f64;
        for ex_idx in 0..num_eval {
            let s = (all_scores[ex_idx][0]).clamp(epsilon, 1.0 - epsilon);
            let y = eval_targets[ex_idx] as f64;
            total_ce += -(y * s.ln() + (1.0 - y) * (1.0 - s).ln());
        }
        let ce = total_ce / num_eval as f64;

        // Use override threshold (from training calibration) or find on eval data (fallback)
        let flat_scores: Vec<f64> = all_scores.iter().map(|s| s[0]).collect();
        let threshold = override_threshold.unwrap_or_else(|| {
            let (t, _f1, _fpr) = find_optimal_threshold_auto(&flat_scores, eval_targets, settings.fitness_weights);
            t
        });

        // Apply threshold for predictions
        let mut correct = 0u64;
        let mut predictions = Vec::with_capacity(num_eval);
        for ex_idx in 0..num_eval {
            let pred = if all_scores[ex_idx][0] >= threshold { 1u32 } else { 0u32 };
            predictions.push(pred);
            if pred as i64 == eval_targets[ex_idx] { correct += 1; }
        }
        let acc = correct as f64 / num_eval as f64;
        let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, 2, settings.normal_class);
        return (ce, acc, f1, fpr, threshold);
    }

    // FAST PATH: If single sparse group covering all clusters, use direct CE computation
    // This avoids the 10GB GPU→CPU transfer by computing CE on GPU
    #[cfg(target_os = "macos")]
    if export.group_info.len() == 1 && export.sparse_exports.len() == 1 {
        let (is_sparse, group_idx, cluster_ids) = &export.group_info[0];
        if *is_sparse && cluster_ids.len() == num_clusters {
            // Check if clusters are contiguous 0..num_clusters (identity mapping)
            let is_contiguous = cluster_ids.iter().enumerate().all(|(i, &c)| c == i);
            if is_contiguous {
                // Use MetalSparseCEEvaluator for direct CE computation
                static CE_EVALUATOR: std::sync::OnceLock<Option<crate::metal_ramlm::MetalSparseCEEvaluator>> = std::sync::OnceLock::new();
                let ce_eval = CE_EVALUATOR.get_or_init(|| {
                    crate::metal_ramlm::MetalSparseCEEvaluator::new().ok()
                });

                if let Some(evaluator) = ce_eval {
                    let group = &export.groups[*group_idx];
                    let sparse_export = &export.sparse_exports[0];

                    let call_start = std::time::Instant::now();
                    let result = evaluator.compute_ce(
                        &packed_eval,
                        &export.connections,
                        &sparse_export.keys,
                        &sparse_export.values,
                        &sparse_export.offsets,
                        &sparse_export.counts,
                        eval_targets,
                        num_eval,
                        words_per_example,
                        group.neurons * num_clusters,  // total neurons
                        group.bits,
                        group.neurons,
                        num_clusters,
                        empty_value,
                        memory_mode,
                    );

                    if let Ok((ce, acc, predictions)) = result {
                        let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, settings.normal_class);
                        if timing_enabled {
                            let elapsed = call_start.elapsed().as_millis();
                            let total_ms = eval_start.elapsed().as_millis();
                            eprintln!(
                                "[EVAL_HYBRID] FAST_PATH total={}ms gpu_ce={}ms (no scatter!)",
                                total_ms, elapsed
                            );
                        }
                        return (ce, acc, f1, fpr, 0.5);
                    }
                    // Fall through to standard path if CE evaluator fails
                }
            }
        }
    }

    // FULL GPU PATH: Write scores directly to shared GPU buffer, compute CE on GPU
    // This avoids the GPU→CPU→GPU round-trip that was slowing down tiered evaluation
    // Full GPU path enabled by default (3x faster). Disable with WNN_GPU_CE=0
    #[cfg(target_os = "macos")]
    {
    let use_full_gpu = std::env::var("WNN_GPU_CE").map(|v| v != "0").unwrap_or(true);

    if use_full_gpu {
        if let Some(group_eval) = get_group_evaluator() {
            let gpu_start = std::time::Instant::now();

            // Get current reset generation to detect when Metal evaluators were reset
            let current_reset_gen = RESET_GENERATION.load(Ordering::SeqCst);

            // Phase timing for detailed analysis (enabled via WNN_PHASE_TIMING env var)
            let phase_timing = std::env::var("WNN_PHASE_TIMING").is_ok();
            let phase_start = std::time::Instant::now();

            // Get or create cached scores buffer (avoids expensive 10GB allocation per eval)
            // The scores buffer is zeroed efficiently using memset instead of creating a new Vec
            // Invalidate cache if Metal was reset (reset_gen changed)
            let scores_buffer = CACHED_SCORES_BUFFER.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some((cached_gen, cached_eval, cached_clusters, ref buffer)) = *cache {
                    if cached_gen == current_reset_gen && cached_eval == num_eval && cached_clusters == num_clusters {
                        // Reuse existing buffer - just zero it
                        group_eval.zero_scores_buffer(buffer, num_eval, num_clusters);
                        return buffer.clone();
                    }
                }
                // Create new buffer and cache it (invalidate on reset_gen mismatch)
                let buffer = group_eval.create_scores_buffer(num_eval, num_clusters);
                *cache = Some((current_reset_gen, num_eval, num_clusters, buffer.clone()));
                buffer
            });

            let zero_time_ms = if phase_timing { phase_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };
            let phase_start = std::time::Instant::now();

            // Get or create cached input buffer (update contents efficiently)
            // Invalidate cache if Metal was reset (reset_gen changed)
            // Uses packed u64 input for GPU
            let input_buffer = CACHED_INPUT_BUFFER.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some((cached_gen, cached_size, ref buffer)) = *cache {
                    if cached_gen == current_reset_gen && cached_size == packed_eval.len() {
                        // Reuse existing buffer - update contents
                        group_eval.update_input_buffer(buffer, &packed_eval);
                        return buffer.clone();
                    }
                }
                // Create new buffer and cache it (invalidate on reset_gen mismatch)
                let buffer = group_eval.create_input_buffer(&packed_eval);
                *cache = Some((current_reset_gen, packed_eval.len(), buffer.clone()));
                buffer
            });

            let input_time_ms = if phase_timing { phase_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };
            let _phase_start = std::time::Instant::now();

            let mut dense_idx: usize;
            let mut sparse_idx = 0usize;
            let all_groups_success = true;
            let mut sparse_time_ms = 0.0f64;
            let mut dense_time_ms = 0.0f64;
            let sparse_call_count;

            // Collect all sparse groups for batched evaluation (single command buffer)
            // This eliminates ~0.5ms overhead per group from separate commit+wait cycles
            let mut sparse_groups: Vec<crate::metal_ramlm::SparseGroupData> = Vec::new();
            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                if *is_sparse {
                    let group = &export.groups[*group_idx];
                    let sparse_export = &export.sparse_exports[sparse_idx];
                    sparse_idx += 1;

                    sparse_groups.push(crate::metal_ramlm::SparseGroupData {
                        connections: &export.connections[group.conn_offset..group.conn_offset + group.conn_size()],
                        keys: &sparse_export.keys,
                        values: &sparse_export.values,
                        offsets: &sparse_export.offsets,
                        counts: &sparse_export.counts,
                        cluster_ids,
                        bits_per_neuron: group.bits,
                        neurons_per_cluster: group.neurons,
                        actual_neurons_per_cluster: group.actual_neurons.as_deref(),
                    });
                }
            }

            // Evaluate all sparse groups in a single batched call
            sparse_call_count = sparse_groups.len();
            if !sparse_groups.is_empty() {
                let sparse_start = std::time::Instant::now();
                group_eval.eval_sparse_groups_batched(
                    &input_buffer,
                    &scores_buffer,
                    &sparse_groups,
                    num_eval,
                    words_per_example,
                    num_clusters,
                    empty_value,
                    memory_mode
                );
                if phase_timing {
                    sparse_time_ms = sparse_start.elapsed().as_micros() as f64 / 1000.0;
                }
            }

            // Evaluate dense groups individually (uses DENSE_BUFFER_CACHE)
            dense_idx = 0;
            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                if !*is_sparse {
                    let group = &export.groups[*group_idx];
                    let dense_words = &export.dense_exports[dense_idx];
                    dense_idx += 1;

                    let dense_start = std::time::Instant::now();
                    group_eval.eval_dense_to_buffer(
                        &input_buffer,
                        &scores_buffer,
                        &export.connections[group.conn_offset..group.conn_offset + group.conn_size()],
                        dense_words,
                        cluster_ids,
                        num_eval,
                        words_per_example,
                        group.bits,
                        group.neurons,
                        num_clusters,
                        group.words_per_neuron,
                        empty_value,
                        memory_mode
                    );

                    if phase_timing {
                        dense_time_ms += dense_start.elapsed().as_micros() as f64 / 1000.0;
                    }
                }
            }

            let ce_start = std::time::Instant::now();

            if all_groups_success {
                // Compute CE directly from GPU buffer
                let result = group_eval.compute_ce_from_buffer(
                    &scores_buffer,
                    eval_targets,
                    num_eval,
                    num_clusters,
                );

                let ce_time_ms = if phase_timing { ce_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };

                if let Ok((ce, acc, predictions)) = result {
                    let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, settings.normal_class);
                    if timing_enabled {
                        let elapsed = gpu_start.elapsed().as_millis();
                        if phase_timing {
                            eprintln!(
                                "[EVAL_PHASE] zero={:.1}ms input={:.1}ms sparse={:.1}ms({} calls) dense={:.1}ms ce={:.1}ms total={}ms",
                                zero_time_ms, input_time_ms, sparse_time_ms, sparse_call_count, dense_time_ms, ce_time_ms, elapsed
                            );
                        } else {
                            eprintln!(
                                "[EVAL_HYBRID] FULL_GPU_PATH total={}ms (no CPU scatter!)",
                                elapsed
                            );
                        }
                    }
                    return (ce, acc, f1, fpr, 0.5);
                }
            }
            // Fall through to CPU path if full GPU fails
        }
    }
    } // cfg(target_os = "macos")

    // CPU FALLBACK PATH: Compute per-example scores using shared function
    let all_scores = compute_per_example_scores(
        export, eval_input_bits, &packed_eval, words_per_example,
        num_eval, num_clusters, total_input_bits, empty_value,
        memory_mode, metal, sparse_metal,
    );

    if timing_enabled {
        let total_ms = eval_start.elapsed().as_millis();
        eprintln!("[EVAL_HYBRID] CPU_FALLBACK total={}ms", total_ms);
    }

    // Compute CE, accuracy, and predictions from pre-computed scores
    let predictions: Vec<u32> = all_scores.par_iter().map(|scores| {
        scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }).collect();

    let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
        let target_idx = eval_targets[ex_idx] as usize;
        let predicted = predictions[ex_idx] as usize;
        let correct: u64 = if predicted == target_idx { 1 } else { 0 };

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let target_prob = exp_scores[target_idx] / sum_exp;
        let ce = -(target_prob + epsilon).ln();

        (ce, correct)
    }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

    let avg_ce = total_ce / num_eval as f64;
    let accuracy = total_correct as f64 / num_eval as f64;
    let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, settings.normal_class);

    (avg_ce, accuracy, f1, fpr, 0.5)
}

/// Predict per-example class indices for a single trained genome.
///
/// Delegates to `compute_per_example_scores` for the shared score computation,
/// then returns argmax predictions instead of CE/accuracy.
/// Used by the bitwise ECOC classifier to combine per-bit predictions.
pub fn predict_genome_hybrid(
    export: &GenomeExport,
    eval_input_bits: &crate::packed_bits::PackedBits,
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&crate::metal_ramlm::MetalSparseEvaluator>,
    single_cluster_threshold: Option<f64>,
) -> Vec<i64> {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;

    let (packed_eval, words_per_example) = crate::neuron_memory::pack_packed_to_u64(eval_input_bits);

    let all_scores = compute_per_example_scores(
        export, eval_input_bits, &packed_eval, words_per_example,
        num_eval, num_clusters, total_input_bits, empty_value,
        memory_mode, metal, sparse_metal,
    );

    // Single-cluster binary discriminator: use provided threshold or default 0.5
    if num_clusters == 1 {
        let threshold = single_cluster_threshold.unwrap_or(0.5);
        return all_scores.par_iter().map(|scores| {
            if scores[0] >= threshold { 1i64 } else { 0i64 }
        }).collect();
    }

    // Return argmax predictions (not CE/accuracy)
    all_scores.par_iter().map(|scores| {
        scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap_or(0)
    }).collect()
}

/// Path 2 adapter: train a single genome via Option B (MarkerHashTable) and
/// return a `GenomeExport`. Drop-in replacement for the dense-path sequence
/// `try_gpu_addresses_adaptive + train_genome_in_slot + export_genome_for_gpu`
/// — bit-exact parity verified for b∈{16,32,48,64,96} × n∈{50,100,200,250}
/// across 9 shape configurations (tests/path2_parity.py, all pass with 0
/// drift on CE/Acc/F1/FPR).
///
/// Benefits over the dense path:
///   - No u32 truncation guard (Option B uses u64 keys natively).
///   - No separate compute_addresses GPU dispatch (train kernel fuses it).
///   - Lower GPU↔CPU traffic (no intermediate address array).
///
/// Callers should switch from the dense sequence to a single call to this
/// function during Path 2 phase 2/3 migrations.
#[allow(clippy::too_many_arguments)]
pub fn train_single_via_marker(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> Result<GenomeExport, String> {
    #[cfg(target_os = "macos")]
    {
        let mut exports = crate::marker_train::batched_train_offspring(
            genomes_bits_flat,
            genomes_neurons_flat,
            genomes_connections_flat,
            1, // single genome
            num_clusters,
            train_input_bits,
            train_targets,
            train_negatives,
            num_train,
            num_negatives,
            total_input_bits,
            empty_value,
            neuron_sample_rate,
            rng_seed,
            class_weights,
        )?;
        exports
            .pop()
            .ok_or_else(|| "batched_train_offspring returned empty Vec for single genome".to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("train_single_via_marker requires macOS / Metal".to_string())
    }
}

/// Train a single genome and return per-example predicted class indices.
///
/// Combines the training path from `evaluate_genomes_parallel_hybrid` (single genome)
/// with `predict_genome_hybrid` for per-example predictions.
#[allow(clippy::too_many_arguments)]
pub fn train_and_predict_single(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> Vec<i64> {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;

    // Extract single genome config
    let neurons_per_cluster = genomes_neurons_flat;
    let per_neuron_bits = genomes_bits_flat;
    let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);
    let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

    let original_connections = genomes_connections_flat.to_vec();

    // Pack training input — still needed for compute_per_example_scores below
    // (single-cluster calibration), even when the train step uses Option B.
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_packed_to_u64(train_input_bits);

    // Path 2 migration (16/05/2026, branch path2-marker-unified): train via
    // Option B (MarkerHashTable / batched_train_offspring) instead of the
    // dense compute_addresses + train_genome_in_slot sequence. Bit-exact
    // parity verified at b∈{16,32,48,64,96} × n∈{50,100,200,250} in
    // tests/path2_parity.py. Falls back to the dense path on error so we
    // keep a safety net during migration.
    let export = match train_single_via_marker(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_clusters,
        train_input_bits,
        train_targets,
        train_negatives,
        num_train,
        num_negatives,
        total_input_bits,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        class_weights,
    ) {
        Ok(e) => e,
        Err(reason) => {
            eprintln!("[PATH2_FALLBACK] train_and_predict_single → dense: {}", reason);
            // Build state needed for the dense fallback path
            let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
            for (group_idx, group) in groups.iter().enumerate() {
                for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                    cluster_to_group[cluster_id] = (group_idx, local_idx);
                }
            }
            let mut memories: Vec<GroupMemory> = groups.iter()
                .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                .collect();

            let gpu_addresses = try_gpu_addresses_adaptive(
                &packed_train_input,
                words_per_example,
                per_neuron_bits,
                &neuron_conn_offsets,
                &original_connections,
                num_train,
            );
            train_genome_in_slot(
                &mut memories,
                &groups,
                &original_connections,
                per_neuron_bits,
                &cluster_neuron_starts,
                &neuron_conn_offsets,
                &cluster_to_group,
                train_input_bits,
                train_targets,
                train_negatives,
                num_train,
                num_negatives,
                total_input_bits,
                gpu_addresses.as_deref(),
                neuron_sample_rate,
                rng_seed,
                memory_mode,
                class_weights,
                true, // parallel: standalone call, safe to use par_iter
            );
            let gpu_connections = reorganize_connections_for_gpu(
                &original_connections,
                per_neuron_bits,
                neurons_per_cluster,
                &groups,
            );
            export_genome_for_gpu(&memories, &groups, &gpu_connections)
        }
    };

    // Get Metal evaluators for GPU prediction
    let metal = get_metal_evaluator();
    let sparse_metal = get_sparse_metal_evaluator();

    // Single-cluster: calibrate threshold on training data before predicting eval
    let threshold = if num_clusters == 1 {
        let train_scores = compute_per_example_scores(
            &export, train_input_bits, &packed_train_input, words_per_example,
            num_train, num_clusters, total_input_bits, empty_value,
            memory_mode, metal.as_deref(), sparse_metal.as_deref(),
        );
        let flat_scores: Vec<f64> = train_scores.iter().map(|s| s[0]).collect();
        let (t, f1, fpr) = find_optimal_threshold_auto(&flat_scores, train_targets, settings.fitness_weights);
        eprintln!(
            "[SINGLE_CLUSTER] Train calibration: threshold={:.4}, train_f1={:.4}, train_fpr={:.4}",
            t, f1, fpr
        );
        Some(t)
    } else {
        None
    };

    predict_genome_hybrid(
        &export,
        eval_input_bits,
        num_eval,
        num_clusters,
        total_input_bits,
        settings,
        metal.as_deref(),
        sparse_metal.as_deref(),
        threshold,
    )
}

/// Train a single genome and return per-example RAW SCORES on eval set.
///
/// Like `train_and_predict_single` but returns `Vec<f64>` scores instead of
/// thresholded class predictions. Used for Platt scaling calibration.
#[allow(clippy::too_many_arguments)]
pub fn train_and_score_single(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> Vec<f64> {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;

    let neurons_per_cluster = genomes_neurons_flat;
    let per_neuron_bits = genomes_bits_flat;
    let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);
    let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

    let original_connections = genomes_connections_flat.to_vec();

    // Path 2 migration — same pattern as train_and_predict_single above.
    let export = match train_single_via_marker(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_clusters,
        train_input_bits,
        train_targets,
        train_negatives,
        num_train,
        num_negatives,
        total_input_bits,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        class_weights,
    ) {
        Ok(e) => e,
        Err(reason) => {
            eprintln!("[PATH2_FALLBACK] train_and_score_single → dense: {}", reason);
            let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
            for (group_idx, group) in groups.iter().enumerate() {
                for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                    cluster_to_group[cluster_id] = (group_idx, local_idx);
                }
            }
            let mut memories: Vec<GroupMemory> = groups.iter()
                .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                .collect();
            let (packed_train_input, words_per_example) =
                crate::neuron_memory::pack_packed_to_u64(train_input_bits);
            let gpu_addresses = try_gpu_addresses_adaptive(
                &packed_train_input, words_per_example,
                per_neuron_bits, &neuron_conn_offsets,
                &original_connections, num_train,
            );
            train_genome_in_slot(
                &mut memories, &groups, &original_connections,
                per_neuron_bits, &cluster_neuron_starts, &neuron_conn_offsets,
                &cluster_to_group,
                train_input_bits, train_targets, train_negatives,
                num_train, num_negatives, total_input_bits,
                gpu_addresses.as_deref(),
                neuron_sample_rate, rng_seed, memory_mode, class_weights,
                true,
            );
            let gpu_connections = reorganize_connections_for_gpu(
                &original_connections, per_neuron_bits, neurons_per_cluster, &groups,
            );
            export_genome_for_gpu(&memories, &groups, &gpu_connections)
        }
    };

    let metal = get_metal_evaluator();
    let sparse_metal = get_sparse_metal_evaluator();

    let (packed_eval, eval_words) = crate::neuron_memory::pack_packed_to_u64(eval_input_bits);

    let all_scores = compute_per_example_scores(
        &export, eval_input_bits, &packed_eval, eval_words,
        num_eval, num_clusters, total_input_bits, empty_value,
        memory_mode, metal.as_deref(), sparse_metal.as_deref(),
    );

    // Return raw score for cluster 0 (single-cluster mode)
    all_scores.iter().map(|scores| scores[0]).collect()
}

/// Train a single genome ONCE and return raw scores for BOTH eval and train sets.
///
/// Equivalent to calling `train_and_score_single` twice (once with eval, once with
/// train_input_bits as the eval set), but trains the memory only once. Used by the
/// IDS validation phase to feed multiple thresholding strategies (train_cal,
/// fixed_05, val_cal/oracle, platt, beta, empirical, empirical_cumulative) without
/// retraining the genome 7+ times.
///
/// Returns (eval_scores, train_scores) — both Vec<f64> of length num_eval and
/// num_train respectively. Single-cluster (binary IDS) mode only.
#[allow(clippy::too_many_arguments)]
pub fn train_and_score_eval_and_train(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> (Vec<f64>, Vec<f64>) {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;

    let neurons_per_cluster = genomes_neurons_flat;
    let per_neuron_bits = genomes_bits_flat;
    let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);
    let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

    let original_connections = genomes_connections_flat.to_vec();

    // Pack train input once — still needed for both compute_per_example_scores
    // calls below (train + eval scoring), regardless of train path.
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_packed_to_u64(train_input_bits);

    // Path 2 migration — same pattern as train_and_predict_single.
    let export = match train_single_via_marker(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_clusters,
        train_input_bits,
        train_targets,
        train_negatives,
        num_train,
        num_negatives,
        total_input_bits,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        class_weights,
    ) {
        Ok(e) => e,
        Err(reason) => {
            eprintln!("[PATH2_FALLBACK] train_and_score_eval_and_train → dense: {}", reason);
            let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
            for (group_idx, group) in groups.iter().enumerate() {
                for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                    cluster_to_group[cluster_id] = (group_idx, local_idx);
                }
            }
            let mut memories: Vec<GroupMemory> = groups.iter()
                .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                .collect();
            let gpu_addresses = try_gpu_addresses_adaptive(
                &packed_train_input, words_per_example,
                per_neuron_bits, &neuron_conn_offsets,
                &original_connections, num_train,
            );
            train_genome_in_slot(
                &mut memories, &groups, &original_connections,
                per_neuron_bits, &cluster_neuron_starts, &neuron_conn_offsets,
                &cluster_to_group,
                train_input_bits, train_targets, train_negatives,
                num_train, num_negatives, total_input_bits,
                gpu_addresses.as_deref(),
                neuron_sample_rate, rng_seed, memory_mode, class_weights,
                true,
            );
            let gpu_connections = reorganize_connections_for_gpu(
                &original_connections, per_neuron_bits, neurons_per_cluster, &groups,
            );
            export_genome_for_gpu(&memories, &groups, &gpu_connections)
        }
    };

    let metal = get_metal_evaluator();
    let sparse_metal = get_sparse_metal_evaluator();

    // Score eval set
    let (packed_eval, eval_words) = crate::neuron_memory::pack_packed_to_u64(eval_input_bits);
    let eval_all_scores = compute_per_example_scores(
        &export, eval_input_bits, &packed_eval, eval_words,
        num_eval, num_clusters, total_input_bits, empty_value,
        memory_mode, metal.as_deref(), sparse_metal.as_deref(),
    );
    let eval_scores: Vec<f64> = eval_all_scores.iter().map(|s| s[0]).collect();

    // Score train set (reuses already-packed train input)
    let train_all_scores = compute_per_example_scores(
        &export, train_input_bits, &packed_train_input, words_per_example,
        num_train, num_clusters, total_input_bits, empty_value,
        memory_mode, metal.as_deref(), sparse_metal.as_deref(),
    );
    let train_scores: Vec<f64> = train_all_scores.iter().map(|s| s[0]).collect();

    (eval_scores, train_scores)
}

/// Compute (CE, accuracy, F1-macro, FPR) for a single-cluster binary classifier
/// from raw scores at a given threshold.
///
/// CE is binary cross-entropy (threshold-independent); accuracy/F1/FPR depend
/// on `threshold`. `normal_class` is 0 by default (set to 1 when flip_labels is
/// active, so FPR always measures false alarms on benign traffic).
pub fn compute_binary_metrics_at_threshold(
    scores: &[f64],
    targets: &[i64],
    threshold: f64,
    normal_class: usize,
) -> (f64, f64, f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let epsilon = 1e-10f64;

    // Binary cross-entropy (independent of threshold)
    let mut total_ce = 0.0f64;
    for i in 0..n {
        let s = scores[i].clamp(epsilon, 1.0 - epsilon);
        let y = targets[i] as f64;
        total_ce += -(y * s.ln() + (1.0 - y) * (1.0 - s).ln());
    }
    let ce = total_ce / n as f64;

    // Predictions + accuracy
    let mut correct = 0u64;
    let mut predictions: Vec<u32> = Vec::with_capacity(n);
    for i in 0..n {
        let pred = if scores[i] >= threshold { 1u32 } else { 0u32 };
        predictions.push(pred);
        if pred as i64 == targets[i] {
            correct += 1;
        }
    }
    let acc = correct as f64 / n as f64;

    let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, targets, 2, normal_class);
    (ce, acc, f1, fpr)
}

/// Evaluate genomes in PARALLEL with CPU+GPU HYBRID evaluation and PIPELINING.
///
/// Strategy:
/// 1. Parallel genome training on CPU (batch of genomes at once)
/// 2. Export to GPU-compatible format
/// 3. GPU batch evaluation (while CPU trains next batch)
/// 4. CPU+GPU hybrid: GPU evaluates, CPU assists with fallback
///
/// Performance benefits:
/// - Parallel training: N genomes trained simultaneously (vs sequential)
/// - GPU acceleration: Both dense and sparse groups on GPU
/// - Persistent worker: Eval thread stays alive across calls (eliminates spawn overhead)
/// - Pipelining: CPU trains batch N+1 while GPU evaluates batch N
///
/// Returns: Vec of (cross_entropy, accuracy) tuples - one per genome
pub fn evaluate_genomes_parallel_hybrid(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> Vec<(f64, f64, f64, f64, f64, u32)> {
    evaluate_genomes_parallel_hybrid_impl(
        genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
        num_genomes, num_clusters,
        train_input_bits, train_targets, train_negatives,
        num_train, num_negatives,
        eval_input_bits, eval_targets, num_eval,
        total_input_bits, settings, neuron_sample_rate, rng_seed,
        class_weights,
        None, // override_threshold: default = calibrate on training data
    )
}

/// Log the resolved WNN_* evaluation flags ONCE per process. Runs' effective
/// configs were previously invisible unless every env var was dumped by hand —
/// this is the reproducibility record for task 3.1 (EvalConfig). Env vars are
/// read at function scope per call (initialization defaults), never mid-loop;
/// the old set_var parameter-passing hack is gone (override_threshold is a
/// real parameter now).
fn log_eval_env_once() {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        let flags = [
            "WNN_BATCH_SIZE", "WNN_HYBRID", "WNN_HYBRID_SPEED_RATIO",
            "WNN_OPTION_B", "WNN_GPU_BATCHED_TRAIN", "WNN_GPU_AFFINITY_RATIO",
            "WNN_SHAPE_GROUP", "WNN_COALESCE_GROUPS", "WNN_NO_METAL",
            "WNN_ORDER_INDEPENDENT_TRAIN", "WNN_TIMING", "WNN_GROUP_LOG",
            "WNN_SPARSE_THRESHOLD", "WNN_ATOMIC_SPARSE",
        ];
        let resolved: Vec<String> = flags.iter()
            .map(|f| match std::env::var(f) {
                Ok(v) => format!("{}={}", f, v),
                Err(_) => format!("{}=<unset>", f),
            })
            .collect();
        eprintln!("[EVAL-ENV] resolved flags: {}", resolved.join(" "));
    });
}

#[allow(clippy::too_many_arguments)]
fn evaluate_genomes_parallel_hybrid_impl(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
    override_threshold: Option<f64>,
) -> Vec<(f64, f64, f64, f64, f64, u32)> {
    let empty_value = settings.empty_value;
    log_eval_env_once();
    // The 6th tuple element is eval_time_ms (best-effort per-genome wall-clock).
    // For batched-GPU paths (marker kernel trains N genomes in one Metal dispatch),
    // the time is approximated as `batch_total_ms / N` since the actual work
    // is fused; for the per-genome CPU fallback path the value is exact.
    let memory_mode = settings.memory_mode;
    if num_genomes == 0 {
        return vec![];
    }

    // Pre-compute genome_bpn_offsets: genomes_bits_flat has total_neurons entries per genome
    // (per-neuron bits), NOT num_clusters entries. This offset table maps each genome to its
    // slice in genomes_bits_flat.
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    debug_assert_eq!(
        genomes_bits_flat.len(),
        *genome_bpn_offsets.last().unwrap(),
        "genomes_bits_flat length ({}) != expected total neurons ({})",
        genomes_bits_flat.len(),
        genome_bpn_offsets.last().unwrap(),
    );

    // Get first genome's config to determine pool sizing (use per-cluster max bits)
    let first_neurons = &genomes_neurons_flat[0..num_clusters];
    let first_per_neuron_bits = &genomes_bits_flat[0..genome_bpn_offsets[1]];
    let first_bits_per_cluster = per_cluster_max_bits(first_per_neuron_bits, first_neurons);

    // Hoisted: cpu_cores is needed both for batch_size computation AND for
    // B11 affinity routing (effective CPU thread count).
    let cpu_cores = rayon::current_num_threads();

    // Calculate memory budget and pool size
    let batch_size = std::env::var("WNN_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let budget_gb = get_available_memory_gb() * 0.6;
            let (_, computed_batch) = calculate_pool_size(
                &first_bits_per_cluster,
                first_neurons,
                num_clusters,
                budget_gb,
                cpu_cores,
            );
            computed_batch
        });

    // Pre-compute connection offsets and sizes for each genome (handles variable configs)
    let use_provided_connections = !genomes_connections_flat.is_empty();

    // Compute per-genome connection offsets: conn_size = sum of per-neuron bits
    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    // Create shared eval data (Arc for zero-copy sharing with persistent worker)
    let eval_data = Arc::new(EvalData {
        eval_input_bits: eval_input_bits.clone(),
        eval_targets: eval_targets.to_vec(),
        num_eval,
        num_clusters,
        total_input_bits,
        settings,
    });

    // Get persistent eval worker (initialized once, stays alive for session).
    // Side-effect only: ensures the lazy global is initialized here rather
    // than in the first hot-path call. Return value intentionally unused.
    let _ = get_eval_worker();

    // Collect all results. Tuple is
    //   (genome_idx, ce, acc, f1, fpr, threshold, per_genome_ms).
    let mut all_results: Vec<(usize, f64, f64, f64, f64, f64, u32)> = Vec::with_capacity(num_genomes);

    // Process genomes in batches
    let num_batches = (num_genomes + batch_size - 1) / batch_size;

    // Log batch configuration if WNN_GROUP_LOG is set
    if std::env::var("WNN_GROUP_LOG").is_ok() && num_batches > 1 {
        eprintln!(
            "[BATCH_CONFIG] genomes={} batch_size={} num_batches={}",
            num_genomes, batch_size, num_batches
        );
    }

    // Timing instrumentation (enabled via WNN_TIMING env var)
    let timing_enabled = std::env::var("WNN_TIMING").is_ok();
    let mut total_train_ms = 0u128;
    let mut total_eval_ms = 0u128;
    let mut total_sparse_keys = 0usize;

    // Pack input bits to u64 once (shared across all genomes for GPU address computation)
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_packed_to_u64(train_input_bits);

    // Progress logging for parallel batch (logs training completion, not final results)
    let progress_log = std::env::var("WNN_PROGRESS_LOG").map(|v| v == "1").unwrap_or(false);
    let log_path = std::env::var("WNN_LOG_PATH").ok();
    let current_gen: usize = std::env::var("WNN_PROGRESS_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let total_gens: usize = std::env::var("WNN_PROGRESS_TOTAL_GENS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let log_type = std::env::var("WNN_PROGRESS_TYPE").unwrap_or_else(|_| "Init".to_string());
    let batch_offset: usize = std::env::var("WNN_PROGRESS_OFFSET").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let total_count: usize = std::env::var("WNN_PROGRESS_TOTAL").ok().and_then(|v| v.parse().ok()).unwrap_or(num_genomes);

    for batch_idx in 0..num_batches {
        // Cooperative SIGTERM cancellation (added 31/05/2026): poll at the
        // batch boundary in evaluate_genomes_parallel_hybrid. When set, leave
        // the outer loop early. Genomes that were never processed get their
        // default (zero) entries in `genome_results` — callers see a short
        // results vec and treat the missing tail as "not evaluated", matching
        // the same shape as a partial offspring search.
        if crate::cancel::check_cancel() {
            break;
        }
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_genomes);
        let current_batch_size = batch_end - batch_start;

        let train_start = std::time::Instant::now();

        // GPU batched train (marker-FSM kernel): when the batch shape favors
        // GPU parallelism, dispatch ONE Metal kernel to train all genomes in
        // this batch. Otherwise fall through to the per-genome par_iter
        // baseline. B11 affinity makes this automatic — no env var needed.
        //
        // Why a per-batch routing decision instead of a global flag:
        //   - Dense regime (max b ≤ SPARSE_THRESHOLD=12): 2^b cells fit in
        //     L1 cache; CPU's direct array indexing beats GPU hashing.
        //   - Baseline scales with min(ng, cpu_cores) effective threads, so
        //     as ng → 16 the baseline CPU path closes the gap. Option B
        //     wins when GPU thread count (ng × n × chunks) substantially
        //     exceeds baseline's effective parallelism.
        //
        // Heuristic: GPU wins when both
        //   - max_bits > 12 (sparse regime where hashing helps)
        //   - effective_gpu_threads / effective_cpu_threads > AFFINITY_RATIO
        //     where effective_gpu_threads ≈ ng × n and
        //           effective_cpu_threads ≈ min(ng, cpu_cores)
        //
        // WNN_GPU_BATCHED_TRAIN env var (still accepted) acts as override:
        //   - "off"/"0"/"false": force CPU baseline always
        //   - "force"/"always":  force GPU always (testing/benchmarks)
        //   - unset or "1":      use B11 affinity (default behavior)
        let mut batch_exports: Vec<(usize, GenomeExport, Option<f64>)>;
        let gpu_override = std::env::var("WNN_GPU_BATCHED_TRAIN").ok().unwrap_or_default();
        let force_cpu = matches!(gpu_override.as_str(), "off" | "0" | "false" | "no");
        let force_gpu = matches!(gpu_override.as_str(), "force" | "always");
        let legacy_off = std::env::var("WNN_OPTION_B").map(|v| matches!(v.as_str(), "off" | "0")).unwrap_or(false);
        // Affinity inputs for this batch
        let _aff_bpn_start = genome_bpn_offsets[batch_start];
        let _aff_bpn_end = genome_bpn_offsets[batch_end];
        let max_bits_in_batch = genomes_bits_flat[_aff_bpn_start.._aff_bpn_end]
            .iter().copied().max().unwrap_or(0);
        let total_neurons_in_batch: usize = genomes_neurons_flat
            [batch_start * num_clusters..batch_end * num_clusters].iter().sum();
        let avg_n_per_genome = (total_neurons_in_batch as f64 / current_batch_size.max(1) as f64) as usize;
        let effective_gpu_threads = current_batch_size * avg_n_per_genome;
        let effective_cpu_threads = current_batch_size.min(cpu_cores).max(1);
        let parallelism_ratio = effective_gpu_threads as f64 / effective_cpu_threads as f64;
        let affinity_ratio_threshold: f64 = std::env::var("WNN_GPU_AFFINITY_RATIO")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(100.0);
        let gpu_wins_here = max_bits_in_batch > SPARSE_THRESHOLD
            && parallelism_ratio >= affinity_ratio_threshold;
        let use_gpu_batched = if force_cpu || legacy_off {
            false
        } else if force_gpu {
            // Honor force-gpu even for dense (caller is explicitly testing).
            true
        } else {
            gpu_wins_here
        };
        let trace = gpu_batched_train_trace();
        if trace {
            let path = if use_gpu_batched { "GPU" } else { "CPU" };
            eprintln!(
                "[GPU_BATCHED_TRACE] B11: batch={}/{} ng={} n={} max_b={} ratio={:.0} → {}",
                batch_idx, num_batches, current_batch_size, avg_n_per_genome,
                max_bits_in_batch, parallelism_ratio, path
            );
        }

        // B12+B13 hybrid decision must be computed BEFORE option_b — when
        // hybrid wins, skip Option B entirely and split the batch between
        // CPU and GPU paths via std::thread::scope. Without this ordering,
        // Option B would always fire when B11 picks GPU, and hybrid would
        // be dead code.
        let hybrid_enabled = std::env::var("WNN_HYBRID")
            .map(|v| !matches!(v.as_str(), "0" | "off" | "false" | "no"))
            .unwrap_or(true);
        let batch_shape: ShapeKey = {
            let neurons_per_cluster: Vec<usize> =
                genomes_neurons_flat[batch_start * num_clusters..(batch_start + 1) * num_clusters].to_vec();
            (neurons_per_cluster, max_bits_in_batch)
        };
        let batch_is_homogeneous = (batch_start..batch_end).all(|g| {
            let off = g * num_clusters;
            &genomes_neurons_flat[off..off + num_clusters] == &genomes_neurons_flat[batch_start * num_clusters..(batch_start + 1) * num_clusters]
        });
        let shape_state_pre = read_shape_state(&batch_shape);
        let speed_ratio = {
            let cpu_us = shape_state_pre.cpu_time_per_genome_us.max(1.0);
            let gpu_us = shape_state_pre.gpu_time_per_genome_us.max(1.0);
            cpu_us.max(gpu_us) / cpu_us.min(gpu_us)
        };
        // speed_ratio guard at 1.5: on Apple Silicon's unified memory, CPU+GPU
        // concurrent access creates bandwidth contention. When one path is
        // already >1.5× faster, splitting tends to regress (the slower path
        // becomes the bottleneck AND gets even slower due to contention).
        // Tunable via WNN_HYBRID_SPEED_RATIO env var (default 1.5).
        let speed_ratio_max: f64 = std::env::var("WNN_HYBRID_SPEED_RATIO")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
        let want_hybrid = hybrid_enabled
            && use_gpu_batched     // hybrid only makes sense when GPU is a viable path
            && current_batch_size >= 4
            && speed_ratio <= speed_ratio_max
            && batch_is_homogeneous;
        if trace && want_hybrid {
            eprintln!("[GPU_BATCHED_TRACE] B12/B13: hybrid eligible (speed_ratio={:.2}, batch={})", speed_ratio, current_batch_size);
        }

        // Option 2 (B14): per-shape-group routing for HETEROGENEOUS batches.
        //
        // When B11 picks GPU and the batch contains multiple shapes (typical
        // GA Neurons generations evolving neuron counts), the standard
        // batched_train_offspring path errors out (non-uniform shape). We
        // recover by grouping genomes by shape and dispatching each group
        // as its own batched_train_offspring call. Per-group results are
        // merged back into batch_exports in original genome_idx order.
        //
        // Set WNN_SHAPE_GROUP=0 to disable (fallback to per-genome CPU path
        // on heterogeneous batches).
        let shape_grouping_enabled = std::env::var("WNN_SHAPE_GROUP")
            .map(|v| !matches!(v.as_str(), "0" | "off" | "false" | "no"))
            .unwrap_or(true);
        let do_shape_grouping = shape_grouping_enabled
            && use_gpu_batched
            && !batch_is_homogeneous
            && !want_hybrid;

        let shape_group_result: Option<Vec<(usize, GenomeExport, Option<f64>)>> = if do_shape_grouping {
            #[cfg(target_os = "macos")]
            {
                // B14 RELAXED (18/05/2026): with batched_train_offspring now
                // handling heterogeneous bpn (commit cf3ff63a), the strict
                // bpn-uniform requirement is gone. Shape key is just
                // `neurons_per_cluster` — genomes with the same neuron layout
                // share a batch regardless of per-neuron bit-width variation.
                // batched_train_offspring pads connections to N × max_bits per
                // genome internally; downstream evaluate_group_sparse_gpu
                // receives padded layouts from reorganize_connections_for_gpu.
                //
                // Original strict-key comment (kept for historical context):
                // > B14 bug fix (15/05/2026): two genomes can share
                // > (neurons_per_cluster, max_bits) but have different bpn
                // > arrays. Including the full bpn tuple in the key forces
                // > each into its own group. ← no longer needed; relaxed.
                type ShapeKey = Vec<usize>;  // just neurons_per_cluster
                let mut shape_to_locals: std::collections::HashMap<ShapeKey, Vec<usize>> =
                    std::collections::HashMap::new();
                for local_idx in 0..current_batch_size {
                    let genome_idx = batch_start + local_idx;
                    let off = genome_idx * num_clusters;
                    let neurons: Vec<usize> = genomes_neurons_flat[off..off + num_clusters].to_vec();
                    shape_to_locals.entry(neurons).or_default().push(local_idx);
                }

                if trace {
                    let sizes: Vec<usize> = shape_to_locals.values().map(|v| v.len()).collect();
                    eprintln!(
                        "[GPU_BATCHED_TRACE] B14 shape-group: batch={} groups={} sizes={:?}",
                        current_batch_size, shape_to_locals.len(), sizes
                    );
                }

                // For each shape group, build contiguous slices and dispatch.
                // Per-group success → use returned exports. Per-group failure → None
                // for those locals (will fall to CPU per-genome).
                let mut per_local_export: Vec<Option<GenomeExport>> = (0..current_batch_size).map(|_| None).collect();
                let mut any_group_failed = false;

                for (shape_neurons, locals) in shape_to_locals.iter() {
                    let group_size = locals.len();
                    // Build per-group flat slices
                    let mut g_bits: Vec<usize> = Vec::new();
                    let mut g_neurons: Vec<usize> = Vec::new();
                    let mut g_conns: Vec<i64> = Vec::new();
                    for &li in locals.iter() {
                        let gi = batch_start + li;
                        let bpn_s = genome_bpn_offsets[gi];
                        let bpn_e = genome_bpn_offsets[gi + 1];
                        g_bits.extend_from_slice(&genomes_bits_flat[bpn_s..bpn_e]);
                        let off = gi * num_clusters;
                        g_neurons.extend_from_slice(&genomes_neurons_flat[off..off + num_clusters]);
                        if use_provided_connections {
                            let cs = conn_offsets[gi];
                            let cn = conn_sizes[gi];
                            g_conns.extend_from_slice(&genomes_connections_flat[cs..cs + cn]);
                        }
                    }
                    // ALWAYS pass actual connections (B14 relaxed — batched_train_offspring
                    // handles heterogeneous-bpn / non-uniform conn-size layouts via
                    // the cf3ff63a fix). Previously we dropped to random connections
                    // if conn-sizes differed across genomes; that loses evolved
                    // connectivity and silently produces wrong results.
                    let g_conns_slice: &[i64] = if use_provided_connections { &g_conns } else { &[] };
                    let _ = shape_neurons;  // already in g_neurons

                    match crate::marker_train::batched_train_offspring(
                        &g_bits, &g_neurons, g_conns_slice,
                        group_size, num_clusters,
                        train_input_bits, train_targets, train_negatives,
                        num_train, num_negatives, total_input_bits, empty_value,
                        neuron_sample_rate, rng_seed.wrapping_add(batch_start as u64),
                        class_weights,
                    ) {
                        Ok(g_exports) => {
                            for (rel_idx, export) in g_exports.into_iter().enumerate() {
                                let li = locals[rel_idx];
                                per_local_export[li] = Some(export);
                            }
                        }
                        Err(e) => {
                            any_group_failed = true;
                            if trace {
                                eprintln!("[GPU_BATCHED_TRACE] B14 group failed ({} genomes, reason: {}) — those locals will use CPU fallback", group_size, e);
                            }
                        }
                    }
                }

                // If ALL groups succeeded, assemble the result.
                // Otherwise pass None back to trigger the CPU per-genome fallback
                // (it will recompute everything; we accept that inefficiency for
                // safety — partial GPU + partial CPU merging is harder to get
                // right than just falling back wholesale).
                if any_group_failed {
                    None
                } else {
                    let mut merged: Vec<(usize, GenomeExport, Option<f64>)> = Vec::with_capacity(current_batch_size);
                    for (local_idx, opt_export) in per_local_export.into_iter().enumerate() {
                        match opt_export {
                            Some(export) => merged.push((batch_start + local_idx, export, None)),
                            None => unreachable!("any_group_failed == false but local has no export"),
                        }
                    }
                    Some(merged)
                }
            }
            #[cfg(not(target_os = "macos"))]
            { None }
        } else { None };

        // Run Option B ONLY when GPU is the chosen path AND hybrid/shape-grouping isn't applicable.
        let option_b_result: Option<Vec<GenomeExport>> = if use_gpu_batched && !want_hybrid && shape_group_result.is_none() && batch_is_homogeneous {
            // Slice flat arrays for this batch's genomes
            let bpn_slice_start = genome_bpn_offsets[batch_start];
            let bpn_slice_end = genome_bpn_offsets[batch_end];
            let bits_slice = &genomes_bits_flat[bpn_slice_start..bpn_slice_end];
            let neurons_slice = &genomes_neurons_flat[batch_start * num_clusters..batch_end * num_clusters];

            // Connections: only pass if all batch genomes share the same
            // conn_per_genome — i.e., genome i+1 starts exactly conn_per_genome
            // after genome i in the flat array. The existing path uses
            // conn_offsets/conn_sizes; for uniform we can pass the slice.
            let first_conn = conn_sizes[batch_start];
            let uniform_conns = (batch_start..batch_end).all(|g| conn_sizes[g] == first_conn);
            let conns_slice: &[i64] = if use_provided_connections && uniform_conns {
                let cs = conn_offsets[batch_start];
                let ce = cs + first_conn * current_batch_size;
                &genomes_connections_flat[cs..ce]
            } else {
                &[]
            };

            #[cfg(target_os = "macos")]
            {
                match crate::marker_train::batched_train_offspring(
                    bits_slice,
                    neurons_slice,
                    conns_slice,
                    current_batch_size,
                    num_clusters,
                    train_input_bits,
                    train_targets,
                    train_negatives,
                    num_train,
                    num_negatives,
                    total_input_bits,
                    empty_value,
                    neuron_sample_rate,
                    rng_seed.wrapping_add(batch_start as u64),
                    class_weights,
                ) {
                    Ok(exports) => Some(exports),
                    Err(e) => {
                        eprintln!("[GPU_BATCHED] batched dispatch fallback (reason: {})", e);
                        None
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            { None }
        } else { None };

        if let Some(merged) = shape_group_result {
            // B14 shape-group routing succeeded for ALL groups.
            batch_exports = merged;
        } else if let Some(exports) = option_b_result {
            // Homogeneous-batch Option B path.
            batch_exports = exports.into_iter().enumerate()
                .map(|(local_idx, export)| (batch_start + local_idx, export, None))
                .collect();
        } else {
        // Existing per-genome par_iter (default path).
        //
        // B12: extracted as a closure so the hybrid CPU+GPU path can call
        // it on a SUBSET of the batch [0..k_cpu] while GPU runs the
        // complementary subset [k_cpu..]. Same body as the original
        // unconditional path — just parameterized on genome_idx.
        let cpu_one_genome = |local_idx: usize| -> (usize, GenomeExport, Option<f64>) {
                let genome_idx = batch_start + local_idx;

                // Cooperative SIGTERM cancellation (added 31/05/2026): poll at
                // the start of each per-genome rayon worker callback. If set,
                // skip the (expensive) training work and return an empty
                // GenomeExport — the outer loop will detect the partial batch
                // and short-circuit further iterations.
                if crate::cancel::check_cancel() {
                    return (genome_idx, GenomeExport::empty(), None);
                }

                // Get this genome's config (per-neuron bits + per-cluster neurons)
                let genome_offset = genome_idx * num_clusters;
                let neurons_per_cluster = &genomes_neurons_flat[genome_offset..genome_offset + num_clusters];

                // Extract per-neuron bits for this genome
                let bpn_start = genome_bpn_offsets[genome_idx];
                let bpn_end = genome_bpn_offsets[genome_idx + 1];
                let per_neuron_bits = &genomes_bits_flat[bpn_start..bpn_end];

                // Compute per-cluster max bits (for build_groups and GPU dispatch)
                let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

                // Build neuron metadata for per-neuron training
                let (cluster_neuron_starts, neuron_conn_offsets) =
                    build_neuron_metadata(per_neuron_bits, neurons_per_cluster);

                // Build config groups for THIS genome (using per-cluster max bits)
                let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

                // Build cluster-to-group mapping for THIS genome
                let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
                for (group_idx, group) in groups.iter().enumerate() {
                    for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                        cluster_to_group[cluster_id] = (group_idx, local_idx);
                    }
                }

                // Create memory for THIS genome
                let mut memories: Vec<GroupMemory> = groups.iter()
                    .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                    .collect();

                // Get original per-neuron connections for this genome
                let original_connections: Vec<i64> = if use_provided_connections {
                    let conn_offset = conn_offsets[genome_idx];
                    let conn_size = conn_sizes[genome_idx];
                    genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec()
                } else {
                    // Generate random per-neuron connections
                    use rand::{Rng, SeedableRng};
                    let mut rng = rand::rngs::SmallRng::seed_from_u64((genome_idx * 12345) as u64);
                    let total_conn: usize = per_neuron_bits.iter().sum();
                    let mut conns = Vec::with_capacity(total_conn);
                    for _ in 0..total_conn {
                        conns.push(rng.gen_range(0..total_input_bits as i64));
                    }
                    conns
                };

                // Path 2 migration: train via Option B (MarkerHashTable /
                // batched_train_offspring) → GenomeExport directly. No
                // separate compute_addresses dispatch, no u32 truncation
                // guard, native u64 keys. Falls back to the dense chunked
                // path on error (memory budget, kernel failure).
                let export = match train_single_via_marker(
                    per_neuron_bits,
                    neurons_per_cluster,
                    &original_connections,
                    num_clusters,
                    train_input_bits,
                    train_targets,
                    train_negatives,
                    num_train,
                    num_negatives,
                    total_input_bits,
                    empty_value,
                    neuron_sample_rate,
                    rng_seed.wrapping_add(genome_idx as u64),
                    class_weights,
                ) {
                    Ok(e) => e,
                    Err(reason) => {
                        eprintln!(
                            "[PATH2_FALLBACK] evaluate_genomes_parallel_hybrid g={} → dense: {}",
                            genome_idx, reason
                        );
                        // Fallback: original dense path (single-shot + chunked variants)
                        let total_neurons = per_neuron_bits.len();
                        let total_addresses_estimate = total_neurons.saturating_mul(num_train);

                        if total_addresses_estimate <= MAX_GPU_ADDRESSES {
                            let gpu_addresses = try_gpu_addresses_adaptive(
                                &packed_train_input,
                                words_per_example,
                                per_neuron_bits,
                                &neuron_conn_offsets,
                                &original_connections,
                                num_train,
                            );
                            train_genome_in_slot(
                                &mut memories,
                                &groups,
                                &original_connections,
                                per_neuron_bits,
                                &cluster_neuron_starts,
                                &neuron_conn_offsets,
                                &cluster_to_group,
                                train_input_bits,
                                train_targets,
                                train_negatives,
                                num_train,
                                num_negatives,
                                total_input_bits,
                                gpu_addresses.as_deref(),
                                neuron_sample_rate,
                                rng_seed.wrapping_add(genome_idx as u64),
                                memory_mode,
                                class_weights,
                                true,
                            );
                        } else {
                            // OI orchestration brackets the entire chunked loop:
                            // counters accumulate across chunks, then commit once.
                            let oi_chunked = crate::neuron_memory::order_independent_training_enabled()
                                && memory_mode == crate::neuron_memory::MODE_QUAD_WEIGHTED;
                            if oi_chunked {
                                for m in memories.iter_mut() { m.init_oi_counters(); }
                            }
                            let chunk_size = (MAX_GPU_ADDRESSES / total_neurons.max(1)).max(1);
                            let mut chunk_start = 0;
                            while chunk_start < num_train {
                                let chunk_end = (chunk_start + chunk_size).min(num_train);
                                let chunk_len = chunk_end - chunk_start;
                                let chunk_packed = &packed_train_input[
                                    chunk_start * words_per_example..chunk_end * words_per_example
                                ];
                                let chunk_addresses = try_gpu_addresses_for_chunk(
                                    chunk_packed,
                                    words_per_example,
                                    per_neuron_bits,
                                    &neuron_conn_offsets,
                                    &original_connections,
                                    chunk_len,
                                );
                                train_genome_in_slot_range(
                                    &memories,
                                    &groups,
                                    &original_connections,
                                    per_neuron_bits,
                                    &cluster_neuron_starts,
                                    &neuron_conn_offsets,
                                    &cluster_to_group,
                                    train_input_bits,
                                    train_targets,
                                    train_negatives,
                                    num_train,
                                    num_negatives,
                                    total_input_bits,
                                    chunk_addresses.as_deref(),
                                    chunk_start..chunk_end,
                                    chunk_len,
                                    neuron_sample_rate,
                                    rng_seed.wrapping_add(genome_idx as u64),
                                    memory_mode,
                                    class_weights,
                                    true,
                                );
                                chunk_start = chunk_end;
                            }
                            if oi_chunked {
                                for m in memories.iter_mut() { m.commit_oi(); }
                            }
                        }
                        let gpu_connections = reorganize_connections_for_gpu(
                            &original_connections,
                            per_neuron_bits,
                            neurons_per_cluster,
                            &groups,
                        );
                        export_genome_for_gpu(&memories, &groups, &gpu_connections)
                    }
                };

                // Threshold calibration done AFTER par_iter to avoid nested parallelism
                (genome_idx, export, None)
            };  // end cpu_one_genome closure

        // B12 hybrid CPU+GPU: when GPU would have been selected by B11 AND
        // the batch is large enough to benefit, split the batch — CPU
        // processes [0..k_cpu] via rayon, GPU processes [k_cpu..] via
        // batched_train_offspring. Both run in parallel via std::thread::scope.
        // Throughput state is updated adaptively for the next batch.
        //
        // Hybrid is OPT-IN via WNN_HYBRID=1. Reason: unified memory on Apple
        // Silicon means CPU and GPU compete for memory bandwidth when running
        // concurrently. For balanced workloads (cpu_time ≈ gpu_time) the
        // split halves wall time (~15-30% measured). For workloads where one
        // path dominates, the slower path becomes the bottleneck and hybrid
        // *hurts*. Auto-detection of "balanced enough" is brittle, so keep
        // off by default; users opt in when they know their workload mix.
        //
        // (B12/B13 hybrid decision was computed earlier — see want_hybrid above.)
        let shape_state = shape_state_pre;

        if want_hybrid {
            // Adaptive split: per-shape state — already loaded above.
            let cpu_time_per_genome_us = shape_state.cpu_time_per_genome_us;
            let gpu_time_per_genome_us = shape_state.gpu_time_per_genome_us;
            // Throughput inverse: faster path gets more genomes.
            // genomes_per_path ∝ 1/time_per_genome
            let cpu_rate = 1.0 / cpu_time_per_genome_us.max(1.0);
            let gpu_rate = 1.0 / gpu_time_per_genome_us.max(1.0);
            let cpu_share = cpu_rate / (cpu_rate + gpu_rate);
            // Clamp to [1/batch_size, 1 - 1/batch_size] so both arms have ≥1 genome
            let min_share = 1.0 / current_batch_size as f64;
            let cpu_share = cpu_share.clamp(min_share, 1.0 - min_share);
            let k_cpu = ((current_batch_size as f64) * cpu_share).round() as usize;
            let k_cpu = k_cpu.max(1).min(current_batch_size - 1);
            let k_gpu = current_batch_size - k_cpu;

            if trace {
                eprintln!(
                    "[GPU_BATCHED_TRACE] B12 hybrid: batch_size={} k_cpu={} k_gpu={} cpu_us/genome≈{:.0} gpu_us/genome≈{:.0}",
                    current_batch_size, k_cpu, k_gpu, cpu_time_per_genome_us, gpu_time_per_genome_us
                );
            }

            // Slice GPU inputs for [k_cpu..batch_end].
            let gpu_batch_start = batch_start + k_cpu;
            let gpu_bpn_start = genome_bpn_offsets[gpu_batch_start];
            let gpu_bpn_end = genome_bpn_offsets[batch_end];
            let gpu_bits_slice = &genomes_bits_flat[gpu_bpn_start..gpu_bpn_end];
            let gpu_neurons_slice = &genomes_neurons_flat[gpu_batch_start * num_clusters..batch_end * num_clusters];
            let gpu_first_conn = conn_sizes[gpu_batch_start];
            let gpu_uniform_conns = (gpu_batch_start..batch_end).all(|g| conn_sizes[g] == gpu_first_conn);
            let gpu_conns_slice: &[i64] = if use_provided_connections && gpu_uniform_conns {
                let cs = conn_offsets[gpu_batch_start];
                let ce = cs + gpu_first_conn * k_gpu;
                &genomes_connections_flat[cs..ce]
            } else {
                &[]
            };

            #[cfg(target_os = "macos")]
            let (cpu_results, gpu_result_or_err, cpu_elapsed, gpu_elapsed) = std::thread::scope(|scope| {
                // GPU thread — blocks on Metal wait. Releases nothing from rayon pool (it's a std::thread).
                let gpu_handle = scope.spawn(|| {
                    let t = std::time::Instant::now();
                    let r = crate::marker_train::batched_train_offspring(
                        gpu_bits_slice,
                        gpu_neurons_slice,
                        gpu_conns_slice,
                        k_gpu,
                        num_clusters,
                        train_input_bits,
                        train_targets,
                        train_negatives,
                        num_train,
                        num_negatives,
                        total_input_bits,
                        empty_value,
                        neuron_sample_rate,
                        rng_seed.wrapping_add(gpu_batch_start as u64),
                        class_weights,
                    );
                    (t.elapsed(), r)
                });
                // CPU work on this thread (rayon's pool is shared and used internally).
                let t_cpu = std::time::Instant::now();
                let cpu_res: Vec<(usize, GenomeExport, Option<f64>)> = (0..k_cpu)
                    .into_par_iter()
                    .map(&cpu_one_genome)
                    .collect();
                let cpu_e = t_cpu.elapsed();
                let (gpu_e, gpu_r) = gpu_handle.join().expect("B12 GPU thread panicked");
                (cpu_res, gpu_r, cpu_e, gpu_e)
            });

            // Update PER-SHAPE throughput state (EMA factor 0.3).
            let cpu_us = cpu_elapsed.as_micros() as f64 / k_cpu as f64;
            let gpu_us = gpu_elapsed.as_micros() as f64 / k_gpu as f64;
            update_shape_state(&batch_shape, cpu_us, gpu_us);
            if trace {
                eprintln!(
                    "[GPU_BATCHED_TRACE] B12 measured: cpu={:.2}ms ({:.0}us/g) gpu={:.2}ms ({:.0}us/g)",
                    cpu_elapsed.as_secs_f64() * 1000.0, cpu_us,
                    gpu_elapsed.as_secs_f64() * 1000.0, gpu_us
                );
            }

            #[cfg(target_os = "macos")]
            match gpu_result_or_err {
                Ok(gpu_exports) => {
                    // Merge: CPU results occupy genome_indices [batch_start..batch_start+k_cpu],
                    // GPU results occupy [batch_start+k_cpu..batch_end].
                    batch_exports = cpu_results;
                    batch_exports.extend(gpu_exports.into_iter().enumerate().map(|(local_idx, export)| {
                        (gpu_batch_start + local_idx, export, None)
                    }));
                }
                Err(e) => {
                    eprintln!("[GPU_BATCHED] B12 hybrid GPU arm failed (reason: {}); recomputing on CPU", e);
                    // Recompute GPU half on CPU.
                    let cpu_fallback: Vec<_> = (k_cpu..current_batch_size).into_par_iter()
                        .map(&cpu_one_genome).collect();
                    batch_exports = cpu_results;
                    batch_exports.extend(cpu_fallback);
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = (cpu_results, gpu_result_or_err, cpu_elapsed, gpu_elapsed);
                batch_exports = (0..current_batch_size).into_par_iter().map(&cpu_one_genome).collect();
            }
        } else {
            // Non-hybrid CPU-only path
            batch_exports = (0..current_batch_size).into_par_iter().map(&cpu_one_genome).collect();
        }
        }  // end else (existing per-genome par_iter path)

        // Single-cluster: calibrate thresholds sequentially (compute_per_example_scores
        // uses par_iter internally, which would deadlock inside the outer par_iter above)
        if num_clusters == 1 {
            // Threshold override: a real parameter (was a set_var/remove_var
            // env-var hack — non-reentrant under rayon and unsafe in Rust 2024).
            let override_t: Option<f64> = override_threshold;

            if let Some(t) = override_t {
                if t < 0.0 {
                    // -1.0: don't set override, let evaluate_genome_hybrid find on eval data
                    // (batch_exports already have None as threshold)
                } else {
                    // Fixed threshold (e.g., 0.5)
                    for (_, _, threshold) in batch_exports.iter_mut() {
                        *threshold = Some(t);
                    }
                }
            } else {
                // Default: calibrate on training data
                let metal_arc = get_metal_evaluator();
                let sparse_metal_arc = get_sparse_metal_evaluator();
                for (_, export, threshold) in batch_exports.iter_mut() {
                    let train_scores = compute_per_example_scores(
                        export, train_input_bits, &packed_train_input, words_per_example,
                        num_train, num_clusters, total_input_bits, empty_value,
                        memory_mode,
                        metal_arc.as_ref().map(|a| a.as_ref()),
                        sparse_metal_arc.as_ref().map(|a| a.as_ref()),
                    );
                    let flat_scores: Vec<f64> = train_scores.iter().map(|s| s[0]).collect();
                    let (t, _f1, _fpr) = find_optimal_threshold_auto(&flat_scores, train_targets, settings.fitness_weights);
                    *threshold = Some(t);
                }
            }
        }

        let train_elapsed = train_start.elapsed();

        // Track sparse export sizes for timing diagnostics
        let sparse_keys_total: usize = if timing_enabled {
            batch_exports.iter()
                .map(|(_, export, _)| export.sparse_exports.iter().map(|se| se.keys.len()).sum::<usize>())
                .sum()
        } else { 0 };

        let eval_start = std::time::Instant::now();

        // Evaluate inline (eval worker thread causes rayon deadlock when
        // compute_per_example_scores tries to use par_iter from a non-rayon thread)
        let metal_arc = get_metal_evaluator();
        let sparse_metal_arc = get_sparse_metal_evaluator();
        let metal_ref = metal_arc.as_ref().map(|a| a.as_ref());
        let sparse_metal_ref = sparse_metal_arc.as_ref().map(|a| a.as_ref());
        let batch_results: Vec<(usize, f64, f64, f64, f64, f64)> = batch_exports
            .into_iter()
            .map(|(genome_idx, export, override_threshold)| {
                let (ce, acc, f1, fpr, threshold) = evaluate_genome_hybrid(
                    &export,
                    &eval_data.eval_input_bits,
                    &eval_data.eval_targets,
                    eval_data.num_eval,
                    eval_data.num_clusters,
                    eval_data.total_input_bits,
                    settings,
                    metal_ref,
                    sparse_metal_ref,
                    override_threshold,
                );
                (genome_idx, ce, acc, f1, fpr, threshold)
            })
            .collect();

        let eval_elapsed_secs = eval_start.elapsed().as_secs_f64();
        let batch_total_secs = train_elapsed.as_secs_f64() + eval_elapsed_secs;
        // Per-genome timing approximation: train+eval batch wall-time divided
        // by the number of genomes in this batch. For batched-GPU paths this is
        // amortized; the per-genome CPU fallback would naturally come out near
        // exact since each genome takes roughly the same fraction of the wall.
        let per_genome_ms: u32 = if current_batch_size > 0 {
            ((batch_total_secs * 1000.0) / current_batch_size as f64).round().clamp(0.0, u32::MAX as f64) as u32
        } else {
            0
        };

        // Log results with CE/Acc after batch completes
        if progress_log {
            use std::io::Write;
            let now = chrono::Local::now();
            let gen_width = total_gens.to_string().len();
            let pos_width = total_count.to_string().len();
            let type_padded = format!("{:<4}", &log_type[..log_type.len().min(4)]);

            for (genome_idx, ce, acc, _f1, _fpr, _threshold) in &batch_results {
                let overall_position = batch_offset + genome_idx + 1;
                let msg = format!(
                    "{} | [Gen {:0gen_width$}/{:0gen_width$}] Genome {:0pos_width$}/{} ({}): CE={:.4}, Acc={:.4}% ({:.1}s)\n",
                    now.format("%H:%M:%S"),
                    current_gen, total_gens,
                    overall_position, total_count,
                    type_padded,
                    ce, acc * 100.0,
                    batch_total_secs,
                    gen_width = gen_width,
                    pos_width = pos_width,
                );
                if let Some(ref path) = log_path {
                    if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
                        let _ = file.write_all(msg.as_bytes());
                        let _ = file.flush();
                    }
                } else {
                    eprint!("{}", msg);
                }
            }
        }

        // Attach per_genome_ms to each tuple before merging into all_results.
        // batch_results: Vec<(usize, f64, f64, f64, f64, f64)>
        // → Vec<(usize, f64, f64, f64, f64, f64, u32)>
        let batch_results_with_time: Vec<(usize, f64, f64, f64, f64, f64, u32)> = batch_results
            .into_iter()
            .map(|(gi, ce, acc, f1, fpr, t)| (gi, ce, acc, f1, fpr, t, per_genome_ms))
            .collect();
        all_results.extend(batch_results_with_time);

        if timing_enabled {
            total_train_ms += train_elapsed.as_millis();
            total_eval_ms += (eval_elapsed_secs * 1000.0) as u128;
            total_sparse_keys += sparse_keys_total;
        }
    }

    // Print timing summary if enabled
    if timing_enabled && num_genomes > 0 {
        let train_per_genome = total_train_ms as f64 / num_genomes as f64;
        let eval_per_genome = total_eval_ms as f64 / num_genomes as f64;
        let sparse_per_genome = total_sparse_keys as f64 / num_genomes as f64;
        eprintln!(
            "[TIMING] batch_size={}, genomes={}: train={:.0}ms/genome, eval={:.0}ms/genome, total={:.0}ms/genome, sparse_keys={:.0}/genome",
            batch_size, num_genomes, train_per_genome, eval_per_genome, train_per_genome + eval_per_genome, sparse_per_genome
        );
    }

    // Sort results by genome index and return. 6-tuple now: 5 metrics + per_genome_ms.
    let mut results: Vec<(f64, f64, f64, f64, f64, u32)> = vec![(0.0, 0.0, 0.0, 0.0, 0.5, 0u32); num_genomes];
    for (genome_idx, ce, acc, f1, fpr, threshold, ms) in all_results {
        results[genome_idx] = (ce, acc, f1, fpr, threshold, ms);
    }

    results
}

/// Same as evaluate_genomes_parallel_hybrid but with optional threshold override.
/// - None: train-calibrated threshold (58d57c6, current default)
/// - Some(-1.0): find optimal on EVAL data (302a36d = data leakage test)
/// - Some(t >= 0): fixed threshold (e.g. 0.5 for 5aa659d)
///
/// Works by running the standard pipeline then overriding the threshold in batch_exports
/// before the inline eval step. For Some(-1.0), passes None to evaluate_genome_hybrid
/// which triggers its internal eval-data threshold sweep.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genomes_parallel_hybrid_with_override(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &crate::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &crate::packed_bits::PackedBits,
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    settings: crate::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
    override_threshold: Option<f64>,
) -> Vec<(f64, f64, f64, f64, f64, u32)> {
    evaluate_genomes_parallel_hybrid_impl(
        genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
        num_genomes, num_clusters,
        train_input_bits, train_targets, train_negatives,
        num_train, num_negatives,
        eval_input_bits, eval_targets, num_eval,
        total_input_bits, settings, neuron_sample_rate, rng_seed,
        class_weights,
        override_threshold,
    )
}
