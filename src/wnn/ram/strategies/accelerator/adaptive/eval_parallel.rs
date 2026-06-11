//! Unified parallel CPU genome evaluation (rayon) + legacy parity path.
//!
//! Split out of adaptive/eval.rs (D3 follow-up, 11/06/2026).

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
