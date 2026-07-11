//! Single-genome paths: hybrid eval/predict, marker training, train-and-score variants, binary metrics.
//!
//! Split out of adaptive/eval.rs (D3 follow-up, 11/06/2026).

use super::*;

pub fn evaluate_genome_hybrid(
    export: &GenomeExport,
    eval_input_bits: &ram_core::packed_bits::PackedBits,
    eval_targets: &[i64],
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    settings: ram_core::neuron_memory::EvalSettings,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&ram_core::metal_sparse::MetalSparseEvaluator>,
    override_threshold: Option<f64>,
) -> (f64, f64, f64, f64, f64) {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;
    let epsilon = 1e-10f64;

    // Detailed timing (enabled via WNN_GROUP_TIMING env var)
    let timing_enabled = std::env::var("WNN_GROUP_TIMING").is_ok();
    let eval_start = std::time::Instant::now();

    // Pack eval input bits to u64 for GPU (pack once, reuse for all GPU paths)
    let (packed_eval, words_per_example) = ram_core::neuron_memory::pack_packed_to_u64(eval_input_bits);
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
                static CE_EVALUATOR: std::sync::OnceLock<Option<crate::metal_genome_eval::MetalSparseCEEvaluator>> = std::sync::OnceLock::new();
                let ce_eval = CE_EVALUATOR.get_or_init(|| {
                    crate::metal_genome_eval::MetalSparseCEEvaluator::new().ok()
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
            let mut sparse_groups: Vec<crate::metal_genome_eval::SparseGroupData> = Vec::new();
            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                if *is_sparse {
                    let group = &export.groups[*group_idx];
                    let sparse_export = &export.sparse_exports[sparse_idx];
                    sparse_idx += 1;

                    sparse_groups.push(crate::metal_genome_eval::SparseGroupData {
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
    eval_input_bits: &ram_core::packed_bits::PackedBits,
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    settings: ram_core::neuron_memory::EvalSettings,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&ram_core::metal_sparse::MetalSparseEvaluator>,
    single_cluster_threshold: Option<f64>,
) -> Vec<i64> {
    let empty_value = settings.empty_value;
    let memory_mode = settings.memory_mode;

    let (packed_eval, words_per_example) = ram_core::neuron_memory::pack_packed_to_u64(eval_input_bits);

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
    train_input_bits: &ram_core::packed_bits::PackedBits,
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
    train_input_bits: &ram_core::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &ram_core::packed_bits::PackedBits,
    num_eval: usize,
    total_input_bits: usize,
    settings: ram_core::neuron_memory::EvalSettings,
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
        ram_core::neuron_memory::pack_packed_to_u64(train_input_bits);

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
    train_input_bits: &ram_core::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &ram_core::packed_bits::PackedBits,
    num_eval: usize,
    total_input_bits: usize,
    settings: ram_core::neuron_memory::EvalSettings,
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
                ram_core::neuron_memory::pack_packed_to_u64(train_input_bits);
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

    let (packed_eval, eval_words) = ram_core::neuron_memory::pack_packed_to_u64(eval_input_bits);

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
/// Returns (eval_scores, train_scores, val_scores) — Vec<f64> of length num_eval
/// and num_train respectively; val_scores is Some(Vec<f64> of length num_val)
/// when `val_input_bits` is provided (Protocol v2: 3-way splits score the val
/// partition from the same trained memory), None otherwise.
/// Single-cluster (binary IDS) mode only.
#[allow(clippy::too_many_arguments)]
pub fn train_and_score_eval_and_train(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &ram_core::packed_bits::PackedBits,
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &ram_core::packed_bits::PackedBits,
    num_eval: usize,
    val_input_bits: Option<&ram_core::packed_bits::PackedBits>,
    num_val: usize,
    total_input_bits: usize,
    settings: ram_core::neuron_memory::EvalSettings,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
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
        ram_core::neuron_memory::pack_packed_to_u64(train_input_bits);

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
    let (packed_eval, eval_words) = ram_core::neuron_memory::pack_packed_to_u64(eval_input_bits);
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

    // Score val set (Protocol v2: same trained memory, packed like the eval set)
    let val_scores: Option<Vec<f64>> = val_input_bits.map(|val_bits| {
        let (packed_val, val_words) = ram_core::neuron_memory::pack_packed_to_u64(val_bits);
        let val_all_scores = compute_per_example_scores(
            &export, val_bits, &packed_val, val_words,
            num_val, num_clusters, total_input_bits, empty_value,
            memory_mode, metal.as_deref(), sparse_metal.as_deref(),
        );
        val_all_scores.iter().map(|s| s[0]).collect()
    });

    (eval_scores, train_scores, val_scores)
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
