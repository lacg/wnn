//! Batched hybrid CPU+GPU genome evaluation (the GA hot path).
//!
//! Split out of adaptive/eval.rs (D3 follow-up, 11/06/2026).

use super::*;

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

/// Loop-invariant per-genome offset tables into the flat genome arrays.
/// Seam 1 of the eval_hybrid decomposition (16/06/2026): pure arithmetic pulled
/// out of the impl prologue so the offset bookkeeping is a named, testable unit.
struct GenomeOffsets {
    /// genome g's per-neuron-bits slice is genomes_bits_flat[bpn[g]..bpn[g+1]];
    /// length num_genomes+1.
    bpn_offsets: Vec<usize>,
    /// genome g's connections start at conn_offsets[g] in genomes_connections_flat.
    conn_offsets: Vec<usize>,
    /// genome g's connection count (= Σ per-neuron bits over its neurons).
    conn_sizes: Vec<usize>,
}

/// Compute the per-genome offset tables. genomes_bits_flat has total_neurons
/// entries per genome (per-neuron bits), NOT num_clusters; conn_size = Σ of those
/// bits. Moved verbatim from the impl — same order, same values.
fn compute_genome_offsets(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    num_genomes: usize,
    num_clusters: usize,
) -> GenomeOffsets {
    let mut bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        bpn_offsets.push(bpn_offsets.last().unwrap() + total_neurons);
    }

    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = bpn_offsets[genome_idx];
        let bpn_end = bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    GenomeOffsets { bpn_offsets, conn_offsets, conn_sizes }
}

/// Loop-invariant inputs shared by every per-genome CPU train. Seam (CPU-fallback)
/// of the eval_hybrid decomposition: bundles the ~20 references the per-genome path
/// used to capture as a closure so it can live as a free function. Built ONCE before
/// the batch loop; all fields are immutable shared borrows ⇒ Sync ⇒ safe to pass to
/// the rayon par_iter that drives the per-genome path.
struct HybridBatchConfig<'a> {
    genomes_bits_flat: &'a [usize],
    genomes_neurons_flat: &'a [usize],
    genomes_connections_flat: &'a [i64],
    num_clusters: usize,
    total_input_bits: usize,
    use_provided_connections: bool,
    memory_mode: u8,
    train_input_bits: &'a crate::packed_bits::PackedBits,
    train_targets: &'a [i64],
    train_negatives: &'a [i64],
    num_train: usize,
    num_negatives: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&'a [u32]>,
    packed_train_input: &'a [u64],
    words_per_example: usize,
    genome_bpn_offsets: &'a [usize],
    conn_offsets: &'a [usize],
    conn_sizes: &'a [usize],
}

/// Train ONE genome on the CPU and export it for eval: Path-2 marker training
/// (MarkerHashTable / batched_train_offspring → GenomeExport) with the original
/// dense single-shot / chunked path as the fallback on error. Moved VERBATIM from
/// the former `cpu_one_genome` closure (captures → cfg.<field>); golden-test-covered
/// via the dense path. Returns (genome_idx, export, threshold=None — calibrated later).
fn train_one_genome_cpu(
    cfg: &HybridBatchConfig,
    genome_idx: usize,
) -> (usize, GenomeExport, Option<f64>) {
    // Cooperative SIGTERM cancellation (added 31/05/2026): poll at the start of
    // each per-genome rayon worker callback. If set, skip the (expensive) training
    // work and return an empty GenomeExport — the outer loop detects the partial
    // batch and short-circuits further iterations.
    if crate::cancel::check_cancel() {
        return (genome_idx, GenomeExport::empty(), None);
    }

    // Get this genome's config (per-neuron bits + per-cluster neurons)
    let genome_offset = genome_idx * cfg.num_clusters;
    let neurons_per_cluster = &cfg.genomes_neurons_flat[genome_offset..genome_offset + cfg.num_clusters];

    // Extract per-neuron bits for this genome
    let bpn_start = cfg.genome_bpn_offsets[genome_idx];
    let bpn_end = cfg.genome_bpn_offsets[genome_idx + 1];
    let per_neuron_bits = &cfg.genomes_bits_flat[bpn_start..bpn_end];

    // Compute per-cluster max bits (for build_groups and GPU dispatch)
    let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

    // Build neuron metadata for per-neuron training
    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);

    // Build config groups for THIS genome (using per-cluster max bits)
    let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

    // Build cluster-to-group mapping for THIS genome
    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); cfg.num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    // Create memory for THIS genome
    let mut memories: Vec<GroupMemory> = groups.iter()
        .map(|g| GroupMemory::new(g.total_neurons(), g.bits, cfg.memory_mode))
        .collect();

    // Get original per-neuron connections for this genome
    let original_connections: Vec<i64> = if cfg.use_provided_connections {
        let conn_offset = cfg.conn_offsets[genome_idx];
        let conn_size = cfg.conn_sizes[genome_idx];
        cfg.genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec()
    } else {
        // Generate random per-neuron connections
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64((genome_idx * 12345) as u64);
        let total_conn: usize = per_neuron_bits.iter().sum();
        let mut conns = Vec::with_capacity(total_conn);
        for _ in 0..total_conn {
            conns.push(rng.gen_range(0..cfg.total_input_bits as i64));
        }
        conns
    };

    // Path 2 migration: train via Option B (MarkerHashTable /
    // batched_train_offspring) → GenomeExport directly. No separate
    // compute_addresses dispatch, no u32 truncation guard, native u64 keys.
    // Falls back to the dense chunked path on error (memory budget, kernel failure).
    let export = match train_single_via_marker(
        per_neuron_bits,
        neurons_per_cluster,
        &original_connections,
        cfg.num_clusters,
        cfg.train_input_bits,
        cfg.train_targets,
        cfg.train_negatives,
        cfg.num_train,
        cfg.num_negatives,
        cfg.total_input_bits,
        cfg.empty_value,
        cfg.neuron_sample_rate,
        cfg.rng_seed.wrapping_add(genome_idx as u64),
        cfg.class_weights,
    ) {
        Ok(e) => e,
        Err(reason) => {
            eprintln!(
                "[PATH2_FALLBACK] evaluate_genomes_parallel_hybrid g={} → dense: {}",
                genome_idx, reason
            );
            // Fallback: original dense path (single-shot + chunked variants)
            let total_neurons = per_neuron_bits.len();
            let total_addresses_estimate = total_neurons.saturating_mul(cfg.num_train);

            if total_addresses_estimate <= MAX_GPU_ADDRESSES {
                let gpu_addresses = try_gpu_addresses_adaptive(
                    cfg.packed_train_input,
                    cfg.words_per_example,
                    per_neuron_bits,
                    &neuron_conn_offsets,
                    &original_connections,
                    cfg.num_train,
                );
                train_genome_in_slot(
                    &mut memories,
                    &groups,
                    &original_connections,
                    per_neuron_bits,
                    &cluster_neuron_starts,
                    &neuron_conn_offsets,
                    &cluster_to_group,
                    cfg.train_input_bits,
                    cfg.train_targets,
                    cfg.train_negatives,
                    cfg.num_train,
                    cfg.num_negatives,
                    cfg.total_input_bits,
                    gpu_addresses.as_deref(),
                    cfg.neuron_sample_rate,
                    cfg.rng_seed.wrapping_add(genome_idx as u64),
                    cfg.memory_mode,
                    cfg.class_weights,
                    true,
                );
            } else {
                // OI orchestration brackets the entire chunked loop:
                // counters accumulate across chunks, then commit once.
                let oi_chunked = crate::neuron_memory::order_independent_training_enabled()
                    && cfg.memory_mode == crate::neuron_memory::MODE_QUAD_WEIGHTED;
                if oi_chunked {
                    for m in memories.iter_mut() { m.init_oi_counters(); }
                }
                let chunk_size = (MAX_GPU_ADDRESSES / total_neurons.max(1)).max(1);
                let mut chunk_start = 0;
                while chunk_start < cfg.num_train {
                    let chunk_end = (chunk_start + chunk_size).min(cfg.num_train);
                    let chunk_len = chunk_end - chunk_start;
                    let chunk_packed = &cfg.packed_train_input[
                        chunk_start * cfg.words_per_example..chunk_end * cfg.words_per_example
                    ];
                    let chunk_addresses = try_gpu_addresses_for_chunk(
                        chunk_packed,
                        cfg.words_per_example,
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
                        cfg.train_input_bits,
                        cfg.train_targets,
                        cfg.train_negatives,
                        cfg.num_train,
                        cfg.num_negatives,
                        cfg.total_input_bits,
                        chunk_addresses.as_deref(),
                        chunk_start..chunk_end,
                        chunk_len,
                        cfg.neuron_sample_rate,
                        cfg.rng_seed.wrapping_add(genome_idx as u64),
                        cfg.memory_mode,
                        cfg.class_weights,
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

    // Seam 1: per-genome offset tables (genome_bpn_offsets / conn_offsets /
    // conn_sizes) precomputed in one pass. Destructured into the same local
    // names the rest of the body already uses → zero downstream changes.
    let GenomeOffsets { bpn_offsets: genome_bpn_offsets, conn_offsets, conn_sizes } =
        compute_genome_offsets(genomes_bits_flat, genomes_neurons_flat, num_genomes, num_clusters);

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

    // Connections are provided when the flat array is non-empty (offsets/sizes
    // already computed above in compute_genome_offsets).
    let use_provided_connections = !genomes_connections_flat.is_empty();

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

    // Loop-invariant config for the per-genome CPU train (seam: CPU-fallback).
    // Built once; the batch loop's `cpu_one_genome` closure just forwards to
    // train_one_genome_cpu(&cfg, genome_idx).
    let cfg = HybridBatchConfig {
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_clusters,
        total_input_bits,
        use_provided_connections,
        memory_mode,
        train_input_bits,
        train_targets,
        train_negatives,
        num_train,
        num_negatives,
        empty_value,
        neuron_sample_rate,
        rng_seed,
        class_weights,
        packed_train_input: &packed_train_input,
        words_per_example,
        genome_bpn_offsets: &genome_bpn_offsets,
        conn_offsets: &conn_offsets,
        conn_sizes: &conn_sizes,
    };

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
        // Existing per-genome par_iter (default path). The per-genome CPU train
        // body now lives in the free fn train_one_genome_cpu(&cfg, genome_idx);
        // this thin closure adapts the batch-local index used by the call sites
        // below (and by the B12 hybrid CPU+GPU split, which runs it on a SUBSET).
        let cpu_one_genome = |local_idx: usize| train_one_genome_cpu(&cfg, batch_start + local_idx);

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
