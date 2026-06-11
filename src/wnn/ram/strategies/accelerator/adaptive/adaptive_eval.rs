//! Training-time adaptation (*genesis) evaluation path.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

use super::*;

// =============================================================================
// Training-Time Adaptation for Adaptive Path (*genesis)
// =============================================================================

/// Result from adaptive evaluation, including (possibly modified) genome parameters.
pub struct AdaptiveGenomeResult {
    pub ce: f64,
    pub accuracy: f64,
    pub f1_macro: f64,
    pub fpr: f64,
    pub adapted_bits: Vec<usize>,
    pub adapted_neurons: Vec<usize>,
    pub adapted_connections: Vec<i64>,
    pub pruned: usize,
    pub grown: usize,
    pub added: usize,
    pub removed: usize,
    pub rewired: usize,
}

/// Compute NeuronStats from adaptive training state (GroupMemory-based).
///
/// For each neuron: fill_rate (from memory), connection_entropy (from input bits),
/// and error_rate (forward pass on sample checking target/negative agreement).
pub(crate) fn compute_neuron_stats_adaptive(
    per_neuron_bits: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    export: &GenomeExport,
    cluster_to_group: &[(usize, usize)],
    cluster_neuron_starts: &[usize],
    neuron_conn_offsets: &[usize],
    packed_input: &[u64],
    words_per_example: usize,
    train_targets: &[i64],
    num_train: usize,
    num_clusters: usize,
    config: &crate::adaptation::AdaptationConfig,
    memory_mode: u8,
    empty_value: f32,
) -> Vec<crate::adaptation::NeuronStats> {
    let total_neurons: usize = neurons_per_cluster.iter().sum();

    // Sample examples for stats
    let sample_indices = crate::adaptation::build_sample_indices(num_train, config.stats_sample_size, 42);
    let n_sample = sample_indices.len();

    // Precompute bit ones for connection entropy
    let bit_ones = crate::adaptation::precompute_bit_ones(
        packed_input, words_per_example, &sample_indices, config.total_input_bits,
    );

    // Build neuron-to-cluster mapping
    let mut neuron_to_cluster = vec![0usize; total_neurons];
    for cluster in 0..num_clusters {
        let base = cluster_neuron_starts[cluster];
        for local_n in 0..neurons_per_cluster[cluster] {
            neuron_to_cluster[base + local_n] = cluster;
        }
    }

    // Parallel per-neuron stats computation
    (0..total_neurons).into_par_iter().map(|global_n| {
        let n_bits = per_neuron_bits[global_n];
        let conn_start = neuron_conn_offsets[global_n];
        let neuron_conns = &connections[conn_start..conn_start + n_bits];
        let cluster = neuron_to_cluster[global_n];

        // Connection entropy (from precomputed bit counts)
        let conn_entropy: Vec<f32> = neuron_conns.iter().map(|&conn_idx| {
            let ones = bit_ones[conn_idx as usize];
            let p = ones as f32 / n_sample.max(1) as f32;
            if p > 0.0 && p < 1.0 {
                -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
            } else {
                0.0
            }
        }).collect();

        // Fill rate from GenomeExport (sparse or dense, handled uniformly).
        let (group_idx, local_cluster) = cluster_to_group[cluster];
        let local_n = global_n - cluster_neuron_starts[cluster];
        let neuron_in_group = local_cluster * export.groups[group_idx].neurons + local_n;
        let fill_rate = export.neuron_fill_rate(group_idx, neuron_in_group, n_bits);

        // Error rate: check neuron output against expected label on sampled examples.
        // For examples where this neuron's cluster is the target: expect TRUE.
        // For all other examples: expect FALSE (this neuron's cluster is a negative).
        let mut errors = 0u32;
        let mut seen = 0u32;
        for &ex in &sample_indices {
            let target = train_targets[ex] as usize;
            let expected_true = target == cluster;

            // Compute address from packed input
            let mut addr = 0usize;
            for (bit_i, &conn_idx) in neuron_conns.iter().enumerate() {
                let idx = conn_idx as usize;
                let word_idx = ex * words_per_example + idx / 64;
                let bit_idx = idx % 64;
                if packed_input[word_idx] >> bit_idx & 1 == 1 {
                    addr |= 1 << (n_bits - 1 - bit_i);
                }
            }

            let cell = export.read_cell_at(group_idx, neuron_in_group, addr as u64);
            let weight = cell_to_weight(cell, memory_mode, empty_value);
            let predicted_true = weight >= 0.5;

            if predicted_true != expected_true {
                errors += 1;
            }
            seen += 1;
        }
        let error_rate = if seen > 0 { errors as f32 / seen as f32 } else { 0.0 };

        crate::adaptation::NeuronStats { fill_rate, error_rate, connection_entropy: conn_entropy }
    }).collect()
}

/// Compute ClusterStats from adaptive training state.
///
/// For each cluster: error_rate (majority vote vs target), mean_fill_rate,
/// neuron_uniqueness (disagrees with majority), neuron_accuracy (matches target).
pub(crate) fn compute_cluster_stats_adaptive(
    neuron_stats: &[crate::adaptation::NeuronStats],
    per_neuron_bits: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    export: &GenomeExport,
    cluster_to_group: &[(usize, usize)],
    cluster_neuron_starts: &[usize],
    neuron_conn_offsets: &[usize],
    packed_input: &[u64],
    words_per_example: usize,
    train_targets: &[i64],
    num_train: usize,
    num_clusters: usize,
    config: &crate::adaptation::AdaptationConfig,
    memory_mode: u8,
    empty_value: f32,
) -> Vec<crate::adaptation::ClusterStats> {
    let sample_indices = crate::adaptation::build_sample_indices(num_train, config.stats_sample_size, 43);
    let n_sample = sample_indices.len();

    (0..num_clusters).into_par_iter().map(|cluster| {
        let n_neurons = neurons_per_cluster[cluster];
        let neuron_base = cluster_neuron_starts[cluster];
        let (group_idx, local_cluster) = cluster_to_group[cluster];
        let group = &export.groups[group_idx];

        // Mean fill rate from pre-computed neuron stats
        let mean_fill = if n_neurons > 0 {
            (0..n_neurons).map(|i| neuron_stats[neuron_base + i].fill_rate).sum::<f32>() / n_neurons as f32
        } else {
            0.0
        };

        // Per-example: compute neuron votes, check cluster majority
        let mut cluster_errors = 0u32;
        let mut neuron_votes: Vec<Vec<bool>> = vec![Vec::with_capacity(n_sample); n_neurons];

        for &ex in &sample_indices {
            let target = train_targets[ex] as usize;
            let target_is_this_cluster = target == cluster;
            let mut votes_true = 0u32;

            for local_n in 0..n_neurons {
                let global_n = neuron_base + local_n;
                let n_bits = per_neuron_bits[global_n];
                let conn_start = neuron_conn_offsets[global_n];
                let neuron_conns = &connections[conn_start..conn_start + n_bits];
                let neuron_in_group = local_cluster * group.neurons + local_n;

                let mut addr = 0usize;
                for (bit_i, &conn_idx) in neuron_conns.iter().enumerate() {
                    let idx = conn_idx as usize;
                    let word_idx = ex * words_per_example + idx / 64;
                    let bit_idx = idx % 64;
                    if packed_input[word_idx] >> bit_idx & 1 == 1 {
                        addr |= 1 << (n_bits - 1 - bit_i);
                    }
                }

                let cell = export.read_cell_at(group_idx, neuron_in_group, addr as u64);
                let weight = cell_to_weight(cell, memory_mode, empty_value);
                let is_true = weight >= 0.5;
                if is_true { votes_true += 1; }
                neuron_votes[local_n].push(is_true);
            }

            // Majority vote for cluster
            let majority_true = votes_true > (n_neurons as u32 / 2);
            if majority_true != target_is_this_cluster { cluster_errors += 1; }
        }

        let cluster_error_rate = if n_sample > 0 {
            cluster_errors as f32 / n_sample as f32
        } else { 0.0 };

        // Precompute majority per sample
        let majority_votes: Vec<bool> = (0..n_sample).map(|s| {
            let v: u32 = (0..n_neurons).map(|j| neuron_votes[j][s] as u32).sum();
            v > (n_neurons as u32 / 2)
        }).collect();

        // Neuron uniqueness + accuracy
        let mut uniqueness = vec![0.0f32; n_neurons];
        let mut accuracy = vec![0.0f32; n_neurons];

        if n_sample > 0 {
            for local_n in 0..n_neurons {
                let mut disagree = 0u32;
                let mut correct = 0u32;
                for s in 0..n_sample {
                    let ex = sample_indices[s];
                    let my_vote = neuron_votes[local_n][s];
                    let target_is_this_cluster = train_targets[ex] as usize == cluster;
                    if my_vote != majority_votes[s] { disagree += 1; }
                    if my_vote == target_is_this_cluster { correct += 1; }
                }
                uniqueness[local_n] = disagree as f32 / n_sample as f32;
                accuracy[local_n] = correct as f32 / n_sample as f32;
            }
        }

        crate::adaptation::ClusterStats {
            error_rate: cluster_error_rate,
            mean_fill_rate: mean_fill,
            neuron_uniqueness: uniqueness,
            neuron_accuracy: accuracy,
        }
    }).collect()
}

/// Axonogenesis for the adaptive (GroupMemory) path.
///
/// Rewires low-value connections to high-entropy input bits using the same 3-stage
/// algorithm as `adaptation::axonogenesis_pass`, but reads from GroupMemory instead
/// of ClusterStorage and uses class-label targets instead of target_bits.
///
/// Returns the number of rewired connections.
#[allow(clippy::too_many_arguments)]
pub(crate) fn axonogenesis_pass_adaptive(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &mut Vec<i64>,
    neuron_stats: &[crate::adaptation::NeuronStats],
    export: &GenomeExport,
    cluster_to_group: &[(usize, usize)],
    _cluster_neuron_starts: &[usize],
    config: &crate::adaptation::AdaptationConfig,
    packed_input: &[u64],
    words_per_example: usize,
    train_targets: &[i64],
    num_examples: usize,
    _num_clusters: usize,
    _memory_mode: u8,
    _empty_value: f32,
    rate: f32,
    rng: &mut impl rand::Rng,
) -> usize {
    use crate::adaptation::{build_sample_indices, precompute_bit_ones};

    let total_neurons = bits_per_neuron.len();
    let mut rewired = 0usize;
    let max_candidates = 20usize;

    // Build connection offsets per neuron
    let mut conn_offsets = vec![0usize];
    for &b in bits_per_neuron.iter() {
        conn_offsets.push(conn_offsets.last().unwrap() + b);
    }

    // Build neuron → cluster mapping
    let mut neuron_cluster = Vec::with_capacity(total_neurons);
    let mut neuron_local_idx = Vec::with_capacity(total_neurons);
    for (c, &nc) in neurons_per_cluster.iter().enumerate() {
        for local in 0..nc {
            neuron_cluster.push(c);
            neuron_local_idx.push(local);
        }
    }

    // Stage 1: Pre-compute marginal entropy for all input bits
    let sample_indices = build_sample_indices(num_examples, config.stats_sample_size, 77);
    let n_sample = sample_indices.len();
    let bit_ones = precompute_bit_ones(packed_input, words_per_example, &sample_indices, config.total_input_bits);
    let bit_entropy: Vec<f32> = bit_ones.iter().map(|&ones| {
        let p = ones as f32 / n_sample.max(1) as f32;
        if p > 0.0 && p < 1.0 {
            -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
        } else {
            0.0
        }
    }).collect();

    let entropy_floor = 0.1f32;

    // Pre-compute bit values for sampled examples (for Stage 3 redundancy)
    let mut bit_vals: Vec<Option<Vec<u8>>> = vec![None; config.total_input_bits];
    for bit_idx in 0..config.total_input_bits {
        if bit_entropy[bit_idx] >= entropy_floor {
            let mut vals = Vec::with_capacity(n_sample);
            for &ex in &sample_indices {
                let row_start = ex * words_per_example;
                let word_idx = bit_idx / 64;
                let bit_pos = bit_idx % 64;
                let bit = ((packed_input[row_start + word_idx] >> bit_pos) & 1) as u8;
                vals.push(bit);
            }
            bit_vals[bit_idx] = Some(vals);
        }
    }

    // Process each neuron
    for n in 0..total_neurons {
        let n_bits = bits_per_neuron[n];
        if n_bits < 2 { continue; }

        let stats = &neuron_stats[n];
        let conn_start = conn_offsets[n];
        let cluster = neuron_cluster[n];
        let local_n = neuron_local_idx[n];

        // Resolve memory location for this neuron via GenomeExport
        let (group_idx, local_cluster) = cluster_to_group[cluster];
        let group = &export.groups[group_idx];
        let neuron_in_group = local_cluster * group.neurons + local_n;

        // Stage 1a: Find weak connections (entropy < median × threshold)
        let median_ent = crate::adaptation::median_of(&stats.connection_entropy);
        let rewire_threshold = median_ent * config.axon_entropy_threshold;

        let mut weak_conns: Vec<(usize, f32)> = stats.connection_entropy.iter()
            .enumerate()
            .filter(|(_, &ent)| ent < rewire_threshold)
            .map(|(idx, &ent)| (idx, ent))
            .collect();
        weak_conns.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        weak_conns.truncate(config.axon_rewire_count);

        if weak_conns.is_empty() { continue; }

        // Stage 1b: Get candidate unused bits sorted by marginal entropy (desc)
        let used: std::collections::HashSet<i64> = connections[conn_start..conn_start + n_bits]
            .iter().copied().collect();

        let mut candidates: Vec<(i64, f32)> = (0..config.total_input_bits)
            .filter(|&b| !used.contains(&(b as i64)) && bit_entropy[b] >= entropy_floor)
            .map(|b| (b as i64, bit_entropy[b]))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_candidates);

        if candidates.is_empty() { continue; }

        // Pre-compute base addresses for this neuron on sampled examples
        let neuron_conns = &connections[conn_start..conn_start + n_bits];
        let mut base_addresses: Vec<usize> = Vec::with_capacity(n_sample);
        for &ex in &sample_indices {
            let row_start = ex * words_per_example;
            let mut addr = 0usize;
            for (i, &conn_idx) in neuron_conns.iter().enumerate() {
                let idx = conn_idx as usize;
                let bit = (packed_input[row_start + idx / 64] >> (idx % 64)) & 1;
                addr |= (bit as usize) << (n_bits - 1 - i);
            }
            base_addresses.push(addr);
        }

        // Stage 3 prep: existing connection bit values
        let existing_bit_vals: Vec<Option<&Vec<u8>>> = (0..n_bits)
            .map(|i| {
                let conn = neuron_conns[i] as usize;
                if conn < config.total_input_bits { bit_vals[conn].as_ref() } else { None }
            })
            .collect();

        // For each weak connection, find the best replacement via accuracy delta + redundancy
        for &(local_idx, _old_entropy) in &weak_conns {
            if rng.gen::<f32>() >= rate { continue; }

            let old_conn = connections[conn_start + local_idx];
            let mut best_candidate: Option<(i64, f32)> = None;

            for &(cand_conn, _cand_entropy) in &candidates {
                if connections[conn_start..conn_start + n_bits].contains(&cand_conn) { continue; }

                // Stage 2: Accuracy delta — measure per-neuron accuracy from swapping
                let mut delta = 0i32;
                for (si, &ex) in sample_indices.iter().enumerate() {
                    let old_addr = base_addresses[si];
                    // Target: is this cluster the correct one for this example?
                    let target_is_this = train_targets[ex] as usize == cluster;

                    // Old prediction (current connection)
                    let old_cell = export.read_cell_at(group_idx, neuron_in_group, old_addr as u64);
                    let old_weight = crate::neuron_memory::QUAD_WEIGHTS[old_cell.clamp(0, 3) as usize];
                    let old_correct = (old_weight >= 0.5) == target_is_this;

                    // Flip address bit at local_idx if old/new input bits differ
                    let row_start = ex * words_per_example;
                    let old_bit = (packed_input[row_start + old_conn as usize / 64]
                        >> (old_conn as usize % 64)) & 1;
                    let new_bit = (packed_input[row_start + cand_conn as usize / 64]
                        >> (cand_conn as usize % 64)) & 1;
                    let new_addr = if old_bit != new_bit {
                        old_addr ^ (1 << (n_bits - 1 - local_idx))
                    } else {
                        old_addr
                    };

                    // New prediction
                    let new_cell = export.read_cell_at(group_idx, neuron_in_group, new_addr as u64);
                    let new_weight = crate::neuron_memory::QUAD_WEIGHTS[new_cell.clamp(0, 3) as usize];
                    let new_correct = (new_weight >= 0.5) == target_is_this;

                    delta += new_correct as i32 - old_correct as i32;
                }

                if delta <= 0 { continue; }

                // Stage 3: Redundancy penalty (Jaccard similarity)
                let mut max_jaccard = 0.0f32;
                if let Some(cand_vals) = &bit_vals[cand_conn as usize] {
                    for (i, existing) in existing_bit_vals.iter().enumerate() {
                        if i == local_idx { continue; }
                        if let Some(existing_vals) = existing {
                            let mut both = 0u32;
                            let mut either = 0u32;
                            for si in 0..n_sample {
                                let a = cand_vals[si];
                                let b = existing_vals[si];
                                both += (a & b) as u32;
                                either += (a | b) as u32;
                            }
                            let jaccard = if either > 0 { both as f32 / either as f32 } else { 0.0 };
                            if jaccard > max_jaccard { max_jaccard = jaccard; }
                        }
                    }
                }

                let redundancy_weight = 0.5f32;
                let adjusted_score = delta as f32 * (1.0 - redundancy_weight * max_jaccard);

                if best_candidate.is_none() || adjusted_score > best_candidate.unwrap().1 {
                    best_candidate = Some((cand_conn, adjusted_score));
                }
            }

            if let Some((new_conn, score)) = best_candidate {
                if score > 0.0 {
                    connections[conn_start + local_idx] = new_conn;
                    rewired += 1;
                }
            }
        }
    }

    if rewired > 0 {
        eprintln!("[Adapt] Axonogenesis: rewired={} (rate={:.2})", rewired, rate);
    }

    rewired
}

/// Evaluate genomes with training-time adaptation (synaptogenesis + neurogenesis + axonogenesis).
///
/// Same interface as `evaluate_genomes_parallel_hybrid` but:
/// 1. After training, computes stats from trained memory
/// 2. Applies adaptation passes (modifying genome architecture)
/// 3. Retrains if genome was modified
/// 4. Returns both scores AND adapted genome parameters
///
/// If adaptation_rate is 0.0 for the given generation (warmup/cooldown), falls
/// through to the standard path with no overhead.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genomes_parallel_hybrid_adaptive(
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
    adapt_config: &crate::adaptation::AdaptationConfig,
    generation: usize,
) -> Vec<AdaptiveGenomeResult> {
    let empty_value = settings.empty_value;
    let rate = crate::adaptation::adaptation_rate(generation, adapt_config);

    // If rate is 0 (warmup/cooldown), use standard path and wrap results
    if rate == 0.0 || (!adapt_config.synaptogenesis_enabled && !adapt_config.neurogenesis_enabled) {
        let standard = evaluate_genomes_parallel_hybrid(
            genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
            num_genomes, num_clusters,
            train_input_bits, train_targets, train_negatives,
            num_train, num_negatives,
            eval_input_bits, eval_targets, num_eval,
            total_input_bits, settings, neuron_sample_rate, rng_seed,
            None, // class_weights: adaptive path doesn't use class balancing
        );

        // Rebuild per-genome bits/neurons/connections from flat arrays
        let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
        genome_bpn_offsets.push(0);
        for g in 0..num_genomes {
            let nc_base = g * num_clusters;
            let total_n: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
            genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_n);
        }
        let mut conn_offset = 0usize;

        return standard.into_iter().enumerate().map(|(g, (ce, acc, f1, fpr, _threshold, _ms))| {
            let bpn_start = genome_bpn_offsets[g];
            let bpn_end = genome_bpn_offsets[g + 1];
            let bits = genomes_bits_flat[bpn_start..bpn_end].to_vec();
            let neurons = genomes_neurons_flat[g * num_clusters..(g + 1) * num_clusters].to_vec();
            let conn_size: usize = bits.iter().sum();
            let conns = genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec();
            conn_offset += conn_size;
            AdaptiveGenomeResult {
                ce, accuracy: acc, f1_macro: f1, fpr,
                adapted_bits: bits, adapted_neurons: neurons, adapted_connections: conns,
                pruned: 0, grown: 0, added: 0, removed: 0, rewired: 0,
            }
        }).collect();
    }

    let memory_mode = settings.memory_mode;
    if num_genomes == 0 {
        return vec![];
    }

    // Pre-compute genome offsets (same as evaluate_genomes_parallel_hybrid)
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    let use_provided_connections = !genomes_connections_flat.is_empty();
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

    // Pack input once
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_packed_to_u64(train_input_bits);

    let eval_data = Arc::new(EvalData {
        eval_input_bits: eval_input_bits.clone(),
        eval_targets: eval_targets.to_vec(),
        num_eval,
        num_clusters,
        total_input_bits,
        settings,
    });
    let eval_worker = get_eval_worker();

    // Process genomes: train → adapt → retrain if needed → export → evaluate
    // Use sequential processing to keep memory bounded (adaptation keeps memory alive longer)
    let batch_size = std::env::var("WNN_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let first_neurons = &genomes_neurons_flat[0..num_clusters];
            let first_per_neuron_bits = &genomes_bits_flat[0..genome_bpn_offsets[1]];
            let first_bits_per_cluster = per_cluster_max_bits(first_per_neuron_bits, first_neurons);
            let budget_gb = get_available_memory_gb() * 0.4; // 40% budget (adaptation keeps memory longer)
            let cpu_cores = rayon::current_num_threads();
            let (_, computed_batch) = calculate_pool_size(
                &first_bits_per_cluster, first_neurons, num_clusters, budget_gb, cpu_cores,
            );
            computed_batch
        });

    let mut all_results: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::with_capacity(num_genomes);
    let mut all_adapted: Vec<(usize, Vec<usize>, Vec<usize>, Vec<i64>, usize, usize, usize, usize, usize)> =
        Vec::with_capacity(num_genomes);
    let num_batches = (num_genomes + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        // Cooperative SIGTERM cancellation (added 31/05/2026): poll at the
        // batch boundary in the adaptive (Lamarckian) eval path. Same shape
        // as the plain hybrid: leave the loop early on cancel; downstream
        // result collation handles a short results vec gracefully.
        if crate::cancel::check_cancel() {
            break;
        }
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_genomes);
        let current_batch_size = batch_end - batch_start;

        // Phase 1: Train all genomes via Option B (Path 2). Each genome's
        // trained memory state lives in its GenomeExport — no separate
        // Vec<GroupMemory>. Stats functions (compute_neuron_stats_adaptive,
        // compute_cluster_stats_adaptive, axonogenesis_pass_adaptive) now
        // consume GenomeExport directly via read_cell_at / neuron_fill_rate.
        struct TrainedState {
            export: GenomeExport,
            cluster_to_group: Vec<(usize, usize)>,
            cluster_neuron_starts: Vec<usize>,
            neuron_conn_offsets: Vec<usize>,
            per_neuron_bits: Vec<usize>,
            neurons_per_cluster: Vec<usize>,
            connections: Vec<i64>,
            genome_idx: usize,
        }

        let trained_states: Vec<TrainedState> = (0..current_batch_size)
            .into_par_iter()
            .map(|local_idx| {
                let genome_idx = batch_start + local_idx;
                let genome_offset = genome_idx * num_clusters;
                let neurons_per_cluster = genomes_neurons_flat[genome_offset..genome_offset + num_clusters].to_vec();
                let bpn_start = genome_bpn_offsets[genome_idx];
                let bpn_end = genome_bpn_offsets[genome_idx + 1];
                let per_neuron_bits = genomes_bits_flat[bpn_start..bpn_end].to_vec();
                let (cluster_neuron_starts, neuron_conn_offsets) =
                    build_neuron_metadata(&per_neuron_bits, &neurons_per_cluster);
                let bits_per_cluster = per_cluster_max_bits(&per_neuron_bits, &neurons_per_cluster);
                let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);
                let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
                for (gi, group) in groups.iter().enumerate() {
                    for (li, &cid) in group.cluster_ids.iter().enumerate() {
                        cluster_to_group[cid] = (gi, li);
                    }
                }
                let connections: Vec<i64> = if use_provided_connections {
                    let co = conn_offsets[genome_idx];
                    let cs = conn_sizes[genome_idx];
                    genomes_connections_flat[co..co + cs].to_vec()
                } else {
                    use rand::{Rng, SeedableRng};
                    let mut rng = rand::rngs::SmallRng::seed_from_u64((genome_idx * 12345) as u64);
                    let total_conn: usize = per_neuron_bits.iter().sum();
                    (0..total_conn).map(|_| rng.gen_range(0..total_input_bits as i64)).collect()
                };
                // Train via Option B; fall back to dense path on Option B error.
                let export = match train_single_via_marker(
                    &per_neuron_bits,
                    &neurons_per_cluster,
                    &connections,
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
                    None, // class_weights: adaptive path doesn't use class balancing
                ) {
                    Ok(e) => e,
                    Err(reason) => {
                        eprintln!(
                            "[PATH2_FALLBACK] evaluate_genomes_parallel_hybrid_adaptive g={} → dense: {}",
                            genome_idx, reason
                        );
                        // Fallback: dense path → GenomeExport via export_genome_for_gpu
                        let mut memories: Vec<GroupMemory> = groups.iter()
                            .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                            .collect();
                        let gpu_addresses = try_gpu_addresses_adaptive(
                            &packed_train_input, words_per_example,
                            &per_neuron_bits, &neuron_conn_offsets, &connections, num_train,
                        );
                        train_genome_in_slot(
                            &mut memories, &groups, &connections, &per_neuron_bits,
                            &cluster_neuron_starts, &neuron_conn_offsets, &cluster_to_group,
                            train_input_bits, train_targets, train_negatives,
                            num_train, num_negatives, total_input_bits,
                            gpu_addresses.as_deref(), neuron_sample_rate,
                            rng_seed.wrapping_add(genome_idx as u64), memory_mode,
                            None, true,
                        );
                        let gpu_connections = reorganize_connections_for_gpu(
                            &connections, &per_neuron_bits, &neurons_per_cluster, &groups,
                        );
                        export_genome_for_gpu(&memories, &groups, &gpu_connections)
                    }
                };
                TrainedState {
                    export, cluster_to_group, cluster_neuron_starts,
                    neuron_conn_offsets, per_neuron_bits, neurons_per_cluster,
                    connections, genome_idx,
                }
            })
            .collect();

        // Phase 2: Stats + Adaptation (per genome, sequential — lightweight)
        struct AdaptedState {
            bits: Vec<usize>,
            neurons: Vec<usize>,
            connections: Vec<i64>,
            changed: bool,
            pruned: usize,
            grown: usize,
            added: usize,
            removed: usize,
            rewired: usize,
            genome_idx: usize,
        }

        let mut adapted_states: Vec<AdaptedState> = Vec::with_capacity(current_batch_size);

        for state in &trained_states {
            let mut adapt_bits = state.per_neuron_bits.clone();
            let mut adapt_neurons = state.neurons_per_cluster.clone();
            let mut adapt_conns = state.connections.clone();
            let initial_neurons = state.neurons_per_cluster.clone();
            let mut cooldowns = vec![0usize; num_clusters];
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
                rng_seed.wrapping_add(state.genome_idx as u64 * 777)
            );
            let mut total_pruned = 0usize;
            let mut total_grown = 0usize;
            let mut total_added = 0usize;
            let mut total_removed = 0usize;
            let mut total_rewired = 0usize;

            // Compute stats from ORIGINAL trained state (not adapted copies).
            // Stats must match the memory architecture — adapted bits/conns may have
            // different sizes than what the memory was trained with.
            let neuron_stats = compute_neuron_stats_adaptive(
                &state.per_neuron_bits, &state.neurons_per_cluster, &state.connections,
                &state.export, &state.cluster_to_group,
                &state.cluster_neuron_starts, &state.neuron_conn_offsets,
                &packed_train_input, words_per_example,
                train_targets, num_train, num_clusters,
                adapt_config, memory_mode, empty_value,
            );

            let cluster_stats = if adapt_config.neurogenesis_enabled {
                Some(compute_cluster_stats_adaptive(
                    &neuron_stats, &state.per_neuron_bits, &state.neurons_per_cluster,
                    &state.connections,
                    &state.export, &state.cluster_to_group,
                    &state.cluster_neuron_starts, &state.neuron_conn_offsets,
                    &packed_train_input, words_per_example,
                    train_targets, num_train, num_clusters,
                    adapt_config, memory_mode, empty_value,
                ))
            } else {
                None
            };

            for _pass in 0..adapt_config.passes_per_eval {
                // Synaptogenesis (prune/grow connections)
                if adapt_config.synaptogenesis_enabled {
                    let (p, g) = crate::adaptation::synaptogenesis_pass(
                        &mut adapt_bits, &mut adapt_conns,
                        &neuron_stats, adapt_config,
                        &packed_train_input, words_per_example, num_train,
                        rate, &mut rng,
                    );
                    total_pruned += p;
                    total_grown += g;
                }

                // Neurogenesis (add/remove neurons)
                if adapt_config.neurogenesis_enabled {
                    let cluster_stats = cluster_stats.as_ref().unwrap();
                    let (a, r) = crate::adaptation::neurogenesis_pass(
                        &mut adapt_bits, &mut adapt_neurons, &mut adapt_conns,
                        &cluster_stats, adapt_config,
                        generation, &mut cooldowns, &initial_neurons,
                        num_train, rate, &mut rng,
                    );
                    total_added += a;
                    total_removed += r;
                }

                // Axonogenesis (MI-guided connection rewiring)
                if adapt_config.axonogenesis_enabled {
                    let rw = axonogenesis_pass_adaptive(
                        &adapt_bits, &adapt_neurons, &mut adapt_conns,
                        &neuron_stats, &state.export,
                        &state.cluster_to_group, &state.cluster_neuron_starts,
                        adapt_config, &packed_train_input, words_per_example,
                        train_targets, num_train, num_clusters,
                        memory_mode, empty_value, rate, &mut rng,
                    );
                    total_rewired += rw;
                }
            }

            let changed = total_pruned > 0 || total_grown > 0 || total_added > 0 || total_removed > 0 || total_rewired > 0;
            adapted_states.push(AdaptedState {
                bits: adapt_bits, neurons: adapt_neurons, connections: adapt_conns,
                changed, pruned: total_pruned, grown: total_grown,
                added: total_added, removed: total_removed, rewired: total_rewired,
                genome_idx: state.genome_idx,
            });
        }

        // Phase 3: Retrain changed genomes + export all (parallel)
        let batch_exports: Vec<(usize, GenomeExport, Option<f64>)> = adapted_states.par_iter()
            .enumerate()
            .map(|(local_idx, adapted)| {
                if adapted.changed {
                    // Retrain with adapted architecture via Option B.
                    let export = train_single_via_marker(
                        &adapted.bits,
                        &adapted.neurons,
                        &adapted.connections,
                        num_clusters,
                        train_input_bits,
                        train_targets,
                        train_negatives,
                        num_train,
                        num_negatives,
                        total_input_bits,
                        empty_value,
                        neuron_sample_rate,
                        rng_seed.wrapping_add(adapted.genome_idx as u64),
                        None, // class_weights
                    ).unwrap_or_else(|reason| {
                        eprintln!(
                            "[PATH2_FALLBACK] adaptive retrain g={} → dense: {}",
                            adapted.genome_idx, reason
                        );
                        // Dense fallback for retrain
                        let bits_per_cluster = per_cluster_max_bits(&adapted.bits, &adapted.neurons);
                        let (cns, nco) = build_neuron_metadata(&adapted.bits, &adapted.neurons);
                        let groups = build_groups(&bits_per_cluster, &adapted.neurons);
                        let mut ctg: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
                        for (gi, group) in groups.iter().enumerate() {
                            for (li, &cid) in group.cluster_ids.iter().enumerate() {
                                ctg[cid] = (gi, li);
                            }
                        }
                        let mut memories: Vec<GroupMemory> = groups.iter()
                            .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                            .collect();
                        let gpu_addresses = try_gpu_addresses_adaptive(
                            &packed_train_input, words_per_example,
                            &adapted.bits, &nco, &adapted.connections, num_train,
                        );
                        train_genome_in_slot(
                            &mut memories, &groups, &adapted.connections, &adapted.bits,
                            &cns, &nco, &ctg,
                            train_input_bits, train_targets, train_negatives,
                            num_train, num_negatives, total_input_bits,
                            gpu_addresses.as_deref(), neuron_sample_rate,
                            rng_seed.wrapping_add(adapted.genome_idx as u64), memory_mode,
                            None, true,
                        );
                        let gpu_conns = reorganize_connections_for_gpu(
                            &adapted.connections, &adapted.bits, &adapted.neurons, &groups,
                        );
                        export_genome_for_gpu(&memories, &groups, &gpu_conns)
                    });
                    (adapted.genome_idx, export, None)
                } else {
                    // Use Phase 1 trained state's already-built GenomeExport.
                    let state = &trained_states[local_idx];
                    (state.genome_idx, state.export.clone(), None)
                }
            })
            .collect();

        // Send to eval worker
        let batch_results = eval_worker.evaluate(batch_exports, Arc::clone(&eval_data));
        all_results.extend(batch_results);

        // Collect adapted genome info
        for adapted in adapted_states {
            all_adapted.push((
                adapted.genome_idx,
                adapted.bits, adapted.neurons, adapted.connections,
                adapted.pruned, adapted.grown, adapted.added, adapted.removed, adapted.rewired,
            ));
        }
    }

    // Sort by genome index and build final results
    all_adapted.sort_by_key(|a| a.0);
    let mut score_map: Vec<(f64, f64, f64, f64)> = vec![(0.0, 0.0, 0.0, 0.0); num_genomes];
    for (idx, ce, acc, f1, fpr, _threshold) in all_results {
        score_map[idx] = (ce, acc, f1, fpr);
    }

    all_adapted.into_iter().map(|(idx, bits, neurons, conns, pruned, grown, added, removed, rewired)| {
        let (ce, acc, f1, fpr) = score_map[idx];
        AdaptiveGenomeResult {
            ce, accuracy: acc, f1_macro: f1, fpr,
            adapted_bits: bits, adapted_neurons: neurons, adapted_connections: conns,
            pruned, grown, added, removed, rewired,
        }
    }).collect()
}
