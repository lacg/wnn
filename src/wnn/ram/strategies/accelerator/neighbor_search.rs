//! Batched Neighbor Search with Threshold Filtering
//!
//! Implements efficient neighbor generation and evaluation entirely in Rust,
//! eliminating Python↔Rust round trips. Generates candidates, evaluates them,
//! filters by accuracy threshold, and logs progress - all in a single call.
//!
//! Key optimization: Instead of Python calling Rust multiple times:
//!   Python: generate 50 → Rust: evaluate → Python: filter → repeat
//! Now it's a single call:
//!   Python: "find 40 candidates above 0.01% threshold, max 200 attempts" → Rust: done

use rand::prelude::*;
use rand::SeedableRng;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use chrono::Local;

/// Configuration for genome mutation.
#[derive(Clone)]
pub struct MutationConfig {
    pub num_clusters: usize,
    /// Which cluster indices can be mutated. None means all clusters.
    /// Supports arbitrary tier optimization (e.g., tier1 = indices 100-499).
    pub mutable_clusters: Option<Vec<usize>>,
    pub min_bits: usize,
    pub max_bits: usize,
    pub min_neurons: usize,
    pub max_neurons: usize,
    pub bits_mutation_rate: f64,
    pub neurons_mutation_rate: f64,
    pub total_input_bits: usize,
}

impl MutationConfig {
    /// Calculate delta range as 10% of (min + max), minimum 1.
    fn bits_delta_max(&self) -> i32 {
        ((0.1 * (self.min_bits + self.max_bits) as f64).round() as i32).max(1)
    }

    fn neurons_delta_max(&self) -> i32 {
        ((0.1 * (self.min_neurons + self.max_neurons) as f64).round() as i32).max(1)
    }
}

/// Result of a successful candidate search.
#[derive(Clone)]
pub struct CandidateResult {
    pub bits_per_cluster: Vec<usize>,
    pub neurons_per_cluster: Vec<usize>,
    pub connections: Vec<i64>,
    pub cross_entropy: f64,
    pub accuracy: f64,
}

/// Result of offspring/neighbor search including counts.
///
/// Provides visibility into how many candidates were evaluated vs how many
/// passed the accuracy threshold. This is critical for understanding whether
/// the threshold is effective or if `return_best_n` fallback is always used.
pub struct OffspringSearchResult {
    /// Candidates returned (passed threshold + fallback if return_best_n)
    pub candidates: Vec<CandidateResult>,
    /// Total candidates evaluated
    pub evaluated: usize,
    /// Candidates that passed accuracy threshold (before fallback)
    pub viable: usize,
}

/// Logger that writes to both stderr and file with proper locking.
/// Uses open/lock/write/close pattern to coordinate with Python writes.
pub struct FileLogger {
    log_path: Option<String>,
}

impl FileLogger {
    pub fn new(log_path: Option<&str>) -> Self {
        Self { log_path: log_path.map(|s| s.to_string()) }
    }

    pub fn log(&mut self, msg: &str) {
        use std::io::Write;
        use fs2::FileExt;
        let timestamp = Local::now().format("%H:%M:%S");
        let line = format!("{} | {}\n", timestamp, msg);

        // Always write to stderr for immediate visibility
        eprint!("{}", line);
        let _ = std::io::stderr().flush();

        // Write to file with proper locking (open fresh, lock, write, close)
        if let Some(ref path) = self.log_path {
            if let Ok(mut file) = OpenOptions::new().append(true).open(path) {
                if file.lock_exclusive().is_ok() {
                    let _ = file.write_all(line.as_bytes());
                    let _ = file.flush();
                    let _ = file.unlock();
                }
                // File closes automatically when dropped
            }
        }
    }
}

/// Type of genome being logged.
#[derive(Clone, Copy)]
pub enum GenomeLogType {
    Initial,    // Initial random population
    EliteCE,    // Elite selected by cross-entropy
    EliteAcc,   // Elite selected by accuracy
    Offspring,  // GA offspring
    Neighbor,   // TS neighbor
    Fallback,   // Best-by-CE fallback (didn't pass accuracy threshold)
}

impl GenomeLogType {
    /// Get the label and type indicator for this log type.
    /// Returns (label, type_indicator) where label is 6 chars and indicator is 4 chars.
    fn format_parts(&self) -> (&'static str, &'static str) {
        match self {
            GenomeLogType::Initial => ("Genome", "Init"),
            GenomeLogType::EliteCE => ("Elite ", "CE  "),
            GenomeLogType::EliteAcc => ("Elite ", "Acc "),
            GenomeLogType::Offspring => ("Genome", "New "),
            GenomeLogType::Neighbor => ("Genome", "Nbr "),
            GenomeLogType::Fallback => ("Genome", "Best"),  // Best-by-CE fallback
        }
    }
}

/// Format a genome log line with consistent padding and alignment.
///
/// All log types are aligned with:
/// - 6-char label (Elite  or Genome)
/// - Position/total with dynamic padding
/// - 4-char type indicator in parentheses (CE, Acc, New, Init, Nbr)
///
/// Returns formatted log string like:
///   "[Gen 001/100] Elite  01/10 (CE  ): CE=10.3417, Acc=0.0180%"
///   "[Gen 001/100] Genome 01/40 (New ): CE=10.3559, Acc=0.0300%"
pub fn format_genome_log(
    generation: usize,
    total_generations: usize,
    log_type: GenomeLogType,
    position: usize,
    total: usize,
    ce: f64,
    acc: f64,
) -> String {
    // Calculate padding widths based on totals
    let gen_width = total_generations.to_string().len();
    let pos_width = total.to_string().len();

    // Get label and type indicator
    let (label, type_ind) = log_type.format_parts();

    // Generation prefix
    let gen_prefix = format!(
        "[Gen {:0width$}/{:0width$}]",
        generation, total_generations, width = gen_width
    );

    format!(
        "{} {} {:0pos_width$}/{} ({}): CE={:.4}, Acc={:.4}%",
        gen_prefix, label, position, total, type_ind, ce, acc * 100.0, pos_width = pos_width
    )
}

/// Format just the generation prefix with dynamic padding.
pub fn format_gen_prefix(generation: usize, total_generations: usize) -> String {
    let gen_width = total_generations.to_string().len();
    format!(
        "[Gen {:0width$}/{:0width$}]",
        generation, total_generations, width = gen_width
    )
}

/// Mutate a genome to create a neighbor.
///
/// Mutations include:
/// - bits_per_cluster: ±delta with mutation_rate probability (only mutable clusters)
/// - neurons_per_cluster: ±delta with mutation_rate probability (only mutable clusters)
/// - connections: adjusted when architecture changes, small perturbations
///
/// mutable_clusters can be any set of indices (e.g., tier1 = 100-499), not just first N.
pub fn mutate_genome(
    bits: &[usize],
    neurons: &[usize],
    connections: &[i64],
    config: &MutationConfig,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let bits_delta_max = config.bits_delta_max();
    let neurons_delta_max = config.neurons_delta_max();

    let mut new_bits = bits.to_vec();
    let mut new_neurons = neurons.to_vec();

    // Track changes for connection adjustment
    let old_bits = bits.to_vec();
    let old_neurons = neurons.to_vec();

    // Mutate architecture - only clusters in mutable_clusters (or all if None)
    let indices: Vec<usize> = match &config.mutable_clusters {
        Some(indices) => indices.clone(),
        None => (0..config.num_clusters).collect(),
    };

    for &i in &indices {
        if i >= config.num_clusters {
            continue; // Safety check
        }

        // Mutate bits
        if rng.gen::<f64>() < config.bits_mutation_rate {
            let delta = rng.gen_range(-bits_delta_max..=bits_delta_max);
            let new_val = (new_bits[i] as i32 + delta)
                .max(config.min_bits as i32)
                .min(config.max_bits as i32);
            new_bits[i] = new_val as usize;
        }

        // Mutate neurons
        if rng.gen::<f64>() < config.neurons_mutation_rate {
            let delta = rng.gen_range(-neurons_delta_max..=neurons_delta_max);
            let new_val = (new_neurons[i] as i32 + delta)
                .max(config.min_neurons as i32)
                .min(config.max_neurons as i32);
            new_neurons[i] = new_val as usize;
        }
    }

    // Adjust connections for architecture changes
    let new_connections = adjust_connections(
        connections,
        &old_bits,
        &old_neurons,
        &new_bits,
        &new_neurons,
        config.total_input_bits,
        rng,
    );

    (new_bits, new_neurons, new_connections)
}

/// Adjust connections when architecture changes.
fn adjust_connections(
    old_connections: &[i64],
    old_bits: &[usize],
    old_neurons: &[usize],
    new_bits: &[usize],
    new_neurons: &[usize],
    total_input_bits: usize,
    rng: &mut impl Rng,
) -> Vec<i64> {
    // If no existing connections, keep them empty (bits-only or neurons-only phase)
    if old_connections.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut old_idx = 0;

    for cluster_idx in 0..new_bits.len() {
        let o_neurons = old_neurons[cluster_idx];
        let o_bits = old_bits[cluster_idx];
        let n_neurons = new_neurons[cluster_idx];
        let n_bits = new_bits[cluster_idx];

        for neuron_idx in 0..n_neurons {
            if neuron_idx < o_neurons {
                // Existing neuron - copy and adjust connections
                for bit_idx in 0..n_bits {
                    if bit_idx < o_bits {
                        // Copy existing connection with 10% chance of small perturbation
                        let conn_idx = old_idx + neuron_idx * o_bits + bit_idx;
                        let old_conn = old_connections[conn_idx];
                        let new_conn = if rng.gen::<f64>() < 0.1 {
                            let delta = rng.gen_range(-2i64..=2i64);
                            (old_conn + delta).max(0).min(total_input_bits as i64 - 1)
                        } else {
                            old_conn
                        };
                        result.push(new_conn);
                    } else {
                        // New bit position - random connection
                        result.push(rng.gen_range(0..total_input_bits as i64));
                    }
                }
            } else {
                // New neuron - copy from random existing with mutations
                if o_neurons > 0 {
                    let template_neuron = rng.gen_range(0..o_neurons);
                    for bit_idx in 0..n_bits {
                        if bit_idx < o_bits {
                            let conn_idx = old_idx + template_neuron * o_bits + bit_idx;
                            let old_conn = old_connections[conn_idx];
                            let delta = *[-2i64, -1, 1, 2].choose(rng).unwrap();
                            let new_conn = (old_conn + delta).max(0).min(total_input_bits as i64 - 1);
                            result.push(new_conn);
                        } else {
                            result.push(rng.gen_range(0..total_input_bits as i64));
                        }
                    }
                } else {
                    // No existing neurons - fully random
                    for _ in 0..n_bits {
                        result.push(rng.gen_range(0..total_input_bits as i64));
                    }
                }
            }
        }

        // Update old_idx for next cluster
        old_idx += o_neurons * o_bits;
    }

    result
}

/// Search for neighbors above accuracy threshold.
///
/// This is the main entry point that eliminates Python↔Rust round trips.
/// Generates candidates, evaluates them, filters by threshold, all in Rust.
///
/// Returns: (passed_candidates, all_candidates_evaluated)
pub fn search_neighbors_with_threshold(
    cache: &crate::token_cache::TokenCache,
    base_bits: &[usize],
    base_neurons: &[usize],
    base_connections: &[i64],
    target_count: usize,
    max_attempts: usize,
    accuracy_threshold: f64,
    mutation_config: &MutationConfig,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    empty_value: f32,
    seed: u64,
    log_path: Option<&str>,
    generation: Option<usize>,
    total_generations: Option<usize>,
) -> (Vec<CandidateResult>, usize) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut logger = FileLogger::new(log_path);

    let mut passed: Vec<CandidateResult> = Vec::new();
    let mut evaluated = 0;
    // Smaller batch = less contention between genome training threads
    // 4 genomes × full parallel training is better than 10 genomes competing
    // Batch size for offspring/neighbor generation - configurable via env var
    // Default 50 (persistent worker eliminates per-call overhead, larger batches more efficient)
    let batch_size: usize = std::env::var("WNN_OFFSPRING_BATCH_SIZE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    // Generation prefix for logs - shows current generation / total generations
    let total_gens = total_generations.unwrap_or(100);
    let current_gen = generation.map(|g| g + 1).unwrap_or(1);
    let gen_prefix = format_gen_prefix(current_gen, total_gens);

    let mut shown_count = 0;  // Sequential count of shown/passed candidates

    while passed.len() < target_count && evaluated < max_attempts {
        // Generate batch based on remaining needed (plus small buffer for efficiency)
        // This avoids generating 50 candidates when we only need 4 more
        let remaining_needed = target_count - passed.len();
        let batch_to_generate = (remaining_needed + 5).min(batch_size).min(max_attempts - evaluated);
        let mut batch_bits: Vec<usize> = Vec::new();
        let mut batch_neurons: Vec<usize> = Vec::new();
        let mut batch_connections: Vec<i64> = Vec::new();
        let mut batch_genomes: Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> = Vec::new();

        for _ in 0..batch_to_generate {
            let (new_bits, new_neurons, new_conns) = mutate_genome(
                base_bits,
                base_neurons,
                base_connections,
                mutation_config,
                &mut rng,
            );

            batch_bits.extend(&new_bits);
            batch_neurons.extend(&new_neurons);
            batch_connections.extend(&new_conns);
            batch_genomes.push((new_bits, new_neurons, new_conns));
        }

        // Evaluate the batch - hybrid with persistent worker (no per-call overhead)
        let results = crate::token_cache::evaluate_genomes_cached_hybrid(
            cache,
            &batch_bits,
            &batch_neurons,
            &batch_connections,
            batch_to_generate,
            train_subset_idx,
            eval_subset_idx,
            empty_value,
        );

        // Process results
        for (i, (ce, acc)) in results.iter().enumerate() {
            evaluated += 1;
            let (bits, neurons, conns) = &batch_genomes[i];

            // Only log and add candidates that pass threshold
            if *acc >= accuracy_threshold {
                shown_count += 1;
                logger.log(&format_genome_log(
                    current_gen, total_gens, GenomeLogType::Neighbor,
                    shown_count, target_count, *ce, *acc
                ));

                passed.push(CandidateResult {
                    bits_per_cluster: bits.clone(),
                    neurons_per_cluster: neurons.clone(),
                    connections: conns.clone(),
                    cross_entropy: *ce,
                    accuracy: *acc,
                });

                if passed.len() >= target_count {
                    break;
                }
            }
        }
    }

    logger.log(&format!(
        "{} Search complete: found {}/{} candidates (evaluated {})",
        gen_prefix, passed.len(), target_count, evaluated
    ));

    (passed, evaluated)
}

/// Search for neighbors, returning best N if threshold not met.
///
/// This version guarantees returning candidates even if none meet threshold.
/// Useful when you need to make progress regardless.
pub fn search_neighbors_best_n(
    cache: &crate::token_cache::TokenCache,
    base_bits: &[usize],
    base_neurons: &[usize],
    base_connections: &[i64],
    target_count: usize,
    max_attempts: usize,
    accuracy_threshold: f64,
    mutation_config: &MutationConfig,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    empty_value: f32,
    seed: u64,
    log_path: Option<&str>,
    generation: Option<usize>,
    total_generations: Option<usize>,
) -> Vec<CandidateResult> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut logger = FileLogger::new(log_path);

    let mut passed: Vec<CandidateResult> = Vec::new();
    let mut all_candidates: Vec<CandidateResult> = Vec::new();
    let mut evaluated = 0;
    // Batch size for offspring/neighbor generation - configurable via env var
    // Default 50 (persistent worker eliminates per-call overhead, larger batches more efficient)
    let batch_size: usize = std::env::var("WNN_OFFSPRING_BATCH_SIZE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    // Generation prefix for logs - shows current generation / total generations
    let total_gens = total_generations.unwrap_or(100);
    let current_gen = generation.map(|g| g + 1).unwrap_or(1);
    let gen_prefix = format_gen_prefix(current_gen, total_gens);

    let mut shown_count = 0;  // Sequential count of shown/passed candidates

    while passed.len() < target_count && evaluated < max_attempts {
        // Generate batch based on remaining needed (plus small buffer for efficiency)
        // This avoids generating 50 candidates when we only need 4 more
        let remaining_needed = target_count - passed.len();
        let batch_to_generate = (remaining_needed + 5).min(batch_size).min(max_attempts - evaluated);
        let mut batch_bits: Vec<usize> = Vec::new();
        let mut batch_neurons: Vec<usize> = Vec::new();
        let mut batch_connections: Vec<i64> = Vec::new();
        let mut batch_genomes: Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> = Vec::new();

        for _ in 0..batch_to_generate {
            let (new_bits, new_neurons, new_conns) = mutate_genome(
                base_bits,
                base_neurons,
                base_connections,
                mutation_config,
                &mut rng,
            );

            batch_bits.extend(&new_bits);
            batch_neurons.extend(&new_neurons);
            batch_connections.extend(&new_conns);
            batch_genomes.push((new_bits, new_neurons, new_conns));
        }

        // Evaluate the batch - hybrid with persistent worker (no per-call overhead)
        let results = crate::token_cache::evaluate_genomes_cached_hybrid(
            cache,
            &batch_bits,
            &batch_neurons,
            &batch_connections,
            batch_to_generate,
            train_subset_idx,
            eval_subset_idx,
            empty_value,
        );

        for (i, (ce, acc)) in results.iter().enumerate() {
            evaluated += 1;
            let (bits, neurons, conns) = &batch_genomes[i];

            let candidate = CandidateResult {
                bits_per_cluster: bits.clone(),
                neurons_per_cluster: neurons.clone(),
                connections: conns.clone(),
                cross_entropy: *ce,
                accuracy: *acc,
            };

            // Only log candidates that pass threshold
            if *acc >= accuracy_threshold {
                shown_count += 1;
                logger.log(&format_genome_log(
                    current_gen, total_gens, GenomeLogType::Neighbor,
                    shown_count, target_count, *ce, *acc
                ));
                passed.push(candidate);
                if passed.len() >= target_count {
                    break;
                }
            } else {
                all_candidates.push(candidate);
            }
        }
    }

    // If we didn't get enough passing candidates, add best from all_candidates
    if passed.len() < target_count {
        // Sort by accuracy (descending), then CE (ascending)
        all_candidates.sort_by(|a, b| {
            b.accuracy.partial_cmp(&a.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cross_entropy.partial_cmp(&b.cross_entropy)
                    .unwrap_or(std::cmp::Ordering::Equal))
        });

        let need = target_count - passed.len();
        let fallback_start = shown_count;  // Continue numbering from where we left off
        for (i, candidate) in all_candidates.into_iter().take(need).enumerate() {
            // Log fallback candidates with (Best) indicator
            logger.log(&format_genome_log(
                current_gen, total_gens, GenomeLogType::Fallback,
                fallback_start + i + 1, target_count,
                candidate.cross_entropy, candidate.accuracy
            ));
            passed.push(candidate);
        }

        logger.log(&format!(
            "{} Threshold not met, returning {} best candidates",
            gen_prefix, passed.len()
        ));
    } else {
        logger.log(&format!(
            "{} Search complete: found {}/{} passing candidates (evaluated {})",
            gen_prefix, passed.len(), target_count, evaluated
        ));
    }

    passed
}

/// Configuration for GA offspring generation.
#[derive(Clone)]
pub struct GAConfig {
    pub num_clusters: usize,
    /// Which cluster indices can be mutated. None means all clusters.
    /// Supports arbitrary tier optimization (e.g., tier1 = indices 100-499).
    pub mutable_clusters: Option<Vec<usize>>,
    pub min_bits: usize,
    pub max_bits: usize,
    pub min_neurons: usize,
    pub max_neurons: usize,
    pub bits_mutation_rate: f64,    // Per-cluster bits mutation rate (0.0 to disable)
    pub neurons_mutation_rate: f64, // Per-cluster neurons mutation rate (0.0 to disable)
    pub crossover_rate: f64,        // Probability of crossover vs clone
    pub tournament_size: usize,     // Tournament selection size
    pub total_input_bits: usize,
}

/// Tournament selection: pick best from random subset.
fn tournament_select<'a>(
    population: &'a [(Vec<usize>, Vec<usize>, Vec<i64>, f64)], // (bits, neurons, conns, fitness)
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a (Vec<usize>, Vec<usize>, Vec<i64>, f64) {
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fitness = population[best_idx].3;

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..population.len());
        if population[idx].3 < best_fitness {  // Lower CE is better
            best_idx = idx;
            best_fitness = population[idx].3;
        }
    }

    &population[best_idx]
}

/// Single-point crossover at cluster boundary.
fn crossover(
    parent1: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    parent2: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    num_clusters: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let crossover_point = rng.gen_range(1..num_clusters);

    // Build child architecture
    let child_bits: Vec<usize> = parent1.0[..crossover_point].iter()
        .chain(parent2.0[crossover_point..].iter())
        .copied()
        .collect();
    let child_neurons: Vec<usize> = parent1.1[..crossover_point].iter()
        .chain(parent2.1[crossover_point..].iter())
        .copied()
        .collect();

    // Build child connections (skip if parents have no connections - bits/neurons only phase)
    let mut child_connections = Vec::new();
    if !parent1.2.is_empty() && !parent2.2.is_empty() {
        let mut p1_idx = 0;
        let mut p2_idx = 0;

        for i in 0..num_clusters {
            let p1_conn_size = parent1.1[i] * parent1.0[i];
            let p2_conn_size = parent2.1[i] * parent2.0[i];

            if i < crossover_point {
                // Take from parent1
                child_connections.extend(&parent1.2[p1_idx..p1_idx + p1_conn_size]);
            } else {
                // Take from parent2
                child_connections.extend(&parent2.2[p2_idx..p2_idx + p2_conn_size]);
            }

            p1_idx += p1_conn_size;
            p2_idx += p2_conn_size;
        }
    }

    (child_bits, child_neurons, child_connections)
}

/// Mutate a genome for GA (reuses MutationConfig logic).
fn mutate_ga(
    bits: &[usize],
    neurons: &[usize],
    connections: &[i64],
    config: &GAConfig,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let mutation_config = MutationConfig {
        num_clusters: config.num_clusters,
        mutable_clusters: config.mutable_clusters.clone(),
        min_bits: config.min_bits,
        max_bits: config.max_bits,
        min_neurons: config.min_neurons,
        max_neurons: config.max_neurons,
        bits_mutation_rate: config.bits_mutation_rate,
        neurons_mutation_rate: config.neurons_mutation_rate,
        total_input_bits: config.total_input_bits,
    };
    mutate_genome(bits, neurons, connections, &mutation_config, rng)
}

/// Search for GA offspring above accuracy threshold.
///
/// Performs tournament selection, crossover, and mutation entirely in Rust.
/// Returns viable offspring (accuracy >= threshold) up to target count.
///
/// Args:
///   - population: Parent population as (bits, neurons, connections, fitness) tuples
///   - target_count: Number of viable offspring needed
///   - max_attempts: Maximum offspring to generate before giving up
///   - accuracy_threshold: Minimum accuracy for viable offspring
///
/// Returns: OffspringSearchResult with candidates, evaluated count, and viable count
pub fn search_offspring(
    cache: &crate::token_cache::TokenCache,
    population: &[(Vec<usize>, Vec<usize>, Vec<i64>, f64)],
    target_count: usize,
    max_attempts: usize,
    accuracy_threshold: f64,
    ga_config: &GAConfig,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    empty_value: f32,
    seed: u64,
    log_path: Option<&str>,
    generation: Option<usize>,
    total_generations: Option<usize>,
    return_best_n: bool,  // If true, return top N by CE when not enough pass threshold
) -> OffspringSearchResult {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut logger = FileLogger::new(log_path);

    let mut passed: Vec<CandidateResult> = Vec::new();
    let mut all_candidates: Vec<CandidateResult> = Vec::new();
    let mut evaluated = 0;
    // Batch size for offspring/neighbor generation - configurable via env var
    // Default 50 (persistent worker eliminates per-call overhead, larger batches more efficient)
    let batch_size: usize = std::env::var("WNN_OFFSPRING_BATCH_SIZE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    // Generation prefix for logs - shows current generation / total generations
    let total_gens = total_generations.unwrap_or(100);
    let current_gen = generation.map(|g| g + 1).unwrap_or(1);
    let gen_prefix = format_gen_prefix(current_gen, total_gens);

    let mut shown_count = 0;  // Sequential count of shown/passed candidates

    while passed.len() < target_count && evaluated < max_attempts {
        // Generate batch based on remaining needed (plus small buffer for efficiency)
        // This avoids generating 50 candidates when we only need 4 more
        let remaining_needed = target_count - passed.len();
        let batch_to_generate = (remaining_needed + 5).min(batch_size).min(max_attempts - evaluated);
        let mut batch_bits: Vec<usize> = Vec::new();
        let mut batch_neurons: Vec<usize> = Vec::new();
        let mut batch_connections: Vec<i64> = Vec::new();
        let mut batch_genomes: Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> = Vec::new();

        for _ in 0..batch_to_generate {
            // Tournament selection
            let p1 = tournament_select(population, ga_config.tournament_size, &mut rng);
            let p2 = tournament_select(population, ga_config.tournament_size, &mut rng);

            // Crossover or clone
            let child = if rng.gen::<f64>() < ga_config.crossover_rate {
                crossover(p1, p2, ga_config.num_clusters, &mut rng)
            } else {
                (p1.0.clone(), p1.1.clone(), p1.2.clone())
            };

            // Mutation
            let (new_bits, new_neurons, new_conns) = mutate_ga(
                &child.0,
                &child.1,
                &child.2,
                ga_config,
                &mut rng,
            );

            batch_bits.extend(&new_bits);
            batch_neurons.extend(&new_neurons);
            batch_connections.extend(&new_conns);
            batch_genomes.push((new_bits, new_neurons, new_conns));
        }

        // Evaluate the batch - hybrid with persistent worker (no per-call overhead)
        let batch_start = std::time::Instant::now();
        let results = crate::token_cache::evaluate_genomes_cached_hybrid(
            cache,
            &batch_bits,
            &batch_neurons,
            &batch_connections,
            batch_to_generate,
            train_subset_idx,
            eval_subset_idx,
            empty_value,
        );
        // Process results - log each viable offspring immediately
        for (i, (ce, acc)) in results.iter().enumerate() {
            evaluated += 1;
            let (bits, neurons, conns) = &batch_genomes[i];

            let candidate = CandidateResult {
                bits_per_cluster: bits.clone(),
                neurons_per_cluster: neurons.clone(),
                connections: conns.clone(),
                cross_entropy: *ce,
                accuracy: *acc,
            };

            // Only log and keep candidates that pass threshold
            if *acc >= accuracy_threshold {
                shown_count += 1;
                logger.log(&format_genome_log(
                    current_gen, total_gens, GenomeLogType::Offspring,
                    shown_count, target_count, *ce, *acc
                ));
                passed.push(candidate);
                if passed.len() >= target_count {
                    break;
                }
            } else {
                all_candidates.push(candidate);
            }
        }
    }

    // Capture viable count BEFORE fallback (candidates that passed accuracy threshold)
    let viable = passed.len();

    // If we didn't get enough passing candidates, add best from all_candidates
    if return_best_n && passed.len() < target_count {
        // Sort by CE (lower is better) - return best N by CE when threshold not met
        all_candidates.sort_by(|a, b| {
            a.cross_entropy.partial_cmp(&b.cross_entropy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let need = target_count - passed.len();
        let fallback_start = shown_count;  // Continue numbering from where we left off
        for (i, candidate) in all_candidates.into_iter().take(need).enumerate() {
            // Log fallback candidates with (Best) indicator
            logger.log(&format_genome_log(
                current_gen, total_gens, GenomeLogType::Fallback,
                fallback_start + i + 1, target_count,
                candidate.cross_entropy, candidate.accuracy
            ));
            passed.push(candidate);
        }
    }

    logger.log(&format!(
        "{} Offspring search: {}/{} viable, {}/{} returned (evaluated {}, threshold {:.4}%)",
        gen_prefix, viable, target_count, passed.len(), target_count, evaluated, accuracy_threshold * 100.0
    ));

    OffspringSearchResult {
        candidates: passed,
        evaluated,
        viable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_config() {
        let config = MutationConfig {
            num_clusters: 100,
            mutable_clusters: 100,
            min_bits: 4,
            max_bits: 20,
            min_neurons: 1,
            max_neurons: 15,
            bits_mutation_rate: 0.1,
            neurons_mutation_rate: 0.05,
            total_input_bits: 64,
        };

        assert_eq!(config.bits_delta_max(), 2); // 10% of 24 = 2.4 -> 2
        assert_eq!(config.neurons_delta_max(), 2); // 10% of 16 = 1.6 -> 2
    }

    #[test]
    fn test_mutate_genome() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = MutationConfig {
            num_clusters: 3,
            mutable_clusters: 3,
            min_bits: 4,
            max_bits: 20,
            min_neurons: 1,
            max_neurons: 15,
            bits_mutation_rate: 1.0, // Always mutate for testing
            neurons_mutation_rate: 1.0,
            total_input_bits: 64,
        };

        let bits = vec![8, 10, 12];
        let neurons = vec![1, 2, 3];
        let connections: Vec<i64> = (0..34).collect(); // 8*1 + 10*2 + 12*3 = 34

        let (new_bits, new_neurons, new_conns) = mutate_genome(
            &bits, &neurons, &connections, &config, &mut rng
        );

        // Should have valid values within bounds
        for &b in &new_bits {
            assert!(b >= config.min_bits && b <= config.max_bits);
        }
        for &n in &new_neurons {
            assert!(n >= config.min_neurons && n <= config.max_neurons);
        }

        // Connection count should match new architecture
        let expected_conns: usize = new_bits.iter()
            .zip(new_neurons.iter())
            .map(|(b, n)| b * n)
            .sum();
        assert_eq!(new_conns.len(), expected_conns);
    }
}
