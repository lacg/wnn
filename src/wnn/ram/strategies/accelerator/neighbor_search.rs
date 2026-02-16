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
use std::fs::OpenOptions;
use chrono::Local;

/// Configuration for genome mutation.
#[derive(Clone)]
pub struct MutationConfig {
    pub num_clusters: usize,
    pub neurons_per_cluster: Vec<usize>,    // Needed to map neuron→cluster
    /// Which cluster indices can be mutated. None means all clusters.
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
    pub bits_per_neuron: Vec<usize>,       // [total_neurons] — per-neuron synapse count
    pub neurons_per_cluster: Vec<usize>,   // [num_clusters]
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

/// Mutate a genome to create a neighbor (per-neuron bits).
///
/// Mutations include:
/// - bits_per_neuron: ±delta per-neuron with mutation_rate (only mutable clusters)
/// - neurons_per_cluster: ±delta per-cluster with mutation_rate (only mutable clusters)
/// - connections: adjusted when architecture changes, small perturbations
pub fn mutate_genome(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    config: &MutationConfig,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let bits_delta_max = config.bits_delta_max();
    let neurons_delta_max = config.neurons_delta_max();

    let mut new_bits = bits_per_neuron.to_vec();
    let mut new_neurons = neurons_per_cluster.to_vec();
    let old_neurons = neurons_per_cluster.to_vec();

    // Build cluster→neuron offset mapping
    let mut neuron_offsets = Vec::with_capacity(config.num_clusters + 1);
    neuron_offsets.push(0usize);
    for &n in neurons_per_cluster {
        neuron_offsets.push(neuron_offsets.last().unwrap() + n);
    }

    // Mutable cluster indices
    let mutable: Vec<usize> = match &config.mutable_clusters {
        Some(indices) => indices.clone(),
        None => (0..config.num_clusters).collect(),
    };

    // Phase 1: Mutate bits per-neuron
    for &c in &mutable {
        if c >= config.num_clusters { continue; }
        for n_idx in neuron_offsets[c]..neuron_offsets[c + 1] {
            if rng.gen::<f64>() < config.bits_mutation_rate {
                let delta = rng.gen_range(-bits_delta_max..=bits_delta_max);
                let new_val = (new_bits[n_idx] as i32 + delta)
                    .max(config.min_bits as i32)
                    .min(config.max_bits as i32);
                new_bits[n_idx] = new_val as usize;
            }
        }
    }

    // Phase 2: Mutate neuron count per-cluster (add/remove neurons)
    for &c in &mutable {
        if c >= config.num_clusters { continue; }
        if rng.gen::<f64>() < config.neurons_mutation_rate {
            let delta = rng.gen_range(-neurons_delta_max..=neurons_delta_max);
            let new_n = (new_neurons[c] as i32 + delta)
                .max(config.min_neurons as i32)
                .min(config.max_neurons as i32) as usize;
            let old_n = old_neurons[c];

            if new_n > old_n {
                // Add neurons: clone random existing neuron's bits
                let cluster_start = neuron_offsets[c];
                let cluster_end = neuron_offsets[c + 1];
                for _ in 0..(new_n - old_n) {
                    let template = rng.gen_range(cluster_start..cluster_end);
                    // Insert at end of new_bits (will be re-ordered below)
                    new_bits.push(new_bits[template]);
                }
            } else if new_n < old_n {
                // Remove trailing neurons from this cluster
                // Mark for removal (we'll rebuild below)
            }
            new_neurons[c] = new_n;
        }
    }

    // Rebuild new_bits to match new_neurons layout if neurons changed
    // (The simple push/remove above doesn't maintain correct ordering)
    let total_old: usize = old_neurons.iter().sum();
    let total_new: usize = new_neurons.iter().sum();
    if total_new != total_old || new_bits.len() != total_new {
        let mut rebuilt_bits = Vec::with_capacity(total_new);
        let mut old_offset = 0;
        for c in 0..config.num_clusters {
            let old_n = old_neurons[c];
            let new_n = new_neurons[c];
            // Copy existing neurons (up to min of old/new)
            let copy_n = old_n.min(new_n);
            for j in 0..copy_n {
                rebuilt_bits.push(new_bits[old_offset + j]);
            }
            // Add new neurons if needed
            if new_n > old_n {
                for _ in 0..(new_n - old_n) {
                    let template = if old_n > 0 {
                        new_bits[old_offset + rng.gen_range(0..old_n)]
                    } else {
                        rng.gen_range(config.min_bits..=config.max_bits)
                    };
                    rebuilt_bits.push(template);
                }
            }
            old_offset += old_n;
        }
        new_bits = rebuilt_bits;
    }

    // Adjust connections for architecture changes
    let new_connections = adjust_connections_per_neuron(
        connections,
        bits_per_neuron,
        &old_neurons,
        &new_bits,
        &new_neurons,
        config.total_input_bits,
        rng,
    );

    (new_bits, new_neurons, new_connections)
}

/// Adjust connections when per-neuron architecture changes.
///
/// Each neuron has its own bit count, so connections are a flat array where
/// neuron i owns `bits_per_neuron[i]` contiguous connections. When bits change
/// (grow/shrink) or neurons are added/removed, this function rebuilds the
/// connection array to match the new architecture.
fn adjust_connections_per_neuron(
    old_connections: &[i64],
    old_bits_per_neuron: &[usize],
    old_neurons_per_cluster: &[usize],
    new_bits_per_neuron: &[usize],
    new_neurons_per_cluster: &[usize],
    total_input_bits: usize,
    rng: &mut impl Rng,
) -> Vec<i64> {
    if old_connections.is_empty() {
        return Vec::new();
    }

    let num_clusters = old_neurons_per_cluster.len();
    let mut result = Vec::new();

    // Build old connection offsets per neuron (cumulative sum of old bits)
    let mut old_conn_offsets = vec![0usize];
    for &b in old_bits_per_neuron {
        old_conn_offsets.push(old_conn_offsets.last().unwrap() + b);
    }

    // Build old/new neuron offsets per cluster
    let mut old_neuron_offsets = vec![0usize];
    for &n in old_neurons_per_cluster {
        old_neuron_offsets.push(old_neuron_offsets.last().unwrap() + n);
    }
    let mut new_neuron_offsets = vec![0usize];
    for &n in new_neurons_per_cluster {
        new_neuron_offsets.push(new_neuron_offsets.last().unwrap() + n);
    }

    for c in 0..num_clusters {
        let old_n = old_neurons_per_cluster[c];
        let new_n = new_neurons_per_cluster[c];
        let old_cluster_start = old_neuron_offsets[c];
        let new_cluster_start = new_neuron_offsets[c];

        for local_idx in 0..new_n {
            let new_global = new_cluster_start + local_idx;
            let n_bits = new_bits_per_neuron[new_global];

            if local_idx < old_n {
                // Existing neuron — copy and adjust connections
                let old_global = old_cluster_start + local_idx;
                let o_bits = old_bits_per_neuron[old_global];
                let old_start = old_conn_offsets[old_global];

                for bit_idx in 0..n_bits {
                    if bit_idx < o_bits {
                        let old_conn = old_connections[old_start + bit_idx];
                        let new_conn = if rng.gen::<f64>() < 0.1 {
                            let delta = rng.gen_range(-2i64..=2i64);
                            (old_conn + delta).max(0).min(total_input_bits as i64 - 1)
                        } else {
                            old_conn
                        };
                        result.push(new_conn);
                    } else {
                        result.push(rng.gen_range(0..total_input_bits as i64));
                    }
                }
            } else {
                // New neuron — clone from random existing + mutate
                if old_n > 0 {
                    let template_local = rng.gen_range(0..old_n);
                    let template_global = old_cluster_start + template_local;
                    let template_bits = old_bits_per_neuron[template_global];
                    let template_start = old_conn_offsets[template_global];

                    for bit_idx in 0..n_bits {
                        if bit_idx < template_bits {
                            let old_conn = old_connections[template_start + bit_idx];
                            let delta = *[-2i64, -1, 1, 2].choose(rng).unwrap();
                            let new_conn = (old_conn + delta).max(0).min(total_input_bits as i64 - 1);
                            result.push(new_conn);
                        } else {
                            result.push(rng.gen_range(0..total_input_bits as i64));
                        }
                    }
                } else {
                    for _ in 0..n_bits {
                        result.push(rng.gen_range(0..total_input_bits as i64));
                    }
                }
            }
        }
    }

    result
}

/// Generate random connections for a genome from scratch.
///
/// Each neuron gets `bits_per_neuron[i]` random connections in `[0, total_input_bits)`.
/// Returns a flat Vec<i64> of length `sum(bits_per_neuron)`.
pub fn generate_random_connections(
    bits_per_neuron: &[usize],
    total_input_bits: usize,
    rng: &mut impl Rng,
) -> Vec<i64> {
    let total: usize = bits_per_neuron.iter().sum();
    let mut connections = Vec::with_capacity(total);
    let limit = total_input_bits as i64;
    for _ in 0..total {
        connections.push(rng.gen_range(0..limit));
    }
    connections
}

/// Search for neighbors above accuracy threshold (generic over eval backend).
///
/// Generates candidates, evaluates via `evaluate_batch` closure, filters by threshold.
/// The closure signature: `(bits_flat, neurons_flat, conns_flat, num_genomes) -> Vec<(ce, acc)>`
///
/// Returns: (passed_candidates, all_candidates_evaluated)
pub fn search_neighbors_with_threshold<F>(
    base_bits: &[usize],
    base_neurons: &[usize],
    base_connections: &[i64],
    target_count: usize,
    max_attempts: usize,
    accuracy_threshold: f64,
    mutation_config: &MutationConfig,
    evaluate_batch: &F,
    seed: u64,
    log_path: Option<&str>,
    generation: Option<usize>,
    total_generations: Option<usize>,
) -> (Vec<CandidateResult>, usize)
where
    F: Fn(&[usize], &[usize], &[i64], usize) -> Vec<(f64, f64)>,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut logger = FileLogger::new(log_path);

    let mut passed: Vec<CandidateResult> = Vec::new();
    let mut evaluated = 0;
    let batch_size: usize = std::env::var("WNN_OFFSPRING_BATCH_SIZE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    let total_gens = total_generations.unwrap_or(100);
    let current_gen = generation.map(|g| g + 1).unwrap_or(1);
    let gen_prefix = format_gen_prefix(current_gen, total_gens);

    let mut shown_count = 0;

    while passed.len() < target_count && evaluated < max_attempts {
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

        let results = evaluate_batch(&batch_bits, &batch_neurons, &batch_connections, batch_to_generate);

        for (i, (ce, acc)) in results.iter().enumerate() {
            evaluated += 1;
            let (bits, neurons, conns) = &batch_genomes[i];

            if *acc >= accuracy_threshold {
                shown_count += 1;
                logger.log(&format_genome_log(
                    current_gen, total_gens, GenomeLogType::Neighbor,
                    shown_count, target_count, *ce, *acc
                ));

                passed.push(CandidateResult {
                    bits_per_neuron: bits.clone(),
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
pub fn search_neighbors_best_n<F>(
    base_bits: &[usize],
    base_neurons: &[usize],
    base_connections: &[i64],
    target_count: usize,
    max_attempts: usize,
    accuracy_threshold: f64,
    mutation_config: &MutationConfig,
    evaluate_batch: &F,
    seed: u64,
    log_path: Option<&str>,
    generation: Option<usize>,
    total_generations: Option<usize>,
) -> Vec<CandidateResult>
where
    F: Fn(&[usize], &[usize], &[i64], usize) -> Vec<(f64, f64)>,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut logger = FileLogger::new(log_path);

    let mut passed: Vec<CandidateResult> = Vec::new();
    let mut all_candidates: Vec<CandidateResult> = Vec::new();
    let mut evaluated = 0;
    let batch_size: usize = std::env::var("WNN_OFFSPRING_BATCH_SIZE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    let total_gens = total_generations.unwrap_or(100);
    let current_gen = generation.map(|g| g + 1).unwrap_or(1);
    let gen_prefix = format_gen_prefix(current_gen, total_gens);

    let mut shown_count = 0;

    while passed.len() < target_count && evaluated < max_attempts {
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

        let results = evaluate_batch(&batch_bits, &batch_neurons, &batch_connections, batch_to_generate);

        for (i, (ce, acc)) in results.iter().enumerate() {
            evaluated += 1;
            let (bits, neurons, conns) = &batch_genomes[i];

            let candidate = CandidateResult {
                bits_per_neuron: bits.clone(),
                neurons_per_cluster: neurons.clone(),
                connections: conns.clone(),
                cross_entropy: *ce,
                accuracy: *acc,
            };

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

    if passed.len() < target_count {
        all_candidates.sort_by(|a, b| {
            b.accuracy.partial_cmp(&a.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cross_entropy.partial_cmp(&b.cross_entropy)
                    .unwrap_or(std::cmp::Ordering::Equal))
        });

        let need = target_count - passed.len();
        let fallback_start = shown_count;
        for (i, candidate) in all_candidates.into_iter().take(need).enumerate() {
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

/// Single-point crossover at cluster boundary (per-neuron bits).
///
/// Population tuples are (bits_per_neuron, neurons_per_cluster, connections, fitness).
/// bits_per_neuron is [total_neurons] and connections is flat [sum(bits_per_neuron)],
/// so we need neuron offsets and connection offsets to slice at cluster boundaries.
fn crossover(
    parent1: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    parent2: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    num_clusters: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let crossover_point = rng.gen_range(1..num_clusters);

    // neurons_per_cluster is per-cluster — straightforward crossover
    let child_neurons: Vec<usize> = parent1.1[..crossover_point].iter()
        .chain(parent2.1[crossover_point..].iter())
        .copied()
        .collect();

    // Compute neuron offsets for each parent (cumulative sum of neurons_per_cluster)
    let p1_neuron_off: Vec<usize> = std::iter::once(0)
        .chain(parent1.1.iter().scan(0, |acc, &n| { *acc += n; Some(*acc) }))
        .collect();
    let p2_neuron_off: Vec<usize> = std::iter::once(0)
        .chain(parent2.1.iter().scan(0, |acc, &n| { *acc += n; Some(*acc) }))
        .collect();

    // bits_per_neuron: take parent1's neurons for clusters < crossover_point, parent2's for rest
    let mut child_bits = Vec::new();
    child_bits.extend_from_slice(&parent1.0[..p1_neuron_off[crossover_point]]);
    child_bits.extend_from_slice(&parent2.0[p2_neuron_off[crossover_point]..]);

    // Connections: need per-neuron connection offsets to slice correctly
    let mut child_connections = Vec::new();
    if !parent1.2.is_empty() && !parent2.2.is_empty() {
        // Connection offsets = cumulative sum of bits_per_neuron
        let p1_conn_off: Vec<usize> = std::iter::once(0)
            .chain(parent1.0.iter().scan(0, |acc, &b| { *acc += b; Some(*acc) }))
            .collect();
        let p2_conn_off: Vec<usize> = std::iter::once(0)
            .chain(parent2.0.iter().scan(0, |acc, &b| { *acc += b; Some(*acc) }))
            .collect();

        // Parent1 connections for neurons 0..p1_neuron_off[crossover_point]
        let p1_conn_end = p1_conn_off[p1_neuron_off[crossover_point]];
        child_connections.extend_from_slice(&parent1.2[..p1_conn_end]);

        // Parent2 connections for neurons p2_neuron_off[crossover_point]..end
        let p2_conn_start = p2_conn_off[p2_neuron_off[crossover_point]];
        child_connections.extend_from_slice(&parent2.2[p2_conn_start..]);
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
        neurons_per_cluster: neurons.to_vec(),
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
pub fn search_offspring<F>(
    population: &[(Vec<usize>, Vec<usize>, Vec<i64>, f64)],
    target_count: usize,
    max_attempts: usize,
    accuracy_threshold: f64,
    ga_config: &GAConfig,
    evaluate_batch: &F,
    seed: u64,
    log_path: Option<&str>,
    generation: Option<usize>,
    total_generations: Option<usize>,
    return_best_n: bool,
) -> OffspringSearchResult
where
    F: Fn(&[usize], &[usize], &[i64], usize) -> Vec<(f64, f64)>,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut logger = FileLogger::new(log_path);

    let mut passed: Vec<CandidateResult> = Vec::new();
    let mut all_candidates: Vec<CandidateResult> = Vec::new();
    let mut evaluated = 0;
    let batch_size: usize = std::env::var("WNN_OFFSPRING_BATCH_SIZE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(50);

    let total_gens = total_generations.unwrap_or(100);
    let current_gen = generation.map(|g| g + 1).unwrap_or(1);
    let gen_prefix = format_gen_prefix(current_gen, total_gens);

    let mut shown_count = 0;

    while passed.len() < target_count && evaluated < max_attempts {
        let remaining_needed = target_count - passed.len();
        let batch_to_generate = (remaining_needed + 5).min(batch_size).min(max_attempts - evaluated);
        let mut batch_bits: Vec<usize> = Vec::new();
        let mut batch_neurons: Vec<usize> = Vec::new();
        let mut batch_connections: Vec<i64> = Vec::new();
        let mut batch_genomes: Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> = Vec::new();

        for _ in 0..batch_to_generate {
            let p1 = tournament_select(population, ga_config.tournament_size, &mut rng);
            let p2 = tournament_select(population, ga_config.tournament_size, &mut rng);

            let child = if rng.gen::<f64>() < ga_config.crossover_rate {
                crossover(p1, p2, ga_config.num_clusters, &mut rng)
            } else {
                (p1.0.clone(), p1.1.clone(), p1.2.clone())
            };

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

        let results = evaluate_batch(&batch_bits, &batch_neurons, &batch_connections, batch_to_generate);

        for (i, (ce, acc)) in results.iter().enumerate() {
            evaluated += 1;
            let (bits, neurons, conns) = &batch_genomes[i];

            let candidate = CandidateResult {
                bits_per_neuron: bits.clone(),
                neurons_per_cluster: neurons.clone(),
                connections: conns.clone(),
                cross_entropy: *ce,
                accuracy: *acc,
            };

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

    let viable = passed.len();

    if return_best_n && passed.len() < target_count {
        all_candidates.sort_by(|a, b| {
            a.cross_entropy.partial_cmp(&b.cross_entropy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let need = target_count - passed.len();
        let fallback_start = shown_count;
        for (i, candidate) in all_candidates.into_iter().take(need).enumerate() {
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
            neurons_per_cluster: vec![5; 100],
            mutable_clusters: None,
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
    fn test_mutate_genome_per_neuron() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let neurons = vec![1, 2, 3]; // 3 clusters, 6 neurons total
        let config = MutationConfig {
            num_clusters: 3,
            neurons_per_cluster: neurons.clone(),
            mutable_clusters: None,
            min_bits: 4,
            max_bits: 20,
            min_neurons: 1,
            max_neurons: 15,
            bits_mutation_rate: 1.0,
            neurons_mutation_rate: 1.0,
            total_input_bits: 64,
        };

        // Per-neuron bits: [neuron0=8, neuron1=10, neuron2=10, neuron3=12, neuron4=12, neuron5=12]
        let bits = vec![8, 10, 10, 12, 12, 12];
        // Connections: 8 + 10 + 10 + 12 + 12 + 12 = 64
        let connections: Vec<i64> = (0..64).collect();

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

        // Total neurons should match sum of new_neurons
        let total_neurons: usize = new_neurons.iter().sum();
        assert_eq!(new_bits.len(), total_neurons);

        // Connection count should match sum of per-neuron bits
        let expected_conns: usize = new_bits.iter().sum();
        assert_eq!(new_conns.len(), expected_conns);
    }

    #[test]
    fn test_crossover_per_neuron() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // Parent1: 3 clusters, neurons=[2, 1, 2], bits=[8,8, 10, 12,12]
        let p1 = (
            vec![8, 8, 10, 12, 12],    // bits_per_neuron
            vec![2, 1, 2],              // neurons_per_cluster
            (0..50i64).collect::<Vec<_>>(), // 8+8+10+12+12 = 50 conns
            10.0,
        );
        // Parent2: 3 clusters, neurons=[1, 3, 1], bits=[6, 14,14,14, 16]
        let p2 = (
            vec![6, 14, 14, 14, 16],
            vec![1, 3, 1],
            (100..164i64).collect::<Vec<_>>(), // 6+14+14+14+16 = 64 conns
            11.0,
        );

        let (child_bits, child_neurons, child_conns) = crossover(&p1, &p2, 3, &mut rng);

        // neurons_per_cluster should be mix of parents
        assert_eq!(child_neurons.len(), 3);

        // Total neurons should match bits length
        let total_neurons: usize = child_neurons.iter().sum();
        assert_eq!(child_bits.len(), total_neurons);

        // Connection count should match sum of bits
        let expected_conns: usize = child_bits.iter().sum();
        assert_eq!(child_conns.len(), expected_conns);
    }
}
