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
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use chrono::Local;

/// Live progress data shared between search functions and Python observer.
///
/// Updated per-batch during search_offspring/search_neighbors. The Python
/// observer thread reads this periodically and POSTs to the dashboard.
#[derive(Clone, Debug)]
pub struct LiveProgress {
    pub experiment_id: i64,
    pub generation: i32,
    pub total_generations: i32,
    pub phase: String,
    pub evaluated: usize,
    pub target_count: usize,
    pub viable: Option<usize>,
    pub best_ce: f64,
    pub best_acc: f64,
    pub elapsed_secs: f64,
    pub updated_at: f64,
}

impl LiveProgress {
    pub fn now_unix() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }
}

/// Phase type for phase-aware crossover and mutation.
///
/// Each optimization phase only touches its own dimension:
/// - Neurons: changes neuron counts, preserves existing neurons' bits + connections
/// - Bits: changes bit counts per neuron, adds/removes connections (no drift)
/// - Connections: perturbs connection targets, preserves architecture
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhaseType {
    Neurons = 0,
    Bits = 1,
    Connections = 2,
}

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
    pub phase_type: PhaseType,
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
    pub f1_macro: f64,
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

/// Compute cumulative neuron offsets per cluster: [0, n0, n0+n1, ...].
fn compute_neuron_offsets(neurons_per_cluster: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(neurons_per_cluster.len() + 1);
    offsets.push(0usize);
    for &n in neurons_per_cluster {
        offsets.push(offsets.last().unwrap() + n);
    }
    offsets
}

/// Compute cumulative connection offsets per neuron: [0, b0, b0+b1, ...].
fn compute_conn_offsets(bits_per_neuron: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(bits_per_neuron.len() + 1);
    offsets.push(0usize);
    for &b in bits_per_neuron {
        offsets.push(offsets.last().unwrap() + b);
    }
    offsets
}

/// Get mutable cluster indices from config.
fn mutable_clusters(config: &MutationConfig) -> Vec<usize> {
    match &config.mutable_clusters {
        Some(indices) => indices.clone(),
        None => (0..config.num_clusters).collect(),
    }
}

/// Mutate a genome to create a neighbor, dispatching by phase type.
///
/// Each phase only touches its own dimension:
/// - Neurons: add/remove neurons, preserve existing bits + connections
/// - Bits: change bit counts, add/remove connections (no drift on existing)
/// - Connections: perturb connection targets, preserve architecture
pub fn mutate_genome(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    config: &MutationConfig,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    match config.phase_type {
        PhaseType::Neurons => mutate_neurons_phase(bits_per_neuron, neurons_per_cluster, connections, config, rng),
        PhaseType::Bits => mutate_bits_phase(bits_per_neuron, neurons_per_cluster, connections, config, rng),
        PhaseType::Connections => mutate_connections_phase(bits_per_neuron, neurons_per_cluster, connections, config, rng),
    }
}

/// Neurons phase mutation: add/remove neurons per cluster.
///
/// - Existing neurons keep their bits + connections untouched
/// - New neurons get the mode bit size + fully random connections
/// - Removed neurons are dropped from end of cluster
fn mutate_neurons_phase(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    config: &MutationConfig,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let neurons_delta_max = config.neurons_delta_max();
    let neuron_off = compute_neuron_offsets(neurons_per_cluster);
    let conn_off = compute_conn_offsets(bits_per_neuron);
    let mutable = mutable_clusters(config);

    let mut new_neurons = neurons_per_cluster.to_vec();

    // Determine which clusters change
    for &c in &mutable {
        if c >= config.num_clusters { continue; }
        if rng.gen::<f64>() < config.neurons_mutation_rate {
            let delta = rng.gen_range(-neurons_delta_max..=neurons_delta_max);
            let new_n = (new_neurons[c] as i32 + delta)
                .max(config.min_neurons as i32)
                .min(config.max_neurons as i32) as usize;
            new_neurons[c] = new_n;
        }
    }

    // Rebuild bits_per_neuron and connections to match new neuron counts
    let mut new_bits = Vec::new();
    let mut new_conns = Vec::new();

    for c in 0..config.num_clusters {
        let old_n = neurons_per_cluster[c];
        let new_n = new_neurons[c];
        let old_start = neuron_off[c];

        // Compute mode bit size for this cluster (for new neurons)
        let cluster_bits: Vec<usize> = (old_start..neuron_off[c + 1])
            .map(|i| bits_per_neuron[i])
            .collect();
        let mode_bits = if cluster_bits.is_empty() {
            (config.min_bits + config.max_bits) / 2
        } else {
            // Simple mode: most common value
            let mut counts = std::collections::HashMap::new();
            for &b in &cluster_bits {
                *counts.entry(b).or_insert(0usize) += 1;
            }
            *counts.iter().max_by_key(|&(_, &count)| count).unwrap().0
        };

        // Copy existing neurons (up to min of old/new)
        let keep_n = old_n.min(new_n);
        for local in 0..keep_n {
            let global = old_start + local;
            new_bits.push(bits_per_neuron[global]);
            // Copy connections verbatim
            let cs = conn_off[global];
            let ce = conn_off[global + 1];
            new_conns.extend_from_slice(&connections[cs..ce]);
        }

        // Add new neurons with mode bit size + random connections
        for _ in keep_n..new_n {
            new_bits.push(mode_bits);
            for _ in 0..mode_bits {
                new_conns.push(rng.gen_range(0..config.total_input_bits as i64));
            }
        }
    }

    (new_bits, new_neurons, new_conns)
}

/// Bits phase mutation: change bit counts per neuron.
///
/// - Add bits → add random connections (one per added bit)
/// - Remove bits → randomly drop connections (Fisher-Yates partial shuffle)
/// - Existing connections are NEVER drifted
fn mutate_bits_phase(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    config: &MutationConfig,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let bits_delta_max = config.bits_delta_max();
    let neuron_off = compute_neuron_offsets(neurons_per_cluster);
    let conn_off = compute_conn_offsets(bits_per_neuron);
    let mutable = mutable_clusters(config);
    let mutable_set: std::collections::HashSet<usize> = mutable.into_iter().collect();

    let mut new_bits = bits_per_neuron.to_vec();
    let new_neurons = neurons_per_cluster.to_vec(); // unchanged

    // Mutate bit counts for neurons in mutable clusters
    for c in 0..config.num_clusters {
        if !mutable_set.contains(&c) { continue; }
        for n_idx in neuron_off[c]..neuron_off[c + 1] {
            if rng.gen::<f64>() < config.bits_mutation_rate {
                let delta = rng.gen_range(-bits_delta_max..=bits_delta_max);
                let new_val = (new_bits[n_idx] as i32 + delta)
                    .max(config.min_bits as i32)
                    .min(config.max_bits as i32);
                new_bits[n_idx] = new_val as usize;
            }
        }
    }

    // Rebuild connections: preserve existing, add/remove as needed
    let mut new_conns = Vec::new();
    let total_neurons: usize = neurons_per_cluster.iter().sum();

    for n_idx in 0..total_neurons {
        let old_b = bits_per_neuron[n_idx];
        let new_b = new_bits[n_idx];
        let cs = conn_off[n_idx];

        if new_b == old_b {
            // Unchanged: copy all connections verbatim
            new_conns.extend_from_slice(&connections[cs..cs + old_b]);
        } else if new_b > old_b {
            // More bits: keep ALL existing connections, add random for new bits
            new_conns.extend_from_slice(&connections[cs..cs + old_b]);
            for _ in 0..(new_b - old_b) {
                new_conns.push(rng.gen_range(0..config.total_input_bits as i64));
            }
        } else {
            // Fewer bits: randomly select which connections to KEEP (Fisher-Yates)
            // Build index array and partial shuffle to select `new_b` to keep
            let mut indices: Vec<usize> = (0..old_b).collect();
            for i in 0..new_b {
                let j = rng.gen_range(i..old_b);
                indices.swap(i, j);
            }
            // Sort the kept indices to preserve relative order
            let mut kept: Vec<usize> = indices[..new_b].to_vec();
            kept.sort_unstable();
            for &idx in &kept {
                new_conns.push(connections[cs + idx]);
            }
        }
    }

    (new_bits, new_neurons, new_conns)
}

/// Connections phase mutation: perturb connection targets.
///
/// - Neurons and bits unchanged
/// - Each connection has bits_mutation_rate chance of ±2 perturbation
fn mutate_connections_phase(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    config: &MutationConfig,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let new_bits = bits_per_neuron.to_vec();
    let new_neurons = neurons_per_cluster.to_vec();

    let mut new_conns = connections.to_vec();
    let limit = config.total_input_bits as i64;
    for conn in &mut new_conns {
        if rng.gen::<f64>() < config.bits_mutation_rate {
            let delta = rng.gen_range(-2i64..=2i64);
            *conn = (*conn + delta).max(0).min(limit - 1);
        }
    }

    (new_bits, new_neurons, new_conns)
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
    live_progress: Option<&Arc<RwLock<Option<LiveProgress>>>>,
) -> (Vec<CandidateResult>, usize)
where
    F: Fn(&[usize], &[usize], &[i64], usize) -> Vec<(f64, f64, f64)>,
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
    let mut best_ce_so_far = f64::MAX;
    let mut best_acc_so_far = 0.0_f64;
    let start_time = Instant::now();

    // Track fingerprints (bits + neurons + connections) to avoid duplicate neighbors
    let mut known_fps: std::collections::HashSet<(Vec<usize>, Vec<usize>, Vec<i64>)> =
        std::collections::HashSet::new();
    known_fps.insert((base_bits.to_vec(), base_neurons.to_vec(), base_connections.to_vec()));

    while passed.len() < target_count && evaluated < max_attempts {
        let remaining_needed = target_count - passed.len();
        let batch_to_generate = (remaining_needed + 5).min(batch_size).min(max_attempts - evaluated);
        let mut batch_bits: Vec<usize> = Vec::new();
        let mut batch_neurons: Vec<usize> = Vec::new();
        let mut batch_connections: Vec<i64> = Vec::new();
        let mut batch_genomes: Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> = Vec::new();

        for _ in 0..batch_to_generate {
            let (mut new_bits, mut new_neurons, mut new_conns) = mutate_genome(
                base_bits,
                base_neurons,
                base_connections,
                mutation_config,
                &mut rng,
            );

            // Re-mutate if duplicate (up to 3 retries)
            for _ in 0..3 {
                let fp = (new_bits.clone(), new_neurons.clone(), new_conns.clone());
                if !known_fps.contains(&fp) {
                    break;
                }
                let remut = mutate_genome(base_bits, base_neurons, base_connections, mutation_config, &mut rng);
                new_bits = remut.0;
                new_neurons = remut.1;
                new_conns = remut.2;
            }
            known_fps.insert((new_bits.clone(), new_neurons.clone(), new_conns.clone()));

            batch_bits.extend(&new_bits);
            batch_neurons.extend(&new_neurons);
            batch_connections.extend(&new_conns);
            batch_genomes.push((new_bits, new_neurons, new_conns));
        }

        let results = evaluate_batch(&batch_bits, &batch_neurons, &batch_connections, batch_to_generate);

        for (i, (ce, acc, f1)) in results.iter().enumerate() {
            evaluated += 1;
            best_ce_so_far = best_ce_so_far.min(*ce);
            best_acc_so_far = best_acc_so_far.max(*acc);
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
                    f1_macro: *f1,
                });

                if passed.len() >= target_count {
                    break;
                }
            }
        }

        // Update live progress after each batch
        if let Some(lp) = live_progress {
            if let Ok(mut guard) = lp.write() {
                if let Some(ref mut p) = *guard {
                    p.evaluated = evaluated;
                    p.viable = Some(passed.len());
                    p.best_ce = best_ce_so_far;
                    p.best_acc = best_acc_so_far;
                    p.elapsed_secs = start_time.elapsed().as_secs_f64();
                    p.updated_at = LiveProgress::now_unix();
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
    live_progress: Option<&Arc<RwLock<Option<LiveProgress>>>>,
) -> Vec<CandidateResult>
where
    F: Fn(&[usize], &[usize], &[i64], usize) -> Vec<(f64, f64, f64)>,
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
    let mut best_ce_so_far = f64::MAX;
    let mut best_acc_so_far = 0.0_f64;
    let start_time = Instant::now();

    // Track fingerprints (bits + neurons + connections) to avoid duplicate neighbors
    let mut known_fps: std::collections::HashSet<(Vec<usize>, Vec<usize>, Vec<i64>)> =
        std::collections::HashSet::new();
    known_fps.insert((base_bits.to_vec(), base_neurons.to_vec(), base_connections.to_vec()));

    while passed.len() < target_count && evaluated < max_attempts {
        let remaining_needed = target_count - passed.len();
        let batch_to_generate = (remaining_needed + 5).min(batch_size).min(max_attempts - evaluated);
        let mut batch_bits: Vec<usize> = Vec::new();
        let mut batch_neurons: Vec<usize> = Vec::new();
        let mut batch_connections: Vec<i64> = Vec::new();
        let mut batch_genomes: Vec<(Vec<usize>, Vec<usize>, Vec<i64>)> = Vec::new();

        for _ in 0..batch_to_generate {
            let (mut new_bits, mut new_neurons, mut new_conns) = mutate_genome(
                base_bits,
                base_neurons,
                base_connections,
                mutation_config,
                &mut rng,
            );

            // Re-mutate if duplicate (up to 3 retries)
            for _ in 0..3 {
                let fp = (new_bits.clone(), new_neurons.clone(), new_conns.clone());
                if !known_fps.contains(&fp) {
                    break;
                }
                let remut = mutate_genome(base_bits, base_neurons, base_connections, mutation_config, &mut rng);
                new_bits = remut.0;
                new_neurons = remut.1;
                new_conns = remut.2;
            }
            known_fps.insert((new_bits.clone(), new_neurons.clone(), new_conns.clone()));

            batch_bits.extend(&new_bits);
            batch_neurons.extend(&new_neurons);
            batch_connections.extend(&new_conns);
            batch_genomes.push((new_bits, new_neurons, new_conns));
        }

        let results = evaluate_batch(&batch_bits, &batch_neurons, &batch_connections, batch_to_generate);

        for (i, (ce, acc, f1)) in results.iter().enumerate() {
            evaluated += 1;
            best_ce_so_far = best_ce_so_far.min(*ce);
            best_acc_so_far = best_acc_so_far.max(*acc);
            let (bits, neurons, conns) = &batch_genomes[i];

            let candidate = CandidateResult {
                bits_per_neuron: bits.clone(),
                neurons_per_cluster: neurons.clone(),
                connections: conns.clone(),
                cross_entropy: *ce,
                accuracy: *acc,
                f1_macro: *f1,
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

        // Update live progress after each batch
        if let Some(lp) = live_progress {
            if let Ok(mut guard) = lp.write() {
                if let Some(ref mut p) = *guard {
                    p.evaluated = evaluated;
                    p.viable = Some(passed.len());
                    p.best_ce = best_ce_so_far;
                    p.best_acc = best_acc_so_far;
                    p.elapsed_secs = start_time.elapsed().as_secs_f64();
                    p.updated_at = LiveProgress::now_unix();
                }
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
    pub phase_type: PhaseType,
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

/// Phase-aware crossover dispatch.
///
/// Population tuples are (bits_per_neuron, neurons_per_cluster, connections, fitness).
fn crossover(
    parent1: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    parent2: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    num_clusters: usize,
    phase_type: PhaseType,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    match phase_type {
        PhaseType::Neurons => crossover_neurons(parent1, parent2, num_clusters, rng),
        PhaseType::Bits => crossover_bits(parent1, parent2, num_clusters, rng),
        PhaseType::Connections => crossover_connections(parent1, parent2, num_clusters, rng),
    }
}

/// Neuron-pool crossover: for each cluster, pool neurons from both parents,
/// shuffle, sample child_n without replacement.
fn crossover_neurons(
    parent1: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    parent2: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    num_clusters: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let p1_neuron_off = compute_neuron_offsets(&parent1.1);
    let p2_neuron_off = compute_neuron_offsets(&parent2.1);
    let p1_conn_off = compute_conn_offsets(&parent1.0);
    let p2_conn_off = compute_conn_offsets(&parent2.0);

    let mut child_bits = Vec::new();
    let mut child_neurons = Vec::new();
    let mut child_conns = Vec::new();

    for c in 0..num_clusters {
        let p1_n = parent1.1[c];
        let p2_n = parent2.1[c];

        // child_n = randomly chosen from {parent1_n, parent2_n}
        let child_n = if rng.gen::<bool>() { p1_n } else { p2_n };
        child_neurons.push(child_n);

        // Pool all neurons from both parents as (bits, connections_slice)
        struct NeuronData<'a> { bits: usize, conns: &'a [i64] }
        let mut pool: Vec<NeuronData> = Vec::with_capacity(p1_n + p2_n);

        for local in 0..p1_n {
            let global = p1_neuron_off[c] + local;
            let cs = p1_conn_off[global];
            let ce = p1_conn_off[global + 1];
            pool.push(NeuronData {
                bits: parent1.0[global],
                conns: if parent1.2.is_empty() { &[] } else { &parent1.2[cs..ce] },
            });
        }
        for local in 0..p2_n {
            let global = p2_neuron_off[c] + local;
            let cs = p2_conn_off[global];
            let ce = p2_conn_off[global + 1];
            pool.push(NeuronData {
                bits: parent2.0[global],
                conns: if parent2.2.is_empty() { &[] } else { &parent2.2[cs..ce] },
            });
        }

        // Shuffle pool and take first child_n (sample without replacement)
        pool.shuffle(rng);
        for neuron in pool.into_iter().take(child_n) {
            child_bits.push(neuron.bits);
            child_conns.extend_from_slice(neuron.conns);
        }
    }

    (child_bits, child_neurons, child_conns)
}

/// Cluster-level crossover: per-cluster coin flip, take whole cluster from one parent.
fn crossover_bits(
    parent1: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    parent2: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    num_clusters: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let p1_neuron_off = compute_neuron_offsets(&parent1.1);
    let p2_neuron_off = compute_neuron_offsets(&parent2.1);
    let p1_conn_off = compute_conn_offsets(&parent1.0);
    let p2_conn_off = compute_conn_offsets(&parent2.0);

    let mut child_bits = Vec::new();
    let mut child_neurons = Vec::new();
    let mut child_conns = Vec::new();

    for c in 0..num_clusters {
        if rng.gen::<bool>() {
            // Take from parent1
            let ns = p1_neuron_off[c];
            let ne = p1_neuron_off[c + 1];
            child_neurons.push(parent1.1[c]);
            child_bits.extend_from_slice(&parent1.0[ns..ne]);
            if !parent1.2.is_empty() {
                let cs = p1_conn_off[ns];
                let ce = p1_conn_off[ne];
                child_conns.extend_from_slice(&parent1.2[cs..ce]);
            }
        } else {
            // Take from parent2
            let ns = p2_neuron_off[c];
            let ne = p2_neuron_off[c + 1];
            child_neurons.push(parent2.1[c]);
            child_bits.extend_from_slice(&parent2.0[ns..ne]);
            if !parent2.2.is_empty() {
                let cs = p2_conn_off[ns];
                let ce = p2_conn_off[ne];
                child_conns.extend_from_slice(&parent2.2[cs..ce]);
            }
        }
    }

    (child_bits, child_neurons, child_conns)
}

/// Connection-level crossover: per-connection coin flip if same architecture,
/// otherwise fall back to cluster-level crossover.
fn crossover_connections(
    parent1: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    parent2: &(Vec<usize>, Vec<usize>, Vec<i64>, f64),
    num_clusters: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    // Check if parents have identical architecture
    let same_arch = parent1.1 == parent2.1 && parent1.0 == parent2.0;

    if same_arch && !parent1.2.is_empty() && !parent2.2.is_empty() {
        // Per-connection coin flip (architecture is identical)
        let child_bits = parent1.0.clone();
        let child_neurons = parent1.1.clone();
        let child_conns: Vec<i64> = parent1.2.iter().zip(parent2.2.iter())
            .map(|(&c1, &c2)| if rng.gen::<bool>() { c1 } else { c2 })
            .collect();
        (child_bits, child_neurons, child_conns)
    } else {
        // Different architectures: fall back to cluster-level
        crossover_bits(parent1, parent2, num_clusters, rng)
    }
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
        phase_type: config.phase_type,
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
    live_progress: Option<&Arc<RwLock<Option<LiveProgress>>>>,
) -> OffspringSearchResult
where
    F: Fn(&[usize], &[usize], &[i64], usize) -> Vec<(f64, f64, f64)>,
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
    let mut best_ce_so_far = f64::MAX;
    let mut best_acc_so_far = 0.0_f64;
    let start_time = Instant::now();

    // Initialize live progress for this search
    if let Some(lp) = live_progress {
        if let Ok(mut guard) = lp.write() {
            if let Some(ref mut p) = *guard {
                p.generation = current_gen as i32;
                p.total_generations = total_gens as i32;
                p.phase = "ga_offspring".into();
                p.target_count = target_count;
                p.evaluated = 0;
                p.viable = Some(0);
                p.best_ce = f64::MAX;
                p.best_acc = 0.0;
                p.elapsed_secs = 0.0;
                p.updated_at = LiveProgress::now_unix();
            }
        }
    }

    // Build fingerprint set (bits + neurons + connections) from population
    let mut known_fps: std::collections::HashSet<(Vec<usize>, Vec<usize>, Vec<i64>)> =
        std::collections::HashSet::new();
    for (bits, neurons, conns, _) in population {
        known_fps.insert((bits.clone(), neurons.clone(), conns.clone()));
    }

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
                crossover(p1, p2, ga_config.num_clusters, ga_config.phase_type, &mut rng)
            } else {
                (p1.0.clone(), p1.1.clone(), p1.2.clone())
            };

            let (mut new_bits, mut new_neurons, mut new_conns) = mutate_ga(
                &child.0,
                &child.1,
                &child.2,
                ga_config,
                &mut rng,
            );

            // Re-mutate if offspring is a duplicate (up to 3 retries)
            for _ in 0..3 {
                let fp = (new_bits.clone(), new_neurons.clone(), new_conns.clone());
                if !known_fps.contains(&fp) {
                    break;
                }
                let remut = mutate_ga(&new_bits, &new_neurons, &new_conns, ga_config, &mut rng);
                new_bits = remut.0;
                new_neurons = remut.1;
                new_conns = remut.2;
            }
            known_fps.insert((new_bits.clone(), new_neurons.clone(), new_conns.clone()));

            batch_bits.extend(&new_bits);
            batch_neurons.extend(&new_neurons);
            batch_connections.extend(&new_conns);
            batch_genomes.push((new_bits, new_neurons, new_conns));
        }

        let results = evaluate_batch(&batch_bits, &batch_neurons, &batch_connections, batch_to_generate);

        for (i, (ce, acc, f1)) in results.iter().enumerate() {
            evaluated += 1;
            best_ce_so_far = best_ce_so_far.min(*ce);
            best_acc_so_far = best_acc_so_far.max(*acc);
            let (bits, neurons, conns) = &batch_genomes[i];

            let candidate = CandidateResult {
                bits_per_neuron: bits.clone(),
                neurons_per_cluster: neurons.clone(),
                connections: conns.clone(),
                cross_entropy: *ce,
                accuracy: *acc,
                f1_macro: *f1,
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

        // Update live progress after each batch
        if let Some(lp) = live_progress {
            if let Ok(mut guard) = lp.write() {
                if let Some(ref mut p) = *guard {
                    p.evaluated = evaluated;
                    p.viable = Some(passed.len());
                    p.best_ce = best_ce_so_far;
                    p.best_acc = best_acc_so_far;
                    p.elapsed_secs = start_time.elapsed().as_secs_f64();
                    p.updated_at = LiveProgress::now_unix();
                }
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
            phase_type: PhaseType::Bits,
        };

        assert_eq!(config.bits_delta_max(), 2); // 10% of 24 = 2.4 -> 2
        assert_eq!(config.neurons_delta_max(), 2); // 10% of 16 = 1.6 -> 2
    }

    /// Helper to create a test config for a specific phase.
    fn test_config(neurons: &[usize], phase: PhaseType) -> MutationConfig {
        MutationConfig {
            num_clusters: neurons.len(),
            neurons_per_cluster: neurons.to_vec(),
            mutable_clusters: None,
            min_bits: 4,
            max_bits: 20,
            min_neurons: 1,
            max_neurons: 15,
            bits_mutation_rate: 1.0,
            neurons_mutation_rate: 1.0,
            total_input_bits: 64,
            phase_type: phase,
        }
    }

    #[test]
    fn test_mutate_neurons_phase() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let neurons = vec![2, 3]; // 2 clusters, 5 neurons total
        let bits = vec![8, 10, 12, 12, 12];
        let connections: Vec<i64> = (0..54).collect(); // 8+10+12+12+12 = 54
        let config = test_config(&neurons, PhaseType::Neurons);

        let (new_bits, new_neurons, new_conns) = mutate_genome(
            &bits, &neurons, &connections, &config, &mut rng
        );

        // Neurons per cluster should be within bounds
        for &n in &new_neurons {
            assert!(n >= config.min_neurons && n <= config.max_neurons);
        }
        // Total neurons should match bits length
        assert_eq!(new_bits.len(), new_neurons.iter().sum::<usize>());
        // Connection count should match sum of bits
        assert_eq!(new_conns.len(), new_bits.iter().sum::<usize>());
        // Existing neurons should keep their EXACT connections
        // (first min(old_n, new_n) neurons per cluster should be verbatim)
        let old_conn_off = compute_conn_offsets(&bits);
        let new_conn_off = compute_conn_offsets(&new_bits);
        let old_neuron_off = compute_neuron_offsets(&neurons);
        let new_neuron_off = compute_neuron_offsets(&new_neurons);
        for c in 0..neurons.len() {
            let keep = neurons[c].min(new_neurons[c]);
            for local in 0..keep {
                let old_g = old_neuron_off[c] + local;
                let new_g = new_neuron_off[c] + local;
                // Bits unchanged
                assert_eq!(new_bits[new_g], bits[old_g]);
                // Connections unchanged
                let old_cs = old_conn_off[old_g];
                let new_cs = new_conn_off[new_g];
                let b = bits[old_g];
                assert_eq!(&new_conns[new_cs..new_cs + b], &connections[old_cs..old_cs + b]);
            }
        }
    }

    #[test]
    fn test_mutate_bits_phase_no_drift() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let neurons = vec![2, 2];
        let bits = vec![8, 8, 8, 8];
        let connections: Vec<i64> = (0..32).collect(); // 4 × 8 = 32
        let config = test_config(&neurons, PhaseType::Bits);

        let (new_bits, new_neurons, new_conns) = mutate_genome(
            &bits, &neurons, &connections, &config, &mut rng
        );

        // Neurons unchanged in bits phase
        assert_eq!(new_neurons, neurons);
        // Check all connections: unchanged connections should be EXACT copies
        let old_conn_off = compute_conn_offsets(&bits);
        let new_conn_off = compute_conn_offsets(&new_bits);
        for n_idx in 0..4 {
            let old_b = bits[n_idx];
            let new_b = new_bits[n_idx];
            let keep = old_b.min(new_b);
            // Existing connections preserved verbatim (no drift!)
            let old_cs = old_conn_off[n_idx];
            let new_cs = new_conn_off[n_idx];
            // If bits decreased, we kept a random subset; if increased, first old_b are preserved
            if new_b >= old_b {
                // All old connections should be at the start
                for i in 0..old_b {
                    assert_eq!(new_conns[new_cs + i], connections[old_cs + i],
                        "Connection {i} of neuron {n_idx} was drifted!");
                }
            }
        }
    }

    #[test]
    fn test_mutate_connections_phase_preserves_architecture() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let neurons = vec![2, 3];
        let bits = vec![8, 10, 12, 12, 12];
        let connections: Vec<i64> = (0..54).collect();
        let config = test_config(&neurons, PhaseType::Connections);

        let (new_bits, new_neurons, new_conns) = mutate_genome(
            &bits, &neurons, &connections, &config, &mut rng
        );

        // Architecture completely unchanged
        assert_eq!(new_bits, bits);
        assert_eq!(new_neurons, neurons);
        // Same number of connections
        assert_eq!(new_conns.len(), connections.len());
        // Some connections should have changed (rate=1.0)
        let changed = new_conns.iter().zip(connections.iter()).filter(|(a, b)| a != b).count();
        assert!(changed > 0, "No connections were perturbed");
    }

    #[test]
    fn test_crossover_neurons_phase() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let p1 = (
            vec![8, 8, 10, 12, 12],
            vec![2, 1, 2],
            (0..50i64).collect::<Vec<_>>(),
            10.0,
        );
        let p2 = (
            vec![6, 14, 14, 14, 16],
            vec![1, 3, 1],
            (100..164i64).collect::<Vec<_>>(),
            11.0,
        );

        let (child_bits, child_neurons, child_conns) = crossover(&p1, &p2, 3, PhaseType::Neurons, &mut rng);
        assert_eq!(child_neurons.len(), 3);
        assert_eq!(child_bits.len(), child_neurons.iter().sum::<usize>());
        assert_eq!(child_conns.len(), child_bits.iter().sum::<usize>());
    }

    #[test]
    fn test_crossover_bits_phase() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let p1 = (
            vec![8, 8, 10, 12, 12],
            vec![2, 1, 2],
            (0..50i64).collect::<Vec<_>>(),
            10.0,
        );
        let p2 = (
            vec![6, 14, 14, 14, 16],
            vec![1, 3, 1],
            (100..164i64).collect::<Vec<_>>(),
            11.0,
        );

        let (child_bits, child_neurons, child_conns) = crossover(&p1, &p2, 3, PhaseType::Bits, &mut rng);
        assert_eq!(child_neurons.len(), 3);
        assert_eq!(child_bits.len(), child_neurons.iter().sum::<usize>());
        assert_eq!(child_conns.len(), child_bits.iter().sum::<usize>());
        // Each cluster should come entirely from one parent
        for c in 0..3 {
            assert!(child_neurons[c] == p1.1[c] || child_neurons[c] == p2.1[c]);
        }
    }

    #[test]
    fn test_crossover_connections_same_arch() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // Same architecture, different connections
        let p1 = (
            vec![4, 4, 4],
            vec![1, 1, 1],
            vec![0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
            10.0,
        );
        let p2 = (
            vec![4, 4, 4],
            vec![1, 1, 1],
            vec![50, 51, 52, 53, 60, 61, 62, 63, 40, 41, 42, 43],
            11.0,
        );

        let (child_bits, child_neurons, child_conns) = crossover(&p1, &p2, 3, PhaseType::Connections, &mut rng);
        // Architecture unchanged
        assert_eq!(child_bits, p1.0);
        assert_eq!(child_neurons, p1.1);
        // Each connection from one parent
        for (i, &conn) in child_conns.iter().enumerate() {
            assert!(conn == p1.2[i] || conn == p2.2[i]);
        }
    }
}
