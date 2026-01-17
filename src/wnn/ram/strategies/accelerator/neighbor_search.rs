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

/// Logger that writes to file with immediate flush.
pub struct FileLogger {
    writer: Option<BufWriter<File>>,
}

impl FileLogger {
    pub fn new(log_path: Option<&str>) -> Self {
        let writer = log_path.and_then(|path| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()
                .map(|f| BufWriter::new(f))
        });
        Self { writer }
    }

    pub fn log(&mut self, msg: &str) {
        // Always print to stderr for immediate visibility
        eprintln!("{}", msg);

        // Also write to file if configured
        if let Some(ref mut writer) = self.writer {
            let timestamp = Local::now().format("%H:%M:%S");
            let _ = writeln!(writer, "{} | {}", timestamp, msg);
            let _ = writer.flush();
        }
    }
}

/// Mutate a genome to create a neighbor.
///
/// Mutations include:
/// - bits_per_cluster: ±delta with mutation_rate probability
/// - neurons_per_cluster: ±delta with mutation_rate probability
/// - connections: adjusted when architecture changes, small perturbations
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

    // Mutate architecture
    for i in 0..config.num_clusters {
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
    let batch_size = 10; // Evaluate in batches for efficiency

    // Generation prefix for logs
    let gen_prefix = match (generation, total_generations) {
        (Some(g), Some(t)) => format!("[Gen {:03}/{:03}]", g + 1, t),
        (Some(g), None) => format!("[Gen {:03}]", g + 1),
        _ => "[Search]".to_string(),
    };

    logger.log(&format!(
        "{} Starting neighbor search: target={}, max={}, threshold={:.2}%",
        gen_prefix, target_count, max_attempts, accuracy_threshold * 100.0
    ));

    while passed.len() < target_count && evaluated < max_attempts {
        // Generate a batch of candidates
        let batch_to_generate = batch_size.min(max_attempts - evaluated);
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

        // Evaluate the batch
        let results = crate::token_cache::evaluate_genomes_cached(
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

            // Log each evaluation
            logger.log(&format!(
                "{} Candidate {:04}/{:04}: CE={:.4}, Acc={:.2}% {}",
                gen_prefix,
                evaluated,
                max_attempts,
                ce,
                acc * 100.0,
                if *acc >= accuracy_threshold { "✓" } else { "" }
            ));

            if *acc >= accuracy_threshold {
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
    let batch_size = 10;

    let gen_prefix = match (generation, total_generations) {
        (Some(g), Some(t)) => format!("[Gen {:03}/{:03}]", g + 1, t),
        (Some(g), None) => format!("[Gen {:03}]", g + 1),
        _ => "[Search]".to_string(),
    };

    logger.log(&format!(
        "{} Starting neighbor search: target={}, max={}, threshold={:.2}%",
        gen_prefix, target_count, max_attempts, accuracy_threshold * 100.0
    ));

    while passed.len() < target_count && evaluated < max_attempts {
        let batch_to_generate = batch_size.min(max_attempts - evaluated);
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

        let results = crate::token_cache::evaluate_genomes_cached(
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

            logger.log(&format!(
                "{} Candidate {:04}/{:04}: CE={:.4}, Acc={:.2}% {}",
                gen_prefix,
                evaluated,
                max_attempts,
                ce,
                acc * 100.0,
                if *acc >= accuracy_threshold { "✓" } else { "" }
            ));

            if *acc >= accuracy_threshold {
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
        passed.extend(all_candidates.into_iter().take(need));

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_config() {
        let config = MutationConfig {
            num_clusters: 100,
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
