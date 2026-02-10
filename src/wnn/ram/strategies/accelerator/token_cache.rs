//! Token Cache - Persistent token storage with subset rotation for RAM evaluation.
//!
//! Holds all tokens (train, eval, test) in Rust memory for the entire session.
//! Pre-encodes tokens to bits once, then provides fast subset selection for
//! per-iteration/generation rotation without any Python→Rust data transfer.
//!
//! Key optimization: All token subsets are pre-computed at cache creation.
//! Per-evaluation calls just use indices to select which subset to use.

use rand::prelude::*;
use rand::SeedableRng;
use std::collections::HashMap;

/// Pre-computed token subset with all data needed for evaluation.
#[derive(Clone)]
pub struct TokenSubset {
    /// Encoded input bits: [num_examples * total_input_bits]
    pub input_bits: Vec<bool>,
    /// Target cluster indices: [num_examples]
    pub targets: Vec<i64>,
    /// Negative samples: [num_examples * num_negatives]
    pub negatives: Vec<i64>,
    /// Number of examples in this subset
    pub num_examples: usize,
}

/// Subset rotator that cycles through parts in randomized order.
pub struct SubsetRotator {
    num_parts: usize,
    current_cycle: Vec<usize>,
    position: usize,
    rng: rand::rngs::StdRng,
}

impl SubsetRotator {
    pub fn new(num_parts: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut cycle: Vec<usize> = (0..num_parts).collect();
        cycle.shuffle(&mut rng);

        Self {
            num_parts,
            current_cycle: cycle,
            position: 0,
            rng,
        }
    }

    /// Get the next subset index, advancing the rotation.
    pub fn next(&mut self) -> usize {
        let idx = self.current_cycle[self.position];
        self.position += 1;

        // Start new cycle if needed
        if self.position >= self.num_parts {
            self.position = 0;
            self.current_cycle.shuffle(&mut self.rng);
        }

        idx
    }

    /// Peek at next index without advancing.
    pub fn peek(&self) -> usize {
        self.current_cycle[self.position]
    }

    /// Reset to beginning with optional new seed.
    pub fn reset(&mut self, seed: Option<u64>) {
        if let Some(s) = seed {
            self.rng = rand::rngs::StdRng::seed_from_u64(s);
        }
        self.current_cycle = (0..self.num_parts).collect();
        self.current_cycle.shuffle(&mut self.rng);
        self.position = 0;
    }
}

/// Persistent token cache holding all tokens for the session.
///
/// Created once at session start, then used for all evaluations.
/// Provides zero-copy subset selection via pre-computed indices.
pub struct TokenCache {
    // Configuration
    vocab_size: usize,
    context_size: usize,
    bits_per_token: usize,
    total_input_bits: usize,
    num_negatives: usize,
    num_parts: usize,

    // Cluster ordering (token_id → cluster_id)
    cluster_map: Vec<i64>,

    // Semantic encoding table (token_id → semantic_bits)
    // If Some, tokens are encoded using this table instead of raw binary.
    // Similar tokens have similar bit patterns, enabling better generalization.
    encoding_table: Option<Vec<u64>>,

    // Pre-computed subsets for train
    train_subsets: Vec<TokenSubset>,
    // Pre-computed subset for eval (typically 1 subset = full eval)
    eval_subsets: Vec<TokenSubset>,
    // Pre-computed subset for test (typically 1 subset = full test)
    test_subsets: Vec<TokenSubset>,

    // Rotators for train/eval
    train_rotator: SubsetRotator,
    eval_rotator: SubsetRotator,

    // Full datasets for final evaluation
    full_train: TokenSubset,
    full_eval: TokenSubset,
}

impl TokenCache {
    /// Create a new token cache with all data pre-encoded and partitioned.
    ///
    /// # Arguments
    /// * `train_tokens` - Full training token sequence
    /// * `eval_tokens` - Full evaluation token sequence
    /// * `test_tokens` - Full test token sequence (optional, can be empty)
    /// * `vocab_size` - Vocabulary size
    /// * `context_size` - Context window size
    /// * `cluster_order` - Token IDs sorted by frequency (most frequent first)
    /// * `num_parts` - Number of subsets to divide train data into (e.g., 3 for thirds)
    /// * `num_negatives` - Number of negative samples per example
    /// * `seed` - Random seed for subset rotation and negative sampling
    /// * `encoding_table` - Optional semantic encoding table (token_id → semantic_bits).
    ///   If provided, tokens are encoded using learned semantic bits instead of raw binary.
    ///   Similar tokens will have similar bit patterns, enabling better generalization.
    /// * `encoding_bits` - Number of bits in semantic encoding (required if encoding_table provided)
    pub fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        _test_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        cluster_order: Vec<usize>,
        num_parts: usize,
        num_negatives: usize,
        seed: u64,
        encoding_table: Option<Vec<u64>>,
        encoding_bits: Option<usize>,
    ) -> Self {
        // Compute bits per token
        // Use encoding_bits if semantic encoding is provided, otherwise compute from vocab_size
        let bits_per_token = if encoding_table.is_some() && encoding_bits.is_some() {
            encoding_bits.unwrap()
        } else {
            (vocab_size as f64).log2().ceil() as usize
        };
        let total_input_bits = context_size * bits_per_token;

        // Build cluster map (token_id → cluster_id)
        let mut cluster_map = vec![0i64; vocab_size];
        for (cluster_id, &token_id) in cluster_order.iter().enumerate() {
            if token_id < vocab_size {
                cluster_map[token_id] = cluster_id as i64;
            }
        }

        // Compute token frequencies for negative sampling
        let mut freq_counts: HashMap<u32, usize> = HashMap::new();
        for &t in &train_tokens {
            *freq_counts.entry(t).or_insert(0) += 1;
        }
        let mut freq_sorted: Vec<(u32, usize)> = freq_counts.into_iter().collect();
        freq_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let top_k: Vec<u32> = freq_sorted.iter().take(1000.min(vocab_size)).map(|(t, _)| *t).collect();

        // Get encoding table reference for encoding functions
        let enc_table_ref = encoding_table.as_deref();

        // Encode full training data
        let full_train = Self::encode_tokens(
            &train_tokens,
            &cluster_map,
            &top_k,
            context_size,
            bits_per_token,
            total_input_bits,
            num_negatives,
            seed,
            enc_table_ref,
        );

        // Encode full eval data
        let full_eval = Self::encode_tokens(
            &eval_tokens,
            &cluster_map,
            &top_k,
            context_size,
            bits_per_token,
            total_input_bits,
            num_negatives,
            seed + 1,
            enc_table_ref,
        );

        // Divide train tokens into subsets
        let train_subsets = Self::create_subsets(
            &train_tokens,
            &cluster_map,
            &top_k,
            context_size,
            bits_per_token,
            total_input_bits,
            num_negatives,
            num_parts,
            seed,
            enc_table_ref,
        );

        // Eval typically uses 1 subset (full eval)
        let eval_subsets = vec![full_eval.clone()];

        // Test typically uses 1 subset (for now, empty)
        let test_subsets = vec![];

        Self {
            vocab_size,
            context_size,
            bits_per_token,
            total_input_bits,
            num_negatives,
            num_parts,
            cluster_map,
            encoding_table,
            train_subsets,
            eval_subsets,
            test_subsets,
            train_rotator: SubsetRotator::new(num_parts, seed + 100),
            eval_rotator: SubsetRotator::new(1, seed + 200), // Eval uses full
            full_train,
            full_eval,
        }
    }

    /// Encode a token sequence into training data.
    ///
    /// If encoding_table is provided, uses semantic encoding (similar tokens → similar bits).
    /// Otherwise falls back to raw binary encoding of token IDs.
    fn encode_tokens(
        tokens: &[u32],
        cluster_map: &[i64],
        top_k: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        num_negatives: usize,
        seed: u64,
        encoding_table: Option<&[u64]>,
    ) -> TokenSubset {
        let num_examples = tokens.len().saturating_sub(context_size);
        if num_examples == 0 {
            return TokenSubset {
                input_bits: vec![],
                targets: vec![],
                negatives: vec![],
                num_examples: 0,
            };
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Encode input bits
        let mut input_bits = vec![false; num_examples * total_input_bits];
        let mut targets = vec![0i64; num_examples];
        let mut negatives = vec![0i64; num_examples * num_negatives];

        for ex_idx in 0..num_examples {
            // Encode context tokens to bits
            for ctx_pos in 0..context_size {
                let token = tokens[ex_idx + ctx_pos] as usize;
                let bit_offset = ex_idx * total_input_bits + ctx_pos * bits_per_token;

                // Get encoded bits: semantic encoding if available, else raw binary
                let encoded_bits = if let Some(table) = encoding_table {
                    if token < table.len() {
                        table[token]
                    } else {
                        // Fallback to raw binary for out-of-vocab tokens
                        token as u64
                    }
                } else {
                    // Raw binary encoding (original behavior)
                    token as u64
                };

                for bit_idx in 0..bits_per_token {
                    let bit_pos = bits_per_token - 1 - bit_idx;
                    input_bits[bit_offset + bit_idx] = ((encoded_bits >> bit_pos) & 1) == 1;
                }
            }

            // Target cluster
            let target_token = tokens[ex_idx + context_size] as usize;
            targets[ex_idx] = if target_token < cluster_map.len() {
                cluster_map[target_token]
            } else {
                target_token as i64
            };

            // Negative samples from top-k
            let neg_offset = ex_idx * num_negatives;
            for k in 0..num_negatives {
                let neg_token = top_k[rng.gen_range(0..top_k.len())] as usize;
                negatives[neg_offset + k] = if neg_token < cluster_map.len() {
                    cluster_map[neg_token]
                } else {
                    neg_token as i64
                };
            }
        }

        TokenSubset {
            input_bits,
            targets,
            negatives,
            num_examples,
        }
    }

    /// Create subsets by dividing tokens into N parts.
    fn create_subsets(
        tokens: &[u32],
        cluster_map: &[i64],
        top_k: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        num_negatives: usize,
        num_parts: usize,
        seed: u64,
        encoding_table: Option<&[u64]>,
    ) -> Vec<TokenSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;

        (0..num_parts).map(|i| {
            let start = i * part_size;
            let end = if i == num_parts - 1 { n } else { (i + 1) * part_size };
            let part_tokens: Vec<u32> = tokens[start..end].to_vec();

            Self::encode_tokens(
                &part_tokens,
                cluster_map,
                top_k,
                context_size,
                bits_per_token,
                total_input_bits,
                num_negatives,
                seed + i as u64 * 1000,
                encoding_table,
            )
        }).collect()
    }

    /// Get the next train subset index (advances rotator).
    pub fn next_train_idx(&mut self) -> usize {
        self.train_rotator.next()
    }

    /// Get the next eval subset index (advances rotator).
    pub fn next_eval_idx(&mut self) -> usize {
        self.eval_rotator.next()
    }

    /// Get train subset by index.
    pub fn train_subset(&self, idx: usize) -> &TokenSubset {
        &self.train_subsets[idx]
    }

    /// Get eval subset by index.
    pub fn eval_subset(&self, idx: usize) -> &TokenSubset {
        &self.eval_subsets[idx.min(self.eval_subsets.len() - 1)]
    }

    /// Get full train data for final evaluation.
    pub fn full_train(&self) -> &TokenSubset {
        &self.full_train
    }

    /// Get full eval data for final evaluation.
    pub fn full_eval(&self) -> &TokenSubset {
        &self.full_eval
    }

    /// Get total input bits.
    pub fn total_input_bits(&self) -> usize {
        self.total_input_bits
    }

    /// Get vocab size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get number of negatives per example.
    pub fn num_negatives(&self) -> usize {
        self.num_negatives
    }

    /// Get number of train subsets.
    pub fn num_train_subsets(&self) -> usize {
        self.train_subsets.len()
    }

    /// Reset rotators with optional new seed.
    pub fn reset(&mut self, seed: Option<u64>) {
        self.train_rotator.reset(seed);
        self.eval_rotator.reset(seed.map(|s| s + 100));
    }
}

/// Evaluate genomes using cached token data.
///
/// This is the main evaluation function that uses pre-cached data.
/// No data copying occurs - just pointer arithmetic to select subsets.
pub fn evaluate_genomes_cached(
    cache: &TokenCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    empty_value: f32,
) -> Vec<(f64, f64)> {
    let train = cache.train_subset(train_subset_idx);
    let eval = cache.eval_subset(eval_subset_idx);

    // Delegate to the existing evaluation function
    crate::adaptive::evaluate_genomes_parallel(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.vocab_size(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_input_bits(),
        empty_value,
    )
}

/// Evaluate genomes using full cached data (for final evaluation).
pub fn evaluate_genomes_cached_full(
    cache: &TokenCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    empty_value: f32,
) -> Vec<(f64, f64)> {
    let train = cache.full_train();
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genomes_parallel(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.vocab_size(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_input_bits(),
        empty_value,
    )
}

/// Evaluate genomes using hybrid CPU+GPU parallel evaluation (4-8x speedup).
///
/// Uses memory pool for parallel training, GPU batch evaluation, and pipelining.
pub fn evaluate_genomes_cached_hybrid(
    cache: &TokenCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    empty_value: f32,
) -> Vec<(f64, f64)> {
    let train = cache.train_subset(train_subset_idx);
    let eval = cache.eval_subset(eval_subset_idx);

    // Always use parallel hybrid (fast, ~3.5s/genome)
    // Progress logging now built into parallel_hybrid when WNN_PROGRESS_LOG=1
    crate::adaptive::evaluate_genomes_parallel_hybrid(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.vocab_size(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_input_bits(),
        empty_value,
    )
}

/// Evaluate genomes using full cached data with hybrid CPU+GPU (4-8x speedup).
pub fn evaluate_genomes_cached_full_hybrid(
    cache: &TokenCache,
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    empty_value: f32,
) -> Vec<(f64, f64)> {
    let train = cache.full_train();
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genomes_parallel_hybrid(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        cache.vocab_size(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_input_bits(),
        empty_value,
    )
}

/// Evaluate a single genome WITH gating, returning both gated and non-gated metrics.
///
/// This function:
/// 1. Trains base RAM on full training data
/// 2. Trains gating model on training data (target gate = true only for target cluster)
/// 3. Evaluates WITHOUT gating → (ce, acc)
/// 4. Evaluates WITH gating → (gated_ce, gated_acc)
///
/// # Returns
/// (ce, accuracy, gated_ce, gated_accuracy)
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genome_with_gating(
    cache: &TokenCache,
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    neurons_per_gate: usize,
    bits_per_gate_neuron: usize,
    vote_threshold_frac: f32,
    empty_value: f32,
    gating_seed: u64,
) -> (f64, f64, f64, f64) {
    let train = cache.full_train();
    let eval = cache.full_eval();

    crate::adaptive::evaluate_genome_with_gating(
        bits_flat,
        neurons_flat,
        connections_flat,
        cache.vocab_size(),
        &train.input_bits,
        &train.targets,
        &train.negatives,
        train.num_examples,
        cache.num_negatives(),
        &eval.input_bits,
        &eval.targets,
        eval.num_examples,
        cache.total_input_bits(),
        empty_value,
        neurons_per_gate,
        bits_per_gate_neuron,
        vote_threshold_frac,
        gating_seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subset_rotator() {
        let mut rotator = SubsetRotator::new(3, 42);

        // Each cycle uses all 3 parts exactly once
        let mut cycle1: Vec<usize> = (0..3).map(|_| rotator.next()).collect();
        cycle1.sort();
        assert_eq!(cycle1, vec![0, 1, 2]);

        let mut cycle2: Vec<usize> = (0..3).map(|_| rotator.next()).collect();
        cycle2.sort();
        assert_eq!(cycle2, vec![0, 1, 2]);
    }

    #[test]
    fn test_token_encoding() {
        let tokens: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let cluster_map: Vec<i64> = (0..256).map(|i| i as i64).collect();
        let top_k: Vec<u32> = (0..100).collect();

        let subset = TokenCache::encode_tokens(
            &tokens,
            &cluster_map,
            &top_k,
            4,  // context_size
            8,  // bits_per_token
            32, // total_input_bits
            5,  // num_negatives
            42, // seed
            None, // encoding_table (no semantic encoding for this test)
        );

        // 8 tokens - 4 context = 4 examples
        assert_eq!(subset.num_examples, 4);
        assert_eq!(subset.targets.len(), 4);
        assert_eq!(subset.negatives.len(), 4 * 5);
    }
}
