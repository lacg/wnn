//! Per-Cluster Optimization Accelerator
//!
//! Rust + Metal acceleration for discriminative per-cluster optimization.
//! Each cluster is optimized independently with discriminative fitness:
//! - Positive: How strongly does this cluster vote when it SHOULD win?
//! - Negative: How strongly does this cluster vote when it should NOT win?
//! - Fitness = positive_strength - negative_strength
//!
//! Architecture:
//! - CPU (rayon): Parallel across clusters, parallel GA/TS evaluation
//! - Metal GPU: Batch address computation and vote aggregation
//! - Hybrid: Best of both for maximum throughput

use rayon::prelude::*;
use rustc_hash::FxHashMap;


// ============================================================================
// Configuration Types
// ============================================================================

/// Fitness computation mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitnessMode {
    /// Only reward correct predictions for this cluster
    PositiveOnly = 1,
    /// Also penalize when cluster wrongly wins
    PenalizeWins = 2,
    /// Penalize high votes even if doesn't win (smoothest gradient)
    PenalizeHighVotes = 3,
    /// Use actual cross-entropy (requires baseline precomputation, DEFAULT)
    CrossEntropy = 4,
    /// Simple discriminative: +vote when correct, -vote when wrong
    /// This is the most intuitive fitness: maximize separation between
    /// "when I should fire" vs "when I should NOT fire"
    SimpleDiscriminative = 5,
    /// Accuracy-based: count how often this cluster wins (highest vote)
    /// when it should win. Directly optimizes for top-1 accuracy.
    /// Requires global baseline to compare against other clusters.
    Accuracy = 6,
}

impl From<i32> for FitnessMode {
    fn from(v: i32) -> Self {
        match v {
            1 => FitnessMode::PositiveOnly,
            2 => FitnessMode::PenalizeWins,
            3 => FitnessMode::PenalizeHighVotes,
            4 => FitnessMode::CrossEntropy,
            5 => FitnessMode::SimpleDiscriminative,
            6 => FitnessMode::Accuracy,
            _ => FitnessMode::CrossEntropy,  // Default to CE
        }
    }
}

/// Configuration for optimizing one tier
#[derive(Clone, Debug)]
pub struct TierOptConfig {
    pub tier: usize,
    pub ga_gens: usize,
    pub ga_population: usize,
    pub ts_iters: usize,
    pub ts_neighbors: usize,
    pub mutation_rate: f64,
    pub enabled: bool,
    pub fitness_mode: FitnessMode,
}

impl Default for TierOptConfig {
    fn default() -> Self {
        Self {
            tier: 0,
            ga_gens: 50,
            ga_population: 30,
            ts_iters: 20,
            ts_neighbors: 30,
            mutation_rate: 0.01,
            enabled: true,
            fitness_mode: FitnessMode::SimpleDiscriminative,
        }
    }
}

/// Result of optimizing a single cluster
#[derive(Clone, Debug)]
pub struct ClusterOptResult {
    pub cluster_id: usize,
    pub tier: usize,
    pub initial_fitness: f64,
    pub final_fitness: f64,
    pub improvement_pct: f64,
    pub final_connectivity: Vec<i64>,
    pub generations_run: usize,
}

/// Result of optimizing a full tier
#[derive(Clone, Debug)]
pub struct TierOptResult {
    pub tier: usize,
    pub clusters_optimized: usize,
    pub total_improvement: f64,
    pub avg_improvement: f64,
    pub cluster_results: Vec<ClusterOptResult>,
}

// ============================================================================
// Core Accelerated Evaluator
// ============================================================================

/// High-performance evaluator for per-cluster optimization
///
/// Stores precomputed data for efficient batch evaluation:
/// - Training contexts as packed bits
/// - Indices per cluster for O(1) lookup
/// - Cached baseline votes for incremental updates
pub struct PerClusterEvaluator {
    // Training data
    train_contexts: Vec<u64>,        // Packed bits: [num_train * words_per_context]
    train_targets: Vec<usize>,       // Target cluster for each example
    num_train: usize,

    // Eval data
    eval_contexts: Vec<u64>,         // Packed bits
    eval_targets: Vec<usize>,
    num_eval: usize,

    // Dimensions
    context_bits: usize,
    words_per_context: usize,        // ceil(context_bits / 64)
    num_clusters: usize,

    // Cluster configuration
    cluster_to_neurons: FxHashMap<usize, (usize, usize)>,  // cluster_id -> (start, end)
    cluster_to_bits: FxHashMap<usize, usize>,              // cluster_id -> bits_per_neuron

    // Precomputed indices for fast lookup
    train_indices: FxHashMap<usize, Vec<usize>>,  // cluster_id -> train example indices
    eval_indices: FxHashMap<usize, Vec<usize>>,   // cluster_id -> eval example indices

    // Baseline votes for CE computation (populated by precompute_tier_baseline)
    // Only stores clusters in the current tier, indexed by position in tier_cluster_ids
    tier_cluster_ids: Vec<usize>,                        // Ordered list of cluster IDs in tier
    tier_cluster_idx: FxHashMap<usize, usize>,           // cluster_id -> index in tier_cluster_ids
    tier_baseline_votes: Vec<f64>,                       // [num_eval * num_tier_clusters] flattened
    tier_baseline_computed: bool,

    // Global baseline votes for true global CE computation
    // Stores votes for ALL clusters (not just current tier)
    global_baseline_votes: Vec<f64>,                     // [num_eval * num_clusters] flattened
    global_baseline_computed: bool,

    // Precomputed values for efficient incremental CE (populated after global baseline)
    // These enable O(num_eval) CE computation instead of O(num_eval × num_clusters)
    global_log_sum_exp: Vec<f64>,     // [num_eval] - log(Σ_c exp(vote_c)) per example
    global_baseline_ce: f64,          // Total CE before any optimization
}

impl PerClusterEvaluator {
    /// Create a new evaluator from training and eval data
    pub fn new(
        train_contexts_flat: &[bool],   // [num_train * context_bits] flattened
        train_targets: &[usize],
        eval_contexts_flat: &[bool],
        eval_targets: &[usize],
        context_bits: usize,
        cluster_to_neurons: FxHashMap<usize, (usize, usize)>,
        cluster_to_bits: FxHashMap<usize, usize>,
        num_clusters: usize,
    ) -> Self {
        let num_train = train_targets.len();
        let num_eval = eval_targets.len();
        let words_per_context = (context_bits + 63) / 64;

        // Pack training contexts into u64 words
        let train_contexts = Self::pack_contexts(train_contexts_flat, num_train, context_bits);
        let eval_contexts = Self::pack_contexts(eval_contexts_flat, num_eval, context_bits);

        // Precompute indices per cluster
        let mut train_indices: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        let mut eval_indices: FxHashMap<usize, Vec<usize>> = FxHashMap::default();

        for (i, &target) in train_targets.iter().enumerate() {
            train_indices.entry(target).or_default().push(i);
        }
        for (i, &target) in eval_targets.iter().enumerate() {
            eval_indices.entry(target).or_default().push(i);
        }

        Self {
            train_contexts,
            train_targets: train_targets.to_vec(),
            num_train,
            eval_contexts,
            eval_targets: eval_targets.to_vec(),
            num_eval,
            context_bits,
            words_per_context,
            num_clusters,
            cluster_to_neurons,
            cluster_to_bits,
            train_indices,
            eval_indices,
            tier_cluster_ids: Vec::new(),
            tier_cluster_idx: FxHashMap::default(),
            tier_baseline_votes: Vec::new(),
            tier_baseline_computed: false,
            global_baseline_votes: Vec::new(),
            global_baseline_computed: false,
            global_log_sum_exp: Vec::new(),
            global_baseline_ce: 0.0,
        }
    }

    /// Precompute baseline votes for all clusters in a tier
    ///
    /// This enables efficient CE computation during optimization.
    /// Only needs to be called once per tier before optimization.
    pub fn precompute_tier_baseline(
        &mut self,
        cluster_ids: &[usize],
        initial_connectivities: &FxHashMap<usize, Vec<i64>>,
    ) {
        let num_tier_clusters = cluster_ids.len();

        // Build index mapping
        self.tier_cluster_ids = cluster_ids.to_vec();
        self.tier_cluster_idx.clear();
        for (idx, &cid) in cluster_ids.iter().enumerate() {
            self.tier_cluster_idx.insert(cid, idx);
        }

        // Allocate baseline votes: [num_eval * num_tier_clusters]
        self.tier_baseline_votes = vec![0.0; self.num_eval * num_tier_clusters];

        // Compute votes for each cluster in parallel
        let votes_per_cluster: Vec<Vec<f64>> = cluster_ids
            .par_iter()
            .map(|&cid| {
                if let Some(conn) = initial_connectivities.get(&cid) {
                    self.train_and_vote(cid, conn)
                } else {
                    vec![0.0; self.num_eval]
                }
            })
            .collect();

        // Copy into flattened storage: [eval_idx * num_tier_clusters + cluster_idx]
        for (cluster_idx, votes) in votes_per_cluster.iter().enumerate() {
            for (eval_idx, &vote) in votes.iter().enumerate() {
                self.tier_baseline_votes[eval_idx * num_tier_clusters + cluster_idx] = vote;
            }
        }

        self.tier_baseline_computed = true;
    }

    /// Precompute baseline votes for ALL clusters (enables true global CE)
    ///
    /// This enables exact global CE computation during optimization.
    /// Memory: num_eval * num_clusters * 8 bytes (e.g., 10K * 50K * 8 = 4GB)
    /// Time: ~3s for 50K clusters (parallelized with rayon)
    ///
    /// Call this ONCE before optimization starts. After this, compute_global_ce_fitness()
    /// will use true global softmax over all 50K clusters.
    pub fn precompute_global_baseline(
        &mut self,
        all_connectivities: &FxHashMap<usize, Vec<i64>>,
    ) {
        // Get all cluster IDs (0..num_clusters)
        let all_cluster_ids: Vec<usize> = (0..self.num_clusters).collect();

        // Allocate global baseline votes: [num_eval * num_clusters]
        // Memory: 10K examples * 50K clusters * 8 bytes = 4GB
        self.global_baseline_votes = vec![0.0; self.num_eval * self.num_clusters];

        // Compute votes for each cluster in parallel
        let votes_per_cluster: Vec<Vec<f64>> = all_cluster_ids
            .par_iter()
            .map(|&cid| {
                if let Some(conn) = all_connectivities.get(&cid) {
                    self.train_and_vote(cid, conn)
                } else {
                    vec![0.0; self.num_eval]
                }
            })
            .collect();

        // Copy into flattened storage: [eval_idx * num_clusters + cluster_id]
        for (cluster_id, votes) in votes_per_cluster.iter().enumerate() {
            for (eval_idx, &vote) in votes.iter().enumerate() {
                self.global_baseline_votes[eval_idx * self.num_clusters + cluster_id] = vote;
            }
        }

        self.global_baseline_computed = true;

        // Precompute log_sum_exp for each example (enables O(num_eval) fitness computation)
        self.global_log_sum_exp = vec![0.0; self.num_eval];
        self.global_baseline_ce = 0.0;

        for eval_idx in 0..self.num_eval {
            // Find max vote for numerical stability
            let mut max_vote = f64::NEG_INFINITY;
            for cid in 0..self.num_clusters {
                let vote = self.get_global_baseline(eval_idx, cid);
                if vote > max_vote {
                    max_vote = vote;
                }
            }

            // Compute log-sum-exp: log(Σ exp(v - max)) + max
            let mut sum_exp = 0.0;
            for cid in 0..self.num_clusters {
                let vote = self.get_global_baseline(eval_idx, cid);
                sum_exp += (vote - max_vote).exp();
            }
            self.global_log_sum_exp[eval_idx] = sum_exp.ln() + max_vote;

            // Compute baseline CE for this example
            let target = self.eval_targets[eval_idx];
            let target_vote = self.get_global_baseline(eval_idx, target);
            self.global_baseline_ce += -target_vote + self.global_log_sum_exp[eval_idx];
        }
    }

    /// Get global baseline vote for a cluster at an eval example
    #[inline]
    fn get_global_baseline(&self, eval_idx: usize, cluster_id: usize) -> f64 {
        self.global_baseline_votes[eval_idx * self.num_clusters + cluster_id]
    }

    /// Update global baseline for a specific cluster (called after optimization)
    ///
    /// Also updates the precomputed log_sum_exp values incrementally.
    pub fn update_global_baseline(&mut self, cluster_id: usize, new_votes: &[f64]) {
        if !self.global_baseline_computed {
            return;
        }

        // Update baseline CE by computing the delta
        for eval_idx in 0..self.num_eval {
            let old_vote = self.get_global_baseline(eval_idx, cluster_id);
            let new_vote = new_votes[eval_idx];

            // Skip if no change
            if (old_vote - new_vote).abs() < 1e-10 {
                continue;
            }

            // Update the stored vote
            self.global_baseline_votes[eval_idx * self.num_clusters + cluster_id] = new_vote;

            // Update log_sum_exp incrementally:
            // old_lse = log(sum_exp) where sum_exp includes old_vote
            // new_lse = log(sum_exp - exp(old_vote) + exp(new_vote))
            //         = log(exp(old_lse) - exp(old_vote) + exp(new_vote))
            let old_lse = self.global_log_sum_exp[eval_idx];

            // Use numerically stable update
            // new_sum_exp = exp(old_lse) * (1 - exp(old_vote - old_lse) + exp(new_vote - old_lse))
            let scale = 1.0 - (old_vote - old_lse).exp() + (new_vote - old_lse).exp();
            let new_lse = old_lse + scale.ln();
            self.global_log_sum_exp[eval_idx] = new_lse;

            // Update baseline CE if target is this cluster
            let target = self.eval_targets[eval_idx];
            if target == cluster_id {
                // CE changes from (-old_vote + old_lse) to (-new_vote + new_lse)
                let old_ce = -old_vote + old_lse;
                let new_ce = -new_vote + new_lse;
                self.global_baseline_ce += new_ce - old_ce;
            } else {
                // CE changes from (-target_vote + old_lse) to (-target_vote + new_lse)
                // Delta = new_lse - old_lse
                self.global_baseline_ce += new_lse - old_lse;
            }
        }
    }

    /// Get baseline vote for a cluster at an eval example
    #[inline]
    fn get_tier_baseline(&self, eval_idx: usize, cluster_idx: usize) -> f64 {
        let num_tier_clusters = self.tier_cluster_ids.len();
        self.tier_baseline_votes[eval_idx * num_tier_clusters + cluster_idx]
    }

    /// Set baseline vote for a cluster at an eval example (for temporary updates)
    #[inline]
    fn set_tier_baseline(&mut self, eval_idx: usize, cluster_idx: usize, value: f64) {
        let num_tier_clusters = self.tier_cluster_ids.len();
        self.tier_baseline_votes[eval_idx * num_tier_clusters + cluster_idx] = value;
    }

    /// Pack bool contexts into u64 words for efficient address computation
    fn pack_contexts(flat: &[bool], num_examples: usize, bits_per_example: usize) -> Vec<u64> {
        let words_per = (bits_per_example + 63) / 64;
        let mut packed = vec![0u64; num_examples * words_per];

        for ex in 0..num_examples {
            let ex_start = ex * bits_per_example;
            let packed_start = ex * words_per;

            for bit in 0..bits_per_example {
                if ex_start + bit < flat.len() && flat[ex_start + bit] {
                    let word_idx = bit / 64;
                    let bit_pos = bit % 64;
                    packed[packed_start + word_idx] |= 1u64 << bit_pos;
                }
            }
        }

        packed
    }

    /// Get packed context for an example
    #[inline]
    fn get_context<'a>(&self, contexts: &'a [u64], example_idx: usize) -> &'a [u64] {
        let start = example_idx * self.words_per_context;
        let end = start + self.words_per_context;
        &contexts[start..end]
    }

    /// Compute RAM address from context and connectivity
    #[inline]
    fn compute_address(context: &[u64], connectivity: &[i64]) -> u64 {
        let mut addr = 0u64;
        for (i, &conn_bit) in connectivity.iter().enumerate() {
            let bit_idx = conn_bit as usize;
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if word_idx < context.len() && (context[word_idx] >> bit_pos) & 1 == 1 {
                addr |= 1u64 << i;
            }
        }
        addr
    }

    // ========================================================================
    // Single Cluster Evaluation
    // ========================================================================

    /// Train a cluster's neurons and compute votes on eval set
    ///
    /// Returns votes for each eval example (sum of neuron activations)
    pub fn train_and_vote(
        &self,
        cluster_id: usize,
        connectivity: &[i64],  // [num_neurons * bits_per_neuron] flattened
    ) -> Vec<f64> {
        let bits_per_neuron = *self.cluster_to_bits.get(&cluster_id).unwrap_or(&8);
        let num_neurons = connectivity.len() / bits_per_neuron;

        // Get training examples for this cluster
        let train_idx = self.train_indices.get(&cluster_id);

        // Train RAMs (one HashMap per neuron: address -> count)
        let mut rams: Vec<FxHashMap<u64, u32>> = vec![FxHashMap::default(); num_neurons];

        if let Some(indices) = train_idx {
            for &ex_idx in indices {
                let ctx = self.get_context(&self.train_contexts, ex_idx);
                for n in 0..num_neurons {
                    let conn_start = n * bits_per_neuron;
                    let conn_end = conn_start + bits_per_neuron;
                    let neuron_conn = &connectivity[conn_start..conn_end];
                    let addr = Self::compute_address(ctx, neuron_conn);
                    *rams[n].entry(addr).or_insert(0) += 1;
                }
            }
        }

        // Compute votes on eval set
        let mut votes = vec![0.0; self.num_eval];

        for ex_idx in 0..self.num_eval {
            let ctx = self.get_context(&self.eval_contexts, ex_idx);
            let mut vote = 0.0;

            for n in 0..num_neurons {
                let conn_start = n * bits_per_neuron;
                let conn_end = conn_start + bits_per_neuron;
                let neuron_conn = &connectivity[conn_start..conn_end];
                let addr = Self::compute_address(ctx, neuron_conn);

                if rams[n].contains_key(&addr) {
                    vote += 1.0;  // Neuron fires
                }
            }

            votes[ex_idx] = vote;
        }

        votes
    }

    /// Compute discriminative fitness for a cluster
    ///
    /// For PositiveOnly/PenalizeWins/PenalizeHighVotes:
    ///   Fitness = positive_strength - negative_strength
    ///
    /// For CrossEntropy (requires precompute_tier_baseline first):
    ///   Fitness = -mean_cross_entropy (higher is better)
    ///   Uses tier-local softmax over clusters in the current tier
    fn compute_fitness(
        &self,
        cluster_id: usize,
        votes: &[f64],
        fitness_mode: FitnessMode,
    ) -> f64 {
        // CrossEntropy mode: compute actual CE using tier baseline
        if fitness_mode == FitnessMode::CrossEntropy {
            return self.compute_ce_fitness(cluster_id, votes);
        }

        let pos_idx = self.eval_indices.get(&cluster_id);

        // Positive: average vote when this cluster should fire
        let pos_strength = if let Some(indices) = pos_idx {
            if indices.is_empty() {
                0.0
            } else {
                indices.iter().map(|&i| votes[i]).sum::<f64>() / indices.len() as f64
            }
        } else {
            0.0
        };

        match fitness_mode {
            FitnessMode::PositiveOnly => pos_strength,

            FitnessMode::PenalizeWins | FitnessMode::PenalizeHighVotes => {
                // Negative: average vote when this cluster should NOT fire
                // Sample for efficiency (max 2000)
                let mut neg_sum = 0.0;
                let mut neg_count = 0;
                let sample_limit = 2000;

                for (ex_idx, &target) in self.eval_targets.iter().enumerate() {
                    if target != cluster_id {
                        neg_sum += votes[ex_idx];
                        neg_count += 1;
                        if neg_count >= sample_limit {
                            break;
                        }
                    }
                }

                let neg_strength = if neg_count > 0 {
                    neg_sum / neg_count as f64
                } else {
                    0.0
                };

                pos_strength - neg_strength
            }

            FitnessMode::SimpleDiscriminative => {
                // Simple discriminative: +vote when correct, -vote when wrong
                // This maximizes the separation between "should fire" vs "should not fire"
                // Uses ALL examples (no sampling) for accurate fitness
                let mut total = 0.0;
                for (ex_idx, &target) in self.eval_targets.iter().enumerate() {
                    if target == cluster_id {
                        total += votes[ex_idx];  // Reward: should fire high
                    } else {
                        total -= votes[ex_idx];  // Penalize: should NOT fire
                    }
                }
                // Return mean (normalized by number of examples)
                total / self.num_eval as f64
            }

            FitnessMode::Accuracy => {
                // Accuracy-based: count how often this cluster wins (highest vote) when it should
                // Directly optimizes for top-1 accuracy
                // Requires global baseline to compare against other clusters
                if !self.global_baseline_computed {
                    // Fall back to simple discriminative if no baseline
                    let mut total = 0.0;
                    for (ex_idx, &target) in self.eval_targets.iter().enumerate() {
                        if target == cluster_id {
                            total += votes[ex_idx];
                        } else {
                            total -= votes[ex_idx];
                        }
                    }
                    return total / self.num_eval as f64;
                }

                // Get positive examples for this cluster
                let pos_indices = match self.eval_indices.get(&cluster_id) {
                    Some(idx) => idx,
                    None => return 0.0,
                };

                if pos_indices.is_empty() {
                    return 0.0;
                }

                let mut wins = 0;
                for &ex_idx in pos_indices.iter() {
                    let our_vote = votes[ex_idx];

                    // Check if we beat all other clusters
                    let mut is_winner = true;
                    for other_cluster in 0..self.num_clusters {
                        if other_cluster != cluster_id {
                            let other_vote = self.global_baseline_votes
                                [ex_idx * self.num_clusters + other_cluster];
                            if other_vote >= our_vote {
                                is_winner = false;
                                break;
                            }
                        }
                    }

                    if is_winner {
                        wins += 1;
                    }
                }

                // Return accuracy (proportion of wins)
                wins as f64 / pos_indices.len() as f64
            }

            FitnessMode::CrossEntropy => unreachable!(), // Handled above
        }
    }

    /// Compute cross-entropy fitness
    ///
    /// Prefers global CE (softmax over all 50K clusters) when available.
    /// Falls back to tier-local CE when only tier baseline is computed.
    ///
    /// Returns negative CE (higher is better, for maximization)
    fn compute_ce_fitness(&self, cluster_id: usize, new_votes: &[f64]) -> f64 {
        // Prefer global CE when available (true global softmax over all clusters)
        if self.global_baseline_computed {
            return self.compute_global_ce_fitness(cluster_id, new_votes);
        }

        // Fall back to tier-local CE if only tier baseline is computed
        if !self.tier_baseline_computed {
            return self.compute_fallback_fitness(cluster_id, new_votes);
        }

        let cluster_idx = match self.tier_cluster_idx.get(&cluster_id) {
            Some(&idx) => idx,
            None => return self.compute_fallback_fitness(cluster_id, new_votes),
        };

        let num_tier_clusters = self.tier_cluster_ids.len();
        let mut total_ce = 0.0;
        let mut count = 0;

        // Only compute CE for examples where target is in this tier
        for eval_idx in 0..self.num_eval {
            let target = self.eval_targets[eval_idx];

            // Check if target is in this tier
            let target_idx = match self.tier_cluster_idx.get(&target) {
                Some(&idx) => idx,
                None => continue, // Skip examples with targets outside this tier
            };

            // Compute softmax denominator using baseline votes + new votes for cluster_id
            let mut max_vote = f64::NEG_INFINITY;
            for cidx in 0..num_tier_clusters {
                let vote = if cidx == cluster_idx {
                    new_votes[eval_idx]
                } else {
                    self.get_tier_baseline(eval_idx, cidx)
                };
                if vote > max_vote {
                    max_vote = vote;
                }
            }

            // Compute log-sum-exp for numerical stability
            let mut sum_exp = 0.0;
            for cidx in 0..num_tier_clusters {
                let vote = if cidx == cluster_idx {
                    new_votes[eval_idx]
                } else {
                    self.get_tier_baseline(eval_idx, cidx)
                };
                sum_exp += (vote - max_vote).exp();
            }

            // Get vote for target cluster
            let target_vote = if target_idx == cluster_idx {
                new_votes[eval_idx]
            } else {
                self.get_tier_baseline(eval_idx, target_idx)
            };

            // CE = -log(softmax(target)) = -target_vote + max_vote + log(sum_exp)
            let ce = -target_vote + max_vote + sum_exp.ln();
            total_ce += ce;
            count += 1;
        }

        if count == 0 {
            return 0.0;
        }

        // Return negative mean CE (higher is better)
        -total_ce / count as f64
    }

    /// Compute cross-entropy fitness using TRUE GLOBAL softmax (over all 50K clusters)
    ///
    /// EFFICIENT INCREMENTAL VERSION: O(num_eval) instead of O(num_eval × num_clusters)
    ///
    /// Uses precomputed log_sum_exp values to compute CE delta when changing
    /// cluster_id's votes from baseline to new_votes.
    ///
    /// - For target == cluster_id: higher votes = better (increases numerator)
    /// - For target != cluster_id: higher votes = worse (increases denominator)
    ///
    /// Requires precompute_global_baseline() to be called first.
    /// Returns negative CE (higher is better, for maximization)
    fn compute_global_ce_fitness(&self, cluster_id: usize, new_votes: &[f64]) -> f64 {
        if !self.global_baseline_computed || self.global_log_sum_exp.is_empty() {
            // Fallback to discriminative fitness if global baseline not available
            return self.compute_fallback_fitness(cluster_id, new_votes);
        }

        // Compute total CE by applying incremental changes from baseline
        let mut total_ce = self.global_baseline_ce;

        for eval_idx in 0..self.num_eval {
            let old_vote = self.get_global_baseline(eval_idx, cluster_id);
            let new_vote = new_votes[eval_idx];

            // Skip if no change (avoids numerical issues)
            if (old_vote - new_vote).abs() < 1e-10 {
                continue;
            }

            let old_lse = self.global_log_sum_exp[eval_idx];

            // Compute new log_sum_exp incrementally:
            // new_sum_exp = exp(old_lse) - exp(old_vote) + exp(new_vote)
            //             = exp(old_lse) * (1 - exp(old_vote - old_lse) + exp(new_vote - old_lse))
            let scale = 1.0 - (old_vote - old_lse).exp() + (new_vote - old_lse).exp();

            // Handle numerical edge case where scale could be <= 0
            if scale <= 0.0 {
                // This shouldn't happen in normal operation, but handle gracefully
                continue;
            }

            let new_lse = old_lse + scale.ln();

            // Update CE delta based on target
            let target = self.eval_targets[eval_idx];
            if target == cluster_id {
                // CE changes from (-old_vote + old_lse) to (-new_vote + new_lse)
                let delta = (-new_vote + new_lse) - (-old_vote + old_lse);
                total_ce += delta;
            } else {
                // CE changes from (-target_vote + old_lse) to (-target_vote + new_lse)
                let delta = new_lse - old_lse;
                total_ce += delta;
            }
        }

        if self.num_eval == 0 {
            return 0.0;
        }

        // Return negative mean CE (higher is better)
        -total_ce / self.num_eval as f64
    }

    /// Fallback fitness when tier baseline not computed
    fn compute_fallback_fitness(&self, cluster_id: usize, votes: &[f64]) -> f64 {
        let pos_idx = self.eval_indices.get(&cluster_id);
        let pos_strength = if let Some(indices) = pos_idx {
            if indices.is_empty() {
                0.0
            } else {
                indices.iter().map(|&i| votes[i]).sum::<f64>() / indices.len() as f64
            }
        } else {
            0.0
        };
        pos_strength
    }

    /// Evaluate a single connectivity variant for a cluster
    pub fn evaluate_cluster_variant(
        &self,
        cluster_id: usize,
        connectivity: &[i64],
        fitness_mode: FitnessMode,
    ) -> f64 {
        let votes = self.train_and_vote(cluster_id, connectivity);
        self.compute_fitness(cluster_id, &votes, fitness_mode)
    }

    // ========================================================================
    // Batch Evaluation (for GA population / TS neighbors)
    // ========================================================================

    /// Evaluate multiple connectivity variants for a cluster in parallel
    ///
    /// This is the key acceleration point - evaluates entire GA population
    /// or TS neighbor set in parallel using rayon.
    pub fn evaluate_variants_batch(
        &self,
        cluster_id: usize,
        variants: &[Vec<i64>],  // [num_variants][neurons * bits]
        fitness_mode: FitnessMode,
    ) -> Vec<f64> {
        variants
            .par_iter()
            .map(|conn| self.evaluate_cluster_variant(cluster_id, conn, fitness_mode))
            .collect()
    }

    // ========================================================================
    // GA Optimization
    // ========================================================================

    /// Optimize a cluster using Genetic Algorithm
    pub fn optimize_cluster_ga(
        &self,
        cluster_id: usize,
        initial_connectivity: &[i64],
        config: &TierOptConfig,
        rng_seed: u64,
    ) -> ClusterOptResult {
        let bits_per_neuron = *self.cluster_to_bits.get(&cluster_id).unwrap_or(&8);
        let num_neurons = initial_connectivity.len() / bits_per_neuron;
        let total_bits = self.context_bits;

        // Simple LCG RNG for reproducibility (using struct to avoid closure issues)
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                self.0
            }
            fn float(&mut self) -> f64 {
                self.next() as f64 / u64::MAX as f64
            }
            fn range(&mut self, max: usize) -> usize {
                (self.next() as usize) % max.max(1)
            }
        }
        let mut rng = Rng(rng_seed);

        // Initialize population
        let pop_size = config.ga_population;
        let mut population: Vec<Vec<i64>> = Vec::with_capacity(pop_size);
        population.push(initial_connectivity.to_vec());

        for _ in 1..pop_size {
            let mut variant = vec![0i64; initial_connectivity.len()];
            for i in 0..variant.len() {
                variant[i] = rng.range(total_bits) as i64;
            }
            population.push(variant);
        }

        // Evaluate initial population
        let mut fitness = self.evaluate_variants_batch(
            cluster_id,
            &population,
            config.fitness_mode,
        );

        let initial_fitness = fitness[0];  // First is the original
        let elitism = (pop_size / 10).max(1);

        // GA generations
        for _gen in 0..config.ga_gens {
            // Sort by fitness (descending)
            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitness[b].partial_cmp(&fitness[a]).unwrap());

            // Create next generation
            let mut next_pop: Vec<Vec<i64>> = Vec::with_capacity(pop_size);

            // Elitism: keep best
            for &idx in indices.iter().take(elitism) {
                next_pop.push(population[idx].clone());
            }

            // Fill rest with crossover + mutation
            while next_pop.len() < pop_size {
                // Tournament selection
                let parent1_idx = {
                    let a = rng.range(pop_size);
                    let b = rng.range(pop_size);
                    if fitness[a] > fitness[b] { a } else { b }
                };
                let parent2_idx = {
                    let a = rng.range(pop_size);
                    let b = rng.range(pop_size);
                    if fitness[a] > fitness[b] { a } else { b }
                };

                // Crossover at neuron boundary
                let crossover_neuron = rng.range(num_neurons.max(1));
                let crossover_point = crossover_neuron * bits_per_neuron;

                let mut child = population[parent1_idx][..crossover_point].to_vec();
                child.extend_from_slice(&population[parent2_idx][crossover_point..]);

                // Mutation
                for i in 0..child.len() {
                    if rng.float() < config.mutation_rate {
                        child[i] = rng.range(total_bits) as i64;
                    }
                }

                next_pop.push(child);
            }

            // Evaluate new population
            population = next_pop;
            fitness = self.evaluate_variants_batch(
                cluster_id,
                &population,
                config.fitness_mode,
            );
        }

        // Find best
        let best_idx = fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let final_fitness = fitness[best_idx];
        let improvement = if initial_fitness != 0.0 {
            ((final_fitness - initial_fitness) / initial_fitness.abs()) * 100.0
        } else {
            0.0
        };

        ClusterOptResult {
            cluster_id,
            tier: config.tier,
            initial_fitness,
            final_fitness,
            improvement_pct: improvement,
            final_connectivity: population[best_idx].clone(),
            generations_run: config.ga_gens,
        }
    }

    /// Optimize a cluster using Tabu Search
    pub fn optimize_cluster_ts(
        &self,
        cluster_id: usize,
        initial_connectivity: &[i64],
        config: &TierOptConfig,
        rng_seed: u64,
    ) -> ClusterOptResult {
        let total_bits = self.context_bits;

        // Simple LCG RNG for reproducibility
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                self.0
            }
            fn range(&mut self, max: usize) -> usize {
                (self.next() as usize) % max.max(1)
            }
        }
        let mut rng = Rng(rng_seed);

        let initial_fitness = self.evaluate_cluster_variant(
            cluster_id,
            initial_connectivity,
            config.fitness_mode,
        );

        let mut current = initial_connectivity.to_vec();
        let mut current_fitness = initial_fitness;
        let mut best = current.clone();
        let mut best_fitness = current_fitness;

        // Tabu list: (position, old_value, new_value)
        let mut tabu_list: Vec<(usize, i64, i64)> = Vec::with_capacity(10);
        let tabu_tenure = 5;

        for _iter in 0..config.ts_iters {
            // Generate neighbors
            let mut neighbors: Vec<Vec<i64>> = Vec::with_capacity(config.ts_neighbors);
            let mut moves: Vec<(usize, i64, i64)> = Vec::with_capacity(config.ts_neighbors);

            for _ in 0..config.ts_neighbors {
                let mut neighbor = current.clone();

                // Make at least one mutation
                let pos = rng.range(neighbor.len());
                let old_val = neighbor[pos];
                let new_val = rng.range(total_bits) as i64;
                neighbor[pos] = new_val;

                // Check if tabu (reverse of recent move)
                let is_tabu = tabu_list.iter().any(|(p, _, nv)| *p == pos && *nv == old_val);

                if !is_tabu {
                    neighbors.push(neighbor);
                    moves.push((pos, old_val, new_val));
                }
            }

            if neighbors.is_empty() {
                continue;
            }

            // Evaluate neighbors in batch
            let neighbor_fitness = self.evaluate_variants_batch(
                cluster_id,
                &neighbors,
                config.fitness_mode,
            );

            // Pick best neighbor (even if worse - TS characteristic)
            let best_neighbor_idx = neighbor_fitness
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            current = neighbors[best_neighbor_idx].clone();
            current_fitness = neighbor_fitness[best_neighbor_idx];

            // Add move to tabu list
            tabu_list.push(moves[best_neighbor_idx]);
            if tabu_list.len() > tabu_tenure {
                tabu_list.remove(0);
            }

            // Update best
            if current_fitness > best_fitness {
                best = current.clone();
                best_fitness = current_fitness;
            }
        }

        let improvement = if initial_fitness != 0.0 {
            ((best_fitness - initial_fitness) / initial_fitness.abs()) * 100.0
        } else {
            0.0
        };

        ClusterOptResult {
            cluster_id,
            tier: config.tier,
            initial_fitness,
            final_fitness: best_fitness,
            improvement_pct: improvement,
            final_connectivity: best,
            generations_run: config.ts_iters,
        }
    }

    // ========================================================================
    // Tier-Level Optimization (parallel across clusters)
    // ========================================================================

    /// Optimize all clusters in a tier using parallel processing
    ///
    /// Each cluster is optimized independently in parallel using rayon.
    pub fn optimize_tier(
        &mut self,
        tier: usize,
        cluster_ids: &[usize],
        initial_connectivities: &FxHashMap<usize, Vec<i64>>,
        config: &TierOptConfig,
        base_seed: u64,
    ) -> TierOptResult {
        // Precompute tier baseline for CE fitness (essential for correct optimization)
        self.precompute_tier_baseline(cluster_ids, initial_connectivities);

        // Optimize all clusters in parallel
        let cluster_results: Vec<ClusterOptResult> = cluster_ids
            .par_iter()
            .enumerate()
            .map(|(_i, &cluster_id)| {
                let seed = base_seed.wrapping_add(cluster_id as u64);
                let initial_conn = initial_connectivities
                    .get(&cluster_id)
                    .expect("Missing connectivity for cluster");

                // Run GA then TS
                let mut result = if config.ga_gens > 0 {
                    self.optimize_cluster_ga(cluster_id, initial_conn, config, seed)
                } else {
                    ClusterOptResult {
                        cluster_id,
                        tier: config.tier,
                        initial_fitness: 0.0,
                        final_fitness: 0.0,
                        improvement_pct: 0.0,
                        final_connectivity: initial_conn.clone(),
                        generations_run: 0,
                    }
                };

                if config.ts_iters > 0 {
                    result = self.optimize_cluster_ts(
                        cluster_id,
                        &result.final_connectivity,
                        config,
                        seed + 1000,
                    );
                }

                result
            })
            .collect();

        // Aggregate results
        let total_improvement: f64 = cluster_results.iter().map(|r| r.improvement_pct).sum();
        let avg_improvement = if cluster_results.is_empty() {
            0.0
        } else {
            total_improvement / cluster_results.len() as f64
        };

        TierOptResult {
            tier,
            clusters_optimized: cluster_results.len(),
            total_improvement,
            avg_improvement,
            cluster_results,
        }
    }

    // ========================================================================
    // Group Optimization (joint optimization of multiple clusters)
    // ========================================================================

    /// Compute joint fitness for a group of clusters
    ///
    /// This computes the MEAN fitness across all clusters in the group.
    /// Used for joint optimization where clusters are optimized together.
    fn compute_group_fitness(
        &self,
        cluster_ids: &[usize],
        cluster_connectivities: &[Vec<i64>],  // One per cluster in group
        fitness_mode: FitnessMode,
    ) -> f64 {
        if !self.global_baseline_computed {
            return 0.0;
        }

        let mut total_fitness = 0.0;
        let mut count = 0;

        for (i, &cluster_id) in cluster_ids.iter().enumerate() {
            // Compute votes for this cluster with its connectivity
            let votes = self.train_and_vote(cluster_id, &cluster_connectivities[i]);

            // Compute fitness using the specified mode
            let fitness = self.compute_fitness(cluster_id, &votes, fitness_mode);
            total_fitness += fitness;
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            total_fitness / count as f64
        }
    }

    /// Optimize a group of clusters jointly using GA
    ///
    /// Instead of optimizing each cluster independently, this optimizes
    /// all clusters in the group together. The fitness is the mean global CE
    /// across all clusters in the group.
    ///
    /// This captures inter-cluster competition: if cluster A becomes stronger,
    /// cluster B must adapt, and vice versa.
    pub fn optimize_group_ga(
        &self,
        cluster_ids: &[usize],
        initial_connectivities: &[Vec<i64>],  // One per cluster
        config: &TierOptConfig,
        rng_seed: u64,
    ) -> Vec<ClusterOptResult> {
        let total_bits = self.context_bits;
        let group_size = cluster_ids.len();

        if group_size == 0 {
            return vec![];
        }

        // Get bits per cluster (assume all same within group)
        let bits_per_cluster: Vec<usize> = cluster_ids
            .iter()
            .map(|&cid| *self.cluster_to_bits.get(&cid).unwrap_or(&8))
            .collect();

        let _neurons_per_cluster: Vec<usize> = cluster_ids
            .iter()
            .map(|&cid| initial_connectivities[cluster_ids.iter().position(|&x| x == cid).unwrap()].len() / bits_per_cluster[cluster_ids.iter().position(|&x| x == cid).unwrap()])
            .collect();

        // Simple LCG RNG
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                self.0
            }
            fn range(&mut self, max: usize) -> usize {
                (self.next() as usize) % max.max(1)
            }
            fn float(&mut self) -> f64 {
                (self.next() as f64) / (u64::MAX as f64)
            }
        }
        let mut rng = Rng(rng_seed);

        let pop_size = config.ga_population;

        // Population: each candidate is a Vec of Vec<i64> (one connectivity per cluster)
        let mut population: Vec<Vec<Vec<i64>>> = Vec::with_capacity(pop_size);

        // First candidate is the initial connectivity
        population.push(initial_connectivities.to_vec());

        // Generate random candidates
        for _ in 1..pop_size {
            let mut candidate: Vec<Vec<i64>> = Vec::with_capacity(group_size);
            for (i, _) in cluster_ids.iter().enumerate() {
                let conn_len = initial_connectivities[i].len();
                let mut conn = vec![0i64; conn_len];
                for j in 0..conn_len {
                    conn[j] = rng.range(total_bits) as i64;
                }
                candidate.push(conn);
            }
            population.push(candidate);
        }

        // Evaluate initial population
        let mut fitness: Vec<f64> = population
            .iter()
            .map(|cand| self.compute_group_fitness(cluster_ids, cand, config.fitness_mode))
            .collect();

        let initial_fitness = fitness[0];
        let elitism = (pop_size / 10).max(1);

        // GA generations
        for _gen in 0..config.ga_gens {
            // Sort by fitness (descending)
            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| fitness[b].partial_cmp(&fitness[a]).unwrap());

            // Create next generation
            let mut next_pop: Vec<Vec<Vec<i64>>> = Vec::with_capacity(pop_size);

            // Elitism
            for &idx in indices.iter().take(elitism) {
                next_pop.push(population[idx].clone());
            }

            // Fill rest with crossover + mutation
            while next_pop.len() < pop_size {
                // Tournament selection
                let parent1_idx = {
                    let a = rng.range(pop_size);
                    let b = rng.range(pop_size);
                    if fitness[a] > fitness[b] { a } else { b }
                };
                let parent2_idx = {
                    let a = rng.range(pop_size);
                    let b = rng.range(pop_size);
                    if fitness[a] > fitness[b] { a } else { b }
                };

                // Crossover: pick clusters from each parent
                let mut child: Vec<Vec<i64>> = Vec::with_capacity(group_size);
                for i in 0..group_size {
                    // 50% chance to take from each parent
                    if rng.float() < 0.5 {
                        child.push(population[parent1_idx][i].clone());
                    } else {
                        child.push(population[parent2_idx][i].clone());
                    }
                }

                // Mutation
                for i in 0..group_size {
                    for j in 0..child[i].len() {
                        if rng.float() < config.mutation_rate {
                            child[i][j] = rng.range(total_bits) as i64;
                        }
                    }
                }

                next_pop.push(child);
            }

            population = next_pop;
            fitness = population
                .iter()
                .map(|cand| self.compute_group_fitness(cluster_ids, cand, config.fitness_mode))
                .collect();
        }

        // Find best
        let best_idx = fitness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let final_fitness = fitness[best_idx];
        let best_candidate = &population[best_idx];

        // Create results for each cluster in group
        cluster_ids
            .iter()
            .enumerate()
            .map(|(i, &cluster_id)| {
                let improvement = if initial_fitness != 0.0 {
                    ((final_fitness - initial_fitness) / initial_fitness.abs()) * 100.0
                } else {
                    0.0
                };

                ClusterOptResult {
                    cluster_id,
                    tier: config.tier,
                    initial_fitness,
                    final_fitness,
                    improvement_pct: improvement,
                    final_connectivity: best_candidate[i].clone(),
                    generations_run: config.ga_gens,
                }
            })
            .collect()
    }

    /// Optimize groups of clusters (for iterative refinement with frequency groups)
    ///
    /// Clusters are grouped by size (group_size), and each group is optimized jointly.
    /// Groups are processed in parallel.
    ///
    /// Args:
    ///     cluster_ids: All clusters to optimize
    ///     initial_connectivities: Connectivity for each cluster
    ///     group_size: Number of clusters per group (e.g., 10)
    ///     config: Optimization config
    ///     base_seed: Random seed
    ///
    /// Returns:
    ///     Results for all clusters
    pub fn optimize_tier_grouped(
        &self,
        cluster_ids: &[usize],
        initial_connectivities: &FxHashMap<usize, Vec<i64>>,
        group_size: usize,
        config: &TierOptConfig,
        base_seed: u64,
    ) -> TierOptResult {
        // Split clusters into groups
        let groups: Vec<Vec<usize>> = cluster_ids
            .chunks(group_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Optimize each group in parallel
        let all_results: Vec<ClusterOptResult> = groups
            .par_iter()
            .enumerate()
            .flat_map(|(group_idx, group)| {
                let seed = base_seed.wrapping_add(group_idx as u64 * 1000);

                // Extract connectivities for this group
                let group_conns: Vec<Vec<i64>> = group
                    .iter()
                    .map(|&cid| initial_connectivities.get(&cid).unwrap().clone())
                    .collect();

                // Joint GA optimization
                let ga_results = self.optimize_group_ga(group, &group_conns, config, seed);

                // Optional: TS refinement for each cluster individually
                // (Group optimization captures competition, TS refines locally)
                ga_results
            })
            .collect();

        // Aggregate results
        let total_improvement: f64 = all_results.iter().map(|r| r.improvement_pct).sum();
        let avg_improvement = if all_results.is_empty() {
            0.0
        } else {
            total_improvement / all_results.len() as f64
        };

        TierOptResult {
            tier: config.tier,
            clusters_optimized: all_results.len(),
            total_improvement,
            avg_improvement,
            cluster_results: all_results,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_contexts() {
        let flat = vec![true, false, true, true, false, false, false, true];
        let packed = PerClusterEvaluator::pack_contexts(&flat, 1, 8);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0b10001101);  // LSB first
    }

    #[test]
    fn test_compute_address() {
        let context = vec![0b10101010u64];
        let connectivity = vec![0i64, 2, 4, 6];  // Select bits 0, 2, 4, 6
        let addr = PerClusterEvaluator::compute_address(&context, &connectivity);
        assert_eq!(addr, 0b0000);  // bits 0,2,4,6 are all 0

        let connectivity2 = vec![1i64, 3, 5, 7];  // Select bits 1, 3, 5, 7
        let addr2 = PerClusterEvaluator::compute_address(&context, &connectivity2);
        assert_eq!(addr2, 0b1111);  // bits 1,3,5,7 are all 1
    }
}
