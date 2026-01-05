//! RAM Neuron implementation in Rust
//!
//! Matches Python's GeneralizedNGramRAM exactly:
//! - Each neuron stores word counts at each address (not binary bits!)
//! - Prediction returns most common word at address
//! - Final prediction votes across neurons for best word

use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::sync::Arc;

/// A single RAM neuron that stores WORD COUNTS at each address
/// This matches Python's: self.ram = defaultdict(Counter)
pub struct RAMNeuron {
    /// Which input bits this neuron observes
    connectivity: Vec<usize>,
    /// Storage: address -> {word_id: count}
    /// Using word_id (u32) instead of String for efficiency
    memory: FxHashMap<u64, HashMap<u32, u32>>,
}

impl RAMNeuron {
    pub fn new(connectivity: Vec<usize>) -> Self {
        Self {
            connectivity,
            memory: FxHashMap::default(),
        }
    }

    /// Compute address from input bits based on connectivity
    /// Matches Python's get_address()
    #[inline]
    pub fn compute_address(&self, input_bits: &[u64]) -> u64 {
        let mut address = 0u64;
        for (i, &bit_idx) in self.connectivity.iter().enumerate() {
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if word_idx < input_bits.len() {
                let bit = (input_bits[word_idx] >> bit_pos) & 1;
                address |= bit << i;
            }
        }
        address
    }

    /// Train: increment count for target word at this address
    /// Matches Python's: self.ram[addr][target] += 1
    pub fn train(&mut self, input_bits: &[u64], target_word_id: u32) {
        let address = self.compute_address(input_bits);
        let counts = self.memory.entry(address).or_insert_with(HashMap::new);
        *counts.entry(target_word_id).or_insert(0) += 1;
    }

    /// Predict: return (most_common_word_id, confidence)
    /// Matches Python's predict() which returns (best_word, count/total)
    pub fn predict(&self, input_bits: &[u64]) -> Option<(u32, f32)> {
        let address = self.compute_address(input_bits);
        self.memory.get(&address).and_then(|counts| {
            if counts.is_empty() {
                return None;
            }
            let total: u32 = counts.values().sum();
            // DETERMINISTIC: Break ties by word_id (lower id wins)
            let (best_word, best_count) = counts
                .iter()
                .max_by(|(id_a, &count_a), (id_b, &count_b)| {
                    count_a.cmp(&count_b).then_with(|| id_b.cmp(id_a))  // Higher count wins, lower id wins ties
                })?;
            Some((*best_word, *best_count as f32 / total as f32))
        })
    }
}

/// Generalized N-gram RAM for language modeling
///
/// IMPORTANT: This now matches Python's approach exactly:
/// - Each neuron stores word counts (not binary cluster bits)
/// - Prediction votes across neurons for best word
pub struct GeneralizedNGramRAM {
    n: usize,
    neurons: Vec<RAMNeuron>,
    word_to_bits: FxHashMap<String, u64>,  // For context encoding
    word_to_id: FxHashMap<String, u32>,     // word -> numeric ID
    id_to_word: Vec<String>,                // ID -> word (for reverse lookup)
}

impl GeneralizedNGramRAM {
    const BITS_PER_WORD: usize = 12;

    pub fn new(
        n: usize,
        connectivity: &[Vec<i64>],
        word_to_bits: &FxHashMap<String, u64>,
        _bits_per_neuron: usize,
    ) -> Self {
        let neurons = connectivity
            .iter()
            .map(|conn| {
                RAMNeuron::new(conn.iter().map(|&x| x as usize).collect())
            })
            .collect();

        // Build word_to_id mapping with DETERMINISTIC order
        // FxHashMap iteration order is non-deterministic, so we must sort keys
        let mut sorted_words: Vec<_> = word_to_bits.keys().cloned().collect();
        sorted_words.sort();

        let mut word_to_id = FxHashMap::default();
        let mut id_to_word = Vec::new();
        for word in sorted_words {
            let id = id_to_word.len() as u32;
            word_to_id.insert(word.clone(), id);
            id_to_word.push(word);
        }

        Self {
            n,
            neurons,
            word_to_bits: word_to_bits.clone(),
            word_to_id,
            id_to_word,
        }
    }

    /// Get word ID, creating new one if needed
    fn get_or_create_word_id(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }
        let id = self.id_to_word.len() as u32;
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.push(word.to_string());
        id
    }

    /// Get precomputed 12-bit encoding for word
    #[inline]
    fn get_word_bits(&self, word: &str) -> u64 {
        self.word_to_bits.get(word).copied().unwrap_or(0)
    }

    /// Encode context words into bit vector
    fn encode_context(&self, context: &[&str]) -> Vec<u64> {
        let total_bits = context.len() * Self::BITS_PER_WORD;
        let num_words = (total_bits + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (word_idx, word) in context.iter().enumerate() {
            let word_bits = self.get_word_bits(word);
            let bit_offset = word_idx * Self::BITS_PER_WORD;

            for bit in 0..Self::BITS_PER_WORD {
                if (word_bits >> bit) & 1 == 1 {
                    let global_bit = bit_offset + bit;
                    let word_pos = global_bit / 64;
                    let bit_pos = global_bit % 64;
                    if word_pos < bits.len() {
                        bits[word_pos] |= 1u64 << bit_pos;
                    }
                }
            }
        }
        bits
    }

    /// Train on token sequence
    /// Each neuron learns to predict the TARGET WORD (not cluster bits!)
    pub fn train(&mut self, tokens: &[String]) {
        if tokens.len() < self.n + 1 {
            return;
        }

        for i in 0..tokens.len() - self.n {
            let context: Vec<&str> = tokens[i..i + self.n].iter().map(|s| s.as_str()).collect();
            let target = &tokens[i + self.n];

            let input_bits = self.encode_context(&context);
            let target_id = self.get_or_create_word_id(target);

            // Train ALL neurons with the same target word
            for neuron in self.neurons.iter_mut() {
                neuron.train(&input_bits, target_id);
            }
        }
    }

    /// Predict next word using voting across neurons
    /// Returns (word_id, confidence) or None
    pub fn predict(&self, context: &[&str]) -> Option<(u32, f32)> {
        if context.len() < self.n {
            return None;
        }

        let ctx = &context[context.len() - self.n..];
        let input_bits = self.encode_context(ctx);

        // Collect votes from all neurons
        let mut votes: HashMap<u32, f32> = HashMap::new();
        let mut num_predictions = 0;

        for neuron in self.neurons.iter() {
            if let Some((word_id, conf)) = neuron.predict(&input_bits) {
                *votes.entry(word_id).or_insert(0.0) += conf;
                num_predictions += 1;
            }
        }

        if votes.is_empty() {
            return None;
        }

        // Return word with highest vote
        // DETERMINISTIC: Break ties by word_id (lower id wins)
        let (best_word_id, best_score) = votes
            .iter()
            .max_by(|(id_a, score_a), (id_b, score_b)| {
                score_a.partial_cmp(score_b)
                    .unwrap()
                    .then_with(|| id_b.cmp(id_a))  // Higher score wins, lower id wins ties
            })?;

        Some((*best_word_id, *best_score / num_predictions as f32))
    }

    /// Get word string from ID
    pub fn get_word(&self, word_id: u32) -> Option<&str> {
        self.id_to_word.get(word_id as usize).map(|s| s.as_str())
    }
}

/// Evaluate a single connectivity pattern
/// Now correctly compares WORDS (not clusters!)
pub fn evaluate_single<S: AsRef<str>>(
    connectivity: &[Vec<i64>],
    word_to_bits: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    _bits_per_neuron: usize,
    eval_subset: usize,
) -> f64 {
    // Build RAM with this connectivity
    let mut ram = GeneralizedNGramRAM::new(4, connectivity, word_to_bits, 14);

    // Train
    let train_strs: Vec<String> = train_tokens.iter().map(|s| s.as_ref().to_string()).collect();
    ram.train(&train_strs);

    // Evaluate - compare predicted WORD to target WORD
    let mut correct = 0u32;
    let mut covered = 0u32;
    let total = eval_subset.min(test_tokens.len().saturating_sub(4));

    for i in 0..total {
        let context: Vec<&str> = test_tokens[i..i + 4]
            .iter()
            .map(|s| s.as_ref())
            .collect();
        let target = test_tokens[i + 4].as_ref();

        if let Some((predicted_word_id, _conf)) = ram.predict(&context) {
            covered += 1;
            // Compare predicted word to target word
            if let Some(predicted_word) = ram.get_word(predicted_word_id) {
                if predicted_word == target {
                    correct += 1;
                }
            }
        }
    }

    let accuracy = if covered > 0 {
        correct as f64 / covered as f64
    } else {
        0.0
    };
    let coverage = if total > 0 {
        covered as f64 / total as f64
    } else {
        0.0
    };

    // Return error = 1 - (accuracy * coverage)
    1.0 - (accuracy * coverage)
}

/// Evaluate cascade of multiple RAMs (n=2,3,4,5,6)
/// Returns 1 - accuracy for the full cascade prediction
pub fn evaluate_cascade<S: AsRef<str>>(
    connectivities: &[&[Vec<i64>]],  // [n2_conn, n3_conn, n4_conn, n5_conn, n6_conn]
    word_to_bits: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    eval_subset: usize,
) -> f64 {
    // Build all 5 RAMs
    let n_values = [2, 3, 4, 5, 6];
    let mut rams: Vec<GeneralizedNGramRAM> = Vec::new();

    for (i, &n) in n_values.iter().enumerate() {
        let mut ram = GeneralizedNGramRAM::new(n, connectivities[i], word_to_bits, 14);
        let train_strs: Vec<String> = train_tokens.iter().map(|s| s.as_ref().to_string()).collect();
        ram.train(&train_strs);
        rams.push(ram);
    }

    // Evaluate cascade - try higher n first
    let mut correct = 0u32;
    let total = eval_subset.min(test_tokens.len().saturating_sub(6));  // Need at least n=6 context

    for i in 0..total {
        let target = test_tokens[i + 6].as_ref();

        // Try each RAM from highest n to lowest
        let mut predicted = false;
        for (ram_idx, &n) in n_values.iter().enumerate().rev() {
            if i + n > test_tokens.len() - 1 {
                continue;
            }
            let context: Vec<&str> = test_tokens[i + 6 - n..i + 6]
                .iter()
                .map(|s| s.as_ref())
                .collect();

            if let Some((word_id, conf)) = rams[ram_idx].predict(&context) {
                if conf > 0.1 {  // Confidence threshold
                    if let Some(pred_word) = rams[ram_idx].get_word(word_id) {
                        if pred_word == target {
                            correct += 1;
                        }
                        predicted = true;
                        break;  // Use first confident prediction
                    }
                }
            }
        }

        // If no RAM predicted, count as wrong
        let _ = predicted;  // Just to show we tried all RAMs
    }

    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    // Return error = 1 - accuracy
    1.0 - accuracy
}

/// Batch evaluate cascade with candidate connectivities for one RAM
/// target_ram_idx: 0=n2, 1=n3, 2=n4, 3=n5, 4=n6
pub fn evaluate_cascade_batch<S: AsRef<str> + Sync>(
    base_connectivities: &[Vec<Vec<i64>>],  // [n2, n3, n4, n5, n6] - fixed connectivities
    candidates: &[Vec<Vec<i64>>],           // candidate connectivities for target RAM
    target_ram_idx: usize,                   // which RAM we're optimizing (0-4)
    word_to_bits: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    eval_subset: usize,
) -> Vec<f64> {
    use rayon::prelude::*;

    candidates.par_iter().map(|candidate_conn| {
        // Build connectivity array with candidate at target position
        let mut conns: Vec<&[Vec<i64>]> = Vec::new();
        for (i, conn) in base_connectivities.iter().enumerate() {
            if i == target_ram_idx {
                conns.push(candidate_conn.as_slice());
            } else {
                conns.push(conn.as_slice());
            }
        }

        evaluate_cascade(&conns, word_to_bits, train_tokens, test_tokens, eval_subset)
    }).collect()
}

/// Evaluate FULL NETWORK (exact + generalized) with pre-computed exact results
/// For positions covered by exact RAMs, uses the pre-computed result
/// For positions not covered, evaluates the generalized RAM cascade
pub fn evaluate_fullnetwork<S: AsRef<str>>(
    connectivities: &[&[Vec<i64>]],  // [n2_conn, n3_conn, n4_conn, n5_conn, n6_conn]
    word_to_bits: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    exact_results: &[Option<bool>],   // Pre-computed: Some(true/false) if exact covers, None if not
    eval_subset: usize,
) -> f64 {
    // Build all 5 RAMs
    let n_values = [2, 3, 4, 5, 6];
    let mut rams: Vec<GeneralizedNGramRAM> = Vec::new();

    for (i, &n) in n_values.iter().enumerate() {
        let mut ram = GeneralizedNGramRAM::new(n, connectivities[i], word_to_bits, 14);
        let train_strs: Vec<String> = train_tokens.iter().map(|s| s.as_ref().to_string()).collect();
        ram.train(&train_strs);
        rams.push(ram);
    }

    // Evaluate full network
    let mut correct = 0u32;
    let total = eval_subset.min(test_tokens.len().saturating_sub(6)).min(exact_results.len());

    for i in 0..total {
        // Check if exact RAM covers this position
        if let Some(exact_correct) = exact_results[i] {
            // Use pre-computed exact result
            if exact_correct {
                correct += 1;
            }
        } else {
            // No exact coverage - try generalized RAM cascade, then voting fallback
            let target = test_tokens[i + 6].as_ref();
            let mut found_confident = false;

            // Try each RAM from highest n to lowest (cascade)
            for (ram_idx, &n) in n_values.iter().enumerate().rev() {
                if i + n > test_tokens.len() - 1 {
                    continue;
                }
                let context: Vec<&str> = test_tokens[i + 6 - n..i + 6]
                    .iter()
                    .map(|s| s.as_ref())
                    .collect();

                if let Some((word_id, conf)) = rams[ram_idx].predict(&context) {
                    if conf > 0.1 {
                        if let Some(pred_word) = rams[ram_idx].get_word(word_id) {
                            if pred_word == target {
                                correct += 1;
                            }
                            found_confident = true;
                            break;  // Use first confident prediction
                        }
                    }
                }
            }

            // Voting fallback: if no confident cascade prediction, vote across all RAMs
            if !found_confident {
                let mut votes: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();

                for (ram_idx, &n) in n_values.iter().enumerate() {
                    if i + 6 < n {
                        continue;
                    }
                    let context: Vec<&str> = test_tokens[i + 6 - n..i + 6]
                        .iter()
                        .map(|s| s.as_ref())
                        .collect();

                    if let Some((word_id, conf)) = rams[ram_idx].predict(&context) {
                        // Weight by conf * n (context length)
                        *votes.entry(word_id).or_insert(0.0) += conf * (n as f32);
                    }
                }

                if !votes.is_empty() {
                    // Find word with highest vote (deterministic tie-breaking)
                    let (best_word_id, _) = votes
                        .iter()
                        .max_by(|(id_a, score_a), (id_b, score_b)| {
                            score_a.partial_cmp(score_b)
                                .unwrap()
                                .then_with(|| id_b.cmp(id_a))  // Lower id wins ties
                        })
                        .unwrap();

                    // Check if voting prediction is correct
                    if let Some(pred_word) = rams[0].get_word(*best_word_id) {
                        if pred_word == target {
                            correct += 1;
                        }
                    }
                }
            }
        }
    }

    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    // Return error = 1 - accuracy
    1.0 - accuracy
}

/// Batch evaluate full network with candidate connectivities for one RAM
pub fn evaluate_fullnetwork_batch<S: AsRef<str> + Sync>(
    base_connectivities: &[Vec<Vec<i64>>],
    candidates: &[Vec<Vec<i64>>],
    target_ram_idx: usize,
    word_to_bits: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    exact_results: &[Option<bool>],
    eval_subset: usize,
) -> Vec<f64> {
    use rayon::prelude::*;

    candidates.par_iter().map(|candidate_conn| {
        let mut conns: Vec<&[Vec<i64>]> = Vec::new();
        for (i, conn) in base_connectivities.iter().enumerate() {
            if i == target_ram_idx {
                conns.push(candidate_conn.as_slice());
            } else {
                conns.push(conn.as_slice());
            }
        }

        evaluate_fullnetwork(&conns, word_to_bits, train_tokens, test_tokens, exact_results, eval_subset)
    }).collect()
}

/// Evaluate full network using PERPLEXITY as the metric
/// exact_probs: Pre-computed P(target|context) for exact RAM positions, None if not covered
/// Returns perplexity = exp(mean cross-entropy), lower is better
pub fn evaluate_fullnetwork_perplexity<S: AsRef<str>>(
    connectivities: &[&[Vec<i64>]],
    word_to_bits: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    exact_probs: &[Option<f64>],   // Pre-computed: Some(prob) if exact covers, None if not
    eval_subset: usize,
    vocab_size: usize,             // For smoothing when no prediction
    cascade_threshold: f64,        // Confidence threshold for cascade
) -> f64 {
    // Build all 5 RAMs
    let n_values = [2, 3, 4, 5, 6];
    let mut rams: Vec<GeneralizedNGramRAM> = Vec::new();

    for (i, &n) in n_values.iter().enumerate() {
        let mut ram = GeneralizedNGramRAM::new(n, connectivities[i], word_to_bits, 14);
        let train_strs: Vec<String> = train_tokens.iter().map(|s| s.as_ref().to_string()).collect();
        ram.train(&train_strs);
        rams.push(ram);
    }

    // Evaluate perplexity
    let mut total_cross_entropy = 0.0f64;
    let total = eval_subset.min(test_tokens.len().saturating_sub(6)).min(exact_probs.len());

    // Minimum probability to avoid -inf (equivalent to vocab_size uniform)
    let min_prob = 1.0 / (vocab_size as f64);

    for i in 0..total {
        let target = test_tokens[i + 6].as_ref();

        // Get probability for this position
        let prob: f64 = if let Some(exact_prob) = exact_probs[i] {
            // Use pre-computed exact RAM probability
            exact_prob.max(min_prob)
        } else {
            // Try generalized RAM cascade, then voting
            let mut found_prob: Option<f64> = None;

            // Try each RAM from highest n to lowest (cascade)
            for (ram_idx, &n) in n_values.iter().enumerate().rev() {
                if i + 6 < n {
                    continue;
                }
                let context: Vec<&str> = test_tokens[i + 6 - n..i + 6]
                    .iter()
                    .map(|s| s.as_ref())
                    .collect();

                if let Some((pred_word_id, conf)) = rams[ram_idx].predict(&context) {
                    if (conf as f64) > cascade_threshold {
                        // FIX: Compare strings, not numeric IDs!
                        if let Some(pred_word) = rams[ram_idx].get_word(pred_word_id) {
                            if pred_word == target {
                                // Correct prediction - use confidence as probability
                                found_prob = Some((conf as f64).max(min_prob));
                            } else {
                                // Wrong prediction - assign low probability to target
                                found_prob = Some(min_prob);
                            }
                        }
                        break;
                    }
                }
            }

            // Voting fallback
            if found_prob.is_none() {
                let mut votes: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
                let mut total_weight = 0.0f32;

                for (ram_idx, &n) in n_values.iter().enumerate() {
                    if i + 6 < n {
                        continue;
                    }
                    let context: Vec<&str> = test_tokens[i + 6 - n..i + 6]
                        .iter()
                        .map(|s| s.as_ref())
                        .collect();

                    if let Some((word_id, conf)) = rams[ram_idx].predict(&context) {
                        // FIX: Use actual word string as vote key
                        if let Some(word) = rams[ram_idx].get_word(word_id) {
                            let weight = conf * (n as f32);
                            *votes.entry(word.to_string()).or_insert(0.0) += weight;
                            total_weight += weight;
                        }
                    }
                }

                if total_weight > 0.0 {
                    // Get probability assigned to target word
                    let target_votes = votes.get(target).copied().unwrap_or(0.0);
                    let prob = (target_votes / total_weight) as f64;
                    found_prob = Some(prob.max(min_prob));
                }
            }

            found_prob.unwrap_or(min_prob)
        };

        // Add cross-entropy: -log(prob)
        total_cross_entropy += -prob.ln();
    }

    // Return perplexity = exp(mean cross-entropy)
    if total > 0 {
        (total_cross_entropy / total as f64).exp()
    } else {
        f64::MAX
    }
}

/// Batch evaluate full network perplexity with candidate connectivities
pub fn evaluate_fullnetwork_perplexity_batch<S: AsRef<str> + Sync>(
    base_connectivities: &[Vec<Vec<i64>>],
    candidates: &[Vec<Vec<i64>>],
    target_ram_idx: usize,
    word_to_bits: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    exact_probs: &[Option<f64>],
    eval_subset: usize,
    vocab_size: usize,
    cascade_threshold: f64,
) -> Vec<f64> {
    use rayon::prelude::*;

    candidates.par_iter().map(|candidate_conn| {
        let mut conns: Vec<&[Vec<i64>]> = Vec::new();
        for (i, conn) in base_connectivities.iter().enumerate() {
            if i == target_ram_idx {
                conns.push(candidate_conn.as_slice());
            } else {
                conns.push(conn.as_slice());
            }
        }

        evaluate_fullnetwork_perplexity(&conns, word_to_bits, train_tokens, test_tokens, exact_probs, eval_subset, vocab_size, cascade_threshold)
    }).collect()
}

// =============================================================================
// BATCH PREDICTION WITH PRE-TRAINED RAMs
// =============================================================================

/// Pre-trained neuron for fast prediction (no training, just lookup)
struct PretrainedNeuron {
    connectivity: Vec<usize>,
    memory: FxHashMap<u64, HashMap<String, u32>>,
}

impl PretrainedNeuron {
    fn new(
        connectivity: &[i64],
        memory: &std::collections::HashMap<u64, std::collections::HashMap<String, u32>>,
    ) -> Self {
        let conn: Vec<usize> = connectivity.iter().map(|&x| x as usize).collect();
        let mem: FxHashMap<u64, HashMap<String, u32>> = memory
            .iter()
            .map(|(&addr, counts)| (addr, counts.clone()))
            .collect();
        Self {
            connectivity: conn,
            memory: mem,
        }
    }

    #[inline]
    fn compute_address(&self, input_bits: &[u64]) -> u64 {
        let mut address = 0u64;
        for (i, &bit_idx) in self.connectivity.iter().enumerate() {
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if word_idx < input_bits.len() {
                let bit = (input_bits[word_idx] >> bit_pos) & 1;
                address |= bit << i;
            }
        }
        address
    }

    fn predict(&self, input_bits: &[u64]) -> Option<(String, f32)> {
        let address = self.compute_address(input_bits);
        self.memory.get(&address).and_then(|counts| {
            if counts.is_empty() {
                return None;
            }
            let total: u32 = counts.values().sum();
            let (best_word, best_count) = counts
                .iter()
                .max_by(|(word_a, &count_a), (word_b, &count_b)| {
                    count_a.cmp(&count_b).then_with(|| word_b.cmp(word_a))
                })?;
            Some((best_word.clone(), *best_count as f32 / total as f32))
        })
    }
}

/// Pre-trained generalized RAM for fast prediction
struct PretrainedGeneralizedRAM {
    n: usize,
    neurons: Vec<PretrainedNeuron>,
}

impl PretrainedGeneralizedRAM {
    const BITS_PER_WORD: usize = 12;

    fn new(
        n: usize,
        connectivity: &[Vec<i64>],
        memory: &[std::collections::HashMap<u64, std::collections::HashMap<String, u32>>],
    ) -> Self {
        let neurons: Vec<PretrainedNeuron> = connectivity
            .iter()
            .zip(memory.iter())
            .map(|(conn, mem)| PretrainedNeuron::new(conn, mem))
            .collect();
        Self { n, neurons }
    }

    fn encode_context(&self, context: &[&str], word_to_bits: &FxHashMap<String, u64>) -> Vec<u64> {
        let total_bits = context.len() * Self::BITS_PER_WORD;
        let num_words = (total_bits + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (word_idx, word) in context.iter().enumerate() {
            let word_bits = word_to_bits.get(*word).copied().unwrap_or(0);
            let bit_offset = word_idx * Self::BITS_PER_WORD;

            for bit in 0..Self::BITS_PER_WORD {
                if (word_bits >> bit) & 1 == 1 {
                    let global_bit = bit_offset + bit;
                    let word_pos = global_bit / 64;
                    let bit_pos = global_bit % 64;
                    if word_pos < bits.len() {
                        bits[word_pos] |= 1u64 << bit_pos;
                    }
                }
            }
        }
        bits
    }

    fn predict(&self, context: &[&str], word_to_bits: &FxHashMap<String, u64>) -> Option<(String, f32)> {
        if context.len() < self.n {
            return None;
        }

        let ctx = &context[context.len() - self.n..];
        let input_bits = self.encode_context(ctx, word_to_bits);

        let mut votes: HashMap<String, f32> = HashMap::new();
        let mut num_predictions = 0;

        for neuron in self.neurons.iter() {
            if let Some((word, conf)) = neuron.predict(&input_bits) {
                *votes.entry(word).or_insert(0.0) += conf;
                num_predictions += 1;
            }
        }

        if votes.is_empty() {
            return None;
        }

        let (best_word, best_score) = votes
            .iter()
            .max_by(|(word_a, score_a), (word_b, score_b)| {
                score_a.partial_cmp(score_b)
                    .unwrap()
                    .then_with(|| word_b.cmp(word_a))
            })?;

        Some((best_word.clone(), *best_score / num_predictions as f32))
    }
}

/// Pre-trained exact RAM for fast prediction
struct PretrainedExactRAM {
    n: usize,
    contexts: FxHashMap<Vec<u64>, HashMap<String, u32>>,
}

impl PretrainedExactRAM {
    const BITS_PER_WORD: usize = 12;

    fn new(
        n: usize,
        contexts: &std::collections::HashMap<Vec<u64>, std::collections::HashMap<String, u32>>,
    ) -> Self {
        let ctx_map: FxHashMap<Vec<u64>, HashMap<String, u32>> = contexts
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Self { n, contexts: ctx_map }
    }

    fn encode_context(&self, context: &[&str], word_to_bits: &FxHashMap<String, u64>) -> Vec<u64> {
        let total_bits = context.len() * Self::BITS_PER_WORD;
        let num_words = (total_bits + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (word_idx, word) in context.iter().enumerate() {
            let word_bits = word_to_bits.get(*word).copied().unwrap_or(0);
            let bit_offset = word_idx * Self::BITS_PER_WORD;

            for bit in 0..Self::BITS_PER_WORD {
                if (word_bits >> bit) & 1 == 1 {
                    let global_bit = bit_offset + bit;
                    let word_pos = global_bit / 64;
                    let bit_pos = global_bit % 64;
                    if word_pos < bits.len() {
                        bits[word_pos] |= 1u64 << bit_pos;
                    }
                }
            }
        }
        bits
    }

    fn predict(&self, context: &[&str], word_to_bits: &FxHashMap<String, u64>) -> Option<(String, f32)> {
        if context.len() < self.n {
            return None;
        }

        let ctx = &context[context.len() - self.n..];
        let ctx_bits = self.encode_context(ctx, word_to_bits);

        self.contexts.get(&ctx_bits).and_then(|counts| {
            if counts.is_empty() {
                return None;
            }
            let total: u32 = counts.values().sum();
            let (best_word, best_count) = counts
                .iter()
                .max_by(|(word_a, &count_a), (word_b, &count_b)| {
                    count_a.cmp(&count_b).then_with(|| word_b.cmp(word_a))
                })?;
            Some((best_word.clone(), *best_count as f32 / total as f32))
        })
    }
}

/// Batch predict using pre-trained RAMs
/// Returns predictions for each position from each RAM
pub fn predict_all_batch(
    generalized_rams: &[(
        usize,
        Vec<Vec<i64>>,
        Vec<std::collections::HashMap<u64, std::collections::HashMap<String, u32>>>,
    )],
    exact_rams: &[(
        usize,
        std::collections::HashMap<Vec<u64>, std::collections::HashMap<String, u32>>,
    )],
    word_to_bits: &FxHashMap<String, u64>,
    test_tokens: &[String],
) -> Vec<std::collections::HashMap<String, (String, f32)>> {
    use rayon::prelude::*;

    // Build pre-trained RAM structures
    let gen_rams: Vec<PretrainedGeneralizedRAM> = generalized_rams
        .iter()
        .map(|(n, conn, mem)| PretrainedGeneralizedRAM::new(*n, conn, mem))
        .collect();

    let exact_ram_structs: Vec<PretrainedExactRAM> = exact_rams
        .iter()
        .map(|(n, contexts)| PretrainedExactRAM::new(*n, contexts))
        .collect();

    // Wrap in Arc for sharing across threads
    let gen_rams = Arc::new(gen_rams);
    let exact_ram_structs = Arc::new(exact_ram_structs);
    let word_to_bits = Arc::new(word_to_bits.clone());
    let test_tokens = Arc::new(test_tokens.to_vec());

    let n_positions = test_tokens.len().saturating_sub(6);

    // Parallel prediction over all positions
    (0..n_positions)
        .into_par_iter()
        .map(|i| {
            let context: Vec<&str> = test_tokens[i..i + 5]
                .iter()
                .map(|s| s.as_str())
                .collect();

            let mut predictions: std::collections::HashMap<String, (String, f32)> =
                std::collections::HashMap::new();

            // Exact RAMs
            for exact_ram in exact_ram_structs.iter() {
                if context.len() >= exact_ram.n {
                    if let Some((word, conf)) = exact_ram.predict(&context, &word_to_bits) {
                        let method = format!("exact_n{}", exact_ram.n);
                        predictions.insert(method, (word, conf));
                    }
                }
            }

            // Generalized RAMs
            for gen_ram in gen_rams.iter() {
                if context.len() >= gen_ram.n {
                    if let Some((word, conf)) = gen_ram.predict(&context, &word_to_bits) {
                        let method = format!("gen_n{}", gen_ram.n);
                        predictions.insert(method, (word, conf));
                    }
                }
            }

            predictions
        })
        .collect()
}
