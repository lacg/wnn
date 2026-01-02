//! RAM Neuron implementation in Rust
//!
//! A RAM (Random Access Memory) neuron is essentially a lookup table:
//! - Input: subset of input bits (determined by connectivity)
//! - Output: stored value at that address
//!
//! This is much faster than Python's dict-based implementation.

use rustc_hash::FxHashMap;

/// A single RAM neuron with configurable connectivity
pub struct RAMNeuron {
    /// Which input bits this neuron observes
    connectivity: Vec<usize>,
    /// Storage: address -> count of TRUE responses
    memory: FxHashMap<u64, (u32, u32)>, // (true_count, total_count)
}

impl RAMNeuron {
    pub fn new(connectivity: Vec<usize>) -> Self {
        Self {
            connectivity,
            memory: FxHashMap::default(),
        }
    }

    /// Compute address from input bits based on connectivity
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

    /// Train: store pattern at address
    pub fn train(&mut self, input_bits: &[u64], target: bool) {
        let address = self.compute_address(input_bits);
        let entry = self.memory.entry(address).or_insert((0, 0));
        if target {
            entry.0 += 1;
        }
        entry.1 += 1;
    }

    /// Predict: lookup address and return (prediction, confidence)
    pub fn predict(&self, input_bits: &[u64]) -> Option<(bool, f32)> {
        let address = self.compute_address(input_bits);
        self.memory.get(&address).map(|&(true_count, total)| {
            let conf = true_count as f32 / total as f32;
            (conf > 0.5, conf.max(1.0 - conf))
        })
    }
}

/// Generalized N-gram RAM for language modeling
pub struct GeneralizedNGramRAM {
    n: usize,
    neurons: Vec<RAMNeuron>,
    word_to_cluster: FxHashMap<String, u64>,
    bits_per_word: usize,
}

impl GeneralizedNGramRAM {
    pub fn new(
        n: usize,
        connectivity: &[Vec<i64>],
        word_to_cluster: &FxHashMap<String, u64>,
        bits_per_neuron: usize,
    ) -> Self {
        let neurons = connectivity
            .iter()
            .map(|conn| {
                RAMNeuron::new(conn.iter().map(|&x| x as usize).collect())
            })
            .collect();

        // Calculate bits needed for word encoding
        let max_cluster = word_to_cluster.values().max().copied().unwrap_or(0);
        let bits_per_word = (64 - max_cluster.leading_zeros()) as usize;

        Self {
            n,
            neurons,
            word_to_cluster: word_to_cluster.clone(),
            bits_per_word,
        }
    }

    /// Encode context words into bit vector
    fn encode_context(&self, context: &[&str]) -> Vec<u64> {
        let total_bits = context.len() * self.bits_per_word;
        let num_words = (total_bits + 63) / 64;
        let mut bits = vec![0u64; num_words];

        for (word_idx, word) in context.iter().enumerate() {
            if let Some(&cluster) = self.word_to_cluster.get(*word) {
                let bit_offset = word_idx * self.bits_per_word;
                for bit in 0..self.bits_per_word {
                    if (cluster >> bit) & 1 == 1 {
                        let global_bit = bit_offset + bit;
                        let word_pos = global_bit / 64;
                        let bit_pos = global_bit % 64;
                        if word_pos < bits.len() {
                            bits[word_pos] |= 1u64 << bit_pos;
                        }
                    }
                }
            }
        }
        bits
    }

    /// Train on token sequence
    pub fn train(&mut self, tokens: &[String]) {
        if tokens.len() < self.n + 1 {
            return;
        }

        for i in 0..tokens.len() - self.n {
            let context: Vec<&str> = tokens[i..i + self.n].iter().map(|s| s.as_str()).collect();
            let target = &tokens[i + self.n];

            let input_bits = self.encode_context(&context);

            // Train each neuron with its target bit
            if let Some(&target_cluster) = self.word_to_cluster.get(target) {
                for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
                    let target_bit = (target_cluster >> neuron_idx) & 1 == 1;
                    neuron.train(&input_bits, target_bit);
                }
            }
        }
    }

    /// Predict next word
    pub fn predict(&self, context: &[&str]) -> Option<(u64, f32)> {
        if context.len() < self.n {
            return None;
        }

        let ctx = &context[context.len() - self.n..];
        let input_bits = self.encode_context(ctx);

        let mut predicted_cluster = 0u64;
        let mut total_conf = 0.0f32;
        let mut count = 0;

        for (neuron_idx, neuron) in self.neurons.iter().enumerate() {
            if let Some((pred, conf)) = neuron.predict(&input_bits) {
                if pred {
                    predicted_cluster |= 1u64 << neuron_idx;
                }
                total_conf += conf;
                count += 1;
            }
        }

        if count > 0 {
            Some((predicted_cluster, total_conf / count as f32))
        } else {
            None
        }
    }
}

/// Evaluate a single connectivity pattern
pub fn evaluate_single<S: AsRef<str>>(
    connectivity: &[Vec<i64>],
    word_to_cluster: &FxHashMap<String, u64>,
    train_tokens: &[S],
    test_tokens: &[S],
    _bits_per_neuron: usize,
    eval_subset: usize,
) -> f64 {
    // Build RAM with this connectivity
    let mut ram = GeneralizedNGramRAM::new(4, connectivity, word_to_cluster, 14);

    // Train
    let train_strs: Vec<String> = train_tokens.iter().map(|s| s.as_ref().to_string()).collect();
    ram.train(&train_strs);

    // Evaluate
    let mut correct = 0u32;
    let mut covered = 0u32;
    let total = eval_subset.min(test_tokens.len().saturating_sub(4));

    for i in 0..total {
        let context: Vec<&str> = test_tokens[i..i + 4]
            .iter()
            .map(|s| s.as_ref())
            .collect();
        let target = test_tokens[i + 4].as_ref();

        if let Some((predicted_cluster, _conf)) = ram.predict(&context) {
            covered += 1;
            if let Some(&target_cluster) = word_to_cluster.get(target) {
                if predicted_cluster == target_cluster {
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
