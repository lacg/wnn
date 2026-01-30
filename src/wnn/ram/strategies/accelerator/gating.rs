//! RAM-based Gating for Cluster Layers
//!
//! Implements binary gating using RAM neurons with majority voting.
//! This is the Rust equivalent of the Python RAMGating class.
//!
//! Architecture:
//! - Each cluster has `neurons_per_gate` RAM neurons
//! - Each neuron has partial connectivity (bits_per_neuron)
//! - Gate = 1 if majority of neurons output TRUE, else 0
//!
//! Key Properties:
//! - Fully weightless (O(1) lookup)
//! - Binary gates via majority voting
//! - Can run in parallel with base RAM evaluation

use std::sync::atomic::{AtomicU8, Ordering};

/// Cell values for gate memory (same as base RAM)
const CELL_EMPTY: u8 = 0;
const CELL_FALSE: u8 = 1;
const CELL_TRUE: u8 = 2;

/// RAM-based gating model
///
/// Uses dedicated RAM neurons to learn which clusters should be active
/// for each input context. Gate output is binary (0 or 1) via majority voting.
pub struct RAMGating {
    /// Number of clusters to gate
    num_clusters: usize,
    /// Number of neurons voting on each cluster's gate
    neurons_per_gate: usize,
    /// Address bits per gate neuron
    bits_per_neuron: usize,
    /// Total input bits
    total_input_bits: usize,
    /// Voting threshold (number of neurons that must fire for gate=1)
    vote_threshold: usize,
    /// Connections: [total_neurons, bits_per_neuron] flattened
    /// Each neuron selects bits_per_neuron bits from input
    connections: Vec<i32>,
    /// Memory cells: [total_neurons, 2^bits_per_neuron] flattened
    /// Uses atomic u8 for thread-safe access
    memory: Vec<AtomicU8>,
    /// Address space size per neuron (2^bits_per_neuron)
    address_space_size: usize,
}

impl RAMGating {
    /// Create a new RAMGating model
    ///
    /// # Arguments
    /// * `num_clusters` - Number of clusters to gate (vocabulary size)
    /// * `neurons_per_gate` - Number of RAM neurons per cluster (default 8)
    /// * `bits_per_neuron` - Address bits per neuron (default 12)
    /// * `total_input_bits` - Total input bits (context_size * bits_per_token)
    /// * `threshold` - Fraction of neurons that must fire (default 0.5)
    /// * `seed` - Random seed for connectivity initialization
    pub fn new(
        num_clusters: usize,
        neurons_per_gate: usize,
        bits_per_neuron: usize,
        total_input_bits: usize,
        threshold: f32,
        seed: Option<u64>,
    ) -> Self {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let total_neurons = num_clusters * neurons_per_gate;
        let actual_bits = bits_per_neuron.min(total_input_bits);
        let address_space_size = 1 << actual_bits;

        // Initialize random number generator
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Initialize connections: each neuron selects random bits from input
        let mut connections = Vec::with_capacity(total_neurons * actual_bits);
        for _ in 0..total_neurons {
            // Sample bits_per_neuron unique indices from [0, total_input_bits)
            let mut selected: Vec<i32> = (0..total_input_bits as i32).collect();
            // Fisher-Yates shuffle for first actual_bits elements
            for i in 0..actual_bits {
                let j = rng.gen_range(i..total_input_bits);
                selected.swap(i, j);
            }
            connections.extend_from_slice(&selected[..actual_bits]);
        }

        // Initialize memory to EMPTY
        let memory_size = total_neurons * address_space_size;
        let memory: Vec<AtomicU8> = (0..memory_size)
            .map(|_| AtomicU8::new(CELL_EMPTY))
            .collect();

        // Compute vote threshold
        let vote_threshold = ((neurons_per_gate as f32 * threshold).ceil() as usize).max(1);

        Self {
            num_clusters,
            neurons_per_gate,
            bits_per_neuron: actual_bits,
            total_input_bits,
            vote_threshold,
            connections,
            memory,
            address_space_size,
        }
    }

    /// Total number of gate neurons
    #[inline]
    pub fn total_neurons(&self) -> usize {
        self.num_clusters * self.neurons_per_gate
    }

    /// Compute address for a single neuron given input bits
    #[inline]
    fn compute_address(&self, neuron_idx: usize, input_bits: &[bool]) -> usize {
        let conn_start = neuron_idx * self.bits_per_neuron;
        let mut address: usize = 0;
        for (bit_pos, &conn_idx) in self.connections[conn_start..conn_start + self.bits_per_neuron]
            .iter()
            .enumerate()
        {
            if conn_idx >= 0 && (conn_idx as usize) < input_bits.len() && input_bits[conn_idx as usize] {
                address |= 1 << bit_pos;
            }
        }
        address
    }

    /// Get memory cell index for a neuron and address
    #[inline]
    fn memory_index(&self, neuron_idx: usize, address: usize) -> usize {
        neuron_idx * self.address_space_size + address
    }

    /// Read a single gate neuron's output
    #[inline]
    fn read_neuron(&self, neuron_idx: usize, address: usize) -> bool {
        let idx = self.memory_index(neuron_idx, address);
        let cell = self.memory[idx].load(Ordering::Relaxed);
        cell == CELL_TRUE
    }

    /// Compute binary gates for a single input
    ///
    /// Returns a vector of gate values (0.0 or 1.0) for each cluster
    pub fn forward_single(&self, input_bits: &[bool]) -> Vec<f32> {
        let mut gates = vec![0.0f32; self.num_clusters];

        for cluster in 0..self.num_clusters {
            let neuron_start = cluster * self.neurons_per_gate;
            let mut true_count = 0usize;

            for n in 0..self.neurons_per_gate {
                let neuron_idx = neuron_start + n;
                let address = self.compute_address(neuron_idx, input_bits);
                if self.read_neuron(neuron_idx, address) {
                    true_count += 1;
                }
            }

            // Majority vote: gate=1 if count >= threshold
            if true_count >= self.vote_threshold {
                gates[cluster] = 1.0;
            }
        }

        gates
    }

    /// Compute binary gates for a batch of inputs (parallel)
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `batch_size` - Number of examples in batch
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters]
    pub fn forward_batch(&self, input_bits_flat: &[bool], batch_size: usize) -> Vec<f32> {
        use rayon::prelude::*;

        let total_input_bits = self.total_input_bits;

        (0..batch_size)
            .into_par_iter()
            .flat_map(|b| {
                let start = b * total_input_bits;
                let end = start + total_input_bits;
                let input = &input_bits_flat[start..end];
                self.forward_single(input)
            })
            .collect()
    }

    /// Train gate neurons for a single example
    ///
    /// # Arguments
    /// * `input_bits` - Input bit pattern
    /// * `target_gates` - Target gate values (true = open, false = closed)
    /// * `allow_override` - Whether to override non-EMPTY cells
    ///
    /// # Returns
    /// Number of cells modified
    pub fn train_single(
        &self,
        input_bits: &[bool],
        target_gates: &[bool],
        allow_override: bool,
    ) -> usize {
        let mut modified = 0;

        for cluster in 0..self.num_clusters {
            let target = target_gates[cluster];
            let target_cell = if target { CELL_TRUE } else { CELL_FALSE };
            let neuron_start = cluster * self.neurons_per_gate;

            for n in 0..self.neurons_per_gate {
                let neuron_idx = neuron_start + n;
                let address = self.compute_address(neuron_idx, input_bits);
                let mem_idx = self.memory_index(neuron_idx, address);

                // Try to write to memory
                let current = self.memory[mem_idx].load(Ordering::Relaxed);

                if current == CELL_EMPTY || allow_override {
                    // Write new value
                    self.memory[mem_idx].store(target_cell, Ordering::Relaxed);
                    if current == CELL_EMPTY {
                        modified += 1;
                    }
                }
            }
        }

        modified
    }

    /// Train gate neurons for a batch of examples
    ///
    /// # Arguments
    /// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
    /// * `target_gates_flat` - Flattened target gates [batch_size * num_clusters]
    /// * `batch_size` - Number of examples
    /// * `allow_override` - Whether to override non-EMPTY cells
    ///
    /// # Returns
    /// Total cells modified across batch
    pub fn train_batch(
        &self,
        input_bits_flat: &[bool],
        target_gates_flat: &[bool],
        batch_size: usize,
        allow_override: bool,
    ) -> usize {
        let mut total_modified = 0;

        for b in 0..batch_size {
            let input_start = b * self.total_input_bits;
            let input_end = input_start + self.total_input_bits;
            let input = &input_bits_flat[input_start..input_end];

            let gate_start = b * self.num_clusters;
            let gate_end = gate_start + self.num_clusters;
            let targets = &target_gates_flat[gate_start..gate_end];

            total_modified += self.train_single(input, targets, allow_override);
        }

        total_modified
    }

    /// Reset all memory cells to EMPTY
    pub fn reset(&self) {
        for cell in &self.memory {
            cell.store(CELL_EMPTY, Ordering::Relaxed);
        }
    }

    /// Get statistics about memory usage
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        let mut empty = 0usize;
        let mut false_count = 0usize;
        let mut true_count = 0usize;

        for cell in &self.memory {
            match cell.load(Ordering::Relaxed) {
                CELL_EMPTY => empty += 1,
                CELL_FALSE => false_count += 1,
                CELL_TRUE => true_count += 1,
                _ => {}
            }
        }

        (empty, false_count, true_count)
    }

    /// Get configuration for serialization
    pub fn config(&self) -> GatingConfig {
        GatingConfig {
            num_clusters: self.num_clusters,
            neurons_per_gate: self.neurons_per_gate,
            bits_per_neuron: self.bits_per_neuron,
            total_input_bits: self.total_input_bits,
            vote_threshold: self.vote_threshold,
        }
    }

    /// Get connections (for Python binding)
    pub fn get_connections(&self) -> &[i32] {
        &self.connections
    }

    /// Export memory state as bytes (for serialization)
    pub fn export_memory(&self) -> Vec<u8> {
        self.memory.iter().map(|c| c.load(Ordering::Relaxed)).collect()
    }

    /// Import memory state from bytes
    pub fn import_memory(&self, data: &[u8]) -> Result<(), &'static str> {
        if data.len() != self.memory.len() {
            return Err("Memory size mismatch");
        }
        for (i, &val) in data.iter().enumerate() {
            self.memory[i].store(val, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Configuration struct for RAMGating
#[derive(Debug, Clone)]
pub struct GatingConfig {
    pub num_clusters: usize,
    pub neurons_per_gate: usize,
    pub bits_per_neuron: usize,
    pub total_input_bits: usize,
    pub vote_threshold: usize,
}

/// Apply gates to cluster scores
///
/// Multiplies each cluster score by its gate value.
/// This is a simple element-wise multiplication.
///
/// # Arguments
/// * `scores` - [batch_size * num_clusters] cluster scores
/// * `gates` - [batch_size * num_clusters] gate values (0.0 or 1.0)
///
/// # Returns
/// Gated scores [batch_size * num_clusters]
pub fn apply_gates(scores: &[f32], gates: &[f32]) -> Vec<f32> {
    scores.iter().zip(gates.iter()).map(|(&s, &g)| s * g).collect()
}

/// Apply gates in-place
pub fn apply_gates_inplace(scores: &mut [f32], gates: &[f32]) {
    for (s, &g) in scores.iter_mut().zip(gates.iter()) {
        *s *= g;
    }
}

/// Compute target gates from target cluster indices
///
/// Creates a boolean vector where target_gates[i][target[i]] = true and all others = false.
/// This is used for training gating from supervised targets.
///
/// # Arguments
/// * `targets` - [batch_size] target cluster indices
/// * `num_clusters` - Number of clusters (vocabulary size)
///
/// # Returns
/// Flattened target gates [batch_size * num_clusters] where only the target cluster is true
pub fn compute_target_gates(targets: &[i64], num_clusters: usize) -> Vec<bool> {
    let batch_size = targets.len();
    let mut result = vec![false; batch_size * num_clusters];

    for (b, &target) in targets.iter().enumerate() {
        if target >= 0 && (target as usize) < num_clusters {
            result[b * num_clusters + target as usize] = true;
        }
    }

    result
}

/// Compute target gates with top-k expansion
///
/// Creates target gates where the target cluster and its k-nearest neighbors are set to true.
/// This allows gating to learn contextual relevance beyond just the exact target.
///
/// # Arguments
/// * `targets` - [batch_size] target cluster indices
/// * `num_clusters` - Number of clusters
/// * `neighbor_clusters` - Optional [batch_size * k] neighbor indices per example
///                         If None, only the target is set to true
///
/// # Returns
/// Flattened target gates [batch_size * num_clusters]
pub fn compute_target_gates_expanded(
    targets: &[i64],
    num_clusters: usize,
    neighbor_clusters: Option<&[i64]>,
    neighbors_per_example: usize,
) -> Vec<bool> {
    let batch_size = targets.len();
    let mut result = vec![false; batch_size * num_clusters];

    for (b, &target) in targets.iter().enumerate() {
        let base = b * num_clusters;

        // Always include the target
        if target >= 0 && (target as usize) < num_clusters {
            result[base + target as usize] = true;
        }

        // Include neighbors if provided
        if let Some(neighbors) = neighbor_clusters {
            let neighbor_start = b * neighbors_per_example;
            for i in 0..neighbors_per_example {
                let neighbor = neighbors[neighbor_start + i];
                if neighbor >= 0 && (neighbor as usize) < num_clusters {
                    result[base + neighbor as usize] = true;
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_creation() {
        let gating = RAMGating::new(100, 8, 12, 64, 0.5, Some(42));
        assert_eq!(gating.num_clusters, 100);
        assert_eq!(gating.neurons_per_gate, 8);
        assert_eq!(gating.total_neurons(), 800);
    }

    #[test]
    fn test_gating_forward_all_empty() {
        let gating = RAMGating::new(10, 4, 8, 32, 0.5, Some(42));
        let input = vec![false; 32];

        // With all EMPTY cells, neurons return FALSE, so all gates should be 0
        let gates = gating.forward_single(&input);
        assert_eq!(gates.len(), 10);
        for g in gates {
            assert_eq!(g, 0.0);
        }
    }

    #[test]
    fn test_gating_train_and_forward() {
        let gating = RAMGating::new(5, 4, 6, 16, 0.5, Some(42));
        let input = vec![true; 16];

        // Train: cluster 2 should be open
        let mut targets = vec![false; 5];
        targets[2] = true;

        let modified = gating.train_single(&input, &targets, false);
        assert!(modified > 0);

        // Forward: cluster 2 should have gate=1
        let gates = gating.forward_single(&input);
        assert_eq!(gates[2], 1.0);
    }

    #[test]
    fn test_apply_gates() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let gates = vec![1.0, 0.0, 1.0, 0.0];
        let gated = apply_gates(&scores, &gates);
        assert_eq!(gated, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_memory_stats() {
        let gating = RAMGating::new(10, 4, 6, 16, 0.5, Some(42));
        let (empty, false_count, true_count) = gating.memory_stats();

        // Initially all empty
        assert!(empty > 0);
        assert_eq!(false_count, 0);
        assert_eq!(true_count, 0);

        // After training
        let input = vec![true; 16];
        let mut targets = vec![true; 10];
        gating.train_single(&input, &targets, false);

        let (empty2, _, true_count2) = gating.memory_stats();
        assert!(empty2 < empty);
        assert!(true_count2 > 0);
    }
}
