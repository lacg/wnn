//! Standalone utility: random connection generation.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

// =============================================================================
// Standalone utility: random connection generation (Rust-accelerated)
// =============================================================================

/// Generate random connections for a genome entirely in Rust.
///
/// Args:
///   bits_per_neuron: List of bit counts per neuron (flat, [total_neurons])
///   total_input_bits: Number of input bits to choose from
///   seed: RNG seed for reproducibility
///
/// Returns: List of random connections in [0, total_input_bits), length = sum(bits_per_neuron)
#[pyfunction]
pub(crate) fn generate_random_connections(
    bits_per_neuron: Vec<usize>,
    total_input_bits: usize,
    seed: u64,
) -> Vec<i64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    neighbor_search::generate_random_connections(&bits_per_neuron, total_input_bits, &mut rng)
}
