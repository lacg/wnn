//! Flat-genome protocol validation at the PyO3 boundary.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

///
/// The internal `debug_assert_eq!` guards are compiled out under
/// `maturin develop --release` (the only supported build), so without this
/// check a misaligned flat array is read with silently shifted offsets —
/// wrong results, not an error. Call this from every PyO3 entry point that
/// accepts the (bits_flat, neurons_flat, connections_flat) triple.
///
/// Invariants (per-NEURON layout — see the offsets comment in
/// `evaluate_genomes_parallel_hybrid`):
/// - `neurons_flat.len() == num_genomes * num_clusters`
/// - `bits_flat.len() == Σ neurons_flat` (one entry per neuron, NOT per cluster)
/// - `connections_flat.len() == Σ bits_flat`, or empty (random-connection fallback)
pub(crate) fn validate_flat_genomes(
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
) -> Result<(), String> {
    let expected_neurons_len = num_genomes * num_clusters;
    if neurons_flat.len() != expected_neurons_len {
        return Err(format!(
            "genomes_neurons_flat length {} != num_genomes ({}) * num_clusters ({}) = {}",
            neurons_flat.len(), num_genomes, num_clusters, expected_neurons_len
        ));
    }
    let total_neurons: usize = neurons_flat.iter().sum();
    if bits_flat.len() != total_neurons {
        return Err(format!(
            "genomes_bits_flat length {} != total neurons {} — bits must be per-NEURON \
             (Σ neurons_per_cluster entries), not per-cluster",
            bits_flat.len(), total_neurons
        ));
    }
    let total_connections: usize = bits_flat.iter().sum();
    if !connections_flat.is_empty() && connections_flat.len() != total_connections {
        return Err(format!(
            "genomes_connections_flat length {} != Σ bits {} (and not empty). A common cause \
             is flattening a genome batch where only SOME genomes have connections — that \
             silently shifts every subsequent genome's offsets",
            connections_flat.len(), total_connections
        ));
    }
    Ok(())
}
