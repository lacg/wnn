//! ram_core — shared WNN substrate primitives.
//!
//! Carved out of the monolithic `ram_accelerator` crate on 2026-06-19 so a
//! controller-only change no longer forces a worker-wheel rebuild + swap. Both
//! `ram_accelerator` (IDS/LM worker) and `ram_controller` link this rlib.
//!
//! Membership rule: a module belongs here iff it is genuinely shared substrate
//! — needed by BOTH the worker and the controller. Today that is the Memory
//! cell semantics (`neuron_memory`), bit packing (`packed_bits`), the sparse
//! Memory backend (`sparse_memory`) and its GPU forward (`metal_sparse`), and
//! the cooperative-cancellation flag (`cancel`). LM- or IDS-specific code does
//! NOT live here (see `metal_ramlm.rs` / `metal_genome_eval.rs` in the worker).

pub mod cancel;
pub mod packed_bits;
pub mod neuron_memory;
pub mod sparse_memory;
// GA/TS search operators (mutation, 8 crossovers, tournament, offspring/
// neighbour drivers). Promoted from ram_accelerator 20/07/2026: it was
// worker-only, so the CONTROLLER could not reach it and grew a parallel
// Python implementation instead (control/arch_strategy.py:201 opted out
// explicitly). Self-contained — no crate:: refs — so the move is mechanical.
pub mod neighbor_search;
// Counter-based RNG shared by both substrates AND both languages (Python
// mirror: src/wnn/ram/counter_rng.py). Order-independent by construction,
// so genome operators can move to Rust and run under rayon without the
// sequential-stream dependency that pinned them to Python.
pub mod counter_rng;

// Metal sparse forward: real on macOS, stub elsewhere (mirrors the worker's
// per-platform Metal gating so the crate still type-checks on non-macOS).
#[cfg(target_os = "macos")]
pub mod metal_sparse;

#[cfg(not(target_os = "macos"))]
pub mod metal_sparse {
    pub fn default_cell_for_mode(memory_mode: u8) -> u32 {
        match memory_mode { 0 => 2, _ => 1 }
    }
    pub struct MetalSparseEvaluator;
    impl MetalSparseEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch_sparse(
            &self, _: &[u64], _: &[i64], _: &[u64], _: &[u8], _: &[u32], _: &[u32],
            _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: u8,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch_general(
            &self, _: &[u64], _: &[i64], _: &[u64], _: &[u8], _: &[u32], _: &[u32],
            _: &[(u32, u32, u32, u32)], _: usize, _: usize, _: usize, _: u8,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
    }
}
