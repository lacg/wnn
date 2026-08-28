//! Adaptive Architecture Accelerator
//!
//! High-performance training and forward pass for AdaptiveClusteredRAM
//! where each cluster can have its own (bits, neurons) configuration.
//!
//! Key optimization: Clusters are grouped by their config to enable
//! efficient batch processing within each group.
//!
//! Memory strategy:
//! - Dense memory (bit-packed Vec) for bits <= SPARSE_THRESHOLD
//! - Sparse memory (DashMap) for bits > SPARSE_THRESHOLD
//! This enables up to 30 bits without memory explosion.
//!
//! Metal GPU Acceleration:
//! - Dense groups can be evaluated on Metal GPU (40 cores on M4 Max)
//! - Sparse groups stay on CPU (hash lookups not GPU-friendly)
//! - Hybrid approach: Metal for dense, CPU for sparse

use dashmap::DashMap;
#[cfg(target_os = "macos")]
use metal;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// Re-export from eval_worker module for backward compatibility
pub use crate::eval_worker::{get_eval_worker, EvalData};

use ram_core::neuron_memory::{
	compute_address, NeuronTrainMeta, BITS_PER_CELL, CELLS_PER_WORD, CELL_MASK, EMPTY, FALSE, TRUE,
};

// Canonical cell→weight conversion lives in neuron_memory.rs (single source
// of truth). Re-exported here for the internal call sites. cell_to_weight_rng
// adds the QSR/PLN seeded coin (byte-identical to cell_to_weight for the
// deterministic modes); qsr_key derives the per-read coin key.
pub(crate) use ram_core::neuron_memory::{cell_to_weight, cell_to_weight_rng, qsr_key};

mod metal_state;
pub use metal_state::*;
mod validation;
pub(crate) use validation::validate_flat_genomes;
mod thresholds;
pub use thresholds::*;
mod groups;
pub use groups::*;
mod memory;
pub use memory::*;
mod eval_parallel;
pub use eval_parallel::*;
pub mod eval_export;
pub use eval_export::*;
mod eval_single;
pub use eval_single::*;
mod eval_hybrid;
pub use eval_hybrid::*;
mod adaptive_eval;
pub use adaptive_eval::*;
mod gating_eval;
pub use gating_eval::*;
#[cfg(test)]
mod tests;
