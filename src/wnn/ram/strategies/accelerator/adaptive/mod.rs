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
pub use crate::eval_worker::{EvalData, get_eval_worker};

use crate::neuron_memory::{
    FALSE, TRUE, EMPTY, BITS_PER_CELL, CELLS_PER_WORD, CELL_MASK,
    compute_address, NeuronTrainMeta,
};

// Canonical cell→weight conversion lives in neuron_memory.rs (single source
// of truth). Re-exported here for the 8 internal call sites.
pub(crate) use crate::neuron_memory::cell_to_weight;


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
mod eval;
pub use eval::*;
mod adaptive_eval;
pub use adaptive_eval::*;
mod gating_eval;
pub use gating_eval::*;
#[cfg(test)]
mod tests;
