//! RAM Accelerator - High-performance RAM neuron evaluation for Apple Silicon
//!
//! Provides GPU-accelerated evaluation of RAM neuron connectivity patterns
//! using Metal compute shaders on M-series Macs.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::SeedableRng;
use std::sync::{Arc, RwLock};
use std::sync::{Mutex, OnceLock};
use numpy::PyReadonlyArray1;

/// Release-mode validation of the flat-genome triple at the PyO3 boundary.
///
/// Wraps `adaptive::validate_flat_genomes` into a `PyValueError` so a
/// misaligned batch fails loudly instead of being scored with silently
/// shifted offsets (the internal `debug_assert`s are compiled out under
/// `--release`).
fn validate_flat_genomes_py(
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
) -> PyResult<()> {
    adaptive::validate_flat_genomes(bits_flat, neurons_flat, connections_flat, num_genomes, num_clusters)
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

// Global cached Metal evaluator for RAMLM (avoids shader recompilation)
// Using OnceLock + Option to handle initialization errors gracefully
static METAL_RAMLM_EVALUATOR: OnceLock<Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>> = OnceLock::new();

fn get_cached_metal_evaluator() -> Result<&'static Mutex<Option<metal_ramlm::MetalRAMLMEvaluator>>, String> {
    Ok(METAL_RAMLM_EVALUATOR.get_or_init(|| {
        Mutex::new(metal_ramlm::MetalRAMLMEvaluator::new().ok())
    }))
}

// Global cached Metal trainer for GPU address computation during training
static METAL_TRAINER: OnceLock<Mutex<Option<metal_train::MetalTrainer>>> = OnceLock::new();

fn get_cached_metal_trainer() -> Result<&'static Mutex<Option<metal_train::MetalTrainer>>, String> {
    Ok(METAL_TRAINER.get_or_init(|| {
        Mutex::new(metal_train::MetalTrainer::new().ok())
    }))
}

// Global cached Metal evaluator for Gating (avoids shader recompilation)
static METAL_GATING_EVALUATOR: OnceLock<Mutex<Option<metal_gating::MetalGatingEvaluator>>> = OnceLock::new();

fn get_cached_metal_gating_evaluator() -> Result<&'static Mutex<Option<metal_gating::MetalGatingEvaluator>>, String> {
    Ok(METAL_GATING_EVALUATOR.get_or_init(|| {
        Mutex::new(metal_gating::MetalGatingEvaluator::new().ok())
    }))
}

#[path = "ramlm.rs"]
mod ramlm;

// Metal evaluator modules: real on macOS, stubs on other platforms
#[cfg(target_os = "macos")]
#[path = "metal_evaluator.rs"]
mod metal_evaluator;

#[cfg(not(target_os = "macos"))]
mod metal_evaluator {
    pub struct MetalEvaluator;
    impl MetalEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn is_available() -> bool { false }
        pub fn device_info() -> Result<String, String> { Err("Metal not available on this platform".into()) }
        pub fn evaluate_batch(
            &self, _: &[Vec<Vec<i64>>], _: &std::collections::HashMap<String, u64>,
            _: &[String], _: &[String], _: usize, _: usize,
        ) -> Result<Vec<f64>, String> { Err("Metal not available on this platform".into()) }
    }
}

#[cfg(target_os = "macos")]
#[path = "metal_ramlm.rs"]
mod metal_ramlm;

#[cfg(not(target_os = "macos"))]
mod metal_ramlm {
    pub struct MetalRAMLMEvaluator;
    impl MetalRAMLMEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
        pub fn is_available() -> bool { false }
        pub fn device_info() -> Result<String, String> { Err("Metal not available on this platform".into()) }
        pub fn forward_batch(
            &self, _: &[u64], _: &[i64], _: &[i64],
            _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize, _: u8,
        ) -> Result<Vec<f32>, String> { Err("Metal not available on this platform".into()) }
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

    pub struct MetalGroupEvaluator;
    impl MetalGroupEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct MetalSparseCEEvaluator;
    impl MetalSparseCEEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct MetalCEReduceEvaluator;
    impl MetalCEReduceEvaluator {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }

    pub struct SparseGroupData<'a> {
        pub connections: &'a [i64],
        pub keys: &'a [u64],
        pub values: &'a [u8],
        pub offsets: &'a [u32],
        pub counts: &'a [u32],
        pub cluster_ids: &'a [usize],
        pub bits_per_neuron: usize,
        pub neurons_per_cluster: usize,
        pub actual_neurons_per_cluster: Option<&'a [u32]>,
    }

    pub fn reset_sparse_buffer_cache() {}
    pub fn get_sparse_cache_generation() -> u64 { 0 }
}

#[path = "neuron_memory.rs"]
mod neuron_memory;

#[path = "sparse_memory.rs"]
mod sparse_memory;

#[path = "adaptive/mod.rs"]
mod adaptive;

#[path = "token_cache.rs"]
mod token_cache;

#[path = "ids_cache.rs"]
mod ids_cache;
mod ids_streaming;
mod packed_bits;
mod atomic_hashtable;

#[path = "neighbor_search.rs"]
mod neighbor_search;

#[path = "gating.rs"]
mod gating;

#[cfg(target_os = "macos")]
#[path = "metal_gating.rs"]
mod metal_gating;

#[path = "eval_worker.rs"]
pub mod eval_worker;

#[path = "bitwise_ramlm.rs"]
mod bitwise_ramlm;

#[path = "multistage.rs"]
mod multistage;

#[path = "adaptation.rs"]
mod adaptation;

#[cfg(target_os = "macos")]
#[path = "metal_stats.rs"]
mod metal_stats;

#[cfg(target_os = "macos")]
#[path = "metal_train.rs"]
mod metal_train;

#[cfg(not(target_os = "macos"))]
mod metal_train {
    pub struct MetalTrainer;
    impl MetalTrainer {
        pub fn new() -> Result<Self, String> { Err("Metal not available on this platform".into()) }
    }
}

#[cfg(target_os = "macos")]
mod metal_atomic_test;

#[cfg(target_os = "macos")]
mod marker_train;

// Drone-controller hot-path (paper #1): attitude sim, controller wrapper,
// Strategy-5 QSR decoder, monotonicity counter, reward.
// See src/wnn/ram/strategies/accelerator/controller.rs and the design
// memory project_drone_controller_paper1.md.
mod controller;
mod controller_training;
mod controller_split;
mod dagger_train;
mod cancel;

// GPU-batched closed-loop controller eval (macOS/Metal only).
#[cfg(target_os = "macos")]
#[path = "metal_controller.rs"]
mod metal_controller;

#[cfg(target_os = "macos")]
pub use metal_evaluator::MetalEvaluator;
#[cfg(target_os = "macos")]
pub use metal_ramlm::MetalRAMLMEvaluator;


// PyO3 surface (all #[pyfunction]/#[pyclass] wrappers) — see pyapi/.
#[path = "pyapi/mod.rs"]
mod pyapi;
use pyapi::*;


/// Python module definition
/// ABI version of the accelerator's Python surface. Bump on any breaking
/// change to an exported signature; wnn/accel.py asserts it at import so a
/// stale build fails loudly instead of silently mis-marshalling.
pub const ABI_VERSION: u32 = 2;

#[pymodule]
fn ram_accelerator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ABI_VERSION", ABI_VERSION)?;
    m.add_function(wrap_pyfunction!(metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(reset_metal_evaluators, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_cores, m)?)?;
    // New batch prediction functions
    // Exact probs acceleration (bit-encoded - deprecated, slow due to export)
    // Exact probs acceleration (word-based - FAST, no export overhead)
    // RAMLM acceleration (proper RAM WNN architecture)
    m.add_function(wrap_pyfunction!(ramlm_train_batch_numpy, m)?)?;  // FAST numpy-based training
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_batch_numpy, m)?)?;  // Bitwise multi-label training (dense)
    m.add_function(wrap_pyfunction!(ramlm_train_batch_tiered_numpy, m)?)?;  // FAST tiered training (all tiers in one call)
    // RAMLM Metal GPU acceleration
    m.add_function(wrap_pyfunction!(ramlm_metal_available, m)?)?;
    // RAMLM NumPy-based acceleration (FAST - zero-copy)
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_metal_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_hybrid_numpy, m)?)?;
    // RAMLM Cached Metal (avoids shader recompilation)
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_metal_cached, m)?)?;
    // Sparse memory backend (for >10 bits per neuron)
    m.add_class::<SparseMemory>()?;
    m.add_function(wrap_pyfunction!(sparse_train_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_bitwise_train_batch, m)?)?;  // Bitwise multi-label training (sparse)
    m.add_function(wrap_pyfunction!(sparse_forward_batch, m)?)?;
    // Tiered sparse memory backend (for variable bits per tier)
    m.add_class::<TieredSparseMemory>()?;
    m.add_function(wrap_pyfunction!(sparse_train_batch_tiered, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_train_batch_tiered_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_forward_batch_tiered, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_forward_batch_tiered_numpy, m)?)?;
    // Metal GPU sparse forward (cached export for fast repeated forward)
    m.add_class::<SparseGpuCache>()?;
    m.add_function(wrap_pyfunction!(sparse_export_for_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_export_groups_for_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_forward_metal_numpy, m)?)?;
    // Parallel GA/TS candidate evaluation (KEY optimization)
    // Parallel GA/TS for TIERED architectures (16 cores parallel)
    // Parallel GA/TS HYBRID CPU+GPU (memory-adaptive, pipelined)
    // Hybrid with explicit memory budget
    // Memory estimation utilities
    // Option C foundation — atomic CAS microbench (CPU+GPU coherence test)
    m.add_function(wrap_pyfunction!(run_atomic_cas_microbench, m)?)?;
    // Option B B0 — MarkerHashTable unit tests
    m.add_function(wrap_pyfunction!(run_marker_hashtable_tests, m)?)?;
    // Option B B2 — Metal marker-train kernel parity test
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(run_marker_train_parity_test, m)?)?;
    // Option B B4-batched — multi-genome Metal kernel parity test
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(run_marker_train_batched_parity_test, m)?)?;
    // Option B B5 — multi-cluster Metal kernel parity test
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(run_marker_train_multicluster_parity_test, m)?)?;
    // Per-cluster optimization (Rust-accelerated discriminative optimization)
    // Global CE with caching (true global softmax over all 50K clusters)
    // Group-based optimization for iterative refinement
    // EMPTY cell value configuration (affects PPL calculation)
    // Adaptive architecture (per-cluster variable bits/neurons)
    m.add_function(wrap_pyfunction!(adaptive_forward_batch, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_train_batch, m)?)?;
    // Parallel genome evaluation (KEY for GA optimization)
    m.add_function(wrap_pyfunction!(evaluate_genomes_parallel, m)?)?;
    // Multi-subset parallel genome evaluation (for per-iteration rotation)
    // Parallel hybrid CPU+GPU genome evaluation (4-8x speedup)
    m.add_function(wrap_pyfunction!(evaluate_genomes_parallel_hybrid, m)?)?;
    // FPGA export: train one genome + return per-neuron sparse (keys, values)
    // Token cache for persistent token storage with subset rotation
    m.add_class::<TokenCacheWrapper>()?;
    // IDS cache for intrusion detection classification
    m.add_class::<IDSCacheWrapper>()?;
    m.add_class::<IDSCacheBuilderWrapper>()?;
    m.add_class::<IDSGenomeStreamerWrapper>()?;
    // RAM-based gating (weightless per-cluster gating with majority voting)
    m.add_class::<RAMGatingWrapper>()?;
    m.add_function(wrap_pyfunction!(compute_target_gates, m)?)?;
    m.add_function(wrap_pyfunction!(gating_metal_available, m)?)?;
    // Bitwise RAMLM evaluation (full Rust+Metal pipeline)
    m.add_class::<BitwiseCacheWrapper>()?;
    // Multi-stage RAMLM evaluation (group prediction + within-group)
    m.add_class::<MultiStageCacheWrapper>()?;
    // Bitwise RAMLM — nudge training + quad forward
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_batch_nudge_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_bitwise_train_and_eval_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_quad_binary_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ramlm_forward_batch_quad_weighted_numpy, m)?)?;
    // Utility: Rust-accelerated random connection generation
    m.add_function(wrap_pyfunction!(generate_random_connections, m)?)?;
    m.add_function(wrap_pyfunction!(find_optimal_threshold_fitness_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_platt_scaling_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_beta_calibration_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_empirical_threshold_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_binary_metrics_at_threshold_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_optimal_threshold_f1_py, m)?)?;

    // Drone-controller hot-path (paper #1). See controller.rs and
    // project_drone_controller_paper1.md.
    m.add_class::<controller::AttitudeSim>()?;
    m.add_class::<controller::WnnController>()?;
    m.add_class::<controller::AttitudePidRs>()?;
    // dagger_train scaffold (B.2 in progress) — see dagger_train.rs.
    m.add_class::<dagger_train::RewardGatedConfigPacked>()?;
    m.add_class::<dagger_train::TrainStats>()?;
    m.add_function(wrap_pyfunction!(dagger_train::dagger_train_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(dagger_train::dagger_train_batch_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(controller::strategy_5_qsr_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(controller::strategy_1_count_true, m)?)?;
    m.add_function(wrap_pyfunction!(controller::monotonicity_violations, m)?)?;
    m.add_function(wrap_pyfunction!(controller::compute_reward, m)?)?;
    #[cfg(target_os = "macos")]
    m.add_function(wrap_pyfunction!(metal_controller::score_controllers_metal, m)?)?;
    // Cooperative cancellation for the controller + IDS evaluators. Python's
    // SIGTERM handler calls set_cancel_flag(), Rust callsites poll at safe
    // boundaries (between genomes / episodes / GPU dispatch chunks) and return
    // partial results.
    m.add_function(wrap_pyfunction!(cancel::set_cancel_flag, m)?)?;
    m.add_function(wrap_pyfunction!(cancel::reset_cancel_flag, m)?)?;
    m.add_function(wrap_pyfunction!(cancel::is_cancelled, m)?)?;

    // EDRA constraint solver (Rust port of Memory._solve_partial_connectivity).
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_trinary_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_qsr_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_trinary_reachable_py, m)?)?;
    m.add_function(wrap_pyfunction!(controller_training::solve_partial_qsr_reachable_py, m)?)?;

    Ok(())
}
