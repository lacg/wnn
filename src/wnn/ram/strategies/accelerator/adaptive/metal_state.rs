//! Resettable Metal evaluator singletons + hybrid CPU/GPU split state.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

use super::*;

// =============================================================================
// Resettable Metal Evaluators
// =============================================================================
//
// Metal evaluators can accumulate driver-level state over long runs, causing
// slowdowns. These use Arc + RwLock to allow periodic reset.
// Call reset_metal_evaluators() every N generations to recreate fresh evaluators.

// Global counter incremented on each reset
pub(crate) static RESET_GENERATION: AtomicU64 = AtomicU64::new(0);

// D2 (10/06/2026): NORMAL_CLASS + FITNESS_* process globals were folded
// into neuron_memory::EvalSettings, threaded per call from the PyO3 boundary.

/// Find optimal threshold using fitness weights if set, otherwise F1.
pub fn find_optimal_threshold_auto(scores: &[f64], labels: &[i64], fitness_weights: Option<(f32, f32, f32, f32)>) -> (f64, f64, f64) {
    if let Some((w_ce, w_f1, w_fpr, w_acc)) = fitness_weights {
        let (t, f1, fpr, _acc, _fitness) = find_optimal_threshold_fitness(scores, labels, w_ce, w_f1, w_fpr, w_acc);
        (t, f1, fpr)
    } else {
        find_optimal_threshold_f1(scores, labels)
    }
}

// Storage for resettable evaluators - uses Arc so callers can hold references
static METAL_EVALUATOR: RwLock<Option<Arc<crate::metal_ramlm::MetalRAMLMEvaluator>>> = RwLock::new(None);
static SPARSE_METAL_EVALUATOR: RwLock<Option<Arc<crate::metal_ramlm::MetalSparseEvaluator>>> = RwLock::new(None);
static GROUP_EVALUATOR: RwLock<Option<Arc<crate::metal_ramlm::MetalGroupEvaluator>>> = RwLock::new(None);

/// Get or initialize the Metal evaluator (resettable, thread-safe)
/// Returns an Arc that can be held across lock boundaries
/// Set WNN_NO_METAL=1 to disable Metal and use CPU-only evaluation (for diagnostics)
pub fn get_metal_evaluator() -> Option<Arc<crate::metal_ramlm::MetalRAMLMEvaluator>> {
    // Check for Metal disable flag (for diagnostics)
    if std::env::var("WNN_NO_METAL").is_ok() {
        return None;
    }

    // Fast path: check if initialized
    {
        let guard = METAL_EVALUATOR.read().unwrap();
        if let Some(ref arc) = *guard {
            return Some(Arc::clone(arc));
        }
    }

    // Slow path: need to initialize
    let mut guard = METAL_EVALUATOR.write().unwrap();
    if guard.is_none() {
        if let Ok(eval) = crate::metal_ramlm::MetalRAMLMEvaluator::new() {
            *guard = Some(Arc::new(eval));
        }
    }
    guard.as_ref().map(Arc::clone)
}

/// Get or initialize the sparse Metal evaluator (resettable, thread-safe)
/// Set WNN_NO_METAL=1 to disable Metal and use CPU-only evaluation (for diagnostics)
pub fn get_sparse_metal_evaluator() -> Option<Arc<crate::metal_ramlm::MetalSparseEvaluator>> {
    // Check for Metal disable flag (for diagnostics)
    if std::env::var("WNN_NO_METAL").is_ok() {
        return None;
    }

    // Fast path: check if initialized
    {
        let guard = SPARSE_METAL_EVALUATOR.read().unwrap();
        if let Some(ref arc) = *guard {
            return Some(Arc::clone(arc));
        }
    }

    // Slow path: need to initialize
    let mut guard = SPARSE_METAL_EVALUATOR.write().unwrap();
    if guard.is_none() {
        if let Ok(eval) = crate::metal_ramlm::MetalSparseEvaluator::new() {
            *guard = Some(Arc::new(eval));
        }
    }
    guard.as_ref().map(Arc::clone)
}

/// Get or initialize the group evaluator (resettable, thread-safe)
/// Set WNN_NO_METAL=1 to disable Metal and use CPU-only evaluation (for diagnostics)
pub(crate) fn get_group_evaluator() -> Option<Arc<crate::metal_ramlm::MetalGroupEvaluator>> {
    // Check for Metal disable flag (for diagnostics)
    if std::env::var("WNN_NO_METAL").is_ok() {
        return None;
    }

    // Fast path: check if initialized
    {
        let guard = GROUP_EVALUATOR.read().unwrap();
        if let Some(ref arc) = *guard {
            return Some(Arc::clone(arc));
        }
    }

    // Slow path: need to initialize
    let mut guard = GROUP_EVALUATOR.write().unwrap();
    if guard.is_none() {
        if let Ok(eval) = crate::metal_ramlm::MetalGroupEvaluator::new() {
            *guard = Some(Arc::new(eval));
        }
    }
    guard.as_ref().map(Arc::clone)
}

/// Reset all Metal evaluators to free accumulated driver state.
///
/// Call this periodically (e.g., every 50 generations) to prevent slowdown
/// from Metal driver state accumulation during long optimization runs.
///
/// The evaluators will be lazily re-initialized on next use.
/// Existing Arc references will continue to work until dropped.
pub fn reset_metal_evaluators() {
    // Increment generation counter (for scores/input buffer cache in evaluate_genome_hybrid)
    RESET_GENERATION.fetch_add(1, Ordering::SeqCst);

    // Also reset the sparse buffer cache (for per-group buffers in eval_sparse_to_buffer)
    crate::metal_ramlm::reset_sparse_buffer_cache();

    // Clear all evaluators - existing Arc holders keep their reference
    // until dropped, then the evaluator is truly freed
    if let Ok(mut guard) = METAL_EVALUATOR.write() {
        *guard = None;
    }
    if let Ok(mut guard) = SPARSE_METAL_EVALUATOR.write() {
        *guard = None;
    }
    if let Ok(mut guard) = GROUP_EVALUATOR.write() {
        *guard = None;
    }
}

/// Threshold for switching to sparse memory (2^12 = 4K addresses)
pub(crate) const SPARSE_THRESHOLD: usize = 12;

/// B12+B13 hybrid split state — PER-SHAPE rolling per-genome wall-time
/// estimates for CPU and GPU paths. Keyed by `(neurons_per_cluster, max_bits)`
/// so different shapes don't cross-contaminate each other's learning curves.
///
/// EMA factor 0.3 on new measurements (slow drift; fast convergence).
/// Defaults seed at (60ms, 60ms) → initial 50/50 split; converges in 1-2
/// batches for any given shape.
pub(crate) type ShapeKey = (Vec<usize>, usize);

#[derive(Clone, Copy)]
pub(crate) struct HybridSplitState {
    pub cpu_time_per_genome_us: f64,
    pub gpu_time_per_genome_us: f64,
}

impl Default for HybridSplitState {
    fn default() -> Self {
        Self {
            cpu_time_per_genome_us: 60_000.0,
            gpu_time_per_genome_us: 60_000.0,
        }
    }
}

pub(crate) static HYBRID_SPLIT_STATE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<ShapeKey, HybridSplitState>>
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Helper: read state for a shape (or default if first time seen).
pub(crate) fn read_shape_state(shape: &ShapeKey) -> HybridSplitState {
    let m = HYBRID_SPLIT_STATE.lock().unwrap();
    m.get(shape).copied().unwrap_or_default()
}

/// Helper: update state for a shape via EMA (factor 0.3 on new measurements).
pub(crate) fn update_shape_state(shape: &ShapeKey, cpu_us: f64, gpu_us: f64) {
    let mut m = HYBRID_SPLIT_STATE.lock().unwrap();
    let s = m.entry(shape.clone()).or_default();
    s.cpu_time_per_genome_us = s.cpu_time_per_genome_us * 0.7 + cpu_us * 0.3;
    s.gpu_time_per_genome_us = s.gpu_time_per_genome_us * 0.7 + gpu_us * 0.3;
}

/// Whether the GPU batched-train path (marker-FSM Metal kernel) is enabled.
///
/// Canonical name: `WNN_GPU_BATCHED_TRAIN`. Backward-compat alias: `WNN_OPTION_B`.
/// Both are treated as enabled only when SET AND NON-EMPTY — `=` (empty string)
/// counts as unset (fixes a subtle bug where shell scripts passing through
/// unset values accidentally enabled the path).
pub(crate) fn gpu_batched_train_enabled() -> bool {
    let nonempty = |name: &str| {
        std::env::var(name).map(|v| !v.is_empty()).unwrap_or(false)
    };
    nonempty("WNN_GPU_BATCHED_TRAIN") || nonempty("WNN_OPTION_B")
}

/// Whether trace output for the GPU batched-train path is enabled.
///
/// Canonical: `WNN_GPU_BATCHED_TRAIN_TRACE`. Alias: `WNN_OPTION_B_TRACE`.
pub(crate) fn gpu_batched_train_trace() -> bool {
    let nonempty = |name: &str| {
        std::env::var(name).map(|v| !v.is_empty()).unwrap_or(false)
    };
    nonempty("WNN_GPU_BATCHED_TRAIN_TRACE") || nonempty("WNN_OPTION_B_TRACE")
}
