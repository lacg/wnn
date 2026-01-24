//! Persistent Evaluation Worker Pool
//!
//! Provides a long-lived worker thread for GPU evaluation that eliminates
//! per-call overhead (thread spawn, channel creation) when calling evaluation
//! functions repeatedly.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────────────────────────────┐
//! │   Caller    │     │         EvalWorkerPool              │
//! │             │     │  ┌─────────────────────────────┐    │
//! │  submit()   │────►│  │   Bounded Request Channel   │    │
//! │             │     │  └─────────────────────────────┘    │
//! │             │     │              │                      │
//! │             │     │              ▼                      │
//! │             │     │  ┌─────────────────────────────┐    │
//! │             │     │  │    Persistent Eval Thread   │    │
//! │             │     │  │  - GPU evaluators cached    │    │
//! │             │     │  │  - Processes batches        │    │
//! │             │     │  └─────────────────────────────┘    │
//! │             │     │              │                      │
//! │             │◄────│  One-shot response channel          │
//! └─────────────┘     └─────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! // Get the singleton worker (lazily initialized)
//! let worker = get_eval_worker();
//!
//! // Create shared eval data (Arc for zero-copy)
//! let eval_data = Arc::new(EvalData { ... });
//!
//! // Submit batch for evaluation
//! let results = worker.evaluate(exports, eval_data);
//! ```
//!
//! # Benefits
//!
//! - **Zero thread spawn overhead**: Worker thread stays alive for session
//! - **Zero channel creation overhead**: Request channel is reused
//! - **GPU warmth**: Metal evaluators stay initialized
//! - **Pipelining**: Bounded channel allows training batch N+1 while evaluating N

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use crate::adaptive::{evaluate_genome_hybrid, GenomeExport, get_metal_evaluator, get_sparse_metal_evaluator};

// ============================================================================
// Configuration
// ============================================================================

/// Default channel capacity for pipelining (train batch N+1 while eval batch N)
const DEFAULT_CHANNEL_CAPACITY: usize = 2;

// ============================================================================
// Data Structures
// ============================================================================

/// Shared evaluation data (immutable during evaluation session)
///
/// Wrapped in Arc for zero-copy sharing between caller and worker.
#[derive(Clone)]
pub struct EvalData {
    pub eval_input_bits: Vec<bool>,
    pub eval_targets: Vec<i64>,
    pub num_eval: usize,
    pub num_clusters: usize,
    pub total_input_bits: usize,
    pub empty_value: f32,
}

impl EvalData {
    /// Create new EvalData from evaluation parameters
    pub fn new(
        eval_input_bits: Vec<bool>,
        eval_targets: Vec<i64>,
        num_eval: usize,
        num_clusters: usize,
        total_input_bits: usize,
        empty_value: f32,
    ) -> Self {
        Self {
            eval_input_bits,
            eval_targets,
            num_eval,
            num_clusters,
            total_input_bits,
            empty_value,
        }
    }

    /// Create from slices (copies data)
    pub fn from_slices(
        eval_input_bits: &[bool],
        eval_targets: &[i64],
        num_eval: usize,
        num_clusters: usize,
        total_input_bits: usize,
        empty_value: f32,
    ) -> Self {
        Self::new(
            eval_input_bits.to_vec(),
            eval_targets.to_vec(),
            num_eval,
            num_clusters,
            total_input_bits,
            empty_value,
        )
    }
}

/// Request to evaluate a batch of genome exports
struct EvalBatchRequest {
    exports: Vec<(usize, GenomeExport)>,
    eval_data: Arc<EvalData>,
    response_tx: mpsc::Sender<EvalBatchResponse>,
}

/// Response containing evaluation results
pub struct EvalBatchResponse {
    pub results: Vec<(usize, f64, f64)>, // (genome_idx, ce, accuracy)
}

// ============================================================================
// Worker Pool
// ============================================================================

/// Persistent worker pool for GPU evaluation
///
/// Manages a long-lived worker thread that processes evaluation requests.
/// The worker stays alive for the entire session, eliminating per-call overhead.
pub struct EvalWorkerPool {
    request_tx: SyncSender<EvalBatchRequest>,
    _worker_handle: JoinHandle<()>,
}

impl EvalWorkerPool {
    /// Create a new worker pool with default configuration
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    /// Create a new worker pool with custom channel capacity
    ///
    /// Higher capacity allows more pipelining but uses more memory.
    pub fn with_capacity(capacity: usize) -> Self {
        let (request_tx, request_rx) = mpsc::sync_channel::<EvalBatchRequest>(capacity);

        let worker_handle = thread::spawn(move || {
            Self::worker_loop(request_rx);
        });

        Self {
            request_tx,
            _worker_handle: worker_handle,
        }
    }

    /// Main worker loop - processes requests until channel closes
    fn worker_loop(request_rx: Receiver<EvalBatchRequest>) {
        // Detailed timing (enabled via WNN_EVAL_TIMING env var)
        let timing_enabled = std::env::var("WNN_EVAL_TIMING").is_ok();

        // Process requests until channel closes
        while let Ok(request) = request_rx.recv() {
            let eval_data = &request.eval_data;
            let num_genomes = request.exports.len();

            // Fetch evaluators each batch (allows reset_metal_evaluators to take effect)
            // These are Arc references to the shared evaluators in adaptive.rs
            let metal_arc = get_metal_evaluator();
            let sparse_metal_arc = get_sparse_metal_evaluator();
            let metal = metal_arc.as_ref().map(|a| a.as_ref());
            let sparse_metal = sparse_metal_arc.as_ref().map(|a| a.as_ref());

            let batch_start = std::time::Instant::now();

            // Evaluate exports sequentially - GPU doesn't benefit from parallel access
            // (multiple threads competing for GPU causes contention and slowdown)
            let results: Vec<(usize, f64, f64)> = request
                .exports
                .into_iter()
                .map(|(genome_idx, export)| {
                    let (ce, acc) = evaluate_genome_hybrid(
                        &export,
                        &eval_data.eval_input_bits,
                        &eval_data.eval_targets,
                        eval_data.num_eval,
                        eval_data.num_clusters,
                        eval_data.total_input_bits,
                        eval_data.empty_value,
                        metal,
                        sparse_metal,
                    );
                    (genome_idx, ce, acc)
                })
                .collect();

            if timing_enabled && num_genomes > 0 {
                let batch_elapsed = batch_start.elapsed();
                let per_genome_ms = batch_elapsed.as_millis() as f64 / num_genomes as f64;
                eprintln!(
                    "[EVAL_WORKER] batch={} genomes, total={:.0}ms, per_genome={:.0}ms",
                    num_genomes, batch_elapsed.as_millis(), per_genome_ms
                );
            }

            // Send results back (ignore error if receiver dropped)
            let _ = request.response_tx.send(EvalBatchResponse { results });
        }
    }

    /// Submit a batch for evaluation and wait for results
    ///
    /// # Arguments
    /// * `exports` - Vector of (genome_idx, GenomeExport) tuples
    /// * `eval_data` - Shared evaluation data (Arc for zero-copy)
    ///
    /// # Returns
    /// Vector of (genome_idx, cross_entropy, accuracy) tuples
    pub fn evaluate(
        &self,
        exports: Vec<(usize, GenomeExport)>,
        eval_data: Arc<EvalData>,
    ) -> Vec<(usize, f64, f64)> {
        // Create one-shot response channel
        let (response_tx, response_rx) = mpsc::channel();

        // Send request
        self.request_tx
            .send(EvalBatchRequest {
                exports,
                eval_data,
                response_tx,
            })
            .expect("Eval worker channel closed unexpectedly");

        // Wait for and return results
        response_rx
            .recv()
            .expect("Eval worker response channel closed unexpectedly")
            .results
    }

    /// Submit a batch for evaluation without waiting (async-style)
    ///
    /// Returns a receiver that will contain results when ready.
    pub fn evaluate_async(
        &self,
        exports: Vec<(usize, GenomeExport)>,
        eval_data: Arc<EvalData>,
    ) -> Receiver<EvalBatchResponse> {
        let (response_tx, response_rx) = mpsc::channel();

        self.request_tx
            .send(EvalBatchRequest {
                exports,
                eval_data,
                response_tx,
            })
            .expect("Eval worker channel closed unexpectedly");

        response_rx
    }
}

impl Default for EvalWorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Global Singleton
// ============================================================================

/// Global persistent eval worker (lazily initialized)
static EVAL_WORKER: OnceLock<EvalWorkerPool> = OnceLock::new();

/// Get or initialize the persistent eval worker pool
///
/// The worker is created on first access and stays alive for the session.
pub fn get_eval_worker() -> &'static EvalWorkerPool {
    EVAL_WORKER.get_or_init(EvalWorkerPool::new)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_data_creation() {
        let data = EvalData::new(
            vec![true, false, true],
            vec![1, 2, 3],
            3,
            10,
            64,
            0.5,
        );
        assert_eq!(data.num_eval, 3);
        assert_eq!(data.num_clusters, 10);
    }

    #[test]
    fn test_eval_data_from_slices() {
        let bits = [true, false];
        let targets = [1i64, 2];
        let data = EvalData::from_slices(&bits, &targets, 2, 5, 32, 0.5);
        assert_eq!(data.eval_input_bits.len(), 2);
        assert_eq!(data.eval_targets.len(), 2);
    }
}
