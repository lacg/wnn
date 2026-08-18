//! Process-wide cooperative-cancellation flag for the controller and IDS evaluators.
//!
//! Plain `static AtomicBool` — every Rust call site polls `check_cancel()` at safe
//! boundaries (between genomes / episodes / GPU dispatch chunks) and returns
//! partial results when the flag is set. Python sets the flag from its SIGTERM
//! handler via the `set_cancel_flag()` PyO3 function.
//!
//! Design choice: ONE global flag rather than per-evaluator tokens. The runtime
//! never has more than one evaluation in flight per process (worker is
//! single-threaded for flow execution; controller GA runs are sequential). A
//! global keeps the API surface tiny — `check_cancel()` from any Rust callsite,
//! `set_cancel_flag()` from Python.
//!
//! Established 2026-05-31 after the Plan A v3 SIGTERM round-trip revealed the
//! 8.5-minute worst-case wait when Rust evaluators don't cooperate with shutdown.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;

/// The single process-wide cancel flag. Read with Relaxed ordering at every
/// per-episode / per-chunk safe point.
pub static GLOBAL_CANCEL: AtomicBool = AtomicBool::new(false);

/// Cheap atomic load. Inline so the call doesn't add overhead in tight loops.
#[inline(always)]
pub fn check_cancel() -> bool
{
	GLOBAL_CANCEL.load(Ordering::Relaxed)
}

/// PyO3: set the flag from Python. Returns immediately; Rust callsites pick it
/// up on their next polling check (typically <= 25 ms on the controller GPU
/// path, faster on CPU paths).
#[pyfunction]
pub fn set_cancel_flag()
{
	GLOBAL_CANCEL.store(true, Ordering::Relaxed);
}

/// PyO3: reset the flag (after handling a previous cancellation, or before
/// starting a new run). Must be called between independent runs in the same
/// process — otherwise an already-set flag would short-circuit the next run.
#[pyfunction]
pub fn reset_cancel_flag()
{
	GLOBAL_CANCEL.store(false, Ordering::Relaxed);
}

/// PyO3: read the current flag state from Python (mostly for tests).
#[pyfunction]
pub fn is_cancelled() -> bool
{
	check_cancel()
}
