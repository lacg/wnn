//! General: availability, resets, thresholds/calibration utilities.
//!
//! Split out of lib.rs (D3, 11/06/2026).

use crate::*;

/// Check if Metal GPU is available
#[pyfunction]
pub(crate) fn metal_available() -> bool
{
	metal_evaluator::MetalEvaluator::is_available()
}

/// Reset Metal evaluators to free accumulated driver state.
///
/// Call this periodically (e.g., every 50 generations) during long optimization
/// runs to prevent slowdown from Metal driver state accumulation.
///
/// The evaluators will be lazily re-initialized on next use.
#[pyfunction]
pub(crate) fn reset_metal_evaluators()
{
	adaptive::reset_metal_evaluators();
}

/// Get number of CPU cores available for rayon
#[pyfunction]
pub(crate) fn cpu_cores() -> usize
{
	rayon::current_num_threads()
}

/// Evaluate cascade with all RAMs, optimizing one at a time
/// base_connectivities: [n2_conn, n3_conn, n4_conn, n5_conn, n6_conn]
/// candidates: candidate connectivities for the RAM at target_ram_idx
/// target_ram_idx: 0=n2, 1=n3, 2=n4, 3=n5, 4=n6
/// Find threshold maximizing weighted fitness (F1, FPR, Acc, CE weights).
/// Returns (threshold, f1, fpr, acc, fitness).
#[pyfunction]
pub(crate) fn find_optimal_threshold_fitness_py(
	scores: Vec<f64>,
	labels: Vec<i64>,
	w_ce: f32,
	w_f1: f32,
	w_fpr: f32,
	w_acc: f32,
) -> PyResult<(f64, f64, f64, f64, f64)>
{
	Ok(adaptive::find_optimal_threshold_fitness(
		&scores, &labels, w_ce, w_f1, w_fpr, w_acc,
	))
}

/// Fit Platt scaling. Returns (threshold, a, b).
#[pyfunction]
pub(crate) fn fit_platt_scaling_py(scores: Vec<f64>, labels: Vec<i64>)
	-> PyResult<(f64, f64, f64)>
{
	Ok(adaptive::fit_platt_scaling(&scores, &labels))
}

/// Fit Beta calibration. Returns (threshold, a, b, c).
#[pyfunction]
pub(crate) fn fit_beta_calibration_py(
	scores: Vec<f64>,
	labels: Vec<i64>,
) -> PyResult<(f64, f64, f64, f64)>
{
	Ok(adaptive::fit_beta_calibration(&scores, &labels))
}

/// Fit empirical threshold. Returns (threshold, n_bins).
#[pyfunction]
pub(crate) fn fit_empirical_threshold_py(
	scores: Vec<f64>,
	labels: Vec<i64>,
) -> PyResult<(f64, usize)>
{
	Ok(adaptive::fit_empirical_threshold(&scores, &labels))
}

/// Compute (CE, accuracy, F1-macro, FPR) for a single-cluster binary classifier
/// from raw scores at a given threshold. CE is binary cross-entropy
/// (threshold-independent); accuracy/F1/FPR depend on `threshold`.
/// `normal_class` is 0 by default, set to 1 when flip_labels is active.
#[pyfunction]
#[pyo3(signature = (scores, labels, threshold, normal_class=0))]
pub(crate) fn compute_binary_metrics_at_threshold_py(
	scores: Vec<f64>,
	labels: Vec<i64>,
	threshold: f64,
	normal_class: usize,
) -> PyResult<(f64, f64, f64, f64)>
{
	Ok(adaptive::compute_binary_metrics_at_threshold(
		&scores,
		&labels,
		threshold,
		normal_class,
	))
}

/// Sweep scores for the F1-macro-optimal threshold (binary classification).
/// Returns (threshold, f1_macro, fpr).
#[pyfunction]
pub(crate) fn find_optimal_threshold_f1_py(
	scores: Vec<f64>,
	labels: Vec<i64>,
) -> PyResult<(f64, f64, f64)>
{
	Ok(adaptive::find_optimal_threshold_f1(&scores, &labels))
}
