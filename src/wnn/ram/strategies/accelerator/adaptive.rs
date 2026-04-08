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

// =============================================================================
// Resettable Metal Evaluators
// =============================================================================
//
// Metal evaluators can accumulate driver-level state over long runs, causing
// slowdowns. These use Arc + RwLock to allow periodic reset.
// Call reset_metal_evaluators() every N generations to recreate fresh evaluators.

// Global counter incremented on each reset
static RESET_GENERATION: AtomicU64 = AtomicU64::new(0);

// Which class index is "normal/benign" for FPR computation.
// 0 = default (normal is class 0), 1 = flip_labels active (normal is class 1).
use std::sync::atomic::AtomicUsize;
static NORMAL_CLASS: AtomicUsize = AtomicUsize::new(0);

pub fn set_normal_class(c: usize) {
    NORMAL_CLASS.store(c, Ordering::Relaxed);
}

pub fn get_normal_class() -> usize {
    NORMAL_CLASS.load(Ordering::Relaxed)
}

// Fitness weights for threshold optimization (global, set per flow).
// When set, threshold sweep maximizes fitness instead of F1.
// Format: [w_ce, w_f1, w_fpr, w_acc] stored as u32 bits.
use std::sync::atomic::AtomicU32;
static FITNESS_W_CE: AtomicU32 = AtomicU32::new(0);    // 0.0 = not set
static FITNESS_W_F1: AtomicU32 = AtomicU32::new(0);
static FITNESS_W_FPR: AtomicU32 = AtomicU32::new(0);
static FITNESS_W_ACC: AtomicU32 = AtomicU32::new(0);
static FITNESS_THRESHOLD_ENABLED: AtomicUsize = AtomicUsize::new(0);

pub fn set_fitness_weights(w_ce: f32, w_f1: f32, w_fpr: f32, w_acc: f32) {
    FITNESS_W_CE.store(w_ce.to_bits(), Ordering::Relaxed);
    FITNESS_W_F1.store(w_f1.to_bits(), Ordering::Relaxed);
    FITNESS_W_FPR.store(w_fpr.to_bits(), Ordering::Relaxed);
    FITNESS_W_ACC.store(w_acc.to_bits(), Ordering::Relaxed);
    FITNESS_THRESHOLD_ENABLED.store(1, Ordering::Relaxed);
}

pub fn clear_fitness_weights() {
    FITNESS_THRESHOLD_ENABLED.store(0, Ordering::Relaxed);
}

/// Find optimal threshold using fitness weights if set, otherwise F1.
pub fn find_optimal_threshold_auto(scores: &[f64], labels: &[i64]) -> (f64, f64, f64) {
    if FITNESS_THRESHOLD_ENABLED.load(Ordering::Relaxed) == 1 {
        let w_ce = f32::from_bits(FITNESS_W_CE.load(Ordering::Relaxed));
        let w_f1 = f32::from_bits(FITNESS_W_F1.load(Ordering::Relaxed));
        let w_fpr = f32::from_bits(FITNESS_W_FPR.load(Ordering::Relaxed));
        let w_acc = f32::from_bits(FITNESS_W_ACC.load(Ordering::Relaxed));
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
fn get_group_evaluator() -> Option<Arc<crate::metal_ramlm::MetalGroupEvaluator>> {
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
const SPARSE_THRESHOLD: usize = 12;

use crate::neuron_memory::{
    FALSE, TRUE, EMPTY, BITS_PER_CELL, CELLS_PER_WORD, CELL_MASK,
    compute_address, NeuronTrainMeta,
};

/// Get the EMPTY cell value from the unified global setting
fn get_empty_value() -> f32 {
    crate::neuron_memory::get_empty_value()
}

/// Convert a raw cell value to a forward-pass weight based on memory mode.
///
/// - TERNARY: FALSE=0.0, TRUE=1.0, EMPTY=empty_value
/// - QUAD_WEIGHTED: QUAD_WEIGHTS[cell] = [0.0, 0.25, 0.75, 1.0]
/// - QUAD_BINARY: same as QUAD_WEIGHTED (uses same 4-state encoding)
#[inline(always)]
fn cell_to_weight(cell: i64, memory_mode: u8, empty_value: f32) -> f32 {
    match memory_mode {
        crate::neuron_memory::MODE_QUAD_BINARY | crate::neuron_memory::MODE_QUAD_WEIGHTED => {
            crate::neuron_memory::QUAD_WEIGHTS[cell.clamp(0, 3) as usize]
        }
        _ => {
            // TERNARY
            match cell {
                FALSE => 0.0,
                TRUE => 1.0,
                _ => empty_value,
            }
        }
    }
}

/// Compute F1-macro from per-example predictions and targets.
///
/// Builds a confusion matrix, computes per-class precision/recall/F1,
/// and returns the macro-average F1 score. Shared by all evaluation paths
/// (GPU FAST_PATH, GPU FULL_PATH, CPU FALLBACK).
pub fn compute_f1_macro(predictions: &[u32], targets: &[i64], num_classes: usize) -> f64 {
    if num_classes == 0 || predictions.is_empty() {
        return 0.0;
    }

    // Build confusion matrix: confusion[true_class * K + predicted_class]
    let mut confusion = vec![0u64; num_classes * num_classes];
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        let t = *target as usize;
        let p = *pred as usize;
        if t < num_classes && p < num_classes {
            confusion[t * num_classes + p] += 1;
        }
    }

    // Compute per-class F1
    let mut f1_sum = 0.0f64;
    let mut num_active_classes = 0usize;

    for c in 0..num_classes {
        let tp = confusion[c * num_classes + c] as f64;
        let fp: f64 = (0..num_classes)
            .filter(|&t| t != c)
            .map(|t| confusion[t * num_classes + c] as f64)
            .sum();
        let fn_count: f64 = (0..num_classes)
            .filter(|&p| p != c)
            .map(|p| confusion[c * num_classes + p] as f64)
            .sum();

        // Skip classes with no support (no true examples)
        if tp + fn_count == 0.0 {
            continue;
        }

        num_active_classes += 1;
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        f1_sum += f1;
    }

    if num_active_classes == 0 { 0.0 } else { f1_sum / num_active_classes as f64 }
}

/// Compute both F1-macro and FPR from the same confusion matrix.
/// FPR = fraction of `normal_class` examples misclassified as any other class.
/// `normal_class` is typically 0, but set to 1 when flip_labels is active
/// so FPR always measures false alarms on the original benign traffic.
/// Returns (f1_macro, fpr).
pub fn compute_f1_fpr(predictions: &[u32], targets: &[i64], num_classes: usize) -> (f64, f64) {
    compute_f1_fpr_with_normal_class(predictions, targets, num_classes, 0)
}

pub fn compute_f1_fpr_with_normal_class(predictions: &[u32], targets: &[i64], num_classes: usize, normal_class: usize) -> (f64, f64) {
    if num_classes == 0 || predictions.is_empty() {
        return (0.0, 0.0);
    }

    // Build confusion matrix: confusion[true_class * K + predicted_class]
    let mut confusion = vec![0u64; num_classes * num_classes];
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        let t = *target as usize;
        let p = *pred as usize;
        if t < num_classes && p < num_classes {
            confusion[t * num_classes + p] += 1;
        }
    }

    let mut f1_sum = 0.0f64;
    let mut num_active_classes = 0usize;

    for c in 0..num_classes {
        let tp = confusion[c * num_classes + c] as f64;
        let fp: f64 = (0..num_classes)
            .filter(|&t| t != c)
            .map(|t| confusion[t * num_classes + c] as f64)
            .sum();
        let fn_count: f64 = (0..num_classes)
            .filter(|&p| p != c)
            .map(|p| confusion[c * num_classes + p] as f64)
            .sum();
        // Skip classes with no support (no true examples)
        if tp + fn_count == 0.0 {
            continue;
        }

        num_active_classes += 1;

        // F1
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        f1_sum += f1;

    }

    if num_active_classes == 0 {
        (0.0, 0.0)
    } else {
        let f1_macro = f1_sum / num_active_classes as f64;

        // IDS FPR: fraction of Normal samples misclassified as any attack class.
        // normal_class is 0 by default, but 1 when flip_labels is active, so
        // FPR always measures false alarms on the original benign traffic.
        //   FPR = (Normal predicted as non-Normal) / (all Normal samples)
        let normal_total: f64 = (0..num_classes)
            .map(|p| confusion[normal_class * num_classes + p] as f64)
            .sum();
        let normal_correct = confusion[normal_class * num_classes + normal_class] as f64;
        let fpr = if normal_total > 0.0 {
            (normal_total - normal_correct) / normal_total
        } else {
            0.0
        };

        (f1_macro, fpr)
    }
}

/// Find the optimal threshold for binary (single-cluster) classification.
///
/// Efficiently sweeps all unique score boundaries in O(n log n) time.
/// Returns (threshold, f1_macro, fpr) where f1_macro is maximized.
/// The threshold is set to the midpoint between adjacent sorted scores.
pub fn find_optimal_threshold_f1(scores: &[f64], labels: &[i64]) -> (f64, f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.5, 0.0, 0.0);
    }

    // Create (score, label) pairs and sort by score ascending
    let mut pairs: Vec<(f64, i64)> = scores.iter().copied()
        .zip(labels.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = labels.iter().filter(|&&l| l == 1).count() as u64;
    let total_neg = (n as u64) - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return (0.5, 0.0, 0.0);
    }

    // Start: threshold = -inf → everything predicted as 1 (attack)
    let mut tp = total_pos;
    let mut fp = total_neg;
    let mut fn_count = 0u64;
    let mut tn = 0u64;

    let mut best_f1 = 0.0f64;
    let mut best_threshold = 0.5f64;
    let mut best_fpr = 1.0f64;

    // Sweep threshold from low to high
    // Moving examples below threshold from "predicted 1" to "predicted 0"
    let mut i = 0;
    while i < n {
        let current_score = pairs[i].0;

        // Process all examples with the same score
        while i < n && pairs[i].0 == current_score {
            if pairs[i].1 == 1 {
                tp -= 1;
                fn_count += 1;
            } else {
                fp -= 1;
                tn += 1;
            }
            i += 1;
        }

        // Compute F1-macro at this threshold boundary
        // F1 for attack class (label=1)
        let f1_pos = if 2 * tp + fp + fn_count > 0 {
            2.0 * tp as f64 / (2.0 * tp as f64 + fp as f64 + fn_count as f64)
        } else {
            0.0
        };
        // F1 for normal class (label=0)
        let f1_neg = if 2 * tn + fn_count + fp > 0 {
            2.0 * tn as f64 / (2.0 * tn as f64 + fn_count as f64 + fp as f64)
        } else {
            0.0
        };
        let f1_macro = (f1_pos + f1_neg) / 2.0;

        if f1_macro > best_f1 {
            best_f1 = f1_macro;
            // Threshold: midpoint between current score and next, or slightly above if last
            best_threshold = if i < n {
                (current_score + pairs[i].0) / 2.0
            } else {
                current_score + 1e-6
            };
            // FPR = normal samples predicted as attack / total normal
            best_fpr = fp as f64 / total_neg as f64;
        }
    }

    (best_threshold, best_f1, best_fpr)
}

/// Find the optimal threshold maximizing weighted fitness instead of F1.
///
/// Same O(n log n) sweep as find_optimal_threshold_f1, but at each boundary
/// computes fitness = w_f1*F1 + w_fpr*(1-FPR) + w_acc*Acc + w_ce*(1-CE_approx)
/// and picks the threshold with the highest fitness.
///
/// Returns (threshold, f1_macro, fpr, accuracy, fitness).
pub fn find_optimal_threshold_fitness(
    scores: &[f64],
    labels: &[i64],
    w_ce: f32,
    w_f1: f32,
    w_fpr: f32,
    w_acc: f32,
) -> (f64, f64, f64, f64, f64) {
    let n = scores.len();
    if n == 0 {
        return (0.5, 0.0, 0.0, 0.0, 0.0);
    }

    let mut pairs: Vec<(f64, i64)> = scores.iter().copied()
        .zip(labels.iter().copied())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = labels.iter().filter(|&&l| l == 1).count() as u64;
    let total_neg = (n as u64) - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return (0.5, 0.0, 0.0, 0.0, 0.0);
    }

    // Precompute CE for the full dataset (doesn't change with threshold)
    let ce: f64 = scores.iter().zip(labels.iter()).map(|(&s, &l)| {
        let p = s.max(1e-10).min(1.0 - 1e-10);
        -(l as f64 * p.ln() + (1.0 - l as f64) * (1.0 - p).ln())
    }).sum::<f64>() / n as f64;
    let ce_score = (1.0 - ce).max(0.0);

    // Start: threshold = -inf → everything predicted as 1 (attack)
    let mut tp = total_pos;
    let mut fp = total_neg;
    let mut fn_count = 0u64;
    let mut tn = 0u64;

    let mut best_fitness = -1.0f64;
    let mut best_threshold = 0.5f64;
    let mut best_f1 = 0.0f64;
    let mut best_fpr_val = 1.0f64;
    let mut best_acc = 0.0f64;

    let n_f64 = n as f64;

    let mut i = 0;
    while i < n {
        let current_score = pairs[i].0;

        while i < n && pairs[i].0 == current_score {
            if pairs[i].1 == 1 {
                tp -= 1;
                fn_count += 1;
            } else {
                fp -= 1;
                tn += 1;
            }
            i += 1;
        }

        // F1-macro
        let f1_pos = if 2 * tp + fp + fn_count > 0 {
            2.0 * tp as f64 / (2.0 * tp as f64 + fp as f64 + fn_count as f64)
        } else { 0.0 };
        let f1_neg = if 2 * tn + fn_count + fp > 0 {
            2.0 * tn as f64 / (2.0 * tn as f64 + fn_count as f64 + fp as f64)
        } else { 0.0 };
        let f1_macro = (f1_pos + f1_neg) / 2.0;

        let fpr = fp as f64 / total_neg as f64;
        let acc = (tp + tn) as f64 / n_f64;

        let fitness = w_f1 as f64 * f1_macro
            + w_fpr as f64 * (1.0 - fpr)
            + w_acc as f64 * acc
            + w_ce as f64 * ce_score;

        if fitness > best_fitness {
            best_fitness = fitness;
            best_threshold = if i < n {
                (current_score + pairs[i].0) / 2.0
            } else {
                current_score + 1e-6
            };
            best_f1 = f1_macro;
            best_fpr_val = fpr;
            best_acc = acc;
        }
    }

    (best_threshold, best_f1, best_fpr_val, best_acc, best_fitness)
}

/// Fit Platt scaling on scores+labels, return threshold.
/// Learns sigmoid P(attack) = 1/(1+exp(-(a*score+b))) via Newton's method.
/// Returns (threshold, a, b) where threshold = -b/a.
pub fn fit_platt_scaling(scores: &[f64], labels: &[i64]) -> (f64, f64, f64) {
    let n = scores.len();
    if n == 0 { return (0.5, 1.0, 0.0); }

    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return (0.5, 1.0, 0.0); }

    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    let mut a = 1.0f64;
    let mut b = 0.0f64;

    for _ in 0..100 {
        let mut g_a = 0.0f64;
        let mut g_b = 0.0f64;
        let mut h_aa = 0.0f64;
        let mut h_ab = 0.0f64;
        let mut h_bb = 0.0f64;

        for i in 0..n {
            let fval = a * scores[i] + b;
            let p = if fval >= 0.0 {
                1.0 / (1.0 + (-fval).exp())
            } else {
                let ef = fval.exp();
                ef / (1.0 + ef)
            };
            let p = p.max(1e-15).min(1.0 - 1e-15);

            let t = if labels[i] == 1 { t_pos } else { t_neg };
            let d = p - t;
            g_a += d * scores[i];
            g_b += d;
            let w = p * (1.0 - p);
            h_aa += w * scores[i] * scores[i];
            h_ab += w * scores[i];
            h_bb += w;
        }

        h_aa += 1e-6;
        h_bb += 1e-6;
        let det = h_aa * h_bb - h_ab * h_ab;
        if det.abs() < 1e-12 { break; }
        let da = -(h_bb * g_a - h_ab * g_b) / det;
        let db = -(h_aa * g_b - h_ab * g_a) / det;
        a += da;
        b += db;
        if da.abs() < 1e-8 && db.abs() < 1e-8 { break; }
    }

    let threshold = if a.abs() > 1e-10 { -b / a } else { 0.5 };
    (threshold, a, b)
}

/// Fit Beta calibration on scores+labels, return threshold.
/// Learns logit(P) = a*ln(s) + b*(-ln(1-s)) + c via gradient descent.
/// Returns (threshold, a, b, c).
pub fn fit_beta_calibration(scores: &[f64], labels: &[i64]) -> (f64, f64, f64, f64) {
    let n = scores.len();
    if n == 0 { return (0.5, 1.0, 1.0, 0.0); }

    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return (0.5, 1.0, 1.0, 0.0); }

    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);

    let eps = 1e-10f64;
    let x1: Vec<f64> = scores.iter().map(|&s| s.max(eps).ln()).collect();
    let x2: Vec<f64> = scores.iter().map(|&s| -(1.0 - s).max(eps).ln()).collect();

    let mut a = 1.0f64;
    let mut b = 1.0f64;
    let mut c = 0.0f64;

    for _ in 0..100 {
        let mut g_a = 0.0f64;
        let mut g_b = 0.0f64;
        let mut g_c = 0.0f64;
        let mut h_aa = 0.0f64;
        let mut h_bb = 0.0f64;
        let mut h_cc = 0.0f64;

        for i in 0..n {
            let fval = a * x1[i] + b * x2[i] + c;
            let p = if fval >= 0.0 {
                1.0 / (1.0 + (-fval).exp())
            } else {
                let ef = fval.exp();
                ef / (1.0 + ef)
            };
            let p = p.max(1e-15).min(1.0 - 1e-15);

            let t = if labels[i] == 1 { t_pos } else { t_neg };
            let d = p - t;
            g_a += d * x1[i];
            g_b += d * x2[i];
            g_c += d;
            let w = p * (1.0 - p);
            h_aa += w * x1[i] * x1[i];
            h_bb += w * x2[i] * x2[i];
            h_cc += w;
        }

        h_aa += 1e-6;
        h_bb += 1e-6;
        h_cc += 1e-6;
        a -= 0.1 * g_a / h_aa;
        b -= 0.1 * g_b / h_bb;
        c -= 0.1 * g_c / h_cc;
        if (g_a / h_aa.max(1e-10)).abs() < 1e-6 && (g_b / h_bb.max(1e-10)).abs() < 1e-6 {
            break;
        }
    }

    // Binary search for threshold where a*ln(s) + b*(-ln(1-s)) + c = 0
    let mut lo = 0.001f64;
    let mut hi = 0.999f64;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let val = a * mid.ln() + b * (-(1.0 - mid).ln()) + c;
        if val < 0.0 { lo = mid; } else { hi = mid; }
    }
    let threshold = (lo + hi) / 2.0;

    (threshold, a, b, c)
}

/// Fit empirical threshold: find lowest score where P(attack|score) >= 0.5.
/// Returns (threshold, n_bins).
pub fn fit_empirical_threshold(scores: &[f64], labels: &[i64]) -> (f64, usize) {
    use std::collections::BTreeMap;
    let mut bins: BTreeMap<i64, (u64, u64)> = BTreeMap::new(); // score_key → (normal, attack)

    for (&s, &l) in scores.iter().zip(labels.iter()) {
        let key = (s * 1_000_000.0) as i64; // round to 6 decimals
        let entry = bins.entry(key).or_insert((0, 0));
        if l == 1 { entry.1 += 1; } else { entry.0 += 1; }
    }

    let n_bins = bins.len();
    let mut threshold = 0.5f64;
    for (&key, &(normal, attack)) in &bins {
        let total = normal + attack;
        if total > 0 && (attack as f64 / total as f64) >= 0.5 {
            threshold = key as f64 / 1_000_000.0;
            break;
        }
    }

    (threshold, n_bins)
}

/// Check if group coalescing is enabled (set WNN_COALESCE_GROUPS=1)
fn use_coalesced_groups() -> bool {
    std::env::var("WNN_COALESCE_GROUPS").is_ok()
}

/// Build config groups with optional coalescing based on environment variable
/// When WNN_COALESCE_GROUPS is set, similar neuron counts are bucketed together
/// to reduce GPU dispatch overhead while preserving accuracy through masking
pub fn build_groups(bits_per_cluster: &[usize], neurons_per_cluster: &[usize]) -> Vec<ConfigGroup> {
    if use_coalesced_groups() {
        build_config_groups_coalesced(bits_per_cluster, neurons_per_cluster)
    } else {
        build_config_groups(bits_per_cluster, neurons_per_cluster)
    }
}

/// Reorganize connections from Python's cluster-order layout to coalesced group layout
///
/// Python generates connections in cluster ID order:
///   [cluster_0_conns, cluster_1_conns, ..., cluster_N_conns]
///   where cluster_i has neurons_per_cluster[i] * bits_per_cluster[i] connections
///
/// Coalesced groups expect connections organized by group with padding:
///   [group_0_cluster_conns, group_1_cluster_conns, ...]
///   where each cluster in group has group.neurons (MAX) * group.bits connections
///   and actual connections are followed by padding (-1) to reach MAX neurons
///
/// Returns: padded connections in group order, ready for coalesced evaluation
pub fn reorganize_connections_for_coalescing(
    original_connections: &[i64],
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
    groups: &[ConfigGroup],
) -> Vec<i64> {
    let num_clusters = bits_per_cluster.len();

    // Build mapping: cluster_id -> offset in original_connections
    let mut cluster_offsets = vec![0usize; num_clusters];
    let mut offset = 0;
    for cluster_id in 0..num_clusters {
        cluster_offsets[cluster_id] = offset;
        offset += neurons_per_cluster[cluster_id] * bits_per_cluster[cluster_id];
    }

    // Total size needed for coalesced layout
    let total_size: usize = groups.iter().map(|g| g.conn_size()).sum();
    let mut result = vec![-1i64; total_size];  // Initialize with padding value

    // For each group, copy connections for each cluster (with padding)
    let mut write_offset = 0;
    for group in groups {
        let max_neurons = group.neurons;
        let bits = group.bits;

        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_idx] as usize
            } else {
                max_neurons  // Uniform case
            };

            // Source: original connections for this cluster
            let src_offset = cluster_offsets[cluster_id];
            let src_size = actual_neurons * bits;

            // Destination: position in coalesced layout
            // Each cluster in group gets max_neurons * bits slots
            let dst_offset = write_offset + local_idx * max_neurons * bits;

            // Copy actual connections
            result[dst_offset..dst_offset + src_size]
                .copy_from_slice(&original_connections[src_offset..src_offset + src_size]);

            // Remaining slots (dst_offset + src_size .. dst_offset + max_neurons * bits)
            // are already -1 (padding)
        }

        write_offset += group.cluster_ids.len() * max_neurons * bits;
    }

    result
}

/// Convert per-neuron bits to per-cluster max bits (for `build_groups`).
///
/// `bits_per_neuron` has length `sum(neurons_per_cluster)` — one entry per neuron.
/// Returns one entry per cluster: the maximum bits among that cluster's neurons.
/// Pattern from `bitwise_ramlm.rs:1391-1400`.
pub(crate) fn per_cluster_max_bits(bits_per_neuron: &[usize], neurons_per_cluster: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(neurons_per_cluster.len());
    let mut offset = 0;
    for &nc in neurons_per_cluster {
        let max_b = bits_per_neuron[offset..offset + nc].iter().copied().max().unwrap_or(0);
        result.push(max_b);
        offset += nc;
    }
    result
}

/// Build per-neuron offset tables for heterogeneous-bits training.
///
/// Returns `(cluster_neuron_starts, neuron_conn_offsets)`:
/// - `cluster_neuron_starts[c]` = first neuron index for cluster `c`
/// - `neuron_conn_offsets[n]` = connection start offset for neuron `n` (cumulative sum of bits)
///
/// Pattern from `bitwise_ramlm.rs:683-704` (`compute_genome_layout`).
pub(crate) fn build_neuron_metadata(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let num_clusters = neurons_per_cluster.len();
    let total_neurons: usize = neurons_per_cluster.iter().sum();

    // cluster_neuron_starts[c] = index of first neuron in cluster c
    let mut cluster_neuron_starts = Vec::with_capacity(num_clusters);
    let mut cumul = 0usize;
    for &nc in neurons_per_cluster {
        cluster_neuron_starts.push(cumul);
        cumul += nc;
    }

    // neuron_conn_offsets[n] = start offset in connections array for neuron n
    let mut neuron_conn_offsets = Vec::with_capacity(total_neurons);
    let mut conn_off = 0usize;
    for &b in bits_per_neuron {
        neuron_conn_offsets.push(conn_off);
        conn_off += b;
    }

    (cluster_neuron_starts, neuron_conn_offsets)
}

/// Pad per-neuron connections to group layout for GPU dispatch.
///
/// Each neuron's `n_bits` connections are padded to `group.bits` (= cluster max_bits) with
/// connection index 0 (harmless padding). Same pattern as `bitwise_ramlm.rs:804-820`.
///
/// This replaces `reorganize_connections_for_coalescing` when per-neuron bits are heterogeneous.
pub(crate) fn reorganize_connections_for_gpu(
    original_connections: &[i64],
    per_neuron_bits: &[usize],
    neurons_per_cluster: &[usize],
    groups: &[ConfigGroup],
) -> Vec<i64> {
    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);

    // Total size needed for group layout
    let total_size: usize = groups.iter().map(|g| g.conn_size()).sum();
    // Initialize with -1 (skipped by GPU shader's `if conn_idx >= 0` check)
    let mut result = vec![-1i64; total_size];

    for group in groups {
        let max_neurons = group.neurons;
        let max_bits = group.bits;

        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_idx] as usize
            } else {
                max_neurons
            };

            let neuron_start = cluster_neuron_starts[cluster_id];

            for n in 0..actual_neurons {
                let global_n = neuron_start + n;
                let n_bits = per_neuron_bits[global_n];
                let conn_start = neuron_conn_offsets[global_n];

                // Destination in group layout: PREFIX-pad with -1, real connections at END.
                // GPU shader computes address bit i as (max_bits-1-i), so real connections
                // at the end match training's bit positions (actual_bits-1-i).
                let dst = group.conn_offset + local_idx * max_neurons * max_bits + n * max_bits;
                let pad_size = max_bits - n_bits;
                // Prefix is already -1 from initialization; copy real connections after it
                result[dst + pad_size..dst + pad_size + n_bits]
                    .copy_from_slice(&original_connections[conn_start..conn_start + n_bits]);
            }
        }
    }

    result
}

/// Configuration group - clusters sharing the same (neurons, bits) config
/// For coalesced groups, neurons is the MAX neurons and actual_neurons stores per-cluster values
#[derive(Clone, Debug)]
pub struct ConfigGroup {
    pub neurons: usize,                   // Max neurons (for memory layout)
    pub bits: usize,
    pub words_per_neuron: usize,
    pub cluster_ids: Vec<usize>,          // Global cluster IDs in this group
    pub actual_neurons: Option<Vec<u32>>, // Per-cluster actual neurons (None = all same as neurons)
    pub memory_offset: usize,             // Offset into flattened memory
    pub conn_offset: usize,               // Offset into flattened connections
}

impl ConfigGroup {
    pub fn new(neurons: usize, bits: usize, cluster_ids: Vec<usize>) -> Self {
        let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
        Self {
            neurons,
            bits,
            words_per_neuron,
            cluster_ids,
            actual_neurons: None,  // Uniform: all clusters have same neurons
            memory_offset: 0,
            conn_offset: 0,
        }
    }

    /// Create a coalesced group where clusters may have different actual neuron counts
    /// neurons = max neurons for memory allocation
    /// actual_neurons[i] = actual neuron count for cluster_ids[i]
    pub fn new_coalesced(neurons: usize, bits: usize, cluster_ids: Vec<usize>, actual_neurons: Vec<u32>) -> Self {
        let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
        Self {
            neurons,
            bits,
            words_per_neuron,
            cluster_ids,
            actual_neurons: Some(actual_neurons),
            memory_offset: 0,
            conn_offset: 0,
        }
    }

    pub fn cluster_count(&self) -> usize {
        self.cluster_ids.len()
    }

    pub fn total_neurons(&self) -> usize {
        self.cluster_count() * self.neurons
    }

    /// True total neurons (sum of actual neurons if coalesced)
    pub fn true_total_neurons(&self) -> usize {
        if let Some(ref actual) = self.actual_neurons {
            actual.iter().map(|&n| n as usize).sum()
        } else {
            self.total_neurons()
        }
    }

    pub fn memory_size(&self) -> usize {
        self.total_neurons() * self.words_per_neuron
    }

    pub fn conn_size(&self) -> usize {
        self.total_neurons() * self.bits
    }

    /// Is this a coalesced group with per-cluster masking?
    pub fn is_coalesced(&self) -> bool {
        self.actual_neurons.is_some()
    }
}

/// Maximum GPU output size: 256M addresses = 1GB output buffer.
/// Beyond this, CPU fallback is used to avoid Metal allocation hangs.
const MAX_GPU_ADDRESSES: usize = 256_000_000;

/// Try to compute training addresses on GPU for adaptive training path.
/// Returns None if GPU is unavailable, disabled, or the problem is too large.
pub(crate) fn try_gpu_addresses_adaptive(
    packed_input: &[u64],
    words_per_example: usize,
    per_neuron_bits: &[usize],
    neuron_conn_offsets: &[usize],
    connections: &[i64],
    num_train: usize,
) -> Option<Vec<u32>> {
    let total_neurons = per_neuron_bits.len();
    if total_neurons < 100 {
        return None;
    }
    // Guard against massive allocations (e.g. 251K neurons × 16K examples = 4B addresses = 16GB)
    if total_neurons.saturating_mul(num_train) > MAX_GPU_ADDRESSES {
        return None;
    }

    let trainer_mutex = crate::get_cached_metal_trainer().ok()?;
    let mut guard = trainer_mutex.lock().ok()?;
    let trainer = guard.as_mut()?;

    let neuron_meta: Vec<NeuronTrainMeta> = (0..total_neurons)
        .map(|n| NeuronTrainMeta {
            bits: per_neuron_bits[n] as u32,
            conn_offset: neuron_conn_offsets[n] as u32,
        })
        .collect();

    trainer.compute_addresses(
        packed_input,
        connections,
        &neuron_meta,
        num_train,
        words_per_example,
    ).ok()
}

/// Read a memory cell value
#[inline]
fn read_cell(memory_words: &[i64], neuron_idx: usize, address: usize, words_per_neuron: usize) -> i64 {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = neuron_idx * words_per_neuron + word_idx;
    let word = memory_words[word_offset];
    (word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
}

/// Write a memory cell value (atomic, for parallel writes)
#[inline]
fn write_cell_atomic(
    memory_words: &[AtomicI64],
    neuron_idx: usize,
    address: usize,
    value: i64,
    words_per_neuron: usize,
    allow_override: bool,
) -> bool {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = neuron_idx * words_per_neuron + word_idx;
    let shift = cell_idx * BITS_PER_CELL;
    let mask = CELL_MASK << shift;
    let new_bits = value << shift;

    loop {
        let old_word = memory_words[word_offset].load(Ordering::Acquire);
        let old_cell = (old_word >> shift) & CELL_MASK;

        if !allow_override && old_cell != EMPTY {
            return false;
        }
        if old_cell == value {
            return false;
        }

        let new_word = (old_word & !mask) | new_bits;
        match memory_words[word_offset].compare_exchange(
            old_word, new_word, Ordering::AcqRel, Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(_) => continue,
        }
    }
}

/// Build config groups from per-cluster configuration
///
/// Groups clusters by their (neurons, bits) to enable efficient batch processing.
pub fn build_config_groups(
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
) -> Vec<ConfigGroup> {
    use std::collections::HashMap;

    let num_clusters = bits_per_cluster.len();
    let mut config_to_clusters: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for cluster_id in 0..num_clusters {
        let key = (neurons_per_cluster[cluster_id], bits_per_cluster[cluster_id]);
        config_to_clusters.entry(key).or_default().push(cluster_id);
    }

    let mut groups: Vec<ConfigGroup> = config_to_clusters
        .into_iter()
        .map(|((neurons, bits), cluster_ids)| ConfigGroup::new(neurons, bits, cluster_ids))
        .collect();

    // Sort by (neurons, bits) for deterministic ordering
    groups.sort_by_key(|g| (g.neurons, g.bits));

    // Compute offsets
    let mut memory_offset = 0;
    let mut conn_offset = 0;
    for group in &mut groups {
        group.memory_offset = memory_offset;
        group.conn_offset = conn_offset;
        memory_offset += group.memory_size();
        conn_offset += group.conn_size();
    }

    // Log group diversity if enabled (helps diagnose slowdown from too many groups)
    if std::env::var("WNN_GROUP_LOG").is_ok() {
        let sparse_count = groups.iter().filter(|g| g.bits > 12).count();
        let dense_count = groups.len() - sparse_count;
        eprintln!(
            "[CONFIG_GROUPS] total={} sparse={} dense={} configs={:?}",
            groups.len(),
            sparse_count,
            dense_count,
            groups.iter().map(|g| (g.neurons, g.bits, g.cluster_ids.len())).collect::<Vec<_>>()
        );
    }

    groups
}

/// Bucket neurons into ranges to reduce group diversity
/// Returns the max neurons for the bucket
fn bucket_neurons(neurons: usize) -> usize {
    // Buckets: 1-5→5, 6-10→10, 11-15→15, 16-20→20, 21-25→25, etc.
    // This gives ~5x fewer unique neuron values
    ((neurons + 4) / 5) * 5
}

/// Build config groups with coalescing - buckets similar neuron counts together
/// This reduces the number of GPU dispatches while preserving accuracy through masking.
///
/// Example: If clusters have neurons [5, 6, 7, 8], they bucket into:
///   - 5→5 (bucket 5), 6-10→10 (bucket for 6,7,8)
///   - Instead of 4 groups, we have 2 groups
///
/// For each coalesced group:
///   - neurons = max in bucket (for memory allocation)
///   - actual_neurons[i] = true neuron count for cluster i (for scoring)
pub fn build_config_groups_coalesced(
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
) -> Vec<ConfigGroup> {
    use std::collections::HashMap;

    let num_clusters = bits_per_cluster.len();

    // Key: (bucket_max, bits) -> list of (cluster_id, actual_neurons)
    let mut bucket_to_clusters: HashMap<(usize, usize), Vec<(usize, u32)>> = HashMap::new();

    for cluster_id in 0..num_clusters {
        let actual = neurons_per_cluster[cluster_id];
        let bucket_max = bucket_neurons(actual);
        let bits = bits_per_cluster[cluster_id];
        let key = (bucket_max, bits);
        bucket_to_clusters.entry(key).or_default().push((cluster_id, actual as u32));
    }

    let mut groups: Vec<ConfigGroup> = bucket_to_clusters
        .into_iter()
        .map(|((max_neurons, bits), entries)| {
            let cluster_ids: Vec<usize> = entries.iter().map(|(id, _)| *id).collect();
            let actual_neurons: Vec<u32> = entries.iter().map(|(_, n)| *n).collect();

            // Check if all actual neurons are the same as max (can use uniform mode)
            let all_same = actual_neurons.iter().all(|&n| n as usize == max_neurons);
            if all_same {
                ConfigGroup::new(max_neurons, bits, cluster_ids)
            } else {
                ConfigGroup::new_coalesced(max_neurons, bits, cluster_ids, actual_neurons)
            }
        })
        .collect();

    // Sort by (neurons, bits) for deterministic ordering
    groups.sort_by_key(|g| (g.neurons, g.bits));

    // Compute offsets
    let mut memory_offset = 0;
    let mut conn_offset = 0;
    for group in &mut groups {
        group.memory_offset = memory_offset;
        group.conn_offset = conn_offset;
        memory_offset += group.memory_size();
        conn_offset += group.conn_size();
    }

    // Log group diversity if enabled
    if std::env::var("WNN_GROUP_LOG").is_ok() {
        let sparse_count = groups.iter().filter(|g| g.bits > 12).count();
        let dense_count = groups.len() - sparse_count;
        let coalesced_count = groups.iter().filter(|g| g.is_coalesced()).count();
        eprintln!(
            "[CONFIG_GROUPS_COALESCED] total={} sparse={} dense={} coalesced={} configs={:?}",
            groups.len(),
            sparse_count,
            dense_count,
            coalesced_count,
            groups.iter().map(|g| (g.neurons, g.bits, g.cluster_ids.len(), g.is_coalesced())).collect::<Vec<_>>()
        );
    }

    groups
}

/// Forward pass for adaptive architecture
///
/// Processes each config group efficiently, then scatters results to output.
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits]
///   connections_flat: All groups' connections concatenated
///   memory_words: All groups' memory concatenated
///   groups: Config groups with cluster assignments
///   num_examples: Number of input examples
///   total_input_bits: Total input bits per example
///   num_clusters: Total number of clusters (vocabulary size)
///
/// Returns: [num_examples * num_clusters] probabilities
pub fn forward_batch_adaptive(
    input_bits_flat: &[bool],
    connections_flat: &[i64],
    memory_words: &[i64],
    groups: &[ConfigGroup],
    num_examples: usize,
    total_input_bits: usize,
    num_clusters: usize,
) -> Vec<f32> {
    let empty_value = get_empty_value();
    let mut probs = vec![0.0f32; num_examples * num_clusters];

    // Build reverse mapping: global_cluster_id -> (group_idx, local_cluster_idx)
    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    // Process all examples in parallel
    probs.par_chunks_mut(num_clusters).enumerate().for_each(|(ex_idx, ex_probs)| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        // Process each config group
        for group in groups {
            let neurons = group.neurons;
            let bits = group.bits;
            let words_per_neuron = group.words_per_neuron;
            let group_memory = &memory_words[group.memory_offset..];
            let group_conns = &connections_flat[group.conn_offset..];

            // For each cluster in this group
            for (local_idx, &global_cluster_id) in group.cluster_ids.iter().enumerate() {
                // Use actual neurons if coalesced, otherwise MAX (uniform case)
                let actual_neurons = if let Some(ref an) = group.actual_neurons {
                    an[local_idx] as usize
                } else {
                    neurons
                };

                let start_neuron = local_idx * neurons;  // Use MAX for memory layout
                let mut count_true = 0u32;
                let mut count_empty = 0u32;

                for neuron_offset in 0..actual_neurons {  // Only iterate actual neurons
                    let local_neuron = start_neuron + neuron_offset;
                    let conn_start = local_neuron * bits;
                    let connections = &group_conns[conn_start..conn_start + bits];

                    let address = compute_address(input_bits, connections, bits);
                    let cell_value = read_cell(group_memory, local_neuron, address, words_per_neuron);

                    if cell_value == TRUE {
                        count_true += 1;
                    } else if cell_value == EMPTY {
                        count_empty += 1;
                    }
                }

                // Divide by actual neurons for correct probability
                ex_probs[global_cluster_id] =
                    (count_true as f32 + empty_value * count_empty as f32) / actual_neurons as f32;
            }
        }
    });

    probs
}

/// Training for adaptive architecture
///
/// Two-phase training: TRUE first, then FALSE (to ensure TRUE priority).
///
/// Args:
///   input_bits_flat: [num_examples * total_input_bits]
///   true_clusters: [num_examples] global cluster indices
///   false_clusters_flat: [num_examples * num_negatives] global cluster indices
///   connections_flat: All groups' connections concatenated
///   memory_words: All groups' memory concatenated (mutable)
///   groups: Config groups with cluster assignments
///
/// Returns: Number of cells modified
pub fn train_batch_adaptive(
    input_bits_flat: &[bool],
    true_clusters: &[i64],
    false_clusters_flat: &[i64],
    connections_flat: &[i64],
    memory_words: &mut [i64],
    groups: &[ConfigGroup],
    num_examples: usize,
    total_input_bits: usize,
    num_negatives: usize,
    num_clusters: usize,
    allow_override: bool,
) -> usize {
    // Build reverse mapping: global_cluster_id -> (group_idx, local_cluster_idx)
    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    // Convert memory to atomic for thread-safe writes
    let atomic_memory: &[AtomicI64] = unsafe {
        std::slice::from_raw_parts(
            memory_words.as_ptr() as *const AtomicI64,
            memory_words.len(),
        )
    };

    // Phase 1: Write all TRUEs
    let true_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];

        let true_cluster = true_clusters[ex_idx] as usize;
        let (group_idx, local_cluster) = cluster_to_group[true_cluster];
        let group = &groups[group_idx];

        let neurons = group.neurons;  // MAX for memory layout
        let bits = group.bits;
        let words_per_neuron = group.words_per_neuron;
        let start_neuron = local_cluster * neurons;
        let group_conns = &connections_flat[group.conn_offset..];

        // Use actual neurons if coalesced, otherwise MAX
        let actual_neurons = if let Some(ref an) = group.actual_neurons {
            an[local_cluster] as usize
        } else {
            neurons
        };

        let mut modified = 0usize;
        for neuron_offset in 0..actual_neurons {  // Only iterate actual neurons
            let local_neuron = start_neuron + neuron_offset;
            let conn_start = local_neuron * bits;
            let connections = &group_conns[conn_start..conn_start + bits];

            let address = compute_address(input_bits, connections, bits);
            let _global_neuron_offset = group.memory_offset / words_per_neuron + local_neuron;

            if write_cell_atomic(
                &atomic_memory[group.memory_offset..],
                local_neuron, address, TRUE, words_per_neuron, allow_override,
            ) {
                modified += 1;
            }
        }
        modified
    }).sum();

    // Phase 2: Write all FALSEs (skip if already TRUE)
    let false_modified: usize = (0..num_examples).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &input_bits_flat[input_start..input_start + total_input_bits];
        let true_cluster = true_clusters[ex_idx] as usize;

        let false_start = ex_idx * num_negatives;
        let mut modified = 0usize;

        for neg_idx in 0..num_negatives {
            let false_cluster = false_clusters_flat[false_start + neg_idx] as usize;
            if false_cluster == true_cluster {
                continue;
            }

            let (group_idx, local_cluster) = cluster_to_group[false_cluster];
            let group = &groups[group_idx];

            let neurons = group.neurons;  // MAX for memory layout
            let bits = group.bits;
            let words_per_neuron = group.words_per_neuron;
            let start_neuron = local_cluster * neurons;
            let group_conns = &connections_flat[group.conn_offset..];

            // Use actual neurons if coalesced, otherwise MAX
            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                neurons
            };

            for neuron_offset in 0..actual_neurons {  // Only iterate actual neurons
                let local_neuron = start_neuron + neuron_offset;
                let conn_start = local_neuron * bits;
                let connections = &group_conns[conn_start..conn_start + bits];

                let address = compute_address(input_bits, connections, bits);

                if write_cell_atomic(
                    &atomic_memory[group.memory_offset..],
                    local_neuron, address, FALSE, words_per_neuron, false, // Never override TRUE
                ) {
                    modified += 1;
                }
            }
        }
        modified
    }).sum();

    true_modified + false_modified
}

/// Dense memory for a config group (bit-packed, fast for bits <= 12)
/// Uses atomic operations for thread-safe concurrent writes.
pub(crate) struct GroupDenseMemory {
    /// Bit-packed memory words [total_neurons * words_per_neuron]
    words: Vec<AtomicI64>,
    words_per_neuron: usize,
}

impl GroupDenseMemory {
    fn new(num_neurons: usize, bits: usize, memory_mode: u8) -> Self {
        let words_per_neuron = (1usize << bits).div_ceil(CELLS_PER_WORD);
        let total_words = num_neurons * words_per_neuron;
        let empty_word = crate::neuron_memory::empty_word_for_mode(memory_mode);
        Self {
            words: (0..total_words).map(|_| AtomicI64::new(empty_word)).collect(),
            words_per_neuron,
        }
    }

    /// Export memory words for Metal GPU (read-only snapshot)
    fn export_for_metal(&self) -> Vec<i64> {
        self.words.iter().map(|w| w.load(Ordering::Relaxed)).collect()
    }

    #[inline]
    fn read(&self, neuron_idx: usize, address: usize) -> i64 {
        let word_idx = address / CELLS_PER_WORD;
        let cell_idx = address % CELLS_PER_WORD;
        let word_offset = neuron_idx * self.words_per_neuron + word_idx;
        let word = self.words[word_offset].load(Ordering::Relaxed);
        (word >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
    }

    /// Thread-safe atomic write using compare-and-swap
    ///
    /// TRUE-wins-over-FALSE semantics:
    /// - TRUE can be written over EMPTY or FALSE
    /// - FALSE can only be written over EMPTY
    /// - TRUE cannot be overwritten by FALSE
    #[inline]
    fn write(&self, neuron_idx: usize, address: usize, value: i64, allow_override: bool) -> bool {
        let word_idx = address / CELLS_PER_WORD;
        let cell_idx = address % CELLS_PER_WORD;
        let word_offset = neuron_idx * self.words_per_neuron + word_idx;
        let shift = cell_idx * BITS_PER_CELL;
        let mask = CELL_MASK << shift;

        loop {
            let old_word = self.words[word_offset].load(Ordering::Relaxed);
            let old_cell = (old_word >> shift) & CELL_MASK;

            // No change needed if same value
            if old_cell == value {
                return false;
            }

            // TRUE wins over FALSE: don't overwrite TRUE with FALSE
            if old_cell == TRUE && value == FALSE {
                return false;
            }

            // If not allow_override:
            // - TRUE can overwrite EMPTY or FALSE (TRUE wins)
            // - FALSE can only overwrite EMPTY
            if !allow_override && value == FALSE && old_cell != EMPTY {
                return false;
            }

            let new_word = (old_word & !mask) | (value << shift);
            if self.words[word_offset]
                .compare_exchange_weak(old_word, new_word, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
            // CAS failed, retry
        }
    }

    /// Thread-safe atomic nudge for quad modes (CAS loop).
    /// Moves cell one step toward target: +1 if target_true, -1 if target_false.
    /// Clamps to [0, 3] (QUAD_FALSE..QUAD_TRUE).
    #[inline]
    fn nudge(&self, neuron_idx: usize, address: usize, target_true: bool) -> bool {
        let word_idx = address / CELLS_PER_WORD;
        let cell_idx = address % CELLS_PER_WORD;
        let word_offset = neuron_idx * self.words_per_neuron + word_idx;
        let shift = cell_idx * BITS_PER_CELL;
        let mask = CELL_MASK << shift;
        let delta = 2 * (target_true as i64) - 1; // +1 or -1

        loop {
            let old_word = self.words[word_offset].load(Ordering::Relaxed);
            let old_cell = (old_word >> shift) & CELL_MASK;

            let new_cell = (old_cell + delta).clamp(crate::neuron_memory::QUAD_FALSE, crate::neuron_memory::QUAD_TRUE);
            if new_cell == old_cell {
                return false; // already at boundary
            }

            let new_word = (old_word & !mask) | (new_cell << shift);
            if self.words[word_offset]
                .compare_exchange_weak(old_word, new_word, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
            // CAS failed, retry
        }
    }
}

/// GPU-compatible sparse memory export (sorted arrays for binary search)
#[derive(Clone)]
pub struct SparseGpuExport {
    /// Sorted keys for all neurons, concatenated
    pub keys: Vec<u64>,
    /// Values corresponding to keys (0=FALSE, 1=TRUE)
    pub values: Vec<u8>,
    /// Start offset for each neuron in keys array
    pub offsets: Vec<u32>,
    /// Number of entries for each neuron
    pub counts: Vec<u32>,
    /// Total number of neurons
    pub num_neurons: usize,
}

impl SparseGpuExport {
    /// CPU binary search lookup
    #[inline]
    pub fn lookup(&self, neuron_idx: usize, address: u64) -> u8 {
        let start = self.offsets[neuron_idx] as usize;
        let count = self.counts[neuron_idx] as usize;

        if count == 0 {
            return EMPTY as u8;
        }

        let end = start + count;
        let keys_slice = &self.keys[start..end];

        match keys_slice.binary_search(&address) {
            Ok(idx) => self.values[start + idx],
            Err(_) => EMPTY as u8,
        }
    }

    /// Total memory size in bytes
    pub fn memory_size(&self) -> usize {
        self.keys.len() * 8 + self.values.len() + self.offsets.len() * 4 + self.counts.len() * 4
    }
}

/// Sparse memory for a config group (concurrent hash-based, for bits > 12)
/// Uses DashMap for thread-safe concurrent access during parallel training.
pub(crate) struct GroupSparseMemory {
    /// Per-neuron concurrent hash maps: address -> cell value
    neurons: Vec<DashMap<u64, u8>>,
    /// Default cell value for unvisited addresses (EMPTY_U8=2 for ternary, 1=QUAD_WEAK_FALSE for quad)
    default_empty: u8,
}

impl GroupSparseMemory {
    fn new(num_neurons: usize, memory_mode: u8) -> Self {
        let default_empty = match memory_mode {
            crate::neuron_memory::MODE_QUAD_BINARY | crate::neuron_memory::MODE_QUAD_WEIGHTED => 1, // QUAD_WEAK_FALSE
            _ => EMPTY as u8, // 2
        };
        Self {
            neurons: (0..num_neurons).map(|_| DashMap::new()).collect(),
            default_empty,
        }
    }

    /// Export to GPU-compatible sorted array format for binary search evaluation
    fn export_for_gpu(&self) -> SparseGpuExport {
        let mut keys: Vec<u64> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(self.neurons.len());
        let mut counts: Vec<u32> = Vec::with_capacity(self.neurons.len());

        for neuron_map in &self.neurons {
            let offset = keys.len() as u32;
            offsets.push(offset);

            // Collect and sort entries for this neuron
            let mut entries: Vec<(u64, u8)> = neuron_map.iter()
                .map(|entry| (*entry.key(), *entry.value()))
                .collect();
            entries.sort_by_key(|(k, _)| *k);

            counts.push(entries.len() as u32);

            for (key, value) in entries {
                keys.push(key);
                values.push(value);
            }
        }

        SparseGpuExport {
            keys,
            values,
            offsets,
            counts,
            num_neurons: self.neurons.len(),
        }
    }

    #[inline]
    fn read(&self, neuron_idx: usize, address: u64) -> u8 {
        *self.neurons[neuron_idx].get(&address).map(|v| *v).as_ref().unwrap_or(&self.default_empty)
    }

    /// Thread-safe write using DashMap
    ///
    /// TRUE-wins-over-FALSE semantics (values: 0=FALSE, 1=TRUE, 2=EMPTY):
    /// - TRUE can be written over EMPTY or FALSE
    /// - FALSE can only be written over EMPTY
    /// - TRUE cannot be overwritten by FALSE
    #[inline]
    fn write(&self, neuron_idx: usize, address: u64, value: u8, allow_override: bool) -> bool {
        let map = &self.neurons[neuron_idx];
        match map.entry(address) {
            dashmap::mapref::entry::Entry::Occupied(mut e) => {
                let current = *e.get();

                // No change needed if same value
                if current == value {
                    return false;
                }

                // TRUE wins over FALSE: don't overwrite TRUE with FALSE
                if current == 1 && value == 0 {
                    return false;
                }

                // If not allow_override:
                // - TRUE (1) can overwrite EMPTY (2) or FALSE (0) (TRUE wins)
                // - FALSE (0) can only overwrite EMPTY (2)
                if !allow_override && value == 0 && current != 2 {
                    return false;
                }

                // Allow TRUE to overwrite FALSE (TRUE wins) or write to EMPTY
                if allow_override || current == 2 || (value == 1 && current == 0) {
                    e.insert(value);
                    return true;
                }
                false
            }
            dashmap::mapref::entry::Entry::Vacant(e) => {
                e.insert(value);
                true
            }
        }
    }

    /// Thread-safe nudge for quad modes using DashMap entry API.
    /// Moves cell one step toward target. For vacant entries, inserts one step
    /// from default (QUAD_WEAK_TRUE=2 if target_true, QUAD_WEAK_FALSE=1 stays if target_false).
    #[inline]
    fn nudge(&self, neuron_idx: usize, address: u64, target_true: bool) -> bool {
        let map = &self.neurons[neuron_idx];
        match map.entry(address) {
            dashmap::mapref::entry::Entry::Occupied(mut e) => {
                let old_cell = *e.get() as i64;
                let delta = 2 * (target_true as i64) - 1;
                let new_cell = (old_cell + delta).clamp(crate::neuron_memory::QUAD_FALSE, crate::neuron_memory::QUAD_TRUE) as u8;
                if new_cell == old_cell as u8 {
                    return false;
                }
                // Remove entry if it matches default_empty (saves memory)
                if new_cell == self.default_empty {
                    e.remove();
                } else {
                    e.insert(new_cell);
                }
                true
            }
            dashmap::mapref::entry::Entry::Vacant(e) => {
                // Default is QUAD_WEAK_FALSE (1). Nudge toward true → insert 2, toward false → insert 0
                let default = self.default_empty as i64;
                let delta = 2 * (target_true as i64) - 1;
                let new_cell = (default + delta).clamp(crate::neuron_memory::QUAD_FALSE, crate::neuron_memory::QUAD_TRUE) as u8;
                if new_cell == self.default_empty {
                    return false; // no change from default
                }
                e.insert(new_cell);
                true
            }
        }
    }
}

/// Hybrid memory - Dense for low bits, Sparse for high bits
/// Both variants support thread-safe concurrent access for parallel training.
pub(crate) enum GroupMemory {
    Dense(GroupDenseMemory),
    Sparse(GroupSparseMemory),
}

impl GroupMemory {
    pub(crate) fn new(num_neurons: usize, bits: usize, memory_mode: u8) -> Self {
        if bits <= SPARSE_THRESHOLD {
            GroupMemory::Dense(GroupDenseMemory::new(num_neurons, bits, memory_mode))
        } else {
            GroupMemory::Sparse(GroupSparseMemory::new(num_neurons, memory_mode))
        }
    }

    /// Check if this is dense memory (can be accelerated with Metal)
    pub(crate) fn is_dense(&self) -> bool {
        matches!(self, GroupMemory::Dense(_))
    }

    /// Export for Metal GPU (only works for Dense, returns None for Sparse)
    pub(crate) fn export_for_metal(&self) -> Option<Vec<i64>> {
        match self {
            GroupMemory::Dense(m) => Some(m.export_for_metal()),
            GroupMemory::Sparse(_) => None,
        }
    }

    /// Export sparse memory for GPU binary search (returns None for Dense)
    pub(crate) fn export_for_gpu_sparse(&self) -> Option<SparseGpuExport> {
        match self {
            GroupMemory::Dense(_) => None,
            GroupMemory::Sparse(m) => Some(m.export_for_gpu()),
        }
    }

    /// Check if this is sparse memory
    pub(crate) fn is_sparse(&self) -> bool {
        matches!(self, GroupMemory::Sparse(_))
    }

    /// Compute fill rate for a single neuron within this group's memory.
    ///
    /// `neuron_idx` is the neuron's position in the group (NOT global).
    /// `bits` is the number of address bits for this neuron.
    pub(crate) fn neuron_fill_rate(&self, neuron_idx: usize, bits: usize) -> f32 {
        let total_cells = 1usize << bits;
        match self {
            GroupMemory::Dense(m) => {
                let wpn = m.words_per_neuron;
                let start = neuron_idx * wpn;
                let mut filled = 0u32;
                for w in 0..wpn {
                    if start + w >= m.words.len() { break; }
                    let word = m.words[start + w].load(Ordering::Relaxed);
                    for c in 0..CELLS_PER_WORD {
                        let cell = (word >> (c * BITS_PER_CELL)) & CELL_MASK;
                        if cell != EMPTY { filled += 1; }
                    }
                }
                filled.min(total_cells as u32) as f32 / total_cells.max(1) as f32
            }
            GroupMemory::Sparse(m) => {
                if neuron_idx >= m.neurons.len() { 0.0 }
                else {
                    m.neurons[neuron_idx].len().min(total_cells) as f32 / total_cells.max(1) as f32
                }
            }
        }
    }

    /// Return (total_capacity, filled_count) for diagnostics
    pub(crate) fn fill_stats(&self) -> (usize, usize) {
        match self {
            GroupMemory::Dense(m) => {
                let total = m.words.len() * CELLS_PER_WORD;
                let empty_cell = EMPTY;
                let mut filled = 0usize;
                for w in &m.words {
                    let word = w.load(std::sync::atomic::Ordering::Relaxed);
                    for i in 0..CELLS_PER_WORD {
                        let cell = (word >> (i * 2)) & 0x3;
                        if cell != empty_cell {
                            filled += 1;
                        }
                    }
                }
                (total, filled)
            }
            GroupMemory::Sparse(m) => {
                let filled: usize = m.neurons.iter().map(|dm| dm.len()).sum();
                // Sparse has no fixed capacity, report filled as both
                (filled, filled)
            }
        }
    }

    #[inline]
    pub(crate) fn read(&self, neuron_idx: usize, address: usize) -> i64 {
        match self {
            GroupMemory::Dense(m) => m.read(neuron_idx, address),
            GroupMemory::Sparse(m) => m.read(neuron_idx, address as u64) as i64,
        }
    }

    /// Thread-safe write (both variants support concurrent access)
    #[inline]
    pub(crate) fn write(&self, neuron_idx: usize, address: usize, value: i64, allow_override: bool) -> bool {
        match self {
            GroupMemory::Dense(m) => m.write(neuron_idx, address, value, allow_override),
            GroupMemory::Sparse(m) => m.write(neuron_idx, address as u64, value as u8, allow_override),
        }
    }

    /// Thread-safe nudge for quad modes — moves cell one step toward target.
    #[inline]
    pub(crate) fn nudge(&self, neuron_idx: usize, address: usize, target_true: bool) -> bool {
        match self {
            GroupMemory::Dense(m) => m.nudge(neuron_idx, address, target_true),
            GroupMemory::Sparse(m) => m.nudge(neuron_idx, address as u64, target_true),
        }
    }
}

/// Evaluate a dense config group using Metal GPU.
///
/// Returns scores for [num_examples × num_clusters_in_group] as f32.
/// The scores are in group-local cluster order (need scattering to global order).
pub(crate) fn evaluate_group_metal(
    metal: &crate::metal_ramlm::MetalRAMLMEvaluator,
    packed_eval: &[u64],
    connections_flat: &[i64],
    memory_words: &[i64],
    group: &ConfigGroup,
    num_eval: usize,
    words_per_example: usize,
    memory_mode: u8,
) -> Result<Vec<f32>, String> {
    let num_clusters = group.cluster_count();
    let num_neurons = group.total_neurons();

    // Extract connections for this group (they're stored contiguously at conn_offset)
    let conn_size = group.conn_size();
    let group_connections = &connections_flat[group.conn_offset..group.conn_offset + conn_size];

    metal.forward_batch(
        packed_eval,
        group_connections,
        memory_words,
        num_eval,
        words_per_example,
        num_neurons,
        group.bits,
        group.neurons,
        num_clusters,
        group.words_per_neuron,
        memory_mode,
    )
}

/// Evaluate a sparse config group using Metal GPU with binary search.
///
/// Returns scores for [num_examples × num_clusters_in_group] as f32.
/// The scores are in group-local cluster order (need scattering to global order).
pub(crate) fn evaluate_group_sparse_gpu(
    sparse_evaluator: &crate::metal_ramlm::MetalSparseEvaluator,
    packed_eval: &[u64],
    connections_flat: &[i64],
    export: &SparseGpuExport,
    group: &ConfigGroup,
    num_eval: usize,
    words_per_example: usize,
    memory_mode: u8,
) -> Result<Vec<f32>, String> {
    let num_clusters = group.cluster_count();

    // Extract connections for this group
    let conn_size = group.conn_size();
    let group_connections = &connections_flat[group.conn_offset..group.conn_offset + conn_size];

    sparse_evaluator.forward_batch_sparse(
        packed_eval,
        group_connections,
        &export.keys,
        &export.values,
        &export.offsets,
        &export.counts,
        num_eval,
        words_per_example,
        export.num_neurons,
        group.bits,
        group.neurons,
        num_clusters,
        memory_mode,
    )
}

/// Evaluate multiple genomes SEQUENTIALLY with METAL GPU ACCELERATION.
///
/// This is the KEY acceleration function for GA optimization.
/// Each genome is evaluated independently with its own memory.
///
/// Hybrid acceleration strategy:
/// - Training: CPU with rayon parallelism (random writes)
/// - Evaluation: Metal GPU for dense groups (40 cores on M4 Max)
///               CPU for sparse groups (hash lookups not GPU-friendly)
///
/// Performance: Metal accelerates evaluation by ~10-20x for dense groups,
/// which typically contain 80%+ of clusters (those with bits <= 12).
///
/// Args:
///   genomes_bits_flat: [num_genomes * num_clusters] bits per cluster for each genome
///   genomes_neurons_flat: [num_genomes * num_clusters] neurons per cluster for each genome
///   genomes_connections_flat: [num_genomes * total_connections] flattened connection indices, or empty for random
///   num_genomes: Number of genomes to evaluate
///   num_clusters: Number of clusters (vocab size)
///   train_input_bits: [num_train * total_input_bits] training contexts
///   train_targets: [num_train] target cluster for each training example
///   train_negatives: [num_train * num_negatives] negative clusters
///   num_train: Number of training examples
///   num_negatives: Number of negative samples per example
///   eval_input_bits: [num_eval * total_input_bits] evaluation contexts
///   eval_targets: [num_eval] target cluster for each eval example
///   num_eval: Number of evaluation examples
///   total_input_bits: Input bits per example
///   empty_value: Value for EMPTY cells (0.0 recommended)
///
/// Returns: [num_genomes] cross-entropy values (lower is better)
/// Evaluate multiple genomes SEQUENTIALLY, returning (CE, accuracy) for each.
///
/// Genomes are evaluated one at a time to:
/// 1. Prevent memory explosion (only 1 genome's memory allocated)
/// 2. Allow full CPU utilization for each genome's training/eval (16 cores)
/// 3. Avoid thread pool contention from nested parallelism
///
/// Each genome's training (200K examples) and evaluation (50K examples)
/// use full rayon parallelism internally.
///
/// IMPORTANT: Connections must be provided for proper evolutionary search.
/// If genomes_connections_flat is empty, random connections are generated.
///
/// Returns Vec of (cross_entropy, accuracy) tuples - one per genome.
pub fn evaluate_genomes_parallel(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64, f64, f64)> {
    let memory_mode = crate::neuron_memory::get_memory_mode();
    use rand::prelude::*;
    use rand::SeedableRng;

    // Check if connections are provided
    let use_provided_connections = !genomes_connections_flat.is_empty();

    // Pre-compute genome_bpn_offsets: genomes_bits_flat has total_neurons entries per genome
    // (per-neuron bits), NOT num_clusters entries.
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    debug_assert_eq!(
        genomes_bits_flat.len(),
        *genome_bpn_offsets.last().unwrap(),
        "genomes_bits_flat length ({}) != expected total neurons ({})",
        genomes_bits_flat.len(),
        genome_bpn_offsets.last().unwrap(),
    );

    // Pre-compute per-genome connection offsets: conn_size = sum of per-neuron bits
    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    // Pack input bits to u64 once (shared across all genomes for GPU address computation)
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_bools_to_u64(train_input_bits, num_train, total_input_bits);

    // Check if progress logging is enabled via env var
    let progress_log = std::env::var("WNN_PROGRESS_LOG").map(|v| v == "1").unwrap_or(false);
    let log_path = std::env::var("WNN_LOG_PATH").ok();
    // Get generation info from env vars (set by Python before calling)
    let current_gen: usize = std::env::var("WNN_PROGRESS_GEN")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let total_gens: usize = std::env::var("WNN_PROGRESS_TOTAL_GENS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    // Log type: Init, New, Nbr, CE, Acc (default Init)
    let log_type = std::env::var("WNN_PROGRESS_TYPE").unwrap_or_else(|_| "Init".to_string());
    // Offset for batch position (e.g., batch starting at genome 11 in a 50-genome set)
    let batch_offset: usize = std::env::var("WNN_PROGRESS_OFFSET")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    // Total count (for showing X/50 instead of X/batch_size)
    let total_count: usize = std::env::var("WNN_PROGRESS_TOTAL")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(num_genomes);
    let _start_time = std::time::Instant::now();

    // SEQUENTIAL genome evaluation - each genome gets full thread pool for token parallelism
    // Parallel genome eval causes contention: 10 genomes × nested token parallelism = thrashing
    // Sequential is faster: ~6s/genome vs ~10s/genome with parallel outer loop
    let results: Vec<(f64, f64, f64, f64)> = (0..num_genomes).map(|genome_idx| {
        let genome_start = std::time::Instant::now();
        // Extract this genome's per-neuron bits and per-cluster neurons
        let genome_offset = genome_idx * num_clusters;
        let neurons_per_cluster: Vec<usize> = genomes_neurons_flat[genome_offset..genome_offset + num_clusters].to_vec();

        // Extract per-neuron bits for this genome
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let per_neuron_bits: Vec<usize> = genomes_bits_flat[bpn_start..bpn_end].to_vec();

        // Compute per-cluster max bits (for build_groups and GPU dispatch)
        let bits_per_cluster = per_cluster_max_bits(&per_neuron_bits, &neurons_per_cluster);

        // Build neuron metadata for per-neuron training and CPU eval
        let (cluster_neuron_starts, neuron_conn_offsets) =
            build_neuron_metadata(&per_neuron_bits, &neurons_per_cluster);

        // Build config groups for this genome (using per-cluster max bits)
        let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);

        // Create hybrid memory for each config group
        let group_memories: Vec<GroupMemory> = groups.iter()
            .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
            .collect();

        // Get original per-neuron connections for this genome
        let original_connections: Vec<i64> = if use_provided_connections {
            let conn_offset = conn_offsets[genome_idx];
            let conn_size = conn_sizes[genome_idx];
            genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec()
        } else {
            // Generate random per-neuron connections (legacy fallback)
            let total_conn: usize = per_neuron_bits.iter().sum();
            let mut rng = rand::rngs::SmallRng::from_entropy();
            let mut conns: Vec<i64> = Vec::with_capacity(total_conn);
            for _ in 0..total_conn {
                conns.push(rng.gen_range(0..total_input_bits as i64));
            }
            conns
        };

        // Build GPU-padded connections (per-neuron → group layout with padding)
        let gpu_connections = reorganize_connections_for_gpu(
            &original_connections,
            &per_neuron_bits,
            &neurons_per_cluster,
            &groups,
        );

        // Build cluster-to-group mapping
        let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
        for (group_idx, group) in groups.iter().enumerate() {
            for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                cluster_to_group[cluster_id] = (group_idx, local_idx);
            }
        }

        // Compute training addresses on GPU (falls back to CPU if unavailable)
        let gpu_addresses = try_gpu_addresses_adaptive(
            &packed_train_input,
            words_per_example,
            &per_neuron_bits,
            &neuron_conn_offsets,
            &original_connections,
            num_train,
        );

        // Train this genome using per-neuron bits (PARALLEL across examples)
        train_genome_in_slot(
            &group_memories,
            &groups,
            &original_connections,
            &per_neuron_bits,
            &cluster_neuron_starts,
            &neuron_conn_offsets,
            &cluster_to_group,
            train_input_bits,
            train_targets,
            train_negatives,
            num_train,
            num_negatives,
            total_input_bits,
            gpu_addresses.as_deref(),
            neuron_sample_rate,
            rng_seed.wrapping_add(genome_idx as u64),
            memory_mode,
            None, // class_weights: only used by IDS via evaluate_genomes_parallel_hybrid
            true, // parallel: safe here, not inside outer par_iter
        );

        // Evaluate this genome - HYBRID Metal/CPU acceleration
        // - Dense groups (bits <= 12): Metal GPU (all examples at once)
        // - Sparse groups (bits > 12): CPU (hash lookups not GPU-friendly)
        let epsilon = 1e-10f64;

        // Pre-compute scores for all examples × clusters
        // Shape: [num_eval][num_clusters]
        let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];

        // Get Metal evaluators (lazy init, thread-safe)
        // These are Arc<T> so we can clone and hold references across the loop
        let metal = get_metal_evaluator();
        let sparse_metal = get_sparse_metal_evaluator();

        // Pack eval input bits to u64 for GPU (pack once, reuse for all groups)
        let (packed_eval, words_per_example) = crate::neuron_memory::pack_bools_to_u64(
            eval_input_bits, num_eval, total_input_bits
        );

        // Process each group - Metal for dense, GPU sparse for sparse, CPU fallback
        for (group_idx, group) in groups.iter().enumerate() {
            let memory = &group_memories[group_idx];

            if let (Some(ref metal_eval), true) = (&metal, memory.is_dense()) {
                // Metal path: evaluate all examples at once for this dense group
                // GPU uses padded connections (group layout with max_bits per neuron)
                if let Some(memory_words) = memory.export_for_metal() {
                    match evaluate_group_metal(
                        metal_eval.as_ref(),
                        &packed_eval,
                        &gpu_connections,
                        &memory_words,
                        group,
                        num_eval,
                        words_per_example,
                        memory_mode,
                    ) {
                        Ok(group_scores) => {
                            for ex_idx in 0..num_eval {
                                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                    let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                    all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                                }
                            }
                            continue;
                        }
                        Err(_e) => {
                        }
                    }
                }
            }

            // GPU sparse path: evaluate sparse groups using binary search on GPU
            if let (Some(ref sparse_eval), true) = (&sparse_metal, memory.is_sparse()) {
                if let Some(export) = memory.export_for_gpu_sparse() {
                    match evaluate_group_sparse_gpu(
                        sparse_eval.as_ref(),
                        &packed_eval,
                        &gpu_connections,
                        &export,
                        group,
                        num_eval,
                        words_per_example,
                        memory_mode,
                    ) {
                        Ok(group_scores) => {
                            for ex_idx in 0..num_eval {
                                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                                    let score_idx = ex_idx * group.cluster_count() + local_cluster;
                                    all_scores[ex_idx][cluster_id] = group_scores[score_idx] as f64;
                                }
                            }
                            continue;
                        }
                        Err(_e) => {
                        }
                    }
                }
            }

            // CPU path: evaluate examples in parallel using per-neuron bits
            all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                let input_start = ex_idx * total_input_bits;
                let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

                for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                    let actual_neurons = if let Some(ref an) = group.actual_neurons {
                        an[local_cluster] as usize
                    } else {
                        group.neurons
                    };

                    let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

                    let mut sum = 0.0f32;
                    for n in 0..actual_neurons {
                        let global_n = cluster_neuron_starts[cluster_id] + n;
                        let n_bits = per_neuron_bits[global_n];
                        let conn_start = neuron_conn_offsets[global_n];
                        let address = compute_address(input_bits, &original_connections[conn_start..], n_bits);
                        let cell = memory.read(neuron_base + n, address);
                        sum += cell_to_weight(cell, memory_mode, empty_value);
                    }

                    scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                }
            });
        }

        // Extract predictions and compute CE/accuracy from pre-computed scores
        let predictions: Vec<u32> = all_scores.par_iter()
            .map(|scores| {
                scores.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            })
            .collect();

        let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
            let target_idx = eval_targets[ex_idx] as usize;
            let predicted = predictions[ex_idx] as usize;
            let correct: u64 = if predicted == target_idx { 1 } else { 0 };

            // Softmax and cross-entropy for this example
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            let target_prob = exp_scores[target_idx] / sum_exp;
            let ce = -(target_prob + epsilon).ln();

            (ce, correct)
        }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

        let avg_ce = total_ce / num_eval as f64;
        let accuracy = total_correct as f64 / num_eval as f64;
        let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, get_normal_class());

        // Progress logging (to log file if WNN_LOG_PATH set, otherwise stderr)
        // Format matches Python's format_genome_log for consistency
        if progress_log {
            use std::io::Write;
            let genome_elapsed = genome_start.elapsed().as_secs_f64();
            let now = chrono::Local::now();

            // Calculate overall position using batch offset
            let overall_position = batch_offset + genome_idx + 1;

            // Calculate padding widths based on totals
            let gen_width = total_gens.to_string().len();
            let pos_width = total_count.to_string().len();

            // Pad type indicator to 4 chars (Init, New , Nbr , CE  , Acc )
            let type_padded = format!("{:<4}", &log_type[..log_type.len().min(4)]);

            // Format: [Gen 001/100] Genome 01/50 (Init): CE=10.6588, Acc=0.0100% (8.7s)
            let msg = format!(
                "{} | [Gen {:0gen_width$}/{:0gen_width$}] Genome {:0pos_width$}/{} ({}): CE={:.4}, Acc={:.4}% ({:.1}s)\n",
                now.format("%H:%M:%S"),
                current_gen, total_gens,
                overall_position, total_count,
                type_padded,
                avg_ce, accuracy * 100.0,
                genome_elapsed,
                gen_width = gen_width,
                pos_width = pos_width,
            );
            if let Some(ref path) = log_path {
                use fs2::FileExt;
                if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
                    // Lock file to prevent interleaved writes with Python
                    if file.lock_exclusive().is_ok() {
                        let _ = file.write_all(msg.as_bytes());
                        let _ = file.flush();
                        let _ = file.unlock();
                    }
                }
            } else {
                eprint!("{}", msg);
            }
        }

        (avg_ce, accuracy, f1, fpr)
    }).collect();

    results
}

/// Evaluate genomes with multi-subset rotation support.
///
/// All token subsets are pre-encoded and passed at once. The train_subset_idx
/// and eval_subset_idx select which subset to use for this batch of evaluations.
///
/// This enables per-generation/iteration rotation of data subsets, acting as
/// a regularizer that forces genomes to generalize across all subsets.
///
/// Args:
///   genomes_bits_flat: [num_genomes * num_clusters] bits per cluster
///   genomes_neurons_flat: [num_genomes * num_clusters] neurons per cluster
///   genomes_connections_flat: Flattened connections (empty = random)
///   num_genomes: Number of genomes to evaluate
///   num_clusters: Vocabulary size
///   train_subsets_flat: [sum(num_train_per_subset) * total_input_bits] all train input bits concatenated
///   train_targets_flat: [sum(num_train_per_subset)] all train targets concatenated
///   train_negatives_flat: [sum(num_train_per_subset) * num_negatives] all train negatives concatenated
///   train_subset_counts: [num_subsets] number of examples in each train subset
///   eval_subsets_flat: [sum(num_eval_per_subset) * total_input_bits] all eval input bits concatenated
///   eval_targets_flat: [sum(num_eval_per_subset)] all eval targets concatenated
///   eval_subset_counts: [num_subsets] number of examples in each eval subset
///   train_subset_idx: Which train subset to use (0-indexed)
///   eval_subset_idx: Which eval subset to use (0-indexed)
///   num_negatives: Number of negative samples per example
///   total_input_bits: Input bits per example
///   empty_value: Value for EMPTY cells (0.0 recommended)
///
/// Returns: Vec of (cross_entropy, accuracy) tuples - one per genome
pub fn evaluate_genomes_parallel_multisubset(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    // Train data - all subsets concatenated
    train_subsets_flat: &[bool],
    train_targets_flat: &[i64],
    train_negatives_flat: &[i64],
    train_subset_counts: &[usize],  // [num_subsets] examples per subset
    // Eval data - all subsets concatenated
    eval_subsets_flat: &[bool],
    eval_targets_flat: &[i64],
    eval_subset_counts: &[usize],  // [num_subsets] examples per subset
    // Subset selection
    train_subset_idx: usize,
    eval_subset_idx: usize,
    // Other params
    num_negatives: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64, f64, f64)> {
    // Compute offsets for train subsets
    let mut train_offsets: Vec<usize> = Vec::with_capacity(train_subset_counts.len() + 1);
    let mut train_target_offsets: Vec<usize> = Vec::with_capacity(train_subset_counts.len() + 1);
    train_offsets.push(0);
    train_target_offsets.push(0);
    for &count in train_subset_counts {
        let last_offset = *train_offsets.last().unwrap();
        train_offsets.push(last_offset + count * total_input_bits);
        let last_target_offset = *train_target_offsets.last().unwrap();
        train_target_offsets.push(last_target_offset + count);
    }

    // Compute offsets for eval subsets
    let mut eval_offsets: Vec<usize> = Vec::with_capacity(eval_subset_counts.len() + 1);
    let mut eval_target_offsets: Vec<usize> = Vec::with_capacity(eval_subset_counts.len() + 1);
    eval_offsets.push(0);
    eval_target_offsets.push(0);
    for &count in eval_subset_counts {
        let last_offset = *eval_offsets.last().unwrap();
        eval_offsets.push(last_offset + count * total_input_bits);
        let last_target_offset = *eval_target_offsets.last().unwrap();
        eval_target_offsets.push(last_target_offset + count);
    }

    // Extract the selected train subset
    let train_start = train_offsets[train_subset_idx];
    let train_end = train_offsets[train_subset_idx + 1];
    let train_input_bits = &train_subsets_flat[train_start..train_end];

    let train_target_start = train_target_offsets[train_subset_idx];
    let train_target_end = train_target_offsets[train_subset_idx + 1];
    let train_targets = &train_targets_flat[train_target_start..train_target_end];

    let num_train = train_subset_counts[train_subset_idx];
    let train_neg_start = train_target_start * num_negatives;
    let train_neg_end = train_target_end * num_negatives;
    let train_negatives = &train_negatives_flat[train_neg_start..train_neg_end];

    // Extract the selected eval subset
    let eval_start = eval_offsets[eval_subset_idx];
    let eval_end = eval_offsets[eval_subset_idx + 1];
    let eval_input_bits = &eval_subsets_flat[eval_start..eval_end];

    let eval_target_start = eval_target_offsets[eval_subset_idx];
    let eval_target_end = eval_target_offsets[eval_subset_idx + 1];
    let eval_targets = &eval_targets_flat[eval_target_start..eval_target_end];

    let num_eval = eval_subset_counts[eval_subset_idx];

    // Now delegate to the existing single-subset function
    evaluate_genomes_parallel(
        genomes_bits_flat,
        genomes_neurons_flat,
        genomes_connections_flat,
        num_genomes,
        num_clusters,
        train_input_bits,
        train_targets,
        train_negatives,
        num_train,
        num_negatives,
        eval_input_bits,
        eval_targets,
        num_eval,
        total_input_bits,
        empty_value,
        neuron_sample_rate,
        rng_seed,
    )
}

// =============================================================================
// PARALLEL HYBRID CPU+GPU EVALUATION
// =============================================================================

/// Export data for a single genome (used in batched GPU evaluation)
#[derive(Clone)]
pub struct GenomeExport {
    /// Connections for all groups, flattened
    pub connections: Vec<i64>,
    /// For each group: (is_sparse, group_idx, cluster_ids)
    pub group_info: Vec<(bool, usize, Vec<usize>)>,
    /// Dense group exports: memory words
    pub dense_exports: Vec<Vec<i64>>,
    /// Sparse group exports: sorted arrays for binary search
    pub sparse_exports: Vec<SparseGpuExport>,
    /// Config groups for this genome
    pub groups: Vec<ConfigGroup>,
}

/// Memory pool for reusable genome memory
struct GenomeMemoryPool {
    /// Memory sets for each pool slot
    memories: Vec<Vec<GroupMemory>>,
    /// Config groups template (same for all genomes with same config)
    groups_template: Vec<ConfigGroup>,
    /// Memory mode (for correct reset empty values)
    memory_mode: u8,
}

impl GenomeMemoryPool {
    /// Create a pool with the given number of slots
    fn new(
        pool_size: usize,
        bits_per_cluster: &[usize],
        neurons_per_cluster: &[usize],
        memory_mode: u8,
    ) -> Self {
        let groups_template = build_groups(bits_per_cluster, neurons_per_cluster);

        let memories = (0..pool_size)
            .map(|_| {
                groups_template.iter()
                    .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                    .collect()
            })
            .collect();

        Self {
            memories,
            groups_template,
            memory_mode,
        }
    }

    /// Reset all memory in a pool slot (clear for reuse)
    fn reset_slot(&self, slot: usize) {
        let empty_word = crate::neuron_memory::empty_word_for_mode(self.memory_mode);
        for memory in &self.memories[slot] {
            match memory {
                GroupMemory::Dense(m) => {
                    for word in &m.words {
                        word.store(empty_word, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                GroupMemory::Sparse(m) => {
                    for neuron_map in &m.neurons {
                        neuron_map.clear();
                    }
                }
            }
        }
    }
}

/// Calculate optimal pool and batch sizes based on memory budget
fn calculate_pool_size(
    bits_per_cluster: &[usize],
    neurons_per_cluster: &[usize],
    _num_clusters: usize,
    budget_gb: f64,
    cpu_cores: usize,
) -> (usize, usize) {
    // Estimate memory per genome (use same grouping strategy as actual training)
    let groups = build_groups(bits_per_cluster, neurons_per_cluster);
    let mut bytes_per_genome = 0usize;

    for group in &groups {
        if group.bits <= SPARSE_THRESHOLD {
            // Dense: 2 bits per cell, 2^bits cells per neuron
            let cells_per_neuron = 1 << group.bits;
            let words_per_neuron = (cells_per_neuron + 30) / 31; // 31 cells per word
            bytes_per_genome += group.total_neurons() * words_per_neuron * 8;
        } else {
            // Sparse: Based on measured data from actual training
            // With 100K training examples + 5 negatives = 600K writes, but many collide
            // Measured: ~1.2K unique entries per neuron on average (8.9M / 7500 neurons)
            // Use 3K as conservative estimate to leave headroom
            // Memory per entry: key(8) + value(1) + DashMap overhead (~24 bytes)
            bytes_per_genome += group.total_neurons() * 3_000 * 32;
        }
    }

    let budget_bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let max_pool_size = (budget_bytes / bytes_per_genome).max(1);

    // Pool size = min(max_pool, cpu_cores) to avoid over-allocation
    // Use WNN_BATCH_SIZE env var to override for testing
    let pool_size = max_pool_size.min(cpu_cores).max(1);

    // Batch size = pool size (process one batch at a time)
    let batch_size = pool_size;

    (pool_size, batch_size)
}

/// Get available memory in GB (macOS specific)
fn get_available_memory_gb() -> f64 {
    // Try to read from sysctl
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = mem_str.trim().parse::<u64>() {
                    return bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                }
            }
        }
    }
    // Fallback: assume 64GB (M4 Max typical)
    64.0
}

/// Compute per-class weights for balanced training: max_count / count.
///
/// Given class labels and number of classes, returns a Vec<u32> where each entry
/// is the integer weight for that class index. Minority classes get higher weights
/// to counteract the effect of class imbalance on address saturation.
///
/// Example: labels with 119K attack (class 1) and 56K normal (class 0)
///   → weights = [2, 1] (119K/56K ≈ 2, 119K/119K = 1)
pub fn compute_class_weights(labels: &[i64], num_classes: usize) -> Vec<u32> {
    compute_class_weights_with_multiplier(labels, num_classes, 1.0)
}

pub fn compute_class_weights_with_multiplier(labels: &[i64], num_classes: usize, multiplier: f32) -> Vec<u32> {
    let mut counts = vec![0u64; num_classes];
    for &label in labels {
        let c = label as usize;
        if c < num_classes {
            counts[c] += 1;
        }
    }
    let max_count = *counts.iter().max().unwrap_or(&1);
    counts.iter().map(|&c| {
        if c == 0 { 1 } else {
            let base = (max_count / c).max(1) as f32;
            (base * multiplier).max(1.0) as u32
        }
    }).collect()
}

/// Train a genome using the given memory slot.
/// When `gpu_addresses` is Some, uses pre-computed GPU addresses instead of CPU compute_address().
/// GPU address layout: addresses[global_neuron_idx * num_train + example_idx].
/// When `parallel` is true, uses rayon par_iter for example-level parallelism.
/// Set `parallel=false` when calling from within an outer par_iter to avoid nested parallelism deadlock.
pub(crate) fn train_genome_in_slot(
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    original_connections: &[i64],    // Per-neuron layout (NOT group layout)
    per_neuron_bits: &[usize],       // Bits per neuron
    cluster_neuron_starts: &[usize], // First neuron idx per cluster
    neuron_conn_offsets: &[usize],   // Conn offset per neuron
    cluster_to_group: &[(usize, usize)],
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    total_input_bits: usize,
    gpu_addresses: Option<&[u32]>,
    neuron_sample_rate: f32,
    rng_seed: u64,
    memory_mode: u8,
    class_weights: Option<&[u32]>,
    parallel: bool,
) {
    let use_sampling = neuron_sample_rate < 1.0;
    let use_nudge = memory_mode != crate::neuron_memory::MODE_TERNARY;

    let train_one_example = |ex_idx: usize| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &train_input_bits[input_start..input_start + total_input_bits];

        let num_clusters = cluster_to_group.len();
        // Single-cluster mode: always target cluster 0, nudge direction = label
        // Multi-cluster mode: target cluster = label, always nudge TRUE
        let true_cluster = if num_clusters == 1 { 0 } else { train_targets[ex_idx] as usize };
        let nudge_direction = if num_clusters == 1 {
            train_targets[ex_idx] == 1  // Attack=TRUE, Normal=FALSE
        } else {
            true  // Always positive for target cluster in multi-cluster mode
        };

        // Train positive example
        {
            let (group_idx, local_cluster) = cluster_to_group[true_cluster];
            let group = &groups[group_idx];
            let memory = &memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

            for n in 0..actual_neurons {
                let global_n = cluster_neuron_starts[true_cluster] + n;

                // Per-(neuron, example) deterministic sampling
                // Uses hash of (rng_seed, neuron_idx, example_idx) for parallel-safe decisions
                if use_sampling {
                    let mut rng = (rng_seed as u32)
                        .wrapping_add(global_n as u32 * 1000003)
                        .wrapping_add(ex_idx as u32 * 2654435761);
                    if rng == 0 { rng = 1; }
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    if (rng >> 8) as f32 / 16777216.0 >= neuron_sample_rate {
                        continue;
                    }
                }

                let address = if let Some(addrs) = gpu_addresses {
                    addrs[global_n * num_train + ex_idx] as usize
                } else {
                    let n_bits = per_neuron_bits[global_n];
                    let conn_start = neuron_conn_offsets[global_n];
                    compute_address(input_bits, &original_connections[conn_start..], n_bits)
                };
                // Weight by original label for class balancing
                let weight_idx = train_targets[ex_idx] as usize;
                let repeats = class_weights.map_or(1u32, |w| w[weight_idx]);
                if use_nudge {
                    for _ in 0..repeats {
                        memory.nudge(neuron_base + n, address, nudge_direction);
                    }
                } else {
                    let value = if nudge_direction { TRUE } else { FALSE };
                    memory.write(neuron_base + n, address, value, false);
                }
            }
        }

        // Train negative examples
        let neg_start = ex_idx * num_negatives;
        for k in 0..num_negatives {
            let false_cluster = train_negatives[neg_start + k] as usize;
            if false_cluster == true_cluster {
                continue;
            }

            let (group_idx, local_cluster) = cluster_to_group[false_cluster];
            let group = &groups[group_idx];
            let memory = &memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;  // Keep MAX for memory layout

            for n in 0..actual_neurons {
                let global_n = cluster_neuron_starts[false_cluster] + n;

                // Same per-(neuron, example) sampling for negative examples
                if use_sampling {
                    let mut rng = (rng_seed as u32)
                        .wrapping_add(global_n as u32 * 1000003)
                        .wrapping_add(ex_idx as u32 * 2654435761);
                    if rng == 0 { rng = 1; }
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    if (rng >> 8) as f32 / 16777216.0 >= neuron_sample_rate {
                        continue;
                    }
                }

                let address = if let Some(addrs) = gpu_addresses {
                    addrs[global_n * num_train + ex_idx] as usize
                } else {
                    let n_bits = per_neuron_bits[global_n];
                    let conn_start = neuron_conn_offsets[global_n];
                    compute_address(input_bits, &original_connections[conn_start..], n_bits)
                };
                // For negative nudges, weight by the TRUE class of the example
                // (the example "belongs to" true_cluster, so its weight applies)
                let repeats = class_weights.map_or(1u32, |w| w[true_cluster]);
                if use_nudge {
                    for _ in 0..repeats {
                        memory.nudge(neuron_base + n, address, false);
                    }
                } else {
                    memory.write(neuron_base + n, address, FALSE, false);
                }
            }
        }
    };

    if parallel {
        let chunk_size = 10_000.max(num_train / 20);
        (0..num_train).into_par_iter()
            .with_min_len(chunk_size)
            .for_each(|ex_idx| train_one_example(ex_idx));
    } else {
        for ex_idx in 0..num_train {
            train_one_example(ex_idx);
        }
    }
}

/// Export trained memory to GPU-compatible format
fn export_genome_for_gpu(
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    connections_flat: &[i64],
) -> GenomeExport {
    let mut dense_exports = Vec::new();
    let mut sparse_exports = Vec::new();
    let mut group_info = Vec::new();

    for (group_idx, (group, memory)) in groups.iter().zip(memories.iter()).enumerate() {
        let is_sparse = memory.is_sparse();
        group_info.push((is_sparse, group_idx, group.cluster_ids.clone()));

        if is_sparse {
            if let Some(export) = memory.export_for_gpu_sparse() {
                sparse_exports.push(export);
            } else {
                // Fallback: empty export
                sparse_exports.push(SparseGpuExport {
                    keys: vec![],
                    values: vec![],
                    offsets: vec![0; group.total_neurons()],
                    counts: vec![0; group.total_neurons()],
                    num_neurons: group.total_neurons(),
                });
            }
        } else {
            if let Some(words) = memory.export_for_metal() {
                dense_exports.push(words);
            } else {
                dense_exports.push(vec![]);
            }
        }
    }

    GenomeExport {
        connections: connections_flat.to_vec(),
        group_info,
        dense_exports,
        sparse_exports,
        groups: groups.to_vec(),
    }
}

// Thread-local cache for GPU buffers to avoid expensive 10GB buffer allocation per evaluation
// The scores buffer is ~10GB (50K examples × 50K clusters × 4 bytes), so reusing it is critical.
// The cache includes the reset generation to invalidate on Metal reset.
#[cfg(target_os = "macos")]
thread_local! {
    // (reset_gen, num_eval, num_clusters, buffer)
    static CACHED_SCORES_BUFFER: std::cell::RefCell<Option<(u64, usize, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
    // (reset_gen, size, buffer)
    static CACHED_INPUT_BUFFER: std::cell::RefCell<Option<(u64, usize, metal::Buffer)>> = std::cell::RefCell::new(None);
}

/// Evaluate a genome export using CPU+GPU hybrid
/// Returns (cross_entropy, accuracy)
/// Compute per-example, per-cluster scores from a trained genome export.
///
/// Shared by `evaluate_genome_hybrid` (CE/accuracy) and `predict_genome_hybrid` (argmax).
/// Tries GPU evaluation (sparse + dense) for each group, falling back to CPU binary search.
fn compute_per_example_scores(
    export: &GenomeExport,
    eval_input_bits: &[bool],
    packed_eval: &[u64],
    words_per_example: usize,
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    empty_value: f32,
    memory_mode: u8,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&crate::metal_ramlm::MetalSparseEvaluator>,
) -> Vec<Vec<f64>> {
    let mut all_scores: Vec<Vec<f64>> = vec![vec![0.0; num_clusters]; num_eval];

    let mut dense_idx = 0usize;
    let mut sparse_idx = 0usize;

    for (is_sparse, group_idx, cluster_ids) in &export.group_info {
        let group = &export.groups[*group_idx];

        if *is_sparse {
            let sparse_export = &export.sparse_exports[sparse_idx];
            sparse_idx += 1;

            let gpu_success = if let Some(sparse_eval) = sparse_metal {
                match evaluate_group_sparse_gpu(
                    sparse_eval,
                    packed_eval,
                    &export.connections,
                    sparse_export,
                    group,
                    num_eval,
                    words_per_example,
                    memory_mode,
                ) {
                    Ok(group_scores) => {
                        let num_group_clusters = group.cluster_count();
                        all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                            for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * num_group_clusters + local_cluster;
                                scores[cluster_id] = group_scores[score_idx] as f64;
                            }
                        });
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };

            if !gpu_success {
                // CPU fallback using binary search
                all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                    let input_start = ex_idx * total_input_bits;
                    let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

                    for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                        let actual_neurons = if let Some(ref an) = group.actual_neurons {
                            an[local_cluster] as usize
                        } else {
                            group.neurons
                        };

                        let neuron_base = local_cluster * group.neurons;
                        let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                        let mut sum = 0.0f32;
                        for n in 0..actual_neurons {
                            let conn_start = conn_base + n * group.bits;
                            let address = compute_address(input_bits, &export.connections[conn_start..], group.bits);
                            let cell = sparse_export.lookup(neuron_base + n, address as u64);
                            sum += cell_to_weight(cell as i64, memory_mode, empty_value);
                        }

                        scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                    }
                });
            }
        } else {
            let dense_words = &export.dense_exports[dense_idx];
            dense_idx += 1;

            let gpu_success = if let Some(metal_eval) = metal {
                match evaluate_group_metal(
                    metal_eval,
                    packed_eval,
                    &export.connections,
                    dense_words,
                    group,
                    num_eval,
                    words_per_example,
                    memory_mode,
                ) {
                    Ok(group_scores) => {
                        let num_group_clusters = group.cluster_count();
                        all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                            for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                                let score_idx = ex_idx * num_group_clusters + local_cluster;
                                scores[cluster_id] = group_scores[score_idx] as f64;
                            }
                        });
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };

            if !gpu_success {
                // CPU fallback for dense groups
                all_scores.par_iter_mut().enumerate().for_each(|(ex_idx, scores)| {
                    let input_start = ex_idx * total_input_bits;
                    let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

                    for (local_cluster, &cluster_id) in cluster_ids.iter().enumerate() {
                        let actual_neurons = if let Some(ref an) = group.actual_neurons {
                            an[local_cluster] as usize
                        } else {
                            group.neurons
                        };

                        let neuron_base = local_cluster * group.neurons;
                        let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                        let mut sum = 0.0f32;
                        for n in 0..actual_neurons {
                            let conn_start = conn_base + n * group.bits;
                            let address = compute_address(input_bits, &export.connections[conn_start..], group.bits);
                            let cell = read_cell(dense_words, neuron_base + n, address, group.words_per_neuron);
                            sum += cell_to_weight(cell, memory_mode, empty_value);
                        }

                        scores[cluster_id] = (sum / actual_neurons as f32) as f64;
                    }
                });
            }
        }
    }

    all_scores
}

pub fn evaluate_genome_hybrid(
    export: &GenomeExport,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    empty_value: f32,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&crate::metal_ramlm::MetalSparseEvaluator>,
    override_threshold: Option<f64>,
) -> (f64, f64, f64, f64, f64) {
    let memory_mode = crate::neuron_memory::get_memory_mode();
    let epsilon = 1e-10f64;

    // Detailed timing (enabled via WNN_GROUP_TIMING env var)
    let timing_enabled = std::env::var("WNN_GROUP_TIMING").is_ok();
    let eval_start = std::time::Instant::now();

    // Pack eval input bits to u64 for GPU (pack once, reuse for all GPU paths)
    let (packed_eval, words_per_example) = crate::neuron_memory::pack_bools_to_u64(
        eval_input_bits, num_eval, total_input_bits
    );

    // SINGLE-CLUSTER BINARY DISCRIMINATOR: use override or find threshold, binary cross-entropy
    if num_clusters == 1 {
        let all_scores = compute_per_example_scores(
            export, eval_input_bits, &packed_eval, words_per_example,
            num_eval, num_clusters, total_input_bits, empty_value,
            memory_mode, metal, sparse_metal,
        );

        // BCE loss (threshold-independent)
        let mut total_ce = 0.0f64;
        for ex_idx in 0..num_eval {
            let s = (all_scores[ex_idx][0]).clamp(epsilon, 1.0 - epsilon);
            let y = eval_targets[ex_idx] as f64;
            total_ce += -(y * s.ln() + (1.0 - y) * (1.0 - s).ln());
        }
        let ce = total_ce / num_eval as f64;

        // Use override threshold (from training calibration) or find on eval data (fallback)
        let flat_scores: Vec<f64> = all_scores.iter().map(|s| s[0]).collect();
        let threshold = override_threshold.unwrap_or_else(|| {
            let (t, _f1, _fpr) = find_optimal_threshold_auto(&flat_scores, eval_targets);
            t
        });

        // Apply threshold for predictions
        let mut correct = 0u64;
        let mut predictions = Vec::with_capacity(num_eval);
        for ex_idx in 0..num_eval {
            let pred = if all_scores[ex_idx][0] >= threshold { 1u32 } else { 0u32 };
            predictions.push(pred);
            if pred as i64 == eval_targets[ex_idx] { correct += 1; }
        }
        let acc = correct as f64 / num_eval as f64;
        let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, 2, get_normal_class());
        return (ce, acc, f1, fpr, threshold);
    }

    // FAST PATH: If single sparse group covering all clusters, use direct CE computation
    // This avoids the 10GB GPU→CPU transfer by computing CE on GPU
    #[cfg(target_os = "macos")]
    if export.group_info.len() == 1 && export.sparse_exports.len() == 1 {
        let (is_sparse, group_idx, cluster_ids) = &export.group_info[0];
        if *is_sparse && cluster_ids.len() == num_clusters {
            // Check if clusters are contiguous 0..num_clusters (identity mapping)
            let is_contiguous = cluster_ids.iter().enumerate().all(|(i, &c)| c == i);
            if is_contiguous {
                // Use MetalSparseCEEvaluator for direct CE computation
                static CE_EVALUATOR: std::sync::OnceLock<Option<crate::metal_ramlm::MetalSparseCEEvaluator>> = std::sync::OnceLock::new();
                let ce_eval = CE_EVALUATOR.get_or_init(|| {
                    crate::metal_ramlm::MetalSparseCEEvaluator::new().ok()
                });

                if let Some(evaluator) = ce_eval {
                    let group = &export.groups[*group_idx];
                    let sparse_export = &export.sparse_exports[0];

                    let call_start = std::time::Instant::now();
                    let result = evaluator.compute_ce(
                        &packed_eval,
                        &export.connections,
                        &sparse_export.keys,
                        &sparse_export.values,
                        &sparse_export.offsets,
                        &sparse_export.counts,
                        eval_targets,
                        num_eval,
                        words_per_example,
                        group.neurons * num_clusters,  // total neurons
                        group.bits,
                        group.neurons,
                        num_clusters,
                        empty_value,
                        memory_mode,
                    );

                    if let Ok((ce, acc, predictions)) = result {
                        let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, get_normal_class());
                        if timing_enabled {
                            let elapsed = call_start.elapsed().as_millis();
                            let total_ms = eval_start.elapsed().as_millis();
                            eprintln!(
                                "[EVAL_HYBRID] FAST_PATH total={}ms gpu_ce={}ms (no scatter!)",
                                total_ms, elapsed
                            );
                        }
                        return (ce, acc, f1, fpr, 0.5);
                    }
                    // Fall through to standard path if CE evaluator fails
                }
            }
        }
    }

    // FULL GPU PATH: Write scores directly to shared GPU buffer, compute CE on GPU
    // This avoids the GPU→CPU→GPU round-trip that was slowing down tiered evaluation
    // Full GPU path enabled by default (3x faster). Disable with WNN_GPU_CE=0
    #[cfg(target_os = "macos")]
    {
    let use_full_gpu = std::env::var("WNN_GPU_CE").map(|v| v != "0").unwrap_or(true);

    if use_full_gpu {
        if let Some(group_eval) = get_group_evaluator() {
            let gpu_start = std::time::Instant::now();

            // Get current reset generation to detect when Metal evaluators were reset
            let current_reset_gen = RESET_GENERATION.load(Ordering::SeqCst);

            // Phase timing for detailed analysis (enabled via WNN_PHASE_TIMING env var)
            let phase_timing = std::env::var("WNN_PHASE_TIMING").is_ok();
            let phase_start = std::time::Instant::now();

            // Get or create cached scores buffer (avoids expensive 10GB allocation per eval)
            // The scores buffer is zeroed efficiently using memset instead of creating a new Vec
            // Invalidate cache if Metal was reset (reset_gen changed)
            let scores_buffer = CACHED_SCORES_BUFFER.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some((cached_gen, cached_eval, cached_clusters, ref buffer)) = *cache {
                    if cached_gen == current_reset_gen && cached_eval == num_eval && cached_clusters == num_clusters {
                        // Reuse existing buffer - just zero it
                        group_eval.zero_scores_buffer(buffer, num_eval, num_clusters);
                        return buffer.clone();
                    }
                }
                // Create new buffer and cache it (invalidate on reset_gen mismatch)
                let buffer = group_eval.create_scores_buffer(num_eval, num_clusters);
                *cache = Some((current_reset_gen, num_eval, num_clusters, buffer.clone()));
                buffer
            });

            let zero_time_ms = if phase_timing { phase_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };
            let phase_start = std::time::Instant::now();

            // Get or create cached input buffer (update contents efficiently)
            // Invalidate cache if Metal was reset (reset_gen changed)
            // Uses packed u64 input for GPU
            let input_buffer = CACHED_INPUT_BUFFER.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some((cached_gen, cached_size, ref buffer)) = *cache {
                    if cached_gen == current_reset_gen && cached_size == packed_eval.len() {
                        // Reuse existing buffer - update contents
                        group_eval.update_input_buffer(buffer, &packed_eval);
                        return buffer.clone();
                    }
                }
                // Create new buffer and cache it (invalidate on reset_gen mismatch)
                let buffer = group_eval.create_input_buffer(&packed_eval);
                *cache = Some((current_reset_gen, packed_eval.len(), buffer.clone()));
                buffer
            });

            let input_time_ms = if phase_timing { phase_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };
            let _phase_start = std::time::Instant::now();

            let mut dense_idx: usize;
            let mut sparse_idx = 0usize;
            let all_groups_success = true;
            let mut sparse_time_ms = 0.0f64;
            let mut dense_time_ms = 0.0f64;
            let sparse_call_count;

            // Collect all sparse groups for batched evaluation (single command buffer)
            // This eliminates ~0.5ms overhead per group from separate commit+wait cycles
            let mut sparse_groups: Vec<crate::metal_ramlm::SparseGroupData> = Vec::new();
            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                if *is_sparse {
                    let group = &export.groups[*group_idx];
                    let sparse_export = &export.sparse_exports[sparse_idx];
                    sparse_idx += 1;

                    sparse_groups.push(crate::metal_ramlm::SparseGroupData {
                        connections: &export.connections[group.conn_offset..group.conn_offset + group.conn_size()],
                        keys: &sparse_export.keys,
                        values: &sparse_export.values,
                        offsets: &sparse_export.offsets,
                        counts: &sparse_export.counts,
                        cluster_ids,
                        bits_per_neuron: group.bits,
                        neurons_per_cluster: group.neurons,
                        actual_neurons_per_cluster: group.actual_neurons.as_deref(),
                    });
                }
            }

            // Evaluate all sparse groups in a single batched call
            sparse_call_count = sparse_groups.len();
            if !sparse_groups.is_empty() {
                let sparse_start = std::time::Instant::now();
                group_eval.eval_sparse_groups_batched(
                    &input_buffer,
                    &scores_buffer,
                    &sparse_groups,
                    num_eval,
                    words_per_example,
                    num_clusters,
                    empty_value,
                    memory_mode
                );
                if phase_timing {
                    sparse_time_ms = sparse_start.elapsed().as_micros() as f64 / 1000.0;
                }
            }

            // Evaluate dense groups individually (uses DENSE_BUFFER_CACHE)
            dense_idx = 0;
            for (is_sparse, group_idx, cluster_ids) in &export.group_info {
                if !*is_sparse {
                    let group = &export.groups[*group_idx];
                    let dense_words = &export.dense_exports[dense_idx];
                    dense_idx += 1;

                    let dense_start = std::time::Instant::now();
                    group_eval.eval_dense_to_buffer(
                        &input_buffer,
                        &scores_buffer,
                        &export.connections[group.conn_offset..group.conn_offset + group.conn_size()],
                        dense_words,
                        cluster_ids,
                        num_eval,
                        words_per_example,
                        group.bits,
                        group.neurons,
                        num_clusters,
                        group.words_per_neuron,
                        empty_value,
                        memory_mode
                    );

                    if phase_timing {
                        dense_time_ms += dense_start.elapsed().as_micros() as f64 / 1000.0;
                    }
                }
            }

            let ce_start = std::time::Instant::now();

            if all_groups_success {
                // Compute CE directly from GPU buffer
                let result = group_eval.compute_ce_from_buffer(
                    &scores_buffer,
                    eval_targets,
                    num_eval,
                    num_clusters,
                );

                let ce_time_ms = if phase_timing { ce_start.elapsed().as_micros() as f64 / 1000.0 } else { 0.0 };

                if let Ok((ce, acc, predictions)) = result {
                    let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, get_normal_class());
                    if timing_enabled {
                        let elapsed = gpu_start.elapsed().as_millis();
                        if phase_timing {
                            eprintln!(
                                "[EVAL_PHASE] zero={:.1}ms input={:.1}ms sparse={:.1}ms({} calls) dense={:.1}ms ce={:.1}ms total={}ms",
                                zero_time_ms, input_time_ms, sparse_time_ms, sparse_call_count, dense_time_ms, ce_time_ms, elapsed
                            );
                        } else {
                            eprintln!(
                                "[EVAL_HYBRID] FULL_GPU_PATH total={}ms (no CPU scatter!)",
                                elapsed
                            );
                        }
                    }
                    return (ce, acc, f1, fpr, 0.5);
                }
            }
            // Fall through to CPU path if full GPU fails
        }
    }
    } // cfg(target_os = "macos")

    // CPU FALLBACK PATH: Compute per-example scores using shared function
    let all_scores = compute_per_example_scores(
        export, eval_input_bits, &packed_eval, words_per_example,
        num_eval, num_clusters, total_input_bits, empty_value,
        memory_mode, metal, sparse_metal,
    );

    if timing_enabled {
        let total_ms = eval_start.elapsed().as_millis();
        eprintln!("[EVAL_HYBRID] CPU_FALLBACK total={}ms", total_ms);
    }

    // Compute CE, accuracy, and predictions from pre-computed scores
    let predictions: Vec<u32> = all_scores.par_iter().map(|scores| {
        scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }).collect();

    let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
        let target_idx = eval_targets[ex_idx] as usize;
        let predicted = predictions[ex_idx] as usize;
        let correct: u64 = if predicted == target_idx { 1 } else { 0 };

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let target_prob = exp_scores[target_idx] / sum_exp;
        let ce = -(target_prob + epsilon).ln();

        (ce, correct)
    }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

    let avg_ce = total_ce / num_eval as f64;
    let accuracy = total_correct as f64 / num_eval as f64;
    let (f1, fpr) = compute_f1_fpr_with_normal_class(&predictions, eval_targets, num_clusters, get_normal_class());

    (avg_ce, accuracy, f1, fpr, 0.5)
}

/// Predict per-example class indices for a single trained genome.
///
/// Delegates to `compute_per_example_scores` for the shared score computation,
/// then returns argmax predictions instead of CE/accuracy.
/// Used by the bitwise ECOC classifier to combine per-bit predictions.
pub fn predict_genome_hybrid(
    export: &GenomeExport,
    eval_input_bits: &[bool],
    num_eval: usize,
    num_clusters: usize,
    total_input_bits: usize,
    empty_value: f32,
    metal: Option<&crate::metal_ramlm::MetalRAMLMEvaluator>,
    sparse_metal: Option<&crate::metal_ramlm::MetalSparseEvaluator>,
    single_cluster_threshold: Option<f64>,
) -> Vec<i64> {
    let memory_mode = crate::neuron_memory::get_memory_mode();

    let (packed_eval, words_per_example) = crate::neuron_memory::pack_bools_to_u64(
        eval_input_bits, num_eval, total_input_bits
    );

    let all_scores = compute_per_example_scores(
        export, eval_input_bits, &packed_eval, words_per_example,
        num_eval, num_clusters, total_input_bits, empty_value,
        memory_mode, metal, sparse_metal,
    );

    // Single-cluster binary discriminator: use provided threshold or default 0.5
    if num_clusters == 1 {
        let threshold = single_cluster_threshold.unwrap_or(0.5);
        return all_scores.par_iter().map(|scores| {
            if scores[0] >= threshold { 1i64 } else { 0i64 }
        }).collect();
    }

    // Return argmax predictions (not CE/accuracy)
    all_scores.par_iter().map(|scores| {
        scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap_or(0)
    }).collect()
}

/// Train a single genome and return per-example predicted class indices.
///
/// Combines the training path from `evaluate_genomes_parallel_hybrid` (single genome)
/// with `predict_genome_hybrid` for per-example predictions.
#[allow(clippy::too_many_arguments)]
pub fn train_and_predict_single(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> Vec<i64> {
    let memory_mode = crate::neuron_memory::get_memory_mode();

    // Extract single genome config
    let neurons_per_cluster = genomes_neurons_flat;
    let per_neuron_bits = genomes_bits_flat;
    let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);
    let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    let memories: Vec<GroupMemory> = groups.iter()
        .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
        .collect();

    let original_connections = genomes_connections_flat.to_vec();

    // Pack training input for GPU address computation
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_bools_to_u64(train_input_bits, num_train, total_input_bits);

    let gpu_addresses = try_gpu_addresses_adaptive(
        &packed_train_input,
        words_per_example,
        per_neuron_bits,
        &neuron_conn_offsets,
        &original_connections,
        num_train,
    );

    // Train
    train_genome_in_slot(
        &memories,
        &groups,
        &original_connections,
        per_neuron_bits,
        &cluster_neuron_starts,
        &neuron_conn_offsets,
        &cluster_to_group,
        train_input_bits,
        train_targets,
        train_negatives,
        num_train,
        num_negatives,
        total_input_bits,
        gpu_addresses.as_deref(),
        neuron_sample_rate,
        rng_seed,
        memory_mode,
        class_weights,
        true, // parallel: standalone call, safe to use par_iter
    );

    // Export and predict
    let gpu_connections = reorganize_connections_for_gpu(
        &original_connections,
        per_neuron_bits,
        neurons_per_cluster,
        &groups,
    );
    let export = export_genome_for_gpu(&memories, &groups, &gpu_connections);

    // Get Metal evaluators for GPU prediction
    let metal = get_metal_evaluator();
    let sparse_metal = get_sparse_metal_evaluator();

    // Single-cluster: calibrate threshold on training data before predicting eval
    let threshold = if num_clusters == 1 {
        let train_scores = compute_per_example_scores(
            &export, train_input_bits, &packed_train_input, words_per_example,
            num_train, num_clusters, total_input_bits, empty_value,
            memory_mode, metal.as_deref(), sparse_metal.as_deref(),
        );
        let flat_scores: Vec<f64> = train_scores.iter().map(|s| s[0]).collect();
        let (t, f1, fpr) = find_optimal_threshold_auto(&flat_scores, train_targets);
        eprintln!(
            "[SINGLE_CLUSTER] Train calibration: threshold={:.4}, train_f1={:.4}, train_fpr={:.4}",
            t, f1, fpr
        );
        Some(t)
    } else {
        None
    };

    predict_genome_hybrid(
        &export,
        eval_input_bits,
        num_eval,
        num_clusters,
        total_input_bits,
        empty_value,
        metal.as_deref(),
        sparse_metal.as_deref(),
        threshold,
    )
}

/// Train a single genome and return per-example RAW SCORES on eval set.
///
/// Like `train_and_predict_single` but returns `Vec<f64>` scores instead of
/// thresholded class predictions. Used for Platt scaling calibration.
#[allow(clippy::too_many_arguments)]
pub fn train_and_score_single(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> Vec<f64> {
    let memory_mode = crate::neuron_memory::get_memory_mode();

    let neurons_per_cluster = genomes_neurons_flat;
    let per_neuron_bits = genomes_bits_flat;
    let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

    let (cluster_neuron_starts, neuron_conn_offsets) =
        build_neuron_metadata(per_neuron_bits, neurons_per_cluster);
    let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    let memories: Vec<GroupMemory> = groups.iter()
        .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
        .collect();

    let original_connections = genomes_connections_flat.to_vec();

    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_bools_to_u64(train_input_bits, num_train, total_input_bits);

    let gpu_addresses = try_gpu_addresses_adaptive(
        &packed_train_input, words_per_example,
        per_neuron_bits, &neuron_conn_offsets,
        &original_connections, num_train,
    );

    // Train
    train_genome_in_slot(
        &memories, &groups, &original_connections,
        per_neuron_bits, &cluster_neuron_starts, &neuron_conn_offsets,
        &cluster_to_group,
        train_input_bits, train_targets, train_negatives,
        num_train, num_negatives, total_input_bits,
        gpu_addresses.as_deref(),
        neuron_sample_rate, rng_seed, memory_mode, class_weights,
        true,
    );

    // Export and compute raw scores
    let gpu_connections = reorganize_connections_for_gpu(
        &original_connections, per_neuron_bits, neurons_per_cluster, &groups,
    );
    let export = export_genome_for_gpu(&memories, &groups, &gpu_connections);

    let metal = get_metal_evaluator();
    let sparse_metal = get_sparse_metal_evaluator();

    let (packed_eval, eval_words) = crate::neuron_memory::pack_bools_to_u64(
        eval_input_bits, num_eval, total_input_bits
    );

    let all_scores = compute_per_example_scores(
        &export, eval_input_bits, &packed_eval, eval_words,
        num_eval, num_clusters, total_input_bits, empty_value,
        memory_mode, metal.as_deref(), sparse_metal.as_deref(),
    );

    // Return raw score for cluster 0 (single-cluster mode)
    all_scores.iter().map(|scores| scores[0]).collect()
}

/// Evaluate genomes in PARALLEL with CPU+GPU HYBRID evaluation and PIPELINING.
///
/// Strategy:
/// 1. Parallel genome training on CPU (batch of genomes at once)
/// 2. Export to GPU-compatible format
/// 3. GPU batch evaluation (while CPU trains next batch)
/// 4. CPU+GPU hybrid: GPU evaluates, CPU assists with fallback
///
/// Performance benefits:
/// - Parallel training: N genomes trained simultaneously (vs sequential)
/// - GPU acceleration: Both dense and sparse groups on GPU
/// - Persistent worker: Eval thread stays alive across calls (eliminates spawn overhead)
/// - Pipelining: CPU trains batch N+1 while GPU evaluates batch N
///
/// Returns: Vec of (cross_entropy, accuracy) tuples - one per genome
pub fn evaluate_genomes_parallel_hybrid(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
) -> Vec<(f64, f64, f64, f64, f64)> {
    let memory_mode = crate::neuron_memory::get_memory_mode();
    if num_genomes == 0 {
        return vec![];
    }

    // Pre-compute genome_bpn_offsets: genomes_bits_flat has total_neurons entries per genome
    // (per-neuron bits), NOT num_clusters entries. This offset table maps each genome to its
    // slice in genomes_bits_flat.
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    debug_assert_eq!(
        genomes_bits_flat.len(),
        *genome_bpn_offsets.last().unwrap(),
        "genomes_bits_flat length ({}) != expected total neurons ({})",
        genomes_bits_flat.len(),
        genome_bpn_offsets.last().unwrap(),
    );

    // Get first genome's config to determine pool sizing (use per-cluster max bits)
    let first_neurons = &genomes_neurons_flat[0..num_clusters];
    let first_per_neuron_bits = &genomes_bits_flat[0..genome_bpn_offsets[1]];
    let first_bits_per_cluster = per_cluster_max_bits(first_per_neuron_bits, first_neurons);

    // Calculate memory budget and pool size
    let batch_size = std::env::var("WNN_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let budget_gb = get_available_memory_gb() * 0.6;
            let cpu_cores = rayon::current_num_threads();
            let (_, computed_batch) = calculate_pool_size(
                &first_bits_per_cluster,
                first_neurons,
                num_clusters,
                budget_gb,
                cpu_cores,
            );
            computed_batch
        });

    // Pre-compute connection offsets and sizes for each genome (handles variable configs)
    let use_provided_connections = !genomes_connections_flat.is_empty();

    // Compute per-genome connection offsets: conn_size = sum of per-neuron bits
    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    // Create shared eval data (Arc for zero-copy sharing with persistent worker)
    let eval_data = Arc::new(EvalData {
        eval_input_bits: eval_input_bits.to_vec(),
        eval_targets: eval_targets.to_vec(),
        num_eval,
        num_clusters,
        total_input_bits,
        empty_value,
    });

    // Get persistent eval worker (initialized once, stays alive for session).
    // Side-effect only: ensures the lazy global is initialized here rather
    // than in the first hot-path call. Return value intentionally unused.
    let _ = get_eval_worker();

    // Collect all results
    let mut all_results: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::with_capacity(num_genomes);

    // Process genomes in batches
    let num_batches = (num_genomes + batch_size - 1) / batch_size;

    // Log batch configuration if WNN_GROUP_LOG is set
    if std::env::var("WNN_GROUP_LOG").is_ok() && num_batches > 1 {
        eprintln!(
            "[BATCH_CONFIG] genomes={} batch_size={} num_batches={}",
            num_genomes, batch_size, num_batches
        );
    }

    // Timing instrumentation (enabled via WNN_TIMING env var)
    let timing_enabled = std::env::var("WNN_TIMING").is_ok();
    let mut total_train_ms = 0u128;
    let mut total_eval_ms = 0u128;
    let mut total_sparse_keys = 0usize;

    // Pack input bits to u64 once (shared across all genomes for GPU address computation)
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_bools_to_u64(train_input_bits, num_train, total_input_bits);

    // Progress logging for parallel batch (logs training completion, not final results)
    let progress_log = std::env::var("WNN_PROGRESS_LOG").map(|v| v == "1").unwrap_or(false);
    let log_path = std::env::var("WNN_LOG_PATH").ok();
    let current_gen: usize = std::env::var("WNN_PROGRESS_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let total_gens: usize = std::env::var("WNN_PROGRESS_TOTAL_GENS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let log_type = std::env::var("WNN_PROGRESS_TYPE").unwrap_or_else(|_| "Init".to_string());
    let batch_offset: usize = std::env::var("WNN_PROGRESS_OFFSET").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let total_count: usize = std::env::var("WNN_PROGRESS_TOTAL").ok().and_then(|v| v.parse().ok()).unwrap_or(num_genomes);

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_genomes);
        let current_batch_size = batch_end - batch_start;

        let train_start = std::time::Instant::now();

        // Train batch in parallel - each genome builds its own config (handles variable architectures)
        let mut batch_exports: Vec<(usize, GenomeExport, Option<f64>)> = (0..current_batch_size)
            .into_par_iter()
            .map(|local_idx| {
                let genome_idx = batch_start + local_idx;

                // Get this genome's config (per-neuron bits + per-cluster neurons)
                let genome_offset = genome_idx * num_clusters;
                let neurons_per_cluster = &genomes_neurons_flat[genome_offset..genome_offset + num_clusters];

                // Extract per-neuron bits for this genome
                let bpn_start = genome_bpn_offsets[genome_idx];
                let bpn_end = genome_bpn_offsets[genome_idx + 1];
                let per_neuron_bits = &genomes_bits_flat[bpn_start..bpn_end];

                // Compute per-cluster max bits (for build_groups and GPU dispatch)
                let bits_per_cluster = per_cluster_max_bits(per_neuron_bits, neurons_per_cluster);

                // Build neuron metadata for per-neuron training
                let (cluster_neuron_starts, neuron_conn_offsets) =
                    build_neuron_metadata(per_neuron_bits, neurons_per_cluster);

                // Build config groups for THIS genome (using per-cluster max bits)
                let groups = build_groups(&bits_per_cluster, neurons_per_cluster);

                // Build cluster-to-group mapping for THIS genome
                let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
                for (group_idx, group) in groups.iter().enumerate() {
                    for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
                        cluster_to_group[cluster_id] = (group_idx, local_idx);
                    }
                }

                // Create memory for THIS genome
                let memories: Vec<GroupMemory> = groups.iter()
                    .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                    .collect();

                // Get original per-neuron connections for this genome
                let original_connections: Vec<i64> = if use_provided_connections {
                    let conn_offset = conn_offsets[genome_idx];
                    let conn_size = conn_sizes[genome_idx];
                    genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec()
                } else {
                    // Generate random per-neuron connections
                    use rand::{Rng, SeedableRng};
                    let mut rng = rand::rngs::SmallRng::seed_from_u64((genome_idx * 12345) as u64);
                    let total_conn: usize = per_neuron_bits.iter().sum();
                    let mut conns = Vec::with_capacity(total_conn);
                    for _ in 0..total_conn {
                        conns.push(rng.gen_range(0..total_input_bits as i64));
                    }
                    conns
                };

                // Compute training addresses on GPU (falls back to CPU if unavailable)
                let gpu_addresses = try_gpu_addresses_adaptive(
                    &packed_train_input,
                    words_per_example,
                    per_neuron_bits,
                    &neuron_conn_offsets,
                    &original_connections,
                    num_train,
                );

                // Train this genome using per-neuron bits
                train_genome_in_slot(
                    &memories,
                    &groups,
                    &original_connections,
                    per_neuron_bits,
                    &cluster_neuron_starts,
                    &neuron_conn_offsets,
                    &cluster_to_group,
                    train_input_bits,
                    train_targets,
                    train_negatives,
                    num_train,
                    num_negatives,
                    total_input_bits,
                    gpu_addresses.as_deref(),
                    neuron_sample_rate,
                    rng_seed.wrapping_add(genome_idx as u64),
                    memory_mode,
                    class_weights,
                    true, // parallel: rayon handles nested parallelism via work-stealing
                );

                // Build GPU-padded connections (per-neuron → group layout with padding)
                let gpu_connections = reorganize_connections_for_gpu(
                    &original_connections,
                    per_neuron_bits,
                    neurons_per_cluster,
                    &groups,
                );

                // Export for GPU using THIS genome's groups
                let export = export_genome_for_gpu(&memories, &groups, &gpu_connections);

                // Threshold calibration done AFTER par_iter to avoid nested parallelism
                (genome_idx, export, None)
            })
            .collect();

        // Single-cluster: calibrate thresholds sequentially (compute_per_example_scores
        // uses par_iter internally, which would deadlock inside the outer par_iter above)
        if num_clusters == 1 {
            // Check for threshold override (set by evaluate_genomes_parallel_hybrid_with_override)
            let override_t: Option<f64> = std::env::var("WNN_OVERRIDE_THRESHOLD")
                .ok().and_then(|v| v.parse().ok());

            if let Some(t) = override_t {
                if t < 0.0 {
                    // -1.0: don't set override, let evaluate_genome_hybrid find on eval data
                    // (batch_exports already have None as threshold)
                } else {
                    // Fixed threshold (e.g., 0.5)
                    for (_, _, threshold) in batch_exports.iter_mut() {
                        *threshold = Some(t);
                    }
                }
            } else {
                // Default: calibrate on training data
                let metal_arc = get_metal_evaluator();
                let sparse_metal_arc = get_sparse_metal_evaluator();
                for (_, export, threshold) in batch_exports.iter_mut() {
                    let train_scores = compute_per_example_scores(
                        export, train_input_bits, &packed_train_input, words_per_example,
                        num_train, num_clusters, total_input_bits, empty_value,
                        memory_mode,
                        metal_arc.as_ref().map(|a| a.as_ref()),
                        sparse_metal_arc.as_ref().map(|a| a.as_ref()),
                    );
                    let flat_scores: Vec<f64> = train_scores.iter().map(|s| s[0]).collect();
                    let (t, _f1, _fpr) = find_optimal_threshold_auto(&flat_scores, train_targets);
                    *threshold = Some(t);
                }
            }
        }

        let train_elapsed = train_start.elapsed();

        // Track sparse export sizes for timing diagnostics
        let sparse_keys_total: usize = if timing_enabled {
            batch_exports.iter()
                .map(|(_, export, _)| export.sparse_exports.iter().map(|se| se.keys.len()).sum::<usize>())
                .sum()
        } else { 0 };

        let eval_start = std::time::Instant::now();

        // Evaluate inline (eval worker thread causes rayon deadlock when
        // compute_per_example_scores tries to use par_iter from a non-rayon thread)
        let metal_arc = get_metal_evaluator();
        let sparse_metal_arc = get_sparse_metal_evaluator();
        let metal_ref = metal_arc.as_ref().map(|a| a.as_ref());
        let sparse_metal_ref = sparse_metal_arc.as_ref().map(|a| a.as_ref());
        let batch_results: Vec<(usize, f64, f64, f64, f64, f64)> = batch_exports
            .into_iter()
            .map(|(genome_idx, export, override_threshold)| {
                let (ce, acc, f1, fpr, threshold) = evaluate_genome_hybrid(
                    &export,
                    &eval_data.eval_input_bits,
                    &eval_data.eval_targets,
                    eval_data.num_eval,
                    eval_data.num_clusters,
                    eval_data.total_input_bits,
                    eval_data.empty_value,
                    metal_ref,
                    sparse_metal_ref,
                    override_threshold,
                );
                (genome_idx, ce, acc, f1, fpr, threshold)
            })
            .collect();

        let eval_elapsed_secs = eval_start.elapsed().as_secs_f64();
        let batch_total_secs = train_elapsed.as_secs_f64() + eval_elapsed_secs;

        // Log results with CE/Acc after batch completes
        if progress_log {
            use std::io::Write;
            let now = chrono::Local::now();
            let gen_width = total_gens.to_string().len();
            let pos_width = total_count.to_string().len();
            let type_padded = format!("{:<4}", &log_type[..log_type.len().min(4)]);

            for (genome_idx, ce, acc, _f1, _fpr, _threshold) in &batch_results {
                let overall_position = batch_offset + genome_idx + 1;
                let msg = format!(
                    "{} | [Gen {:0gen_width$}/{:0gen_width$}] Genome {:0pos_width$}/{} ({}): CE={:.4}, Acc={:.4}% ({:.1}s)\n",
                    now.format("%H:%M:%S"),
                    current_gen, total_gens,
                    overall_position, total_count,
                    type_padded,
                    ce, acc * 100.0,
                    batch_total_secs,
                    gen_width = gen_width,
                    pos_width = pos_width,
                );
                if let Some(ref path) = log_path {
                    if let Ok(mut file) = std::fs::OpenOptions::new().append(true).open(path) {
                        let _ = file.write_all(msg.as_bytes());
                        let _ = file.flush();
                    }
                } else {
                    eprint!("{}", msg);
                }
            }
        }

        all_results.extend(batch_results);

        if timing_enabled {
            total_train_ms += train_elapsed.as_millis();
            total_eval_ms += (eval_elapsed_secs * 1000.0) as u128;
            total_sparse_keys += sparse_keys_total;
        }
    }

    // Print timing summary if enabled
    if timing_enabled && num_genomes > 0 {
        let train_per_genome = total_train_ms as f64 / num_genomes as f64;
        let eval_per_genome = total_eval_ms as f64 / num_genomes as f64;
        let sparse_per_genome = total_sparse_keys as f64 / num_genomes as f64;
        eprintln!(
            "[TIMING] batch_size={}, genomes={}: train={:.0}ms/genome, eval={:.0}ms/genome, total={:.0}ms/genome, sparse_keys={:.0}/genome",
            batch_size, num_genomes, train_per_genome, eval_per_genome, train_per_genome + eval_per_genome, sparse_per_genome
        );
    }

    // Sort results by genome index and return
    let mut results: Vec<(f64, f64, f64, f64, f64)> = vec![(0.0, 0.0, 0.0, 0.0, 0.5); num_genomes];
    for (genome_idx, ce, acc, f1, fpr, threshold) in all_results {
        results[genome_idx] = (ce, acc, f1, fpr, threshold);
    }

    results
}

/// Same as evaluate_genomes_parallel_hybrid but with optional threshold override.
/// - None: train-calibrated threshold (58d57c6, current default)
/// - Some(-1.0): find optimal on EVAL data (302a36d = data leakage test)
/// - Some(t >= 0): fixed threshold (e.g. 0.5 for 5aa659d)
///
/// Works by running the standard pipeline then overriding the threshold in batch_exports
/// before the inline eval step. For Some(-1.0), passes None to evaluate_genome_hybrid
/// which triggers its internal eval-data threshold sweep.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genomes_parallel_hybrid_with_override(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    class_weights: Option<&[u32]>,
    override_threshold: Option<f64>,
) -> Vec<(f64, f64, f64, f64, f64)> {
    if override_threshold.is_none() {
        return evaluate_genomes_parallel_hybrid(
            genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
            num_genomes, num_clusters,
            train_input_bits, train_targets, train_negatives,
            num_train, num_negatives,
            eval_input_bits, eval_targets, num_eval,
            total_input_bits, empty_value, neuron_sample_rate, rng_seed,
            class_weights,
        );
    }

    // Set OVERRIDE_THRESHOLD env var so the inline eval in evaluate_genomes_parallel_hybrid
    // can pick it up. This is a hack but avoids duplicating 300 lines of code.
    let t = override_threshold.unwrap();
    let env_key = "WNN_OVERRIDE_THRESHOLD";
    std::env::set_var(env_key, format!("{}", t));
    let results = evaluate_genomes_parallel_hybrid(
        genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
        num_genomes, num_clusters,
        train_input_bits, train_targets, train_negatives,
        num_train, num_negatives,
        eval_input_bits, eval_targets, num_eval,
        total_input_bits, empty_value, neuron_sample_rate, rng_seed,
        class_weights,
    );
    std::env::remove_var(env_key);
    results
}

// =============================================================================
// Training-Time Adaptation for Adaptive Path (*genesis)
// =============================================================================

/// Result from adaptive evaluation, including (possibly modified) genome parameters.
pub struct AdaptiveGenomeResult {
    pub ce: f64,
    pub accuracy: f64,
    pub f1_macro: f64,
    pub fpr: f64,
    pub adapted_bits: Vec<usize>,
    pub adapted_neurons: Vec<usize>,
    pub adapted_connections: Vec<i64>,
    pub pruned: usize,
    pub grown: usize,
    pub added: usize,
    pub removed: usize,
    pub rewired: usize,
}

/// Compute NeuronStats from adaptive training state (GroupMemory-based).
///
/// For each neuron: fill_rate (from memory), connection_entropy (from input bits),
/// and error_rate (forward pass on sample checking target/negative agreement).
pub(crate) fn compute_neuron_stats_adaptive(
    per_neuron_bits: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    cluster_to_group: &[(usize, usize)],
    cluster_neuron_starts: &[usize],
    neuron_conn_offsets: &[usize],
    packed_input: &[u64],
    words_per_example: usize,
    train_targets: &[i64],
    num_train: usize,
    num_clusters: usize,
    config: &crate::adaptation::AdaptationConfig,
    memory_mode: u8,
    empty_value: f32,
) -> Vec<crate::adaptation::NeuronStats> {
    let total_neurons: usize = neurons_per_cluster.iter().sum();

    // Sample examples for stats
    let sample_indices = crate::adaptation::build_sample_indices(num_train, config.stats_sample_size, 42);
    let n_sample = sample_indices.len();

    // Precompute bit ones for connection entropy
    let bit_ones = crate::adaptation::precompute_bit_ones(
        packed_input, words_per_example, &sample_indices, config.total_input_bits,
    );

    // Build neuron-to-cluster mapping
    let mut neuron_to_cluster = vec![0usize; total_neurons];
    for cluster in 0..num_clusters {
        let base = cluster_neuron_starts[cluster];
        for local_n in 0..neurons_per_cluster[cluster] {
            neuron_to_cluster[base + local_n] = cluster;
        }
    }

    // Parallel per-neuron stats computation
    (0..total_neurons).into_par_iter().map(|global_n| {
        let n_bits = per_neuron_bits[global_n];
        let conn_start = neuron_conn_offsets[global_n];
        let neuron_conns = &connections[conn_start..conn_start + n_bits];
        let cluster = neuron_to_cluster[global_n];

        // Connection entropy (from precomputed bit counts)
        let conn_entropy: Vec<f32> = neuron_conns.iter().map(|&conn_idx| {
            let ones = bit_ones[conn_idx as usize];
            let p = ones as f32 / n_sample.max(1) as f32;
            if p > 0.0 && p < 1.0 {
                -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
            } else {
                0.0
            }
        }).collect();

        // Fill rate from GroupMemory
        let (group_idx, local_cluster) = cluster_to_group[cluster];
        let local_n = global_n - cluster_neuron_starts[cluster];
        let neuron_in_group = local_cluster * groups[group_idx].neurons + local_n;
        let fill_rate = memories[group_idx].neuron_fill_rate(neuron_in_group, n_bits);

        // Error rate: check neuron output against expected label on sampled examples.
        // For examples where this neuron's cluster is the target: expect TRUE.
        // For all other examples: expect FALSE (this neuron's cluster is a negative).
        let mut errors = 0u32;
        let mut seen = 0u32;
        for &ex in &sample_indices {
            let target = train_targets[ex] as usize;
            let expected_true = target == cluster;

            // Compute address from packed input
            let mut addr = 0usize;
            for (bit_i, &conn_idx) in neuron_conns.iter().enumerate() {
                let idx = conn_idx as usize;
                let word_idx = ex * words_per_example + idx / 64;
                let bit_idx = idx % 64;
                if packed_input[word_idx] >> bit_idx & 1 == 1 {
                    addr |= 1 << (n_bits - 1 - bit_i);
                }
            }

            let cell = memories[group_idx].read(neuron_in_group, addr);
            let weight = cell_to_weight(cell, memory_mode, empty_value);
            let predicted_true = weight >= 0.5;

            if predicted_true != expected_true {
                errors += 1;
            }
            seen += 1;
        }
        let error_rate = if seen > 0 { errors as f32 / seen as f32 } else { 0.0 };

        crate::adaptation::NeuronStats { fill_rate, error_rate, connection_entropy: conn_entropy }
    }).collect()
}

/// Compute ClusterStats from adaptive training state.
///
/// For each cluster: error_rate (majority vote vs target), mean_fill_rate,
/// neuron_uniqueness (disagrees with majority), neuron_accuracy (matches target).
pub(crate) fn compute_cluster_stats_adaptive(
    neuron_stats: &[crate::adaptation::NeuronStats],
    per_neuron_bits: &[usize],
    neurons_per_cluster: &[usize],
    connections: &[i64],
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    cluster_to_group: &[(usize, usize)],
    cluster_neuron_starts: &[usize],
    neuron_conn_offsets: &[usize],
    packed_input: &[u64],
    words_per_example: usize,
    train_targets: &[i64],
    num_train: usize,
    num_clusters: usize,
    config: &crate::adaptation::AdaptationConfig,
    memory_mode: u8,
    empty_value: f32,
) -> Vec<crate::adaptation::ClusterStats> {
    let sample_indices = crate::adaptation::build_sample_indices(num_train, config.stats_sample_size, 43);
    let n_sample = sample_indices.len();

    (0..num_clusters).into_par_iter().map(|cluster| {
        let n_neurons = neurons_per_cluster[cluster];
        let neuron_base = cluster_neuron_starts[cluster];
        let (group_idx, local_cluster) = cluster_to_group[cluster];
        let group = &groups[group_idx];
        let memory = &memories[group_idx];

        // Mean fill rate from pre-computed neuron stats
        let mean_fill = if n_neurons > 0 {
            (0..n_neurons).map(|i| neuron_stats[neuron_base + i].fill_rate).sum::<f32>() / n_neurons as f32
        } else {
            0.0
        };

        // Per-example: compute neuron votes, check cluster majority
        let mut cluster_errors = 0u32;
        let mut neuron_votes: Vec<Vec<bool>> = vec![Vec::with_capacity(n_sample); n_neurons];

        for &ex in &sample_indices {
            let target = train_targets[ex] as usize;
            let target_is_this_cluster = target == cluster;
            let mut votes_true = 0u32;

            for local_n in 0..n_neurons {
                let global_n = neuron_base + local_n;
                let n_bits = per_neuron_bits[global_n];
                let conn_start = neuron_conn_offsets[global_n];
                let neuron_conns = &connections[conn_start..conn_start + n_bits];
                let neuron_in_group = local_cluster * group.neurons + local_n;

                let mut addr = 0usize;
                for (bit_i, &conn_idx) in neuron_conns.iter().enumerate() {
                    let idx = conn_idx as usize;
                    let word_idx = ex * words_per_example + idx / 64;
                    let bit_idx = idx % 64;
                    if packed_input[word_idx] >> bit_idx & 1 == 1 {
                        addr |= 1 << (n_bits - 1 - bit_i);
                    }
                }

                let cell = memory.read(neuron_in_group, addr);
                let weight = cell_to_weight(cell, memory_mode, empty_value);
                let is_true = weight >= 0.5;
                if is_true { votes_true += 1; }
                neuron_votes[local_n].push(is_true);
            }

            // Majority vote for cluster
            let majority_true = votes_true > (n_neurons as u32 / 2);
            if majority_true != target_is_this_cluster { cluster_errors += 1; }
        }

        let cluster_error_rate = if n_sample > 0 {
            cluster_errors as f32 / n_sample as f32
        } else { 0.0 };

        // Precompute majority per sample
        let majority_votes: Vec<bool> = (0..n_sample).map(|s| {
            let v: u32 = (0..n_neurons).map(|j| neuron_votes[j][s] as u32).sum();
            v > (n_neurons as u32 / 2)
        }).collect();

        // Neuron uniqueness + accuracy
        let mut uniqueness = vec![0.0f32; n_neurons];
        let mut accuracy = vec![0.0f32; n_neurons];

        if n_sample > 0 {
            for local_n in 0..n_neurons {
                let mut disagree = 0u32;
                let mut correct = 0u32;
                for s in 0..n_sample {
                    let ex = sample_indices[s];
                    let my_vote = neuron_votes[local_n][s];
                    let target_is_this_cluster = train_targets[ex] as usize == cluster;
                    if my_vote != majority_votes[s] { disagree += 1; }
                    if my_vote == target_is_this_cluster { correct += 1; }
                }
                uniqueness[local_n] = disagree as f32 / n_sample as f32;
                accuracy[local_n] = correct as f32 / n_sample as f32;
            }
        }

        crate::adaptation::ClusterStats {
            error_rate: cluster_error_rate,
            mean_fill_rate: mean_fill,
            neuron_uniqueness: uniqueness,
            neuron_accuracy: accuracy,
        }
    }).collect()
}

/// Axonogenesis for the adaptive (GroupMemory) path.
///
/// Rewires low-value connections to high-entropy input bits using the same 3-stage
/// algorithm as `adaptation::axonogenesis_pass`, but reads from GroupMemory instead
/// of ClusterStorage and uses class-label targets instead of target_bits.
///
/// Returns the number of rewired connections.
#[allow(clippy::too_many_arguments)]
pub(crate) fn axonogenesis_pass_adaptive(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    connections: &mut Vec<i64>,
    neuron_stats: &[crate::adaptation::NeuronStats],
    memories: &[GroupMemory],
    groups: &[ConfigGroup],
    cluster_to_group: &[(usize, usize)],
    _cluster_neuron_starts: &[usize],
    config: &crate::adaptation::AdaptationConfig,
    packed_input: &[u64],
    words_per_example: usize,
    train_targets: &[i64],
    num_examples: usize,
    _num_clusters: usize,
    _memory_mode: u8,
    _empty_value: f32,
    rate: f32,
    rng: &mut impl rand::Rng,
) -> usize {
    use crate::adaptation::{build_sample_indices, precompute_bit_ones};

    let total_neurons = bits_per_neuron.len();
    let mut rewired = 0usize;
    let max_candidates = 20usize;

    // Build connection offsets per neuron
    let mut conn_offsets = vec![0usize];
    for &b in bits_per_neuron.iter() {
        conn_offsets.push(conn_offsets.last().unwrap() + b);
    }

    // Build neuron → cluster mapping
    let mut neuron_cluster = Vec::with_capacity(total_neurons);
    let mut neuron_local_idx = Vec::with_capacity(total_neurons);
    for (c, &nc) in neurons_per_cluster.iter().enumerate() {
        for local in 0..nc {
            neuron_cluster.push(c);
            neuron_local_idx.push(local);
        }
    }

    // Stage 1: Pre-compute marginal entropy for all input bits
    let sample_indices = build_sample_indices(num_examples, config.stats_sample_size, 77);
    let n_sample = sample_indices.len();
    let bit_ones = precompute_bit_ones(packed_input, words_per_example, &sample_indices, config.total_input_bits);
    let bit_entropy: Vec<f32> = bit_ones.iter().map(|&ones| {
        let p = ones as f32 / n_sample.max(1) as f32;
        if p > 0.0 && p < 1.0 {
            -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) / std::f32::consts::LN_2
        } else {
            0.0
        }
    }).collect();

    let entropy_floor = 0.1f32;

    // Pre-compute bit values for sampled examples (for Stage 3 redundancy)
    let mut bit_vals: Vec<Option<Vec<u8>>> = vec![None; config.total_input_bits];
    for bit_idx in 0..config.total_input_bits {
        if bit_entropy[bit_idx] >= entropy_floor {
            let mut vals = Vec::with_capacity(n_sample);
            for &ex in &sample_indices {
                let row_start = ex * words_per_example;
                let word_idx = bit_idx / 64;
                let bit_pos = bit_idx % 64;
                let bit = ((packed_input[row_start + word_idx] >> bit_pos) & 1) as u8;
                vals.push(bit);
            }
            bit_vals[bit_idx] = Some(vals);
        }
    }

    // Process each neuron
    for n in 0..total_neurons {
        let n_bits = bits_per_neuron[n];
        if n_bits < 2 { continue; }

        let stats = &neuron_stats[n];
        let conn_start = conn_offsets[n];
        let cluster = neuron_cluster[n];
        let local_n = neuron_local_idx[n];

        // Resolve GroupMemory location for this neuron
        let (group_idx, local_cluster) = cluster_to_group[cluster];
        let group = &groups[group_idx];
        let memory = &memories[group_idx];
        let neuron_in_group = local_cluster * group.neurons + local_n;

        // Stage 1a: Find weak connections (entropy < median × threshold)
        let median_ent = crate::adaptation::median_of(&stats.connection_entropy);
        let rewire_threshold = median_ent * config.axon_entropy_threshold;

        let mut weak_conns: Vec<(usize, f32)> = stats.connection_entropy.iter()
            .enumerate()
            .filter(|(_, &ent)| ent < rewire_threshold)
            .map(|(idx, &ent)| (idx, ent))
            .collect();
        weak_conns.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        weak_conns.truncate(config.axon_rewire_count);

        if weak_conns.is_empty() { continue; }

        // Stage 1b: Get candidate unused bits sorted by marginal entropy (desc)
        let used: std::collections::HashSet<i64> = connections[conn_start..conn_start + n_bits]
            .iter().copied().collect();

        let mut candidates: Vec<(i64, f32)> = (0..config.total_input_bits)
            .filter(|&b| !used.contains(&(b as i64)) && bit_entropy[b] >= entropy_floor)
            .map(|b| (b as i64, bit_entropy[b]))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_candidates);

        if candidates.is_empty() { continue; }

        // Pre-compute base addresses for this neuron on sampled examples
        let neuron_conns = &connections[conn_start..conn_start + n_bits];
        let mut base_addresses: Vec<usize> = Vec::with_capacity(n_sample);
        for &ex in &sample_indices {
            let row_start = ex * words_per_example;
            let mut addr = 0usize;
            for (i, &conn_idx) in neuron_conns.iter().enumerate() {
                let idx = conn_idx as usize;
                let bit = (packed_input[row_start + idx / 64] >> (idx % 64)) & 1;
                addr |= (bit as usize) << (n_bits - 1 - i);
            }
            base_addresses.push(addr);
        }

        // Stage 3 prep: existing connection bit values
        let existing_bit_vals: Vec<Option<&Vec<u8>>> = (0..n_bits)
            .map(|i| {
                let conn = neuron_conns[i] as usize;
                if conn < config.total_input_bits { bit_vals[conn].as_ref() } else { None }
            })
            .collect();

        // For each weak connection, find the best replacement via accuracy delta + redundancy
        for &(local_idx, _old_entropy) in &weak_conns {
            if rng.gen::<f32>() >= rate { continue; }

            let old_conn = connections[conn_start + local_idx];
            let mut best_candidate: Option<(i64, f32)> = None;

            for &(cand_conn, _cand_entropy) in &candidates {
                if connections[conn_start..conn_start + n_bits].contains(&cand_conn) { continue; }

                // Stage 2: Accuracy delta — measure per-neuron accuracy from swapping
                let mut delta = 0i32;
                for (si, &ex) in sample_indices.iter().enumerate() {
                    let old_addr = base_addresses[si];
                    // Target: is this cluster the correct one for this example?
                    let target_is_this = train_targets[ex] as usize == cluster;

                    // Old prediction (current connection)
                    let old_cell = memory.read(neuron_in_group, old_addr);
                    let old_weight = crate::neuron_memory::QUAD_WEIGHTS[old_cell.clamp(0, 3) as usize];
                    let old_correct = (old_weight >= 0.5) == target_is_this;

                    // Flip address bit at local_idx if old/new input bits differ
                    let row_start = ex * words_per_example;
                    let old_bit = (packed_input[row_start + old_conn as usize / 64]
                        >> (old_conn as usize % 64)) & 1;
                    let new_bit = (packed_input[row_start + cand_conn as usize / 64]
                        >> (cand_conn as usize % 64)) & 1;
                    let new_addr = if old_bit != new_bit {
                        old_addr ^ (1 << (n_bits - 1 - local_idx))
                    } else {
                        old_addr
                    };

                    // New prediction
                    let new_cell = memory.read(neuron_in_group, new_addr);
                    let new_weight = crate::neuron_memory::QUAD_WEIGHTS[new_cell.clamp(0, 3) as usize];
                    let new_correct = (new_weight >= 0.5) == target_is_this;

                    delta += new_correct as i32 - old_correct as i32;
                }

                if delta <= 0 { continue; }

                // Stage 3: Redundancy penalty (Jaccard similarity)
                let mut max_jaccard = 0.0f32;
                if let Some(cand_vals) = &bit_vals[cand_conn as usize] {
                    for (i, existing) in existing_bit_vals.iter().enumerate() {
                        if i == local_idx { continue; }
                        if let Some(existing_vals) = existing {
                            let mut both = 0u32;
                            let mut either = 0u32;
                            for si in 0..n_sample {
                                let a = cand_vals[si];
                                let b = existing_vals[si];
                                both += (a & b) as u32;
                                either += (a | b) as u32;
                            }
                            let jaccard = if either > 0 { both as f32 / either as f32 } else { 0.0 };
                            if jaccard > max_jaccard { max_jaccard = jaccard; }
                        }
                    }
                }

                let redundancy_weight = 0.5f32;
                let adjusted_score = delta as f32 * (1.0 - redundancy_weight * max_jaccard);

                if best_candidate.is_none() || adjusted_score > best_candidate.unwrap().1 {
                    best_candidate = Some((cand_conn, adjusted_score));
                }
            }

            if let Some((new_conn, score)) = best_candidate {
                if score > 0.0 {
                    connections[conn_start + local_idx] = new_conn;
                    rewired += 1;
                }
            }
        }
    }

    if rewired > 0 {
        eprintln!("[Adapt] Axonogenesis: rewired={} (rate={:.2})", rewired, rate);
    }

    rewired
}

/// Evaluate genomes with training-time adaptation (synaptogenesis + neurogenesis + axonogenesis).
///
/// Same interface as `evaluate_genomes_parallel_hybrid` but:
/// 1. After training, computes stats from trained memory
/// 2. Applies adaptation passes (modifying genome architecture)
/// 3. Retrains if genome was modified
/// 4. Returns both scores AND adapted genome parameters
///
/// If adaptation_rate is 0.0 for the given generation (warmup/cooldown), falls
/// through to the standard path with no overhead.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genomes_parallel_hybrid_adaptive(
    genomes_bits_flat: &[usize],
    genomes_neurons_flat: &[usize],
    genomes_connections_flat: &[i64],
    num_genomes: usize,
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neuron_sample_rate: f32,
    rng_seed: u64,
    adapt_config: &crate::adaptation::AdaptationConfig,
    generation: usize,
) -> Vec<AdaptiveGenomeResult> {
    let rate = crate::adaptation::adaptation_rate(generation, adapt_config);

    // If rate is 0 (warmup/cooldown), use standard path and wrap results
    if rate == 0.0 || (!adapt_config.synaptogenesis_enabled && !adapt_config.neurogenesis_enabled) {
        let standard = evaluate_genomes_parallel_hybrid(
            genomes_bits_flat, genomes_neurons_flat, genomes_connections_flat,
            num_genomes, num_clusters,
            train_input_bits, train_targets, train_negatives,
            num_train, num_negatives,
            eval_input_bits, eval_targets, num_eval,
            total_input_bits, empty_value, neuron_sample_rate, rng_seed,
            None, // class_weights: adaptive path doesn't use class balancing
        );

        // Rebuild per-genome bits/neurons/connections from flat arrays
        let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
        genome_bpn_offsets.push(0);
        for g in 0..num_genomes {
            let nc_base = g * num_clusters;
            let total_n: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
            genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_n);
        }
        let mut conn_offset = 0usize;

        return standard.into_iter().enumerate().map(|(g, (ce, acc, f1, fpr, _threshold))| {
            let bpn_start = genome_bpn_offsets[g];
            let bpn_end = genome_bpn_offsets[g + 1];
            let bits = genomes_bits_flat[bpn_start..bpn_end].to_vec();
            let neurons = genomes_neurons_flat[g * num_clusters..(g + 1) * num_clusters].to_vec();
            let conn_size: usize = bits.iter().sum();
            let conns = genomes_connections_flat[conn_offset..conn_offset + conn_size].to_vec();
            conn_offset += conn_size;
            AdaptiveGenomeResult {
                ce, accuracy: acc, f1_macro: f1, fpr,
                adapted_bits: bits, adapted_neurons: neurons, adapted_connections: conns,
                pruned: 0, grown: 0, added: 0, removed: 0, rewired: 0,
            }
        }).collect();
    }

    let memory_mode = crate::neuron_memory::get_memory_mode();
    if num_genomes == 0 {
        return vec![];
    }

    // Pre-compute genome offsets (same as evaluate_genomes_parallel_hybrid)
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let total_neurons: usize = genomes_neurons_flat[nc_base..nc_base + num_clusters].iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    let use_provided_connections = !genomes_connections_flat.is_empty();
    let mut conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut conn_sizes: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut running_offset = 0usize;
    for genome_idx in 0..num_genomes {
        conn_offsets.push(running_offset);
        let bpn_start = genome_bpn_offsets[genome_idx];
        let bpn_end = genome_bpn_offsets[genome_idx + 1];
        let conn_size: usize = genomes_bits_flat[bpn_start..bpn_end].iter().sum();
        conn_sizes.push(conn_size);
        running_offset += conn_size;
    }

    // Pack input once
    let (packed_train_input, words_per_example) =
        crate::neuron_memory::pack_bools_to_u64(train_input_bits, num_train, total_input_bits);

    let eval_data = Arc::new(EvalData {
        eval_input_bits: eval_input_bits.to_vec(),
        eval_targets: eval_targets.to_vec(),
        num_eval,
        num_clusters,
        total_input_bits,
        empty_value,
    });
    let eval_worker = get_eval_worker();

    // Process genomes: train → adapt → retrain if needed → export → evaluate
    // Use sequential processing to keep memory bounded (adaptation keeps memory alive longer)
    let batch_size = std::env::var("WNN_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            let first_neurons = &genomes_neurons_flat[0..num_clusters];
            let first_per_neuron_bits = &genomes_bits_flat[0..genome_bpn_offsets[1]];
            let first_bits_per_cluster = per_cluster_max_bits(first_per_neuron_bits, first_neurons);
            let budget_gb = get_available_memory_gb() * 0.4; // 40% budget (adaptation keeps memory longer)
            let cpu_cores = rayon::current_num_threads();
            let (_, computed_batch) = calculate_pool_size(
                &first_bits_per_cluster, first_neurons, num_clusters, budget_gb, cpu_cores,
            );
            computed_batch
        });

    let mut all_results: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::with_capacity(num_genomes);
    let mut all_adapted: Vec<(usize, Vec<usize>, Vec<usize>, Vec<i64>, usize, usize, usize, usize, usize)> =
        Vec::with_capacity(num_genomes);
    let num_batches = (num_genomes + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * batch_size;
        let batch_end = (batch_start + batch_size).min(num_genomes);
        let current_batch_size = batch_end - batch_start;

        // Phase 1: Train all genomes in parallel, keeping memory alive
        struct TrainedState {
            memories: Vec<GroupMemory>,
            groups: Vec<ConfigGroup>,
            cluster_to_group: Vec<(usize, usize)>,
            cluster_neuron_starts: Vec<usize>,
            neuron_conn_offsets: Vec<usize>,
            per_neuron_bits: Vec<usize>,
            neurons_per_cluster: Vec<usize>,
            connections: Vec<i64>,
            genome_idx: usize,
        }

        let trained_states: Vec<TrainedState> = (0..current_batch_size)
            .into_par_iter()
            .map(|local_idx| {
                let genome_idx = batch_start + local_idx;
                let genome_offset = genome_idx * num_clusters;
                let neurons_per_cluster = genomes_neurons_flat[genome_offset..genome_offset + num_clusters].to_vec();
                let bpn_start = genome_bpn_offsets[genome_idx];
                let bpn_end = genome_bpn_offsets[genome_idx + 1];
                let per_neuron_bits = genomes_bits_flat[bpn_start..bpn_end].to_vec();
                let bits_per_cluster = per_cluster_max_bits(&per_neuron_bits, &neurons_per_cluster);
                let (cluster_neuron_starts, neuron_conn_offsets) =
                    build_neuron_metadata(&per_neuron_bits, &neurons_per_cluster);
                let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);
                let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
                for (gi, group) in groups.iter().enumerate() {
                    for (li, &cid) in group.cluster_ids.iter().enumerate() {
                        cluster_to_group[cid] = (gi, li);
                    }
                }
                let memories: Vec<GroupMemory> = groups.iter()
                    .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                    .collect();
                let connections: Vec<i64> = if use_provided_connections {
                    let co = conn_offsets[genome_idx];
                    let cs = conn_sizes[genome_idx];
                    genomes_connections_flat[co..co + cs].to_vec()
                } else {
                    use rand::{Rng, SeedableRng};
                    let mut rng = rand::rngs::SmallRng::seed_from_u64((genome_idx * 12345) as u64);
                    let total_conn: usize = per_neuron_bits.iter().sum();
                    (0..total_conn).map(|_| rng.gen_range(0..total_input_bits as i64)).collect()
                };
                let gpu_addresses = try_gpu_addresses_adaptive(
                    &packed_train_input, words_per_example,
                    &per_neuron_bits, &neuron_conn_offsets, &connections, num_train,
                );
                train_genome_in_slot(
                    &memories, &groups, &connections, &per_neuron_bits,
                    &cluster_neuron_starts, &neuron_conn_offsets, &cluster_to_group,
                    train_input_bits, train_targets, train_negatives,
                    num_train, num_negatives, total_input_bits,
                    gpu_addresses.as_deref(), neuron_sample_rate,
                    rng_seed.wrapping_add(genome_idx as u64), memory_mode,
                    None, // class_weights: adaptive path doesn't use class balancing
                    true, // parallel: rayon work-stealing handles nested parallelism
                );
                TrainedState {
                    memories, groups, cluster_to_group, cluster_neuron_starts,
                    neuron_conn_offsets, per_neuron_bits, neurons_per_cluster,
                    connections, genome_idx,
                }
            })
            .collect();

        // Phase 2: Stats + Adaptation (per genome, sequential — lightweight)
        struct AdaptedState {
            bits: Vec<usize>,
            neurons: Vec<usize>,
            connections: Vec<i64>,
            changed: bool,
            pruned: usize,
            grown: usize,
            added: usize,
            removed: usize,
            rewired: usize,
            genome_idx: usize,
        }

        let mut adapted_states: Vec<AdaptedState> = Vec::with_capacity(current_batch_size);

        for state in &trained_states {
            let mut adapt_bits = state.per_neuron_bits.clone();
            let mut adapt_neurons = state.neurons_per_cluster.clone();
            let mut adapt_conns = state.connections.clone();
            let initial_neurons = state.neurons_per_cluster.clone();
            let mut cooldowns = vec![0usize; num_clusters];
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
                rng_seed.wrapping_add(state.genome_idx as u64 * 777)
            );
            let mut total_pruned = 0usize;
            let mut total_grown = 0usize;
            let mut total_added = 0usize;
            let mut total_removed = 0usize;
            let mut total_rewired = 0usize;

            // Compute stats from ORIGINAL trained state (not adapted copies).
            // Stats must match the memory architecture — adapted bits/conns may have
            // different sizes than what the memory was trained with.
            let neuron_stats = compute_neuron_stats_adaptive(
                &state.per_neuron_bits, &state.neurons_per_cluster, &state.connections,
                &state.memories, &state.groups, &state.cluster_to_group,
                &state.cluster_neuron_starts, &state.neuron_conn_offsets,
                &packed_train_input, words_per_example,
                train_targets, num_train, num_clusters,
                adapt_config, memory_mode, empty_value,
            );

            let cluster_stats = if adapt_config.neurogenesis_enabled {
                Some(compute_cluster_stats_adaptive(
                    &neuron_stats, &state.per_neuron_bits, &state.neurons_per_cluster,
                    &state.connections,
                    &state.memories, &state.groups, &state.cluster_to_group,
                    &state.cluster_neuron_starts, &state.neuron_conn_offsets,
                    &packed_train_input, words_per_example,
                    train_targets, num_train, num_clusters,
                    adapt_config, memory_mode, empty_value,
                ))
            } else {
                None
            };

            for _pass in 0..adapt_config.passes_per_eval {
                // Synaptogenesis (prune/grow connections)
                if adapt_config.synaptogenesis_enabled {
                    let (p, g) = crate::adaptation::synaptogenesis_pass(
                        &mut adapt_bits, &mut adapt_conns,
                        &neuron_stats, adapt_config,
                        &packed_train_input, words_per_example, num_train,
                        rate, &mut rng,
                    );
                    total_pruned += p;
                    total_grown += g;
                }

                // Neurogenesis (add/remove neurons)
                if adapt_config.neurogenesis_enabled {
                    let cluster_stats = cluster_stats.as_ref().unwrap();
                    let (a, r) = crate::adaptation::neurogenesis_pass(
                        &mut adapt_bits, &mut adapt_neurons, &mut adapt_conns,
                        &cluster_stats, adapt_config,
                        generation, &mut cooldowns, &initial_neurons,
                        num_train, rate, &mut rng,
                    );
                    total_added += a;
                    total_removed += r;
                }

                // Axonogenesis (MI-guided connection rewiring)
                if adapt_config.axonogenesis_enabled {
                    let rw = axonogenesis_pass_adaptive(
                        &adapt_bits, &adapt_neurons, &mut adapt_conns,
                        &neuron_stats, &state.memories, &state.groups,
                        &state.cluster_to_group, &state.cluster_neuron_starts,
                        adapt_config, &packed_train_input, words_per_example,
                        train_targets, num_train, num_clusters,
                        memory_mode, empty_value, rate, &mut rng,
                    );
                    total_rewired += rw;
                }
            }

            let changed = total_pruned > 0 || total_grown > 0 || total_added > 0 || total_removed > 0 || total_rewired > 0;
            adapted_states.push(AdaptedState {
                bits: adapt_bits, neurons: adapt_neurons, connections: adapt_conns,
                changed, pruned: total_pruned, grown: total_grown,
                added: total_added, removed: total_removed, rewired: total_rewired,
                genome_idx: state.genome_idx,
            });
        }

        // Phase 3: Retrain changed genomes + export all (parallel)
        let batch_exports: Vec<(usize, GenomeExport, Option<f64>)> = adapted_states.par_iter()
            .enumerate()
            .map(|(local_idx, adapted)| {
                if adapted.changed {
                    // Rebuild groups and memory with adapted architecture, retrain
                    let bits_per_cluster = per_cluster_max_bits(&adapted.bits, &adapted.neurons);
                    let (cns, nco) = build_neuron_metadata(&adapted.bits, &adapted.neurons);
                    let groups = build_groups(&bits_per_cluster, &adapted.neurons);
                    let mut ctg: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
                    for (gi, group) in groups.iter().enumerate() {
                        for (li, &cid) in group.cluster_ids.iter().enumerate() {
                            ctg[cid] = (gi, li);
                        }
                    }
                    let memories: Vec<GroupMemory> = groups.iter()
                        .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
                        .collect();
                    let gpu_addresses = try_gpu_addresses_adaptive(
                        &packed_train_input, words_per_example,
                        &adapted.bits, &nco, &adapted.connections, num_train,
                    );
                    train_genome_in_slot(
                        &memories, &groups, &adapted.connections, &adapted.bits,
                        &cns, &nco, &ctg,
                        train_input_bits, train_targets, train_negatives,
                        num_train, num_negatives, total_input_bits,
                        gpu_addresses.as_deref(), neuron_sample_rate,
                        rng_seed.wrapping_add(adapted.genome_idx as u64), memory_mode,
                        None, // class_weights: adaptive path doesn't use class balancing
                        true, // parallel: rayon work-stealing handles nested parallelism
                    );
                    let gpu_conns = reorganize_connections_for_gpu(
                        &adapted.connections, &adapted.bits, &adapted.neurons, &groups,
                    );
                    let export = export_genome_for_gpu(&memories, &groups, &gpu_conns);
                    (adapted.genome_idx, export, None)
                } else {
                    // Use Phase 1 trained state directly
                    let state = &trained_states[local_idx];
                    let gpu_conns = reorganize_connections_for_gpu(
                        &state.connections, &state.per_neuron_bits,
                        &state.neurons_per_cluster, &state.groups,
                    );
                    let export = export_genome_for_gpu(&state.memories, &state.groups, &gpu_conns);
                    (state.genome_idx, export, None)
                }
            })
            .collect();

        // Send to eval worker
        let batch_results = eval_worker.evaluate(batch_exports, Arc::clone(&eval_data));
        all_results.extend(batch_results);

        // Collect adapted genome info
        for adapted in adapted_states {
            all_adapted.push((
                adapted.genome_idx,
                adapted.bits, adapted.neurons, adapted.connections,
                adapted.pruned, adapted.grown, adapted.added, adapted.removed, adapted.rewired,
            ));
        }
    }

    // Sort by genome index and build final results
    all_adapted.sort_by_key(|a| a.0);
    let mut score_map: Vec<(f64, f64, f64, f64)> = vec![(0.0, 0.0, 0.0, 0.0); num_genomes];
    for (idx, ce, acc, f1, fpr, _threshold) in all_results {
        score_map[idx] = (ce, acc, f1, fpr);
    }

    all_adapted.into_iter().map(|(idx, bits, neurons, conns, pruned, grown, added, removed, rewired)| {
        let (ce, acc, f1, fpr) = score_map[idx];
        AdaptiveGenomeResult {
            ce, accuracy: acc, f1_macro: f1, fpr,
            adapted_bits: bits, adapted_neurons: neurons, adapted_connections: conns,
            pruned, grown, added, removed, rewired,
        }
    }).collect()
}

/// Evaluate a SINGLE genome WITH gating, returning both gated and non-gated metrics.
///
/// This function:
/// 1. Trains base RAM on training data
/// 2. Trains gating model on training data (target gate = true only for target cluster)
/// 3. Evaluates WITHOUT gating → (ce, acc)
/// 4. Evaluates WITH gating → (gated_ce, gated_acc)
///
/// # Returns
/// (ce, accuracy, gated_ce, gated_accuracy)
#[allow(clippy::too_many_arguments)]
pub fn evaluate_genome_with_gating(
    bits_flat: &[usize],
    neurons_flat: &[usize],
    connections_flat: &[i64],
    num_clusters: usize,
    train_input_bits: &[bool],
    train_targets: &[i64],
    train_negatives: &[i64],
    num_train: usize,
    num_negatives: usize,
    eval_input_bits: &[bool],
    eval_targets: &[i64],
    num_eval: usize,
    total_input_bits: usize,
    empty_value: f32,
    neurons_per_gate: usize,
    bits_per_gate_neuron: usize,
    vote_threshold_frac: f32,
    gating_seed: u64,
) -> (f64, f64, f64, f64) {
    use crate::gating::RAMGating;

    let epsilon = 1e-10f64;
    let memory_mode = crate::neuron_memory::get_memory_mode();

    // ========================================================================
    // Step 1: Train base RAM (same as existing evaluation)
    // ========================================================================

    // Build config groups for this genome
    let bits_per_cluster: Vec<usize> = bits_flat.to_vec();
    let neurons_per_cluster: Vec<usize> = neurons_flat.to_vec();
    let groups = build_groups(&bits_per_cluster, &neurons_per_cluster);

    // Create hybrid memory for each config group
    let group_memories: Vec<GroupMemory> = groups.iter()
        .map(|g| GroupMemory::new(g.total_neurons(), g.bits, memory_mode))
        .collect();

    // Reorganize connections for coalescing if needed
    let connections: Vec<i64> = if use_coalesced_groups() {
        reorganize_connections_for_coalescing(
            connections_flat,
            &bits_per_cluster,
            &neurons_per_cluster,
            &groups,
        )
    } else {
        connections_flat.to_vec()
    };

    // Build cluster-to-group mapping
    let mut cluster_to_group: Vec<(usize, usize)> = vec![(0, 0); num_clusters];
    for (group_idx, group) in groups.iter().enumerate() {
        for (local_idx, &cluster_id) in group.cluster_ids.iter().enumerate() {
            cluster_to_group[cluster_id] = (group_idx, local_idx);
        }
    }

    // Train: iterate over training examples (parallel)
    let use_nudge = memory_mode != crate::neuron_memory::MODE_TERNARY;
    (0..num_train).into_par_iter().for_each(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &train_input_bits[input_start..input_start + total_input_bits];

        let true_cluster = train_targets[ex_idx] as usize;

        // Train positive example
        {
            let (group_idx, local_cluster) = cluster_to_group[true_cluster];
            let group = &groups[group_idx];
            let memory = &group_memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;
            let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

            for n in 0..actual_neurons {
                let conn_start = conn_base + n * group.bits;
                let address = compute_address(input_bits, &connections[conn_start..], group.bits);
                if use_nudge {
                    memory.nudge(neuron_base + n, address, true);
                } else {
                    memory.write(neuron_base + n, address, TRUE, false);
                }
            }
        }

        // Train negative examples
        let neg_start = ex_idx * num_negatives;
        for k in 0..num_negatives {
            let false_cluster = train_negatives[neg_start + k] as usize;
            if false_cluster == true_cluster {
                continue;
            }

            let (group_idx, local_cluster) = cluster_to_group[false_cluster];
            let group = &groups[group_idx];
            let memory = &group_memories[group_idx];

            let actual_neurons = if let Some(ref an) = group.actual_neurons {
                an[local_cluster] as usize
            } else {
                group.neurons
            };

            let neuron_base = local_cluster * group.neurons;
            let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

            for n in 0..actual_neurons {
                let conn_start = conn_base + n * group.bits;
                let address = compute_address(input_bits, &connections[conn_start..], group.bits);
                if use_nudge {
                    memory.nudge(neuron_base + n, address, false);
                } else {
                    memory.write(neuron_base + n, address, FALSE, false);
                }
            }
        }
    });

    // ========================================================================
    // Step 2: Train gating model (parallel)
    // ========================================================================

    let gating = RAMGating::new(
        num_clusters,
        neurons_per_gate,
        bits_per_gate_neuron,
        total_input_bits,
        vote_threshold_frac,
        Some(gating_seed),
    );

    // Build target gates in parallel: for each example, target_gate[target] = true
    let target_gates_flat: Vec<bool> = (0..num_train)
        .into_par_iter()
        .flat_map(|ex_idx| {
            let target = train_targets[ex_idx] as usize;
            let mut gates = vec![false; num_clusters];
            if target < num_clusters {
                gates[target] = true;
            }
            gates
        })
        .collect();

    // Train gating using parallel batch training
    gating.train_batch(train_input_bits, &target_gates_flat, num_train, false);

    // ========================================================================
    // Step 3: Evaluate WITHOUT gating - pre-compute all scores
    // ========================================================================

    let all_scores: Vec<Vec<f64>> = (0..num_eval).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];

        let mut scores = vec![0.0f64; num_clusters];

        for (group_idx, group) in groups.iter().enumerate() {
            let memory = &group_memories[group_idx];

            for (local_cluster, &cluster_id) in group.cluster_ids.iter().enumerate() {
                let actual_neurons = if let Some(ref an) = group.actual_neurons {
                    an[local_cluster] as usize
                } else {
                    group.neurons
                };

                let neuron_base = local_cluster * group.neurons;
                let conn_base = group.conn_offset + local_cluster * group.neurons * group.bits;

                let mut sum = 0.0f32;
                for n in 0..actual_neurons {
                    let conn_start = conn_base + n * group.bits;
                    let address = compute_address(input_bits, &connections[conn_start..], group.bits);
                    let cell = memory.read(neuron_base + n, address);
                    sum += cell_to_weight(cell, memory_mode, empty_value);
                }

                scores[cluster_id] = (sum / actual_neurons as f32) as f64;
            }
        }

        scores
    }).collect();

    // Compute CE and accuracy without gating
    let (total_ce, total_correct): (f64, u64) = all_scores.par_iter().enumerate().map(|(ex_idx, scores)| {
        let target_idx = eval_targets[ex_idx] as usize;

        // Find prediction (argmax) for accuracy
        let predicted = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let correct: u64 = if predicted == target_idx { 1 } else { 0 };

        // Softmax and cross-entropy
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        let target_prob = exp_scores[target_idx] / sum_exp;
        let ce = -(target_prob + epsilon).ln();

        (ce, correct)
    }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

    let ce = total_ce / num_eval as f64;
    let accuracy = total_correct as f64 / num_eval as f64;

    // ========================================================================
    // Step 4: Evaluate WITH gating
    // ========================================================================

    let (total_gated_ce, total_gated_correct): (f64, u64) = (0..num_eval).into_par_iter().map(|ex_idx| {
        let input_start = ex_idx * total_input_bits;
        let input_bits = &eval_input_bits[input_start..input_start + total_input_bits];
        let target_idx = eval_targets[ex_idx] as usize;

        // Get scores for this example
        let scores = &all_scores[ex_idx];

        // Compute gates for this input
        let gates = gating.forward_single(input_bits);

        // Apply gates to scores (multiply)
        let gated_scores: Vec<f64> = scores.iter().zip(gates.iter())
            .map(|(&s, &g)| s * g as f64)
            .collect();

        // Check if any gates are open
        let any_open: bool = gates.iter().any(|&g| g > 0.0);

        if !any_open {
            // No gates open - use original scores (fallback)
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let target_prob = exp_scores[target_idx] / sum_exp;
            let ce = -(target_prob + epsilon).ln();

            let predicted = scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let correct: u64 = if predicted == target_idx { 1 } else { 0 };

            (ce, correct)
        } else {
            // Gated evaluation
            let predicted = gated_scores.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let correct: u64 = if predicted == target_idx { 1 } else { 0 };

            // Softmax on gated scores
            let max_score = gated_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = gated_scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            let target_prob = exp_scores[target_idx] / sum_exp;
            let ce = -(target_prob + epsilon).ln();

            (ce, correct)
        }
    }).reduce(|| (0.0, 0), |(ce1, c1), (ce2, c2)| (ce1 + ce2, c1 + c2));

    let gated_ce = total_gated_ce / num_eval as f64;
    let gated_accuracy = total_gated_correct as f64 / num_eval as f64;

    (ce, accuracy, gated_ce, gated_accuracy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_config_groups() {
        // 5 clusters with 3 different configs
        let bits = vec![8, 8, 10, 10, 8];
        let neurons = vec![5, 5, 3, 3, 5];

        let groups = build_config_groups(&bits, &neurons);

        assert_eq!(groups.len(), 2); // (5,8) and (3,10)

        // Find the (5,8) group
        let group_5_8 = groups.iter().find(|g| g.neurons == 5 && g.bits == 8).unwrap();
        assert_eq!(group_5_8.cluster_ids, vec![0, 1, 4]);

        // Find the (3,10) group
        let group_3_10 = groups.iter().find(|g| g.neurons == 3 && g.bits == 10).unwrap();
        assert_eq!(group_3_10.cluster_ids, vec![2, 3]);
    }
}
