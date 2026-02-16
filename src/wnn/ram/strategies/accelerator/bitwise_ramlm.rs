//! Bitwise RAMLM Accelerator — Full Rust+Metal genome evaluation
//!
//! Evaluates BitwiseRAMLM genomes entirely in Rust, avoiding Python overhead.
//! Pipeline per genome:
//!   1. Train (CPU, sequential per genome) — majority vote / QUAD nudging
//!   2. Forward pass (Metal GPU, with CPU fallback) — dense memory reads + probability
//!   3. Reconstruction + CE (Metal GPU) — 50K vocab × 16 bits matmul
//!
//! Parallelism is at the GENOME level only (outer rayon par_iter).
//! Each genome runs train+forward sequentially with non-atomic memory ops.
//!
//! BitwiseTokenCache holds pre-encoded tokens (created once).

#[cfg(target_os = "macos")]
use metal::*;
use rand::SeedableRng;
use rayon::prelude::*;
#[cfg(target_os = "macos")]
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_os = "macos")]
use std::sync::{Arc, RwLock};

use crate::neuron_memory::{
    FALSE, TRUE, EMPTY, QUAD_WEAK_TRUE, QUAD_WEIGHTS,
    CELLS_PER_WORD,
    MODE_TERNARY, MODE_QUAD_BINARY, MODE_QUAD_WEIGHTED,
    ClusterStorage, auto_sparse_threshold,
};

// =============================================================================
// Metal Bitwise CE Evaluator
// =============================================================================

#[cfg(target_os = "macos")]
mod metal_bitwise_ce_impl {
    use super::*;

    /// Metal-accelerated reconstruction + cross-entropy for bitwise evaluation.
    /// Computes log-product reconstruction over 50K vocab and CE in a single dispatch.
    pub struct MetalBitwiseCE {
        device: Device,
        command_queue: CommandQueue,
        pipeline: ComputePipelineState,
    }

    static METAL_BITWISE_CE: RwLock<Option<Arc<MetalBitwiseCE>>> = RwLock::new(None);

    /// Get or initialize the Metal bitwise CE evaluator (lazy, thread-safe).
    /// Returns None if WNN_NO_METAL is set or Metal is unavailable.
    pub fn get_metal_bitwise_ce() -> Option<Arc<MetalBitwiseCE>> {
        if std::env::var("WNN_NO_METAL").is_ok() {
            return None;
        }

        // Fast path
        {
            let guard = METAL_BITWISE_CE.read().unwrap();
            if let Some(ref arc) = *guard {
                return Some(Arc::clone(arc));
            }
        }

        // Slow path
        let mut guard = METAL_BITWISE_CE.write().unwrap();
        if guard.is_none() {
            match MetalBitwiseCE::new() {
                Ok(ce) => { *guard = Some(Arc::new(ce)); }
                Err(e) => { eprintln!("[BitwiseCE] Metal init failed: {e}"); }
            }
        }
        guard.as_ref().map(Arc::clone)
    }

    impl MetalBitwiseCE {
        pub fn new() -> Result<Self, String> {
            let device = Device::system_default().ok_or("No Metal device")?;
            let queue = device.new_command_queue();
            let src = include_str!("shaders/bitwise_ce.metal");
            let lib = device
                .new_library_with_source(src, &CompileOptions::new())
                .map_err(|e| format!("Shader compile: {e}"))?;
            let func = lib
                .get_function("bitwise_reconstruct_ce", None)
                .map_err(|e| format!("Kernel: {e}"))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("Pipeline: {e}"))?;
            Ok(Self { device, command_queue: queue, pipeline })
        }

        /// Compute CE for multiple genomes in one GPU dispatch.
        ///
        /// All genomes share the same eval data (targets, token_bits).
        /// bit_scores differ per genome (from forward pass).
        pub fn compute_ce_batch(
            &self,
            all_bit_scores: &[f32], // [num_genomes * num_eval * num_bits]
            token_bits: &[u8],      // [vocab_size * num_bits]
            targets: &[u32],        // [num_eval]
            num_genomes: usize,
            num_eval: usize,
            num_bits: usize,
            vocab_size: usize,
        ) -> Result<Vec<(f64, f64)>, String> {
            let total_work = num_genomes * num_eval;
            if total_work == 0 {
                return Ok(vec![(0.0, 0.0); num_genomes]);
            }

            #[repr(C)]
            struct BitwiseCEParams {
                num_examples: u32,
                num_bits: u32,
                vocab_size: u32,
                num_genomes: u32,
            }

            let params = BitwiseCEParams {
                num_examples: num_eval as u32,
                num_bits: num_bits as u32,
                vocab_size: vocab_size as u32,
                num_genomes: num_genomes as u32,
            };

            // Create GPU buffers
            let scores_buf = self.device.new_buffer_with_data(
                all_bit_scores.as_ptr() as *const _,
                (all_bit_scores.len() * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let token_bits_buf = self.device.new_buffer_with_data(
                token_bits.as_ptr() as *const _,
                (token_bits.len() * mem::size_of::<u8>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let targets_buf = self.device.new_buffer_with_data(
                targets.as_ptr() as *const _,
                (targets.len() * mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let params_buf = self.device.new_buffer_with_data(
                &params as *const _ as *const _,
                mem::size_of::<BitwiseCEParams>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let ce_buf = self.device.new_buffer(
                (total_work * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let correct_buf = self.device.new_buffer(
                (total_work * mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Dispatch
            let cmd = self.command_queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&scores_buf), 0);
            enc.set_buffer(1, Some(&token_bits_buf), 0);
            enc.set_buffer(2, Some(&targets_buf), 0);
            enc.set_buffer(3, Some(&params_buf), 0);
            enc.set_buffer(4, Some(&ce_buf), 0);
            enc.set_buffer(5, Some(&correct_buf), 0);

            let grid = MTLSize::new(total_work as u64, 1, 1);
            let max_threads = self.pipeline.max_total_threads_per_threadgroup();
            let group = MTLSize::new(max_threads.min(total_work as u64), 1, 1);
            enc.dispatch_threads(grid, group);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Read results
            let ce_ptr = ce_buf.contents() as *const f32;
            let correct_ptr = correct_buf.contents() as *const u32;
            let ce_values = unsafe { std::slice::from_raw_parts(ce_ptr, total_work) };
            let correct_values = unsafe { std::slice::from_raw_parts(correct_ptr, total_work) };

            // Reduce per genome
            let mut results = Vec::with_capacity(num_genomes);
            for g in 0..num_genomes {
                let start = g * num_eval;
                let end = start + num_eval;
                let total_ce: f64 = ce_values[start..end].iter().map(|&c| c as f64).sum();
                let total_correct: u32 = correct_values[start..end].iter().sum();
                results.push((
                    total_ce / num_eval as f64,
                    total_correct as f64 / num_eval as f64,
                ));
            }

            Ok(results)
        }
    }
}

#[cfg(target_os = "macos")]
pub use metal_bitwise_ce_impl::get_metal_bitwise_ce;

#[cfg(not(target_os = "macos"))]
pub fn get_metal_bitwise_ce() -> Option<std::sync::Arc<()>> { None }

// =============================================================================
// Metal Forward Pass Evaluator (dense memory, reuses ramlm.metal kernels)
// =============================================================================

#[cfg(target_os = "macos")]
mod metal_forward_impl {
	use super::*;

	static METAL_FORWARD_EVAL: RwLock<Option<Arc<crate::metal_ramlm::MetalRAMLMEvaluator>>> = RwLock::new(None);

	pub fn get_metal_forward_evaluator() -> Option<Arc<crate::metal_ramlm::MetalRAMLMEvaluator>> {
		if std::env::var("WNN_NO_METAL").is_ok() {
			return None;
		}
		{
			let guard = METAL_FORWARD_EVAL.read().unwrap();
			if let Some(ref arc) = *guard {
				return Some(Arc::clone(arc));
			}
		}
		let mut guard = METAL_FORWARD_EVAL.write().unwrap();
		if guard.is_none() {
			match crate::metal_ramlm::MetalRAMLMEvaluator::new() {
				Ok(eval) => { *guard = Some(Arc::new(eval)); }
				Err(e) => { eprintln!("[BitwiseForward] MetalRAMLMEvaluator init failed: {e}"); }
			}
		}
		guard.as_ref().map(Arc::clone)
	}
}

#[cfg(target_os = "macos")]
pub use metal_forward_impl::get_metal_forward_evaluator;

// =============================================================================
// Metal Sparse Forward Pass Evaluator (for sparse clusters)
// =============================================================================

#[cfg(target_os = "macos")]
mod metal_sparse_impl {
	use super::*;

	static METAL_SPARSE_EVAL: RwLock<Option<Arc<crate::metal_ramlm::MetalSparseEvaluator>>> = RwLock::new(None);

	pub fn get_metal_sparse_evaluator() -> Option<Arc<crate::metal_ramlm::MetalSparseEvaluator>> {
		if std::env::var("WNN_NO_METAL").is_ok() {
			return None;
		}
		{
			let guard = METAL_SPARSE_EVAL.read().unwrap();
			if let Some(ref arc) = *guard {
				return Some(Arc::clone(arc));
			}
		}
		let mut guard = METAL_SPARSE_EVAL.write().unwrap();
		if guard.is_none() {
			match crate::metal_ramlm::MetalSparseEvaluator::new() {
				Ok(eval) => { *guard = Some(Arc::new(eval)); }
				Err(e) => { eprintln!("[BitwiseForward] MetalSparseEvaluator init failed: {e}"); }
			}
		}
		guard.as_ref().map(Arc::clone)
	}
}

#[cfg(target_os = "macos")]
pub use metal_sparse_impl::get_metal_sparse_evaluator;

// =============================================================================
// Token Cache
// =============================================================================

/// Pre-encoded token subset for bitwise training.
pub struct BitwiseSubset {
    pub input_bits: Vec<bool>,   // [N * total_input_bits]
    pub packed_input: Vec<u64>,  // [N * words_per_example] packed bit representation
    pub target_bits: Vec<u8>,    // [N * num_bits]
    pub num_examples: usize,
    pub words_per_example: usize,
}

/// Pre-encoded eval subset for bitwise CE computation.
/// Unlike BitwiseSubset, stores u32 token IDs (needed for CE) instead of target_bits.
pub struct BitwiseEvalSubset {
    pub packed_input: Vec<u64>,  // [N * words_per_example] packed bit representation
    pub targets: Vec<u32>,       // [N] token IDs
    pub num_examples: usize,
    pub words_per_example: usize,
}

/// Persistent token cache for bitwise genome evaluation.
///
/// Created once with all tokens, pre-encodes into binary contexts + target bits.
/// Provides zero-overhead evaluation via pre-computed data.
pub struct BitwiseTokenCache {
    pub vocab_size: usize,
    pub context_size: usize,
    pub bits_per_token: usize,
    pub total_input_bits: usize,
    pub num_bits: usize, // = bits_per_token (e.g., 16 for GPT-2)

    /// Binary patterns for all tokens: [vocab_size * num_bits] (row-major)
    pub token_bits: Vec<u8>,

    /// Training data subsets (for rotation)
    pub train_subsets: Vec<BitwiseSubset>,
    /// Full training data (for evaluate_genomes_full)
    pub full_train: BitwiseSubset,

    /// Eval data subsets (for rotation during GA/TS)
    pub eval_subsets: Vec<BitwiseEvalSubset>,
    /// Full eval data (for evaluate_genomes_full — typically validation split)
    pub full_eval: BitwiseEvalSubset,

    pub num_parts: usize,
    pub num_eval_parts: usize,
    current_train_idx: AtomicUsize,
    current_eval_idx: AtomicUsize,
}

fn bits_needed(n: usize) -> usize {
    if n <= 1 { return 1; }
    ((n as f64).log2().ceil()) as usize
}

/// Pack boolean input bits into u64 words for cache-efficient address computation.
/// Reduces memory footprint 8x (1 byte/bit → 1 bit/bit).
fn pack_input_bits(input_bits: &[bool], num_examples: usize, total_input_bits: usize) -> (Vec<u64>, usize) {
    let words_per_example = (total_input_bits + 63) / 64;
    let mut packed = vec![0u64; num_examples * words_per_example];
    for ex in 0..num_examples {
        let bits_off = ex * total_input_bits;
        let pack_off = ex * words_per_example;
        for i in 0..total_input_bits {
            if input_bits[bits_off + i] {
                packed[pack_off + i / 64] |= 1u64 << (i % 64);
            }
        }
    }
    (packed, words_per_example)
}

impl BitwiseTokenCache {
    pub fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        num_parts: usize,
        num_eval_parts: usize,
        _pad_token_id: u32,
    ) -> Self {
        let bits_per_token = bits_needed(vocab_size);
        let total_input_bits = context_size * bits_per_token;
        let num_bits = bits_per_token;

        // Build token_bits table: [vocab_size * num_bits]
        let mut token_bits = vec![0u8; vocab_size * num_bits];
        for t in 0..vocab_size {
            for b in 0..num_bits {
                token_bits[t * num_bits + b] = ((t >> (num_bits - 1 - b)) & 1) as u8;
            }
        }

        // Encode full training data
        let (full_input_bits, _full_targets, full_target_bits, full_n) =
            Self::encode_sequence(&train_tokens, context_size, bits_per_token, total_input_bits, num_bits);
        let (full_packed, full_wpe) = pack_input_bits(&full_input_bits, full_n, total_input_bits);
        drop(full_input_bits);  // free unpacked bits (~77 MB for ctx=2)
        let full_train = BitwiseSubset {
            input_bits: Vec::new(),  // dropped after packing — only packed_input is used
            packed_input: full_packed,
            target_bits: full_target_bits,
            num_examples: full_n,
            words_per_example: full_wpe,
        };

        // Split training data into subsets
        let train_subsets = Self::split_and_encode(
            &train_tokens, context_size, bits_per_token, total_input_bits, num_bits, num_parts,
        );

        // Encode full eval data
        let (full_eval_input_bits, full_eval_targets, _, full_eval_n) =
            Self::encode_sequence(&eval_tokens, context_size, bits_per_token, total_input_bits, num_bits);
        let (full_eval_packed, full_eval_wpe) = pack_input_bits(&full_eval_input_bits, full_eval_n, total_input_bits);
        drop(full_eval_input_bits);  // free unpacked bits — only packed_input is used
        let full_eval = BitwiseEvalSubset {
            packed_input: full_eval_packed,
            targets: full_eval_targets,
            num_examples: full_eval_n,
            words_per_example: full_eval_wpe,
        };

        // Split eval data into subsets
        let eval_subsets = Self::split_and_encode_eval(
            &eval_tokens, context_size, bits_per_token, total_input_bits, num_bits, num_eval_parts,
        );

        let eval_subset_n: usize = eval_subsets.iter().map(|s| s.num_examples).sum();
        eprintln!(
            "[BitwiseCache] vocab={vocab_size}, context={context_size}, bits={num_bits}, \
             train={full_n} examples ({num_parts} subsets), eval={full_eval_n} examples ({num_eval_parts} subsets, {eval_subset_n} total)"
        );

        Self {
            vocab_size, context_size, bits_per_token, total_input_bits, num_bits,
            token_bits, train_subsets, full_train,
            eval_subsets, full_eval,
            num_parts, num_eval_parts,
            current_train_idx: AtomicUsize::new(0),
            current_eval_idx: AtomicUsize::new(0),
        }
    }

    /// Encode a token sequence into (input_bits, targets, target_bits, num_examples).
    fn encode_sequence(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        num_bits: usize,
    ) -> (Vec<bool>, Vec<u32>, Vec<u8>, usize) {
        if tokens.len() <= context_size {
            return (vec![], vec![], vec![], 0);
        }

        let num_ex = tokens.len() - context_size;
        let mut input_bits = vec![false; num_ex * total_input_bits];
        let mut targets = vec![0u32; num_ex];
        let mut target_bits = vec![0u8; num_ex * num_bits];

        // Parallel encoding
        input_bits
            .par_chunks_mut(total_input_bits)
            .zip(targets.par_iter_mut())
            .zip(target_bits.par_chunks_mut(num_bits))
            .enumerate()
            .for_each(|(i, ((inp, tgt), tb))| {
                // Encode context tokens
                for ctx in 0..context_size {
                    let token = tokens[i + ctx] as usize;
                    let off = ctx * bits_per_token;
                    for b in 0..bits_per_token {
                        inp[off + b] = ((token >> (bits_per_token - 1 - b)) & 1) == 1;
                    }
                }
                // Target token ID and its bits
                let target = tokens[i + context_size];
                *tgt = target;
                let t = target as usize;
                for b in 0..num_bits {
                    tb[b] = ((t >> (num_bits - 1 - b)) & 1) as u8;
                }
            });

        (input_bits, targets, target_bits, num_ex)
    }

    /// Split token sequence into num_parts subsets and encode each.
    fn split_and_encode(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        num_bits: usize,
        num_parts: usize,
    ) -> Vec<BitwiseSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;
        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let (input_bits, _, target_bits, num_ex) =
                    Self::encode_sequence(part, context_size, bits_per_token, total_input_bits, num_bits);
                let (packed, wpe) = pack_input_bits(&input_bits, num_ex, total_input_bits);
                BitwiseSubset { input_bits: Vec::new(), packed_input: packed, target_bits, num_examples: num_ex, words_per_example: wpe }
            })
            .collect()
    }

    /// Split eval token sequence into num_parts subsets and encode each.
    /// Returns BitwiseEvalSubset (with u32 targets for CE computation).
    fn split_and_encode_eval(
        tokens: &[u32],
        context_size: usize,
        bits_per_token: usize,
        total_input_bits: usize,
        num_bits: usize,
        num_parts: usize,
    ) -> Vec<BitwiseEvalSubset> {
        let n = tokens.len();
        let part_size = n / num_parts;
        (0..num_parts)
            .map(|i| {
                let start = i * part_size;
                let end = if i < num_parts - 1 { start + part_size } else { n };
                let part = &tokens[start..end];
                let (input_bits, targets, _, num_ex) =
                    Self::encode_sequence(part, context_size, bits_per_token, total_input_bits, num_bits);
                let (packed, wpe) = pack_input_bits(&input_bits, num_ex, total_input_bits);
                BitwiseEvalSubset { packed_input: packed, targets, num_examples: num_ex, words_per_example: wpe }
            })
            .collect()
    }

    /// Get next training subset index (advances rotator).
    pub fn next_train_idx(&self) -> usize {
        self.current_train_idx.fetch_add(1, Ordering::Relaxed) % self.num_parts
    }

    /// Get next eval subset index (advances rotator).
    pub fn next_eval_idx(&self) -> usize {
        self.current_eval_idx.fetch_add(1, Ordering::Relaxed) % self.num_eval_parts
    }

    /// Reset subset rotation (both train and eval).
    pub fn reset(&self) {
        self.current_train_idx.store(0, Ordering::Relaxed);
        self.current_eval_idx.store(0, Ordering::Relaxed);
    }
}

// =============================================================================
// Genome Evaluation
// =============================================================================

/// CPU fallback for reconstruction + CE computation.
fn compute_ce_cpu(
    bit_scores: &[f32],  // [num_eval * num_bits]
    token_bits: &[u8],   // [vocab_size * num_bits]
    targets: &[u32],     // [num_eval]
    num_eval: usize,
    num_bits: usize,
    vocab_size: usize,
) -> (f64, f64) {
    let eps = 1e-7f32;

    let results: Vec<(f64, u32)> = (0..num_eval)
        .into_par_iter()
        .map(|ex| {
            let score_off = ex * num_bits;
            let target = targets[ex] as usize;

            // Precompute diff[b] = log(p) - log(1-p) and base = sum(log(1-p))
            let mut base = 0.0f32;
            let mut diff = [0.0f32; 32];
            for b in 0..num_bits {
                let p = bit_scores[score_off + b].clamp(eps, 1.0 - eps);
                let lp0 = (1.0 - p).ln();
                diff[b] = p.ln() - lp0;
                base += lp0;
            }

            // Online log-sum-exp over all vocab tokens
            let mut max_lp = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            let mut target_lp = 0.0f32;
            let mut predicted = 0usize;
            let mut predicted_lp = f32::NEG_INFINITY;

            for t in 0..vocab_size {
                let tb_off = t * num_bits;
                let mut lp = base;
                for b in 0..num_bits {
                    lp += token_bits[tb_off + b] as f32 * diff[b];
                }

                if t == target { target_lp = lp; }
                if lp > predicted_lp { predicted_lp = lp; predicted = t; }

                if lp > max_lp {
                    sum_exp = sum_exp * (max_lp - lp).exp() + 1.0;
                    max_lp = lp;
                } else {
                    sum_exp += (lp - max_lp).exp();
                }
            }

            let ce = (max_lp + sum_exp.ln() - target_lp) as f64;
            let correct = if predicted == target { 1u32 } else { 0u32 };
            (ce, correct)
        })
        .collect();

    let total_ce: f64 = results.iter().map(|(c, _)| *c).sum();
    let total_correct: u32 = results.iter().map(|(_, c)| *c).sum();
    (total_ce / num_eval as f64, total_correct as f64 / num_eval as f64)
}

// =============================================================================
// Sequential per-genome train+forward (no inner rayon, no atomics)
//
// Each genome gets its own memory — no concurrent writes — so we can use
// direct (non-atomic) memory operations. This avoids CAS overhead and,
// critically, avoids nested rayon parallelism that serializes genome-level
// processing on the outer par_iter.
// =============================================================================


/// Inline xorshift32 PRNG — returns uniform float in (0, 1)
#[inline]
fn xorshift32_seq(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state >> 8) as f32 / 16777216.0
}

/// Geometric skip: sample gap to next selected example.
/// Returns floor(ln(rng) / ln(1 - rate)) using precomputed inv_log_complement = 1/ln(1-rate).
/// One RNG call gives the number of examples to skip, replacing per-example branching.
#[inline]
fn geometric_skip(rng: &mut u32, inv_log_complement: f32) -> usize {
    let u = xorshift32_seq(rng).max(f32::MIN_POSITIVE); // avoid ln(0)
    (u.ln() * inv_log_complement) as usize
}

/// Compute memory address for a single neuron given input bits (same as ramlm::compute_address).
#[inline]
#[allow(dead_code)]
fn compute_address_seq(input_bits: &[bool], connections: &[i64], bits_per_neuron: usize) -> usize {
    let mut address: usize = 0;
    for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
        if input_bits[conn_idx as usize] {
            address |= 1 << (bits_per_neuron - 1 - i);
        }
    }
    address
}

/// Compute memory address from packed u64 input bits (8x less memory bandwidth).
#[inline]
fn compute_address_packed(packed_words: &[u64], connections: &[i64], bits_per_neuron: usize) -> usize {
    let mut address: usize = 0;
    for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
        let idx = conn_idx as usize;
        let bit = (packed_words[idx / 64] >> (idx % 64)) & 1;
        address |= (bit as usize) << (bits_per_neuron - 1 - i);
    }
    address
}

/// Pre-computed per-cluster layout for heterogeneous configs.
///
/// With per-neuron bits, each neuron can have a different bit count.
/// `conn_offsets` is per-neuron (length total_neurons + 1).
/// `cluster_max_bits` holds the max bits within each cluster (used for ClusterStorage allocation).
struct GenomeLayout {
    /// Cumulative neuron offsets: neuron_offsets[c] = first global neuron index for cluster c.
    /// Length: num_clusters + 1 (last entry = total_neurons).
    neuron_offsets: Vec<usize>,
    /// Cumulative memory word offsets: mem_offsets[c] = first memory word for cluster c (dense only)
    mem_offsets: Vec<usize>,
    /// Per-neuron cumulative connection offsets: conn_offsets[n] = first connection index for neuron n.
    /// Length: total_neurons + 1 (last entry = total_connections).
    conn_offsets: Vec<usize>,
    /// Words per neuron for each cluster (based on cluster_max_bits).
    /// Used for ClusterStorage dense memory layout.
    words_per_neuron: Vec<usize>,
    /// Max bits within each cluster (for ClusterStorage allocation).
    cluster_max_bits: Vec<usize>,
    /// Total memory words needed (for dense-only fallback estimate)
    total_memory_words: usize,
    /// Total connections needed
    total_connections: usize,
    /// Estimated total bytes for this genome (accounts for dense/sparse mix)
    estimated_bytes: u64,
    /// Whether each cluster uses sparse storage
    cluster_is_sparse: Vec<bool>,
}

/// Compute per-cluster layout from per-neuron bit counts.
///
/// `bits_per_neuron` has length `sum(neurons_per_cluster)` — one entry per neuron.
/// Connection offsets are per-neuron. ClusterStorage uses max bits within each cluster.
fn compute_genome_layout(
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    sparse_threshold: usize,
    expected_train_examples: usize,
) -> GenomeLayout {
    let num_clusters = neurons_per_cluster.len();
    let total_neurons: usize = neurons_per_cluster.iter().sum();

    let mut neuron_offsets = Vec::with_capacity(num_clusters + 1);
    let mut mem_offsets = Vec::with_capacity(num_clusters);
    let mut words_per_neuron_vec = Vec::with_capacity(num_clusters);
    let mut cluster_max_bits_vec = Vec::with_capacity(num_clusters);
    let mut cluster_is_sparse = Vec::with_capacity(num_clusters);

    // Build per-neuron connection offsets
    let mut conn_offsets = Vec::with_capacity(total_neurons + 1);
    conn_offsets.push(0usize);
    for &b in bits_per_neuron {
        conn_offsets.push(conn_offsets.last().unwrap() + b);
    }
    let total_conn = *conn_offsets.last().unwrap();

    let mut cumul_neurons: usize = 0;
    let mut cumul_mem: usize = 0;
    let mut total_estimated: u64 = 0;

    for c in 0..num_clusters {
        let neurons = neurons_per_cluster[c];
        neuron_offsets.push(cumul_neurons);

        // Max bits within this cluster (for ClusterStorage allocation)
        let cluster_neuron_start = cumul_neurons;
        let cluster_neuron_end = cumul_neurons + neurons;
        let max_bits = bits_per_neuron[cluster_neuron_start..cluster_neuron_end]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);

        let addresses = 1usize << max_bits;
        let wpn = (addresses + CELLS_PER_WORD - 1) / CELLS_PER_WORD;
        let is_sparse = max_bits > sparse_threshold;

        mem_offsets.push(cumul_mem);
        words_per_neuron_vec.push(wpn);
        cluster_max_bits_vec.push(max_bits);
        cluster_is_sparse.push(is_sparse);

        // Estimate bytes for budget (use max_bits as representative)
        let est = ClusterStorage::estimated_bytes(
            neurons, max_bits, sparse_threshold, expected_train_examples,
        );
        total_estimated += est;

        cumul_neurons += neurons;
        cumul_mem += neurons * wpn;
    }
    neuron_offsets.push(cumul_neurons); // sentinel

    GenomeLayout {
        neuron_offsets,
        mem_offsets,
        conn_offsets,
        words_per_neuron: words_per_neuron_vec,
        cluster_max_bits: cluster_max_bits_vec,
        total_memory_words: cumul_mem,
        total_connections: total_conn,
        estimated_bytes: total_estimated,
        cluster_is_sparse,
    }
}

/// GPU forward pass for heterogeneous per-cluster configs (dense + sparse).
///
/// Groups clusters by (cluster_max_bits, neurons, is_dense), dispatches `forward_batch()`
/// for dense groups and `forward_batch_sparse()` for sparse groups.
///
/// With per-neuron bits, connections are variable-length per neuron. For GPU dispatch,
/// connections are re-packed into fixed-stride format (cluster_max_bits per neuron),
/// padding short neurons with connection index 0 (harmless — address bits beyond the
/// neuron's actual bit count are ignored by the memory lookup since ClusterStorage
/// uses cluster_max_bits for allocation).
#[cfg(target_os = "macos")]
fn gpu_forward_heterogeneous(
	evaluator: &crate::metal_ramlm::MetalRAMLMEvaluator,
	packed_input: &[u64],
	connections: &[i64],
	bits_per_neuron: &[usize],
	clusters: &[ClusterStorage],
	neurons_per_cluster: &[usize],
	layout: &GenomeLayout,
	num_eval: usize,
	words_per_example: usize,
	num_clusters: usize,
	memory_mode: u8,
	out_probs: &mut [f32],
) -> bool {
	// Group clusters by (cluster_max_bits, neurons, is_dense) for uniform GPU dispatch
	let mut dense_groups: Vec<(usize, usize, usize, Vec<usize>)> = Vec::new(); // (max_bits, neurons, wpn, indices)
	let mut sparse_groups: Vec<(usize, usize, Vec<usize>)> = Vec::new(); // (max_bits, neurons, indices)

	for c in 0..num_clusters {
		let max_bits = layout.cluster_max_bits[c];
		let neurons = neurons_per_cluster[c];
		if clusters[c].is_dense() {
			let wpn = layout.words_per_neuron[c];
			if let Some(g) = dense_groups.iter_mut().find(|(b, n, w, _)| *b == max_bits && *n == neurons && *w == wpn) {
				g.3.push(c);
			} else {
				dense_groups.push((max_bits, neurons, wpn, vec![c]));
			}
		} else {
			if let Some(g) = sparse_groups.iter_mut().find(|(b, n, _)| *b == max_bits && *n == neurons) {
				g.2.push(c);
			} else {
				sparse_groups.push((max_bits, neurons, vec![c]));
			}
		}
	}

	// Helper: gather connections for a cluster, padding each neuron to max_bits stride
	let gather_cluster_connections = |c: usize, max_bits: usize| -> Vec<i64> {
		let neurons = neurons_per_cluster[c];
		let neuron_start = layout.neuron_offsets[c];
		let mut out = Vec::with_capacity(neurons * max_bits);
		for n in 0..neurons {
			let global_n = neuron_start + n;
			let n_bits = bits_per_neuron[global_n];
			let conn_start = layout.conn_offsets[global_n];
			out.extend_from_slice(&connections[conn_start..conn_start + n_bits]);
			// Pad to max_bits with 0 (harmless connection index)
			for _ in n_bits..max_bits {
				out.push(0);
			}
		}
		out
	};

	// Dense groups: gather contiguous memory + connections, dispatch forward_batch
	for (max_bits, neurons, wpn, cluster_indices) in &dense_groups {
		let group_size = cluster_indices.len();
		let mut group_memory: Vec<i64> = Vec::with_capacity(neurons * group_size * wpn);
		let mut group_connections: Vec<i64> = Vec::with_capacity(neurons * group_size * max_bits);

		for &c in cluster_indices {
			group_memory.extend_from_slice(clusters[c].dense_words());
			group_connections.extend_from_slice(&gather_cluster_connections(c, *max_bits));
		}

		match evaluator.forward_batch(
			packed_input, &group_connections, &group_memory,
			num_eval, words_per_example,
			neurons * group_size, *max_bits, *neurons, group_size, *wpn, memory_mode,
		) {
			Ok(probs) => {
				for ex in 0..num_eval {
					for (i, &c) in cluster_indices.iter().enumerate() {
						out_probs[ex * num_clusters + c] = probs[ex * group_size + i];
					}
				}
			}
			Err(e) => {
				eprintln!("[BitwiseForward] GPU dense group failed: {e}");
				return false;
			}
		}
	}

	// Sparse groups: export to SparseGpuExport, dispatch forward_batch_sparse
	if !sparse_groups.is_empty() {
		let sparse_eval = get_metal_sparse_evaluator();
		if let Some(ref sparse_evaluator) = sparse_eval {
			for (max_bits, neurons, cluster_indices) in &sparse_groups {
				let group_size = cluster_indices.len();

				// Merge SparseGpuExport from all clusters in group
				let mut all_keys: Vec<u64> = Vec::new();
				let mut all_values: Vec<u8> = Vec::new();
				let mut all_offsets: Vec<u32> = Vec::new();
				let mut all_counts: Vec<u32> = Vec::new();
				let mut group_connections: Vec<i64> = Vec::with_capacity(neurons * group_size * max_bits);

				for &c in cluster_indices {
					let export = clusters[c].export_sparse_gpu();
					let base_offset = all_keys.len() as u32;
					for &off in &export.offsets {
						all_offsets.push(off + base_offset);
					}
					all_counts.extend_from_slice(&export.counts);
					all_keys.extend_from_slice(&export.keys);
					all_values.extend_from_slice(&export.values);

					group_connections.extend_from_slice(&gather_cluster_connections(c, *max_bits));
				}

				match sparse_evaluator.forward_batch_sparse(
					packed_input, &group_connections,
					&all_keys, &all_values, &all_offsets, &all_counts,
					num_eval, words_per_example,
					neurons * group_size, *max_bits, *neurons, group_size, memory_mode,
				) {
					Ok(probs) => {
						for ex in 0..num_eval {
							for (i, &c) in cluster_indices.iter().enumerate() {
								out_probs[ex * num_clusters + c] = probs[ex * group_size + i];
							}
						}
					}
					Err(e) => {
						eprintln!("[BitwiseForward] GPU sparse group failed: {e}");
						return false;
					}
				}
			}
		} else {
			// No sparse Metal evaluator — will fall through to CPU
			return false;
		}
	}

	true
}

/// Train a single genome into pre-allocated memory, then forward pass into output slice.
///
/// Per-neuron heterogeneous version: each neuron can have a different bit count.
/// Train neuron memory from training data.
///
/// Resets all cluster storage, then writes votes (TERNARY) or nudges (QUAD) into
/// each neuron's memory cells based on training examples.
///
/// `bits_per_neuron` has length `sum(neurons_per_cluster)`.
/// Each cluster independently uses dense or sparse storage via ClusterStorage.
///
/// Fully sequential (no rayon, no atomics) — designed to be called from
/// an outer par_iter over genomes.
fn train_into(
    connections: &[i64],
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    num_clusters: usize,
    layout: &GenomeLayout,
    train_subset: &BitwiseSubset,
    clusters: &mut [ClusterStorage],
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) {
    let num_examples = train_subset.num_examples;

    // Reset all cluster storage
    for cs in clusters.iter_mut() {
        cs.reset();
    }

    // Precompute geometric skip divisor: 1 / ln(1 - rate)
    let use_sampling = neuron_sample_rate < 1.0;
    let inv_log_complement = if use_sampling {
        1.0 / (1.0f32 - neuron_sample_rate).ln()
    } else {
        0.0
    };

    match memory_mode {
        MODE_TERNARY => {
            // ===== TRAINING: Majority vote (neuron-major for L1 cache locality) =====
            // For dense clusters: use dense vote array (fast, pre-allocated)
            // For sparse clusters: use per-neuron FxHashMap votes (no 2^bits allocation)
            let max_dense_votes = (0..num_clusters)
                .filter(|&c| clusters[c].is_dense())
                .map(|c| {
                    let max_bits = layout.cluster_max_bits[c];
                    neurons_per_cluster[c] * (1usize << max_bits)
                })
                .max()
                .unwrap_or(0);
            let mut dense_votes = vec![0i32; max_dense_votes];
            let wpe = train_subset.words_per_example;

            for cluster in 0..num_clusters {
                let c_neurons = neurons_per_cluster[cluster];
                let neuron_base = layout.neuron_offsets[cluster];

                if clusters[cluster].is_dense() {
                    // Dense path: vote array sized by cluster_max_bits
                    let c_max_bits = layout.cluster_max_bits[cluster];
                    let c_addresses = 1usize << c_max_bits;
                    let vote_size = c_neurons * c_addresses;
                    dense_votes[..vote_size].fill(0);

                    for n in 0..c_neurons {
                        let global_n = neuron_base + n;
                        let n_bits = bits_per_neuron[global_n];
                        let conn_start = layout.conn_offsets[global_n];
                        let conns = &connections[conn_start..conn_start + n_bits];
                        let vote_base = n * c_addresses;

                        let mut neuron_rng = (rng_seed as u32)
                            .wrapping_add(global_n as u32 * 1000003);
                        if neuron_rng == 0 { neuron_rng = 1; }

                        if use_sampling {
                            let mut ex = geometric_skip(&mut neuron_rng, inv_log_complement);
                            while ex < num_examples {
                                let packed = &train_subset.packed_input[ex * wpe..(ex + 1) * wpe];
                                let target_bit = train_subset.target_bits[ex * num_clusters + cluster];
                                let vote: i32 = if target_bit == 1 { 1 } else { -1 };
                                let addr = compute_address_packed(packed, conns, n_bits);
                                dense_votes[vote_base + addr] += vote;
                                ex += geometric_skip(&mut neuron_rng, inv_log_complement) + 1;
                            }
                        } else {
                            for ex in 0..num_examples {
                                let packed = &train_subset.packed_input[ex * wpe..(ex + 1) * wpe];
                                let target_bit = train_subset.target_bits[ex * num_clusters + cluster];
                                let vote: i32 = if target_bit == 1 { 1 } else { -1 };
                                let addr = compute_address_packed(packed, conns, n_bits);
                                dense_votes[vote_base + addr] += vote;
                            }
                        }
                    }

                    for n in 0..c_neurons {
                        let global_n = neuron_base + n;
                        let n_bits = bits_per_neuron[global_n];
                        let c_addresses = 1usize << n_bits;
                        let c_max_addresses = 1usize << c_max_bits;
                        // Only iterate over addresses reachable by this neuron's bit count
                        for addr in 0..c_addresses {
                            let v = dense_votes[n * c_max_addresses + addr];
                            if v > 0 {
                                clusters[cluster].write_cell(n, addr, TRUE);
                            } else if v < 0 {
                                clusters[cluster].write_cell(n, addr, FALSE);
                            }
                        }
                    }
                } else {
                    // Sparse path: per-neuron HashMap votes (no 2^bits allocation)
                    use rustc_hash::FxHashMap;
                    let mut sparse_votes: Vec<FxHashMap<u32, i32>> =
                        (0..c_neurons).map(|_| FxHashMap::default()).collect();

                    for n in 0..c_neurons {
                        let global_n = neuron_base + n;
                        let n_bits = bits_per_neuron[global_n];
                        let conn_start = layout.conn_offsets[global_n];
                        let conns = &connections[conn_start..conn_start + n_bits];

                        let mut neuron_rng = (rng_seed as u32)
                            .wrapping_add(global_n as u32 * 1000003);
                        if neuron_rng == 0 { neuron_rng = 1; }

                        if use_sampling {
                            let mut ex = geometric_skip(&mut neuron_rng, inv_log_complement);
                            while ex < num_examples {
                                let packed = &train_subset.packed_input[ex * wpe..(ex + 1) * wpe];
                                let target_bit = train_subset.target_bits[ex * num_clusters + cluster];
                                let vote: i32 = if target_bit == 1 { 1 } else { -1 };
                                let addr = compute_address_packed(packed, conns, n_bits);
                                *sparse_votes[n].entry(addr as u32).or_insert(0) += vote;
                                ex += geometric_skip(&mut neuron_rng, inv_log_complement) + 1;
                            }
                        } else {
                            for ex in 0..num_examples {
                                let packed = &train_subset.packed_input[ex * wpe..(ex + 1) * wpe];
                                let target_bit = train_subset.target_bits[ex * num_clusters + cluster];
                                let vote: i32 = if target_bit == 1 { 1 } else { -1 };
                                let addr = compute_address_packed(packed, conns, n_bits);
                                *sparse_votes[n].entry(addr as u32).or_insert(0) += vote;
                            }
                        }
                    }

                    // Write non-zero votes into cluster storage
                    for n in 0..c_neurons {
                        for (&addr, &v) in &sparse_votes[n] {
                            if v > 0 {
                                clusters[cluster].write_cell(n, addr as usize, TRUE);
                            } else if v < 0 {
                                clusters[cluster].write_cell(n, addr as usize, FALSE);
                            }
                        }
                    }
                }
            }
        }
        MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => {
            // ===== TRAINING: Sequential nudging (neuron-major for L1 cache locality) =====
            let wpe = train_subset.words_per_example;

            for cluster in 0..num_clusters {
                let c_neurons = neurons_per_cluster[cluster];
                let neuron_base = layout.neuron_offsets[cluster];

                for n in 0..c_neurons {
                    let global_n = neuron_base + n;
                    let n_bits = bits_per_neuron[global_n];
                    let conn_start = layout.conn_offsets[global_n];
                    let conns = &connections[conn_start..conn_start + n_bits];

                    let mut neuron_rng = (rng_seed as u32)
                        .wrapping_add(global_n as u32 * 1000003);
                    if neuron_rng == 0 { neuron_rng = 1; }

                    if use_sampling {
                        let mut ex = geometric_skip(&mut neuron_rng, inv_log_complement);
                        while ex < num_examples {
                            let packed = &train_subset.packed_input[ex * wpe..(ex + 1) * wpe];
                            let target_true = train_subset.target_bits[ex * num_clusters + cluster] == 1;
                            let addr = compute_address_packed(packed, conns, n_bits);
                            clusters[cluster].nudge_cell(n, addr, target_true);
                            ex += geometric_skip(&mut neuron_rng, inv_log_complement) + 1;
                        }
                    } else {
                        for ex in 0..num_examples {
                            let packed = &train_subset.packed_input[ex * wpe..(ex + 1) * wpe];
                            let target_true = train_subset.target_bits[ex * num_clusters + cluster] == 1;
                            let addr = compute_address_packed(packed, conns, n_bits);
                            clusters[cluster].nudge_cell(n, addr, target_true);
                        }
                    }
                }
            }
        }
        _ => {
            eprintln!("[BitwiseRAMLM] Unknown memory_mode={memory_mode}, falling back to TERNARY");
        }
    }
}

/// Forward-evaluate trained neuron memory on eval data, writing per-cluster probabilities.
///
/// Reads from immutable cluster storage. Tries Metal GPU first, falls back to CPU.
/// Output: `out_probs[ex * num_clusters + cluster]` = P(bit=1) for that cluster.
fn forward_eval_into(
    connections: &[i64],
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    num_clusters: usize,
    layout: &GenomeLayout,
    eval_subset: &BitwiseEvalSubset,
    clusters: &[ClusterStorage],
    out_probs: &mut [f32],
    memory_mode: u8,
) {
    // ===== GPU FORWARD PASS on eval data (try GPU, fall through to CPU if unavailable) =====
    #[cfg(target_os = "macos")]
    {
        if let Some(evaluator) = get_metal_forward_evaluator() {
            let num_eval = eval_subset.num_examples;
            let wpe = eval_subset.words_per_example;

            // Uniform fast-path: all clusters have same max_bits AND neurons AND all are dense
            let uniform = layout.cluster_max_bits.iter().all(|&b| b == layout.cluster_max_bits[0])
                && neurons_per_cluster.iter().all(|&n| n == neurons_per_cluster[0])
                && bits_per_neuron.iter().all(|&b| b == bits_per_neuron[0])
                && clusters.iter().all(|c| c.is_dense());

            let gpu_ok = if uniform {
                // All neurons are uniform + all clusters dense: gather into one flat memory
                let bits = bits_per_neuron[0]; // uniform, so any neuron's bits works
                let neurons = neurons_per_cluster[0];
                let wpn = layout.words_per_neuron[0];
                let mut flat_memory: Vec<i64> = Vec::with_capacity(neurons * num_clusters * wpn);
                for c in 0..num_clusters {
                    flat_memory.extend_from_slice(clusters[c].dense_words());
                }
                match evaluator.forward_batch(
                    &eval_subset.packed_input,
                    connections,
                    &flat_memory,
                    num_eval, wpe,
                    neurons * num_clusters, bits, neurons, num_clusters, wpn, memory_mode,
                ) {
                    Ok(probs) => {
                        out_probs[..probs.len()].copy_from_slice(&probs);
                        true
                    }
                    Err(e) => {
                        eprintln!("[BitwiseForward] GPU uniform failed: {e}");
                        false
                    }
                }
            } else {
                gpu_forward_heterogeneous(
                    &evaluator, &eval_subset.packed_input, connections, bits_per_neuron,
                    clusters, neurons_per_cluster, layout,
                    num_eval, wpe, num_clusters, memory_mode, out_probs,
                )
            };

            if gpu_ok {
                return;
            }
        }
    }

    // ===== CPU FORWARD PASS on eval data (fallback) =====
    let num_eval = eval_subset.num_examples;
    let empty_value = crate::neuron_memory::get_empty_value();
    let wpe_eval = eval_subset.words_per_example;

    for ex in 0..num_eval {
        let packed = &eval_subset.packed_input[ex * wpe_eval..(ex + 1) * wpe_eval];
        let ex_probs = &mut out_probs[ex * num_clusters..(ex + 1) * num_clusters];

        for cluster in 0..num_clusters {
            let c_neurons = neurons_per_cluster[cluster];
            let neuron_base = layout.neuron_offsets[cluster];

            match memory_mode {
                MODE_TERNARY => {
                    let mut count_true = 0u32;
                    let mut count_empty = 0u32;
                    for n in 0..c_neurons {
                        let global_n = neuron_base + n;
                        let n_bits = bits_per_neuron[global_n];
                        let conn_start = layout.conn_offsets[global_n];
                        let conns = &connections[conn_start..conn_start + n_bits];
                        let addr = compute_address_packed(packed, conns, n_bits);
                        let cell = clusters[cluster].read_cell(n, addr);
                        if cell == TRUE {
                            count_true += 1;
                        } else if cell == EMPTY {
                            count_empty += 1;
                        }
                    }
                    ex_probs[cluster] = (count_true as f32 + empty_value * count_empty as f32)
                        / c_neurons as f32;
                }
                MODE_QUAD_BINARY => {
                    let mut count_true = 0u32;
                    for n in 0..c_neurons {
                        let global_n = neuron_base + n;
                        let n_bits = bits_per_neuron[global_n];
                        let conn_start = layout.conn_offsets[global_n];
                        let conns = &connections[conn_start..conn_start + n_bits];
                        let addr = compute_address_packed(packed, conns, n_bits);
                        let cell = clusters[cluster].read_cell(n, addr);
                        if cell >= QUAD_WEAK_TRUE {
                            count_true += 1;
                        }
                    }
                    ex_probs[cluster] = count_true as f32 / c_neurons as f32;
                }
                MODE_QUAD_WEIGHTED => {
                    let mut weighted_sum = 0.0f32;
                    for n in 0..c_neurons {
                        let global_n = neuron_base + n;
                        let n_bits = bits_per_neuron[global_n];
                        let conn_start = layout.conn_offsets[global_n];
                        let conns = &connections[conn_start..conn_start + n_bits];
                        let addr = compute_address_packed(packed, conns, n_bits);
                        let cell = clusters[cluster].read_cell(n, addr);
                        weighted_sum += QUAD_WEIGHTS[cell as usize];
                    }
                    ex_probs[cluster] = weighted_sum / c_neurons as f32;
                }
                _ => {
                    ex_probs[cluster] = 0.5;
                }
            }
        }
    }
}

/// Train neuron memory and forward-evaluate on eval data (convenience wrapper).
///
/// Equivalent to calling `train_into` followed by `forward_eval_into`.
/// Kept for backward compatibility with callers that don't need the split.
fn train_and_forward_into(
    cache: &BitwiseTokenCache,
    connections: &[i64],
    bits_per_neuron: &[usize],
    neurons_per_cluster: &[usize],
    layout: &GenomeLayout,
    train_subset: &BitwiseSubset,
    eval_subset: &BitwiseEvalSubset,
    clusters: &mut [ClusterStorage],
    out_probs: &mut [f32],
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) {
    let num_clusters = cache.num_bits;
    train_into(
        connections, bits_per_neuron, neurons_per_cluster, num_clusters,
        layout, train_subset, clusters, memory_mode, neuron_sample_rate, rng_seed,
    );
    forward_eval_into(
        connections, bits_per_neuron, neurons_per_cluster, num_clusters,
        layout, eval_subset, clusters, out_probs, memory_mode,
    );
}

/// Evaluate multiple genomes with per-neuron heterogeneous configs (subset training + eval).
///
/// `bits_per_neuron_flat` is variable-length: for genome g, there are
/// `sum(neurons_per_cluster_flat[g*num_clusters..(g+1)*num_clusters])` entries.
/// `neurons_per_cluster_flat` is `[num_genomes * num_clusters]`.
/// `connections_flat` is variable-length: total connections across all genomes.
pub fn evaluate_genomes(
    cache: &BitwiseTokenCache,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    train_subset_idx: usize,
    eval_subset_idx: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold_override: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    let train_subset = &cache.train_subsets[train_subset_idx % cache.num_parts];
    let eval_subset = &cache.eval_subsets[eval_subset_idx % cache.num_eval_parts];
    evaluate_genomes_with_subset(
        cache, bits_per_neuron_flat, neurons_per_cluster_flat,
        connections_flat, num_genomes, train_subset, eval_subset,
        memory_mode, neuron_sample_rate, rng_seed, sparse_threshold_override,
    )
}

/// Evaluate multiple genomes with per-neuron heterogeneous configs (full training + full eval).
pub fn evaluate_genomes_full(
    cache: &BitwiseTokenCache,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold_override: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    evaluate_genomes_with_subset(
        cache, bits_per_neuron_flat, neurons_per_cluster_flat,
        connections_flat, num_genomes, &cache.full_train, &cache.full_eval,
        memory_mode, neuron_sample_rate, rng_seed, sparse_threshold_override,
    )
}

/// Core evaluation: parallel train+forward (per-neuron heterogeneous), then batched CE.
///
/// Each genome can have different bits per neuron and neurons per cluster.
/// Layouts are pre-computed per genome, and memory is allocated per genome
/// based on its actual layout.
///
/// `bits_per_neuron_flat` is variable-length: entries for genome g start at
/// `genome_bpn_offsets[g]` and have length `total_neurons_g`.
///
/// Sparse threshold is auto-computed per genome to fit all genomes in the memory budget
/// simultaneously (maximizing parallelism). If sparse_threshold_override is Some(t),
/// uses that fixed threshold for all genomes instead.
///
/// Returns: Vec<(ce, accuracy, weighted_bit_accuracy)> per genome.
fn evaluate_genomes_with_subset(
    cache: &BitwiseTokenCache,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    train_subset: &BitwiseSubset,
    eval_subset: &BitwiseEvalSubset,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold_override: Option<usize>,
) -> Vec<(f64, f64, f64)> {
    let num_clusters = cache.num_bits;
    let num_eval = eval_subset.num_examples;
    let scores_per_genome = num_eval * num_clusters;
    let expected_train = train_subset.num_examples;

    // Compute empty_word based on memory mode (needed for dense ClusterStorage)
    let empty_word: i64 = crate::neuron_memory::empty_word_for_mode(memory_mode);

    // Memory budget for concurrent genome training
    let mem_budget: u64 = 40 * 1024 * 1024 * 1024; // 40 GiB (Mac Studio has 64 GiB)
    let target_per_genome = if num_genomes > 0 { mem_budget / num_genomes as u64 } else { mem_budget };

    // Pre-compute per-genome offsets into bits_per_neuron_flat.
    // Each genome has total_neurons = sum(neurons_per_cluster[g*nc..(g+1)*nc]) entries.
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let base = g * num_clusters;
        let neurons_slice = &neurons_per_cluster_flat[base..base + num_clusters];
        let total_neurons: usize = neurons_slice.iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    // Pre-compute per-genome layouts, thresholds, and connection offsets
    let mut genome_layouts: Vec<GenomeLayout> = Vec::with_capacity(num_genomes);
    let mut genome_conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut genome_thresholds: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut cumul_conn: usize = 0;

    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let neurons_slice = &neurons_per_cluster_flat[nc_base..nc_base + num_clusters];
        let bpn_start = genome_bpn_offsets[g];
        let bpn_end = genome_bpn_offsets[g + 1];
        let bpn_slice = &bits_per_neuron_flat[bpn_start..bpn_end];

        // For auto_sparse_threshold, compute per-cluster max bits
        let cluster_max_bits: Vec<usize> = {
            let mut v = Vec::with_capacity(num_clusters);
            let mut off = 0;
            for &nc in neurons_slice {
                let max_b = bpn_slice[off..off + nc].iter().copied().max().unwrap_or(0);
                v.push(max_b);
                off += nc;
            }
            v
        };

        let threshold = match sparse_threshold_override {
            Some(t) => t,
            None => auto_sparse_threshold(&cluster_max_bits, neurons_slice, target_per_genome, expected_train),
        };

        let layout = compute_genome_layout(bpn_slice, neurons_slice, threshold, expected_train);
        genome_conn_offsets.push(cumul_conn);
        cumul_conn += layout.total_connections;
        genome_layouts.push(layout);
        genome_thresholds.push(threshold);
    }

    // Allocate output scores (uniform across genomes — always num_eval * num_clusters)
    let mut all_scores: Vec<f32> = vec![0.0f32; num_genomes * scores_per_genome];

    // Phase 1: Train + Forward (CPU, parallel over genomes with memory budgeting)
    // Each genome allocates ClusterStorage per cluster (dense or sparse).
    // Adaptive batching: pack as many genomes as fit within budget per batch.
    let genome_mem_bytes: Vec<u64> = genome_layouts.iter()
        .map(|l| l.estimated_bytes)
        .collect();

    let mut batch_start = 0;
    let mut batch_num = 0usize;
    let num_batches_total = {
        // Pre-count batches for logging
        let mut cnt = 0;
        let mut s = 0;
        while s < num_genomes {
            let mut m: u64 = 0;
            let mut e = s;
            while e < num_genomes {
                let gm = genome_mem_bytes[e];
                if m + gm > mem_budget && e > s { break; }
                m += gm;
                e += 1;
            }
            cnt += 1;
            s = e;
        }
        cnt
    };
    if num_batches_total > 1 {
        eprintln!("[BitwiseEval] {} genomes split into {} batches (budget: {} GiB)",
            num_genomes, num_batches_total, mem_budget / (1024 * 1024 * 1024));
    }
    while batch_start < num_genomes {
        // Pack genomes into this batch until budget is reached
        let mut batch_mem: u64 = 0;
        let mut batch_end = batch_start;
        while batch_end < num_genomes {
            let gm = genome_mem_bytes[batch_end];
            if batch_mem + gm > mem_budget && batch_end > batch_start {
                break;
            }
            batch_mem += gm;
            batch_end += 1;
        }
        if num_batches_total > 1 {
            let batch_time = std::time::Instant::now();
            eprintln!("[BitwiseEval] Batch {}/{}: genomes {}..{} ({:.1} GiB)",
                batch_num + 1, num_batches_total, batch_start, batch_end,
                batch_mem as f64 / (1024.0 * 1024.0 * 1024.0));
            let _ = batch_time; // suppress unused warning
        }
        batch_num += 1;

        let batch_scores = &mut all_scores[batch_start * scores_per_genome..batch_end * scores_per_genome];

        batch_scores
            .par_chunks_mut(scores_per_genome)
            .enumerate()
            .for_each(|(i, score_slice)| {
                let g = batch_start + i;
                let layout = &genome_layouts[g];
                let nc_base = g * num_clusters;
                let neurons_slice = &neurons_per_cluster_flat[nc_base..nc_base + num_clusters];
                let bpn_start = genome_bpn_offsets[g];
                let bpn_end = genome_bpn_offsets[g + 1];
                let bpn_slice = &bits_per_neuron_flat[bpn_start..bpn_end];
                let conn_start = genome_conn_offsets[g];
                let conn_end = conn_start + layout.total_connections;
                let connections = &connections_flat[conn_start..conn_end];

                // Allocate per-genome ClusterStorage (dense/sparse per auto-threshold)
                // Uses cluster_max_bits for ClusterStorage allocation
                let threshold = genome_thresholds[g];
                let mut cluster_storage: Vec<ClusterStorage> = (0..num_clusters)
                    .map(|c| ClusterStorage::new(
                        neurons_slice[c], layout.cluster_max_bits[c],
                        threshold, empty_word, memory_mode,
                    ))
                    .collect();

                let genome_rng_seed = rng_seed.wrapping_add(g as u64 * 1000007);
                train_and_forward_into(
                    cache, connections, bpn_slice, neurons_slice, layout,
                    train_subset, eval_subset, &mut cluster_storage, score_slice,
                    memory_mode, neuron_sample_rate, genome_rng_seed,
                );
            });

        batch_start = batch_end;
    }

    // Phase 1.5: Compute weighted BitAcc per genome
    // Weight each bit by its entropy in the eval targets: balanced bits (50/50) matter
    // more than degenerate bits (always 0 or always 1).
    let num_bits = cache.num_bits;
    let mut bit_weights = vec![0.0f64; num_bits];
    {
        let mut bit_ones = vec![0u64; num_bits];
        for ex in 0..num_eval {
            let target = eval_subset.targets[ex] as usize;
            let tb_off = target * num_bits;
            for b in 0..num_bits {
                bit_ones[b] += cache.token_bits[tb_off + b] as u64;
            }
        }
        for b in 0..num_bits {
            let p = bit_ones[b] as f64 / num_eval as f64;
            if p > 0.0 && p < 1.0 {
                bit_weights[b] = (-p * p.ln() - (1.0 - p) * (1.0 - p).ln())
                    / std::f64::consts::LN_2;
            }
            // else: degenerate bit → weight 0 (always same value, no information)
        }
    }
    let total_weight: f64 = bit_weights.iter().sum();

    // Compute weighted BitAcc per genome (parallel over genomes)
    let weighted_bit_accs: Vec<f64> = (0..num_genomes)
        .into_par_iter()
        .map(|g| {
            let score_base = g * scores_per_genome;
            let mut weighted_correct = 0.0f64;
            for ex in 0..num_eval {
                let target = eval_subset.targets[ex] as usize;
                let tb_off = target * num_bits;
                let sc_off = score_base + ex * num_bits;
                for b in 0..num_bits {
                    let predicted = all_scores[sc_off + b] > 0.5;
                    let actual = cache.token_bits[tb_off + b] > 0;
                    if predicted == actual {
                        weighted_correct += bit_weights[b];
                    }
                }
            }
            if total_weight > 0.0 {
                weighted_correct / (num_eval as f64 * total_weight)
            } else {
                0.0
            }
        })
        .collect();

    // Phase 2: Reconstruction + CE
    #[cfg(target_os = "macos")]
    if let Some(metal) = get_metal_bitwise_ce() {
        match metal.compute_ce_batch(
            &all_scores,
            &cache.token_bits,
            &eval_subset.targets,
            num_genomes,
            num_eval,
            cache.num_bits,
            cache.vocab_size,
        ) {
            Ok(ce_results) => {
                return ce_results.into_iter()
                    .zip(weighted_bit_accs)
                    .map(|((ce, acc), bit_acc)| (ce, acc, bit_acc))
                    .collect();
            }
            Err(e) => {
                eprintln!("[BitwiseCE] Metal failed, CPU fallback: {e}");
            }
        }
    }

    // CPU fallback (parallel over genomes)
    let ce_results: Vec<(f64, f64)> = all_scores
        .par_chunks(scores_per_genome)
        .map(|scores| {
            compute_ce_cpu(
                scores,
                &cache.token_bits,
                &eval_subset.targets,
                num_eval,
                cache.num_bits,
                cache.vocab_size,
            )
        })
        .collect();

    ce_results.into_iter()
        .zip(weighted_bit_accs)
        .map(|((ce, acc), bit_acc)| (ce, acc, bit_acc))
        .collect()
}

/// Result of adaptive evaluation for a single genome.
pub struct AdaptiveResult {
    pub ce: f64,
    pub acc: f64,
    pub bit_acc: f64,
    pub adapted_bits: Vec<usize>,
    pub adapted_neurons: Vec<usize>,
    pub adapted_connections: Vec<i64>,
    pub pruned: usize,
    pub grown: usize,
    pub added: usize,
    pub removed: usize,
}

/// Evaluate multiple genomes with per-genome adaptation (Baldwin effect).
///
/// For each genome (in parallel):
///   1. Train neuron memory from training data
///   2. Compute neuron/cluster stats from trained memory
///   3. Run synaptogenesis/neurogenesis passes to adapt architecture
///   4. If architecture changed: re-allocate storage, retrain, re-forward
///   5. Forward-evaluate on eval data
///
/// The flat score buffer layout is invariant under adaptation (always
/// `num_eval * num_clusters` per genome), so Metal CE batching is preserved.
///
/// Returns one `AdaptiveResult` per genome containing adapted architecture + scores.
pub fn evaluate_genomes_adaptive(
    cache: &BitwiseTokenCache,
    bits_per_neuron_flat: &[usize],
    neurons_per_cluster_flat: &[usize],
    connections_flat: &[i64],
    num_genomes: usize,
    train_subset: &BitwiseSubset,
    eval_subset: &BitwiseEvalSubset,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
    sparse_threshold_override: Option<usize>,
    adapt_config: &crate::adaptation::AdaptationConfig,
    generation: usize,
) -> Vec<AdaptiveResult> {
    use crate::adaptation;
    use std::sync::Mutex;

    let num_clusters = cache.num_bits;
    let num_eval = eval_subset.num_examples;
    let scores_per_genome = num_eval * num_clusters;
    let expected_train = train_subset.num_examples;
    let empty_word: i64 = crate::neuron_memory::empty_word_for_mode(memory_mode);
    let needs_adaptation = adapt_config.synaptogenesis_enabled || adapt_config.neurogenesis_enabled;

    // GPU stats accelerator (macOS only) — shared across genome threads
    #[cfg(target_os = "macos")]
    let metal_stats: Option<std::sync::Arc<crate::metal_stats::MetalStatsComputer>> =
        if needs_adaptation {
            match crate::metal_stats::MetalStatsComputer::new() {
                Ok(m) => {
                    eprintln!("[Adaptation] GPU stats accelerator initialized");
                    Some(std::sync::Arc::new(m))
                }
                Err(e) => {
                    eprintln!("[Adaptation] GPU stats init failed (CPU fallback): {e}");
                    None
                }
            }
        } else {
            None
        };

    // Memory budget for concurrent genome training
    let mem_budget: u64 = 40 * 1024 * 1024 * 1024; // 40 GiB (Mac Studio has 64 GiB)
    let target_per_genome = if num_genomes > 0 { mem_budget / num_genomes as u64 } else { mem_budget };

    // Pre-compute per-genome offsets into bits_per_neuron_flat
    let mut genome_bpn_offsets: Vec<usize> = Vec::with_capacity(num_genomes + 1);
    genome_bpn_offsets.push(0);
    for g in 0..num_genomes {
        let base = g * num_clusters;
        let neurons_slice = &neurons_per_cluster_flat[base..base + num_clusters];
        let total_neurons: usize = neurons_slice.iter().sum();
        genome_bpn_offsets.push(genome_bpn_offsets.last().unwrap() + total_neurons);
    }

    // Pre-compute per-genome layouts, thresholds, and connection offsets
    let mut genome_layouts: Vec<GenomeLayout> = Vec::with_capacity(num_genomes);
    let mut genome_conn_offsets: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut genome_thresholds: Vec<usize> = Vec::with_capacity(num_genomes);
    let mut cumul_conn: usize = 0;

    for g in 0..num_genomes {
        let nc_base = g * num_clusters;
        let neurons_slice = &neurons_per_cluster_flat[nc_base..nc_base + num_clusters];
        let bpn_start = genome_bpn_offsets[g];
        let bpn_end = genome_bpn_offsets[g + 1];
        let bpn_slice = &bits_per_neuron_flat[bpn_start..bpn_end];

        let cluster_max_bits: Vec<usize> = {
            let mut v = Vec::with_capacity(num_clusters);
            let mut off = 0;
            for &nc in neurons_slice {
                let max_b = bpn_slice[off..off + nc].iter().copied().max().unwrap_or(0);
                v.push(max_b);
                off += nc;
            }
            v
        };

        let threshold = match sparse_threshold_override {
            Some(t) => t,
            None => auto_sparse_threshold(&cluster_max_bits, neurons_slice, target_per_genome, expected_train),
        };

        let layout = compute_genome_layout(bpn_slice, neurons_slice, threshold, expected_train);
        genome_conn_offsets.push(cumul_conn);
        cumul_conn += layout.total_connections;
        genome_layouts.push(layout);
        genome_thresholds.push(threshold);
    }

    // Allocate output scores (uniform: always num_eval * num_clusters per genome)
    let mut all_scores: Vec<f32> = vec![0.0f32; num_genomes * scores_per_genome];

    // Per-genome adapted data slots (Mutex per genome — zero contention since each
    // thread writes a unique index)
    let adapted_data: Vec<Mutex<Option<(Vec<usize>, Vec<usize>, Vec<i64>, usize, usize, usize, usize)>>> =
        (0..num_genomes).map(|_| Mutex::new(None)).collect();

    // Phase 1: Train + Adapt + Forward (parallel over genomes with memory budgeting)
    let genome_mem_bytes: Vec<u64> = genome_layouts.iter()
        .map(|l| l.estimated_bytes)
        .collect();

    let mut batch_start = 0;
    while batch_start < num_genomes {
        // Pack genomes into this batch until budget is reached
        let mut batch_mem: u64 = 0;
        let mut batch_end = batch_start;
        while batch_end < num_genomes {
            let gm = genome_mem_bytes[batch_end];
            // Double budget estimate for adaptive: may need re-allocate after adaptation
            let gm_adaptive = if needs_adaptation { gm * 2 } else { gm };
            if batch_mem + gm_adaptive > mem_budget && batch_end > batch_start {
                break;
            }
            batch_mem += gm_adaptive;
            batch_end += 1;
        }

        let batch_scores = &mut all_scores[batch_start * scores_per_genome..batch_end * scores_per_genome];

        batch_scores
            .par_chunks_mut(scores_per_genome)
            .enumerate()
            .for_each(|(i, score_slice)| {
                let g = batch_start + i;
                let layout = &genome_layouts[g];
                let nc_base = g * num_clusters;
                let neurons_slice = &neurons_per_cluster_flat[nc_base..nc_base + num_clusters];
                let bpn_start = genome_bpn_offsets[g];
                let bpn_end = genome_bpn_offsets[g + 1];
                let bpn_slice = &bits_per_neuron_flat[bpn_start..bpn_end];
                let conn_start = genome_conn_offsets[g];
                let conn_end = conn_start + layout.total_connections;
                let connections = &connections_flat[conn_start..conn_end];

                let threshold = genome_thresholds[g];
                let genome_rng_seed = rng_seed.wrapping_add(g as u64 * 1000007);

                // Step 1: Allocate storage and train
                let mut cluster_storage: Vec<ClusterStorage> = (0..num_clusters)
                    .map(|c| ClusterStorage::new(
                        neurons_slice[c], layout.cluster_max_bits[c],
                        threshold, empty_word, memory_mode,
                    ))
                    .collect();

                train_into(
                    connections, bpn_slice, neurons_slice, num_clusters,
                    layout, train_subset, &mut cluster_storage,
                    memory_mode, neuron_sample_rate, genome_rng_seed,
                );

                if !needs_adaptation {
                    // No adaptation — forward directly (same as non-adaptive path)
                    forward_eval_into(
                        connections, bpn_slice, neurons_slice, num_clusters,
                        layout, eval_subset, &cluster_storage, score_slice, memory_mode,
                    );
                    // Store original genome as "adapted" (no changes)
                    *adapted_data[g].lock().unwrap() = Some((
                        bpn_slice.to_vec(), neurons_slice.to_vec(), connections.to_vec(),
                        0, 0, 0, 0,
                    ));
                    return;
                }

                // Warmup guard: skip all adaptation (and expensive stats) before warmup
                if generation < adapt_config.warmup_generations {
                    forward_eval_into(
                        connections, bpn_slice, neurons_slice, num_clusters,
                        layout, eval_subset, &cluster_storage, score_slice, memory_mode,
                    );
                    *adapted_data[g].lock().unwrap() = Some((
                        bpn_slice.to_vec(), neurons_slice.to_vec(), connections.to_vec(),
                        0, 0, 0, 0,
                    ));
                    return;
                }

                // Step 2: Adapt architecture
                let mut adapted_bits = bpn_slice.to_vec();
                let mut adapted_neurons = neurons_slice.to_vec();
                let mut adapted_connections = connections.to_vec();
                let mut total_pruned = 0usize;
                let mut total_grown = 0usize;
                let mut total_added = 0usize;
                let mut total_removed = 0usize;

                let mut adapt_rng = rand::rngs::StdRng::seed_from_u64(
                    genome_rng_seed.wrapping_add(999)
                );
                // Fresh cooldowns per genome (each genome is new in GA/TS population)
                let mut cooldowns = vec![0usize; num_clusters];

                for _pass in 0..adapt_config.passes_per_eval {
                    // Compute stats — GPU (combined neuron+cluster) or CPU fallback
                    #[cfg(target_os = "macos")]
                    let gpu_stats = metal_stats.as_ref().map(|metal| {
                        adaptation::compute_neuron_and_cluster_stats_gpu(
                            metal, &adapted_bits, &adapted_neurons, &adapted_connections,
                            &cluster_storage, &train_subset.packed_input,
                            train_subset.words_per_example, &train_subset.target_bits,
                            train_subset.num_examples, num_clusters, adapt_config, memory_mode,
                        )
                    });
                    #[cfg(not(target_os = "macos"))]
                    let gpu_stats: Option<(Vec<adaptation::NeuronStats>, Vec<adaptation::ClusterStats>)> = None;

                    let (mut neuron_stats, mut cluster_stats_cache) = match gpu_stats {
                        Some((ns, cs)) => (ns, Some(cs)),
                        None => {
                            let ns = adaptation::compute_neuron_stats(
                                &adapted_bits, &adapted_neurons, &adapted_connections,
                                &cluster_storage, &train_subset.packed_input,
                                train_subset.words_per_example, &train_subset.target_bits,
                                train_subset.num_examples, num_clusters, adapt_config,
                            );
                            (ns, None)
                        }
                    };

                    if adapt_config.synaptogenesis_enabled {
                        let (pruned, grown) = adaptation::synaptogenesis_pass(
                            &mut adapted_bits, &mut adapted_connections,
                            &neuron_stats, adapt_config,
                            &train_subset.packed_input, train_subset.words_per_example,
                            train_subset.num_examples, &mut adapt_rng,
                        );
                        total_pruned += pruned;
                        total_grown += grown;

                        // Recompute stats if synaptogenesis changed architecture
                        if adapt_config.neurogenesis_enabled && (pruned > 0 || grown > 0) {
                            #[cfg(target_os = "macos")]
                            let recomputed = metal_stats.as_ref().map(|metal| {
                                adaptation::compute_neuron_and_cluster_stats_gpu(
                                    metal, &adapted_bits, &adapted_neurons, &adapted_connections,
                                    &cluster_storage, &train_subset.packed_input,
                                    train_subset.words_per_example, &train_subset.target_bits,
                                    train_subset.num_examples, num_clusters, adapt_config, memory_mode,
                                )
                            });
                            #[cfg(not(target_os = "macos"))]
                            let recomputed: Option<(Vec<adaptation::NeuronStats>, Vec<adaptation::ClusterStats>)> = None;

                            match recomputed {
                                Some((ns, cs)) => {
                                    neuron_stats = ns;
                                    cluster_stats_cache = Some(cs);
                                }
                                None => {
                                    neuron_stats = adaptation::compute_neuron_stats(
                                        &adapted_bits, &adapted_neurons, &adapted_connections,
                                        &cluster_storage, &train_subset.packed_input,
                                        train_subset.words_per_example, &train_subset.target_bits,
                                        train_subset.num_examples, num_clusters, adapt_config,
                                    );
                                    cluster_stats_cache = None;
                                }
                            }
                        }
                    }

                    if adapt_config.neurogenesis_enabled {
                        let cluster_stats = cluster_stats_cache.unwrap_or_else(|| {
                            adaptation::compute_cluster_stats(
                                &neuron_stats, &adapted_neurons, &adapted_bits,
                                &adapted_connections, &cluster_storage,
                                &train_subset.packed_input, train_subset.words_per_example,
                                &train_subset.target_bits, train_subset.num_examples, num_clusters,
                                adapt_config,
                            )
                        });
                        let (added, removed) = adaptation::neurogenesis_pass(
                            &mut adapted_bits, &mut adapted_neurons, &mut adapted_connections,
                            &cluster_stats, adapt_config, generation, &mut cooldowns,
                            &mut adapt_rng,
                        );
                        total_added += added;
                        total_removed += removed;
                    }
                }

                let architecture_changed = total_pruned > 0 || total_grown > 0
                    || total_added > 0 || total_removed > 0;

                // Step 3: Re-train with adapted architecture if changed, then forward
                if architecture_changed {
                    let new_cluster_max_bits: Vec<usize> = {
                        let mut v = Vec::with_capacity(num_clusters);
                        let mut off = 0;
                        for &nc in &adapted_neurons {
                            let max_b = adapted_bits[off..off + nc].iter().copied().max().unwrap_or(0);
                            v.push(max_b);
                            off += nc;
                        }
                        v
                    };
                    let new_threshold = match sparse_threshold_override {
                        Some(t) => t,
                        None => auto_sparse_threshold(
                            &new_cluster_max_bits, &adapted_neurons,
                            target_per_genome, expected_train,
                        ),
                    };
                    let new_layout = compute_genome_layout(
                        &adapted_bits, &adapted_neurons, new_threshold, expected_train,
                    );

                    let mut new_storage: Vec<ClusterStorage> = (0..num_clusters)
                        .map(|c| ClusterStorage::new(
                            adapted_neurons[c], new_layout.cluster_max_bits[c],
                            new_threshold, empty_word, memory_mode,
                        ))
                        .collect();

                    train_into(
                        &adapted_connections, &adapted_bits, &adapted_neurons, num_clusters,
                        &new_layout, train_subset, &mut new_storage,
                        memory_mode, neuron_sample_rate, genome_rng_seed.wrapping_add(1),
                    );
                    forward_eval_into(
                        &adapted_connections, &adapted_bits, &adapted_neurons, num_clusters,
                        &new_layout, eval_subset, &new_storage, score_slice, memory_mode,
                    );
                } else {
                    // No architecture change — forward with original storage
                    forward_eval_into(
                        &adapted_connections, &adapted_bits, &adapted_neurons, num_clusters,
                        layout, eval_subset, &cluster_storage, score_slice, memory_mode,
                    );
                }

                // Store adapted genome data
                *adapted_data[g].lock().unwrap() = Some((
                    adapted_bits, adapted_neurons, adapted_connections,
                    total_pruned, total_grown, total_added, total_removed,
                ));
            });

        batch_start = batch_end;
    }

    // Phase 1.5: Compute weighted BitAcc per genome (same as non-adaptive)
    let num_bits = cache.num_bits;
    let mut bit_weights = vec![0.0f64; num_bits];
    {
        let mut bit_ones = vec![0u64; num_bits];
        for ex in 0..num_eval {
            let target = eval_subset.targets[ex] as usize;
            let tb_off = target * num_bits;
            for b in 0..num_bits {
                bit_ones[b] += cache.token_bits[tb_off + b] as u64;
            }
        }
        for b in 0..num_bits {
            let p = bit_ones[b] as f64 / num_eval as f64;
            if p > 0.0 && p < 1.0 {
                bit_weights[b] = (-p * p.ln() - (1.0 - p) * (1.0 - p).ln())
                    / std::f64::consts::LN_2;
            }
        }
    }
    let total_weight: f64 = bit_weights.iter().sum();

    let weighted_bit_accs: Vec<f64> = (0..num_genomes)
        .into_par_iter()
        .map(|g| {
            let score_base = g * scores_per_genome;
            let mut weighted_correct = 0.0f64;
            for ex in 0..num_eval {
                let target = eval_subset.targets[ex] as usize;
                let tb_off = target * num_bits;
                let sc_off = score_base + ex * num_bits;
                for b in 0..num_bits {
                    let predicted = all_scores[sc_off + b] > 0.5;
                    let actual = cache.token_bits[tb_off + b] > 0;
                    if predicted == actual {
                        weighted_correct += bit_weights[b];
                    }
                }
            }
            if total_weight > 0.0 {
                weighted_correct / (num_eval as f64 * total_weight)
            } else {
                0.0
            }
        })
        .collect();

    // Phase 2: CE computation (Metal GPU or CPU fallback)
    let ce_results: Vec<(f64, f64)>;
    #[cfg(target_os = "macos")]
    {
        if let Some(metal) = get_metal_bitwise_ce() {
            match metal.compute_ce_batch(
                &all_scores,
                &cache.token_bits,
                &eval_subset.targets,
                num_genomes,
                num_eval,
                cache.num_bits,
                cache.vocab_size,
            ) {
                Ok(results) => {
                    ce_results = results;
                }
                Err(e) => {
                    eprintln!("[BitwiseCE] Metal failed, CPU fallback: {e}");
                    ce_results = all_scores
                        .par_chunks(scores_per_genome)
                        .map(|scores| compute_ce_cpu(
                            scores, &cache.token_bits, &eval_subset.targets,
                            num_eval, cache.num_bits, cache.vocab_size,
                        ))
                        .collect();
                }
            }
        } else {
            ce_results = all_scores
                .par_chunks(scores_per_genome)
                .map(|scores| compute_ce_cpu(
                    scores, &cache.token_bits, &eval_subset.targets,
                    num_eval, cache.num_bits, cache.vocab_size,
                ))
                .collect();
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        ce_results = all_scores
            .par_chunks(scores_per_genome)
            .map(|scores| compute_ce_cpu(
                scores, &cache.token_bits, &eval_subset.targets,
                num_eval, cache.num_bits, cache.vocab_size,
            ))
            .collect();
    }

    // Combine CE results + adapted data → Vec<AdaptiveResult>
    (0..num_genomes)
        .map(|g| {
            let (ce, acc) = ce_results[g];
            let bit_acc = weighted_bit_accs[g];
            let (a_bits, a_neurons, a_conns, pruned, grown, added, removed) =
                adapted_data[g].lock().unwrap().take().unwrap();
            AdaptiveResult {
                ce, acc, bit_acc,
                adapted_bits: a_bits,
                adapted_neurons: a_neurons,
                adapted_connections: a_conns,
                pruned, grown, added, removed,
            }
        })
        .collect()
}
