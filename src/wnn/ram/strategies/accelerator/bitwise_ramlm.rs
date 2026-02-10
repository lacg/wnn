//! Bitwise RAMLM Accelerator — Full Rust+Metal genome evaluation
//!
//! Evaluates BitwiseRAMLM genomes entirely in Rust, avoiding Python overhead.
//! Pipeline per genome:
//!   1. Train (CPU, sequential per genome) — two-phase TRUE/FALSE writes
//!   2. Forward pass (CPU, sequential per genome) — memory reads + probability
//!   3. Reconstruction + CE (Metal GPU) — 50K vocab × 16 bits matmul
//!
//! Parallelism is at the GENOME level only (outer rayon par_iter).
//! Each genome runs train+forward sequentially with non-atomic memory ops.
//! This avoids nested rayon which serialized genome processing.
//!
//! BitwiseTokenCache holds pre-encoded tokens (created once).

use metal::*;
use rayon::prelude::*;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use crate::ramlm;

// =============================================================================
// Metal Bitwise CE Evaluator
// =============================================================================

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

// =============================================================================
// Token Cache
// =============================================================================

/// Pre-encoded token subset for bitwise training.
pub struct BitwiseSubset {
    pub input_bits: Vec<bool>,  // [N * total_input_bits]
    pub target_bits: Vec<u8>,   // [N * num_bits]
    pub num_examples: usize,
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

    /// Eval data
    pub eval_input_bits: Vec<bool>, // [num_eval * total_input_bits]
    pub eval_targets: Vec<u32>,     // [num_eval] token IDs
    pub num_eval: usize,

    pub num_parts: usize,
    current_train_idx: AtomicUsize,
}

fn bits_needed(n: usize) -> usize {
    if n <= 1 { return 1; }
    ((n as f64).log2().ceil()) as usize
}

impl BitwiseTokenCache {
    pub fn new(
        train_tokens: Vec<u32>,
        eval_tokens: Vec<u32>,
        vocab_size: usize,
        context_size: usize,
        num_parts: usize,
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
        let (full_input, _full_targets, full_target_bits, full_n) =
            Self::encode_sequence(&train_tokens, context_size, bits_per_token, total_input_bits, num_bits);
        let full_train = BitwiseSubset {
            input_bits: full_input,
            target_bits: full_target_bits,
            num_examples: full_n,
        };

        // Split training data into subsets
        let train_subsets = Self::split_and_encode(
            &train_tokens, context_size, bits_per_token, total_input_bits, num_bits, num_parts,
        );

        // Encode eval data
        let (eval_input, eval_targets, _, num_eval) =
            Self::encode_sequence(&eval_tokens, context_size, bits_per_token, total_input_bits, num_bits);

        eprintln!(
            "[BitwiseCache] vocab={vocab_size}, context={context_size}, bits={num_bits}, \
             train={full_n} examples ({num_parts} subsets), eval={num_eval} examples"
        );

        Self {
            vocab_size, context_size, bits_per_token, total_input_bits, num_bits,
            token_bits, train_subsets, full_train,
            eval_input_bits: eval_input, eval_targets, num_eval,
            num_parts,
            current_train_idx: AtomicUsize::new(0),
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
                BitwiseSubset { input_bits, target_bits, num_examples: num_ex }
            })
            .collect()
    }

    /// Get next training subset index (advances rotator).
    pub fn next_train_idx(&self) -> usize {
        self.current_train_idx.fetch_add(1, Ordering::Relaxed) % self.num_parts
    }

    /// Reset subset rotation.
    pub fn reset(&self) {
        self.current_train_idx.store(0, Ordering::Relaxed);
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

const BITS_PER_CELL: usize = 2;
const CELLS_PER_WORD: usize = 31;
const CELL_MASK: i64 = 0b11;
const FALSE: i64 = 0;
const TRUE: i64 = 1;
const EMPTY: i64 = 2;

// Quad mode constants
const QUAD_FALSE: i64 = 0;
const QUAD_WEAK_FALSE: i64 = 1;
const QUAD_WEAK_TRUE: i64 = 2;
const QUAD_TRUE: i64 = 3;
const QUAD_WEIGHTS: [f32; 4] = [0.0, 0.25, 0.75, 1.0];

// Memory modes
const MODE_TERNARY: u8 = 0;
const MODE_QUAD_BINARY: u8 = 1;
const MODE_QUAD_WEIGHTED: u8 = 2;

/// Inline xorshift32 PRNG
#[inline]
fn xorshift32_seq(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state >> 8) as f32 / 16777216.0
}

/// Compute memory address for a single neuron given input bits (same as ramlm::compute_address).
#[inline]
fn compute_address_seq(input_bits: &[bool], connections: &[i64], bits_per_neuron: usize) -> usize {
    let mut address: usize = 0;
    for (i, &conn_idx) in connections.iter().take(bits_per_neuron).enumerate() {
        if input_bits[conn_idx as usize] {
            address |= 1 << (bits_per_neuron - 1 - i);
        }
    }
    address
}

/// Read a memory cell value (non-atomic, for single-threaded per-genome use).
#[inline]
fn read_cell_seq(memory: &[i64], neuron_idx: usize, address: usize, words_per_neuron: usize) -> i64 {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = neuron_idx * words_per_neuron + word_idx;
    (memory[word_offset] >> (cell_idx * BITS_PER_CELL)) & CELL_MASK
}

/// Write a memory cell value (non-atomic, for single-threaded per-genome use).
/// Returns true if the cell was actually modified.
#[inline]
fn write_cell_seq(
    memory: &mut [i64],
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
    let old_cell = (memory[word_offset] >> shift) & CELL_MASK;

    if !allow_override && old_cell != EMPTY {
        return false;
    }
    if old_cell == value {
        return false;
    }

    let mask = CELL_MASK << shift;
    memory[word_offset] = (memory[word_offset] & !mask) | (value << shift);
    true
}

/// Nudge a memory cell one step toward target (non-atomic, sequential).
/// target_true: cell = min(cell + 1, 3)
/// target_false: cell = max(cell - 1, 0)
#[inline]
fn nudge_cell_seq(
    memory: &mut [i64],
    neuron_idx: usize,
    address: usize,
    target_true: bool,
    words_per_neuron: usize,
) -> bool {
    let word_idx = address / CELLS_PER_WORD;
    let cell_idx = address % CELLS_PER_WORD;
    let word_offset = neuron_idx * words_per_neuron + word_idx;
    let shift = cell_idx * BITS_PER_CELL;
    let old_cell = (memory[word_offset] >> shift) & CELL_MASK;

    let new_cell = if target_true {
        (old_cell + 1).min(QUAD_TRUE)
    } else {
        (old_cell - 1).max(QUAD_FALSE)
    };

    if new_cell == old_cell {
        return false;
    }

    let mask = CELL_MASK << shift;
    memory[word_offset] = (memory[word_offset] & !mask) | (new_cell << shift);
    true
}

/// Train a single genome into pre-allocated memory, then forward pass into output slice.
///
/// Supports three memory modes:
///   MODE_TERNARY (0): Majority vote (existing, default)
///   MODE_QUAD_BINARY (1): 4-state nudging, binary threshold forward
///   MODE_QUAD_WEIGHTED (2): 4-state nudging, weighted confidence forward
///
/// Fully sequential (no rayon, no atomics) — designed to be called from
/// an outer par_iter over genomes.
fn train_and_forward_into(
    cache: &BitwiseTokenCache,
    connections: &[i64],
    neurons_per_cluster: usize,
    bits_per_neuron: usize,
    train_subset: &BitwiseSubset,
    memory: &mut [i64],
    empty_word: i64,
    words_per_neuron: usize,
    out_probs: &mut [f32],
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) {
    let num_clusters = cache.num_bits;
    let num_examples = train_subset.num_examples;
    let total_input_bits = cache.total_input_bits;
    let address_space = 1usize << bits_per_neuron;

    // Reset memory
    memory.fill(empty_word);

    // Per-genome PRNG state
    let mut rng_state = (rng_seed as u32).wrapping_add(17);
    if rng_state == 0 { rng_state = 1; }

    match memory_mode {
        MODE_TERNARY => {
            // ===== TRAINING: Majority vote, one cluster at a time =====
            let mut votes = vec![0i32; neurons_per_cluster * address_space];

            for cluster in 0..num_clusters {
                votes.fill(0);
                let start_neuron = cluster * neurons_per_cluster;

                for ex in 0..num_examples {
                    let input_bits = &train_subset.input_bits[ex * total_input_bits..(ex + 1) * total_input_bits];
                    let target_bit = train_subset.target_bits[ex * num_clusters + cluster];
                    let vote: i32 = if target_bit == 1 { 1 } else { -1 };

                    for n in 0..neurons_per_cluster {
                        if neuron_sample_rate < 1.0 {
                            if xorshift32_seq(&mut rng_state) >= neuron_sample_rate {
                                continue;
                            }
                        }
                        let neuron_idx = start_neuron + n;
                        let conn_start = neuron_idx * bits_per_neuron;
                        let conns = &connections[conn_start..conn_start + bits_per_neuron];
                        let addr = compute_address_seq(input_bits, conns, bits_per_neuron);
                        votes[n * address_space + addr] += vote;
                    }
                }

                for n in 0..neurons_per_cluster {
                    let neuron_idx = start_neuron + n;
                    for addr in 0..address_space {
                        let v = votes[n * address_space + addr];
                        if v > 0 {
                            write_cell_seq(memory, neuron_idx, addr, TRUE, words_per_neuron, true);
                        } else if v < 0 {
                            write_cell_seq(memory, neuron_idx, addr, FALSE, words_per_neuron, true);
                        }
                    }
                }
            }
        }
        MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => {
            // ===== TRAINING: Sequential nudging =====
            for cluster in 0..num_clusters {
                let start_neuron = cluster * neurons_per_cluster;

                for ex in 0..num_examples {
                    let input_bits = &train_subset.input_bits[ex * total_input_bits..(ex + 1) * total_input_bits];
                    let target_bit = train_subset.target_bits[ex * num_clusters + cluster];
                    let target_true = target_bit == 1;

                    for n in 0..neurons_per_cluster {
                        if neuron_sample_rate < 1.0 {
                            if xorshift32_seq(&mut rng_state) >= neuron_sample_rate {
                                continue;
                            }
                        }
                        let neuron_idx = start_neuron + n;
                        let conn_start = neuron_idx * bits_per_neuron;
                        let conns = &connections[conn_start..conn_start + bits_per_neuron];
                        let addr = compute_address_seq(input_bits, conns, bits_per_neuron);
                        nudge_cell_seq(memory, neuron_idx, addr, target_true, words_per_neuron);
                    }
                }
            }
        }
        _ => {
            // Unknown mode — treat as ternary
            eprintln!("[BitwiseRAMLM] Unknown memory_mode={memory_mode}, falling back to TERNARY");
        }
    }

    // ===== FORWARD PASS on eval data =====
    let num_eval = cache.num_eval;
    let empty_value = ramlm::get_empty_value();

    for ex in 0..num_eval {
        let input_bits = &cache.eval_input_bits[ex * total_input_bits..(ex + 1) * total_input_bits];
        let ex_probs = &mut out_probs[ex * num_clusters..(ex + 1) * num_clusters];

        for cluster in 0..num_clusters {
            let start_neuron = cluster * neurons_per_cluster;

            match memory_mode {
                MODE_TERNARY => {
                    let mut count_true = 0u32;
                    let mut count_empty = 0u32;
                    for n in 0..neurons_per_cluster {
                        let neuron_idx = start_neuron + n;
                        let conn_start = neuron_idx * bits_per_neuron;
                        let conns = &connections[conn_start..conn_start + bits_per_neuron];
                        let addr = compute_address_seq(input_bits, conns, bits_per_neuron);
                        let cell = read_cell_seq(memory, neuron_idx, addr, words_per_neuron);
                        if cell == TRUE {
                            count_true += 1;
                        } else if cell == EMPTY {
                            count_empty += 1;
                        }
                    }
                    ex_probs[cluster] = (count_true as f32 + empty_value * count_empty as f32)
                        / neurons_per_cluster as f32;
                }
                MODE_QUAD_BINARY => {
                    let mut count_true = 0u32;
                    for n in 0..neurons_per_cluster {
                        let neuron_idx = start_neuron + n;
                        let conn_start = neuron_idx * bits_per_neuron;
                        let conns = &connections[conn_start..conn_start + bits_per_neuron];
                        let addr = compute_address_seq(input_bits, conns, bits_per_neuron);
                        let cell = read_cell_seq(memory, neuron_idx, addr, words_per_neuron);
                        if cell >= QUAD_WEAK_TRUE {
                            count_true += 1;
                        }
                    }
                    ex_probs[cluster] = count_true as f32 / neurons_per_cluster as f32;
                }
                MODE_QUAD_WEIGHTED => {
                    let mut weighted_sum = 0.0f32;
                    for n in 0..neurons_per_cluster {
                        let neuron_idx = start_neuron + n;
                        let conn_start = neuron_idx * bits_per_neuron;
                        let conns = &connections[conn_start..conn_start + bits_per_neuron];
                        let addr = compute_address_seq(input_bits, conns, bits_per_neuron);
                        let cell = read_cell_seq(memory, neuron_idx, addr, words_per_neuron);
                        weighted_sum += QUAD_WEIGHTS[cell as usize];
                    }
                    ex_probs[cluster] = weighted_sum / neurons_per_cluster as f32;
                }
                _ => {
                    ex_probs[cluster] = 0.5; // fallback
                }
            }
        }
    }
}

/// Evaluate multiple genomes using subset training data.
pub fn evaluate_genomes(
    cache: &BitwiseTokenCache,
    all_connections_flat: &[i64],
    num_genomes: usize,
    neurons_per_cluster: usize,
    bits_per_neuron: usize,
    train_subset_idx: usize,
) -> Vec<(f64, f64)> {
    let train_subset = &cache.train_subsets[train_subset_idx % cache.num_parts];
    evaluate_genomes_with_subset(
        cache, all_connections_flat, num_genomes,
        neurons_per_cluster, bits_per_neuron, train_subset,
        MODE_TERNARY, 1.0, 42,
    )
}

/// Evaluate with mode and sample rate (subset).
pub fn evaluate_genomes_with_mode(
    cache: &BitwiseTokenCache,
    all_connections_flat: &[i64],
    num_genomes: usize,
    neurons_per_cluster: usize,
    bits_per_neuron: usize,
    train_subset_idx: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64)> {
    let train_subset = &cache.train_subsets[train_subset_idx % cache.num_parts];
    evaluate_genomes_with_subset(
        cache, all_connections_flat, num_genomes,
        neurons_per_cluster, bits_per_neuron, train_subset,
        memory_mode, neuron_sample_rate, rng_seed,
    )
}

/// Evaluate multiple genomes using full training data.
pub fn evaluate_genomes_full(
    cache: &BitwiseTokenCache,
    all_connections_flat: &[i64],
    num_genomes: usize,
    neurons_per_cluster: usize,
    bits_per_neuron: usize,
) -> Vec<(f64, f64)> {
    evaluate_genomes_with_subset(
        cache, all_connections_flat, num_genomes,
        neurons_per_cluster, bits_per_neuron, &cache.full_train,
        MODE_TERNARY, 1.0, 42,
    )
}

/// Evaluate full with mode and sample rate.
pub fn evaluate_genomes_full_with_mode(
    cache: &BitwiseTokenCache,
    all_connections_flat: &[i64],
    num_genomes: usize,
    neurons_per_cluster: usize,
    bits_per_neuron: usize,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64)> {
    evaluate_genomes_with_subset(
        cache, all_connections_flat, num_genomes,
        neurons_per_cluster, bits_per_neuron, &cache.full_train,
        memory_mode, neuron_sample_rate, rng_seed,
    )
}

/// Core evaluation: parallel train+forward, then batched CE.
fn evaluate_genomes_with_subset(
    cache: &BitwiseTokenCache,
    all_connections_flat: &[i64],
    num_genomes: usize,
    neurons_per_cluster: usize,
    bits_per_neuron: usize,
    train_subset: &BitwiseSubset,
    memory_mode: u8,
    neuron_sample_rate: f32,
    rng_seed: u64,
) -> Vec<(f64, f64)> {
    let num_clusters = cache.num_bits;
    let conns_per_genome = num_clusters * neurons_per_cluster * bits_per_neuron;
    let total_neurons = num_clusters * neurons_per_cluster;
    let addresses = 1usize << bits_per_neuron;
    let words_per_neuron = (addresses + CELLS_PER_WORD - 1) / CELLS_PER_WORD;
    let memory_size = total_neurons * words_per_neuron;
    let scores_per_genome = cache.num_eval * num_clusters;

    // Compute empty_word based on memory mode
    let empty_word: i64 = match memory_mode {
        MODE_QUAD_BINARY | MODE_QUAD_WEIGHTED => {
            // All cells = QUAD_WEAK_FALSE (1)
            (0..31i64).fold(0i64, |acc, i| acc | (QUAD_WEAK_FALSE << (i * 2)))
        }
        _ => {
            // All cells = EMPTY (2) — ternary mode
            (0..31i64).fold(0i64, |acc, i| acc | (EMPTY << (i * 2)))
        }
    };

    // Pre-allocate ALL memory and output buffers before parallel execution.
    let mut all_memory: Vec<i64> = vec![empty_word; num_genomes * memory_size];
    let mut all_scores: Vec<f32> = vec![0.0f32; num_genomes * scores_per_genome];

    // Phase 1: Train + Forward (CPU, parallel over genomes)
    all_memory
        .par_chunks_mut(memory_size)
        .zip(all_scores.par_chunks_mut(scores_per_genome))
        .enumerate()
        .for_each(|(g, (mem_slice, score_slice))| {
            let conn_start = g * conns_per_genome;
            let connections = &all_connections_flat[conn_start..conn_start + conns_per_genome];
            // Per-genome RNG seed
            let genome_rng_seed = rng_seed.wrapping_add(g as u64 * 1000007);
            train_and_forward_into(
                cache, connections, neurons_per_cluster, bits_per_neuron,
                train_subset, mem_slice, empty_word, words_per_neuron, score_slice,
                memory_mode, neuron_sample_rate, genome_rng_seed,
            );
        });

    // Phase 2: Reconstruction + CE
    if let Some(metal) = get_metal_bitwise_ce() {
        match metal.compute_ce_batch(
            &all_scores,
            &cache.token_bits,
            &cache.eval_targets,
            num_genomes,
            cache.num_eval,
            cache.num_bits,
            cache.vocab_size,
        ) {
            Ok(results) => return results,
            Err(e) => {
                eprintln!("[BitwiseCE] Metal failed, CPU fallback: {e}");
            }
        }
    }

    // CPU fallback (parallel over genomes)
    all_scores
        .par_chunks(scores_per_genome)
        .map(|scores| {
            compute_ce_cpu(
                scores,
                &cache.token_bits,
                &cache.eval_targets,
                cache.num_eval,
                cache.num_bits,
                cache.vocab_size,
            )
        })
        .collect()
}
