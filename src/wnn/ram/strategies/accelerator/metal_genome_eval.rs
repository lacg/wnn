//! GPU genome-batch evaluation for GA architecture search (IDS classification).
//!
//! IDS-domain, NOT language modeling: `MetalSparseCEEvaluator` scores
//! CE/accuracy on-GPU and `MetalGroupEvaluator` dispatches many genomes in one
//! batch (with a reusable buffer cache). These drive the GA search; the dense
//! LM forward (`MetalRAMLMEvaluator`) lives in `metal_ramlm.rs`, and the shared
//! sparse forward (`MetalSparseEvaluator`) lives in `ram_core::metal_sparse`.
//! Extracted from the former monolithic metal_ramlm.rs (2026-06-19 split).

use metal::*;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};
use ram_core::metal_sparse::default_cell_for_mode;

// =============================================================================
// Buffer Cache for Per-Group Evaluation
// =============================================================================

/// Global counter for cache invalidation (incremented by reset_sparse_buffer_cache)
static SPARSE_CACHE_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Reset the sparse buffer cache (call when Metal evaluators are reset)
pub fn reset_sparse_buffer_cache() {
    SPARSE_CACHE_GENERATION.fetch_add(1, Ordering::SeqCst);
}

/// Get current cache generation (for cache validation)
pub fn get_sparse_cache_generation() -> u64 {
    SPARSE_CACHE_GENERATION.load(Ordering::SeqCst)
}

/// Cached buffer with capacity tracking
struct CachedBuffer {
    buffer: Buffer,
    capacity_bytes: u64,
    cache_gen: u64,
}

/// Thread-local cache for dense evaluation buffers
/// Avoids 4 buffer allocations per dense group call
struct DenseBufferCache {
    conn_buffer: Option<CachedBuffer>,      // i32 connections
    memory_buffer: Option<CachedBuffer>,    // i64 memory words
    cluster_ids_buffer: Option<CachedBuffer>, // u32 cluster IDs
    params_buffer: Option<CachedBuffer>,    // DenseToBufferParams struct
}

impl DenseBufferCache {
    fn new() -> Self {
        Self {
            conn_buffer: None,
            memory_buffer: None,
            cluster_ids_buffer: None,
            params_buffer: None,
        }
    }
}

/// Thread-local cache for CE reduction buffers
/// Avoids buffer allocations per genome in compute_ce_from_buffer
struct CEBufferCache {
    targets_buffer: Option<CachedBuffer>,
    ce_buffer: Option<CachedBuffer>,
    correct_buffer: Option<CachedBuffer>,
    predicted_buffer: Option<CachedBuffer>,
    // Track the targets data to know if we need to update
    cached_targets_hash: u64,
}

impl CEBufferCache {
    fn new() -> Self {
        Self {
            targets_buffer: None,
            ce_buffer: None,
            correct_buffer: None,
            predicted_buffer: None,
            cached_targets_hash: 0,
        }
    }
}

/// Thread-local pool for batched sparse evaluation
/// Maintains multiple buffer sets (one per sparse group in a batch)
/// Buffers grow as needed but are reused across batches
struct BatchedSparseBufferPool {
    conn_buffers: Vec<Option<CachedBuffer>>,
    keys_buffers: Vec<Option<CachedBuffer>>,
    values_buffers: Vec<Option<CachedBuffer>>,
    offsets_buffers: Vec<Option<CachedBuffer>>,
    counts_buffers: Vec<Option<CachedBuffer>>,
    cluster_ids_buffers: Vec<Option<CachedBuffer>>,
    params_buffers: Vec<Option<CachedBuffer>>,
    actual_neurons_buffers: Vec<Option<CachedBuffer>>,  // For masked groups
}

impl BatchedSparseBufferPool {
    fn new() -> Self {
        Self {
            conn_buffers: Vec::new(),
            keys_buffers: Vec::new(),
            values_buffers: Vec::new(),
            offsets_buffers: Vec::new(),
            counts_buffers: Vec::new(),
            cluster_ids_buffers: Vec::new(),
            params_buffers: Vec::new(),
            actual_neurons_buffers: Vec::new(),
        }
    }

    /// Ensure pool has at least `count` slots
    fn ensure_capacity(&mut self, count: usize) {
        while self.conn_buffers.len() < count {
            self.conn_buffers.push(None);
            self.keys_buffers.push(None);
            self.values_buffers.push(None);
            self.offsets_buffers.push(None);
            self.counts_buffers.push(None);
            self.cluster_ids_buffers.push(None);
            self.params_buffers.push(None);
            self.actual_neurons_buffers.push(None);
        }
    }

    /// Clear buffers beyond the given count to prevent memory accumulation
    /// from previous batches with more sparse groups
    fn clear_beyond(&mut self, count: usize) {
        for i in count..self.conn_buffers.len() {
            self.conn_buffers[i] = None;
            self.keys_buffers[i] = None;
            self.values_buffers[i] = None;
            self.offsets_buffers[i] = None;
            self.counts_buffers[i] = None;
            self.cluster_ids_buffers[i] = None;
            self.params_buffers[i] = None;
            self.actual_neurons_buffers[i] = None;
        }
    }
}

thread_local! {
    static DENSE_BUFFER_CACHE: std::cell::RefCell<DenseBufferCache> =
        std::cell::RefCell::new(DenseBufferCache::new());
    static CE_BUFFER_CACHE: std::cell::RefCell<CEBufferCache> =
        std::cell::RefCell::new(CEBufferCache::new());
    static BATCHED_SPARSE_BUFFER_POOL: std::cell::RefCell<BatchedSparseBufferPool> =
        std::cell::RefCell::new(BatchedSparseBufferPool::new());
}

/// Get or create a cached buffer, writing data directly to it
/// Returns the buffer to use for the GPU operation
fn get_or_create_buffer<T>(
    device: &Device,
    cached: &mut Option<CachedBuffer>,
    data: &[T],
    current_gen: u64,
) -> Buffer {
    let required_bytes = (data.len() * mem::size_of::<T>()) as u64;

    // Check if cached buffer can be reused
    if let Some(ref cache) = cached {
        if cache.cache_gen == current_gen && cache.capacity_bytes >= required_bytes {
            // Reuse buffer - write data directly to contents
            let ptr = cache.buffer.contents() as *mut T;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
            return cache.buffer.clone();
        }
    }

    // Need new buffer - allocate with 50% headroom for future reuse
    // This reduces buffer thrashing when genome sizes vary within a batch
    let alloc_bytes = ((required_bytes as f64 * 1.5) as u64).max(1024);

    // Create buffer with headroom capacity
    let buffer = device.new_buffer(
        alloc_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    // Copy data to the buffer
    let ptr = buffer.contents() as *mut T;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }

    *cached = Some(CachedBuffer {
        buffer: buffer.clone(),
        capacity_bytes: alloc_bytes,
        cache_gen: current_gen,
    });

    buffer
}

// ============================================================================
// Batched Sparse Evaluator - Multiple Genomes in One Dispatch
// ============================================================================

// ============================================================================
// Sparse CE Evaluator - Computes CE/Accuracy Directly on GPU
// ============================================================================

/// Parameters for sparse CE computation (must match Metal struct)
#[repr(C)]
struct SparseCEParams {
    num_examples: u32,
    words_per_example: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    neurons_per_cluster: u32,
    num_clusters: u32,
    empty_value: f32,
    memory_mode: u32,
    default_cell_value: u32,
    run_seed: u64,
}

/// Metal evaluator that computes CE and accuracy directly on GPU
///
/// Instead of returning all probabilities (10GB for 50K×50K), this evaluator
/// computes cross-entropy and accuracy ON THE GPU and returns just the results.
/// This eliminates the massive GPU→CPU data transfer.
pub struct MetalSparseCEEvaluator {
    device: Device,
    command_queue: CommandQueue,
    ce_online_pipeline: ComputePipelineState,
}

impl MetalSparseCEEvaluator {
    /// Create new sparse CE evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile sparse CE shader
        let shader_source = concat!(include_str!("core/shaders/common.metal"), "\n", include_str!("shaders/sparse_ce.metal"));
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile sparse CE shader: {}", e))?;

        let ce_online_kernel = library
            .get_function("sparse_forward_with_ce_online", None)
            .map_err(|e| format!("Failed to get sparse_forward_with_ce_online: {}", e))?;

        let ce_online_pipeline = device
            .new_compute_pipeline_state_with_function(&ce_online_kernel)
            .map_err(|e| format!("Failed to create CE online pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            ce_online_pipeline,
        })
    }

    /// Compute CE and accuracy directly on GPU
    ///
    /// Returns (average_ce, accuracy) instead of all probabilities.
    /// This eliminates the 10GB data transfer for 50K×50K.
    ///
    /// Args:
    ///   packed_input: [num_examples * words_per_example] packed u64
    ///   connections: [num_neurons * bits_per_neuron]
    ///   keys: Sorted addresses for all neurons
    ///   values: Corresponding cell values
    ///   offsets: [num_neurons] start index per neuron
    ///   counts: [num_neurons] count of entries per neuron
    ///   targets: [num_examples] target cluster for each example
    ///   params: Evaluation parameters
    ///
    /// Returns: (average_ce, accuracy)
    pub fn compute_ce(
        &self,
        packed_input: &[u64],
        connections: &[i64],
        keys: &[u64],
        values: &[u8],
        offsets: &[u32],
        counts: &[u32],
        targets: &[i64],
        num_examples: usize,
        words_per_example: usize,
        num_neurons: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        empty_value: f32,
        memory_mode: u8,
        run_seed: u64,
    ) -> Result<(f64, f64, Vec<u32>), String> {
        if num_examples == 0 {
            return Ok((0.0, 0.0, vec![]));
        }

        // Convert connections to i32 for Metal
        let connections_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();

        // Convert targets to i32
        let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();

        // Create Metal buffers
        let input_buffer = self.device.new_buffer_with_data(
            packed_input.as_ptr() as *const _,
            (packed_input.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let connections_buffer = self.device.new_buffer_with_data(
            connections_i32.as_ptr() as *const _,
            (connections_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let keys_buffer = self.device.new_buffer_with_data(
            keys.as_ptr() as *const _,
            (keys.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let values_buffer = self.device.new_buffer_with_data(
            values.as_ptr() as *const _,
            (values.len() * mem::size_of::<u8>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let offsets_buffer = self.device.new_buffer_with_data(
            offsets.as_ptr() as *const _,
            (offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let counts_buffer = self.device.new_buffer_with_data(
            counts.as_ptr() as *const _,
            (counts.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let targets_buffer = self.device.new_buffer_with_data(
            targets_i32.as_ptr() as *const _,
            (targets_i32.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params = SparseCEParams {
            num_examples: num_examples as u32,
            words_per_example: words_per_example as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_clusters: num_clusters as u32,
            empty_value,
            memory_mode: memory_mode as u32,
            default_cell_value: default_cell_for_mode(memory_mode),
            run_seed,
        };

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<SparseCEParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffers: one float CE, one uint correct, one uint predicted per example
        let ce_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let correct_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let predicted_buffer = self.device.new_buffer(
            (num_examples * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ce_online_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&connections_buffer), 0);
        encoder.set_buffer(2, Some(&keys_buffer), 0);
        encoder.set_buffer(3, Some(&values_buffer), 0);
        encoder.set_buffer(4, Some(&offsets_buffer), 0);
        encoder.set_buffer(5, Some(&counts_buffer), 0);
        encoder.set_buffer(6, Some(&targets_buffer), 0);
        encoder.set_buffer(7, Some(&params_buffer), 0);
        encoder.set_buffer(8, Some(&ce_buffer), 0);
        encoder.set_buffer(9, Some(&correct_buffer), 0);
        encoder.set_buffer(10, Some(&predicted_buffer), 0);

        // Grid: one thread per example
        let grid_size = MTLSize::new(num_examples as u64, 1, 1);
        let max_threads = self.ce_online_pipeline.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(num_examples as u64), 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results and reduce on CPU
        let ce_ptr = ce_buffer.contents() as *const f32;
        let correct_ptr = correct_buffer.contents() as *const u32;
        let predicted_ptr = predicted_buffer.contents() as *const u32;

        let (total_ce, total_correct, predictions): (f64, u64, Vec<u32>) = unsafe {
            let ce_slice = std::slice::from_raw_parts(ce_ptr, num_examples);
            let correct_slice = std::slice::from_raw_parts(correct_ptr, num_examples);
            let predicted_slice = std::slice::from_raw_parts(predicted_ptr, num_examples);

            let total_ce: f64 = ce_slice.iter().map(|&c| c as f64).sum();
            let total_correct: u64 = correct_slice.iter().map(|&c| c as u64).sum();
            let predictions: Vec<u32> = predicted_slice.to_vec();

            (total_ce, total_correct, predictions)
        };

        let avg_ce = total_ce / num_examples as f64;
        let accuracy = total_correct as f64 / num_examples as f64;

        Ok((avg_ce, accuracy, predictions))
    }
}

// ============================================================================
// CE Reduction Evaluator - For Tiered Configs
// ============================================================================

/// Parameters for CE reduction (must match Metal struct)
#[repr(C)]
struct CEReduceParams {
    num_examples: u32,
    num_clusters: u32,
}

// ============================================================================
// Unified Group Evaluator - Writes Directly to Shared GPU Buffer
// ============================================================================

/// Parameters for sparse forward to buffer (must match Metal struct)
#[repr(C)]
struct SparseToBufferParams {
    num_examples: u32,
    words_per_example: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    neurons_per_cluster: u32,
    num_group_clusters: u32,
    total_clusters: u32,
    empty_value: f32,
    memory_mode: u32,
    default_cell_value: u32,
    run_seed: u64,
}

/// Parameters for sparse forward to buffer with per-cluster masking (must match Metal struct)
/// Used when clusters are coalesced by neuron bucket (e.g., 5-7 neurons → max 7)
#[repr(C)]
struct SparseToBufferMaskedParams {
    num_examples: u32,
    words_per_example: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    max_neurons_per_cluster: u32,  // Max neurons for memory layout
    num_group_clusters: u32,
    total_clusters: u32,
    empty_value: f32,
    memory_mode: u32,
    default_cell_value: u32,
    run_seed: u64,
}

/// Parameters for dense forward to buffer (must match Metal struct)
#[repr(C)]
struct DenseToBufferParams {
    num_examples: u32,
    words_per_example: u32,
    num_neurons: u32,
    bits_per_neuron: u32,
    neurons_per_cluster: u32,
    num_group_clusters: u32,
    total_clusters: u32,
    words_per_neuron: u32,
    empty_value: f32,
    memory_mode: u32,
    run_seed: u64,
}

/// Data for a single sparse group in batched evaluation
pub struct SparseGroupData<'a> {
    pub connections: &'a [i64],
    pub keys: &'a [u64],
    pub values: &'a [u8],
    pub offsets: &'a [u32],
    pub counts: &'a [u32],
    pub cluster_ids: &'a [usize],
    pub bits_per_neuron: usize,
    pub neurons_per_cluster: usize,
    /// Actual neurons per cluster (for masked groups), None if uniform
    pub actual_neurons_per_cluster: Option<&'a [u32]>,
}

/// Unified Metal evaluator that writes group results directly to shared GPU buffer
///
/// This avoids the GPU→CPU→GPU round-trip that was slowing down tiered evaluation.
/// All groups write to the same shared buffer on GPU, then CE is computed once.
pub struct MetalGroupEvaluator {
    device: Device,
    command_queue: CommandQueue,
    sparse_to_buffer_pipeline: ComputePipelineState,
    sparse_to_buffer_masked_pipeline: ComputePipelineState,  // For coalesced groups with per-cluster masking
    dense_to_buffer_pipeline: ComputePipelineState,
    ce_reduce_pipeline: ComputePipelineState,
}

impl MetalGroupEvaluator {
    /// Create new unified group evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile sparse forward shader
        let sparse_shader = concat!(include_str!("core/shaders/common.metal"), "\n", include_str!("core/shaders/sparse_forward.metal"));
        let sparse_library = device
            .new_library_with_source(sparse_shader, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile sparse shader: {}", e))?;

        let sparse_to_buffer_kernel = sparse_library
            .get_function("sparse_forward_to_buffer", None)
            .map_err(|e| format!("Failed to get sparse_forward_to_buffer: {}", e))?;

        let sparse_to_buffer_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_to_buffer_kernel)
            .map_err(|e| format!("Failed to create sparse to buffer pipeline: {}", e))?;

        let sparse_to_buffer_masked_kernel = sparse_library
            .get_function("sparse_forward_to_buffer_masked", None)
            .map_err(|e| format!("Failed to get sparse_forward_to_buffer_masked: {}", e))?;

        let sparse_to_buffer_masked_pipeline = device
            .new_compute_pipeline_state_with_function(&sparse_to_buffer_masked_kernel)
            .map_err(|e| format!("Failed to create sparse masked pipeline: {}", e))?;

        // Compile dense forward shader
        let dense_shader = concat!(include_str!("core/shaders/common.metal"), "\n", include_str!("shaders/ramlm.metal"));
        let dense_library = device
            .new_library_with_source(dense_shader, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile dense shader: {}", e))?;

        let dense_to_buffer_kernel = dense_library
            .get_function("ramlm_forward_to_buffer", None)
            .map_err(|e| format!("Failed to get ramlm_forward_to_buffer: {}", e))?;

        let dense_to_buffer_pipeline = device
            .new_compute_pipeline_state_with_function(&dense_to_buffer_kernel)
            .map_err(|e| format!("Failed to create dense to buffer pipeline: {}", e))?;

        // Compile CE reduce shader
        let ce_shader = include_str!("shaders/ce_reduce.metal");
        let ce_library = device
            .new_library_with_source(ce_shader, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile CE reduce shader: {}", e))?;

        let ce_reduce_kernel = ce_library
            .get_function("reduce_scores_to_ce", None)
            .map_err(|e| format!("Failed to get reduce_scores_to_ce: {}", e))?;

        let ce_reduce_pipeline = device
            .new_compute_pipeline_state_with_function(&ce_reduce_kernel)
            .map_err(|e| format!("Failed to create CE reduce pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            sparse_to_buffer_pipeline,
            sparse_to_buffer_masked_pipeline,
            dense_to_buffer_pipeline,
            ce_reduce_pipeline,
        })
    }

    /// Create shared scores buffer initialized to 0
    pub fn create_scores_buffer(&self, num_examples: usize, num_clusters: usize) -> Buffer {
        let size = num_examples * num_clusters;
        let zeros: Vec<f32> = vec![0.0; size];
        self.device.new_buffer_with_data(
            zeros.as_ptr() as *const _,
            (size * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Zero an existing scores buffer (much faster than creating a new one)
    /// Uses direct memory write since StorageModeShared buffers are CPU-accessible
    pub fn zero_scores_buffer(&self, buffer: &Buffer, num_examples: usize, num_clusters: usize) {
        let size = num_examples * num_clusters;
        let ptr = buffer.contents() as *mut f32;
        // Safety: buffer was created with StorageModeShared, size matches
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
    }

    /// Update an existing input buffer with new packed u64 data (much faster than creating a new one)
    pub fn update_input_buffer(&self, buffer: &Buffer, packed_input: &[u64]) {
        let ptr = buffer.contents() as *mut u64;
        // Safety: buffer was created with StorageModeShared, size matches
        unsafe {
            std::ptr::copy_nonoverlapping(packed_input.as_ptr(), ptr, packed_input.len());
        }
    }

    /// Create packed u64 input buffer (shared across all group evaluations)
    pub fn create_input_buffer(&self, packed_input: &[u64]) -> Buffer {
        self.device.new_buffer_with_data(
            packed_input.as_ptr() as *const _,
            (packed_input.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Batch evaluate multiple sparse groups with a SINGLE Metal command buffer
    ///
    /// This eliminates the ~0.5ms overhead per group from separate commit+wait cycles.
    /// With 34 sparse groups, this reduces sparse time from ~27ms to ~2-3ms.
    ///
    /// Uses BATCHED_SPARSE_BUFFER_POOL to cache buffers across batches, preventing
    /// memory leaks that would occur from allocating new buffers each call.
    pub fn eval_sparse_groups_batched(
        &self,
        input_buffer: &Buffer,
        scores_buffer: &Buffer,
        sparse_groups: &[SparseGroupData],
        num_examples: usize,
        words_per_example: usize,
        num_clusters: usize,
        empty_value: f32,
        memory_mode: u8,
        run_seed: u64,
    ) {
        if sparse_groups.is_empty() {
            return;
        }

        let batch_timing = std::env::var("WNN_SPARSE_TIMING").is_ok();
        let t0 = std::time::Instant::now();

        // Get current cache generation for validation
        let current_gen = get_sparse_cache_generation();

        // Pre-allocate converted data to keep it alive during GPU execution
        // For masked groups, we use different params struct
        enum ParamsBytes {
            Uniform(Vec<u8>),
            Masked(Vec<u8>),
        }
        let default_cell = default_cell_for_mode(memory_mode);
        let eval_memory_mode: u32 = memory_mode as u32;
        let mut converted_data: Vec<(Vec<i32>, Vec<u32>, ParamsBytes)> = Vec::with_capacity(sparse_groups.len());
        for group in sparse_groups {
            let connections_i32: Vec<i32> = group.connections.iter().map(|&c| c as i32).collect();
            let cluster_ids_u32: Vec<u32> = group.cluster_ids.iter().map(|&c| c as u32).collect();

            let params_bytes = if group.actual_neurons_per_cluster.is_some() {
                // Masked mode: use SparseToBufferMaskedParams
                let params = SparseToBufferMaskedParams {
                    num_examples: num_examples as u32,
                    words_per_example: words_per_example as u32,
                    num_neurons: (group.cluster_ids.len() * group.neurons_per_cluster) as u32,
                    bits_per_neuron: group.bits_per_neuron as u32,
                    max_neurons_per_cluster: group.neurons_per_cluster as u32,
                    num_group_clusters: group.cluster_ids.len() as u32,
                    total_clusters: num_clusters as u32,
                    empty_value,
                    memory_mode: eval_memory_mode,
                    default_cell_value: default_cell,
                    run_seed,
                };
                ParamsBytes::Masked(unsafe {
                    std::slice::from_raw_parts(
                        &params as *const SparseToBufferMaskedParams as *const u8,
                        mem::size_of::<SparseToBufferMaskedParams>(),
                    ).to_vec()
                })
            } else {
                // Uniform mode: use SparseToBufferParams
                let params = SparseToBufferParams {
                    num_examples: num_examples as u32,
                    words_per_example: words_per_example as u32,
                    num_neurons: (group.cluster_ids.len() * group.neurons_per_cluster) as u32,
                    bits_per_neuron: group.bits_per_neuron as u32,
                    neurons_per_cluster: group.neurons_per_cluster as u32,
                    num_group_clusters: group.cluster_ids.len() as u32,
                    total_clusters: num_clusters as u32,
                    empty_value,
                    memory_mode: eval_memory_mode,
                    default_cell_value: default_cell,
                    run_seed,
                };
                ParamsBytes::Uniform(unsafe {
                    std::slice::from_raw_parts(
                        &params as *const SparseToBufferParams as *const u8,
                        mem::size_of::<SparseToBufferParams>(),
                    ).to_vec()
                })
            };
            converted_data.push((connections_i32, cluster_ids_u32, params_bytes));
        }

        // Get cached buffers from pool (all buffers must stay alive until command completes)
        let all_buffers = BATCHED_SPARSE_BUFFER_POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.ensure_capacity(sparse_groups.len());

            let mut buffers: Vec<(Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Buffer, Option<Buffer>)> = Vec::with_capacity(sparse_groups.len());

            for (idx, group) in sparse_groups.iter().enumerate() {
                let (ref connections_i32, ref cluster_ids_u32, ref params_bytes) = converted_data[idx];
                let params_slice = match params_bytes {
                    ParamsBytes::Uniform(v) => v.as_slice(),
                    ParamsBytes::Masked(v) => v.as_slice(),
                };

                let conn = get_or_create_buffer(&self.device, &mut pool.conn_buffers[idx], connections_i32, current_gen);
                let keys = get_or_create_buffer(&self.device, &mut pool.keys_buffers[idx], group.keys, current_gen);
                let values = get_or_create_buffer(&self.device, &mut pool.values_buffers[idx], group.values, current_gen);
                let offsets = get_or_create_buffer(&self.device, &mut pool.offsets_buffers[idx], group.offsets, current_gen);
                let counts = get_or_create_buffer(&self.device, &mut pool.counts_buffers[idx], group.counts, current_gen);
                let cluster_ids = get_or_create_buffer(&self.device, &mut pool.cluster_ids_buffers[idx], cluster_ids_u32, current_gen);
                let params = get_or_create_buffer(&self.device, &mut pool.params_buffers[idx], params_slice, current_gen);

                // For masked groups, also create the actual_neurons buffer
                let actual_neurons_buf = if let Some(actual_neurons) = group.actual_neurons_per_cluster {
                    Some(get_or_create_buffer(&self.device, &mut pool.actual_neurons_buffers[idx], actual_neurons, current_gen))
                } else {
                    None
                };

                buffers.push((conn, keys, values, offsets, counts, cluster_ids, params, actual_neurons_buf));
            }

            buffers
        });

        // Create a single command buffer for all sparse groups
        let command_buffer = self.command_queue.new_command_buffer();

        for (idx, group) in sparse_groups.iter().enumerate() {
            let num_group_clusters = group.cluster_ids.len();

            let (ref conn_buffer, ref keys_buffer, ref values_buffer, ref offsets_buffer,
                 ref counts_buffer, ref cluster_ids_buffer, ref params_buffer, ref actual_neurons_buf) = all_buffers[idx];

            // Encode compute pass for this group
            let encoder = command_buffer.new_compute_command_encoder();

            // Choose pipeline based on whether this is a masked group
            if let Some(ref actual_neurons_buffer) = actual_neurons_buf {
                // Masked pipeline with per-cluster neuron counts
                encoder.set_compute_pipeline_state(&self.sparse_to_buffer_masked_pipeline);
                encoder.set_buffer(0, Some(input_buffer), 0);
                encoder.set_buffer(1, Some(conn_buffer), 0);
                encoder.set_buffer(2, Some(keys_buffer), 0);
                encoder.set_buffer(3, Some(values_buffer), 0);
                encoder.set_buffer(4, Some(offsets_buffer), 0);
                encoder.set_buffer(5, Some(counts_buffer), 0);
                encoder.set_buffer(6, Some(cluster_ids_buffer), 0);
                encoder.set_buffer(7, Some(actual_neurons_buffer), 0);
                encoder.set_buffer(8, Some(params_buffer), 0);
                encoder.set_buffer(9, Some(scores_buffer), 0);
            } else {
                // Uniform pipeline (all clusters have same neurons)
                encoder.set_compute_pipeline_state(&self.sparse_to_buffer_pipeline);
                encoder.set_buffer(0, Some(input_buffer), 0);
                encoder.set_buffer(1, Some(conn_buffer), 0);
                encoder.set_buffer(2, Some(keys_buffer), 0);
                encoder.set_buffer(3, Some(values_buffer), 0);
                encoder.set_buffer(4, Some(offsets_buffer), 0);
                encoder.set_buffer(5, Some(counts_buffer), 0);
                encoder.set_buffer(6, Some(cluster_ids_buffer), 0);
                encoder.set_buffer(7, Some(params_buffer), 0);
                encoder.set_buffer(8, Some(scores_buffer), 0);
            }

            // Grid: X = examples (SIMD-coalesced), Y = clusters
            // SIMD groups of 32 threads span X, so all threads in a group
            // access the SAME cluster's key arrays → cache-friendly binary search
            let grid_size = MTLSize::new(num_examples as u64, num_group_clusters as u64, 1);
            let thread_group_size = MTLSize::new(
                32.min(num_examples as u64),
                8.min(num_group_clusters as u64),
                1,
            );
            encoder.dispatch_threads(grid_size, thread_group_size);
            encoder.end_encoding();
        }

        // Single commit + wait for all groups
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Clear unused buffer slots to prevent memory accumulation from batches
        // with more sparse groups than the current batch
        let num_groups = sparse_groups.len();
        BATCHED_SPARSE_BUFFER_POOL.with(|pool| {
            pool.borrow_mut().clear_beyond(num_groups);
        });

        if batch_timing {
            let elapsed = t0.elapsed();
            let total_keys: usize = sparse_groups.iter().map(|g| g.keys.len()).sum();
            eprintln!(
                "[SPARSE_BATCHED] groups={} total_keys={} time={:.1}ms",
                sparse_groups.len(),
                total_keys,
                elapsed.as_micros() as f64 / 1000.0
            );
        }
    }

    /// Evaluate dense group and write directly to shared buffer on GPU
    ///
    /// Uses thread-local buffer caching to avoid 4 buffer allocations per call.
    pub fn eval_dense_to_buffer(
        &self,
        input_buffer: &Buffer,
        scores_buffer: &Buffer,
        connections: &[i64],
        memory_words: &[i64],
        cluster_ids: &[usize],
        num_examples: usize,
        words_per_example: usize,
        bits_per_neuron: usize,
        neurons_per_cluster: usize,
        num_clusters: usize,
        words_per_neuron: usize,
        empty_value: f32,
        memory_mode: u8,
        run_seed: u64,
    ) {
        let num_group_clusters = cluster_ids.len();
        let num_neurons = num_group_clusters * neurons_per_cluster;

        // Convert to GPU-friendly formats
        let connections_i32: Vec<i32> = connections.iter().map(|&c| c as i32).collect();
        let cluster_ids_u32: Vec<u32> = cluster_ids.iter().map(|&c| c as u32).collect();

        let params = DenseToBufferParams {
            num_examples: num_examples as u32,
            words_per_example: words_per_example as u32,
            num_neurons: num_neurons as u32,
            bits_per_neuron: bits_per_neuron as u32,
            neurons_per_cluster: neurons_per_cluster as u32,
            num_group_clusters: num_group_clusters as u32,
            total_clusters: num_clusters as u32,
            words_per_neuron: words_per_neuron as u32,
            empty_value,
            memory_mode: memory_mode as u32,
            run_seed,
        };
        // Convert params to slice for get_or_create_buffer
        let params_slice = unsafe {
            std::slice::from_raw_parts(
                &params as *const DenseToBufferParams as *const u8,
                mem::size_of::<DenseToBufferParams>(),
            )
        };

        // Get current cache generation for validation
        let current_gen = get_sparse_cache_generation();

        // Use cached buffers from thread-local storage
        let (conn_buffer, memory_buffer, cluster_ids_buffer, params_buffer) =
            DENSE_BUFFER_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();

                let conn = get_or_create_buffer(&self.device, &mut cache.conn_buffer, &connections_i32, current_gen);
                let memory = get_or_create_buffer(&self.device, &mut cache.memory_buffer, memory_words, current_gen);
                let cluster_ids = get_or_create_buffer(&self.device, &mut cache.cluster_ids_buffer, &cluster_ids_u32, current_gen);
                let params = get_or_create_buffer(&self.device, &mut cache.params_buffer, params_slice, current_gen);

                (conn, memory, cluster_ids, params)
            });

        // Dispatch kernel
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.dense_to_buffer_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(&conn_buffer), 0);
        encoder.set_buffer(2, Some(&memory_buffer), 0);
        encoder.set_buffer(3, Some(&cluster_ids_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);
        encoder.set_buffer(5, Some(scores_buffer), 0);

        let grid_size = MTLSize::new(num_group_clusters as u64, num_examples as u64, 1);
        let thread_group_size = MTLSize::new(
            32.min(num_group_clusters as u64),
            8.min(num_examples as u64),
            1,
        );
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Compute CE, accuracy, and per-example predictions from accumulated scores buffer.
    /// Uses cached buffers to avoid allocation overhead.
    /// Returns (avg_ce, accuracy, predictions_vec).
    pub fn compute_ce_from_buffer(
        &self,
        scores_buffer: &Buffer,
        targets: &[i64],
        num_examples: usize,
        num_clusters: usize,
    ) -> Result<(f64, f64, Vec<u32>), String> {
        let current_gen = get_sparse_cache_generation();

        // Simple hash of targets for cache invalidation (first + last + len)
        let targets_hash = if targets.is_empty() {
            0u64
        } else {
            (targets[0] as u64)
                .wrapping_add((targets[targets.len() - 1] as u64).wrapping_mul(31))
                .wrapping_add((targets.len() as u64).wrapping_mul(997))
        };

        // Get or create cached buffers
        let (targets_buffer, ce_buffer, correct_buffer, predicted_buffer) = CE_BUFFER_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let targets_i32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
            let required_targets_bytes = (targets_i32.len() * mem::size_of::<i32>()) as u64;
            let required_ce_bytes = (num_examples * mem::size_of::<f32>()) as u64;
            let required_correct_bytes = (num_examples * mem::size_of::<u32>()) as u64;
            let required_predicted_bytes = (num_examples * mem::size_of::<u32>()) as u64;

            // Check targets buffer - simplified logic to avoid borrow issues
            // First check if we can reuse the existing buffer
            let can_reuse = cache.targets_buffer.as_ref().map_or(false, |cached| {
                cached.cache_gen == current_gen
                    && cached.capacity_bytes >= required_targets_bytes
                    && cache.cached_targets_hash == targets_hash
            });

            let tgt_buf = if can_reuse {
                // Targets unchanged, reuse buffer
                cache.targets_buffer.as_ref().unwrap().buffer.clone()
            } else {
                // Need to update or create buffer
                let buf = self.device.new_buffer_with_data(
                    targets_i32.as_ptr() as *const _,
                    required_targets_bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                cache.targets_buffer = Some(CachedBuffer {
                    buffer: buf.clone(),
                    capacity_bytes: required_targets_bytes,
                    cache_gen: current_gen,
                });
                cache.cached_targets_hash = targets_hash;
                buf
            };

            // CE buffer - just needs to be large enough
            let ce_buf = if let Some(ref cached) = cache.ce_buffer {
                if cached.cache_gen == current_gen && cached.capacity_bytes >= required_ce_bytes {
                    cached.buffer.clone()
                } else {
                    let buf = self.device.new_buffer(
                        required_ce_bytes,
                        MTLResourceOptions::StorageModeShared,
                    );
                    cache.ce_buffer = Some(CachedBuffer {
                        buffer: buf.clone(),
                        capacity_bytes: required_ce_bytes,
                        cache_gen: current_gen,
                    });
                    buf
                }
            } else {
                let buf = self.device.new_buffer(
                    required_ce_bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                cache.ce_buffer = Some(CachedBuffer {
                    buffer: buf.clone(),
                    capacity_bytes: required_ce_bytes,
                    cache_gen: current_gen,
                });
                buf
            };

            // Correct buffer - just needs to be large enough
            let correct_buf = if let Some(ref cached) = cache.correct_buffer {
                if cached.cache_gen == current_gen && cached.capacity_bytes >= required_correct_bytes {
                    cached.buffer.clone()
                } else {
                    let buf = self.device.new_buffer(
                        required_correct_bytes,
                        MTLResourceOptions::StorageModeShared,
                    );
                    cache.correct_buffer = Some(CachedBuffer {
                        buffer: buf.clone(),
                        capacity_bytes: required_correct_bytes,
                        cache_gen: current_gen,
                    });
                    buf
                }
            } else {
                let buf = self.device.new_buffer(
                    required_correct_bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                cache.correct_buffer = Some(CachedBuffer {
                    buffer: buf.clone(),
                    capacity_bytes: required_correct_bytes,
                    cache_gen: current_gen,
                });
                buf
            };

            // Predicted buffer - just needs to be large enough
            let pred_buf = if let Some(ref cached) = cache.predicted_buffer {
                if cached.cache_gen == current_gen && cached.capacity_bytes >= required_predicted_bytes {
                    cached.buffer.clone()
                } else {
                    let buf = self.device.new_buffer(
                        required_predicted_bytes,
                        MTLResourceOptions::StorageModeShared,
                    );
                    cache.predicted_buffer = Some(CachedBuffer {
                        buffer: buf.clone(),
                        capacity_bytes: required_predicted_bytes,
                        cache_gen: current_gen,
                    });
                    buf
                }
            } else {
                let buf = self.device.new_buffer(
                    required_predicted_bytes,
                    MTLResourceOptions::StorageModeShared,
                );
                cache.predicted_buffer = Some(CachedBuffer {
                    buffer: buf.clone(),
                    capacity_bytes: required_predicted_bytes,
                    cache_gen: current_gen,
                });
                buf
            };

            (tgt_buf, ce_buf, correct_buf, pred_buf)
        });

        // Params buffer is small, just create it each time
        let params = CEReduceParams {
            num_examples: num_examples as u32,
            num_clusters: num_clusters as u32,
        };
        let params_buffer = self.device.new_buffer_with_data(
            &params as *const _ as *const _,
            mem::size_of::<CEReduceParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ce_reduce_pipeline);
        encoder.set_buffer(0, Some(scores_buffer), 0);
        encoder.set_buffer(1, Some(&targets_buffer), 0);
        encoder.set_buffer(2, Some(&params_buffer), 0);
        encoder.set_buffer(3, Some(&ce_buffer), 0);
        encoder.set_buffer(4, Some(&correct_buffer), 0);
        encoder.set_buffer(5, Some(&predicted_buffer), 0);

        let grid_size = MTLSize::new(num_examples as u64, 1, 1);
        let max_threads = self.ce_reduce_pipeline.max_total_threads_per_threadgroup();
        let thread_group_size = MTLSize::new(max_threads.min(num_examples as u64), 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Sum CE and correct on CPU, read predictions
        let ce_ptr = ce_buffer.contents() as *const f32;
        let correct_ptr = correct_buffer.contents() as *const u32;
        let predicted_ptr = predicted_buffer.contents() as *const u32;

        let (total_ce, total_correct, predictions): (f64, u64, Vec<u32>) = unsafe {
            let ce_slice = std::slice::from_raw_parts(ce_ptr, num_examples);
            let correct_slice = std::slice::from_raw_parts(correct_ptr, num_examples);
            let predicted_slice = std::slice::from_raw_parts(predicted_ptr, num_examples);

            let total_ce: f64 = ce_slice.iter().map(|&c| c as f64).sum();
            let total_correct: u64 = correct_slice.iter().map(|&c| c as u64).sum();
            let predictions: Vec<u32> = predicted_slice.to_vec();

            (total_ce, total_correct, predictions)
        };

        let avg_ce = total_ce / num_examples as f64;
        let accuracy = total_correct as f64 / num_examples as f64;

        Ok((avg_ce, accuracy, predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metal_present() -> bool {
        metal::Device::system_default().is_some()
    }

    #[test]
    fn test_sparse_ce_evaluator_creation() {
        if metal_present() {
            let evaluator = MetalSparseCEEvaluator::new();
            assert!(evaluator.is_ok(), "Failed to create sparse CE evaluator: {:?}", evaluator.err());
        }
    }

    #[test]
    fn test_sparse_ce_small() {
        if !metal_present() {
            return;
        }
        let evaluator = MetalSparseCEEvaluator::new().unwrap();
        let num_examples = 3;
        let num_clusters = 4;
        let neurons_per_cluster = 2;
        let bits = 3;
        let total_neurons = num_clusters * neurons_per_cluster;
        let words_per_example = 1;
        let packed_input: Vec<u64> = vec![0x3FF; num_examples * words_per_example];
        let connections: Vec<i64> = (0..total_neurons).flat_map(|_| vec![0i64, 1, 2]).collect();
        let keys: Vec<u64> = vec![7; total_neurons];
        let values: Vec<u8> = vec![1; total_neurons];
        let offsets: Vec<u32> = (0..total_neurons).map(|i| i as u32).collect();
        let counts: Vec<u32> = vec![1; total_neurons];
        let targets: Vec<i64> = vec![0, 1, 2];
        let result = evaluator.compute_ce(
            &packed_input, &connections, &keys, &values, &offsets, &counts, &targets,
            num_examples, words_per_example, total_neurons, bits, neurons_per_cluster,
            num_clusters, 0.5, 0, 0,
        );
        assert!(result.is_ok(), "compute_ce failed: {:?}", result.err());
        let (avg_ce, accuracy, _predictions) = result.unwrap();
        println!("Small test: avg_ce={:.4}, accuracy={:.4}", avg_ce, accuracy);
        assert!(avg_ce > 1.0 && avg_ce < 2.0, "CE should be around 1.386");
    }

    // CPU reference for one cluster's QSR score — mirrors compute_cluster_score's
    // QSR branch exactly (same qsr_key derivation + qsr_coin). Used to prove the
    // GPU shader draws the identical coin at a fixed run_seed.
    #[allow(clippy::too_many_arguments)]
    fn cpu_qsr_cluster_score(
        packed_input: &[u64],
        connections: &[i64],
        keys: &[u64],
        values: &[u8],
        offsets: &[u32],
        counts: &[u32],
        start_neuron: usize,
        neurons_per_cluster: usize,
        bits_per_neuron: usize,
        default_cell: u8,
        run_seed: u64,
        example_idx: usize,
    ) -> f32 {
        use ram_core::neuron_memory::{qsr_coin, qsr_key};
        let mut sum = 0.0f32;
        for n in 0..neurons_per_cluster {
            let neuron_idx = start_neuron + n;
            let conn = &connections[neuron_idx * bits_per_neuron..(neuron_idx + 1) * bits_per_neuron];
            // address = pack observed bits MSB-first (matches wnn_compute_address_u64)
            let mut address = 0u64;
            for &bit in conn {
                let b = bit as usize;
                let word = packed_input[b / 64];
                let set = (word >> (b % 64)) & 1;
                address = (address << 1) | set;
            }
            let mem_start = offsets[neuron_idx] as usize;
            let mem_count = counts[neuron_idx] as usize;
            let mut cell = default_cell;
            for i in 0..mem_count {
                if keys[mem_start + i] == address {
                    cell = values[mem_start + i];
                    break;
                }
            }
            if cell > 3 { cell = default_cell; }
            let rng = qsr_key(run_seed, neuron_idx as u64, address, example_idx as u64);
            sum += qsr_coin(cell as i64, rng);
        }
        sum / neurons_per_cluster as f32
    }

    #[test]
    fn test_qsr_cpu_gpu_coin_parity() {
        // At a fixed run_seed the GPU coin must match the CPU coin bit-for-bit.
        // We verify via the observable accuracy: a QSR run and a CPU reference of
        // the same clusters, at the same seed, agree on which cluster wins.
        use ram_core::neuron_memory::QSR;
        if !metal_present() { return; }
        let evaluator = MetalSparseCEEvaluator::new().unwrap();

        let num_examples = 8usize;
        let num_clusters = 4usize;
        let neurons_per_cluster = 6usize;
        let bits = 4usize;
        let total_neurons = num_clusters * neurons_per_cluster;
        let words_per_example = 1usize;
        // Distinct inputs per example so addresses vary.
        let packed_input: Vec<u64> = (0..num_examples).map(|e| (0xA5u64).wrapping_mul(e as u64 + 1) & 0xFFFF).collect();
        let connections: Vec<i64> = (0..total_neurons).flat_map(|_| vec![0i64, 1, 2, 3]).collect();
        // Sparse memory: one stored entry per neuron, cell = a WEAK state (1 or 2)
        // so the coin actually fires stochastically (not deterministic 0/3).
        let keys: Vec<u64> = (0..total_neurons).map(|i| (i % 16) as u64).collect();
        let values: Vec<u8> = (0..total_neurons).map(|i| if i % 2 == 0 { 1u8 } else { 2u8 }).collect();
        let offsets: Vec<u32> = (0..total_neurons).map(|i| i as u32).collect();
        let counts: Vec<u32> = vec![1; total_neurons];
        let targets: Vec<i64> = (0..num_examples).map(|e| (e % num_clusters) as i64).collect();
        let run_seed = 0xC0FFEE_1234_5678u64;

        let (_ce, _acc, gpu_pred) = evaluator.compute_ce(
            &packed_input, &connections, &keys, &values, &offsets, &counts, &targets,
            num_examples, words_per_example, total_neurons, bits, neurons_per_cluster,
            num_clusters, 0.5, QSR, run_seed,
        ).unwrap();

        // CPU argmax over cluster scores using the shared qsr_key + qsr_coin.
        let default_cell = default_cell_for_mode(QSR) as u8;
        for e in 0..num_examples {
            let input = &packed_input[e * words_per_example..(e + 1) * words_per_example];
            let mut best_c = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for c in 0..num_clusters {
                let s = cpu_qsr_cluster_score(
                    input, &connections, &keys, &values, &offsets, &counts,
                    c * neurons_per_cluster, neurons_per_cluster, bits,
                    default_cell, run_seed, e,
                );
                if s > best_s { best_s = s; best_c = c; }
            }
            assert_eq!(
                gpu_pred[e] as usize, best_c,
                "QSR CPU/GPU prediction mismatch at example {e}: gpu={} cpu={best_c}", gpu_pred[e]
            );
        }
    }
}
