//! Metal GPU Accelerator for RAM-based Gating
//!
//! Uses Metal compute shaders to evaluate gating on Apple Silicon GPUs.
//! The M4 Max has 40 GPU cores which can process many (example, cluster)
//! pairs in parallel.
//!
//! Buffer Caching Strategy:
//! - Connections buffer: Cached permanently (immutable after creation)
//! - Memory buffer: Cached with generation tracking (re-upload after training)
//! - Input/Output buffers: Cached and resized as needed
//!
//! On M4 Max with unified memory, StorageModeShared means CPU and GPU share
//! the same physical memory. The main overhead is buffer object creation,
//! which we avoid by reusing cached buffers.

use metal::*;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};

use super::gating::RAMGating;

// =============================================================================
// Buffer Cache Infrastructure
// =============================================================================

/// Global generation counter for cache invalidation
static GATING_CACHE_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Increment cache generation (call after training to force memory re-upload)
pub fn invalidate_gating_cache() {
    GATING_CACHE_GENERATION.fetch_add(1, Ordering::SeqCst);
}

/// Get current cache generation
fn get_cache_generation() -> u64 {
    GATING_CACHE_GENERATION.load(Ordering::SeqCst)
}

/// Reset and clear all gating buffer caches to free GPU memory
/// Call this when done with gating operations to prevent memory leaks
pub fn reset_gating_buffer_cache() {
    // Increment generation to invalidate any existing references
    invalidate_gating_cache();

    // Clear thread-local cache
    GATING_BUFFER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        cache.conn_buffer = None;
        cache.conn_data_hash = 0;
        cache.memory_buffer = None;
        cache.memory_generation = 0;
        cache.input_buffer = None;
        cache.params_buffer = None;
        cache.output_buffer = None;
    });
}

/// Cached buffer with capacity and generation tracking
struct CachedBuffer {
    buffer: Buffer,
    capacity_bytes: u64,
    generation: u64,
}

/// Thread-local buffer cache for gating evaluation
struct GatingBufferCache {
    // Static data (connections) - cached once
    conn_buffer: Option<CachedBuffer>,
    conn_data_hash: u64,  // Hash to detect if connections changed

    // Memory data - cached with generation tracking
    memory_buffer: Option<CachedBuffer>,
    memory_generation: u64,  // Generation when memory was last uploaded

    // Dynamic data - reuse if capacity sufficient
    input_buffer: Option<CachedBuffer>,
    params_buffer: Option<CachedBuffer>,
    output_buffer: Option<CachedBuffer>,
}

impl GatingBufferCache {
    fn new() -> Self {
        Self {
            conn_buffer: None,
            conn_data_hash: 0,
            memory_buffer: None,
            memory_generation: 0,
            input_buffer: None,
            params_buffer: None,
            output_buffer: None,
        }
    }
}

thread_local! {
    static GATING_BUFFER_CACHE: std::cell::RefCell<GatingBufferCache> =
        std::cell::RefCell::new(GatingBufferCache::new());
}

/// Simple hash for detecting connection changes
fn hash_slice<T: Copy>(data: &[T]) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    data.len().hash(&mut hasher);
    if !data.is_empty() {
        // Hash first, middle, last elements for quick detection
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * mem::size_of::<T>()
            )
        };
        bytes[0..8.min(bytes.len())].hash(&mut hasher);
        if bytes.len() > 16 {
            bytes[bytes.len()/2..bytes.len()/2+8].hash(&mut hasher);
            bytes[bytes.len()-8..].hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Get or create a cached buffer, updating contents if needed
fn get_or_create_buffer<T: Copy>(
    device: &Device,
    cached: &mut Option<CachedBuffer>,
    data: &[T],
    current_gen: u64,
) -> Buffer {
    let required_bytes = (data.len() * mem::size_of::<T>()) as u64;

    // Check if cached buffer can be reused
    if let Some(ref cache) = cached {
        if cache.generation == current_gen && cache.capacity_bytes >= required_bytes {
            // Reuse buffer - write data directly to contents (zero-copy on unified memory)
            let ptr = cache.buffer.contents() as *mut T;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
            return cache.buffer.clone();
        }
    }

    // Need new buffer - allocate with 50% headroom for future reuse
    let alloc_bytes = ((required_bytes as f64 * 1.5) as u64).max(1024);

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
        generation: current_gen,
    });

    buffer
}

/// Get or create output buffer (no data copy needed, just allocation)
fn get_or_create_output_buffer(
    device: &Device,
    cached: &mut Option<CachedBuffer>,
    required_elements: usize,
    current_gen: u64,
) -> Buffer {
    let required_bytes = (required_elements * mem::size_of::<f32>()) as u64;

    if let Some(ref cache) = cached {
        if cache.capacity_bytes >= required_bytes {
            return cache.buffer.clone();
        }
    }

    let alloc_bytes = ((required_bytes as f64 * 1.5) as u64).max(1024);

    let buffer = device.new_buffer(
        alloc_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    *cached = Some(CachedBuffer {
        buffer: buffer.clone(),
        capacity_bytes: alloc_bytes,
        generation: current_gen,
    });

    buffer
}

// =============================================================================
// Metal Gating Evaluator
// =============================================================================

/// Parameters struct matching the Metal shader
#[repr(C)]
struct GatingParams {
    num_clusters: u32,
    neurons_per_gate: u32,
    bits_per_neuron: u32,
    words_per_example: u32,
    vote_threshold: u32,
    address_space_size: u32,
    batch_size: u32,
    _padding: u32,
}

/// Metal-based Gating evaluator with buffer caching
pub struct MetalGatingEvaluator {
    device: Device,
    command_queue: CommandQueue,
    forward_pipeline: ComputePipelineState,
    forward_per_example_pipeline: ComputePipelineState,
    train_pipeline: ComputePipelineState,
}

impl MetalGatingEvaluator {
    /// Check if Metal is available
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    /// Create new Metal gating evaluator
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();

        // Compile gating shader
        let shader_source = include_str!("shaders/gating.metal");
        let library = device
            .new_library_with_source(shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile gating shader: {}", e))?;

        // Get kernel functions
        let forward_kernel = library
            .get_function("gating_forward", None)
            .map_err(|e| format!("Failed to get gating_forward kernel: {}", e))?;

        let forward_per_example_kernel = library
            .get_function("gating_forward_per_example", None)
            .map_err(|e| format!("Failed to get gating_forward_per_example kernel: {}", e))?;

        let train_kernel = library
            .get_function("gating_train", None)
            .map_err(|e| format!("Failed to get gating_train kernel: {}", e))?;

        // Create pipelines
        let forward_pipeline = device
            .new_compute_pipeline_state_with_function(&forward_kernel)
            .map_err(|e| format!("Failed to create forward pipeline: {}", e))?;

        let forward_per_example_pipeline = device
            .new_compute_pipeline_state_with_function(&forward_per_example_kernel)
            .map_err(|e| format!("Failed to create forward_per_example pipeline: {}", e))?;

        let train_pipeline = device
            .new_compute_pipeline_state_with_function(&train_kernel)
            .map_err(|e| format!("Failed to create train pipeline: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            forward_pipeline,
            forward_per_example_pipeline,
            train_pipeline,
        })
    }

    /// Get device reference for external use
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Forward pass on GPU with buffer caching
    ///
    /// # Arguments
    /// * `gating` - The RAMGating model (provides memory and connections)
    /// * `packed_input` - Packed u64 input [batch_size * words_per_example]
    /// * `batch_size` - Number of examples in batch
    /// * `words_per_example` - Number of u64 words per example
    ///
    /// # Returns
    /// Flattened gate values [batch_size * num_clusters]
    pub fn forward_batch(
        &self,
        gating: &RAMGating,
        packed_input: &[u64],
        batch_size: usize,
        words_per_example: usize,
    ) -> Result<Vec<f32>, String> {
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let config = gating.config();
        let current_gen = get_cache_generation();

        // Get connections as i32
        let connections: Vec<i32> = gating.get_connections().to_vec();
        let conn_hash = hash_slice(&connections);

        // Export memory as u8
        let memory = gating.export_memory();

        // Prepare params
        let params = GatingParams {
            num_clusters: config.num_clusters as u32,
            neurons_per_gate: config.neurons_per_gate as u32,
            bits_per_neuron: config.bits_per_neuron as u32,
            words_per_example: words_per_example as u32,
            vote_threshold: config.vote_threshold as u32,
            address_space_size: (1 << config.bits_per_neuron) as u32,
            batch_size: batch_size as u32,
            _padding: 0,
        };

        let output_size = batch_size * config.num_clusters;

        // Use thread-local cache
        GATING_BUFFER_CACHE.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();

            // Connections buffer: only update if hash changed
            let conn_buffer = if cache.conn_data_hash == conn_hash {
                if let Some(ref cb) = cache.conn_buffer {
                    cb.buffer.clone()
                } else {
                    let buf = get_or_create_buffer(&self.device, &mut cache.conn_buffer, &connections, current_gen);
                    cache.conn_data_hash = conn_hash;
                    buf
                }
            } else {
                let buf = get_or_create_buffer(&self.device, &mut cache.conn_buffer, &connections, current_gen);
                cache.conn_data_hash = conn_hash;
                buf
            };

            // Memory buffer: update if generation changed (after training)
            let memory_buffer = if cache.memory_generation == current_gen {
                if let Some(ref mb) = cache.memory_buffer {
                    if mb.capacity_bytes >= memory.len() as u64 {
                        // Reuse buffer, update contents
                        let ptr = mb.buffer.contents() as *mut u8;
                        unsafe {
                            std::ptr::copy_nonoverlapping(memory.as_ptr(), ptr, memory.len());
                        }
                        mb.buffer.clone()
                    } else {
                        get_or_create_buffer(&self.device, &mut cache.memory_buffer, &memory, current_gen)
                    }
                } else {
                    let buf = get_or_create_buffer(&self.device, &mut cache.memory_buffer, &memory, current_gen);
                    cache.memory_generation = current_gen;
                    buf
                }
            } else {
                let buf = get_or_create_buffer(&self.device, &mut cache.memory_buffer, &memory, current_gen);
                cache.memory_generation = current_gen;
                buf
            };

            // Input buffer: always update contents (packed u64)
            let input_buffer = get_or_create_buffer(&self.device, &mut cache.input_buffer, packed_input, current_gen);

            // Params buffer: always update
            let params_slice = unsafe {
                std::slice::from_raw_parts(
                    &params as *const GatingParams as *const u8,
                    mem::size_of::<GatingParams>()
                )
            };
            let params_buffer = get_or_create_buffer(&self.device, &mut cache.params_buffer, params_slice, current_gen);

            // Output buffer: just need capacity
            let output_buffer = get_or_create_output_buffer(&self.device, &mut cache.output_buffer, output_size, current_gen);

            // Choose kernel based on problem size
            let use_per_example = config.num_clusters > 1000;

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            if use_per_example {
                encoder.set_compute_pipeline_state(&self.forward_per_example_pipeline);
            } else {
                encoder.set_compute_pipeline_state(&self.forward_pipeline);
            }

            encoder.set_buffer(0, Some(&conn_buffer), 0);
            encoder.set_buffer(1, Some(&memory_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&params_buffer), 0);
            encoder.set_buffer(4, Some(&output_buffer), 0);

            if use_per_example {
                let grid_size = MTLSize::new(batch_size as u64, 1, 1);
                let max_threads = self.forward_per_example_pipeline.max_total_threads_per_threadgroup();
                let thread_group_size = MTLSize::new(max_threads.min(batch_size as u64), 1, 1);
                encoder.dispatch_threads(grid_size, thread_group_size);
            } else {
                let grid_size = MTLSize::new(config.num_clusters as u64, batch_size as u64, 1);
                let max_threads = self.forward_pipeline.max_total_threads_per_threadgroup();
                let threads_x = (max_threads as f64).sqrt() as u64;
                let threads_y = max_threads / threads_x;
                let thread_group_size = MTLSize::new(
                    threads_x.min(config.num_clusters as u64),
                    threads_y.min(batch_size as u64),
                    1,
                );
                encoder.dispatch_threads(grid_size, thread_group_size);
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read results
            let ptr = output_buffer.contents() as *const f32;
            let results = unsafe { std::slice::from_raw_parts(ptr, output_size) };
            Ok(results.to_vec())
        })
    }

    /// Train gating neurons on GPU
    ///
    /// Uses the gating_train Metal kernel with atomic memory writes.
    /// The kernel processes (cluster, batch) pairs in parallel.
    ///
    /// # Arguments
    /// * `gating` - The RAMGating model to train into
    /// * `packed_input` - Packed u64 input [batch_size * words_per_example]
    /// * `target_gates_flat` - Flattened target gate values [batch_size * num_clusters]
    /// * `batch_size` - Number of training examples
    /// * `words_per_example` - Number of u64 words per example
    ///
    /// # Returns
    /// Updated memory (caller must import into gating model)
    pub fn train_batch(
        &self,
        gating: &RAMGating,
        packed_input: &[u64],
        target_gates_flat: &[bool],
        batch_size: usize,
        words_per_example: usize,
    ) -> Result<Vec<u8>, String> {
        if batch_size == 0 {
            return Ok(gating.export_memory());
        }

        let config = gating.config();

        // Convert target gates to u8 for GPU
        let target_gates_u8: Vec<u8> = target_gates_flat.iter().map(|&b| b as u8).collect();

        // Get connections as i32
        let connections: Vec<i32> = gating.get_connections().to_vec();

        // Export current memory state and pack into u32s for atomic access
        let memory_bytes = gating.export_memory();
        let memory_words: Vec<u32> = pack_u8_to_u32(&memory_bytes);

        // Prepare params
        let params = GatingParams {
            num_clusters: config.num_clusters as u32,
            neurons_per_gate: config.neurons_per_gate as u32,
            bits_per_neuron: config.bits_per_neuron as u32,
            words_per_example: words_per_example as u32,
            vote_threshold: config.vote_threshold as u32,
            address_space_size: (1 << config.bits_per_neuron) as u32,
            batch_size: batch_size as u32,
            _padding: 0,
        };

        // Create buffers (using StorageModeShared for unified memory)
        let conn_buffer = self.device.new_buffer_with_data(
            connections.as_ptr() as *const _,
            (connections.len() * mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let memory_buffer = self.device.new_buffer_with_data(
            memory_words.as_ptr() as *const _,
            (memory_words.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let input_buffer = self.device.new_buffer_with_data(
            packed_input.as_ptr() as *const _,
            (packed_input.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let target_buffer = self.device.new_buffer_with_data(
            target_gates_u8.as_ptr() as *const _,
            target_gates_u8.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let params_buffer = self.device.new_buffer_with_data(
            &params as *const GatingParams as *const _,
            mem::size_of::<GatingParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.train_pipeline);
        encoder.set_buffer(0, Some(&conn_buffer), 0);
        encoder.set_buffer(1, Some(&memory_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&target_buffer), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);

        // Dispatch: one thread per (cluster, batch) pair
        let grid_size = MTLSize::new(config.num_clusters as u64, batch_size as u64, 1);
        let max_threads = self.train_pipeline.max_total_threads_per_threadgroup();
        let threads_x = (max_threads as f64).sqrt() as u64;
        let threads_y = max_threads / threads_x;
        let thread_group_size = MTLSize::new(
            threads_x.min(config.num_clusters as u64),
            threads_y.min(batch_size as u64),
            1,
        );
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back updated memory and unpack from u32 to u8
        let ptr = memory_buffer.contents() as *const u32;
        let result_words = unsafe { std::slice::from_raw_parts(ptr, memory_words.len()) };
        let result_bytes = unpack_u32_to_u8(result_words, memory_bytes.len());

        Ok(result_bytes)
    }
}

/// Pack u8 slice into u32 slice (4 bytes per u32)
fn pack_u8_to_u32(bytes: &[u8]) -> Vec<u32> {
    let word_count = (bytes.len() + 3) / 4;
    let mut words = vec![0u32; word_count];

    for (i, &b) in bytes.iter().enumerate() {
        let word_idx = i / 4;
        let byte_offset = i % 4;
        words[word_idx] |= (b as u32) << (byte_offset * 8);
    }

    words
}

/// Unpack u32 slice back to u8 slice
fn unpack_u32_to_u8(words: &[u32], target_len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(target_len);

    for &word in words {
        for byte_offset in 0..4 {
            if bytes.len() >= target_len {
                break;
            }
            bytes.push(((word >> (byte_offset * 8)) & 0xFF) as u8);
        }
    }

    bytes.truncate(target_len);
    bytes
}

/// Hybrid CPU+GPU forward pass
///
/// Splits the batch between CPU (rayon) and GPU (Metal) for maximum throughput.
///
/// # Arguments
/// * `gating` - The RAMGating model
/// * `metal_eval` - Metal evaluator
/// * `input_bits_flat` - Flattened input bits [batch_size * total_input_bits]
/// * `batch_size` - Number of examples
/// * `cpu_fraction` - Fraction of batch to process on CPU (0.0-1.0)
///
/// # Returns
/// Flattened gate values [batch_size * num_clusters]
pub fn forward_batch_hybrid(
    gating: &RAMGating,
    metal_eval: &MetalGatingEvaluator,
    input_bits_flat: &[bool],
    batch_size: usize,
    cpu_fraction: f32,
) -> Result<Vec<f32>, String> {
    use super::neuron_memory;

    if batch_size == 0 {
        return Ok(vec![]);
    }

    let config = gating.config();
    let total_input_bits = config.total_input_bits;
    let num_clusters = config.num_clusters;

    // Determine split point
    let cpu_batch_size = ((batch_size as f32 * cpu_fraction) as usize).max(1).min(batch_size);
    let gpu_batch_size = batch_size - cpu_batch_size;

    // If GPU batch is empty, just use CPU
    if gpu_batch_size == 0 {
        return Ok(gating.forward_batch(input_bits_flat, batch_size));
    }

    // Split input
    let cpu_input_end = cpu_batch_size * total_input_bits;
    let cpu_input = &input_bits_flat[..cpu_input_end];
    let gpu_input = &input_bits_flat[cpu_input_end..];

    // Pack GPU portion to u64
    let (packed_gpu, words_per_example) = neuron_memory::pack_bools_to_u64(
        gpu_input, gpu_batch_size, total_input_bits
    );

    // Allocate output
    let mut results = vec![0.0f32; batch_size * num_clusters];

    // Run CPU and GPU in parallel using rayon's join
    let (cpu_results, gpu_results) = rayon::join(
        || gating.forward_batch(cpu_input, cpu_batch_size),
        || metal_eval.forward_batch(gating, &packed_gpu, gpu_batch_size, words_per_example),
    );

    // Check GPU result
    let gpu_results = gpu_results?;

    // Copy results
    let cpu_output_end = cpu_batch_size * num_clusters;
    results[..cpu_output_end].copy_from_slice(&cpu_results);
    results[cpu_output_end..].copy_from_slice(&gpu_results);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::neuron_memory;

    #[test]
    fn test_metal_gating_available() {
        let available = MetalGatingEvaluator::is_available();
        println!("Metal gating available: {}", available);
    }

    #[test]
    fn test_metal_gating_forward() {
        if !MetalGatingEvaluator::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let total_input_bits = 32;
        let gating = RAMGating::new(10, 4, 6, total_input_bits, 0.5, Some(42));
        let input = vec![true; total_input_bits];

        // Train some gates
        let mut targets = vec![false; 10];
        targets[3] = true;
        targets[7] = true;
        gating.train_single(&input, &targets, false);

        // Invalidate cache after training
        invalidate_gating_cache();

        // Pack input and test Metal forward
        let (packed, wpe) = neuron_memory::pack_bools_to_u64(&input, 1, total_input_bits);
        let metal_eval = MetalGatingEvaluator::new().unwrap();
        let gates = metal_eval.forward_batch(&gating, &packed, 1, wpe).unwrap();

        assert_eq!(gates.len(), 10);
        assert_eq!(gates[3], 1.0);
        assert_eq!(gates[7], 1.0);
    }

    #[test]
    fn test_buffer_caching() {
        if !MetalGatingEvaluator::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let total_input_bits = 64;
        let gating = RAMGating::new(100, 8, 10, total_input_bits, 0.5, Some(42));
        let metal_eval = MetalGatingEvaluator::new().unwrap();

        // Multiple forward calls should reuse buffers
        for i in 0..5 {
            let input: Vec<bool> = (0..total_input_bits).map(|j| ((i + j) % 2) == 0).collect();
            let (packed, wpe) = neuron_memory::pack_bools_to_u64(&input, 1, total_input_bits);
            let gates = metal_eval.forward_batch(&gating, &packed, 1, wpe).unwrap();
            assert_eq!(gates.len(), 100);
        }

        // After training, invalidate cache
        let input = vec![true; total_input_bits];
        let mut targets = vec![false; 100];
        targets[50] = true;
        gating.train_single(&input, &targets, false);
        invalidate_gating_cache();

        // Forward should still work
        let (packed, wpe) = neuron_memory::pack_bools_to_u64(&input, 1, total_input_bits);
        let gates = metal_eval.forward_batch(&gating, &packed, 1, wpe).unwrap();
        assert_eq!(gates[50], 1.0);
    }

    #[test]
    fn test_metal_cpu_equivalence() {
        if !MetalGatingEvaluator::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let total_input_bits = 64;
        let gating = RAMGating::new(100, 8, 10, total_input_bits, 0.5, Some(42));

        // Train with random patterns
        for i in 0..50 {
            let input: Vec<bool> = (0..total_input_bits).map(|j| ((i * 7 + j) % 3) == 0).collect();
            let mut targets = vec![false; 100];
            targets[(i * 3) % 100] = true;
            targets[(i * 7) % 100] = true;
            gating.train_single(&input, &targets, false);
        }

        invalidate_gating_cache();

        // Test batch
        let batch_size = 20;
        let input_flat: Vec<bool> = (0..batch_size * total_input_bits)
            .map(|i| ((i * 11) % 5) == 0)
            .collect();

        // CPU forward
        let cpu_gates = gating.forward_batch(&input_flat, batch_size);

        // GPU forward (pack first)
        let (packed, wpe) = neuron_memory::pack_bools_to_u64(&input_flat, batch_size, total_input_bits);
        let metal_eval = MetalGatingEvaluator::new().unwrap();
        let gpu_gates = metal_eval.forward_batch(&gating, &packed, batch_size, wpe).unwrap();

        // Should be identical
        assert_eq!(cpu_gates.len(), gpu_gates.len());
        for (i, (&cpu, &gpu)) in cpu_gates.iter().zip(gpu_gates.iter()).enumerate() {
            assert_eq!(cpu, gpu, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_metal_gpu_training() {
        if !MetalGatingEvaluator::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let total_input_bits = 32;
        let gating_gpu = RAMGating::new(10, 4, 6, total_input_bits, 0.5, Some(42));
        let gating_cpu = RAMGating::new(10, 4, 6, total_input_bits, 0.5, Some(42));

        let metal_eval = MetalGatingEvaluator::new().unwrap();

        // Create training batch
        let batch_size = 5;
        let input_flat: Vec<bool> = (0..batch_size * total_input_bits)
            .map(|i| ((i * 3) % 2) == 0)
            .collect();

        // Target gates: each example has one target cluster open
        let mut target_gates: Vec<bool> = vec![false; batch_size * 10];
        for b in 0..batch_size {
            target_gates[b * 10 + (b * 2) % 10] = true;
        }

        // Train on CPU
        gating_cpu.train_batch(&input_flat, &target_gates, batch_size, false);

        // Train on GPU (pack first)
        let (packed, wpe) = neuron_memory::pack_bools_to_u64(&input_flat, batch_size, total_input_bits);
        let gpu_memory = metal_eval.train_batch(&gating_gpu, &packed, &target_gates, batch_size, wpe).unwrap();
        gating_gpu.import_memory(&gpu_memory).unwrap();

        invalidate_gating_cache();

        // Test input
        let test_input: Vec<bool> = (0..total_input_bits).map(|i| ((i * 3) % 2) == 0).collect();

        // Forward on both should give same results
        let cpu_gates = gating_cpu.forward_single(&test_input);
        let gpu_gates = gating_gpu.forward_single(&test_input);

        assert_eq!(cpu_gates, gpu_gates, "CPU and GPU trained models should produce same output");
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let original: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let packed = super::pack_u8_to_u32(&original);
        let unpacked = super::unpack_u32_to_u8(&packed, original.len());
        assert_eq!(original, unpacked);
    }
}
