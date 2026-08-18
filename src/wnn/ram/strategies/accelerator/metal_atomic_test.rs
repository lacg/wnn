//! Microbenchmark validating that Metal `atomic_ulong` CAS operations are
//! coherent with CPU `AtomicU64::compare_exchange` on a SHARED buffer in
//! Apple Silicon unified memory.
//!
//! This is the foundational assumption for Option C (CPU+GPU concurrent
//! training on the same AtomicHashTable). If this test fails — i.e., total
//! increment count is less than expected, or claim-test sees multiple
//! winners per slot — atomic semantics don't hold cross-platform and
//! Option C's architecture is unsound.
//!
//! Three tests:
//!   1. GPU-only CAS increment (sanity — does Metal atomic CAS work at all?)
//!   2. Concurrent CPU+GPU CAS increment on the same shared buffer
//!   3. Claim-with-contention (CAS from EMPTY → claimer's value): exactly
//!      one winner per slot, no double-claim
//!
//! Apple Silicon's unified memory + `StorageModeShared` is documented to
//! support cross-CPU/GPU atomic visibility, but cross-platform CAS has not
//! been validated in this codebase. Hence this test.

#[cfg(target_os = "macos")]
mod metal_impl
{

	use metal::{
		CompileOptions, ComputePipelineState, Device, MTLLanguageVersion, MTLResourceOptions, MTLSize,
	};
	use std::sync::atomic::{AtomicU64, Ordering};

	const SHADER_SOURCE: &str = include_str!("shaders/atomic_cas_test.metal");
	const EMPTY_SENTINEL: u64 = u64::MAX;

	pub struct MicrobenchResult
	{
		pub test_name: String,
		pub passed: bool,
		pub expected: u64,
		pub observed: u64,
		pub details: String,
		pub elapsed_ms: f64,
	}

	pub fn run_microbench(
		num_slots: usize,
		gpu_threads: usize,
		cpu_threads: usize,
		iterations: usize,
	) -> Result<Vec<MicrobenchResult>, String>
	{
		let device = Device::system_default().ok_or("No Metal device available")?;
		let command_queue = device.new_command_queue();

		// 64-bit atomics require MSL 3.0+. Apple Silicon M-series GPUs support
		// 64-bit atomics with `memory_order_relaxed` only (no acquire/release on
		// 64-bit). Cross-CPU/GPU coherence via `memory_scope_system` is NOT
		// supported for atomic CAS ops in any MSL version — that's the limit
		// that drives Option B's per-genome architecture (each genome's table
		// is exclusive to either CPU or GPU; no shared CAS).
		let compile_opts = CompileOptions::new();
		compile_opts.set_language_version(MTLLanguageVersion::V3_1);
		let library = device
			.new_library_with_source(SHADER_SOURCE, &compile_opts)
			.map_err(|e| format!("Metal shader compile failed: {}", e))?;

		let inc_kernel = library
			.get_function("atomic_cas_increment", None)
			.map_err(|e| format!("get_function increment: {}", e))?;
		let inc_pipeline = device
			.new_compute_pipeline_state_with_function(&inc_kernel)
			.map_err(|e| format!("pipeline increment: {}", e))?;

		let claim_kernel = library
			.get_function("atomic_cas_claim", None)
			.map_err(|e| format!("get_function claim: {}", e))?;
		let claim_pipeline = device
			.new_compute_pipeline_state_with_function(&claim_kernel)
			.map_err(|e| format!("pipeline claim: {}", e))?;

		let mut results = Vec::new();

		// Test 1: GPU-only increment — does Metal CAS work at all?
		results.push(run_gpu_only_increment(
			&device,
			&command_queue,
			&inc_pipeline,
			num_slots,
			gpu_threads,
			iterations,
		)?);

		// Test 2: CPU + GPU concurrent increment on the same buffer
		results.push(run_concurrent_increment(
			&device,
			&command_queue,
			&inc_pipeline,
			num_slots,
			gpu_threads,
			cpu_threads,
			iterations,
		)?);

		// Test 3: claim-with-contention — verifies "first-writer-wins" CAS
		let claim_threads = gpu_threads.max(num_slots * 4); // ensure contention
		results.push(run_claim_contention(
			&device,
			&command_queue,
			&claim_pipeline,
			num_slots,
			claim_threads,
		)?);

		Ok(results)
	}

	fn run_gpu_only_increment(
		device: &Device,
		command_queue: &metal::CommandQueueRef,
		pipeline: &ComputePipelineState,
		num_slots: usize,
		gpu_threads: usize,
		iterations: usize,
	) -> Result<MicrobenchResult, String>
	{
		let t0 = std::time::Instant::now();

		// Allocate shared buffer, init to 0
		let buf_bytes = (num_slots * 8) as u64;
		let buffer = device.new_buffer(buf_bytes, MTLResourceOptions::StorageModeShared);
		let ptr = buffer.contents() as *mut u64;
		unsafe {
			std::ptr::write_bytes(ptr, 0, num_slots);
		}

		// Params buffers
		let n_slots_u32 = num_slots as u32;
		let iters_u32 = iterations as u32;
		let n_buf = device.new_buffer_with_data(
			&n_slots_u32 as *const _ as *const _,
			4,
			MTLResourceOptions::StorageModeShared,
		);
		let i_buf = device.new_buffer_with_data(
			&iters_u32 as *const _ as *const _,
			4,
			MTLResourceOptions::StorageModeShared,
		);

		// Dispatch
		let cmd = command_queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(pipeline);
		enc.set_buffer(0, Some(&buffer), 0);
		enc.set_buffer(1, Some(&n_buf), 0);
		enc.set_buffer(2, Some(&i_buf), 0);
		let grid = MTLSize::new(gpu_threads as u64, 1, 1);
		let tg = MTLSize::new((gpu_threads as u64).min(64), 1, 1);
		enc.dispatch_threads(grid, tg);
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		// Sum slot values
		let slice = unsafe { std::slice::from_raw_parts(ptr, num_slots) };
		let total: u64 = slice.iter().map(|&v| v as u64).sum();
		let expected = (gpu_threads * iterations) as u64;
		let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

		Ok(MicrobenchResult {
			test_name: "gpu_only_increment".into(),
			passed: total == expected,
			expected,
			observed: total,
			details: format!(
				"GPU {} threads × {} iters = {} expected; got {}",
				gpu_threads, iterations, expected, total
			),
			elapsed_ms,
		})
	}

	fn run_concurrent_increment(
		device: &Device,
		command_queue: &metal::CommandQueueRef,
		pipeline: &ComputePipelineState,
		num_slots: usize,
		gpu_threads: usize,
		cpu_threads: usize,
		iterations: usize,
	) -> Result<MicrobenchResult, String>
	{
		use rayon::prelude::*;
		let t0 = std::time::Instant::now();

		let buf_bytes = (num_slots * 8) as u64;
		let buffer = device.new_buffer(buf_bytes, MTLResourceOptions::StorageModeShared);
		let ptr = buffer.contents() as *mut u64;
		unsafe {
			std::ptr::write_bytes(ptr, 0, num_slots);
		}

		let n_slots_u32 = num_slots as u32;
		let iters_u32 = iterations as u32;
		let n_buf = device.new_buffer_with_data(
			&n_slots_u32 as *const _ as *const _,
			4,
			MTLResourceOptions::StorageModeShared,
		);
		let i_buf = device.new_buffer_with_data(
			&iters_u32 as *const _ as *const _,
			4,
			MTLResourceOptions::StorageModeShared,
		);

		// Dispatch GPU (async — doesn't wait)
		let cmd = command_queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(pipeline);
		enc.set_buffer(0, Some(&buffer), 0);
		enc.set_buffer(1, Some(&n_buf), 0);
		enc.set_buffer(2, Some(&i_buf), 0);
		let grid = MTLSize::new(gpu_threads as u64, 1, 1);
		let tg = MTLSize::new((gpu_threads as u64).min(64), 1, 1);
		enc.dispatch_threads(grid, tg);
		enc.end_encoding();
		cmd.commit();

		// Meanwhile, CPU rayon increments the SAME buffer
		// SAFETY: we're holding the buffer alive for the duration; AtomicU64 is
		// layout-compatible with u64; we have unique mutable access via the buffer
		// pointer (Metal's shared storage model permits concurrent CPU+GPU atomic ops).
		let atomic_slice: &[AtomicU64] =
			unsafe { std::slice::from_raw_parts(ptr as *const AtomicU64, num_slots) };

		(0..cpu_threads).into_par_iter().for_each(|tid| {
			let slot = &atomic_slice[tid % num_slots];
			for _ in 0..iterations
			{
				let _ = slot.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| Some(v + 1));
			}
		});

		// Wait for GPU
		cmd.wait_until_completed();

		// Sum
		let slice = unsafe { std::slice::from_raw_parts(ptr, num_slots) };
		let total: u64 = slice.iter().map(|&v| v as u64).sum();
		let expected = ((gpu_threads + cpu_threads) * iterations) as u64;
		let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

		Ok(MicrobenchResult {
			test_name: "concurrent_cpu_gpu_increment".into(),
			passed: total == expected,
			expected,
			observed: total,
			details: format!(
				"GPU {} + CPU {} threads × {} iters = {} expected; got {} (delta {})",
				gpu_threads,
				cpu_threads,
				iterations,
				expected,
				total,
				expected as i64 - total as i64,
			),
			elapsed_ms,
		})
	}

	fn run_claim_contention(
		device: &Device,
		command_queue: &metal::CommandQueueRef,
		pipeline: &ComputePipelineState,
		num_slots: usize,
		claim_threads: usize,
	) -> Result<MicrobenchResult, String>
	{
		let t0 = std::time::Instant::now();

		let buf_bytes = (num_slots * 8) as u64;
		let buffer = device.new_buffer(buf_bytes, MTLResourceOptions::StorageModeShared);
		let ptr = buffer.contents() as *mut u64;
		// Init to EMPTY_SENTINEL (0xFFFFFFFFFFFFFFFF)
		unsafe {
			for i in 0..num_slots
			{
				*ptr.add(i) = EMPTY_SENTINEL;
			}
		}

		let success_bytes = (claim_threads * 4) as u64;
		let success_buf = device.new_buffer(success_bytes, MTLResourceOptions::StorageModeShared);
		let success_ptr = success_buf.contents() as *mut u32;
		unsafe {
			std::ptr::write_bytes(success_ptr, 0, claim_threads);
		}

		let n_slots_u32 = num_slots as u32;
		let n_buf = device.new_buffer_with_data(
			&n_slots_u32 as *const _ as *const _,
			4,
			MTLResourceOptions::StorageModeShared,
		);

		let cmd = command_queue.new_command_buffer();
		let enc = cmd.new_compute_command_encoder();
		enc.set_compute_pipeline_state(pipeline);
		enc.set_buffer(0, Some(&buffer), 0);
		enc.set_buffer(1, Some(&success_buf), 0);
		enc.set_buffer(2, Some(&n_buf), 0);
		let grid = MTLSize::new(claim_threads as u64, 1, 1);
		let tg = MTLSize::new((claim_threads as u64).min(64), 1, 1);
		enc.dispatch_threads(grid, tg);
		enc.end_encoding();
		cmd.commit();
		cmd.wait_until_completed();

		// Verify exactly num_slots successes (one winner per slot)
		let success_slice = unsafe { std::slice::from_raw_parts(success_ptr, claim_threads) };
		let total_winners: u32 = success_slice.iter().sum();

		// Verify each slot has a value != EMPTY_SENTINEL and the winning tid was indeed
		// flagged as a winner
		let slot_slice = unsafe { std::slice::from_raw_parts(ptr, num_slots) };
		let mut consistent = true;
		for (slot_idx, &val) in slot_slice.iter().enumerate()
		{
			if val == EMPTY_SENTINEL
			{
				consistent = false;
				break;
			}
			// val == winner_tid + 1
			let winner_tid = (val - 1) as usize;
			if winner_tid >= claim_threads
			{
				consistent = false;
				break;
			}
			if success_slice[winner_tid] != 1
			{
				consistent = false;
				break;
			}
			// Validate slot assignment matches
			if winner_tid % num_slots != slot_idx
			{
				consistent = false;
				break;
			}
		}

		let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
		let passed = total_winners as usize == num_slots && consistent;

		Ok(MicrobenchResult {
			test_name: "claim_contention".into(),
			passed,
			expected: num_slots as u64,
			observed: total_winners as u64,
			details: format!(
			"{} threads contended for {} slots; expected {} winners, got {}, slot/winner consistency: {}",
			claim_threads, num_slots, num_slots, total_winners, consistent,
		),
			elapsed_ms,
		})
	}
} // mod metal_impl

#[cfg(target_os = "macos")]
pub use metal_impl::run_microbench;
#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub use metal_impl::MicrobenchResult;
