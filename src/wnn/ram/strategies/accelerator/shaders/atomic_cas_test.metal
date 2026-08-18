#include <metal_stdlib>
using namespace metal;

// Diagnostic — error early if we accidentally compile on too-old MSL.
#if __METAL_VERSION__ < 300
#error "MSL 3.0+ required for 64-bit atomics"
#endif
#if !defined(__HAVE_ATOMIC_ULONG__)
#error "__HAVE_ATOMIC_ULONG__ not defined — 64-bit atomics unavailable on this target"
#endif

// Atomic CAS microbenchmark — validates that Metal `atomic_ulong` CAS
// (64-bit atomic CAS, requires MSL 3.0+) produces coherent updates on a
// shared buffer that is ALSO being CAS'd from CPU rayon workers.
//
// This is the foundational test for Option B: each genome's training writes
// will go through 64-bit atomic CAS on per-genome AtomicHashTable buffers.
// We need 64-bit because address keys for b > 32 architectures don't fit
// in u32.
//
// MSL 3.0+ 64-bit atomics only support `memory_order_relaxed`. Cross-CPU/GPU
// coherence via `memory_scope_system` is not supported for atomic CAS ops
// (per MSL spec). So this test will confirm 64-bit GPU-internal CAS works
// AND confirm the cross-CPU/GPU coherence limit (test 2 expected to fail
// the same way it did with u32 — coherence is a scope issue, not a size
// issue).

kernel void atomic_cas_increment(
		device atomic_ulong *slots [[buffer(0)]],
		constant uint &num_slots [[buffer(1)]],
		constant uint &iterations [[buffer(2)]],
		uint tid [[thread_position_in_grid]])
{
	uint slot_idx = tid % num_slots;
	for (uint i = 0; i < iterations; i++)
	{
		ulong current = atomic_load_explicit(&slots[slot_idx], memory_order_relaxed);
		ulong new_val;
		do
		{
			new_val = current + 1;
		} while (!atomic_compare_exchange_weak_explicit(
				&slots[slot_idx],
				&current,
				new_val,
				memory_order_relaxed,
				memory_order_relaxed));
	}
}

constant ulong EMPTY_SENTINEL = 0xFFFFFFFFFFFFFFFFul;

kernel void atomic_cas_claim(
		device atomic_ulong *slots [[buffer(0)]],
		device uint *success_flags [[buffer(1)]],
		constant uint &num_slots [[buffer(2)]],
		uint tid [[thread_position_in_grid]])
{
	uint slot_idx = tid % num_slots;
	ulong my_value = (ulong)(tid + 1);
	bool won = false;

	while (true)
	{
		ulong expected = EMPTY_SENTINEL;
		bool ok = atomic_compare_exchange_weak_explicit(
				&slots[slot_idx],
				&expected,
				my_value,
				memory_order_relaxed,
				memory_order_relaxed);
		if (ok)
		{
			won = true;
			break;
		}
		if (expected != EMPTY_SENTINEL)
		{
			won = false;
			break;
		}
	}
	success_flags[tid] = won ? 1u : 0u;
}
