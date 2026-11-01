// =============================================================================
// marker_direct.metal — DIRECT-addressed slot regions (IDS-34, 01/11/2026)
// =============================================================================
//
// Worker-only (IDS trainer + probe-eval). Prepended AFTER
// core/shaders/marker_slots.metal (MARKER_* constants) and BEFORE the kernel
// body, in BOTH marker_train.metal's and marker_probe_eval.metal's pipelines —
// the trainer and every reader of a trained table must agree on the layout,
// so the predicate lives here once.
//
// When a neuron's whole key space fits its slot region (2^bits <=
// slot_capacity) the key IS the slot: slot = slot_offset + key. That is the
// regime where marker_capacity_for_train caps the region at 2 x 2^bits (more
// training visits than addresses), and where the hashed find_or_claim_slot was
// NOT safe: with z (up to 1024) threads per neuron, a claimer that out-waits a
// pending CLAIMED probes forward and publishes a duplicate-key slot; hundreds
// of those filled the region, find_or_claim_slot returned table-full and the
// write was silently dropped — which writes, depended on GPU scheduling. Same
// genome, same data, different memory every call (b = 8 always, some b = 12).
//
// Direct addressing has no hash, no probe, no claim race: every visitor of a
// key writes the same key into the same slot and publishes FINAL (idempotent),
// so there is never a duplicate and never a full table. The export walks FINAL
// slots and sorts by key, so it is layout-agnostic: wherever the hashed table
// never overflowed, the exported memory is bit-identical
// (marker_repeatability_tests golden).

inline bool marker_region_is_direct(uint bits, uint slot_capacity)
{
	return bits < 32u && (1u << bits) <= slot_capacity;
}

// The slot of `key` in a direct region, published FINAL. No CAS needed: all
// writers store the identical key, and the marker only ever moves to FINAL.
inline uint claim_direct_slot(
		device atomic_uint *slot_markers,
		device ulong *slot_keys,
		uint slot_offset,
		ulong key)
{
	uint slot = slot_offset + uint(key);
	if (atomic_load_explicit(&slot_markers[slot], memory_order_relaxed) != MARKER_FINAL)
	{
		slot_keys[slot] = key;
		atomic_store_explicit(&slot_markers[slot], MARKER_FINAL, memory_order_relaxed);
	}
	return slot;
}
