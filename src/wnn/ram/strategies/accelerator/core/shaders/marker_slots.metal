//! marker_slots.metal — canonical GPU cell-write primitives (marker-FSM slot
//! claim + clamped/OI nudge). SINGLE SOURCE shared by the IDS trainer
//! (marker_train.metal) and the controller trainer. Prepended AFTER common.metal
//! (so WNN cell constants are available) and BEFORE the consuming kernel body.
//! Moved here 2026-06-20 from marker_train.metal (verbatim) to stop duplication.

constant uint MARKER_EMPTY = 0u;
constant uint MARKER_CLAIMED = 1u;
constant uint MARKER_FINAL = 0xFFFFFFFFu;

// Murmur3 finalizer — identical mixer to Rust MarkerInner::hash.
inline uint slot_hash(ulong key, uint mask)
{
	ulong x = key;
	x ^= x >> 33;
	x *= 0xff51afd7ed558ccdul;
	x ^= x >> 33;
	x *= 0xc4ceb9fe1a85ec53ul;
	x ^= x >> 33;
	return uint(x) & mask;
}

// Find-or-claim slot for (neuron, key). Returns slot index, or 0xFFFFFFFF on
// table-full. Caller must check the return value before writing.
//
// Mirrors Rust MarkerInner::find_or_claim_slot but adapted for GPU:
//   - No spin_loop() hint; the CLAIMED wait below is bounded (it waits on
//     ANOTHER thread's progress — unbounded could deadlock a diverged SIMD
//     group). The claim CAS itself retries until decided (lock-free).
//   - On CLAIMED slot, probe forward instead of waiting (loses some slot
//     coalescing but avoids deadlock; load factor accommodates)
inline uint find_or_claim_slot(
		device atomic_uint *slot_markers,
		device ulong *slot_keys,
		uint slot_offset,
		uint slot_capacity,
		ulong key)
{
	uint mask = slot_capacity - 1;
	uint idx = slot_hash(key, mask);
	for (uint probe = 0; probe < slot_capacity; probe++)
	{
		uint slot = slot_offset + idx;
		uint m = atomic_load_explicit(&slot_markers[slot], memory_order_relaxed);
		if (m == MARKER_FINAL)
		{
			if (slot_keys[slot] == key)
				return slot;
			// Different key — probe forward.
		}
		else if (m == MARKER_EMPTY)
		{
			uint expected = MARKER_EMPTY;
			bool won = false;
			// Weak CAS: retry until it either wins or sees another claimer. A
			// bounded retry (was 4) fell through on spurious failures with the
			// slot still EMPTY and probed forward -> a duplicate-key slot.
			while (true)
			{
				if (atomic_compare_exchange_weak_explicit(
								&slot_markers[slot], &expected, MARKER_CLAIMED,
								memory_order_relaxed, memory_order_relaxed))
				{
					won = true;
					break;
				}
				if (expected != MARKER_EMPTY)
					break; // someone else won
			}
			if (won)
			{
				// Exclusive access — write key, then publish FINAL.
				slot_keys[slot] = key;
				atomic_store_explicit(&slot_markers[slot], MARKER_FINAL, memory_order_relaxed);
				return slot;
			}
			// Lost the race; re-examine
			if (expected == MARKER_FINAL && slot_keys[slot] == key)
				return slot;
			// If still CLAIMED, fall through to wait below.
			m = expected;
		}
		if (m != MARKER_EMPTY && m != MARKER_FINAL)
		{
			// CLAIMED — bounded wait to let the writer resolve. Avoids
			// most same-key duplicate inserts that would happen if we just
			// probed forward. Bounded to prevent warp deadlock.
			uint resolved = m;
			for (uint w = 0; w < 64; w++)
			{
				resolved = atomic_load_explicit(&slot_markers[slot], memory_order_relaxed);
				if (resolved == MARKER_FINAL || resolved == MARKER_EMPTY)
					break;
			}
			if (resolved == MARKER_FINAL && slot_keys[slot] == key)
				return slot;
			// If resolved == EMPTY (writer aborted? shouldn't happen) or
			// FINAL with different key, fall through to probe forward.
		}
		// Probe forward
		idx = (idx + 1) & mask;
	}
	return 0xFFFFFFFFu; // table full
}

// Apply nudge (clamped delta) to the value field. Mirrors CPU MarkerInner::nudge.
//
// Every CAS loop below is UNBOUNDED (12/10/2026; were 8 / 8 / 256 retries). A
// bounded loop SILENTLY DROPPED the write when contention outlasted it — the
// cross-run nondeterminism oi_z_parity used to flake on. Unbounded is safe: the
// loops are lock-free, not locks — a failed CAS means another thread's CAS on
// the same slot SUCCEEDED, so some thread always progresses and no thread ever
// waits on another's progress (no SIMD-divergence deadlock).
inline void slot_nudge(device atomic_uint *slot_values, uint slot, bool target_true)
{
	int delta = target_true ? 1 : -1;
	while (true)
	{
		uint current = atomic_load_explicit(&slot_values[slot], memory_order_relaxed);
		int new_cell = clamp(int(current) + delta, 0, 3);
		if (uint(new_cell) == current)
			return; // saturated, no change
		uint exp = current;
		if (atomic_compare_exchange_weak_explicit(
						&slot_values[slot], &exp, uint(new_cell),
						memory_order_relaxed, memory_order_relaxed))
			return;
	}
}

// TERNARY lattice write (12/07/2026 GPU-train generalization): mirrors the
// CPU IDS trainer's TRUE-wins semantics (GroupMemory::write, allow_override=
// false): TRUE absorbs anything; FALSE only lands on non-TRUE. Order-
// independent (monotone lattice join) → exact under any thread interleaving.
// Cell codes are the ternary encoding: FALSE=0, TRUE=1 (EMPTY=2 is only the
// unclaimed/init default — every claimed slot is written immediately).
inline void slot_write_ternary(device atomic_uint *slot_values, uint slot, bool target_true)
{
	if (target_true)
	{
		atomic_store_explicit(&slot_values[slot], 1u, memory_order_relaxed);
		return;
	}
	while (true)
	{
		uint current = atomic_load_explicit(&slot_values[slot], memory_order_relaxed);
		if (current == 1u || current == 0u)
			return; // TRUE wins / already FALSE
		uint exp = current;
		if (atomic_compare_exchange_weak_explicit(
						&slot_values[slot], &exp, 0u,
						memory_order_relaxed, memory_order_relaxed))
			return;
	}
}

// BINARY (classical WiSARD) one-shot set: idempotent TRUE store. FALSE-
// direction participants never reach here (skipped before slot claim).
inline void slot_write_binary(device atomic_uint *slot_values, uint slot)
{
	atomic_store_explicit(&slot_values[slot], 1u, memory_order_relaxed);
}

// OI (order-independent) packed counter constants — must match
// neuron_memory.rs OI_* constants.
constant uint OI_NET_MASK = 0x3FFFFFFFu;
constant uint OI_OBS_GE_1_BIT = 30u;
constant uint OI_OBS_GE_2_BIT = 31u;
constant int OI_NET_MAX_INT = (1 << 29) - 1;
constant int OI_NET_MIN_INT = -(1 << 29);

// Pure transition function for the packed (obs, net) counter. Returns the
// new packed value given an old one and a signed weight delta. Mirrors
// neuron_memory.rs::oi_apply_nudge exactly.
inline uint oi_apply_nudge_inline(uint old, int delta)
{
	uint net30 = old & OI_NET_MASK;
	int net;
	if ((net30 & (1u << 29)) != 0u)
	{
		// Sign-extend 30-bit → 32-bit.
		net = int(net30 | (~OI_NET_MASK));
	}
	else
	{
		net = int(net30);
	}
	bool old_obs1 = ((old >> OI_OBS_GE_1_BIT) & 1u) != 0u;
	bool old_obs2 = ((old >> OI_OBS_GE_2_BIT) & 1u) != 0u;

	// saturating_add at 30-bit boundary.
	int new_net = net + delta;
	if (new_net > OI_NET_MAX_INT)
		new_net = OI_NET_MAX_INT;
	if (new_net < OI_NET_MIN_INT)
		new_net = OI_NET_MIN_INT;

	bool new_obs1 = true;
	bool new_obs2 = old_obs1 || old_obs2;

	uint new_net30 = uint(new_net) & OI_NET_MASK;
	uint o1 = uint(new_obs1) << OI_OBS_GE_1_BIT;
	uint o2 = uint(new_obs2) << OI_OBS_GE_2_BIT;
	return o2 | o1 | new_net30;
}

// Order-independent nudge: accumulates ±weight into the packed counter via
// CAS. Replaces the clamped slot_nudge when TrainParams.oi_mode == 1.
inline void slot_nudge_oi(device atomic_uint *slot_values, uint slot, int delta)
{
	// Unbounded (was 16, 64, then 256 retries — each exhausted loop silently
	// dropped a nudge: ~1 in 4/20 oi_z_parity runs at 64). See slot_nudge.
	while (true)
	{
		uint old = atomic_load_explicit(&slot_values[slot], memory_order_relaxed);
		uint nw = oi_apply_nudge_inline(old, delta);
		if (nw == old)
			return; // saturated, no change
		uint exp = old;
		if (atomic_compare_exchange_weak_explicit(
						&slot_values[slot], &exp, nw,
						memory_order_relaxed, memory_order_relaxed))
			return;
	}
}
