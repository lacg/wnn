//! Lock-free atomic open-addressed hash table for sparse neuron memory.
//!
//! Drop-in replacement for `Vec<DashMap<u64, u8>>` used by `GroupSparseMemory`,
//! designed so the same flat-array representation can be used from CPU rayon
//! workers today and from Metal GPU shaders later (Apple Silicon supports
//! 64-bit atomics in MSL via `atomic_ulong`).
//!
//! Layout: per-neuron table holds parallel `Vec<AtomicU64>` keys and
//! `Vec<AtomicU8>` values, sized as a power-of-two for fast bitwise modulo.
//! Empty slot sentinel is `EMPTY_KEY = u64::MAX`. Capacity grows 2× when load
//! factor exceeds `RESIZE_LOAD_FACTOR` (0.75).
//!
//! Concurrency model:
//! - Normal `write` / `nudge` hold a *read* lock on the inner table — multiple
//!   writers run concurrently and coordinate via per-slot CAS only
//! - Resize takes a *write* lock, swaps the inner table, then drops the lock
//! - Readers/writers that race with a resize see either the old or new table;
//!   each call is single-shot (acquire lock → one slot op → release), so the
//!   table identity is consistent within a call. The wrapping `write` /
//!   `nudge` retry the operation after a resize.
//!
//! Capacity sizing helper `estimate_capacity` matches the existing
//! `calculate_pool_size` heuristic (~3K unique addresses per neuron at sparse
//! threshold) but accepts a hint based on actual `num_train` so per-flow
//! sizing scales to 46M-row workloads without forcing the maximum case at
//! every level.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;
use rayon::prelude::*;

pub const EMPTY_KEY: u64 = u64::MAX;
const RESIZE_LOAD_FACTOR: f32 = 0.75;
const MIN_CAPACITY: usize = 16;

/// Inner table — flat parallel arrays. All operations are lock-free atomic
/// CAS at the slot level. Capacity is fixed for the lifetime of an `Inner`;
/// growing the table requires building a fresh `Inner` and swapping it under
/// the outer `RwLock`.
struct Inner {
	capacity: usize,
	mask: u64,
	keys: Vec<AtomicU64>,
	values: Vec<AtomicU8>,
	// KEPT-API: miss-default; concurrent-map contract field
	#[allow(dead_code)]
	default_value: u8,
	approx_count: AtomicUsize,
	/// OI training counters: optional parallel u32 array, one packed counter
	/// per slot. Allocated by `init_oi_counters`, drained and dropped at
	/// `commit_oi`. Same slot indexing as `values` — find_or_claim_slot's
	/// return value indexes both arrays.
	counters: Option<Vec<AtomicU32>>,
}

impl Inner {
	fn new(capacity: usize, default_value: u8) -> Self {
		let cap = capacity.next_power_of_two().max(MIN_CAPACITY);
		let keys = (0..cap).map(|_| AtomicU64::new(EMPTY_KEY)).collect();
		let values = (0..cap).map(|_| AtomicU8::new(default_value)).collect();
		Self {
			capacity: cap,
			mask: (cap - 1) as u64,
			keys,
			values,
			default_value,
			approx_count: AtomicUsize::new(0),
			counters: None,
		}
	}

	/// Allocate the OI counter buffer in place. Idempotent.
	fn init_oi_counters(&mut self) {
		if self.counters.is_some() { return; }
		let buf: Vec<AtomicU32> = (0..self.capacity)
			.map(|_| AtomicU32::new(ram_core::neuron_memory::OI_INITIAL))
			.collect();
		self.counters = Some(buf);
	}

	/// Order-independent nudge: lock-free CAS on the per-slot packed counter.
	/// Returns true if the counter was updated (always, unless table is full).
	fn nudge_oi(&self, key: u64, delta: i32) -> bool {
		let slot = match self.find_or_claim_slot(key) {
			Some(s) => s,
			None => return false,
		};
		let counters = self.counters.as_ref()
			.expect("nudge_oi called without init_oi_counters");
		ram_core::neuron_memory::oi_nudge_atomic(&counters[slot], delta);
		true
	}

	/// Commit pass: for every claimed slot, bin its counter into the 2-bit
	/// value field, then drop the counter buffer. Slots with counter ==
	/// OI_INITIAL (untouched during OI) leave their existing value alone.
	fn commit_oi(&mut self) {
		let Some(counters) = self.counters.take() else { return; };
		for slot in 0..self.capacity {
			let k = self.keys[slot].load(Ordering::Relaxed);
			if k == EMPTY_KEY { continue; }
			let packed = counters[slot].load(Ordering::Relaxed);
			if packed == ram_core::neuron_memory::OI_INITIAL { continue; }
			let cell = ram_core::neuron_memory::oi_bin_to_cell(packed) as u8;
			self.values[slot].store(cell, Ordering::Relaxed);
		}
	}

	/// Murmur3-style finalizer — fast and good distribution for u64 keys.
	#[inline]
	fn hash(&self, key: u64) -> usize {
		let mut x = key;
		x ^= x >> 33;
		x = x.wrapping_mul(0xff51afd7ed558ccd);
		x ^= x >> 33;
		x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
		x ^= x >> 33;
		(x & self.mask) as usize
	}

	/// Lock-free probe: returns Some(slot) for an existing key, attempts to
	/// claim a fresh slot for new keys, or returns None if the table is
	/// (effectively) full.
	#[inline]
	fn find_or_claim_slot(&self, key: u64) -> Option<usize> {
		let mut idx = self.hash(key);
		for _ in 0..self.capacity {
			let current = self.keys[idx].load(Ordering::Acquire);
			if current == key {
				return Some(idx);
			}
			if current == EMPTY_KEY {
				match self.keys[idx].compare_exchange(
					EMPTY_KEY, key,
					Ordering::AcqRel, Ordering::Acquire,
				) {
					Ok(_) => {
						self.approx_count.fetch_add(1, Ordering::Relaxed);
						return Some(idx);
					}
					Err(actual) if actual == key => return Some(idx),
					Err(_) => { /* slot taken by another key; probe on */ }
				}
			}
			idx = (idx + 1) & (self.mask as usize);
		}
		None
	}

	/// Write following the existing TRUE-wins-over-FALSE protocol from
	/// `GroupSparseMemory::write`. Returns true if the slot's value was
	/// modified, false otherwise (including "table full" via find_or_claim).
	fn write(&self, key: u64, new_value: u8, allow_override: bool) -> bool {
		let slot = match self.find_or_claim_slot(key) {
			Some(s) => s,
			None => return false,
		};
		loop {
			let current = self.values[slot].load(Ordering::Acquire);
			if current == new_value { return false; }
			// TRUE (1) cannot be overwritten by FALSE (0)
			if current == 1 && new_value == 0 { return false; }
			// Without allow_override, FALSE only writes over EMPTY (2)
			if !allow_override && new_value == 0 && current != 2 { return false; }
			match self.values[slot].compare_exchange(
				current, new_value, Ordering::AcqRel, Ordering::Acquire,
			) {
				Ok(_) => return true,
				Err(_) => continue,
			}
		}
	}

	/// Nudge a cell one step toward target. Mirrors the QUAD nudge in
	/// `GroupSparseMemory::nudge` — values clamp into 0..=3.
	fn nudge(&self, key: u64, target_true: bool) -> bool {
		let slot = match self.find_or_claim_slot(key) {
			Some(s) => s,
			None => return false,
		};
		let delta: i64 = if target_true { 1 } else { -1 };
		loop {
			let current = self.values[slot].load(Ordering::Acquire);
			let new_cell = (current as i64 + delta).clamp(0, 3) as u8;
			if new_cell == current { return false; }
			match self.values[slot].compare_exchange(
				current, new_cell, Ordering::AcqRel, Ordering::Acquire,
			) {
				Ok(_) => return true,
				Err(_) => continue,
			}
		}
	}

	#[inline]
	fn read(&self, key: u64) -> u8 {
		let mut idx = self.hash(key);
		for _ in 0..self.capacity {
			let k = self.keys[idx].load(Ordering::Acquire);
			if k == key { return self.values[idx].load(Ordering::Acquire); }
			if k == EMPTY_KEY { return self.default_value; }
			idx = (idx + 1) & (self.mask as usize);
		}
		self.default_value
	}

	fn load_factor(&self) -> f32 {
		self.approx_count.load(Ordering::Relaxed) as f32 / self.capacity as f32
	}

	fn is_overloaded(&self) -> bool {
		self.load_factor() >= RESIZE_LOAD_FACTOR
	}

	/// Build a new Inner with 2× capacity and copy all live entries over.
	/// Caller must guarantee no concurrent writes are in flight (we hold the
	/// outer write lock).
	fn grow_2x(&self) -> Self {
		let mut new_inner = Self::new(self.capacity * 2, self.default_value);
		// If OI counters were active, allocate fresh counters in the new table.
		if self.counters.is_some() {
			new_inner.init_oi_counters();
		}
		for i in 0..self.capacity {
			let k = self.keys[i].load(Ordering::Relaxed);
			if k != EMPTY_KEY {
				let v = self.values[i].load(Ordering::Relaxed);
				// Unchecked insert: we know the new table is freshly empty and
				// (key, value) pairs are unique, so a direct probe+claim is
				// safe and fast.
				if let Some(slot) = new_inner.find_or_claim_slot(k) {
					new_inner.values[slot].store(v, Ordering::Relaxed);
					// Copy counter too if OI is active.
					if let (Some(ref old_ctr), Some(ref new_ctr)) = (&self.counters, &new_inner.counters) {
						let packed = old_ctr[i].load(Ordering::Relaxed);
						new_ctr[slot].store(packed, Ordering::Relaxed);
					}
				}
			}
		}
		new_inner
	}

	/// Snapshot for GPU export. Returns sorted (key, value) pairs.
	fn snapshot_sorted(&self) -> Vec<(u64, u8)> {
		let mut out = Vec::with_capacity(self.approx_count.load(Ordering::Relaxed));
		for i in 0..self.capacity {
			let k = self.keys[i].load(Ordering::Relaxed);
			if k != EMPTY_KEY {
				let v = self.values[i].load(Ordering::Relaxed);
				out.push((k, v));
			}
		}
		out.sort_by_key(|(k, _)| *k);
		out
	}

	fn len(&self) -> usize {
		self.approx_count.load(Ordering::Relaxed)
	}
}

/// Public hash table. The outer `RwLock` only synchronises rare resize events;
/// all hot-path operations hold the *read* lock and run lock-free via atomic
/// CAS at the slot level.
pub struct AtomicHashTable {
	inner: RwLock<Inner>,
	// KEPT-API: miss-default; map contract field (AtomicHashTable)
	#[allow(dead_code)]
	default_value: u8,
}

impl AtomicHashTable {
	pub fn new(initial_capacity: usize, default_value: u8) -> Self {
		Self {
			inner: RwLock::new(Inner::new(initial_capacity, default_value)),
			default_value,
		}
	}

	pub fn write(&self, key: u64, value: u8, allow_override: bool) -> bool {
		loop {
			// Hot path: read lock, attempt write
			let needs_resize = {
				let guard = self.inner.read().expect("AtomicHashTable RwLock poisoned");
				let result = guard.write(key, value, allow_override);
				// Decide whether to trigger a resize based on post-op load.
				// Returning a successful write while the table is now full is
				// fine — next call will resize before doing more work.
				if result { return result; }
				// Result was false: either no-op (same value, protocol reject) or
				// table full. Distinguish by load_factor.
				guard.is_overloaded()
			};
			if needs_resize {
				self.maybe_resize();
				continue;
			}
			return false;
		}
	}

	pub fn nudge(&self, key: u64, target_true: bool) -> bool {
		loop {
			let needs_resize = {
				let guard = self.inner.read().expect("AtomicHashTable RwLock poisoned");
				let result = guard.nudge(key, target_true);
				if result { return result; }
				guard.is_overloaded()
			};
			if needs_resize {
				self.maybe_resize();
				continue;
			}
			return false;
		}
	}

	/// Allocate the OI counter buffer. Idempotent. Takes the write lock once.
	pub fn init_oi_counters(&self) {
		let mut guard = self.inner.write().expect("AtomicHashTable RwLock poisoned");
		guard.init_oi_counters();
	}

	/// Order-independent nudge: lock-free CAS on per-slot packed counter.
	/// Resizes the table on overload (same protocol as `nudge`).
	pub fn nudge_oi(&self, key: u64, delta: i32) -> bool {
		loop {
			let needs_resize = {
				let guard = self.inner.read().expect("AtomicHashTable RwLock poisoned");
				let result = guard.nudge_oi(key, delta);
				if result { return result; }
				guard.is_overloaded()
			};
			if needs_resize {
				self.maybe_resize();
				continue;
			}
			return false;
		}
	}

	/// Commit pass: bin per-slot counters into the 2-bit value field and
	/// drop the counter buffer. Takes the write lock once.
	pub fn commit_oi(&self) {
		let mut guard = self.inner.write().expect("AtomicHashTable RwLock poisoned");
		guard.commit_oi();
	}

	pub fn read(&self, key: u64) -> u8 {
		let guard = self.inner.read().expect("AtomicHashTable RwLock poisoned");
		guard.read(key)
	}

	// KEPT-API: concurrent-map API completeness (len/clear/capacity family)
	#[allow(dead_code)]
	pub fn len(&self) -> usize {
		let guard = self.inner.read().expect("AtomicHashTable RwLock poisoned");
		guard.len()
	}

	/// Reset to empty, keeping the current capacity. Used by GenomePool slot
	/// reuse so we avoid reallocating on every iteration.
	// KEPT-API: map API completeness (GenomePool slot reuse)
	#[allow(dead_code)]
	pub fn clear(&self) {
		let mut guard = self.inner.write().expect("AtomicHashTable RwLock poisoned");
		let cap = guard.capacity;
		let dv = self.default_value;
		*guard = Inner::new(cap, dv);
	}

	/// Sorted snapshot for GPU export. Caller decides how to integrate into
	/// the SparseGpuExport layout.
	pub fn snapshot_sorted(&self) -> Vec<(u64, u8)> {
		let guard = self.inner.read().expect("AtomicHashTable RwLock poisoned");
		guard.snapshot_sorted()
	}

	fn maybe_resize(&self) {
		let mut guard = self.inner.write().expect("AtomicHashTable RwLock poisoned");
		// Double-check under write lock — another thread may have resized while
		// we waited.
		if !guard.is_overloaded() { return; }
		let new_inner = guard.grow_2x();
		*guard = new_inner;
	}
}

/// Capacity heuristic — scaled by num_train (sparse_keys grows roughly with
/// sqrt(num_train)); floor of 3K unique addresses per neuron at the sparse
/// threshold. (calculate_pool_size now sizes via marker_capacity_for_train
/// — dataset-size-aware since the 05/07/2026 jetsam root-cause fix.)
pub fn estimate_capacity(num_train: usize) -> usize {
	let expected = ((num_train as f64).sqrt() * 20.0).max(3000.0) as usize;
	((expected as f32 / 0.25) as usize).next_power_of_two().max(MIN_CAPACITY)
}

// =============================================================================
// MarkerHashTable — GPU-compatible variant (Option B-marker)
// =============================================================================
//
// Why a separate type from AtomicHashTable: the Apple Metal stdlib (as of
// macOS 26.3 / Metal SDK 26.5) does NOT expose 64-bit atomic CAS — the
// `_valid_load_type` and `_valid_compare_exchange_type` SFINAE templates only
// list 32-bit integer types. We need 64-bit keys (b > 32 architectures), so
// AtomicHashTable's u64-CAS approach can't run on GPU.
//
// Workaround: keep keys as plain u64 (no CAS on them). Coordinate writers via
// a separate 32-bit marker per slot. The marker FSM (EMPTY → CLAIMED → FINAL)
// gives exclusive access between claim and finalize, during which the writer
// safely writes the key non-atomically. The same code path works on CPU and
// GPU because 32-bit atomic CAS is available in MSL on both.
//
// Memory cost: 16 bytes per slot vs AtomicHashTable's 9. The 1.8× memory
// overhead is the price of GPU compatibility.

/// Marker FSM values. Anything outside {EMPTY, FINAL} represents a slot
/// transiently held by a writer (the value is conventionally `writer_tid + 1`
/// but only EMPTY/FINAL are protocol-significant — any non-sentinel value
/// works for the "claimed mid-write" state).
pub const MARKER_EMPTY: u32 = 0;
pub const MARKER_FINAL: u32 = u32::MAX;
const MARKER_CLAIMED_SENTINEL: u32 = 1;

/// Backing storage for MarkerHashTable's three parallel arrays. Heap variant
/// is the CPU-only path (Vec-backed); Metal variant uses Apple Silicon's
/// unified-memory `MTLResourceOptions::StorageModeShared` buffers so the
/// same memory can be read/written from both CPU rayon workers and Metal
/// GPU kernels. Both expose `&[AtomicU32]` and raw `*mut u64` for key
/// access — the only difference is who owns the underlying allocation.
enum MarkerStorage {
	Heap {
		markers: Vec<AtomicU32>,
		keys: Vec<std::cell::UnsafeCell<u64>>,
		values: Vec<AtomicU32>,
	},
	#[cfg(target_os = "macos")]
	Metal {
		markers_buf: metal::Buffer,
		keys_buf: metal::Buffer,
		values_buf: metal::Buffer,
		capacity: usize,
	},
}

// SAFETY: Metal buffer pointers stay alive for the lifetime of the buffer
// handle (refcounted via metal-rs). The buffer's memory is in unified
// storage; the AtomicU32 slice we hand out is a view into it. Concurrent
// CPU access is safe because the values are AtomicU32. Concurrent GPU
// access is safe because Metal's atomic ops on the same buffer are
// memory_scope_device — coherent within GPU execution. Cross-CPU/GPU
// coherence is *not* relied upon: the marker FSM serializes via 32-bit
// CAS (which works on both), and key writes are protected by marker
// state.
#[cfg(target_os = "macos")]
unsafe impl Send for MarkerStorage {}
#[cfg(target_os = "macos")]
unsafe impl Sync for MarkerStorage {}

impl MarkerStorage {
	fn new_heap(capacity: usize) -> Self {
		Self::Heap {
			markers: (0..capacity).map(|_| AtomicU32::new(MARKER_EMPTY)).collect(),
			keys: (0..capacity).map(|_| std::cell::UnsafeCell::new(0u64)).collect(),
			values: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
		}
	}

	#[cfg(target_os = "macos")]
	fn new_metal(device: &metal::Device, capacity: usize, default_value: u8) -> Self {
		use metal::MTLResourceOptions;
		let markers_bytes = (capacity * 4) as u64;
		let keys_bytes = (capacity * 8) as u64;
		let values_bytes = (capacity * 4) as u64;

		let markers_buf = device.new_buffer(markers_bytes, MTLResourceOptions::StorageModeShared);
		let keys_buf = device.new_buffer(keys_bytes, MTLResourceOptions::StorageModeShared);
		let values_buf = device.new_buffer(values_bytes, MTLResourceOptions::StorageModeShared);

		// Init markers to EMPTY (0) — zero-init is already the default for
		// a fresh buffer in shared storage mode, but explicit for clarity.
		unsafe {
			std::ptr::write_bytes(markers_buf.contents() as *mut u32, 0, capacity);
			// Keys are uninitialized — the marker FSM forbids access until CLAIMED.
			// Values initialized to default_value (low 8 bits) so reads on
			// not-yet-claimed slots return the right thing.
			let v_ptr = values_buf.contents() as *mut u32;
			let dv = default_value as u32;
			for i in 0..capacity {
				*v_ptr.add(i) = dv;
			}
		}

		Self::Metal { markers_buf, keys_buf, values_buf, capacity }
	}

	#[inline]
	fn markers(&self) -> &[AtomicU32] {
		match self {
			Self::Heap { markers, .. } => markers,
			#[cfg(target_os = "macos")]
			Self::Metal { markers_buf, capacity, .. } => unsafe {
				std::slice::from_raw_parts(markers_buf.contents() as *const AtomicU32, *capacity)
			},
		}
	}

	#[inline]
	fn values(&self) -> &[AtomicU32] {
		match self {
			Self::Heap { values, .. } => values,
			#[cfg(target_os = "macos")]
			Self::Metal { values_buf, capacity, .. } => unsafe {
				std::slice::from_raw_parts(values_buf.contents() as *const AtomicU32, *capacity)
			},
		}
	}

	/// Read a key by slot index. Caller must guarantee the slot's marker
	/// is FINAL (via the FSM protocol) — otherwise reads may race with the
	/// writer's non-atomic key store.
	#[inline]
	unsafe fn key_at(&self, slot: usize) -> u64 {
		match self {
			Self::Heap { keys, .. } => *keys[slot].get(),
			#[cfg(target_os = "macos")]
			Self::Metal { keys_buf, .. } => {
				let ptr = keys_buf.contents() as *const u64;
				*ptr.add(slot)
			},
		}
	}

	/// Write a key by slot index. Caller must guarantee the slot's marker
	/// is CLAIMED by this caller (FSM exclusive access) — otherwise the
	/// non-atomic store races with concurrent writers.
	#[inline]
	unsafe fn set_key_at(&self, slot: usize, key: u64) {
		match self {
			Self::Heap { keys, .. } => *keys[slot].get() = key,
			#[cfg(target_os = "macos")]
			Self::Metal { keys_buf, .. } => {
				let ptr = keys_buf.contents() as *mut u64;
				*ptr.add(slot) = key;
			},
		}
	}

	#[inline]
	// KEPT-API: map API completeness
	#[allow(dead_code)]
	fn capacity(&self) -> usize {
		match self {
			Self::Heap { markers, .. } => markers.len(),
			#[cfg(target_os = "macos")]
			Self::Metal { capacity, .. } => *capacity,
		}
	}

	/// Metal buffer accessors — used by future GPU training kernel to bind
	/// these arrays to compute encoder slots. Returns None for Heap storage.
	#[cfg(target_os = "macos")]
	#[allow(dead_code)]
	pub fn metal_buffers(&self) -> Option<(&metal::Buffer, &metal::Buffer, &metal::Buffer)> {
		match self {
			Self::Heap { .. } => None,
			Self::Metal { markers_buf, keys_buf, values_buf, .. } => {
				Some((markers_buf, keys_buf, values_buf))
			},
		}
	}
}

struct MarkerInner {
	capacity: usize,
	mask: u64,
	storage: MarkerStorage,
	// KEPT-API: miss-default; map contract field
	#[allow(dead_code)]
	default_value: u8,
	approx_count: AtomicUsize,
}

// SAFETY: per-slot key access is serialized by the slot's marker FSM —
// a slot's key is only written by the unique writer that transitioned
// marker EMPTY→CLAIMED, and only read once marker == FINAL. Readers and
// writers never overlap on the same key cell; key writes happen-before
// marker FINAL via Release/Acquire on the marker store/load.
unsafe impl Sync for MarkerInner {}

impl MarkerInner {
	fn new(capacity: usize, default_value: u8) -> Self {
		let cap = capacity.next_power_of_two().max(MIN_CAPACITY);
		let storage = MarkerStorage::new_heap(cap);
		let inner = Self {
			capacity: cap,
			mask: (cap - 1) as u64,
			storage,
			default_value,
			approx_count: AtomicUsize::new(0),
		};
		// Initialize values to default — Heap variant inits to 0; set to dv
		for v in inner.storage.values() {
			v.store(default_value as u32, Ordering::Relaxed);
		}
		inner
	}

	#[cfg(target_os = "macos")]
	fn new_metal(device: &metal::Device, capacity: usize, default_value: u8) -> Self {
		let cap = capacity.next_power_of_two().max(MIN_CAPACITY);
		let storage = MarkerStorage::new_metal(device, cap, default_value);
		Self {
			capacity: cap,
			mask: (cap - 1) as u64,
			storage,
			default_value,
			approx_count: AtomicUsize::new(0),
		}
	}

	#[inline]
	fn hash(&self, key: u64) -> usize {
		// Same murmur-style mixer as AtomicHashTable for consistent slot
		// distribution under workload comparisons.
		let mut x = key;
		x ^= x >> 33;
		x = x.wrapping_mul(0xff51afd7ed558ccd);
		x ^= x >> 33;
		x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
		x ^= x >> 33;
		(x & self.mask) as usize
	}

	#[inline]
	fn markers(&self) -> &[AtomicU32] { self.storage.markers() }
	#[inline]
	fn values(&self) -> &[AtomicU32] { self.storage.values() }
	/// SAFETY: caller must guarantee marker[slot] == FINAL (key has been
	/// fully written by its claiming writer).
	#[inline]
	unsafe fn key_at(&self, slot: usize) -> u64 { self.storage.key_at(slot) }
	/// SAFETY: caller must hold this slot's CLAIMED marker (FSM exclusive
	/// access between CAS to CLAIMED and CAS/store to FINAL).
	#[inline]
	unsafe fn set_key_at(&self, slot: usize, key: u64) { self.storage.set_key_at(slot, key) }

	/// Wait for a slot to leave the CLAIMED state. Returns the resolved marker
	/// (EMPTY if the writer aborted — shouldn't happen in our protocol — or
	/// FINAL once the writer finalizes).
	#[inline]
	fn wait_for_resolution(&self, slot: usize) -> u32 {
		// Bounded spin — in practice CLAIMED is held for just a few writes
		// (key + value store + finalize CAS). Microseconds at most.
		let markers = self.markers();
		loop {
			let m = markers[slot].load(Ordering::Acquire);
			if m == MARKER_FINAL || m == MARKER_EMPTY {
				return m;
			}
			std::hint::spin_loop();
		}
	}

	/// Find the slot owning `key`, claiming a fresh one if not found.
	/// Returns Some(slot) where marker is now FINAL with key == requested key,
	/// or None on table-full.
	///
	/// Claim semantics: on a probed-into EMPTY slot, this function tries to
	/// CAS to CLAIMED. If it wins, it writes the key and CASes to FINAL
	/// before returning. The caller then atomically updates the slot's value
	/// per the application's protocol (TRUE-wins / nudge).
	fn find_or_claim_slot(&self, key: u64) -> Option<usize> {
		let markers = self.markers();
		let mut idx = self.hash(key);
		for _ in 0..self.capacity {
			let m = markers[idx].load(Ordering::Acquire);
			if m == MARKER_FINAL {
				// SAFETY: marker == FINAL means a prior writer fully wrote
				// the key; no one else mutates it. Read is safe.
				let stored_key = unsafe { self.key_at(idx) };
				if stored_key == key {
					return Some(idx);
				}
				// Different key in this slot — probe forward.
			} else if m == MARKER_EMPTY {
				// Try to claim
				match markers[idx].compare_exchange(
					MARKER_EMPTY,
					MARKER_CLAIMED_SENTINEL,
					Ordering::AcqRel,
					Ordering::Acquire,
				) {
					Ok(_) => {
						// We own the slot. Write key non-atomically; the
						// FINAL store later serves as the release fence.
						// SAFETY: we hold exclusive access to this slot's
						// key cell — no other writer can be in CLAIMED here,
						// and readers respect the marker.
						unsafe { self.set_key_at(idx, key); }
						// Release: ensures the key write happens-before the
						// FINAL store visible to other threads.
						markers[idx].store(MARKER_FINAL, Ordering::Release);
						self.approx_count.fetch_add(1, Ordering::Relaxed);
						return Some(idx);
					}
					Err(_) => {
						// Lost the race — re-examine this slot
						let m2 = markers[idx].load(Ordering::Acquire);
						if m2 == MARKER_FINAL {
							let stored_key = unsafe { self.key_at(idx) };
							if stored_key == key { return Some(idx); }
						} else if m2 != MARKER_EMPTY && m2 != MARKER_FINAL {
							// Someone else is mid-claim. Wait for resolution,
							// then re-check.
							let resolved = self.wait_for_resolution(idx);
							if resolved == MARKER_FINAL {
								let stored_key = unsafe { self.key_at(idx) };
								if stored_key == key { return Some(idx); }
							}
						}
						// Probe forward in any other case.
					}
				}
			} else {
				// CLAIMED by another writer. Wait for resolution; if it
				// resolves to FINAL with our key, we found our slot. Otherwise
				// probe forward.
				let resolved = self.wait_for_resolution(idx);
				if resolved == MARKER_FINAL {
					let stored_key = unsafe { self.key_at(idx) };
					if stored_key == key { return Some(idx); }
				}
			}
			idx = (idx + 1) & (self.mask as usize);
		}
		None
	}

	/// Write following the TRUE-wins-over-FALSE protocol from
	/// `GroupSparseMemory::write`. Values: 0=FALSE, 1=WEAK_FALSE/EMPTY, 2=WEAK_TRUE,
	/// 3=TRUE (mode-dependent).
	fn write(&self, key: u64, new_value: u8, allow_override: bool) -> bool {
		let slot = match self.find_or_claim_slot(key) {
			Some(s) => s,
			None => return false,
		};
		let values = self.values();
		loop {
			let current = values[slot].load(Ordering::Acquire) as u8;
			if current == new_value { return false; }
			if current == 1 && new_value == 0 { return false; }
			if !allow_override && new_value == 0 && current != 2 { return false; }
			match values[slot].compare_exchange(
				current as u32,
				new_value as u32,
				Ordering::AcqRel,
				Ordering::Acquire,
			) {
				Ok(_) => return true,
				Err(_) => continue,
			}
		}
	}

	/// Nudge a cell one step toward target. Mirrors the QUAD nudge in
	/// `GroupSparseMemory::nudge` — values clamp into 0..=3.
	fn nudge(&self, key: u64, target_true: bool) -> bool {
		let slot = match self.find_or_claim_slot(key) {
			Some(s) => s,
			None => return false,
		};
		let values = self.values();
		let delta: i32 = if target_true { 1 } else { -1 };
		loop {
			let current = values[slot].load(Ordering::Acquire) as i32;
			let new_cell = (current + delta).clamp(0, 3) as u32;
			if new_cell == current as u32 { return false; }
			match values[slot].compare_exchange(
				current as u32,
				new_cell,
				Ordering::AcqRel,
				Ordering::Acquire,
			) {
				Ok(_) => return true,
				Err(_) => continue,
			}
		}
	}

	#[inline]
	fn read(&self, key: u64) -> u8 {
		let markers = self.markers();
		let values = self.values();
		let mut idx = self.hash(key);
		for _ in 0..self.capacity {
			let m = markers[idx].load(Ordering::Acquire);
			if m == MARKER_FINAL {
				let stored_key = unsafe { self.key_at(idx) };
				if stored_key == key {
					return values[idx].load(Ordering::Acquire) as u8;
				}
			} else if m == MARKER_EMPTY {
				return self.default_value;
			}
			// CLAIMED: treat as "not yet our slot" — probe forward without
			// waiting (reads should be fast; if probe runs out, we miss the
			// claim and return default, which is acceptable for sparse
			// memory's read-while-training contract).
			idx = (idx + 1) & (self.mask as usize);
		}
		self.default_value
	}

	fn load_factor(&self) -> f32 {
		self.approx_count.load(Ordering::Relaxed) as f32 / self.capacity as f32
	}

	fn is_overloaded(&self) -> bool {
		self.load_factor() >= RESIZE_LOAD_FACTOR
	}

	/// Build a new MarkerInner with 2× capacity and migrate live entries.
	/// Caller must hold the outer write lock (no concurrent writers).
	///
	/// Note: resize always produces a Heap-backed table. For Metal-backed
	/// tables, capacity must be sized correctly up-front (resize would
	/// re-allocate Metal buffers which is expensive — and the GPU might
	/// hold pointer references to the old buffer). For Option B's flow,
	/// AtomicHashTable instances live only for a single genome's training
	/// and are sized via `estimate_capacity(num_train)`.
	fn grow_2x(&self) -> Self {
		let markers = self.markers();
		let values = self.values();
		let new_inner = Self::new(self.capacity * 2, self.default_value);
		for i in 0..self.capacity {
			let m = markers[i].load(Ordering::Relaxed);
			if m == MARKER_FINAL {
				let k = unsafe { self.key_at(i) };
				let v = values[i].load(Ordering::Relaxed) as u8;
				if let Some(slot) = new_inner.find_or_claim_slot(k) {
					new_inner.values()[slot].store(v as u32, Ordering::Relaxed);
				}
			}
		}
		new_inner
	}

	fn snapshot_sorted(&self) -> Vec<(u64, u8)> {
		let markers = self.markers();
		let values = self.values();
		let mut out = Vec::with_capacity(self.approx_count.load(Ordering::Relaxed));
		for i in 0..self.capacity {
			let m = markers[i].load(Ordering::Relaxed);
			if m == MARKER_FINAL {
				let k = unsafe { self.key_at(i) };
				let v = values[i].load(Ordering::Relaxed) as u8;
				out.push((k, v));
			}
		}
		out.sort_by_key(|(k, _)| *k);
		out
	}

	fn len(&self) -> usize {
		self.approx_count.load(Ordering::Relaxed)
	}
}

/// GPU-compatible variant of AtomicHashTable using 3-state marker FSM
/// instead of 64-bit atomic CAS on keys. Same public API surface as
/// AtomicHashTable for ergonomic A/B swapping in callers.
pub struct MarkerHashTable {
	inner: RwLock<MarkerInner>,
	// KEPT-API: miss-default; map contract field
	#[allow(dead_code)]
	default_value: u8,
}

impl MarkerHashTable {
	/// Heap-backed (CPU-only) constructor. Use for testing and pure-CPU
	/// workloads. For GPU integration, use `new_metal`.
	pub fn new(initial_capacity: usize, default_value: u8) -> Self {
		Self {
			inner: RwLock::new(MarkerInner::new(initial_capacity, default_value)),
			default_value,
		}
	}

	/// Metal-backed constructor. The hashtable's marker/key/value arrays
	/// live in Apple Silicon unified memory (`StorageModeShared`) so the
	/// same memory can be read/written by both CPU rayon workers and
	/// Metal GPU kernels. CPU operations work identically to `new()`;
	/// GPU access goes through the buffers returned by `metal_buffers()`.
	///
	/// Note: Metal-backed tables don't currently support `clear()` or
	/// `grow_2x` (these would reallocate the Metal buffer, invalidating
	/// any GPU pointer references). Size via `estimate_capacity(num_train)`
	/// with adequate headroom for the genome's expected sparse_keys count.
	#[cfg(target_os = "macos")]
	pub fn new_metal(device: &metal::Device, initial_capacity: usize, default_value: u8) -> Self {
		Self {
			inner: RwLock::new(MarkerInner::new_metal(device, initial_capacity, default_value)),
			default_value,
		}
	}

	/// Expose the underlying Metal buffers for binding to a compute encoder.
	/// Returns None for Heap-backed tables. Order: (markers, keys, values).
	#[cfg(target_os = "macos")]
	pub fn metal_buffers(&self) -> Option<(metal::Buffer, metal::Buffer, metal::Buffer)> {
		let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
		guard.storage.metal_buffers().map(|(m, k, v)| (m.clone(), k.clone(), v.clone()))
	}

	/// Build a `SparseGpuExport`-compatible representation by walking the
	/// flat buffer per-neuron. `slot_offsets` and `slot_capacities` define
	/// each neuron's region within the flat buffer (same metadata used by
	/// the GPU training kernel via `NeuronTrainMeta`).
	///
	/// Returns (keys, values, offsets, counts) where keys/values are
	/// sorted per neuron and offsets/counts give per-neuron slicing
	/// — matching the existing `SparseGpuExport` shape that
	/// `MetalSparseEvaluator` already consumes.
	pub fn export_per_neuron(
		&self,
		slot_offsets: &[u32],
		slot_capacities: &[u32],
	) -> (Vec<u64>, Vec<u8>, Vec<u32>, Vec<u32>) {
		assert_eq!(slot_offsets.len(), slot_capacities.len(),
			"slot_offsets and slot_capacities must have same length");
		let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
		let markers = guard.storage.markers();
		let values = guard.storage.values();
		let storage_ref = &guard.storage;
		let markers_len = markers.len();

		let num_neurons = slot_offsets.len();

		// B6: scan + sort each neuron's slot region in parallel. The marker
		// FSM guarantees that any slot in MARKER_FINAL state has its key
		// fully written, so the read side is purely lock-free atomic loads —
		// safe to execute concurrently across neurons.
		let per_neuron: Vec<Vec<(u64, u8)>> = (0..num_neurons)
			.into_par_iter()
			.map(|n| {
				let off = slot_offsets[n] as usize;
				let cap = slot_capacities[n] as usize;
				let end = (off + cap).min(markers_len);
				// Pre-size to ~25% of cap to amortize reallocs without
				// wasting too much for under-loaded regions.
				let mut entries: Vec<(u64, u8)> = Vec::with_capacity(cap / 4);
				for slot in off..end {
					if markers[slot].load(Ordering::Relaxed) == MARKER_FINAL {
						// SAFETY: marker == FINAL → key is fully written.
						let k = unsafe { storage_ref.key_at(slot) };
						let v = values[slot].load(Ordering::Relaxed) as u8;
						entries.push((k, v));
					}
				}
				entries.sort_by_key(|(k, _)| *k);
				entries
			})
			.collect();

		// Serial concatenation (cheap — just copying into pre-sized buffers).
		let total: usize = per_neuron.iter().map(|v| v.len()).sum();
		let mut keys_out: Vec<u64> = Vec::with_capacity(total);
		let mut values_out: Vec<u8> = Vec::with_capacity(total);
		let mut offsets_out: Vec<u32> = Vec::with_capacity(num_neurons);
		let mut counts_out: Vec<u32> = Vec::with_capacity(num_neurons);

		for entries in per_neuron {
			offsets_out.push(keys_out.len() as u32);
			counts_out.push(entries.len() as u32);
			for (k, v) in entries {
				keys_out.push(k);
				values_out.push(v);
			}
		}

		(keys_out, values_out, offsets_out, counts_out)
	}

	pub fn write(&self, key: u64, value: u8, allow_override: bool) -> bool {
		loop {
			let needs_resize = {
				let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
				let result = guard.write(key, value, allow_override);
				if result { return result; }
				guard.is_overloaded()
			};
			if needs_resize {
				self.maybe_resize();
				continue;
			}
			return false;
		}
	}

	pub fn nudge(&self, key: u64, target_true: bool) -> bool {
		loop {
			let needs_resize = {
				let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
				let result = guard.nudge(key, target_true);
				if result { return result; }
				guard.is_overloaded()
			};
			if needs_resize {
				self.maybe_resize();
				continue;
			}
			return false;
		}
	}

	/// OI commit pass: scan every MARKER_FINAL slot in the given per-neuron
	/// regions and rewrite its value from a packed `(obs, net)` counter to
	/// the binned 2-bit cell. Called once on the host side after the Metal
	/// kernel completes with `oi_mode=1`. Slots in MARKER_EMPTY/CLAIMED are
	/// skipped — only fully published slots are rewritten.
	///
	/// `slot_offsets` and `slot_capacities` define per-neuron slot regions
	/// (same metadata used by the kernel and by `export_per_neuron`).
	// KEPT-API: OI machinery symmetry with MarkerHashTable's atomic path
	#[allow(dead_code)]
	pub fn commit_oi(&self, slot_offsets: &[u32], slot_capacities: &[u32]) {
		assert_eq!(slot_offsets.len(), slot_capacities.len(),
			"slot_offsets and slot_capacities must have same length");
		let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
		let markers = guard.storage.markers();
		let values = guard.storage.values();
		let markers_len = markers.len();
		let num_neurons = slot_offsets.len();

		// Parallel sweep across neurons — slot regions are disjoint.
		(0..num_neurons).into_par_iter().for_each(|n| {
			let off = slot_offsets[n] as usize;
			let cap = slot_capacities[n] as usize;
			let end = (off + cap).min(markers_len);
			for slot in off..end {
				if markers[slot].load(Ordering::Relaxed) == MARKER_FINAL {
					let packed = values[slot].load(Ordering::Relaxed);
					let cell = ram_core::neuron_memory::oi_bin_to_cell(packed) as u32;
					values[slot].store(cell, Ordering::Relaxed);
				}
			}
		});
	}

	/// OI commit pass over the entire backing buffer (every slot, regardless of
	/// per-neuron region grouping). Useful for batched dispatches where
	/// constructing the per-(genome, neuron) offsets list just for commit
	/// would be wasteful — this iterates the flat slot array once in parallel.
	pub fn commit_oi_all(&self) {
		let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
		let markers = guard.storage.markers();
		let values = guard.storage.values();
		let len = markers.len();
		(0..len).into_par_iter().for_each(|slot| {
			if markers[slot].load(Ordering::Relaxed) == MARKER_FINAL {
				let packed = values[slot].load(Ordering::Relaxed);
				let cell = ram_core::neuron_memory::oi_bin_to_cell(packed) as u32;
				values[slot].store(cell, Ordering::Relaxed);
			}
		});
	}

	pub fn read(&self, key: u64) -> u8 {
		let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
		guard.read(key)
	}

	pub fn len(&self) -> usize {
		let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
		guard.len()
	}

	pub fn snapshot_sorted(&self) -> Vec<(u64, u8)> {
		let guard = self.inner.read().expect("MarkerHashTable RwLock poisoned");
		guard.snapshot_sorted()
	}

	// KEPT-API: map API completeness (mirrors AtomicHashTable::clear)
	#[allow(dead_code)]
	pub fn clear(&self) {
		let mut guard = self.inner.write().expect("MarkerHashTable RwLock poisoned");
		let cap = guard.capacity;
		let dv = self.default_value;
		*guard = MarkerInner::new(cap, dv);
	}

	fn maybe_resize(&self) {
		let mut guard = self.inner.write().expect("MarkerHashTable RwLock poisoned");
		if !guard.is_overloaded() { return; }
		let new_inner = guard.grow_2x();
		*guard = new_inner;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_basic_write_read() {
		let t = AtomicHashTable::new(64, 2);
		assert!(t.write(42, 1, false));
		assert_eq!(t.read(42), 1);
		assert_eq!(t.read(99), 2);
		assert_eq!(t.len(), 1);
	}

	#[test]
	fn test_true_wins_over_false() {
		let t = AtomicHashTable::new(64, 2);
		assert!(t.write(7, 1, false));      // write TRUE
		assert!(!t.write(7, 0, false));     // FALSE rejected (TRUE wins)
		assert_eq!(t.read(7), 1);
	}

	#[test]
	fn test_no_override_blocks_false_on_nonempty() {
		let t = AtomicHashTable::new(64, 2);
		assert!(t.write(7, 1, false));
		// FALSE over TRUE blocked by TRUE-wins
		assert!(!t.write(7, 0, false));
		// allow_override still respects TRUE-wins guard? In existing protocol
		// allow_override triggers the lower clause; the TRUE-wins check still
		// runs first.
		assert!(!t.write(7, 0, true));
		assert_eq!(t.read(7), 1);
	}

	#[test]
	fn test_nudge_clamping() {
		let t = AtomicHashTable::new(64, 1);  // default = QUAD_WEAK_FALSE
		assert!(t.nudge(11, true));
		assert_eq!(t.read(11), 2);            // 1 → 2 (WEAK_TRUE)
		assert!(t.nudge(11, true));
		assert_eq!(t.read(11), 3);            // 2 → 3 (TRUE)
		assert!(!t.nudge(11, true));          // saturated, no change
		assert_eq!(t.read(11), 3);
		assert!(t.nudge(11, false));
		assert_eq!(t.read(11), 2);            // 3 → 2
	}

	#[test]
	fn test_resize_under_load() {
		// Start tiny; force several resizes by inserting many distinct keys.
		let t = AtomicHashTable::new(16, 2);
		for k in 0u64..1000 {
			assert!(t.write(k, 1, false), "write {} failed", k);
		}
		assert_eq!(t.len(), 1000);
		for k in 0u64..1000 {
			assert_eq!(t.read(k), 1, "read {} mismatch", k);
		}
	}

	#[test]
	fn test_parallel_writes_distinct_keys() {
		let t = AtomicHashTable::new(64, 2);
		(0u64..10_000).into_par_iter().for_each(|k| {
			t.write(k, 1, false);
		});
		assert_eq!(t.len(), 10_000);
		// All keys must be readable
		for k in 0u64..10_000 {
			assert_eq!(t.read(k), 1);
		}
	}

	#[test]
	fn test_parallel_nudges_same_keys() {
		let t = AtomicHashTable::new(64, 1);
		// Many threads nudge the same 10 keys toward TRUE — final value should be 3
		let work: Vec<u64> = (0u64..10).cycle().take(10_000).collect();
		work.par_iter().for_each(|&k| {
			t.nudge(k, true);
		});
		for k in 0u64..10 {
			assert_eq!(t.read(k), 3, "key {} should saturate at TRUE", k);
		}
	}

	#[test]
	fn test_snapshot_sorted() {
		let t = AtomicHashTable::new(64, 2);
		for (k, v) in &[(100u64, 1u8), (50, 0), (75, 1), (25, 0)] {
			t.write(*k, *v, true);
		}
		let snap = t.snapshot_sorted();
		assert_eq!(snap, vec![(25, 0), (50, 0), (75, 1), (100, 1)]);
	}

	#[test]
	fn test_estimate_capacity_scales() {
		// Small workload — floor at 3K expected → 16K capacity (next pow2 of 12K)
		assert!(estimate_capacity(100_000) >= 16_384);
		// Mid workload (1M)
		assert!(estimate_capacity(1_000_000) >= 16_384);
		// Large workload (46M) — expected ~20*sqrt(46M)=~136K → 524K capacity
		let cap_46m = estimate_capacity(46_000_000);
		assert!(cap_46m >= 524_288, "got {}", cap_46m);
		// All capacities power of two
		assert_eq!(cap_46m & (cap_46m - 1), 0);
	}

	// ------------------------------------------------------------------
	// MarkerHashTable tests (mirror AtomicHashTable tests + protocol checks)
	// ------------------------------------------------------------------

	#[test]
	fn test_marker_basic_write_read() {
		let t = MarkerHashTable::new(64, 2);
		assert!(t.write(42, 1, false));
		assert_eq!(t.read(42), 1);
		assert_eq!(t.read(99), 2);
		assert_eq!(t.len(), 1);
	}

	#[test]
	fn test_marker_true_wins_over_false() {
		let t = MarkerHashTable::new(64, 2);
		assert!(t.write(7, 1, false));
		assert!(!t.write(7, 0, false));
		assert_eq!(t.read(7), 1);
	}

	#[test]
	fn test_marker_no_override_blocks_false_on_nonempty() {
		let t = MarkerHashTable::new(64, 2);
		assert!(t.write(7, 1, false));
		assert!(!t.write(7, 0, false));
		assert!(!t.write(7, 0, true));
		assert_eq!(t.read(7), 1);
	}

	#[test]
	fn test_marker_nudge_clamping() {
		let t = MarkerHashTable::new(64, 1);
		assert!(t.nudge(11, true));
		assert_eq!(t.read(11), 2);
		assert!(t.nudge(11, true));
		assert_eq!(t.read(11), 3);
		assert!(!t.nudge(11, true));
		assert_eq!(t.read(11), 3);
		assert!(t.nudge(11, false));
		assert_eq!(t.read(11), 2);
	}

	#[test]
	fn test_marker_resize_under_load() {
		let t = MarkerHashTable::new(16, 2);
		for k in 0u64..1000 {
			assert!(t.write(k, 1, false), "write {} failed", k);
		}
		assert_eq!(t.len(), 1000);
		for k in 0u64..1000 {
			assert_eq!(t.read(k), 1, "read {} mismatch", k);
		}
	}

	#[test]
	fn test_marker_parallel_writes_distinct_keys() {
		let t = MarkerHashTable::new(64, 2);
		(0u64..10_000).into_par_iter().for_each(|k| {
			t.write(k, 1, false);
		});
		assert_eq!(t.len(), 10_000);
		for k in 0u64..10_000 {
			assert_eq!(t.read(k), 1);
		}
	}

	#[test]
	fn test_marker_parallel_nudges_same_keys() {
		// Many threads nudge the same 10 keys — final value should saturate
		// at TRUE for every key. This exercises the marker FSM's contention
		// path: many threads hit the same hash slot near-simultaneously.
		let t = MarkerHashTable::new(64, 1);
		let work: Vec<u64> = (0u64..10).cycle().take(10_000).collect();
		work.par_iter().for_each(|&k| {
			t.nudge(k, true);
		});
		for k in 0u64..10 {
			assert_eq!(t.read(k), 3, "key {} should saturate at TRUE", k);
		}
	}

	#[test]
	fn test_marker_snapshot_sorted() {
		let t = MarkerHashTable::new(64, 2);
		for (k, v) in &[(100u64, 1u8), (50, 0), (75, 1), (25, 0)] {
			t.write(*k, *v, true);
		}
		let snap = t.snapshot_sorted();
		assert_eq!(snap, vec![(25, 0), (50, 0), (75, 1), (100, 1)]);
	}

	#[test]
	fn test_marker_clear_resets() {
		let t = MarkerHashTable::new(64, 2);
		for k in 0u64..100 {
			t.write(k, 1, false);
		}
		assert_eq!(t.len(), 100);
		t.clear();
		assert_eq!(t.len(), 0);
		for k in 0u64..100 {
			assert_eq!(t.read(k), 2);  // default_value
		}
	}

	#[test]
	fn test_marker_parity_with_atomic_random_workload() {
		// Run identical sequence on AtomicHashTable and MarkerHashTable;
		// final snapshot_sorted() must agree exactly.
		let a = AtomicHashTable::new(64, 1);
		let m = MarkerHashTable::new(64, 1);
		// Deterministic pseudo-random sequence
		let mut state: u64 = 0xC0FFEE;
		for _ in 0..5000 {
			state ^= state << 13; state ^= state >> 7; state ^= state << 17;
			let key = state & 0xFFFFFFFFFFFF;  // 48-bit address
			let nudge_true = (state >> 32) & 1 == 1;
			a.nudge(key, nudge_true);
			m.nudge(key, nudge_true);
		}
		let snap_a = a.snapshot_sorted();
		let snap_m = m.snapshot_sorted();
		assert_eq!(snap_a, snap_m, "marker and atomic diverged on identical workload");
	}
}
