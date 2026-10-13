//! Page-aligned, zero-padded `u64` storage — the backing buffer of `PackedBits`.
//!
//! Why its own allocation instead of `Vec<u8>` (13/10/2026 memory fixes):
//!   * the GPU wants input rows as `u64` words. A `Vec<u8>` is only 1-byte
//!     aligned, so every train/eval call copied the whole matrix into a fresh
//!     `Vec<u64>` (`pack_packed_to_u64`) — 6.9 GB per k-fold call at 46M rows,
//!     plus another 6.9 GB per marker-train chunk.
//!   * Metal on unified memory can wrap host memory without a copy
//!     (`newBufferWithBytesNoCopy`), but only a PAGE-ALIGNED pointer whose
//!     length is a whole number of pages. A `Vec` gives neither.
//!
//! Invariants (every constructor and mutator keeps them):
//!   * the allocation is `PAGE_BYTES`-aligned and `cap_bytes` is a multiple of
//!     `PAGE_BYTES` (or zero for the empty buffer);
//!   * every byte in `[len_words * 8, cap_bytes)` is ZERO, so the page span
//!     handed to Metal is fully initialised and deterministic.

use std::alloc::{alloc_zeroed, dealloc, realloc, Layout};
use std::ptr::NonNull;

/// Alignment AND length granule of the allocation. 16 KiB is the Apple-Silicon
/// VM page; it is a multiple of every smaller page size (4 KiB on x86), so the
/// same constant satisfies Metal's no-copy requirement on every host.
pub const PAGE_BYTES: usize = 16 * 1024;

/// Growable `u64` buffer with the page-alignment invariants above.
pub struct AlignedWords
{
	ptr: NonNull<u64>,
	len_words: usize,
	cap_bytes: usize,
}

// SAFETY: AlignedWords uniquely owns its allocation (like Vec<u64>); shared
// references only hand out `&[u64]` / `&[u8]`, mutation needs `&mut self`.
unsafe impl Send for AlignedWords {}
unsafe impl Sync for AlignedWords {}

impl AlignedWords
{
	/// The empty buffer (no allocation).
	pub fn new() -> Self
	{
		Self {
			ptr: NonNull::dangling(),
			len_words: 0,
			cap_bytes: 0,
		}
	}

	/// `len_words` zero words.
	pub fn zeroed(len_words: usize) -> Self
	{
		let mut out = Self::new();
		out.reserve_words(len_words);
		out.len_words = len_words;
		out
	}

	/// Number of words in use.
	#[inline]
	pub fn len(&self) -> usize
	{
		self.len_words
	}

	/// True when no words are in use.
	#[inline]
	pub fn is_empty(&self) -> bool
	{
		self.len_words == 0
	}

	/// The words in use.
	#[inline]
	pub fn as_slice(&self) -> &[u64]
	{
		// SAFETY: ptr is valid for cap_bytes >= len_words * 8 initialised bytes
		// (or dangling with len 0, which from_raw_parts permits).
		unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len_words) }
	}

	/// The words in use, mutably.
	#[inline]
	pub fn as_mut_slice(&mut self) -> &mut [u64]
	{
		// SAFETY: as as_slice, plus &mut self guarantees uniqueness.
		unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len_words) }
	}

	/// The words in use as little-endian bytes (`len() * 8` bytes).
	#[inline]
	pub fn as_bytes(&self) -> &[u8]
	{
		// SAFETY: u8 has no alignment requirement; same region as as_slice.
		unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len_words * 8) }
	}

	/// The words in use as mutable bytes.
	#[inline]
	pub fn as_bytes_mut(&mut self) -> &mut [u8]
	{
		// SAFETY: as as_bytes, plus &mut self guarantees uniqueness.
		unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len_words * 8) }
	}

	/// The WHOLE allocation: page-aligned start, a whole number of pages,
	/// zero past `len() * 8`. Empty when nothing is allocated. This is the
	/// region a no-copy GPU buffer may wrap.
	#[inline]
	pub fn page_span(&self) -> &[u8]
	{
		// SAFETY: every byte of the allocation is initialised (alloc_zeroed /
		// tail zeroing in grow_to), and &self freezes it.
		unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.cap_bytes) }
	}

	/// Append `words` (amortised growth, doubling like `Vec`).
	pub fn extend_from_slice(&mut self, words: &[u64])
	{
		let start = self.len_words;
		self.reserve_words(start + words.len());
		self.len_words = start + words.len();
		self.as_mut_slice()[start..].copy_from_slice(words);
	}

	/// Append `n` zero words; returns the new (zeroed) tail for in-place fill.
	pub fn extend_zeroed(&mut self, n: usize) -> &mut [u64]
	{
		let start = self.len_words;
		self.reserve_words(start + n);
		self.len_words = start + n;
		&mut self.as_mut_slice()[start..]
	}

	/// Ensure capacity for `total_words` without changing `len()`.
	pub fn reserve_words(&mut self, total_words: usize)
	{
		let need = total_words.checked_mul(8).expect("AlignedWords: size overflow");
		if need <= self.cap_bytes
		{
			return;
		}
		let doubled = self.cap_bytes.saturating_mul(2);
		self.grow_to(round_up_to_page(need.max(doubled)));
	}

	/// Reallocate to exactly `new_cap` bytes (a page multiple > cap_bytes),
	/// zeroing the newly exposed tail.
	fn grow_to(&mut self, new_cap: usize)
	{
		let new_layout = page_layout(new_cap);
		// SAFETY: layouts are non-zero-sized with PAGE_BYTES alignment; realloc
		// is given the exact layout the block was allocated with.
		let raw = unsafe {
			if self.cap_bytes == 0
			{
				alloc_zeroed(new_layout)
			}
			else
			{
				let grown = realloc(self.ptr.as_ptr() as *mut u8, page_layout(self.cap_bytes), new_cap);
				if !grown.is_null()
				{
					std::ptr::write_bytes(grown.add(self.cap_bytes), 0, new_cap - self.cap_bytes);
				}
				grown
			}
		};
		self.ptr = NonNull::new(raw as *mut u64).unwrap_or_else(|| std::alloc::handle_alloc_error(new_layout));
		self.cap_bytes = new_cap;
	}
}

impl Default for AlignedWords
{
	fn default() -> Self
	{
		Self::new()
	}
}

impl Drop for AlignedWords
{
	fn drop(&mut self)
	{
		if self.cap_bytes > 0
		{
			// SAFETY: allocated by alloc_zeroed/realloc with this exact layout.
			unsafe { dealloc(self.ptr.as_ptr() as *mut u8, page_layout(self.cap_bytes)) };
		}
	}
}

impl Clone for AlignedWords
{
	fn clone(&self) -> Self
	{
		let mut out = Self::zeroed(self.len_words);
		out.as_mut_slice().copy_from_slice(self.as_slice());
		out
	}
}

impl std::fmt::Debug for AlignedWords
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
	{
		write!(f, "AlignedWords {{ len_words: {}, cap_bytes: {} }}", self.len_words, self.cap_bytes)
	}
}

/// Smallest page multiple >= `bytes` (and >= one page).
fn round_up_to_page(bytes: usize) -> usize
{
	bytes.max(1).div_ceil(PAGE_BYTES) * PAGE_BYTES
}

/// The allocation layout for a `cap_bytes`-byte block.
fn page_layout(cap_bytes: usize) -> Layout
{
	Layout::from_size_align(cap_bytes, PAGE_BYTES).expect("AlignedWords: bad layout")
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn page_span_is_aligned_page_multiple_and_zero_past_len()
	{
		let mut w = AlignedWords::new();
		assert!(w.page_span().is_empty());
		w.extend_from_slice(&[1, 2, 3]);
		let span = w.page_span();
		assert_eq!(span.as_ptr() as usize % PAGE_BYTES, 0);
		assert_eq!(span.len() % PAGE_BYTES, 0);
		assert!(span[24..].iter().all(|&b| b == 0));
		assert_eq!(w.as_slice(), &[1, 2, 3]);
	}

	#[test]
	fn growth_keeps_contents_alignment_and_zero_tail()
	{
		let mut w = AlignedWords::new();
		for i in 0..5000u64
		{
			w.extend_from_slice(&[i, !i]);
		}
		assert_eq!(w.len(), 10000);
		assert_eq!(w.as_slice()[9998], 4999);
		assert_eq!(w.as_slice()[9999], !4999);
		let span = w.page_span();
		assert_eq!(span.as_ptr() as usize % PAGE_BYTES, 0);
		assert!(span[80000..].iter().all(|&b| b == 0));
		let c = w.clone();
		assert_eq!(c.as_slice(), w.as_slice());
	}
}
