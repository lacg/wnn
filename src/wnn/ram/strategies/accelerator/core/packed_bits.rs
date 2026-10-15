//! Bit-packed storage for encoded WNN input features.
//!
//! Replaces `Vec<bool>` (1 byte per bit, 8x memory bloat) with a `Vec<u8>`
//! storing 8 bits per byte. Little-endian bit order within each byte to
//! match `np.packbits(bitorder='little')` on the Python side.
//!
//! For 30.8M × 4416-bit canonical-Neto training data this reduces in-process
//! storage from ~165 GB to ~21 GB.
//!
//! ## Bit ordering
//!
//! Bit j of row i lives at byte `i * bytes_per_row + j / 8`, bit `j & 7`
//! (LSB-first within the byte). Example with 5 logical bits encoded:
//!   bit values: [1, 0, 1, 1, 0]
//!   byte: 0b00001101 (=13)
//!
//! ## Why not BitVec or bitvec crate?
//!
//! Both add a dependency and impose their own iteration semantics. For a
//! hot-path bit read inside a neuron-address loop, a hand-rolled struct
//! with inlined `bit()` and `row()` methods gives us the SIMD-friendly
//! pattern the compiler already vectorizes nicely. The 50-line cost of
//! defining it ourselves is paid back in transparency.

use std::ops::Range;

use rayon::prelude::*;

use crate::aligned_words::AlignedWords;

/// Bit-packed feature matrix, row-major, with logical width `total_bits`.
///
/// Storage (13/10/2026): rows live in a page-aligned `AlignedWords` buffer
/// with a row STRIDE of `words_per_row * 8` bytes — each row's logical
/// `bytes_per_row` bytes followed by zero padding up to the next `u64`. The
/// bytes are exactly the old unpadded row bytes, so `bit()` / `packed_row()`
/// read the same values; the padding makes the matrix ALSO the `u64`-word
/// layout every GPU kernel consumes (`as_u64_words`, zero-copy — it used to be
/// a full re-pack per call) and lets Metal wrap it without a copy
/// (`page_span`). Trailing bits in the final logical byte are zero by
/// convention (should not be read).
#[derive(Clone, Debug)]
pub struct PackedBits
{
	/// Row-major words. Length = `num_rows * words_per_row`.
	words: AlignedWords,
	/// Logical width of each row in bits.
	total_bits: usize,
	/// Logical packed bytes per row: `(total_bits + 7) / 8`.
	bytes_per_row: usize,
	/// `u64` words per row (row stride / 8): `(total_bits + 63) / 64`.
	words_per_row: usize,
	/// Number of rows.
	num_rows: usize,
}

impl PackedBits
{
	/// A zero matrix of `num_rows` rows (the common constructor body).
	fn zeroed(num_rows: usize, total_bits: usize) -> Self
	{
		let words_per_row = (total_bits + 63) / 64;
		Self {
			words: AlignedWords::zeroed(num_rows * words_per_row),
			total_bits,
			bytes_per_row: (total_bits + 7) / 8,
			words_per_row,
			num_rows,
		}
	}

	/// Construct from a packed byte buffer (typically `np.packbits` output).
	///
	/// `data.len()` must be a multiple of `(total_bits + 7) / 8`.
	pub fn from_packed_bytes(data: Vec<u8>, total_bits: usize) -> Self
	{
		Self::from_packed_slice(&data, total_bits)
	}

	/// `from_packed_bytes` without taking ownership — callers holding a numpy
	/// view no longer need a `to_vec()` staging copy of the whole matrix.
	pub fn from_packed_slice(data: &[u8], total_bits: usize) -> Self
	{
		let bytes_per_row = (total_bits + 7) / 8;
		assert!(bytes_per_row > 0 || total_bits == 0, "total_bits={} produced bytes_per_row=0", total_bits);
		let num_rows = if bytes_per_row == 0 { 0 } else { data.len() / bytes_per_row };
		debug_assert_eq!(
			data.len(),
			num_rows * bytes_per_row,
			"PackedBits: data.len()={} not divisible by bytes_per_row={}",
			data.len(),
			bytes_per_row
		);
		let mut out = Self::zeroed(num_rows, total_bits);
		out.fill_rows(|i, row| row.copy_from_slice(&data[i * bytes_per_row..(i + 1) * bytes_per_row]));
		out
	}

	/// Construct from a `&[bool]` slice (transitional helper for legacy callers).
	pub fn from_bool_slice(bools: &[bool], total_bits: usize) -> Self
	{
		assert!(
			bools.len() % total_bits == 0,
			"from_bool_slice: bools.len()={} not divisible by total_bits={}",
			bools.len(),
			total_bits
		);
		let num_rows = if total_bits == 0 { 0 } else { bools.len() / total_bits };
		let mut out = Self::zeroed(num_rows, total_bits);
		out.fill_rows(|i, row| pack_row(row, |j| bools[i * total_bits + j], total_bits));
		out
	}

	/// Construct from a flat `&[u8]` slice where each byte is a logical bool
	/// (zero ⇒ false, non-zero ⇒ true). `bytes.len()` = num_rows × total_bits.
	pub fn from_bool_bytes(bytes: &[u8], total_bits: usize) -> Self
	{
		assert!(
			bytes.len() % total_bits.max(1) == 0,
			"from_bool_bytes: bytes.len()={} not divisible by total_bits={}",
			bytes.len(),
			total_bits
		);
		let num_rows = if total_bits == 0 { 0 } else { bytes.len() / total_bits };
		let mut out = Self::zeroed(num_rows, total_bits);
		out.fill_rows(|i, row| pack_row(row, |j| bytes[i * total_bits + j] != 0, total_bits));
		out
	}

	/// Empty PackedBits with the given logical width (used for placeholder subsets).
	pub fn empty(total_bits: usize) -> Self
	{
		Self::zeroed(0, total_bits)
	}

	/// Fill every row's logical bytes in parallel: `f(row_index, row_bytes)`.
	fn fill_rows<F: Fn(usize, &mut [u8]) + Sync>(&mut self, f: F)
	{
		let (stride, bpr) = (self.words_per_row * 8, self.bytes_per_row);
		if stride == 0
		{
			return;
		}
		self.words.as_bytes_mut().par_chunks_mut(stride).enumerate().for_each(|(i, row)| f(i, &mut row[..bpr]));
	}

	/// Read bit `j` of row `i`. Hot path — inlined.
	#[inline(always)]
	pub fn bit(&self, i: usize, j: usize) -> bool
	{
		debug_assert!(i < self.num_rows, "PackedBits::bit row {} >= {}", i, self.num_rows);
		debug_assert!(j < self.total_bits, "PackedBits::bit col {} >= {}", j, self.total_bits);
		let byte = unsafe { *self.words.as_bytes().get_unchecked(i * self.words_per_row * 8 + (j >> 3)) };
		(byte >> (j & 7)) & 1 != 0
	}

	/// Read bit at flat index (i * total_bits + j) — for callers that
	/// previously did `bools[i * total_bits + j]`.
	#[inline(always)]
	pub fn bit_flat(&self, flat_idx: usize) -> bool
	{
		let i = flat_idx / self.total_bits;
		let j = flat_idx % self.total_bits;
		self.bit(i, j)
	}

	/// Return packed bytes for row `i` as `&[u8]` of length `bytes_per_row`
	/// (the logical bytes — the row's word padding is not included).
	#[inline]
	pub fn packed_row(&self, i: usize) -> &[u8]
	{
		debug_assert!(i < self.num_rows);
		let start = i * self.words_per_row * 8;
		&self.words.as_bytes()[start..start + self.bytes_per_row]
	}

	/// Row `i` as its `words_per_row` little-endian `u64` words.
	#[inline]
	pub fn row_words(&self, i: usize) -> &[u64]
	{
		&self.words.as_slice()[i * self.words_per_row..(i + 1) * self.words_per_row]
	}

	/// Decode an entire row into a fresh `Vec<bool>`.
	pub fn row_as_bools(&self, i: usize) -> Vec<bool>
	{
		(0..self.total_bits).map(|j| self.bit(i, j)).collect()
	}

	/// Extend `self` by appending another PackedBits (must have same `total_bits`).
	pub fn extend_from(&mut self, other: &PackedBits)
	{
		assert_eq!(
			self.total_bits, other.total_bits,
			"PackedBits::extend_from total_bits mismatch ({} vs {})",
			self.total_bits, other.total_bits
		);
		self.words.extend_from_slice(other.words.as_slice());
		self.num_rows += other.num_rows;
	}

	/// Append a single row from another PackedBits.
	pub fn extend_row_from(&mut self, src: &PackedBits, src_row: usize)
	{
		assert_eq!(self.total_bits, src.total_bits);
		self.words.extend_from_slice(src.row_words(src_row));
		self.num_rows += 1;
	}

	/// Extract a contiguous row range as a new PackedBits.
	pub fn slice_rows(&self, range: Range<usize>) -> PackedBits
	{
		assert!(range.end <= self.num_rows);
		let start = range.start;
		self.gather_rows(range.end - start, |k| start + k)
	}

	/// Materialize a row-subset selected by indices. Used by stratified
	/// partitioning and undersampling in IDSCache.
	pub fn select_rows(&self, indices: &[usize]) -> PackedBits
	{
		self.gather_rows(indices.len(), |k| indices[k])
	}

	/// Materialize `count` rows where output row `k` is source row `row_of(k)`
	/// — the one gather every row-subset builder goes through. Parallel over
	/// output rows; the result is byte-identical to appending the rows in order.
	pub fn gather_rows<F: Fn(usize) -> usize + Sync>(&self, count: usize, row_of: F) -> PackedBits
	{
		let mut out = Self::zeroed(count, self.total_bits);
		let wpr = self.words_per_row;
		if wpr == 0
		{
			return out;
		}
		out.words.as_mut_slice().par_chunks_mut(wpr).enumerate().for_each(|(k, dst)| {
			let i = row_of(k);
			assert!(i < self.num_rows, "gather_rows: index {} >= {}", i, self.num_rows);
			dst.copy_from_slice(self.row_words(i));
		});
		out
	}

	/// Number of rows.
	#[inline]
	pub fn num_rows(&self) -> usize
	{
		self.num_rows
	}

	/// Logical width per row (bits).
	#[inline]
	pub fn total_bits(&self) -> usize
	{
		self.total_bits
	}

	/// Logical packed bytes per row (`(total_bits + 7) / 8`).
	#[inline]
	pub fn bytes_per_row(&self) -> usize
	{
		self.bytes_per_row
	}

	/// `u64` words per row (`(total_bits + 63) / 64`) — the GPU row stride.
	#[inline]
	pub fn words_per_row(&self) -> usize
	{
		self.words_per_row
	}

	/// Return whether the matrix has zero rows.
	#[inline]
	pub fn is_empty(&self) -> bool
	{
		self.num_rows == 0
	}

	/// The whole matrix as `num_rows * words_per_row` little-endian `u64`
	/// words — exactly what the old `pack_packed_to_u64` re-pack produced, but
	/// borrowed (zero-copy).
	#[inline]
	pub fn as_u64_words(&self) -> &[u64]
	{
		self.words.as_slice()
	}

	/// The page-aligned allocation holding `as_u64_words()` at its start
	/// (whole pages, zero past the words). Wrap THIS in a no-copy GPU buffer.
	#[inline]
	pub fn page_span(&self) -> &[u8]
	{
		self.words.page_span()
	}
}

/// Pack one row's bits (`bit_at(j)` for j < total_bits) into its zeroed bytes.
#[inline]
fn pack_row<F: Fn(usize) -> bool>(row: &mut [u8], bit_at: F, total_bits: usize)
{
	for j in 0..total_bits
	{
		if bit_at(j)
		{
			row[j >> 3] |= 1 << (j & 7);
		}
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	/// Concatenated logical row bytes (the pre-13/10 `as_bytes()` view).
	fn rows_bytes(pb: &PackedBits) -> Vec<u8>
	{
		(0..pb.num_rows()).flat_map(|i| pb.packed_row(i).to_vec()).collect()
	}

	/// The pre-13/10 `pack_packed_to_u64` re-pack, kept verbatim as the oracle
	/// for the zero-copy word view.
	fn legacy_pack_to_u64(pb: &PackedBits) -> Vec<u64>
	{
		let wpe = (pb.total_bits() + 63) / 64;
		let bpr = pb.bytes_per_row();
		let mut out = vec![0u64; pb.num_rows() * wpe];
		for ex in 0..pb.num_rows()
		{
			let row = pb.packed_row(ex);
			for w in 0..wpe
			{
				let mut word = [0u8; 8];
				let take = 8usize.min(bpr.saturating_sub(w * 8));
				if take > 0
				{
					word[..take].copy_from_slice(&row[w * 8..w * 8 + take]);
				}
				out[ex * wpe + w] = u64::from_le_bytes(word);
			}
		}
		out
	}

	fn le_bytes(w: &[u64]) -> Vec<u8>
	{
		w.iter().flat_map(|x| x.to_le_bytes()).collect()
	}

	#[test]
	fn word_view_equals_legacy_repack_for_every_width()
	{
		use rand::{Rng, SeedableRng};
		let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
		for total_bits in [1usize, 5, 8, 9, 63, 64, 65, 127, 128, 200, 1870]
		{
			let bools: Vec<bool> = (0..37 * total_bits).map(|_| rng.gen()).collect();
			let pb = PackedBits::from_bool_slice(&bools, total_bits);
			assert_eq!(pb.as_u64_words(), legacy_pack_to_u64(&pb).as_slice(), "bits={}", total_bits);
			assert_eq!(pb.words_per_row(), (total_bits + 63) / 64);
			let again = PackedBits::from_packed_slice(&rows_bytes(&pb), total_bits);
			assert_eq!(again.as_u64_words(), pb.as_u64_words());
			let span = pb.page_span();
			assert_eq!(span.as_ptr() as usize % crate::aligned_words::PAGE_BYTES, 0);
			assert_eq!(&span[..pb.as_u64_words().len() * 8], le_bytes(pb.as_u64_words()).as_slice());
		}
	}

	#[test]
	fn gather_rows_equals_sequential_append()
	{
		let bools: Vec<bool> = (0..50 * 70).map(|i| (i * 7919) % 3 == 0).collect();
		let pb = PackedBits::from_bool_slice(&bools, 70);
		let idx = [4usize, 49, 0, 17, 17, 3];
		let mut seq = PackedBits::empty(70);
		for &i in &idx
		{
			seq.extend_row_from(&pb, i);
		}
		let gathered = pb.select_rows(&idx);
		assert_eq!(gathered.as_u64_words(), seq.as_u64_words());
		assert_eq!(gathered.num_rows(), 6);
	}

	#[test]
	fn from_packed_bytes_roundtrip()
	{
		// 2 rows × 5 bits; bytes_per_row=1
		// row 0: [1,0,1,1,0] → 0b00001101 = 13
		// row 1: [0,1,1,0,1] → 0b00010110 = 22
		let pb = PackedBits::from_packed_bytes(vec![13, 22], 5);
		assert_eq!(pb.num_rows(), 2);
		assert_eq!(pb.total_bits(), 5);
		assert_eq!(pb.bytes_per_row(), 1);
		assert!(pb.bit(0, 0));
		assert!(!pb.bit(0, 1));
		assert!(pb.bit(0, 2));
		assert!(pb.bit(0, 3));
		assert!(!pb.bit(0, 4));
		assert!(!pb.bit(1, 0));
		assert!(pb.bit(1, 1));
		assert!(pb.bit(1, 2));
		assert!(!pb.bit(1, 3));
		assert!(pb.bit(1, 4));
	}

	#[test]
	fn from_bool_slice_roundtrip()
	{
		let bools = vec![
			true, false, true, true, false, // row 0
			false, true, true, false, true, // row 1
		];
		let pb = PackedBits::from_bool_slice(&bools, 5);
		for i in 0..2
		{
			for j in 0..5
			{
				assert_eq!(pb.bit(i, j), bools[i * 5 + j]);
			}
		}
	}

	#[test]
	fn from_bool_bytes_roundtrip()
	{
		// Numpy-style uint8: each byte is 0 or 1 representing a logical bool.
		let bytes: Vec<u8> = vec![
			1, 0, 1, 1, 0, // row 0
			0, 1, 1, 0, 1, // row 1
		];
		let pb = PackedBits::from_bool_bytes(&bytes, 5);
		assert_eq!(pb.num_rows(), 2);
		assert_eq!(pb.bytes_per_row(), 1);
		for i in 0..2
		{
			for j in 0..5
			{
				assert_eq!(pb.bit(i, j), bytes[i * 5 + j] != 0, "row {} bit {}", i, j);
			}
		}
	}

	#[test]
	fn select_rows()
	{
		let pb = PackedBits::from_packed_bytes(vec![10, 20, 30, 40], 8);
		let sub = pb.select_rows(&[2, 0, 3]);
		assert_eq!(sub.num_rows(), 3);
		assert_eq!(rows_bytes(&sub), vec![30, 10, 40]);
	}

	#[test]
	fn slice_rows_contiguous()
	{
		let pb = PackedBits::from_packed_bytes(vec![10, 20, 30, 40], 8);
		let sub = pb.slice_rows(1..3);
		assert_eq!(sub.num_rows(), 2);
		assert_eq!(rows_bytes(&sub), vec![20, 30]);
	}

	#[test]
	fn extend_from()
	{
		let mut a = PackedBits::from_packed_bytes(vec![1, 2], 8);
		let b = PackedBits::from_packed_bytes(vec![3, 4], 8);
		a.extend_from(&b);
		assert_eq!(a.num_rows(), 4);
		assert_eq!(rows_bytes(&a), vec![1, 2, 3, 4]);
	}

	#[test]
	fn wide_row_bits()
	{
		// 1 row × 20 bits → 3 bytes (with 4 trailing pad bits).
		// bits [0..20]: alternating 1,0,1,0,...
		// bytes: 0b01010101 = 85, 0b01010101 = 85, 0b0001 (lower 4 bits) = 5
		let pb = PackedBits::from_packed_bytes(vec![85, 85, 5], 20);
		assert_eq!(pb.num_rows(), 1);
		assert_eq!(pb.bytes_per_row(), 3);
		for j in 0..20
		{
			let expected = j % 2 == 0;
			assert_eq!(pb.bit(0, j), expected, "bit {} mismatch", j);
		}
	}
}
