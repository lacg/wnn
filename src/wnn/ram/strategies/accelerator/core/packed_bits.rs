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

/// Bit-packed feature matrix, row-major, with logical width `total_bits`.
///
/// Storage: `data` has length `num_rows * bytes_per_row` where
/// `bytes_per_row = (total_bits + 7) / 8`. Trailing bits in the final byte
/// of each row are unused (zero-padded by convention; should not be read).
#[derive(Clone, Debug)]
pub struct PackedBits
{
	/// Packed bytes. Length = `num_rows * bytes_per_row`.
	data: Vec<u8>,
	/// Logical width of each row in bits.
	total_bits: usize,
	/// Bytes per row stride (precomputed).
	bytes_per_row: usize,
	/// Number of rows.
	num_rows: usize,
}

impl PackedBits
{
	/// Construct from a packed byte buffer (typically `np.packbits` output).
	///
	/// `data.len()` must be a multiple of `(total_bits + 7) / 8`.
	pub fn from_packed_bytes(data: Vec<u8>, total_bits: usize) -> Self
	{
		let bytes_per_row = (total_bits + 7) / 8;
		assert!(
			bytes_per_row > 0 || total_bits == 0,
			"total_bits={} produced bytes_per_row=0",
			total_bits
		);
		let num_rows = if bytes_per_row == 0
		{
			0
		}
		else
		{
			data.len() / bytes_per_row
		};
		debug_assert_eq!(
			data.len(),
			num_rows * bytes_per_row,
			"PackedBits: data.len()={} not divisible by bytes_per_row={}",
			data.len(),
			bytes_per_row
		);
		Self {
			data,
			total_bits,
			bytes_per_row,
			num_rows,
		}
	}

	/// Construct from a `&[bool]` slice (transitional helper for legacy callers).
	///
	/// Allocates a new `Vec<u8>` and packs the bits. For new code prefer
	/// `from_packed_bytes` to avoid the bool→packed conversion at the
	/// Python boundary.
	pub fn from_bool_slice(bools: &[bool], total_bits: usize) -> Self
	{
		assert!(
			bools.len() % total_bits == 0,
			"from_bool_slice: bools.len()={} not divisible by total_bits={}",
			bools.len(),
			total_bits
		);
		let num_rows = if total_bits == 0
		{
			0
		}
		else
		{
			bools.len() / total_bits
		};
		let bytes_per_row = (total_bits + 7) / 8;
		let mut data = vec![0u8; num_rows * bytes_per_row];
		for i in 0..num_rows
		{
			let row_byte_offset = i * bytes_per_row;
			let row_bool_offset = i * total_bits;
			for j in 0..total_bits
			{
				if bools[row_bool_offset + j]
				{
					data[row_byte_offset + (j >> 3)] |= 1 << (j & 7);
				}
			}
		}
		Self {
			data,
			total_bits,
			bytes_per_row,
			num_rows,
		}
	}

	/// Construct from a flat `&[u8]` slice where each byte is a logical bool
	/// (zero ⇒ false, non-zero ⇒ true). Avoids the bool-vec intermediate when
	/// data comes from a numpy uint8 buffer via PyO3.
	///
	/// `bytes.len()` must be a multiple of `total_bits` (= num_rows × total_bits).
	pub fn from_bool_bytes(bytes: &[u8], total_bits: usize) -> Self
	{
		assert!(
			bytes.len() % total_bits.max(1) == 0,
			"from_bool_bytes: bytes.len()={} not divisible by total_bits={}",
			bytes.len(),
			total_bits
		);
		let num_rows = if total_bits == 0
		{
			0
		}
		else
		{
			bytes.len() / total_bits
		};
		let bytes_per_row = (total_bits + 7) / 8;
		let mut data = vec![0u8; num_rows * bytes_per_row];
		for i in 0..num_rows
		{
			let row_byte_offset = i * bytes_per_row;
			let row_input_offset = i * total_bits;
			for j in 0..total_bits
			{
				if unsafe { *bytes.get_unchecked(row_input_offset + j) } != 0
				{
					data[row_byte_offset + (j >> 3)] |= 1 << (j & 7);
				}
			}
		}
		Self {
			data,
			total_bits,
			bytes_per_row,
			num_rows,
		}
	}

	/// Empty PackedBits with the given logical width (used for placeholder subsets).
	pub fn empty(total_bits: usize) -> Self
	{
		let bytes_per_row = (total_bits + 7) / 8;
		Self {
			data: Vec::new(),
			total_bits,
			bytes_per_row,
			num_rows: 0,
		}
	}

	/// Read bit `j` of row `i`. Hot path — inlined.
	#[inline(always)]
	pub fn bit(&self, i: usize, j: usize) -> bool
	{
		debug_assert!(
			i < self.num_rows,
			"PackedBits::bit row {} >= {}",
			i,
			self.num_rows
		);
		debug_assert!(
			j < self.total_bits,
			"PackedBits::bit col {} >= {}",
			j,
			self.total_bits
		);
		let byte = unsafe { *self.data.get_unchecked(i * self.bytes_per_row + (j >> 3)) };
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

	/// Return packed bytes for row `i` as `&[u8]` of length `bytes_per_row`.
	/// Useful for downstream code that wants to consume a row's worth of bits.
	#[inline]
	pub fn packed_row(&self, i: usize) -> &[u8]
	{
		debug_assert!(i < self.num_rows);
		let start = i * self.bytes_per_row;
		&self.data[start..start + self.bytes_per_row]
	}

	/// Decode an entire row into a fresh `Vec<bool>`. Allocates `total_bits`
	/// bytes — use only when the caller really needs a bool slice (e.g. legacy
	/// inner loops not yet migrated). Prefer `bit()` for single-bit reads.
	pub fn row_as_bools(&self, i: usize) -> Vec<bool>
	{
		let mut out = Vec::with_capacity(self.total_bits);
		for j in 0..self.total_bits
		{
			out.push(self.bit(i, j));
		}
		out
	}

	/// Extend `self` by appending another PackedBits (must have same `total_bits`).
	pub fn extend_from(&mut self, other: &PackedBits)
	{
		assert_eq!(
			self.total_bits, other.total_bits,
			"PackedBits::extend_from total_bits mismatch ({} vs {})",
			self.total_bits, other.total_bits
		);
		self.data.extend_from_slice(&other.data);
		self.num_rows += other.num_rows;
	}

	/// Append a single row from another PackedBits. Identical to `extend_from`
	/// for the case of a single-row source.
	pub fn extend_row_from(&mut self, src: &PackedBits, src_row: usize)
	{
		assert_eq!(self.total_bits, src.total_bits);
		let start = src_row * src.bytes_per_row;
		self
			.data
			.extend_from_slice(&src.data[start..start + src.bytes_per_row]);
		self.num_rows += 1;
	}

	/// Extract a contiguous row range as a new PackedBits.
	pub fn slice_rows(&self, range: Range<usize>) -> PackedBits
	{
		assert!(range.end <= self.num_rows);
		let byte_start = range.start * self.bytes_per_row;
		let byte_end = range.end * self.bytes_per_row;
		PackedBits {
			data: self.data[byte_start..byte_end].to_vec(),
			total_bits: self.total_bits,
			bytes_per_row: self.bytes_per_row,
			num_rows: range.end - range.start,
		}
	}

	/// Materialize a row-subset selected by indices. Used by stratified
	/// partitioning and undersampling in IDSCache.
	pub fn select_rows(&self, indices: &[usize]) -> PackedBits
	{
		let mut data = Vec::with_capacity(indices.len() * self.bytes_per_row);
		for &i in indices
		{
			debug_assert!(
				i < self.num_rows,
				"select_rows: index {} >= {}",
				i,
				self.num_rows
			);
			let start = i * self.bytes_per_row;
			data.extend_from_slice(&self.data[start..start + self.bytes_per_row]);
		}
		PackedBits {
			data,
			total_bits: self.total_bits,
			bytes_per_row: self.bytes_per_row,
			num_rows: indices.len(),
		}
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

	/// Packed bytes per row (precomputed stride).
	#[inline]
	pub fn bytes_per_row(&self) -> usize
	{
		self.bytes_per_row
	}

	/// Raw byte buffer length.
	#[inline]
	pub fn data_len(&self) -> usize
	{
		self.data.len()
	}

	/// Return whether the matrix has zero rows.
	#[inline]
	pub fn is_empty(&self) -> bool
	{
		self.num_rows == 0
	}

	/// Borrow the underlying byte buffer (for callers that want zero-copy
	/// access to the packed representation, e.g. GPU staging).
	#[inline]
	pub fn as_bytes(&self) -> &[u8]
	{
		&self.data
	}

	/// Mutable byte access for in-place construction (used by tests).
	#[cfg(test)]
	pub fn data_mut(&mut self) -> &mut Vec<u8>
	{
		&mut self.data
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

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
		assert_eq!(sub.as_bytes(), &[30, 10, 40]);
	}

	#[test]
	fn slice_rows_contiguous()
	{
		let pb = PackedBits::from_packed_bytes(vec![10, 20, 30, 40], 8);
		let sub = pb.slice_rows(1..3);
		assert_eq!(sub.num_rows(), 2);
		assert_eq!(sub.as_bytes(), &[20, 30]);
	}

	#[test]
	fn extend_from()
	{
		let mut a = PackedBits::from_packed_bytes(vec![1, 2], 8);
		let b = PackedBits::from_packed_bytes(vec![3, 4], 8);
		a.extend_from(&b);
		assert_eq!(a.num_rows(), 4);
		assert_eq!(a.as_bytes(), &[1, 2, 3, 4]);
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
