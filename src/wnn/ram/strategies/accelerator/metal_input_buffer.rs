//! Zero-copy Metal view of a `PackedBits` input matrix (13/10/2026 memory fix).
//!
//! Apple Silicon has unified memory: the GPU can read the host allocation
//! directly. `new_buffer_with_data` nevertheless COPIED the whole training
//! matrix into a fresh Metal buffer on every marker-train dispatch (6.9 GB
//! per chunk at 46M rows, on top of the u64 re-pack it was made from).
//! `PackedBits` now lives in a page-aligned, page-multiple allocation already
//! in the kernels' u64 word layout, so this wraps it with
//! `newBufferWithBytesNoCopy` instead.
//!
//! Lifetime: the wrapper borrows the matrix (`PhantomData<&'a PackedBits>`),
//! so the borrow checker forbids the Metal buffer outliving — or the matrix
//! being mutated/reallocated under — it. Every dispatch that reads it is
//! committed and `wait_until_completed` before the wrapper drops (the marker
//! trainer and probe are synchronous). No deallocator is registered: Metal
//! never frees the memory, `PackedBits` does.

use std::marker::PhantomData;

use metal::{Buffer, DeviceRef, MTLResourceOptions};
use ram_core::aligned_words::PAGE_BYTES;
use ram_core::packed_bits::PackedBits;

/// A shared-storage Metal buffer aliasing a `PackedBits`' words (no copy).
pub struct NoCopyInputBuffer<'a>
{
	buffer: Buffer,
	_input: PhantomData<&'a PackedBits>,
}

impl<'a> NoCopyInputBuffer<'a>
{
	/// Wrap `input`'s page span. Panics (never silently copies) if the span
	/// is not page-aligned or Metal did not alias it. An empty matrix gets a
	/// one-word placeholder (there is nothing to copy).
	pub fn new(device: &DeviceRef, input: &'a PackedBits) -> Self
	{
		let span = input.page_span();
		let buffer = if span.is_empty()
		{
			device.new_buffer(8, MTLResourceOptions::StorageModeShared)
		}
		else
		{
			Self::wrap(device, span)
		};
		Self {
			buffer,
			_input: PhantomData,
		}
	}

	/// `newBufferWithBytesNoCopy` over `span`, asserting the alias.
	fn wrap(device: &DeviceRef, span: &[u8]) -> Buffer
	{
		assert_eq!(span.as_ptr() as usize % PAGE_BYTES, 0, "no-copy input: span not page-aligned");
		assert_eq!(span.len() % PAGE_BYTES, 0, "no-copy input: span not a page multiple");
		let buffer = device.new_buffer_with_bytes_no_copy(
			span.as_ptr() as *const std::ffi::c_void,
			span.len() as u64,
			MTLResourceOptions::StorageModeShared,
			None,
		);
		assert_eq!(
			buffer.contents() as usize,
			span.as_ptr() as usize,
			"no-copy input: Metal did not alias the host matrix"
		);
		buffer
	}

	/// The Metal buffer (valid only while this wrapper lives).
	pub fn buffer(&self) -> &Buffer
	{
		&self.buffer
	}
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn aliases_the_packed_words()
	{
		let Some(device) = metal::Device::system_default() else { return };
		let bools: Vec<bool> = (0..300 * 130).map(|i| i % 7 == 0).collect();
		let pb = PackedBits::from_bool_slice(&bools, 130);
		let wrapped = NoCopyInputBuffer::new(&device, &pb);
		assert_eq!(wrapped.buffer().contents() as *const u64, pb.as_u64_words().as_ptr());
		let words = unsafe { std::slice::from_raw_parts(wrapped.buffer().contents() as *const u64, pb.as_u64_words().len()) };
		assert_eq!(words, pb.as_u64_words());
	}
}
