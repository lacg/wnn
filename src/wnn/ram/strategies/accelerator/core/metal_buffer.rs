//! Host slice -> shared-storage Metal buffer, safe for an EMPTY slice (IDS-81).
//!
//! Metal returns nil for a zero-length buffer (metal-rs then aborts on the null
//! pointer), and an empty slice's pointer is dangling, so copying even one byte
//! from it (the old `size_of_val(data).max(1)` workaround) reads unmapped
//! memory. An untrained genome's sparse export is exactly such a slice (no
//! keys, no values). Every kernel bounds its reads by a count that is 0 for an
//! empty input, so the placeholder buffer is never read.

use metal::{Buffer, DeviceRef, MTLResourceOptions};

/// Size of the placeholder buffer bound for an empty slice (covers the widest
/// element any kernel binds, a u64/ulong).
pub const EMPTY_PLACEHOLDER_BYTES: u64 = 16;

/// A shared-storage buffer holding a copy of `data`; an empty slice gets a
/// fresh `EMPTY_PLACEHOLDER_BYTES` buffer instead of a copy.
pub fn shared_buffer_of<T>(device: &DeviceRef, data: &[T]) -> Buffer
{
	if data.is_empty()
	{
		return device.new_buffer(EMPTY_PLACEHOLDER_BYTES, MTLResourceOptions::StorageModeShared);
	}
	device.new_buffer_with_data(
		data.as_ptr() as *const std::ffi::c_void,
		std::mem::size_of_val(data) as u64,
		MTLResourceOptions::StorageModeShared,
	)
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn empty_and_nonempty_slices_both_yield_a_buffer()
	{
		let Some(device) = metal::Device::system_default() else { return };
		let empty: [u64; 0] = [];
		assert_eq!(shared_buffer_of(&device, &empty).length(), EMPTY_PLACEHOLDER_BYTES);
		let words = [7u32, 9, 11];
		let b = shared_buffer_of(&device, &words);
		assert_eq!(b.length(), 12);
		let got = unsafe { std::slice::from_raw_parts(b.contents() as *const u32, 3) };
		assert_eq!(got, &words);
	}
}
