//! ABI 35: one bptt window's MEMORY-INDEPENDENT inputs, computed once on the CPU.
//!
//! `bptt_train_window` used to interleave two kinds of work in its forward roll:
//! the feature pipeline (thresholded frames, the K-frame sensor window, accumulators,
//! observers, replayed observations) and the recurrent STATE reads. Only the second
//! depends on memory. Splitting them lets the CPU walk and the GPU walk consume the
//! SAME prepared window, which is what makes the two bit-comparable: the GPU never
//! re-derives a feature, it only re-does what depends on cells.
//!
//! Everything here is state-independent by construction: compute_features reads the
//! replayed sensors/observations and the controller's own accumulators (pwm, d-hat,
//! integrals), none of which the walk touches; the teacher target bits read the
//! recorded pwms. `reset` replays `reset_state` (prev_state ← 0 before the window).
pub(crate) struct BpttWindowPlan
{
	/// The window started from a reset (prev_state zeroed before its first record).
	pub reset: bool,
	/// Decision records in the window (action-repeat holds produce none).
	pub n_rec: usize,
	/// Physical step of each record (L4 reads |err| at it; sn=0 only).
	pub rec_step: Vec<usize>,
	/// [n_rec * sensor_window] — the K-frame window, padded exactly as the ring was.
	pub sensor: Vec<bool>,
	/// [n_rec * frame_bits] — the current frame of each record.
	pub frame: Vec<bool>,
	/// [n_rec * num_out] — the teacher target bit of every output neuron.
	pub out_target: Vec<bool>,
	/// Option A (WNN_STATE_INTEGRAL_TARGET=1): [n_rec * state_neurons] direct state
	/// target cell, or NO_STATE_OVERRIDE where the walk's d_s decides.
	pub state_target: Option<Vec<u8>>,
}

/// Marker in `state_target`: no Option-A override for this (record, neuron).
pub(crate) const NO_STATE_OVERRIDE: u8 = 255;

/// L4 (magnitude-priority writes) parameters of one window — sn=0 only; the walk
/// ignores them when a state layer exists (records are order-coupled there).
pub(crate) struct BpttL4<'a>
{
	pub att_errs: Option<&'a [f32]>,
	/// Physical steps in the window (att_errs must be this long to count).
	pub window_len: usize,
	pub write_priority_err: bool,
	pub write_err_floor_deg: f32,
}

impl BpttL4<'_>
{
	/// The legacy walk: no L4 ordering.
	pub(crate) fn off() -> BpttL4<'static>
	{
		BpttL4 { att_errs: None, window_len: 0, write_priority_err: false, write_err_floor_deg: 0.0 }
	}
}

/// Whether the GPU sn>0 trainer is requested for this process:
/// WNN_CONTROLLER_GPU_BPTT "1" ⇒ GPU, "0" ⇒ CPU, unset ⇒ crate::BPTT_DEVICE_DEFAULT
/// (phased_ga --bptt-device always sets it). Read per training call.
pub(crate) fn gpu_bptt_requested() -> bool
{
	match std::env::var("WNN_CONTROLLER_GPU_BPTT").as_deref()
	{
		Ok("1") => true,
		Ok("0") => false,
		_ => crate::BPTT_DEVICE_DEFAULT == "gpu",
	}
}
