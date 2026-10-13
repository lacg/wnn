//! `TrainKnobs` — the GPU trainer's tuning knobs as a per-call VALUE.
//!
//! The batched marker trainer used to read `WNN_EXAMPLE_CHUNKS` and
//! `WNN_EXPORT_SKIP_WF` from the process environment deep inside the dispatch
//! and the export. Tests drove them with `std::env::set_var`, which is process-
//! global: under parallel `cargo test` one test's z sweep / wF switch leaked
//! into every other test training at that moment (12/10/2026: plain-vs-plain
//! exports differed, oi_z_parity_with_collisions failed ~3/5). Production
//! still reads the env — once per call, at the entry (`from_env`) — and tests
//! pass explicit knobs instead of mutating shared state.

/// Per-call knobs for `train_batch_to_table` and the export after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrainKnobs
{
	/// Threads per (genome, neuron) along the example axis; `None` = the
	/// occupancy heuristic. Env: `WNN_EXAMPLE_CHUNKS` (> 0).
	pub example_chunks: Option<u32>,
	/// OI export drops cells that read as the miss default (WEAK_FALSE) —
	/// eval-identical, halves the export. Env: `WNN_EXPORT_SKIP_WF=0` disables.
	pub export_skip_wf: bool,
}

impl TrainKnobs
{
	/// The production knobs: the env, read now.
	pub(crate) fn from_env() -> Self
	{
		let example_chunks = std::env::var("WNN_EXAMPLE_CHUNKS")
			.ok()
			.and_then(|s| s.parse::<u32>().ok())
			.filter(|&v| v > 0);
		let export_skip_wf = std::env::var("WNN_EXPORT_SKIP_WF").ok().as_deref() != Some("0");
		Self { example_chunks, export_skip_wf }
	}

	/// Explicit knobs (tests, benches): z threads per neuron, wF filter.
	#[cfg(test)]
	pub(crate) fn with(example_chunks: Option<u32>, export_skip_wf: bool) -> Self
	{
		Self { example_chunks, export_skip_wf }
	}
}
