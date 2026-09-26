//! ram_core::counter_rng bridge for the WORKER wheel (26/09/2026, WNN-1).
//!
//! The controller wheel has exported these since the counter_rng landed; the
//! worker wheel did not, so the shared GA template (OptimizationTemplate, used by
//! IDS AND the controller) had no Rust source for its per-generation seeds on the
//! IDS side — which is how the IDS Rust offspring generator ended up seeded from
//! the wall clock. Both wheels now expose the same three functions over the same
//! ram_core implementation; the template derives every generation's seeds from
//! `counter_rng_draw_u64(run_seed, generation, stage, stream, 0, 0)`, so a resumed
//! generation replays its draws exactly with no RNG state in the checkpoint.
//!
//! Still NOT a per-cell Python draw API: operators belong in Rust. The template
//! makes a handful of draws per generation (one seed per stream).

use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn counter_rng_draw_u64(
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
	index: u64,
	sub: u64,
) -> u64
{
	ram_core::counter_rng::draw_u64(seed, generation, genome, layer, index, sub)
}

#[pyfunction]
pub(crate) fn counter_rng_uniform(
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
	index: u64,
	sub: u64,
) -> f64
{
	ram_core::counter_rng::uniform(seed, generation, genome, layer, index, sub)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn counter_rng_below(
	n: u64,
	seed: u64,
	generation: u64,
	genome: u64,
	layer: u64,
	index: u64,
	sub: u64,
) -> u64
{
	ram_core::counter_rng::below(n, seed, generation, genome, layer, index, sub)
}
