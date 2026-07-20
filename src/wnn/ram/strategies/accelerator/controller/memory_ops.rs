//! GA-Memory cell-value operators (paradigm B), in Rust.
//!
//! These replace the per-cell Python loops in control/ga_memory.py::mutate /
//! crossover and control/recurrent_genome.py::_mutate_memory / crossover_memory.
//! At production scale (~500k cells/genome x 50 genomes x 250 generations) those
//! loops run order 10^9 interpreter iterations, each with a numpy `rng.random()`
//! call — the single largest Rust-first violation in the controller stack.
//!
//! RNG: ram_core::counter_rng, NOT a sequential stream. Every draw is a pure
//! function of (seed, generation, genome, layer, index, sub), so:
//!   * the gate and the direction draw are independent coordinates (sub 0 and 1)
//!     rather than consecutive stream positions — the Python version's draw COUNT
//!     was data-dependent (the direction draw happened only when the gate passed),
//!     which made the stream position depend on the data. That coupling is gone.
//!   * cells can be mutated in any order, or in parallel, and still reproduce.
//!
//! This changes RESULTS relative to the numpy PCG64 path — deliberately, and
//! only where the caller opts in. It is the versioned lineage break.

use ram_core::counter_rng;

/// Layer tags, so the state and output layers of one genome never share draw
/// coordinates.
pub const LAYER_STATE: u64 = 0;
pub const LAYER_OUTPUT: u64 = 1;

/// Nudge ~`rate` of the cell values one step.
///
/// QUAD (QUAD_WEIGHTED / QUAD_BINARY / QSR): ±1, clamped to 0..3.
/// Otherwise (TERNARY / BINARY / PLN): 2-state flip `1 - (v & 1)`. The low-bit
/// mask FIRST is load-bearing — an EMPTY cell (EMPTY_U8 = 2, the untrained
/// baseline carried in the universe) must flip to a definite TRUE(1), not to
/// 1-2 = -1, which would overflow the u8 write path. Mirrors the Python fix at
/// ga_memory.py:186 exactly.
pub fn mutate_values(
	values: &mut [u8], quad: bool, rate: f64,
	seed: u64, generation: u64, genome: u64, layer: u64,
) {
	for (i, v) in values.iter_mut().enumerate() {
		let idx = i as u64;
		if counter_rng::uniform(seed, generation, genome, layer, idx, 0) >= rate {
			continue;
		}
		*v = if quad {
			let up = counter_rng::uniform(seed, generation, genome, layer, idx, 1) < 0.5;
			let d: i16 = if up { 1 } else { -1 };
			(*v as i16 + d).clamp(0, 3) as u8
		} else {
			1 - (*v & 1)
		};
	}
}

/// Uniform per-cell crossover over two index-aligned value vectors.
///
/// Index alignment holds because a MEMORY phase freezes the architecture, so the
/// whole population shares one address universe (see MemoryPayload's docstring).
/// Returns a fresh vector; `a` and `b` must be the same length.
pub fn crossover_values(
	a: &[u8], b: &[u8],
	seed: u64, generation: u64, genome: u64, layer: u64,
) -> Vec<u8> {
	a.iter()
		.zip(b.iter())
		.enumerate()
		.map(|(i, (&x, &y))| {
			if counter_rng::uniform(seed, generation, genome, layer, i as u64, 0) < 0.5 { x } else { y }
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn quad_stays_in_range_and_moves_by_one() {
		let mut v = vec![0u8, 1, 2, 3, 2, 1];
		let before = v.clone();
		mutate_values(&mut v, true, 1.0, 42, 0, 0, LAYER_STATE);
		for (i, (&b, &a)) in before.iter().zip(v.iter()).enumerate() {
			assert!(a <= 3, "cell {i} out of QUAD range: {a}");
			let d = a as i16 - b as i16;
			// clamped at the ends, so 0 is legal there
			assert!(d.abs() <= 1, "cell {i} moved by {d}, expected +-1");
		}
	}

	#[test]
	fn two_state_flip_never_underflows_on_empty() {
		// EMPTY_U8 = 2 must become 1, NOT -1 (the u8 overflow the Python fix guards)
		let mut v = vec![2u8, 2, 2, 2];
		mutate_values(&mut v, false, 1.0, 7, 0, 0, LAYER_OUTPUT);
		assert!(v.iter().all(|&x| x == 1), "EMPTY must flip to TRUE(1), got {v:?}");
	}

	#[test]
	fn two_state_flip_is_involutive() {
		let mut v = vec![0u8, 1, 0, 1];
		mutate_values(&mut v, false, 1.0, 9, 0, 0, LAYER_STATE);
		assert_eq!(v, vec![1u8, 0, 1, 0]);
	}

	#[test]
	fn rate_zero_is_a_no_op_and_rate_one_touches_all() {
		let mut v = vec![0u8; 500];
		mutate_values(&mut v, true, 0.0, 3, 0, 0, LAYER_STATE);
		assert!(v.iter().all(|&x| x == 0), "rate=0 must not mutate");

		let mut w = vec![1u8; 500];
		mutate_values(&mut w, true, 1.0, 3, 0, 0, LAYER_STATE);
		assert!(w.iter().any(|&x| x != 1), "rate=1 must mutate");
	}

	/// The property the counter RNG buys: mutating in a different ORDER (or on a
	/// different thread) yields the same result. A sequential stream would not.
	#[test]
	fn mutation_is_order_independent() {
		let base: Vec<u8> = (0..1000).map(|i| (i % 4) as u8).collect();
		let mut fwd = base.clone();
		mutate_values(&mut fwd, true, 0.3, 11, 5, 2, LAYER_STATE);

		// same draws, applied back-to-front
		let mut rev = base.clone();
		for i in (0..rev.len()).rev() {
			let mut one = [rev[i]];
			// reproduce the per-cell call with the SAME coordinates
			let idx = i as u64;
			if counter_rng::uniform(11, 5, 2, LAYER_STATE, idx, 0) < 0.3 {
				let up = counter_rng::uniform(11, 5, 2, LAYER_STATE, idx, 1) < 0.5;
				let d: i16 = if up { 1 } else { -1 };
				one[0] = (one[0] as i16 + d).clamp(0, 3) as u8;
			}
			rev[i] = one[0];
		}
		assert_eq!(fwd, rev, "order changed the result — RNG is not counter-based");
	}

	#[test]
	fn crossover_takes_from_both_parents() {
		let a = vec![0u8; 2000];
		let b = vec![3u8; 2000];
		let c = crossover_values(&a, &b, 5, 0, 0, LAYER_OUTPUT);
		assert!(c.iter().any(|&x| x == 0) && c.iter().any(|&x| x == 3),
			"crossover must draw from both parents");
		assert!(c.iter().all(|&x| x == 0 || x == 3), "crossover invented a value");
	}
}
