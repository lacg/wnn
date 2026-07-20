//! Counter-based RNG shared by every substrate and both languages.
//!
//! WHY THIS EXISTS
//! ---------------
//! Genome operators (mutation, crossover, tournament) draw randoms per cell and
//! per connection — order 10^9 draws in a production controller run. Python does
//! them today on numpy's PCG64, and PCG64 is not reproduced in Rust, so porting
//! an operator changes results. The codebase's two existing parity strategies
//! both fail at this volume:
//!
//!   * INJECTION (pre-draw in numpy, pass the array over FFI) — used for episode
//!     initial conditions in dagger_train. Fine for 12 episodes; absurd for 10^9
//!     draws, which is ~8 GB of f64 crossing the boundary to avoid computing it.
//!   * SEQUENTIAL STREAM — inherently order-dependent, so results depend on the
//!     order draws are consumed. Under rayon that order is not guaranteed, which
//!     is precisely why the RNG keeps forcing operators to stay single-threaded
//!     in Python.
//!
//! A COUNTER-BASED generator solves both. Every draw is a pure function of its
//! COORDINATES — (seed, generation, genome, layer, index, sub-draw) — so:
//!   * identical in Rust and Python (integer ops only, no stream state),
//!   * ORDER-INDEPENDENT: any thread computes cell i's draw without knowing what
//!     any other cell did, so rayon may schedule freely,
//!   * reproducible across a different thread count or chunk size,
//!   * one RNG contract for IDS and the controller instead of two.
//!
//! It follows the house counter-hash style (`dist_hash_u32` in the controller,
//! `qsr_coin` in neuron_memory) but 64-bit, using the SplitMix64 finaliser so the
//! Python mirror is a handful of masked integer ops.
//!
//! NOT bit-compatible with numpy PCG64 — by construction. Adopting it is a
//! ONE-TIME, VERSIONED lineage break, not a silent drift.

/// Golden-ratio odd constant (SplitMix64's increment).
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// SplitMix64 finaliser: avalanches a counter into a well-distributed u64.
#[inline]
pub fn splitmix64(x: u64) -> u64 {
	let mut z = x.wrapping_add(GAMMA);
	z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
	z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
	z ^ (z >> 31)
}

/// Fold draw COORDINATES into one u64 key, then avalanche.
///
/// The multipliers are distinct large odd constants so different coordinates
/// cannot alias by construction (the same trick as `dist_hash_u32`). Argument
/// order is part of the contract — the Python mirror must fold identically.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn draw_u64(seed: u64, generation: u64, genome: u64, layer: u64, index: u64, sub: u64) -> u64 {
	let key = seed
		.wrapping_add(generation.wrapping_mul(0x9E37_79B1))
		.wrapping_add(genome.wrapping_mul(0x85EB_CA6B))
		.wrapping_add(layer.wrapping_mul(0xC2B2_AE35))
		.wrapping_add(index.wrapping_mul(0x27D4_EB2F))
		.wrapping_add(sub.wrapping_mul(0x1656_67B1));
	splitmix64(key)
}

/// Uniform f64 in [0, 1).
///
/// Uses the top 53 bits — the same construction numpy uses for `random()`, so
/// the DISTRIBUTION and precision match exactly even though the stream does not.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn uniform(seed: u64, generation: u64, genome: u64, layer: u64, index: u64, sub: u64) -> f64 {
	(draw_u64(seed, generation, genome, layer, index, sub) >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// Unbiased integer in [0, n), via Lemire's multiply-shift with rejection.
///
/// Plain `x % n` is biased whenever n does not divide 2^64. Lemire takes the
/// high half of a 128-bit product and rejects only the short interval that would
/// skew it; rejection re-draws by advancing `sub`, which keeps the whole thing a
/// pure function of the coordinates (no hidden state). Returns 0 for n == 0.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn below(n: u64, seed: u64, generation: u64, genome: u64, layer: u64, index: u64, sub: u64) -> u64 {
	if n == 0 {
		return 0;
	}
	let threshold = n.wrapping_neg() % n;
	let mut k = sub;
	loop {
		let x = draw_u64(seed, generation, genome, layer, index, k);
		let m = (x as u128).wrapping_mul(n as u128);
		if (m as u64) >= threshold {
			return (m >> 64) as u64;
		}
		k = k.wrapping_add(0x1_0000_0000); // disjoint from normal sub-draw stepping
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Distinct coordinates must not collide — the property the multipliers buy.
	#[test]
	fn coordinates_do_not_alias() {
		let mut seen = std::collections::HashSet::new();
		for g in 0..8u64 {
			for gen in 0..8u64 {
				for idx in 0..64u64 {
					assert!(seen.insert(draw_u64(42, gen, g, 0, idx, 0)),
						"collision at gen={gen} genome={g} idx={idx}");
				}
			}
		}
	}

	/// Same coordinates ⇒ same draw, regardless of when or on which thread.
	#[test]
	fn draws_are_pure() {
		let a = draw_u64(7, 3, 11, 1, 9999, 2);
		let b = draw_u64(7, 3, 11, 1, 9999, 2);
		assert_eq!(a, b);
	}

	#[test]
	fn uniform_is_in_range_and_spread() {
		let mut lo = 0usize;
		let n = 20_000;
		for i in 0..n {
			let u = uniform(1, 0, 0, 0, i as u64, 0);
			assert!((0.0..1.0).contains(&u), "u={u} out of range");
			if u < 0.5 { lo += 1; }
		}
		let frac = lo as f64 / n as f64;
		assert!((0.45..0.55).contains(&frac), "half-split {frac} is skewed");
	}

	/// `below` must cover the range and stay unbiased enough to detect a modulo
	/// skew. n is deliberately NOT a power of two.
	#[test]
	fn below_is_unbiased_over_a_non_power_of_two() {
		let n = 7u64;
		let mut counts = [0usize; 7];
		let trials = 70_000;
		for i in 0..trials {
			counts[below(n, 5, 0, 0, 0, i as u64, 0) as usize] += 1;
		}
		let expect = trials as f64 / n as f64;
		for (v, &c) in counts.iter().enumerate() {
			let dev = (c as f64 - expect).abs() / expect;
			assert!(dev < 0.05, "value {v} deviates {:.3} from uniform", dev);
		}
	}

	#[test]
	fn below_zero_is_zero() {
		assert_eq!(below(0, 1, 2, 3, 4, 5, 6), 0);
	}
}
