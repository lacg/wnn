"""Python mirror of ram_core::counter_rng — MUST agree bit-for-bit.

This exists for two reasons only:
  1. to VERIFY the Rust implementation from Python (tests/test_counter_rng_parity.py),
  2. as a transition shim while an operator still runs in Python.

It is deliberately NOT the place to add new draws. The Rust-first rule stands:
operators belong in ram_core, and this mirror is the proof that moving them there
does not change what a draw means.

Why counter-based at all (the short version — see core/counter_rng.rs for the
full rationale): a sequential stream like numpy's PCG64 makes results depend on
the ORDER draws are consumed, which blocks parallel operators and cannot be
reproduced in Rust. Here every draw is a pure function of its coordinates
(seed, generation, genome, layer, index, sub), so it is identical in both
languages, order-independent, and safe under any rayon schedule.

NOT bit-compatible with numpy PCG64, by construction.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1

GAMMA = 0x9E3779B97F4A7C15
MIX_A = 0xBF58476D1CE4E5B9
MIX_B = 0x94D049BB133111EB

_TWO53 = float(1 << 53)


def splitmix64(x: int) -> int:
	"""SplitMix64 finaliser — mirrors core/counter_rng.rs::splitmix64."""
	z = (x + GAMMA) & MASK64
	z = ((z ^ (z >> 30)) * MIX_A) & MASK64
	z = ((z ^ (z >> 27)) * MIX_B) & MASK64
	return z ^ (z >> 31)


def draw_u64(seed: int, generation: int, genome: int, layer: int, index: int, sub: int) -> int:
	"""Fold draw coordinates into a key, then avalanche. Argument order is part
	of the contract — it must match the Rust fold exactly."""
	key = (
		seed
		+ generation * 0x9E3779B1
		+ genome * 0x85EBCA6B
		+ layer * 0xC2B2AE35
		+ index * 0x27D4EB2F
		+ sub * 0x165667B1
	) & MASK64
	return splitmix64(key)


def uniform(seed: int, generation: int, genome: int, layer: int, index: int, sub: int) -> float:
	"""Uniform in [0, 1) from the top 53 bits (numpy's own construction)."""
	return (draw_u64(seed, generation, genome, layer, index, sub) >> 11) / _TWO53


def below(n: int, seed: int, generation: int, genome: int, layer: int, index: int, sub: int) -> int:
	"""Unbiased integer in [0, n) — Lemire multiply-shift with rejection.

	`x % n` is biased unless n divides 2^64; Lemire rejects only the short
	skewing interval. Rejection advances `sub` by a disjoint stride so the result
	stays a pure function of the coordinates."""
	if n <= 0:
		return 0
	# Mirror Rust's `n.wrapping_neg() % n`, i.e. (2^64 - n) % n on UNSIGNED types.
	# Python's `(-n) % n` is NOT the same thing: `%` here is non-negative and -n
	# divides evenly, so it is always 0 — the mirror would never reject, and
	# diverged from Rust exactly on the draws Lemire is meant to discard.
	threshold = ((1 << 64) - n) % n
	k = sub
	while True:
		x = draw_u64(seed, generation, genome, layer, index, k)
		m = x * n
		if (m & MASK64) >= threshold:
			return m >> 64
		k = (k + 0x100000000) & MASK64
