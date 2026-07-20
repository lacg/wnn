"""ram_core::counter_rng and wnn.ram.counter_rng must agree BIT-FOR-BIT.

This is the contract that lets genome operators move from Python to Rust without
changing what a draw means. If it ever fails, a ported operator is silently
drawing different numbers than its Python twin.

Run: PYTHONPATH=src python tests/test_counter_rng_parity.py
"""

from __future__ import annotations

import sys

from wnn.control import _accel as ra
from wnn.ram import counter_rng as py

# Coordinates chosen to exercise: zero, small, large, and values that overflow a
# u64 fold (so the Python mask and the Rust wrapping_* must agree).
COORDS = [
	(0, 0, 0, 0, 0, 0),
	(1, 0, 0, 0, 0, 0),
	(42, 7, 13, 1, 999_999, 3),
	(2**63, 2**20, 2**20, 3, 2**32, 7),
	(2**64 - 1, 2**32 - 1, 2**32 - 1, 2**16, 2**40, 2**8),
	(12345, 250, 49, 1, 599_999, 1),
]


def check_draw_u64() -> int:
	bad = 0
	for c in COORDS:
		r = ra.counter_rng_draw_u64(*c)
		p = py.draw_u64(*c)
		if r != p:
			print(f"  ✗ draw_u64{c}: rust={r} python={p}")
			bad += 1
	# bulk sweep
	for i in range(20_000):
		c = (7, i % 251, (i * 31) % 97, i % 4, i, i % 5)
		if ra.counter_rng_draw_u64(*c) != py.draw_u64(*c):
			print(f"  ✗ draw_u64 sweep mismatch at {c}")
			bad += 1
			break
	print(f"  draw_u64: {'OK' if bad == 0 else 'MISMATCH'} "
	      f"({len(COORDS)} edge cases + 20,000 sweep)")
	return bad


def check_uniform() -> int:
	bad = 0
	for i in range(20_000):
		c = (11, i % 97, i % 53, i % 3, i, 0)
		r = ra.counter_rng_uniform(*c)
		p = py.uniform(*c)
		if r != p:   # exact equality: same integer -> same float construction
			print(f"  ✗ uniform{c}: rust={r!r} python={p!r}")
			bad += 1
			break
		if not (0.0 <= r < 1.0):
			print(f"  ✗ uniform{c} out of range: {r}")
			bad += 1
			break
	print(f"  uniform : {'OK' if bad == 0 else 'MISMATCH'} (20,000 exact-float compares)")
	return bad


def check_below() -> int:
	bad = 0
	# include non-powers-of-two and 1 and a huge n (rejection path)
	for n in (1, 2, 3, 7, 10, 64, 1000, 2**32 + 1, 2**63 + 12345):
		for i in range(400):
			c = (5, i % 17, i % 11, 1, i, 0)
			r = ra.counter_rng_below(n, *c)
			p = py.below(n, *c)
			if r != p:
				print(f"  ✗ below(n={n}){c}: rust={r} python={p}")
				bad += 1
				break
			if not (0 <= r < n):
				print(f"  ✗ below(n={n}) out of range: {r}")
				bad += 1
				break
		if bad:
			break
	print(f"  below   : {'OK' if bad == 0 else 'MISMATCH'} (9 moduli x 400 draws, incl. rejection path)")
	return bad


def check_order_independence() -> int:
	"""The property that makes rayon safe: a draw depends ONLY on coordinates,
	so computing them in any order (or on any thread) yields the same values."""
	coords = [(3, 1, g, 0, i, 0) for g in range(20) for i in range(200)]
	forward = [ra.counter_rng_draw_u64(*c) for c in coords]
	backward = [ra.counter_rng_draw_u64(*c) for c in reversed(coords)][::-1]
	ok = forward == backward
	print(f"  order-independent: {'OK' if ok else 'FAILED'} "
	      f"({len(coords)} draws, forward vs reverse evaluation)")
	return 0 if ok else 1


def main() -> int:
	print(f"counter_rng parity — ram_controller ABI {ra.ABI_VERSION}")
	bad = check_draw_u64() + check_uniform() + check_below() + check_order_independence()
	print("\n" + ("✓ COUNTER_RNG PARITY: Rust and Python agree bit-for-bit"
	              if bad == 0 else f"✗ {bad} PARITY FAILURE(S)"))
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
