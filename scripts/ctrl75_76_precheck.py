"""CTRL-75 (L2) / CTRL-76 (L3) offline kill-checks — VAL diagnosis only, never a result.

The G+T s1 VAL traces (CTRL-73 trace_val.py: 3 val units x 100 eps, never TEST) are pushed
through the IMPLEMENTED ladders (phased_ga's own threshold fit for the flown G+T argv, flags
off vs on). Gates from docs/ctrl73_gap_analysis_gt.md §6:
  L2 (both ladders, dense and dense_hold, side by side with today)
     (a) beyond-outer share <= 25 % in every transient window (0-1.6 s) and <= 5 % in the hold;
     (b) hold occupancy of the inner two pairs not coarser than today's (largest inner-bin
         share no more than 1 pp above today's).
  L3 share of 0-0.3 s ticks beyond the new yaw outer cut <= 10 %, and the hold occupancy of
     the +-0.10 / +-0.25 deg bins unchanged.
yaw is the TRUE yaw from the trace (the input is the Mahony-estimated yaw error: a proxy, as in
CTRL-73 §5). Run: PYTHONPATH=src nice -n 10 python scripts/ctrl75_76_precheck.py [trace.npz]
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from controller_ctrl75_76_ladders import XY_FLAG, XYH_FLAG, YO_FLAG, _blocks, dump  # noqa: E402

W = [(0, 150), (150, 300), (300, 600), (600, 1000), (1000, 1600), (1600, 2000)]


def _yaw_deg(q: np.ndarray) -> np.ndarray:
	w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
	return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _beyond(x: np.ndarray, cuts: list[float]) -> float:
	return 100.0 * float(((x < cuts[0]) | (x > cuts[-1])).mean())


def _occupancy(x: np.ndarray, cuts: list[float]) -> np.ndarray:
	return 100.0 * np.bincount(np.searchsorted(np.asarray(cuts), x), minlength=len(cuts) + 1) / len(x)


def _windows(x: np.ndarray, t: np.ndarray, cuts: list[float]) -> list[float]:
	return [_beyond(x[(t >= a) & (t < b)], cuts) for a, b in W]


def _inner_max(x: np.ndarray, cuts: list[float]) -> float:
	"""Largest hold share among the bins bounded by the inner two pairs (cuts[2..5] of 8)."""
	return float(_occupancy(x, cuts)[3:6].max())


def _entropy(x: np.ndarray, cuts: list[float]) -> float:
	p = _occupancy(x, cuts) / 100.0
	p = p[p > 0]
	return float(-(p * np.log2(p)).sum())


def _l2_gate(x: np.ndarray, t: np.ndarray, c0: list[float], c1: list[float]) -> bool:
	hold, b1 = t >= 1600, _windows(x, t, c1)
	return max(b1[:5]) <= 25.0 and b1[5] <= 5.0 and _inner_max(x[hold], c1) <= _inner_max(x[hold], c0) + 1.0


def _l2_rows(x: np.ndarray, t: np.ndarray, ladders: dict) -> None:
	hold = t >= 1600
	print(f"    {'ladder':10s} | beyond-outer % 0-.15/.15-.3/.3-.6/.6-1/1-1.6/hold | hold H bits | "
	      f"max hold-bin % inside +-0.013 | H bits .15-.3 / .3-.6")
	for lab, c in ladders.items():
		b, H = _windows(x, t, c), [_entropy(x[(t >= a) & (t < bb)], c) for a, bb in W[1:3]]
		print(f"    {lab:10s} | {'/'.join(f'{v:.1f}' for v in b):34s} | {_entropy(x[hold], c):11.2f} | "
		      f"{_inner_max(x[hold], c):29.1f} | {H[0]:.2f} / {H[1]:.2f}")


def check_l2(T: np.ndarray, ladders: dict) -> dict:
	"""ladders: label -> {feature: cuts}, the first being today. Prints the values table; returns per-variant PASS."""
	t, labs, ok = T[:, 2], list(ladders), {}
	for col, name in ((3, "gyro_x"), (4, "gyro_y")):
		print(f"  {name}")
		_l2_rows(T[:, col], t, {lab: ladders[lab][name] for lab in labs})
		for lab in labs[1:]:
			ok[lab] = ok.get(lab, True) and _l2_gate(T[:, col], t, ladders[labs[0]][name], ladders[lab][name])
	return ok


def check_l3(T: np.ndarray, off: dict, on: dict) -> bool:
	t, yw, hold = T[:, 2], np.radians(_yaw_deg(T[:, 6:10])), T[:, 2] >= 1600
	c0, c1 = off["yaw_err"], on["yaw_err"]
	early = t < 300
	b0, b1 = _beyond(yw[early], c0), _beyond(yw[early], c1)
	h0, h1 = _occupancy(yw[hold], c0)[2:7], _occupancy(yw[hold], c1)[2:7]
	ok = b1 <= 10.0 and np.allclose(h0, h1)
	print(f"  yaw_err outer {math.degrees(c0[-1]):.2f} -> {math.degrees(c1[-1]):.2f} deg: 0-0.3 s beyond "
	      f"{b0:.1f} -> {b1:.1f} %; by window {'/'.join(f'{v:.1f}' for v in _windows(yw, t, c0))} -> "
	      f"{'/'.join(f'{v:.1f}' for v in _windows(yw, t, c1))}")
	print(f"  hold occupancy of the +-0.10/+-0.25 deg bins {' '.join(f'{v:.1f}' for v in h0)} -> "
	      f"{' '.join(f'{v:.1f}' for v in h1)}")
	return ok


def main(path: str) -> None:
	T = np.load(path)["trace"].astype(float)
	off, yo = _blocks(dump([])), _blocks(dump(YO_FLAG))
	ladders = {"today": off, "L2-a dense": _blocks(dump(XY_FLAG)), "L2-b hold": _blocks(dump(XYH_FLAG))}
	print(f"VAL trace {path}: {len(T)} ticks, units {sorted(set(T[:, 0].astype(int)))}")
	print("L2 --gyro-xy-ladder dense (L2-a) | dense_hold (L2-b)")
	for lab, ok in check_l2(T, ladders).items():
		print(f"L2 {lab}: {'PASS' if ok else 'FAIL'}")
	print("L3 --yaw-err-outer-cut 4.5")
	print(f"L3 {'PASS' if check_l3(T, off, yo) else 'FAIL'}")


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ctrl73/trace_gt.npz")
