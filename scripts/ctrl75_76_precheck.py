"""CTRL-75 (L2) / CTRL-76 (L3) offline kill-checks — VAL diagnosis only, never a result.

The G+T s1 VAL traces (CTRL-73 trace_val.py: 3 val units x 100 eps, never TEST) are pushed
through the IMPLEMENTED ladders (phased_ga's own threshold fit for the flown G+T argv, flags
off vs on). Gates from docs/ctrl73_gap_analysis_gt.md §6:
  L2 (a) beyond-outer share <= 25 % in every transient window (0-1.6 s) and <= 5 % in the hold;
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
from controller_ctrl75_76_ladders import XY_FLAG, YO_FLAG, _blocks, dump  # noqa: E402

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


def check_l2(T: np.ndarray, off: dict, on: dict) -> bool:
	t, ok, hold = T[:, 2], True, T[:, 2] >= 1600
	for col, name in ((3, "gyro_x"), (4, "gyro_y")):
		x, c0, c1 = T[:, col], off[name], on[name]
		b0, b1 = _windows(x, t, c0), _windows(x, t, c1)
		a = max(b1[:5]) <= 25.0 and b1[5] <= 5.0
		i0, i1 = _inner_max(x[hold], c0), _inner_max(x[hold], c1)
		ok &= a and i1 <= i0 + 1.0
		print(f"  {name}: beyond-outer % by window {'/'.join(f'{v:.1f}' for v in b0)} -> "
		      f"{'/'.join(f'{v:.1f}' for v in b1)}  (a) {'PASS' if a else 'FAIL'}")
		print(f"    hold occupancy today {' '.join(f'{v:.1f}' for v in _occupancy(x[hold], c0))}")
		print(f"    hold occupancy new   {' '.join(f'{v:.1f}' for v in _occupancy(x[hold], c1))}"
		      f"  (b) largest inner-bin share {i0:.1f} -> {i1:.1f} % {'PASS' if i1 <= i0 + 1.0 else 'FAIL'}")
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
	off, xy, yo = _blocks(dump([])), _blocks(dump(XY_FLAG)), _blocks(dump(YO_FLAG))
	print(f"VAL trace {path}: {len(T)} ticks, units {sorted(set(T[:, 0].astype(int)))}")
	print("L2 --gyro-xy-ladder dense")
	print(f"L2 {'PASS' if check_l2(T, off, xy) else 'FAIL'}")
	print("L3 --yaw-err-outer-cut 4.5")
	print(f"L3 {'PASS' if check_l3(T, off, yo) else 'FAIL'}")


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ctrl73/trace_gt.npz")
