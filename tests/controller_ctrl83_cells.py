"""CTRL-83/89 (ABI 70): every joint stress cell and its score_cell.py gate read ONE source.

For each joint cell, the stressed config's scorer kwargs (EpisodeConfig.stress_kwargs, what both
the WNN and the rival scorers receive) must carry EXACTLY the inertia / latency levels
cells.jitter_levels(cell) declares (the dict score_cell._gate_jitter checks), and the per-episode
lag range cells.CELL_LAG_RANGE[cell]. Pins the per-cell policy (Luiz 28/11): the golden `all`
carries inertia 0.20 + latency (8, 12) ms; `allz90` its Zhang-derived levels; `all_v1` / `all_a69`
none. Needs the ABI-70 controller wheel on PYTHONPATH.

Run: PYTHONPATH=<abi70 wheel dir>:src python tests/controller_ctrl83_cells.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "logs", "controller", "ctrl83"))
import cells as C  # noqa: E402

WANT_JITTER = {
	"all": {"inertia_jitter": 0.20, "control_latency_ms": (8.0, 12.0)},
	"allz90": {"inertia_jitter": 0.44, "control_latency_ms": (6.4, 14.4)},
	"all_v1": {},
	"all_a69": {},
}


def _nominal_ec():
	from wnn.control.training import EpisodeConfig, DisturbanceConfig
	return EpisodeConfig(translation=True, disturbance=DisturbanceConfig.preset("L4C", seed=911), motor_lag_s=0.20)


def _joint_cells():
	return [c[0] for c in C.CELLS if c[1] == "joint"]


def test_policy_pinned():
	assert sorted(_joint_cells()) == sorted(WANT_JITTER), _joint_cells()
	assert sorted(C.CELL_LEVELS) == sorted(WANT_JITTER)
	for cell, want in WANT_JITTER.items():
		assert C.jitter_levels(cell) == want, (cell, C.jitter_levels(cell))
	print(f"✓ per-cell inertia/latency policy: {WANT_JITTER}")


def test_cell_matches_gate():
	ec0 = _nominal_ec()
	for cell in _joint_cells():
		kw = C.apply(cell, ec0).stress_kwargs()
		want = C.jitter_levels(cell)
		for k in ("inertia_jitter", "control_latency_ms"):
			got = kw.get(k)
			if k in want:
				assert got is not None and np.allclose(got, want[k]), (cell, k, got, want[k])
			else:
				assert got is None or not np.any(np.asarray(got, dtype=float) != 0.0), (cell, k, got)
		assert np.allclose(kw["train_motor_lag_range"], C.CELL_LAG_RANGE[cell]), (cell, kw["train_motor_lag_range"])
		print(f"✓ {cell:8s} stress_kwargs == gate: jitter {want or 'none'}, lag {C.CELL_LAG_RANGE[cell]}")


def test_nominal_carries_none():
	kw = C.apply("nom", _nominal_ec()).stress_kwargs()
	assert "inertia_jitter" not in kw and "control_latency_ms" not in kw, kw
	print("✓ nom: no inertia / latency kwargs (byte-identical call)")


if __name__ == "__main__":
	test_policy_pinned()
	test_cell_matches_gate()
	test_nominal_carries_none()
