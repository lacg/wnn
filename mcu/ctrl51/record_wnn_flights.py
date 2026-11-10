#!/usr/bin/env python3
"""CTRL-51: record the WNN cascade winner's closed-loop input stream for the M4 bench.

The Rust WnnController (installed ram_controller wheel) flies the run's own plant —
mcu/parity/test_cascade_parity.py's recorder, unchanged — and the per-tick inputs
plus the commands it produced are written for bench_ctrl51.c (which replays them
into the C twin and checks every motor command bit for bit on the Cortex-M4).

File: u32 n_eps; per episode u32 steps, f32 q0[4], f32 yaw0, f32 anchor, then per
tick f32 gyro[3] accel[3] target[3] vert[3] horiz[4] pwm[4].

Usage: PYTHONPATH=src/wnn python mcu/ctrl51/record_wnn_flights.py --winner <ckpt> --out f.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mcu/parity"))
import test_cascade_parity as tcp  # noqa: E402

FIELDS = ("gyro", "accel", "target", "vert", "horiz")


def write(path: Path, flights: list):
	with open(path, "wb") as f:
		f.write(struct.pack("<I", len(flights)))
		for fl in flights:
			f.write(struct.pack("<I", len(fl.pwm)))
			f.write(np.asarray(fl.q0, "<f4").tobytes())
			f.write(np.asarray([fl.yaw0, fl.anchor], "<f4").tobytes())
			rows = np.concatenate([fl.ins[k] for k in FIELDS] + [fl.pwm], axis=1).astype("<f4")
			f.write(rows.tobytes())


def main():
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--winner", type=Path, required=True)
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--episodes", type=int, default=5)
	ap.add_argument("--steps", type=int, default=2000)
	ap.add_argument("--seed", type=int, default=51)
	a = ap.parse_args()
	model, genome, argv = tcp.load_winner(a.winner)
	ec = tcp.run_ec(argv)
	flights = tcp.record_flights(genome, ec, model.motor_lag_s, a.episodes, a.steps, a.seed)
	write(a.out, flights)
	print(f"wrote {a.out}: {len(flights)} episodes, {sum(len(f.pwm) for f in flights)} ticks")


if __name__ == "__main__":
	main()
