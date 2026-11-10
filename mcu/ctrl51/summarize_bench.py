#!/usr/bin/env python3
"""CTRL-51: turn mcu/run_bench_ctrl51.sh's per-call instruction counts into the cost
table (WNN cascade vs MPCOF cascade teacher). Reads only measured files; the ONLY
assumptions are the stated cycles/instruction and the datasheet power figure.

Region layout (bench_ctrl51.c): line 0 = the empty marker region (overhead, subtracted);
then per tick MPCOF [observe, decide-if-due, shadow], WNN [wnn_tick, observer twin,
Mahony twin]. Each line "insns soft libm vfp".
Usage: python mcu/ctrl51/summarize_bench.py [--bench /tmp/ctrl51/bench] [--ref /tmp/ctrl51/ref]
"""

import argparse
import re
import struct
from pathlib import Path

import numpy as np

CPI = 1.5                     # cycles / instruction: an ESTIMATE (M4 loads ~2, branches 2-3, flash waits)
CLOCKS_MHZ = (16, 48, 168)
# STM32F405xx/407xx datasheet DS8626, "Typical and maximum current consumption in Run
# mode, code with data processing running from Flash memory (ART accelerator enabled)":
# 168 MHz, VDD 3.3 V, typ 25 C: ~40 mA all peripherals disabled, ~87 mA all enabled.
MW_PER_MHZ = 40.0 * 3.3 / 168.0       # 0.786 mW/MHz, core + flash, peripherals off
MW_PER_MHZ_PERIPH = 87.0 * 3.3 / 168.0
WINDOW_TICKS, N_O = 10, 10


def regions(path: Path, per_tick: int) -> np.ndarray:
	"""(ticks, per_tick, 4) counts with the empty-region overhead subtracted from insns."""
	a = np.loadtxt(path, dtype=np.int64).reshape(-1, 4)
	a[1:, 0] -= a[0, 0]
	return a[1:].reshape(-1, per_tick, 4)


def episode_steps(traj: Path, wnn: bool) -> list:
	b = traj.read_bytes()
	n, off, out = struct.unpack_from("<I", b, 0)[0], 4, []
	rec = 4 * 20 if wnn else 4 * 14 + 8 * 15
	for _ in range(n):
		t = struct.unpack_from("<I", b, off)[0]
		out.append(t)
		off += 4 + (24 if wnn else 0) + t * rec
	return out


def decision_mask(steps: list) -> np.ndarray:
	return np.concatenate([np.arange(t) % N_O == 0 for t in steps])


def windows(per_tick: np.ndarray, steps: list) -> np.ndarray:
	"""Cost of every full 10-tick window starting at a decision tick."""
	out, i = [], 0
	for t in steps:
		full = (t // WINDOW_TICKS) * WINDOW_TICKS
		out.append(per_tick[i:i + full].reshape(-1, WINDOW_TICKS).sum(1))
		i += t
	return np.concatenate(out)


def stat(x) -> str:
	x = np.asarray(x)
	return f"{x.mean():9.0f} {np.percentile(x, 99):9.0f} {x.max():9.0f}"


def pct(instr: float, mhz: float, window_s: float) -> float:
	return 100.0 * instr * CPI / (mhz * 1e6 * window_s)


def nj(instr: float, k=MW_PER_MHZ) -> float:
	return instr * CPI * k   # E[nJ] = cycles * (mW/MHz): linear-in-f model, f cancels


def row(name: str, win: np.ndarray, dec_tick: np.ndarray):
	m, w = win.mean(), win.max()
	clocks = " ".join(f"{pct(m, f, 0.01):6.2f}/{pct(w, f, 0.01):6.2f}" for f in CLOCKS_MHZ)
	d1 = " ".join(f"{pct(dec_tick.max(), f, 0.001):6.1f}" for f in CLOCKS_MHZ)
	print(f"  {name:34s} win {m:9.0f} / {w:9.0f}  %10ms(mean/worst)@16/48/168: {clocks}"
	      f"  | worst decision tick %1ms: {d1} | {nj(m) / 1e3:6.1f} uJ/window ({nj(m, MW_PER_MHZ_PERIPH) / 1e3:.1f} periph-on)")


def mpcof(bench: Path, ref: Path, prec: str):
	print(f"\n=== MPCOF ({prec}) — per call: mean / p99 / max instructions")
	for regime in ("nominal", "stress", "upset"):
		r = regions(bench / f"mpcof_{prec}_{regime}.regions", 3)
		steps = episode_steps(ref / f"traj_{regime}.bin", False)
		dec = decision_mask(steps)
		obs, decide, shadow = r[:, 0, 0], r[:, 1, 0], r[:, 2, 0]
		tick = obs + decide + shadow
		qp = np.loadtxt(bench / f"dec_{regime}.txt", dtype=int).reshape(-1, 3)[:, 0] > 0
		dd = decide[dec]
		print(f" [{regime}] {len(tick)} ticks, {dec.sum()} decisions, QP ran on {qp.sum()}")
		print(f"   observer/tick        {stat(obs)}   shadow/tick {stat(shadow)}")
		print(f"   decision (plan)      {stat(dd)}   no-QP {stat(dd[~qp])}" + (f"   QP {stat(dd[qp])}" if qp.any() else ""))
		print(f"   hold-tick total      {stat(tick[~dec])}   decision-tick total {stat(tick[dec])}")
		print(f"   per decision: soft-double calls {stat(r[dec, 1, 1])}  libm {stat(r[dec, 1, 2])}  VFP {stat(r[dec, 1, 3])}")
		print(f"   observer soft-double calls/tick {r[:, 0, 1].mean():.0f}; shadow {r[:, 2, 1].mean():.0f}")
		row("full (obs+plan+shadow)", windows(tick, steps), tick[dec])
		core = obs + decide
		row("core (shadow = firmware loop)", windows(core, steps), core[dec])


def wnn(bench: Path, ref: Path, name: str):
	r = regions(bench / f"{name}.regions", 3)
	steps = episode_steps(ref / "wnn_a49_flights.bin", True)
	dec = decision_mask(steps)
	tick, obs, est = r[:, 0, 0], r[:, 1, 0], r[:, 2, 0]
	print(f"\n=== WNN {name} — {len(tick)} ticks, {dec.sum()} decisions; mean / p99 / max")
	print(f"   wnn_tick hold        {stat(tick[~dec])}   decision {stat(tick[dec])}")
	print(f"   of which: d-hat observer (twin) {stat(obs)}   Mahony f64 (twin) {stat(est)}")
	lookup = tick[dec].mean() - tick[~dec].mean()
	print(f"   lookup+encode+decode (decision - hold, means) {lookup:.0f}; other features "
	      f"{tick[~dec].mean() - obs.mean() - est.mean():.0f}")
	print(f"   soft-double calls: hold {r[~dec, 0, 1].mean():.0f} decision {r[dec, 0, 1].mean():.0f}; "
	      f"VFP hold {r[~dec, 0, 3].mean():.0f} decision {r[dec, 0, 3].mean():.0f}")
	row("full wnn_tick", windows(tick, steps), tick[dec])
	no_est = tick - est
	row("without its Mahony (q given)", windows(no_est, steps), no_est[dec])


SECTION = re.compile(r"^ (\.(?:text|rodata|data|bss)\S*)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)")
SECTION2 = re.compile(r"^\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)")


def footprint(mapfile: Path) -> dict:
	"""Bytes by (kind, owner) from the GNU ld map: kind flash (.text/.rodata) or ram
	(.data/.bss); owner = object basename or archive name."""
	acc, pending = {}, None
	for line in mapfile.read_text(errors="replace").splitlines():
		if line.startswith("Discarded input sections"):
			continue
		m = SECTION.match(line)
		if m:
			add(acc, m.group(1), int(m.group(3), 16), m.group(4))
			pending = None
			continue
		if re.match(r"^ \.(text|rodata|data|bss)\S*$", line):
			pending = line.strip()
			continue
		m2 = SECTION2.match(line)
		if pending and m2:
			add(acc, pending, int(m2.group(2), 16), m2.group(3))
		pending = None
	return acc


def add(acc: dict, sec: str, size: int, src: str):
	if size == 0 or "/" not in src and "(" not in src:
		return
	kind = "flash" if sec.startswith((".text", ".rodata")) else "ram"
	owner = re.sub(r"\(.*", "", src.split("/")[-1])
	acc[(kind, owner)] = acc.get((kind, owner), 0) + size


def print_footprint(bench: Path):
	"""Every object's flash/RAM. MPCOF = mpcof_cascade + cascade_dhat + wnn_ratefw (its
	shadow). WNN = the bench TU (it #includes wnn_cascade.c: model tables + code, plus a
	~1.5 KB driver) + cascade_dhat; its wnn_ratefw is the FIRMWARE rate loop, not the WNN.
	The controller state lives in the driver's .bss (STATE_BYTES in the .log)."""
	print("\n=== footprint (GNU ld map; bytes per object/archive: flash = .text+.rodata, ram = .data+.bss)")
	for name in ("mpcof_f64_nominal", "mpcof_f32_nominal", "wnn_a49_f64obs", "wnn_a49_f32obs"):
		acc = footprint(bench / f"{name}.map")
		items = ", ".join(f"{o}:{k}={v}" for (k, o), v in sorted(acc.items(), key=lambda x: (x[0][1], x[0][0])))
		print(f"  {name:20s} {items}")


def main():
	ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
	ap.add_argument("--bench", type=Path, default=Path("/tmp/ctrl51/bench"))
	ap.add_argument("--ref", type=Path, default=Path("/tmp/ctrl51/ref"))
	a = ap.parse_args()
	print(f"CPI {CPI} (estimate); energy k = {MW_PER_MHZ:.3f} mW/MHz (DS8626 typ, periph off), "
	      f"{MW_PER_MHZ_PERIPH:.3f} periph on")
	for prec in ("f64", "f32"):
		mpcof(a.bench, a.ref, prec)
	for name in ("wnn_a49_f64obs", "wnn_a49_f32obs"):
		wnn(a.bench, a.ref, name)
	print_footprint(a.bench)


if __name__ == "__main__":
	main()
