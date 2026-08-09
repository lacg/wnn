#!/usr/bin/env python3
"""Minimise each per-neuron PLA with ABC and report the deployment footprint.

WHAT THIS MEASURES. A RAM-WNN controller's memory is an ON-set (all populated
cells are TRUE in BINARY mode), i.e. one Boolean function per output neuron. Its
real hardware cost is that function AFTER logic minimisation, not the number of
addresses you would have to store in a naive key/value table. This runs ABC on
each neuron's PLA and reports:

  cubes    product terms in the minimised SOP  (the espresso-style number)
  luts     LUT6 count after technology mapping (the FPGA number; per the project
           rule, FPGA size is LUTs -- never n x 2^bits)
  levels   logic depth = the combinational delay, and on an FPGA the whole
           controller resolves in ONE clock because all neurons are parallel

The naive-sparse column is printed alongside purely to show what the minimisation
buys; it is not a representation anyone would deploy.

Usage:
  python scripts/minimize_controller_pla.py --pla-dir /tmp/pla [--jobs 2]
"""

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# `ps` prints e.g.
#   neuron000 : i/o = 30/ 1  lat = 0  nd = 412  edge = 1980  cube = 903  lev = 9
# After read_pla the network is a single SOP node, so `cube` is the product-term
# count; after `if -K 6` the network is LUT-mapped, so `nd` is the LUT6 count.
_ND = re.compile(r"\bnd\s*=\s*(\d+)")
_CUBE = re.compile(r"\bcube\s*=\s*(\d+)")
_LEV = re.compile(r"\blev\s*=\s*(\d+)")

# resyn2, spelled out. Homebrew's yosys-abc ships without abc.rc, where the
# b/rw/rf aliases live, so the shorthand silently fails with a file-not-found.
_RESYN2 = ("balance; rewrite; refactor; balance; rewrite; rewrite -z; "
           "balance; refactor -z; rewrite -z; balance")
# Chosen by measurement on this design, not by habit: dch+resyn2 beat plain dc2
# (73 vs 82 LUT6 on neuron015) and costs <2 s even on the 6,464-minterm neuron.
_MAP = f"strash; dch; {_RESYN2}; if -K 6; mfs2"


def parse_args():
	ap = argparse.ArgumentParser(description="ABC-minimise per-neuron PLAs")
	ap.add_argument("--pla-dir", required=True)
	ap.add_argument("--abc", default="yosys-abc", help="ABC binary (yosys ships yosys-abc)")
	# Default 2: a controller run usually owns this box, and ABC is single-threaded
	# per file -- do not starve the live search.
	ap.add_argument("--jobs", type=int, default=2)
	return ap.parse_args()


def run_abc(abc: str, pla: Path) -> dict:
	"""Minimise one PLA; return {'cubes', 'luts', 'levels'} (None on failure)."""
	# Two passes: the as-read SOP gives the product-term count, the mapped
	# network gives LUT6s + depth.
	script_sop = f"read_pla {pla}; ps"
	script_lut = f"read_pla {pla}; {_MAP}; ps"
	out = {"neuron": int(pla.stem.replace("neuron", "")), "pla": str(pla)}
	for key, script in (("sop", script_sop), ("lut", script_lut)):
		try:
			r = subprocess.run([abc, "-q", script], capture_output=True, text=True, timeout=1800)
			txt = r.stdout + r.stderr
		except (subprocess.TimeoutExpired, FileNotFoundError) as e:
			out[f"{key}_error"] = str(e)
			continue
		if key == "sop":
			m = _CUBE.search(txt)
			out["cubes"] = int(m.group(1)) if m else None
		else:
			m, lev = _ND.search(txt), _LEV.search(txt)
			out["luts"] = int(m.group(1)) if m else None
			out["levels"] = int(lev.group(1)) if lev else None
		if m is None:
			out[f"{key}_raw"] = txt[-400:]
	return out


def main():
	args = parse_args()
	d = Path(args.pla_dir)
	meta = json.loads((d / "index.json").read_text())
	onset = {e["neuron"]: e["onset"] for e in meta["files"]}
	plas = sorted(d.glob("neuron*.pla"))
	print(f"minimising {len(plas)} neurons ({meta['bits']} inputs, "
	      f"{meta['total_onset']} ON minterms, unlisted={meta['offset_semantics']}) "
	      f"with {args.abc}, jobs={args.jobs}\n")

	with ThreadPoolExecutor(max_workers=args.jobs) as ex:
		rows = sorted(ex.map(lambda p: run_abc(args.abc, p), plas),
		              key=lambda r: r["neuron"])

	bits = meta["bits"]
	print(f"{'neu':>3} {'ON-set':>7} {'cubes':>7} {'ratio':>6} {'LUT6':>7} {'lev':>4}")
	tot_on = tot_cubes = tot_luts = 0
	bad = []
	for r in rows:
		n, on = r["neuron"], onset[r["neuron"]]
		c, l, lv = r.get("cubes"), r.get("luts"), r.get("levels")
		if c is None or l is None:
			bad.append(r); continue
		tot_on += on; tot_cubes += c; tot_luts += l
		print(f"{n:3d} {on:7d} {c:7d} {on/max(c,1):6.1f} {l:7d} {lv if lv is not None else '-':>4}")
	print(f"\nTOTAL  ON-set {tot_on}  cubes {tot_cubes}  LUT6 {tot_luts}")
	if tot_cubes:
		print(f"  minterm->cube compression: {tot_on/tot_cubes:.1f}x")
	print(f"  naive sparse (keys+values, NOT deployed): "
	      f"{(tot_on*((bits+7)//8) + (tot_on+7)//8)/1024:.0f} KB")
	if bad:
		print(f"\n{len(bad)} neuron(s) FAILED — not counted:")
		for r in bad:
			print(f"  neuron {r['neuron']}: {r.get('sop_error') or r.get('lut_error') or ''}"
			      f"{(r.get('sop_raw') or r.get('lut_raw') or '')[:200]}")
	(d / "minimized.json").write_text(json.dumps(
		{"meta": meta, "rows": rows,
		 "totals": {"onset": tot_on, "cubes": tot_cubes, "luts": tot_luts,
		            "failed": len(bad)}}, indent=1) + "\n")
	print(f"\nwrote {d/'minimized.json'}")


if __name__ == "__main__":
	main()
