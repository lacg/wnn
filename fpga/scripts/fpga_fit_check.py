#!/usr/bin/env python3
"""Refuse to synthesise a WNN export whose model cannot physically fit the part.

WHY THIS GATE EXISTS (31/08/2026). Every utilization number in fpga/results/
reports 0 BRAM and 0 LUT-as-Memory, and the LUT counts are far too small to hold
the model: CICIOT46M 500n x 34b needs 151 Mb of keys against the Z-7020's 4.9 Mb
of BRAM, yet "synthesised" to 50,527 LUTs. The memory had been optimised away
(conditional $readmemh + ram_style on a ROM -- both fixed in rtl/wnn_neuron.sv),
so those numbers measure the binary-search FSM and address formation ONLY, with
the model absent.

Two independent failures produced that, and this gate closes the second one:
a design 31x too large for the part should never have reached synthesis at all.
Run this BEFORE synth; a design that cannot fit must be reported as "needs
external memory", never as a LUT count.

Usage:  python3 fpga/scripts/fpga_fit_check.py [export_dir ...]
        (no args = check every export under fpga/export/)
Exit 0 if all checked designs fit, 1 otherwise.
"""
import glob, os, sys

# Xilinx Zynq Z-7020 (xc7z020clg400-1)
BRAM_MB = 4.9          # block RAM, Mb
LUT6 = 53_200          # each LUT6 can serve as 64 bits of ROM
LUTROM_MB = LUT6 * 64 / 1e6
VALUE_BITS = 8

def measure(d):
	keys = sorted(glob.glob(os.path.join(d, 'mem', '*_keys.mem')))
	if not keys:
		return None
	entries = sum(sum(1 for _ in open(f)) for f in keys)
	first = open(keys[0]).readline().strip()
	key_bits = len(first) * 4           # hex chars -> bits
	largest = max(sum(1 for _ in open(f)) for f in keys)
	return dict(neurons=len(keys), entries=entries, key_bits=key_bits,
	            mb=entries * (key_bits + VALUE_BITS) / 1e6,
	            largest_mb=largest * (key_bits + VALUE_BITS) / 1e6)

def main(argv):
	dirs = argv[1:] or sorted(
		p for p in glob.glob(os.path.join(os.path.dirname(__file__), '..', 'export', '*'))
		if os.path.isdir(os.path.join(p, 'mem')))
	print(f"{'export':<34}{'neur':>5}{'entries':>12}{'key':>5}{'Mb':>9}  verdict")
	bad = 0
	for d in dirs:
		m = measure(d)
		if m is None:
			print(f"{os.path.basename(d):<34}  no mem/ -- SKIP")
			continue
		if m['mb'] <= BRAM_MB:
			v = f"FITS BRAM ({m['mb']/BRAM_MB*100:.0f}%)"
		elif m['mb'] <= BRAM_MB + LUTROM_MB:
			v = f"tight -- needs BRAM+LUTROM ({m['mb']:.1f} of {BRAM_MB+LUTROM_MB:.1f} Mb)"
		else:
			v = f"DOES NOT FIT -- {m['mb']/BRAM_MB:.0f}x the BRAM; needs external memory"
			bad += 1
		print(f"{os.path.basename(d):<34}{m['neurons']:>5}{m['entries']:>12,}"
		      f"{m['key_bits']:>5}{m['mb']:>9.2f}  {v}")
	print(f"\nZ-7020: {BRAM_MB} Mb BRAM + {LUTROM_MB:.2f} Mb if ALL {LUT6:,} LUTs were ROM.")
	if bad:
		print(f"\n{bad} design(s) CANNOT fit. Do not quote a LUT count for these — a\n"
		      f"synthesis that 'succeeds' on them has dropped the model.")
	return 1 if bad else 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
