"""Build a 'logic-only' synth variant of an exported genome.

The full 64-bit sparse genomes need 13,618 RAMB18 (> any FPGA), so they can't
be synthesized on-chip. But the DEPLOYED on-chip footprint is just the N
binary-search FSMs (64-bit comparator + state machine + address counter) — the
sorted tables live in external DRAM. This script isolates that logic so it
synthesizes on a Z-7020:

  - caps each neuron's NUM_ENTRIES to CAP (default 32) so the memories are tiny
  - maps them to distributed RAM (ram_style="distributed") instead of block RAM
    (64-bit width otherwise forces >=2 RAMB18/neuron -> >280 total)
  - truncates each .mem to CAP lines so $readmemh matches the capped array

The 64-bit datapath (comparator critical path -> Fmax) and the per-neuron FSM
are preserved, so LUT/FF/Fmax/Power are representative of the real on-chip
search logic. The address counter shrinks (log2(CAP) vs 17 bits) — a few LUTs
per neuron — noted as a small under-count.
"""

from __future__ import annotations
import argparse, re, shutil
from pathlib import Path


BLOCK_ROM = 'rom_style = "block"'
DISTRIBUTED_ROM = 'rom_style = "distributed"'


def distributed_neuron(neuron_sv: str) -> str:
	"""Steer the neuron's key/value ROMs to distributed (LUT) ROM.

	The arrays are ROMs (only $readmemh writes them), so the attribute that
	steers them is rom_style (wnn_neuron.sv, fixed 31/08/2026). This used to
	replace `ram_style = "block"`, which no longer exists in the RTL: the
	replace silently did nothing and every "logic-only" variant still asked
	for block ROM. Refuse instead of emitting a variant that is not one.
	"""
	if BLOCK_ROM not in neuron_sv:
		raise ValueError(f"neuron RTL has no `{BLOCK_ROM}` attribute to steer to distributed ROM")
	return neuron_sv.replace(BLOCK_ROM, DISTRIBUTED_ROM)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--export-dir", required=True)
	ap.add_argument("--cap", type=int, default=32)
	ap.add_argument("--rtl", default="fpga/rtl/wnn_neuron.sv")
	args = ap.parse_args()

	src = Path(args.export_dir)
	dst = Path(str(src) + "_logic")
	if dst.exists():
		shutil.rmtree(dst)
	(dst / "mem").mkdir(parents=True)

	# 1) distributed-ROM variant of the neuron (same module name).
	(dst / "wnn_neuron.sv").write_text(distributed_neuron(Path(args.rtl).read_text()))

	# 2) cap NUM_ENTRIES in the classifier instantiations.
	cls = (src / "wnn_classifier_impl.sv").read_text()
	def cap_entries(m):
		v = int(m.group(1))
		return f".NUM_ENTRIES({min(v, args.cap)})"
	cls_new, n = re.subn(r"\.NUM_ENTRIES\((\d+)\)", cap_entries, cls)
	(dst / "wnn_classifier_impl.sv").write_text(cls_new)
	print(f"capped {n} NUM_ENTRIES instantiations to <= {args.cap}")

	# 3) truncate every .mem to CAP lines.
	mem_files = sorted((src / "mem").glob("*.mem"))
	for mf in mem_files:
		lines = mf.read_text().splitlines()[: args.cap]
		(dst / "mem" / mf.name).write_text("\n".join(lines) + "\n")
	print(f"truncated {len(mem_files)} .mem files to {args.cap} lines")
	print(f"logic-only variant: {dst}")


if __name__ == "__main__":
	main()
