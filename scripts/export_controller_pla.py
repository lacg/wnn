#!/usr/bin/env python3
"""Export a controller winner's memory as one PLA per output neuron, for logic
minimisation (espresso / ABC).

WHY. A RAM-WNN controller's memory is not a key->value store: every populated
cell in BINARY mode is TRUE, so the memory IS a set — 64 Boolean functions
f_n : {0,1}^b -> {0,1}, each with |ON_n| true minterms. Its deployment footprint
is therefore the MINIMISED CHARACTERISTIC FUNCTION (cubes -> LUTs), not
"keys x 4 bytes + values". Counting stored keys measures a representation nobody
would ship; logic synthesis is what the FPGA flow was already exploiting.

SEMANTICS OF THE UNLISTED MINTERMS. Two framings, and they are NOT interchangeable:

  --offset default  (.type f)  unlisted = 0. The synthesised function reproduces
      the controller BIT-EXACTLY on every input, reachable or not. This is the
      honest headline number.
  --offset dc       (.type fd) unlisted = don't-care. Far smaller, but it CHANGES
      the controller's output on addresses it never visited during recording. That
      is a behavioural change, so any number from it is an upper bound on
      compressibility and must be re-scored before it can be claimed.

Usage:
  PYTHONPATH=src/wnn python scripts/export_controller_pla.py \
      --winner logs/.../stage4_memory.yaml.gz --out /tmp/pla [--offset default|dc]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
	ap = argparse.ArgumentParser(description="Export controller memory as per-neuron PLA")
	ap.add_argument("--winner", required=True, help="schema-2 winner/stage yaml.gz")
	ap.add_argument("--out", required=True, help="output directory for the .pla files")
	ap.add_argument("--offset", choices=["default", "dc"], default="default",
	                help="unlisted minterms: 'default' = OFF (bit-exact), "
	                     "'dc' = don't-care (upper bound only)")
	return ap.parse_args()


def load_onsets(path: str):
	"""(bits_per_neuron, {neuron: [address, ...]}) from a winner checkpoint."""
	from wnn.control.checkpoint_io import load_controller_checkpoint
	ckpt = load_controller_checkpoint(path, skip_population=True)
	if ckpt is None:
		raise SystemExit(f"could not load {path}")
	spec, cells = ckpt["spec"], ckpt["best_genome"].cells
	uni, val = cells.output_universe, cells.output_values
	if uni is None or len(uni) == 0:
		raise SystemExit("winner has no output cells — nothing to minimise")
	non_true = sum(1 for v in val if int(v) != 1)
	if non_true:
		# BINARY memories store only TRUE cells; anything else means the caller
		# handed us a different memory mode and the ON-set framing is wrong.
		raise SystemExit(f"expected an all-TRUE ON-set, found {non_true} other values "
		                 f"— is this a BINARY winner?")
	onsets = defaultdict(list)
	for (n, addr) in uni:
		onsets[int(n)].append(int(addr))
	return int(spec.output_bits_per_neuron), onsets


def write_pla(path: Path, bits: int, addrs: list, offset: str) -> None:
	"""One PLA: `bits` inputs, 1 output, one line per ON minterm (MSB-first)."""
	lines = [f".i {bits}", ".o 1", ".type " + ("fd" if offset == "dc" else "f"),
	         f".p {len(addrs)}"]
	for a in addrs:
		lines.append(format(a, f"0{bits}b") + " 1")
	lines.append(".e")
	path.write_text("\n".join(lines) + "\n")


def main():
	args = parse_args()
	bits, onsets = load_onsets(args.winner)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	index = []
	for n in sorted(onsets):
		addrs = sorted(set(onsets[n]))
		p = out / f"neuron{n:03d}.pla"
		write_pla(p, bits, addrs, args.offset)
		index.append({"neuron": n, "onset": len(addrs), "pla": str(p)})
	meta = {"winner": args.winner, "bits": bits, "neurons": len(index),
	        "offset_semantics": args.offset,
	        "total_onset": sum(e["onset"] for e in index), "files": index}
	(out / "index.json").write_text(json.dumps(meta, indent=1) + "\n")
	print(f"wrote {len(index)} PLAs to {out}  ({bits} inputs, "
	      f"{meta['total_onset']} total ON minterms, unlisted={args.offset})")


if __name__ == "__main__":
	main()
