#!/usr/bin/env python3
"""pef64 key-store EXACTNESS: the partitioned Elias-Fano store must answer every
membership query exactly as binary search on the uint32 TRUE-key arrays does.

Per winner (TRUE on-set from export_controller_c.true_onset, i.e. the shipped keys):
  keys     every stored key                       -> TRUE
  near     every key-1 and key+1                  -> as the uint32 set says
  trace    every distinct (neuron, address) of the recorded VAL traces
           (logs/compression_probe/data/ctrl_trace_<tag>_u*.npz, when present)
  random   --random uniform (neuron, address < 2^bits) pairs (default 1M)
Reference = numpy searchsorted on each neuron's uint32 array (binary search).
  Python   scripts/pef_decoder.py (the mirror of mcu/wnn_pef.h) on every query
  C        (--c) pef_host_check: the uint32 C binary search vs mcu/wnn_pef.h,
           both compiled with clang on the host, on the same queries
Plus flash bytes per winner: uint32 (keys*4 + off + conn) vs pef64 (all arrays + conn).

  PYTHONPATH=src/wnn python mcu/parity/test_pef64_exactness.py --c
  python -m pytest mcu/parity/test_pef64_exactness.py   # synthetic sets, no winners needed
"""

import argparse
import glob
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from pef_decoder import PefDecoder  # noqa: E402
from pef_encoder import PefEncoder  # noqa: E402

MAIN = Path("/Users/lacg/wnn")
WINNER_DIR = MAIN / "logs/controller/ctrl48"
TRACE_DIR = MAIN / "logs/compression_probe/data"
TAG = "C-S2-d_wposx2_a59_rf_c200_dc_biasin_ms01_P_both_L1_{arm}_full_s{seed}"
WINNERS = [TAG.format(arm="ffF_yld", seed=s) for s in range(1, 6)] + \
          [TAG.format(arm="dmy03", seed=s) for s in (1, 2)]


def parse_args():
	ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	ap.add_argument("--winner-dir", default=str(WINNER_DIR))
	ap.add_argument("--trace-dir", default=str(TRACE_DIR))
	ap.add_argument("--tags", nargs="*", default=WINNERS)
	ap.add_argument("--random", type=int, default=1_000_000)
	ap.add_argument("--c", action="store_true", help="also run the C host check (clang)")
	ap.add_argument("--seed", type=int, default=20261120)
	return ap.parse_args()


# ------------------------------------------------------------- queries --

def neuron_arrays(keys: list, off: list) -> list:
	k = np.asarray(keys, np.uint64)
	return [k[off[n]:off[n + 1]] for n in range(len(off) - 1)]


def reference(per: list, n: np.ndarray, a: np.ndarray) -> np.ndarray:
	"""Binary search on each neuron's sorted uint32 keys."""
	out = np.zeros(len(n), bool)
	for j in np.unique(n):
		sel, k = n == j, per[j]
		if len(k):
			r = np.searchsorted(k, a[sel].astype(np.uint64))
			out[sel] = (r < len(k)) & (k[np.minimum(r, len(k) - 1)] == a[sel])
	return out


def key_queries(off: list, keys: list) -> tuple:
	n = np.repeat(np.arange(len(off) - 1), np.diff(off))
	return n, np.asarray(keys, np.int64)


def near_queries(off: list, keys: list, bits: int) -> tuple:
	n, k = key_queries(off, keys)
	n2, a = np.concatenate([n, n]), np.concatenate([k - 1, k + 1])
	ok = (a >= 0) & (a < (1 << bits))
	return n2[ok], a[ok]


def random_queries(neurons: int, bits: int, count: int, rng) -> tuple:
	return rng.integers(0, neurons, count), rng.integers(0, 1 << bits, count, dtype=np.int64)


def trace_queries(trace_dir: str, tag: str, neurons: int) -> tuple:
	"""Every distinct (neuron, address) the winner produced on its VAL traces."""
	paths = sorted(glob.glob(os.path.join(trace_dir, f"ctrl_trace_{tag}_u*.npz")))
	cols = [[np.zeros(0, np.uint32)] for _ in range(neurons)]
	for p in paths:
		addr = np.load(p)["addr"]
		if addr.shape[1] != neurons:
			raise SystemExit(f"{p}: {addr.shape[1]} neurons != winner's {neurons}")
		for j in range(neurons):
			cols[j].append(np.unique(addr[:, j]))
	uniq = [np.unique(np.concatenate(c)) for c in cols]
	n = np.concatenate([np.full(len(u), j) for j, u in enumerate(uniq)])
	return (n, np.concatenate(uniq).astype(np.int64)), len(paths)


# ------------------------------------------------------------- checks --

def python_mismatches(dec: PefDecoder, n: np.ndarray, a: np.ndarray, expected: np.ndarray) -> int:
	got = np.fromiter((dec.contains(int(x), int(y)) for x, y in zip(n, a)), bool, len(n))
	return int((got != expected).sum())


def check_set(name: str, dec, per, q: tuple, must_be_true: bool) -> dict:
	n, a = q
	exp = reference(per, n, a)
	bad_ref = int((~exp).sum()) if must_be_true else 0
	bad = python_mismatches(dec, n, a, exp)
	print(f"  python {name:<7} checked {len(n):>9}  TRUE {int(exp.sum()):>8}  mismatches {bad}"
	      + (f"  (stored keys not TRUE in ref: {bad_ref})" if must_be_true else ""), flush=True)
	return dict(name=name, n=n, a=a, exp=exp, bad=bad + bad_ref)


# ------------------------------------------------------------- C host --

def build_c(onset: dict, pef, work: Path) -> Path:
	import export_controller_c as exp
	(work / "u32.h").write_text(exp.render_header(onset, "pef64 exactness", None))
	(work / "pef.h").write_text(exp.render_header(onset, "pef64 exactness", None, pef))
	here, cc = Path(__file__).parent, ["clang", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-c"]
	for hdr, fn in (("u32.h", "u32"), ("pef.h", "pef")):
		subprocess.run(cc + [f'-DWNN_MODEL_HEADER="{work / hdr}"', f"-DLOOKUP_FN={fn}_present",
		                     f"-DNEURONS_FN={fn}_neurons", str(here / "pef_lookup_tu.c"),
		                     "-o", str(work / f"{fn}.o")], check=True)
	exe = work / "pef_host_check"
	subprocess.run(["clang", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", str(here / "pef_host_check.c"),
	                str(work / "u32.o"), str(work / "pef.o"), "-o", str(exe)], check=True)
	return exe


def run_c(exe: Path, sets: list, work: Path) -> int:
	trip = np.concatenate([np.stack([s["n"], s["a"], s["exp"]], 1).astype(np.uint32) for s in sets])
	path = work / "triples.u32"
	trip.astype("<u4").tofile(path)
	r = subprocess.run([str(exe), str(path)], capture_output=True, text=True)
	print("  " + (r.stdout.strip() or r.stderr.strip()), flush=True)
	return r.returncode


# ------------------------------------------------------------- sizes --

def size_line(tag: str, onset: dict, pef) -> str:
	conn = len(onset["conn"])          # uint8 connectivity here (in_bits <= 256)
	u32 = len(onset["keys"]) * 4 + len(onset["off"]) * 4 + conn
	p = pef.nbytes() + conn
	return (f"SIZE {tag}: keys {len(onset['keys'])}  uint32 {u32 / 1024:.0f} KB  pef64 {p / 1024:.0f} KB  "
	        f"x{p / u32:.3f}  ({8 * pef.nbytes() / max(1, len(onset['keys'])):.2f} b/key incl. index)")


def check_winner(tag: str, args, rng) -> int:
	from export_controller_c import true_onset
	onset = true_onset(os.path.join(args.winner_dir, tag + "_winner.yaml.gz"))
	pef = PefEncoder().encode(onset["keys"], onset["off"])
	dec, per, bits = PefDecoder(pef), neuron_arrays(onset["keys"], onset["off"]), onset["bits"]
	print(f"{tag}: {onset['neurons']}n x {bits}b  keys {len(onset['keys'])}  blocks {pef.num_blocks()}", flush=True)
	tq, ntr = trace_queries(args.trace_dir, tag, onset["neurons"])
	sets = [check_set("keys", dec, per, key_queries(onset["off"], onset["keys"]), True),
	        check_set("near", dec, per, near_queries(onset["off"], onset["keys"], bits), False),
	        check_set(f"trace{ntr}", dec, per, tq, False),
	        check_set("random", dec, per, random_queries(onset["neurons"], bits, args.random, rng), False)]
	bad = sum(s["bad"] for s in sets)
	if args.c:
		with tempfile.TemporaryDirectory(prefix="pef64_") as d:
			bad += run_c(build_c(onset, pef, Path(d)), sets, Path(d)) != 0
	print(size_line(tag, onset, pef), flush=True)
	return bad


def main():
	args = parse_args()
	rng = np.random.default_rng(args.seed)
	failures = {tag: check_winner(tag, args, rng) for tag in args.tags}
	bad = {t: b for t, b in failures.items() if b}
	print(f"\nPEF64 EXACTNESS: {'PASS' if not bad else 'FAIL'}  ({len(failures) - len(bad)}/{len(failures)} winners clean)")
	return 1 if bad else 0


# ------------------------------------------------------------- pytest --

def synthetic_sets(rng) -> tuple:
	keys, off = [], [0]
	for bits, count in [(4, 0), (4, 16), (10, 1), (10, 64), (10, 65), (24, 129), (24, 5000), (32, 300)]:
		keys += sorted(rng.sample(range(1 << bits), count))
		off.append(len(keys))
	keys += list(range(7000, 7200))            # one dense run (l = 0 everywhere)
	off.append(len(keys))
	return keys, off


def test_pef64_synthetic():
	rng = random.Random(7)
	keys, off = synthetic_sets(rng)
	dec = PefDecoder(PefEncoder().encode(keys, off))
	for n in range(len(off) - 1):
		k = set(keys[off[n]:off[n + 1]])
		probe = k | {x + 1 for x in k} | {max(0, x - 1) for x in k} | {rng.randrange(1 << 32) for _ in range(500)}
		assert all(dec.contains(n, a) == (a in k) for a in probe | {0, (1 << 32) - 1})


def test_pef64_mutant_is_caught():
	"""A flipped bitstream bit MUST change some answer — else the check is blind."""
	rng = random.Random(11)
	keys, off = synthetic_sets(rng)
	t = PefEncoder().encode(keys, off)
	t.data[len(t.data) // 2] ^= 1 << 7
	dec = PefDecoder(t)
	assert any(not dec.contains(n, a) for n in range(len(off) - 1) for a in keys[off[n]:off[n + 1]])


if __name__ == "__main__":
	sys.exit(main())
