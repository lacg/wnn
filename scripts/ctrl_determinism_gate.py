"""Controller DETERMINISM GATE helper (Luiz 17/11/2026) — used by scripts/ctrl_determinism_gate.sh.

  argv  <base.out> <run_dir>          print the gate's phased_ga argv (shell-quoted, one line)
  fp    <run_dir>                     write <run_dir>/fingerprints.json from the stage checkpoints
  compare <run_dir> <run_dir> [...]   compare runs; exit 1 on any mismatch

The gate run is the base recipe's EXACT argv (the run's `[argv]` line) shrunk to a small
budget — grid + 3 NEURONS gens, memory skipped, pop 8, 1000-step episodes, 2 VAL units —
on base seed 31337901 and test seeds 77770001-2: outside every report range (the report
units are 99990101..99990400), so a gate never touches a TEST unit.

What must be identical across runs (timing and memory-budget figures are stripped):
  * every GA gen line, GRID WINNER / expand line, stage-select HEADLINE and held-out line;
  * every `[gen-digest]` line (only runs that print them — ABI >= 56 Python);
  * fingerprints.json: per stage, sha256 over (genome.fingerprint()) of the saved population.
"""
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

GATE_SEED = "31337901"
GATE_TEST_SEEDS = ["77770001", "77770002"]
SET = {"--pop": "8", "--neurons-gens": "3", "--neurons-patience": "9", "--memory-gens": "1",
       "--eval-episodes": "20", "--memory-eval-episodes": "20", "--test-episodes": "10",
       "--calib-episodes": "30", "--val-units": "2", "--holdout-pop-sample": "2", "--steps": "1000",
       "--check-interval": "1", "--base-seed": GATE_SEED, "--skip-stages": "bits,connections,memory"}
DROP = ("--save-stage-checkpoints", "--save-winner", "--test-seeds")
KEEP = re.compile(r"Gen \d+/\d+:|GRID WINNER|\[expand |\[gen-digest\]|HEADLINE|MULTI-SEED|RESULT —")
STRIP = [re.compile(p) for p in (
	r"\| [\d.]+s \(offspring[^\]]*\]", r"\[elapsed:[^\]]*\]", r"\(\d+(\.\d+)?s\)", r"\(\d+(\.\d+)?s/gen\)",
	r"\d+(\.\d+)?s/gen", r"in [\d.]+s")]


def build_argv(base_out: str, run_dir: str) -> list:
	toks = [l for l in open(base_out, errors="replace").read().splitlines() if l.startswith("[argv] ")]
	if len(toks) != 1:
		raise SystemExit(f"{base_out}: expected one [argv] line")
	toks = toks[0][len("[argv] "):].split()
	out, i = [], 0
	while i < len(toks):
		t = toks[i]
		if t in SET:
			out += [t, SET[t]]
			i += 2
			continue
		if t in DROP:
			i += 1
			while i < len(toks) and not toks[i].startswith("--"):
				i += 1
			continue
		out.append(t)
		i += 1
	for k, v in SET.items():
		if k not in out:
			out += [k, v]
	return out + ["--test-seeds", *GATE_TEST_SEEDS, "--save-stage-checkpoints", f"{run_dir}/ckpt",
	              "--save-winner", f"{run_dir}/winner.yaml.gz"]


def fingerprints(run_dir: str) -> dict:
	from wnn.control.phased_ga import _stage_entries_from_checkpoints
	out = {}
	for label, _spec, res in _stage_entries_from_checkpoints(f"{run_dir}/ckpt", top_k=10_000):
		pop = list(getattr(res, "final_population", None) or [res.best_genome])
		h = hashlib.sha256()
		for g in pop:
			h.update(repr(g.fingerprint()).encode() + b"\n")
		out[label] = {"n": len(pop), "sha": h.hexdigest()[:16]}
	return out


def norm_lines(run_dir: str) -> list:
	lines = []
	for l in open(f"{run_dir}/run.out", errors="replace").read().splitlines():
		if KEEP.search(l):
			for p in STRIP:
				l = p.sub("", l)
			lines.append(re.sub(r"\s+", " ", l).strip())
	return lines


def compare(dirs: list) -> int:
	ref = dirs[0]
	ref_lines, ref_fp = norm_lines(ref), json.load(open(f"{ref}/fingerprints.json"))
	bad = 0
	for d in dirs[1:]:
		lines, fp = norm_lines(d), json.load(open(f"{d}/fingerprints.json"))
		has_dig = any("[gen-digest]" in l for l in lines) and any("[gen-digest]" in l for l in ref_lines)
		a = [l for l in ref_lines if has_dig or "[gen-digest]" not in l]
		b = [l for l in lines if has_dig or "[gen-digest]" not in l]
		print(f"[gate] {ref} vs {d}: {len(a)} vs {len(b)} compared lines (digests {'compared' if has_dig else 'NOT both present'})")
		for k, (x, y) in enumerate(zip(a, b)):
			if x != y:
				print(f"[gate] FIRST DIVERGENCE at compared line {k}:\n  A: {x}\n  B: {y}")
				bad += 1
				break
		if len(a) != len(b):
			print(f"[gate] line count differs: {len(a)} vs {len(b)}")
			bad += 1
		if fp != ref_fp:
			print(f"[gate] checkpoint fingerprints differ:\n  A: {ref_fp}\n  B: {fp}")
			bad += 1
		else:
			print(f"[gate] checkpoint fingerprints identical: {fp}")
	print("[gate] PASS" if bad == 0 else f"[gate] FAIL ({bad})")
	return 1 if bad else 0


if __name__ == "__main__":
	cmd = sys.argv[1]
	if cmd == "argv":
		print(shlex.join(build_argv(sys.argv[2], sys.argv[3])))
	elif cmd == "fp":
		json.dump(fingerprints(sys.argv[2]), open(f"{sys.argv[2]}/fingerprints.json", "w"), indent=1)
	elif cmd == "compare":
		sys.exit(compare(sys.argv[2:]))
	else:
		raise SystemExit(__doc__)
