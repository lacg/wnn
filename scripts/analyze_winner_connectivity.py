#!/usr/bin/env python3
"""Forensic connectivity histogram over saved winner architectures (14/08/2026).

QUESTION (Luiz): do the GA's winning architectures SPREAD their sampled bits
evenly across the input features, or CONCENTRATE them on a few? Spread supports
the "starved — give every neuron more bits" reading of the b=30 grid ceiling;
concentration says selection already prefers specialist-shaped neurons, which is
the arm-C hypothesis of the specialist experiment (task #15).

METHOD. For each winner file: take the best genome's output-layer sampled
suffixes (sn=0 ⇒ no forced prefix; every connection is a sampled sensor bit),
map bit index -> (frame, feature, threshold) via the K·F·b layout, and compare
the OBSERVED per-neuron feature concentration against the RANDOM-SAMPLING NULL
(same suffix widths, uniform without replacement over the same input space,
simulated). The null is the honest yardstick: connections were RANDOM AT INIT
(the connections GA stage is skipped in these runs), so any deviation is pure
SELECTION — which genomes survived — not mutation pressure.

Caveat printed with the results: selection acts on whole genomes, so per-neuron
concentration signal is diluted; a null-consistent result does NOT prove
specialisation is useless, only that selection over random maps never found it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "wnn"))
from wnn.control.checkpoint_io import load_controller_checkpoint  # noqa: E402

# Feature names come from the ONE table (feature_layout.py, held equal to Rust
# compute_features by tests/controller_feature_layout.py).
from wnn.control.feature_layout import feature_names  # noqa: E402


def per_neuron_stats(suffixes: list[list[int]], F: int, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""(distinct features, effective feature count, top3 share) per neuron."""
	dis, eff, top3 = [], [], []
	for s in suffixes:
		feats = (np.asarray(s) % (F * b)) // b
		counts = np.bincount(feats, minlength=F).astype(float)
		p = counts / counts.sum()
		nz = p[p > 0]
		dis.append(int((counts > 0).sum()))
		eff.append(float(np.exp(-(nz * np.log(nz)).sum())))
		top3.append(float(np.sort(counts)[-3:].sum() / counts.sum()))
	return np.array(dis), np.array(eff), np.array(top3)


def null_stats(widths: list[int], space: int, F: int, b: int, sims: int = 2000, seed: int = 7):
	"""Same stats under uniform sampling without replacement (the init law)."""
	rng = np.random.default_rng(seed)
	dis, eff, top3 = [], [], []
	for _ in range(sims):
		w = widths[rng.integers(len(widths))]
		s = rng.choice(space, size=w, replace=False)
		d, e, t = per_neuron_stats([list(s)], F, b)
		dis.append(d[0]); eff.append(e[0]); top3.append(t[0])
	return np.array(dis), np.array(eff), np.array(top3)


def analyze(path: str) -> None:
	p = load_controller_checkpoint(path)
	spec, g = p["spec"], p["best_genome"]
	b, k = spec.bits_per_feature, spec.input_window_k
	names = feature_names(spec)
	F = len(names)
	# OUTPUT layer space is ONE frame (evaluator.py: output_input_space = nf*b);
	# only the STATE layer samples the k-frame window, and these runs fly sn=0.
	# First cut of this script used k*F*b here and produced artifact z-scores —
	# the null must draw from the SAME space the init sampler used.
	space = F * b
	suffixes = g.output_sampled
	widths = sorted({len(s) for s in suffixes})
	allbits = np.concatenate([np.asarray(s) for s in suffixes])
	assert allbits.max() < space, f"output bit {allbits.max()} outside one-frame space {space}"
	feats = (allbits % (F * b)) // b
	share = np.bincount(feats, minlength=F) / len(allbits)

	d_o, e_o, t_o = per_neuron_stats(suffixes, F, b)
	d_n, e_n, t_n = null_stats([len(s) for s in suffixes], space, F, b)

	tag = Path(path).name.replace("_winner.yaml.gz", "")
	print(f"\n== {tag}")
	print(f"   neurons={len(suffixes)}  suffix_widths={widths}  F={F} b={b} k={k} space={space}")
	order = np.argsort(share)[::-1]
	row = "  ".join(f"{names[i]}={share[i]*100:.1f}%" for i in order)
	print(f"   feature share (uniform would be {100/F:.1f}% each):\n     {row}")

	# --- THE HISTOGRAMS (Luiz 14/08) -------------------------------------------
	# (1) per-feature TOTAL connection counts across all neurons — cross-feature
	# clustering at the architecture level. (2) the within-neuron distribution:
	# how many (neuron, feature) pairs carry exactly k bits — the histogram that
	# decides which minimum-per-feature rule would actually BIND (min-1 is moot
	# if random already leaves few features at 0; min-2 binds on the mass at 1).
	counts_total = np.bincount(feats, minlength=F)
	print("   per-feature TOTAL connections (all neurons):")
	for i in np.argsort(counts_total)[::-1]:
		bar = "#" * int(round(counts_total[i] / max(counts_total.max(), 1) * 40))
		print(f"     {names[i]:<11} {counts_total[i]:>4}  {bar}")
	pf = np.zeros((len(suffixes), F), dtype=int)
	for j, s in enumerate(suffixes):
		pf[j] = np.bincount((np.asarray(s) % (F * b)) // b, minlength=F)
	obs_hist = np.bincount(pf.ravel(), minlength=b + 1)
	rng = np.random.default_rng(11)
	null_pf = []
	for _ in range(2000):
		s = rng.choice(space, size=len(suffixes[0]), replace=False)
		null_pf.append(np.bincount((s % (F * b)) // b, minlength=F))
	nul_hist = np.bincount(np.concatenate(null_pf), minlength=b + 1).astype(float)
	nul_hist *= pf.size / nul_hist.sum()
	print("   bits-per-feature WITHIN a neuron  (pairs of neuron x feature):")
	print("     k bits :  " + "  ".join(f"{k:>5}" for k in range(b + 1)))
	print("     winner :  " + "  ".join(f"{obs_hist[k]:>5}" for k in range(b + 1)))
	print("     null   :  " + "  ".join(f"{nul_hist[k]:>5.0f}" for k in range(b + 1)))

	def cmp(label, obs, nul, fmt=".2f"):
		zn = (obs.mean() - nul.mean()) / (nul.std() / max(np.sqrt(len(obs)), 1) + 1e-12)
		print(f"   {label:<28} winner {obs.mean():{fmt}} ± {obs.std():{fmt}}   "
		      f"null {nul.mean():{fmt}} ± {nul.std():{fmt}}   z≈{zn:+.1f}")
	cmp("distinct features / neuron", d_o, d_n)
	cmp("effective features (expH)", e_o, e_n)
	cmp("top-3 feature share", t_o, t_n, ".3f")


if __name__ == "__main__":
	files = sys.argv[1:] or sorted(
		str(f) for f in Path("logs/controller").glob("*/*_winner.yaml.gz"))
	print("NOTE: connections in these runs are RANDOM AT INIT (connections GA skipped);")
	print("deviations from the null are SELECTION ONLY, and genome-level selection")
	print("dilutes per-neuron signal — read direction, not magnitude.")
	for f in files:
		try:
			analyze(f)
		except Exception as e:
			print(f"\n== {Path(f).name}: FAILED ({e})")
