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

# Canonical feature order (controller.rs): base 9, then enabled extras in the
# num_features expression's order. Names resolved per spec's obs flags.
BASE = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "tgt_r", "tgt_p", "tgt_y"]


def feature_names(spec) -> list[str]:
	n = list(BASE)
	if getattr(spec, "obs_tilt_p", False): n.append("tilt")
	if getattr(spec, "obs_tilt_i", False): n.append("tilt_i")
	if getattr(spec, "obs_peraxis_p", False):
		n += ["roll_err", "pitch_err"] + (["yaw_err_pa"] if getattr(spec, "obs_peraxis_yaw", False) else [])
	if getattr(spec, "obs_peraxis_i", False):
		n += ["roll_i", "pitch_i"] + (["yaw_i_pa"] if getattr(spec, "obs_peraxis_yaw", False) else [])
	if getattr(spec, "obs_pwm", False):
		n += [f"pwm_{m}" for m in range(4)]
	if getattr(spec, "obs_yaw_err", False): n.append("yaw_err")
	if getattr(spec, "obs_yaw_err_i", False): n.append("yaw_err_i")
	if getattr(spec, "dhat_b", None) is not None:
		n += ["dhat_r", "dhat_p", "dhat_y"]
	if getattr(spec, "obs_collective_cmd", False): n.append("coll_cmd")
	if getattr(spec, "obs_alt_err", False): n.append("alt_err")
	if getattr(spec, "obs_vz", False): n.append("vz")
	if getattr(spec, "obs_pos_err_xy", False): n += ["e_x", "e_y"]
	if getattr(spec, "obs_vel_xy", False): n += ["v_x", "v_y"]
	return n


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
