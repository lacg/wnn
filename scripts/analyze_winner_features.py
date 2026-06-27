#!/usr/bin/env python3
"""Per-FEATURE connectivity + memory stats for a saved controller winner.

Answers: now that the frame fix is in and the feature-balance cap is OFF, how are
connections distributed across input features (is any feature over/under-wired —
the old '2.14x yaw_err' concern), and how big is the learned memory?

Usage: analyze_winner_features.py <winner.yaml.gz> [<winner.yaml.gz> ...]
"""
import gzip
import sys

import yaml

sys.path.insert(0, "/Users/lacg/wnn/src/wnn")
from wnn.control.recurrent_genome import RecurrentArchGenome, _feature_of  # noqa: E402

BASE9 = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "tgt_roll", "tgt_pitch", "tgt_yaw"]


def feature_names(spec: dict, nf: int) -> list[str]:
	"""Canonical compute_features order: base 9 + enabled extras."""
	names = list(BASE9)
	pax = 3 if spec.get("obs_peraxis_yaw", True) else 2
	if spec.get("obs_tilt_p"):    names.append("tilt_p")
	if spec.get("obs_tilt_i"):    names.append("tilt_i")
	if spec.get("obs_peraxis_p"): names += [f"peraxis_p[{a}]" for a in (["r", "p", "y"][:pax])]
	if spec.get("obs_peraxis_i"): names += [f"peraxis_i[{a}]" for a in (["r", "p", "y"][:pax])]
	if spec.get("obs_pwm"):       names += [f"pwm[{m}]" for m in range(spec.get("num_motors", 4))]
	if spec.get("obs_yaw_err"):   names.append("yaw_err")
	if spec.get("obs_yaw_err_i"): names.append("yaw_err_i")
	while len(names) < nf:  # safety pad
		names.append(f"feat[{len(names)}]")
	return names[:nf]


def per_feature_counts(sampled: list[list[int]], frame_bits: int, bpf: int, nf: int) -> list[int]:
	counts = [0] * nf
	for suf in sampled:
		for b in suf:
			counts[_feature_of(b, frame_bits, bpf)] += 1
	return counts


def analyze(path: str) -> None:
	d = yaml.safe_load(gzip.open(path))
	g = RecurrentArchGenome.deserialize(d["best_genome"])
	spec = d.get("extra", {}).get("spec", {})
	bpf = spec.get("bits_per_feature", 8)
	frame_bits = g.shape.output_input_space          # = nf * bpf
	nf = frame_bits // bpf
	names = feature_names(spec, nf)

	sc = per_feature_counts(g.state_sampled, frame_bits, bpf, nf)
	oc = per_feature_counts(g.output_sampled, frame_bits, bpf, nf)
	tot = [s + o for s, o in zip(sc, oc)]

	print(f"\n===== {path}")
	print(f"arch: state_neurons={g.state_neurons} (suffix {g.state_suffix_width}b + forced prefix "
	      f"{g.forced_prefix}b = {g.state_bits_per_neuron}b/neuron)  output_neurons={g.output_neurons} "
	      f"(suffix {g.output_suffix_width}b + {g.forced_prefix}b = {g.output_bits_per_neuron}b/neuron)")
	print(f"features: nf={nf}  frame_bits={frame_bits}  bpf={bpf}  state_input_space={g.shape.state_input_space}")
	# memory
	if g.cells is not None:
		ns, no = len(g.cells.state_values), len(g.cells.output_values)
		from collections import Counter
		spn = Counter(n for n, _ in g.cells.state_universe)
		print(f"memory: state_cells={ns:,}  output_cells={no:,}  total={ns+no:,}  "
		      f"(state cells/neuron: min={min(spn.values()) if spn else 0} "
		      f"max={max(spn.values()) if spn else 0} mean={ns//max(g.state_neurons,1)})")
	else:
		print("memory: none (paradigm A — cells trained at eval)")
	# per-feature wiring
	print(f"\n  {'feature':<14} {'state':>6} {'out':>5} {'total':>6} {'share%':>7}")
	gtot = sum(tot) or 1
	for i in range(nf):
		bar = "#" * round(40 * tot[i] / max(tot))
		print(f"  {names[i]:<14} {sc[i]:>6} {oc[i]:>5} {tot[i]:>6} {100*tot[i]/gtot:>6.1f}%  {bar}")
	nz = [t for t in tot if t > 0] or [0]
	print(f"\n  balance: max/min(nonzero) = {max(tot)}/{min(nz)} = "
	      f"{max(tot)/max(min(nz),1):.2f}x  | unwired features: {sum(1 for t in tot if t==0)}/{nf}")


if __name__ == "__main__":
	for p in sys.argv[1:]:
		analyze(p)
