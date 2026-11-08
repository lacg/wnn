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

from dataclasses import fields  # noqa: E402

from wnn.control.evaluator import ControllerSpec  # noqa: E402
from wnn.control.feature_layout import feature_names as _layout_names  # noqa: E402


def feature_names(spec: dict, nf: int) -> list[str]:
	"""Canonical compute_features order from the ONE table (feature_layout.py),
	for the spec dict a winner file carries."""
	known = {f.name for f in fields(ControllerSpec)}
	names = _layout_names(ControllerSpec(**{k: v for k, v in spec.items() if k in known}))
	if len(names) != nf:
		raise ValueError(f"winner spec names {len(names)} features, its genome space holds {nf}")
	return names


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
