"""Bit-identity of the ABI-44 genome operators vs the ABI-43 ones where nothing
should move (08/11/2026).

The bank-growth fix (B1) changes output neurogenesis and different-level-count
crossover ONLY. Everything else must reproduce the previous lineage exactly:
random(), mutate(BITS / CONNECTIONS / MEMORY), NEURONS steps that leave the output
count unchanged, same-shape crossover / crossover_average, same-shape Lamarckian
cell inheritance, and --receptive-fields off (no RNG draw).

The reference is the PREVIOUS recurrent_genome.py, read from git (`git show
<ref>:...`, ref = $CTRL45_BASE_REF or HEAD~0 of the source checkout given by
$CTRL45_BASE_REPO, default /Users/lacg/wnn), loaded as a separate module. Both
modules call the same Rust operators, so any difference is the Python change.

Run: PYTHONPATH=src python tests/controller_ctrl45_head_parity.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

import wnn.control.recurrent_genome as new_rg
from wnn.ram.strategies.optimization_dimension import OptimizationDimension as Dim

BASE_REPO = os.environ.get("CTRL45_BASE_REPO", "/Users/lacg/wnn")
BASE_REF = os.environ.get("CTRL45_BASE_REF", "dc4fdb2e")


def _load_base():
	src = subprocess.run(["git", "-C", BASE_REPO, "show", f"{BASE_REF}:src/wnn/control/recurrent_genome.py"],
	                     check=True, capture_output=True, text=True).stdout
	path = os.path.join(tempfile.mkdtemp(), "rg_base.py")
	with open(path, "w") as f:
		f.write(src)
	spec = importlib.util.spec_from_file_location("rg_base", path)
	mod = importlib.util.module_from_spec(spec)
	sys.modules["rg_base"] = mod
	spec.loader.exec_module(mod)
	return mod


old_rg = _load_base()
SHAPE = dict(prefix_factor=1, state_input_space=704, output_input_space=176, output_quantum=8)


def _pair(sn: int, on: int, ss: int, os_: int, seed: int, cfg_kw: dict | None = None):
	"""The same random genome built by both modules (with a cells payload)."""
	out = []
	for rg in (old_rg, new_rg):
		shape = rg.RecurrentArchShape(**SHAPE)
		cfg = rg.RecurrentArchConfig(**(cfg_kw or {}))
		g = rg.RecurrentArchGenome.random(shape, sn, on, ss, os_, np.random.default_rng(seed), config=cfg)
		r = np.random.default_rng(seed + 1)
		ou = [(int(n), int(a)) for n, a in zip(r.integers(0, on, 300), r.integers(0, 1 << (sn + os_), 300))]
		ou = sorted(set(ou))
		g.cells = rg.MemoryPayload([], ou, [], [int(v) for v in r.integers(0, 2, len(ou))])
		out.append((g, cfg))
	return out


def _same(a, b, what: str) -> None:
	assert a.fingerprint() == b.fingerprint(), f"{what}: lineage moved"


def test_unchanged_operators():
	for seed in range(6):
		(go, co), (gn, cn) = _pair(0, 256, 0, 24, seed, dict(output_block_delta=0, memory_mode="BINARY"))
		_same(go, gn, "random")
		for dim in (Dim.BITS, Dim.CONNECTIONS, Dim.MEMORY, Dim.NEURONS):
			mo = go.mutate(dim, 0.5, co, np.random.default_rng(seed + 10))
			mn = gn.mutate(dim, 0.5, cn, np.random.default_rng(seed + 10))
			_same(mo, mn, f"mutate {dim.name} (output count fixed)")
	print("✓ random + mutate(BITS/CONNECTIONS/MEMORY/NEURONS w/o output growth) bit-identical")


def test_same_shape_crossover_unchanged():
	for seed in range(6):
		(ao, _), (an, _) = _pair(0, 256, 0, 24, 100 + seed)
		(bo, _), (bn, _) = _pair(0, 256, 0, 24, 200 + seed)
		off = new_rg.RecurrentArchConfig()          # receptive_fields="off"
		for op in ("crossover", "crossover_average"):
			co = getattr(old_rg.RecurrentArchGenome, op)(ao, bo, np.random.default_rng(seed))
			for cfg in (None, off):
				cn = getattr(new_rg.RecurrentArchGenome, op)(an, bn, np.random.default_rng(seed), cfg)
				assert (co.output_sampled, co.state_sampled) == (cn.output_sampled, cn.state_sampled), \
					f"{op} (config={cfg is not None}): same-shape child moved"
	print("✓ same-shape crossover / crossover_average bit-identical (no config, and rf off)")


def test_same_shape_lamarckian_inheritance_unchanged():
	"""_inherit_cells == the old index-keyed filter when the shapes agree."""
	from wnn.control.arch_strategy import _inherit_cells, _filter_inherited_cells
	for seed in range(6):
		(_, _), (a, _) = _pair(0, 256, 0, 24, 300 + seed)
		(_, _), (b, _) = _pair(0, 256, 0, 24, 400 + seed)
		child = new_rg.RecurrentArchGenome.crossover(a, b, np.random.default_rng(seed))
		co = {i for i in range(child.output_neurons) if child.output_sampled[i] != a.output_sampled[i]}
		old = _filter_inherited_cells(child, a, changed_state=set(), changed_output=co)
		assert _inherit_cells(child, a).fingerprint() == old.fingerprint()
	print("✓ same-shape Lamarckian inheritance bit-identical")


if __name__ == "__main__":
	test_unchanged_operators()
	test_same_shape_crossover_unchanged()
	test_same_shape_lamarckian_inheritance_unchanged()
	print("\nAll CTRL-45 head-parity tests passed.")
