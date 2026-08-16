"""Scoped GA-connectivity (16/08/2026, Luiz's connectivity types).

Proves the full Python->Rust path: a CONNECTIONS-dimension mutation under
conn_mutation_scope="feature" moves thresholds but never the feature map;
"window" never crosses time; "free" stays the legacy draw.
"""
import sys
sys.path.insert(0, "/Users/lacg/wnn/src/wnn")
import numpy as np
from wnn.control.recurrent_genome import (
	RecurrentArchShape, RecurrentArchGenome, RecurrentArchConfig,
)
from wnn.ram.strategies.optimization_dimension import OptimizationDimension

BPF, NFEAT, K, N = 8, 18, 4, 32
FRAME = NFEAT * BPF
SPACE = K * FRAME

shape = RecurrentArchShape(prefix_factor=0, state_input_space=0,
                           output_input_space=SPACE, output_quantum=4)


def feature_runs(g):
	"""per-neuron sorted list of (window, feature) pairs"""
	return [sorted({(b // FRAME, (b % FRAME) // BPF) for b in s}) for s in g.output_sampled]


def windows(g):
	return [sorted({b // FRAME for b in s}) for s in g.output_sampled]


def cfg(scope):
	return RecurrentArchConfig(bits_per_feature=BPF, input_window_k=K,
	                           conn_mutation_scope=scope,
	                           min_suffix=1, max_suffix=64)


base = RecurrentArchGenome.random(shape, 0, N, 0, 20, np.random.default_rng(3),
                                  config=cfg("free"))

# --- feature scope: the (window, feature) map is invariant under mutation -----
g = base.clone()
runs_before = feature_runs(g)
for i in range(5):
	g = g.mutate(OptimizationDimension.CONNECTIONS, 1.0, cfg("feature"),
	             np.random.default_rng(100 + i))
assert feature_runs(g) == runs_before, "feature scope moved a feature/window"
assert g.output_sampled != base.output_sampled, "rate 1.0 x5 should have moved thresholds"
for s in g.output_sampled:
	assert len(set(s)) == len(s), "distinctness broken"
print("1. feature scope: thresholds moved, (window, feature) map FROZEN ✓")

# --- window scope: windows invariant, features may move -----------------------
g = base.clone()
win_before = windows(g)
moved_feature = False
for i in range(5):
	g = g.mutate(OptimizationDimension.CONNECTIONS, 1.0, cfg("window"),
	             np.random.default_rng(200 + i))
assert windows(g) == win_before, "window scope crossed time"
assert feature_runs(g) != runs_before, "window scope should explore features"
print("2. window scope: time frozen, features explored ✓")

# --- free scope: legacy — can cross windows -----------------------------------
g = base.clone()
for i in range(5):
	g = g.mutate(OptimizationDimension.CONNECTIONS, 1.0, cfg("free"),
	             np.random.default_rng(300 + i))
assert windows(g) != win_before, "free scope should cross windows at rate 1.0 x5"
print("3. free scope: legacy behaviour (windows crossed) ✓")

print("ALL PASS")
