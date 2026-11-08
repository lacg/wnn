"""Grid and GA share ONE RecurrentArchConfig builder (08/11/2026 increment-0 fix B2).

controller_grid_search._random_genome used to build its own RecurrentArchConfig
and omitted input_window_k, so `--conn-policy framed1` drew grid genomes with
k=1 = min1 over the WHOLE K-frame window instead of one frame per neuron. The
fix routes the grid through arch_strategy.default_controller_arch_config.

Also pinned here: that builder's input_window_k is the number of frames in the
OUTPUT space (k under arm D, else 1). Passing the state window (k=4) to a
one-frame output space made conn-scope "window" cut it into quarter-"frames".

Run: PYTHONPATH=src python tests/controller_grid_shared_config.py
"""

from __future__ import annotations

import numpy as np

from wnn.control.phased_ga import build_arg_parser
from wnn.control.controller_grid_search import ControllerGridSearch
from wnn.control.arch_strategy import default_controller_arch_config
from wnn.control.recurrent_genome import (
	RecurrentArchGenome, RecurrentArchConfig, conn_policy_from_name,
)


def _grid(argv: list[str]) -> ControllerGridSearch:
	"""A grid search with only the genome-drawing state (no threshold fit, no evaluator)."""
	args = build_arg_parser().parse_args(argv)
	args._conn_policy, args._conn_policy_min = conn_policy_from_name(args.conn_policy)
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._rng_master = args, np.random.default_rng(7)
	gs._valid_pairs, _ = gs._compute_valid_pairs()
	return gs


def test_framed1_grid_genome_is_frame_pure():
	gs = _grid(["--conn-policy", "framed1", "--output-full-window", "--input-window-k", "4",
	            "--grid-state-neurons", "0", "--max-state-neurons", "0", "--grid-bits", "9"])
	point = gs._enumerate_points()[0]
	frame = point.shape.output_input_space // 4
	g = gs._random_genome(point)
	assert g.output_neurons > 0
	for n, suffix in enumerate(g.output_sampled):
		frames = {b // frame for b in suffix}
		assert len(frames) == 1, f"framed1 grid neuron {n} draws from frames {sorted(frames)}"
	print(f"✓ framed1 grid genome: all {g.output_neurons} neurons frame-pure")


def test_spread_grid_genome_unchanged():
	"""The production recipe's grid (spread, one output frame): the shared builder
	draws the SAME genome as the old inline config (bit-identical grid stage)."""
	gs = _grid(["--grid-state-neurons", "0", "--max-state-neurons", "0", "--grid-bits", "24",
	            "--memory-mode", "BINARY"])
	point = gs._enumerate_points()[0]
	args = gs._args
	seed = int(np.random.default_rng(3).integers(0, 2**32 - 1))
	old_cfg = RecurrentArchConfig(conn_policy=args._conn_policy, conn_policy_min=args._conn_policy_min,
	                              feature_balance_ratio=args.feature_balance_ratio,
	                              bits_per_feature=args.bits_per_feature, memory_mode=args.memory_mode)
	draw = lambda cfg: RecurrentArchGenome.random(
		point.shape, point.sn, point.on, point.state_suffix, point.output_suffix,
		np.random.default_rng(seed), config=cfg)
	assert draw(old_cfg).fingerprint() == draw(default_controller_arch_config(point.spec)).fingerprint()
	print("✓ spread grid genome bit-identical to the old inline config")


def test_arch_config_window_k_is_output_frames():
	gs = _grid(["--input-window-k", "4", "--grid-state-neurons", "0", "--max-state-neurons", "0"])
	assert default_controller_arch_config(gs._enumerate_points()[0].spec).input_window_k == 1
	gs = _grid(["--input-window-k", "4", "--output-full-window",
	            "--grid-state-neurons", "0", "--max-state-neurons", "0"])
	assert default_controller_arch_config(gs._enumerate_points()[0].spec).input_window_k == 4
	print("✓ arch config input_window_k = frames in the OUTPUT space (1 legacy, k under arm D)")


if __name__ == "__main__":
	test_framed1_grid_genome_is_frame_pure()
	test_spread_grid_genome_unchanged()
	test_arch_config_window_k_is_output_frames()
	print("\nAll grid shared-config tests passed.")
