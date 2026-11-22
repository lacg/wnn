"""Fit the run's thermometer cuts (teacher calibration ladder) and dump them as JSON.
Usage: python fit_cuts.py <run.out> <out.json>
Uses whichever wnn src + ram_controller wheel come first on sys.path (ABI 60 vs 61)."""
import json
import sys

import numpy as np


def argv_tokens(path):
	with open(path, "rb") as f:
		lines = [l.decode("utf-8", "replace").rstrip("\n") for l in f if l.startswith(b"[argv] ")]
	assert len(lines) == 1, len(lines)
	return lines[0][len("[argv] "):].split()


def main():
	from wnn.control import _accel
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args, _resolve_ctbr
	from wnn.control.controller_grid_search import ControllerGridSearch, _calib_ec
	from wnn.control.evaluator import fit_calibration
	from wnn.control.threshold_calibration import calibration_from_args
	from wnn.control.feature_layout import feature_names

	args = build_arg_parser().parse_args(argv_tokens(sys.argv[1]))
	args.cli_argv = []
	ec = episode_config_from_args(args)
	_resolve_ctbr(args, ec)
	args._dhat_b = None
	assert not getattr(args, 'obs_dhat', False)
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._seed = args, ec, args.base_seed
	_, probe = gs._compute_valid_pairs()
	fit = fit_calibration(probe, calibration=calibration_from_args(args), seed=args.base_seed,
	                      geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None),
	                      episode_config=_calib_ec(args, ec))
	names = feature_names(probe)
	th = np.asarray(fit.thresholds, dtype=np.float64).reshape(len(names), -1)
	json.dump({"abi": _accel.ABI_VERSION, "bpf": int(probe.bits_per_feature), "names": list(names),
	           "constant": [names[i] for i in fit.constant_features], "thresholds": th.tolist()},
	          open(sys.argv[2], "w"))
	print("ABI", _accel.ABI_VERSION, "features", len(names), "bpf", probe.bits_per_feature,
	      "constant", [names[i] for i in fit.constant_features])


main()
