"""CTRL-69 collector: teacher-driven DAgger rollouts in the N0 s1 regime, traced per step.

Usage: python collect.py <run.out> <scratch_pkg_dir> <out.npz> <seed> <episodes> [winner.yaml.gz]
The scratch ram_controller (with the CTRL-69 diagnostic export trace_dagger_cascade,
logs/controller/ctrl69/trace_dagger_cascade.patch) is imported from <scratch_pkg_dir>,
prepended to sys.path; the venv's installed wheel is never touched.
Probe layout = the N0 argv + --obs-rate-i --obs-cascade-dhat-residual (r and I straight out
of compute_features). Teacher = the run's (mpcof), expert_drives=True.
With a 6th arg (post-hoc STUDENT-DRIVEN arm): <run.out> is that winner's own run, the controller
is the trained winner (its own layout — must already carry cresid), expert_drives=False, no
exploration: labels are the teacher's at the student-visited states, e/I are the flown (student's)
rate loop."""
import sys

sys.path.insert(0, sys.argv[2])
import numpy as np  # noqa: E402


def argv_tokens(path):
	with open(path, "rb") as f:
		lines = [l.decode("utf-8", "replace").rstrip("\n") for l in f if l.startswith(b"[argv] ")]
	assert len(lines) == 1, len(lines)
	return lines[0][len("[argv] "):].split()


def probe_spec(args, ec):
	from wnn.control.controller_grid_search import ControllerGridSearch
	gs = ControllerGridSearch.__new__(ControllerGridSearch)
	gs._args, gs._ec, gs._seed = args, ec, args.base_seed
	_, probe = gs._compute_valid_pairs()
	return probe


def feature_controller(probe):
	"""evaluator._feature_controller, but at the spec's action_repeat (the CTBR decision
	rate): the teacher's ExpertPilot holds its label for the controller's action_repeat."""
	from wnn.control import evaluator
	orig = evaluator.WnnController
	evaluator.WnnController = lambda **kw: orig(**kw, action_repeat=int(probe.action_repeat))
	try:
		return evaluator._feature_controller(probe)
	finally:
		evaluator.WnnController = orig


def main():
	from wnn.control import _accel as ra
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args, _resolve_ctbr
	from wnn.control.evaluator import reference_packed_config
	from wnn.control.feature_layout import feature_names
	assert hasattr(ra, "trace_dagger_cascade"), f"not the scratch wheel: {ra.__name__}"
	student = len(sys.argv) > 6
	toks = argv_tokens(sys.argv[1]) + ([] if student else ["--obs-rate-i", "--obs-cascade-dhat-residual"])
	args = build_arg_parser().parse_args(toks)
	args.cli_argv = []
	ec = episode_config_from_args(args)
	_resolve_ctbr(args, ec)
	args._dhat_b = None
	probe = probe_spec(args, ec)
	names = list(feature_names(probe))
	cfg = reference_packed_config(ec, args.teacher, args.teacher_hover)
	if student:
		from wnn.control.checkpoint_io import load_controller_checkpoint
		from wnn.control.evaluator import build_controller
		from wnn.control.evaluator import controller_genome_from_arch, fit_calibration, calib_episode_config
		from wnn.control.threshold_calibration import calibration_from_args
		pay = load_controller_checkpoint(sys.argv[6])
		spec = pay["spec"]
		# thresholds are not stored: re-fit them exactly as the run did (fit_cuts.py pattern)
		fit = fit_calibration(probe, calibration=calibration_from_args(args), seed=args.base_seed,
		                      geometry=getattr(ec, "geometry", None), alloc=getattr(ec, "alloc_residual", None),
		                      episode_config=calib_episode_config(args, ec))
		assert list(feature_names(spec)) == names, "winner layout != the run argv's layout"
		ctl = build_controller(controller_genome_from_arch(pay["best_genome"], spec, list(fit.thresholds)))
		cfg.expert_drives = False
		cfg.explore_eps = 0.0
	else:
		ctl = feature_controller(probe)
	seed, n = int(sys.argv[4]), int(sys.argv[5])
	assert not (31337001 <= seed < 31338000) and not (99990101 <= seed < 99990200), seed
	# the residual block's airframe constants, from the same argv + the residual flag
	from wnn.control.evaluator import ctbr_arg
	rargs = build_arg_parser().parse_args(argv_tokens(sys.argv[1]) + ["--obs-cascade-dhat-residual"])
	rargs.cli_argv = []
	rec = episode_config_from_args(rargs)
	_resolve_ctbr(rargs, rec)
	rargs._dhat_b = None
	plant = ctbr_arg(probe_spec(rargs, rec)).obs_dhat_residual_plant
	assert plant is not None
	feats, labels, lbase, phys, shadow = ra.trace_dagger_cascade(ctl, cfg, n, seed, list(plant))
	np.savez_compressed(sys.argv[3], feats=np.asarray(feats, np.float32), labels=np.asarray(labels, np.float32),
	                    label_base=np.asarray(lbase, np.float32), phys=np.asarray(phys, np.float32),
	                    shadow_resid=np.asarray(shadow, np.float32),
	                    names=np.asarray(names), action_repeat=int(probe.action_repeat),
	                    dt=float(ec.dt), steps=int(ec.steps_per_episode), seed=seed, episodes=n,
	                    tilt_deg=float(np.degrees(ec.max_initial_tilt_rad)))
	print("ABI", ra.ABI_VERSION, "features", len(names), "records", len(feats), "phys", len(phys),
	      "action_repeat", probe.action_repeat, "tilt", np.degrees(ec.max_initial_tilt_rad))
	print("names", names)
	if "cresid_x" in names:
		F = np.asarray(feats, np.float32)
		r = F[:, [names.index(f"cresid_{a}") for a in "xyz"]]
		print("PARITY shadow residual vs cresid feature: max|diff| =", float(np.abs(r - np.asarray(shadow)).max()))


main()
