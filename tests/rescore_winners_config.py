"""rescore_winners.py: --airframe / --translation / --trained-* resolve through
phased_ga's OWN parser + episode_config_from_args, ladder winner tags parse, and
--dry-run prints both regimes and exits without loading a winner.

Run: PYTHONPATH=src/wnn python tests/rescore_winners_config.py
"""

import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "wnn"))
SCRIPT = os.path.join(ROOT, "scripts", "rescore_winners.py")

_spec = importlib.util.spec_from_file_location("rw", SCRIPT)
rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rw)

FAILS = 0


def check(label, got, want):
	global FAILS
	ok = got == want
	print(f"  {'ok  ' if ok else 'FAIL'} {label:<64} -> {got!r}" + ("" if ok else f" (expected {want!r})"))
	if not ok:
		FAILS += 1


def _args(**over):
	base = dict(steps=2000, tilt=5.0, body_rate=0.5, yaw_rate=0.3, episodes=100,
	            disturbance="L4A", airframe="cf21_brushless", translation=True,
	            trained_disturbance=None, trained_airframe=None, trained_translation=None)
	base.update(over)
	return SimpleNamespace(**base)


def test_parse_tag_accepts_dfa1l_cells_and_ladder_winners():
	d = rw._parse_tag("/x/dfa_9feat_QUAD_s31337002_winner.yaml.gz")
	check("dfa1l cell coordinates", (d["sub"], d["feat"], d["mode"], d["seed"]), ("dfa", "9feat", "QUAD", 31337002))
	l = rw._parse_tag("/x/SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_hd29_winner.yaml.gz")
	check("ladder tag keeps its full name", l["tag"], "SL_C_b24n256_cf21_brushless_L4C_g10_s31337005_hd29")
	check("ladder seed extracted from _s<seed>", l["seed"], 31337005)
	check("ladder tag has no cell coordinates", (l["sub"], l["feat"], l["mode"]), (None, None, None))
	check("no seed token -> None (skipped by the caller)", rw._parse_tag("/x/random_winner.yaml.gz"), None)


def test_trained_regime_defaults_fieldwise_to_the_scoring_one():
	score, trained = rw._conditions(_args())
	check("plain replay: trained == scoring", trained, score)
	score, trained = rw._conditions(_args(trained_disturbance="L4C"))
	check("A-cross: only the disturbance differs",
	      (trained.disturbance, trained.airframe, trained.translation), ("L4C", "cf21_brushless", True))
	score, trained = rw._conditions(_args(airframe="cf2x_firmware", trained_airframe="cf21_brushless"))
	check("airframe cross: scored on cf2x_firmware", score.airframe, "cf2x_firmware")
	check("airframe cross: address function from cf21_brushless", trained.airframe, "cf21_brushless")
	_, trained = rw._conditions(_args(translation=False, trained_translation=True))
	check("--trained-translation overrides a False scoring flag", trained.translation, True)
	check("argv carries only the set fields",
	      rw.Condition("L4B", None, False).argv(), ["--disturbance", "L4B"])
	check("argv with airframe + translation",
	      rw.Condition("L4A", "cf2x_urdf", True).argv(),
	      ["--disturbance", "L4A", "--airframe", "cf2x_urdf", "--translation"])


def test_regime_resolves_through_phased_ga():
	from wnn.control.phased_ga import build_arg_parser, episode_config_from_args
	a = _args()
	reg = rw._regime(rw.Condition("L4A", "cf21_brushless", True), a)
	check("airframe preset on the ec", reg.ec.airframe.name, "cf21_brushless")
	check("translation on the ec", reg.ec.translation, True)
	check("steps forwarded", reg.ec.steps_per_episode, 2000)
	check("tilt forwarded (deg)", round(math.degrees(reg.ec.max_initial_tilt_rad), 6), 5.0)
	check("rates forwarded", (reg.ec.max_initial_body_rate, reg.ec.max_initial_yaw_rate), (0.5, 0.3))
	check("report episodes = --episodes", reg.report_episodes, 100)
	check("K-fold is 5", reg.args.num_eval_folds, 5)
	# Same argv through phased_ga directly must give the SAME plant (no hand copy).
	ref = episode_config_from_args(build_arg_parser().parse_args(
		["--disturbance", "L4A", "--airframe", "cf21_brushless", "--translation",
		 "--steps", "2000", "--tilt", "5.0", "--body-rate", "0.5", "--yaw-rate", "0.3",
		 "--report-episodes", "100", "--num-eval-folds", "5"]))
	same = all(getattr(ref, f) == getattr(reg.ec, f) for f in
	           ("steps_per_episode", "max_initial_tilt_rad", "translation", "max_initial_alt_offset_m",
	            "max_initial_vz", "collective_cmd_jitter", "mass_jitter", "target_altitude"))
	check("ec fields identical to phased_ga's own assembly", same, True)
	check("disturbance preset identical", type(ref.disturbance).__name__, type(reg.ec.disturbance).__name__)
	legacy = rw._regime(rw.Condition("L4C", None, False), a)
	check("legacy plant: no airframe", legacy.ec.airframe, None)
	try:
		rw._regime(rw.Condition("L4A", None, True), a)
		check("translation without airframe refused", False, True)
	except SystemExit as e:
		check("translation without airframe refused (phased_ga's guard wording)",
		      "requires --airframe" in str(e), True)


def _cli(argv, cwd):
	env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src", "wnn"))
	return subprocess.run([sys.executable, SCRIPT] + argv, capture_output=True, text=True, env=env, cwd=cwd)


def test_dry_run_prints_both_regimes_and_writes_nothing():
	with tempfile.TemporaryDirectory() as td:
		for tag in ("SL_C_b24n256_cf21_brushless_L4C_g10_s31337002_hd29", "dfa_9feat_QUAD_s31337003", "junk"):
			open(os.path.join(td, tag + "_winner.yaml.gz"), "wb").write(b"not a checkpoint")
		out = os.path.join(td, "out.json")
		r = _cli(["--glob", os.path.join(td, "*_winner.yaml.gz"), "--airframe", "cf21_brushless",
		          "--translation", "--disturbance", "L4A", "--trained-disturbance", "L4C",
		          "--dry-run", "--out", out], td)
		check("dry-run exits 0", r.returncode, 0)
		check("dry-run header", "DRY RUN" in r.stdout, True)
		check("scoring regime printed", "score:   L4A / cf21_brushless / translation" in r.stdout, True)
		check("trained regime printed", "trained: L4C / cf21_brushless / translation" in r.stdout, True)
		check("matched count", "*_winner.yaml.gz': 3" in r.stdout, True)
		check("ladder winner parsed", "seed=31337002" in r.stdout, True)
		check("unparseable winner flagged", "UNPARSEABLE" in r.stdout, True)
		check("nothing scored: no 'scoring' line", "scoring " in r.stdout, False)
		check("nothing written", os.path.exists(out), False)


def test_parser_rejects_unknown_airframe_and_needs_out():
	with tempfile.TemporaryDirectory() as td:
		r = _cli(["--airframe", "cf9_imaginary", "--dry-run", "--out", "x.json"], td)
		check("unknown airframe -> argparse error (2)", r.returncode, 2)
		check("error names the choices", "cf21_brushless" in r.stderr, True)
		r = _cli(["--dry-run"], td)
		check("--out still required", r.returncode, 2)
		r = _cli(["--translation", "--dry-run", "--out", "x.json"], td)
		check("--translation without --airframe refused", r.returncode, 1)
		check("refusal names the rule", "requires --airframe" in (r.stderr + r.stdout), True)


if __name__ == "__main__":
	for name, fn in list(globals().items()):
		if name.startswith("test_") and callable(fn):
			print(f"=== {name}")
			fn()
	print()
	if FAILS:
		print(f"FAILED ({FAILS})"); sys.exit(1)
	print("ALL PASS — regimes resolve through phased_ga, ladder tags parse, dry-run is inert")
