"""CTRL-83 (ABI 66): the cf21 FLAG-OFF pin vs ABI 65, measured across the two builds on the
flown l2b recipe — the CTRL-86 harness (tests/controller_ctrl86_flag_off.py: the same argv,
the same shrink, the same dump), re-pointed one ABI up:
  ABI 65: $CTRL83_BASE_PYTHONPATH (the ABI-65 src tree; the ABI-65 wheel = the venv's);
  ABI 66: this tree with the ABI-66 wheel first on PYTHONPATH.
Every dump is EQUAL: feature names, fitted thresholds, calibration token, resolved ctbr,
every parsed argument (minus the new --anchor-mass, which must resolve to 'true'), the
provenance tokens, 40 steps of features, the Lamarckian write-back digest after one short
DAgger and the scorer rows. A second ABI-66 dump spells --anchor-mass true out: equal again.
(The scorer-only D1/D3 options have no CLI: they are pinned in cargo, ctrl83_flag_off_pins.)

Run: CTRL83_BASE_PYTHONPATH=<abi65 src> PYTHONPATH=<abi66 wheel dir>:src python tests/controller_ctrl83_flag_off.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_ctrl86_flag_off as H  # noqa: E402  the CTRL-86 harness (argv, shrink, dump)

VARIANTS = {"off": [], "anchor_true": ["--anchor-mass", "true"]}
NEW_KEYS = ("anchor_mass",)


def dump(variant: str) -> dict:
	H.VARIANTS = {variant: VARIANTS[variant]}
	H.NEW_KEYS = H.NEW_KEYS + NEW_KEYS
	d = H.dump(variant)
	args, ec = H._args(VARIANTS[variant])
	d["_anchor"] = repr((getattr(args, "anchor_mass", None), getattr(ec, "anchor_mass", None)))
	return d


def _dump_in(pythonpath: str, variant: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", variant], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def test_flag_off_is_abi65() -> dict:
	base_pp = os.environ.get("CTRL83_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL83_BASE_PYTHONPATH to the ABI-65 src tree (its wheel = the venv's)")
	new_pp = os.environ.get("PYTHONPATH", H._here())
	old, new = _dump_in(base_pp, "off"), _dump_in(new_pp, "off")
	assert (old["_abi"], new["_abi"]) == (65, 66), (old["_abi"], new["_abi"])
	assert old["_src"] != new["_src"], "both dumps imported the same source tree"
	for key in old:
		if not key.startswith("_"):
			assert old[key] == new[key], f"off: {key} moved between ABI 65 and ABI 66"
	assert new["_anchor"] == "('true', 'true')", new["_anchor"]
	print(f"✓ flag off == ABI 65: names ({len(new['names'])}), thresholds ({len(new['thresholds'])}), token, ctbr, "
	      f"args ({len(new['args'])}), provenance, features ({len(new['features'])} steps), train digest "
	      f"{new['train_digest']}, scorer rows; --anchor-mass resolves to 'true'")
	return new


def test_anchor_true_spelled_out(off: dict):
	spelled = _dump_in(os.environ.get("PYTHONPATH", H._here()), "anchor_true")
	for key in H.BEHAVIOUR:
		assert spelled[key] == off[key], f"--anchor-mass true moved {key}"
	print("✓ --anchor-mass true: every behaviour key (provenance included) equal to flag off")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2])
		print(json.dumps(d))
		sys.exit(0)
	off = test_flag_off_is_abi65()
	test_anchor_true_spelled_out(off)
	print("\nCTRL-83: cf21 flag-off pin vs ABI 65 passed.")
