"""CTRL-84 (ABI 71): the cf21 FLAG-OFF pin vs ABI 69, measured across the two builds on the flown
l2b recipe — the CTRL-87 harness (tests/controller_ctrl87_flag_off.py → the CTRL-86 harness: the same
argv, shrink and dump), with no MLP head anywhere:
  ABI 69: $CTRL84_BASE_PYTHONPATH (the ABI-69 src tree; the ABI-69 wheel = the venv's);
  ABI 71: this tree with the ABI-71 wheel first on PYTHONPATH (e.g. an UNPACKED wheel).
Every dump key is EQUAL: feature names, fitted thresholds, calibration token, resolved ctbr, every
parsed argument, the provenance tokens, 40 steps of features, the Lamarckian write-back digest after
one short DAgger, and the GA-search scorer rows (default device: Metal unless WNN_CONTROLLER_GPU_EVAL=0).
The refactors this branch made in evaluator.py (packed_train_config / new_train_controller /
_metrics_from_scored) sit on exactly these paths. (The cargo twin: ctrl84_mlp_tests.rs.)

Run: CTRL84_BASE_PYTHONPATH=/Users/lacg/wnn/src \\
     PYTHONPATH=/tmp/ctrl84_site:src python tests/controller_ctrl84_flag_off.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_ctrl87_flag_off as H87  # noqa: E402


def _dump_in(pythonpath: str, cpu: bool = False) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	if cpu:
		env["WNN_CONTROLLER_GPU_EVAL"] = "0"
	r = subprocess.run([sys.executable, os.path.abspath(H87.__file__), "--dump", "off"], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"off @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def test_flag_off_is_abi69(cpu: bool) -> None:
	base_pp = os.environ.get("CTRL84_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL84_BASE_PYTHONPATH to the ABI-69 src tree (its wheel = the venv's)")
	new_pp = os.environ["PYTHONPATH"]
	old, new = _dump_in(base_pp, cpu), _dump_in(new_pp, cpu)
	assert (old["_abi"], new["_abi"]) == (69, 71), (old["_abi"], new["_abi"])
	assert old["_src"] != new["_src"], "both dumps imported the same source tree"
	moved = [k for k in old if not k.startswith("_") and old[k] != new[k]]
	assert not moved, f"flag off moved between ABI 69 and ABI 71: {moved}"
	print(f"✓ [{'cpu' if cpu else 'default device'}] flag off == ABI 69: {sorted(k for k in old if not k.startswith('_'))}; "
	      f"train digest {new['train_digest']}, scorer err {new['scorer_rows'][0]['mean_attitude_error_deg']}")


if __name__ == "__main__":
	test_flag_off_is_abi69(cpu=True)
	test_flag_off_is_abi69(cpu=False)
	print("\nCTRL-84: cf21 flag-off pin vs ABI 69 passed.")
