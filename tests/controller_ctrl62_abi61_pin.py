"""ABI-62 flags OFF == ABI 61, measured across the two builds (CTRL-69).

The CTRL-48 two-build probe (tests/controller_ctrl48_abi44_pin.py: grid ladder, 40 steps
of features, the Lamarckian write-back digest after one short DAgger, the scorer rows),
run in subprocesses on:
  * ABI 61: $CTRL62_BASE_PYTHONPATH (an ABI-61 wheel dir + the ABI-61 source tree);
  * ABI 62: this tree with whatever wheel is first on PYTHONPATH;
for every cascade layout an ABI-61 run can fly: no blocks, rate-I (all axes — the
ABI-62 default), rate-I + d-hat, and the CTRL-48 residual. Every dump must be EQUAL.

Run: CTRL62_BASE_PYTHONPATH=/tmp/ctrl61_site:/tmp/ctrl61_wt/src \\
     PYTHONPATH=/tmp/ctrl62_site:src python tests/controller_ctrl62_abi61_pin.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_ctrl48_abi44_pin as pin  # noqa: E402

VARIANTS = {"base": [], "rate_i": ["--obs-rate-i"], "cascade_blocks": ["--obs-rate-i", "--obs-cascade-dhat"],
            "dres": ["--obs-cascade-dhat-residual"]}


def _run(variant: str, pythonpath: str) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", variant], env=env,
	                   capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def test_flags_off_is_abi61():
	base_pp = os.environ.get("CTRL62_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL62_BASE_PYTHONPATH to an ABI-61 wheel dir + the ABI-61 src tree")
	here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
	new_pp = os.environ.get("PYTHONPATH", here)
	for variant in VARIANTS:
		old = _run(variant, base_pp)
		new = _run(variant, new_pp)
		for key in old:
			assert old[key] == new[key], f"{variant}: {key} moved between ABI 61 and ABI 62 (flags off)"
		print(f"✓ {variant}: thresholds ({len(new['thresholds'])}), features ({len(new['features'])} steps), "
		      f"train digest {new['train_digest']}, scorer rows identical to ABI 61")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		pin.VARIANTS.update(VARIANTS)
		with contextlib.redirect_stdout(io.StringIO()):
			d = pin.dump(sys.argv[2])
		print(json.dumps(d))
		sys.exit(0)
	test_flags_off_is_abi61()
	print("\nABI-62 flags-off pin vs ABI 61 passed.")
