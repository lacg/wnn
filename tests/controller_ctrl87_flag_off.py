"""CTRL-87 (ABI 68): the cf21 FLAG-OFF pin vs the PREVIOUS build (ABI 66 when written), measured across the two builds on the
flown l2b recipe — the CTRL-86 harness (tests/controller_ctrl86_flag_off.py: the same argv, the
same shrink, the same dump), re-pointed two ABIs up (67 is the per-bank-width branch's):
  previous: $CTRL87_BASE_PYTHONPATH (the previous src tree; its wheel = the venv's, or first on the
            path); $CTRL87_BASE_ABI pins which ABI that must be (optional);
  this:     this tree with its own wheel first on PYTHONPATH (ABI = this tree's EXPECTED_ABI).
Every dump is EQUAL with neither --train-motor-asym-mag nor --train-motor-lag-range: feature
names, fitted thresholds, calibration token, resolved ctbr, every parsed argument (minus the two
new keys, which must resolve to None), the provenance tokens, 40 steps of features, the
Lamarckian write-back digest after one short DAgger and the GA-search scorer rows (default
device: Metal unless WNN_CONTROLLER_GPU_EVAL=0).

A second pair spells the spread out AT the nominal values (--train-motor-asym-mag 0.2
--train-motor-lag-range 0.20 0.20) and compares it with flag-off on the CPU scorer: every
behaviour key equal (the Metal FC_EP_LAG specialization agrees to ~1e-7, pinned in cargo:
ctrl87_train_spread_tests::ctrl87_degenerate_range_is_the_fixed_lag), provenance records the
spelling (train_lag=uniform:0.2,0.2) and nothing else.

Run: CTRL87_BASE_PYTHONPATH=/tmp/ctrl87_base/src \\
     PYTHONPATH=/tmp/ctrl87_site:src python tests/controller_ctrl87_flag_off.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controller_ctrl86_flag_off as H  # noqa: E402  the CTRL-86 harness (argv, shrink, dump)

VARIANTS = {"off": [], "nominal": ["--train-motor-asym-mag", "0.2", "--train-motor-lag-range", "0.20", "0.20"],
            "spread": ["--train-motor-asym-mag", "0.30", "--train-motor-lag-range", "0.15", "0.30"]}
NEW_KEYS = ("anchor_mass", "train_motor_asym_mag", "train_motor_lag_range",
            # later arguments a previous build may lack (they must resolve to None here).
            "train_inertia_jitter", "train_latency_range", "train_mass_jitter", "train_noise_scale",
            "train_motor_asym_range")


def dump(variant: str) -> dict:
	H.VARIANTS = {variant: VARIANTS[variant]}
	H.NEW_KEYS = H.NEW_KEYS + NEW_KEYS
	d = H.dump(variant)
	args, ec = H._args(VARIANTS[variant])
	d["_spread"] = repr((getattr(ec, "train_motor_asym_mag", None), getattr(ec, "train_motor_lag_range", None)))
	try:
		from wnn.control.evaluator import train_asym_token, train_lag_token
		d["prov"] = {**d["prov"], "train_asym": train_asym_token(ec)}
		if train_lag_token(ec):
			d["prov"]["train_lag"] = train_lag_token(ec)
	except ImportError:  # the ABI-66 tree has no train_lag_token
		from wnn.control.evaluator import train_asym_token
		d["prov"] = {**d["prov"], "train_asym": train_asym_token(ec)}
	return d


def _dump_in(pythonpath: str, variant: str, cpu: bool = False) -> dict:
	env = dict(os.environ, PYTHONPATH=pythonpath)
	if cpu:
		env["WNN_CONTROLLER_GPU_EVAL"] = "0"
	r = subprocess.run([sys.executable, os.path.abspath(__file__), "--dump", variant], env=env,
	                   capture_output=True, text=True, cwd="/tmp")
	if r.returncode != 0:
		raise RuntimeError(f"{variant} @ {pythonpath}: rc={r.returncode}\n{r.stderr[-3000:]}")
	return json.loads(r.stdout.strip().splitlines()[-1])


def test_flag_off_is_previous_build() -> dict:
	from controller_ctrl89_flag_off import previous_build_pair
	base_pp = os.environ.get("CTRL87_BASE_PYTHONPATH")
	if not base_pp:
		raise SystemExit("set CTRL87_BASE_PYTHONPATH to the previous build's src tree (its wheel = the venv's)")
	new_pp = os.environ.get("PYTHONPATH", H._here())
	old, new = _dump_in(base_pp, "off"), _dump_in(new_pp, "off")
	o, n = previous_build_pair(old, new, "CTRL87_BASE_ABI")
	for key in old:
		if not key.startswith("_"):
			assert old[key] == new[key], f"off: {key} moved between ABI {o} and ABI {n}"
	assert new["_spread"] == "(None, None)", new["_spread"]
	assert new["_new"]["train_motor_asym_mag"] == "None" and new["_new"]["train_motor_lag_range"] == "None", new["_new"]
	print(f"✓ flag off == ABI {o} (this tree ABI {n}): names ({len(new['names'])}), thresholds ({len(new['thresholds'])}), token, ctbr, "
	      f"args ({len(new['args'])}), provenance {new['prov'].get('train_asym')}, features ({len(new['features'])} steps), "
	      f"train digest {new['train_digest']}, scorer rows {new['scorer_rows'][0]['mean_attitude_error_deg']}; "
	      f"both new flags resolve to None")
	return new


def test_nominal_spelled_out():
	pp = os.environ.get("PYTHONPATH", H._here())
	off, spelled = _dump_in(pp, "off", cpu=True), _dump_in(pp, "nominal", cpu=True)
	for key in H.BEHAVIOUR:
		if key != "prov":
			assert spelled[key] == off[key], f"the spread spelled out at nominal moved {key}"
	assert spelled["prov"] == {**off["prov"], "train_lag": "uniform:0.2,0.2"}, spelled["prov"]
	assert spelled["_spread"] == "(0.2, (0.2, 0.2))", spelled["_spread"]
	print("✓ --train-motor-asym-mag 0.2 --train-motor-lag-range 0.20 0.20 (CPU scorer): every behaviour key equal "
	      "to flag off; provenance adds train_lag=uniform:0.2,0.2 only")


def test_spread_reaches_every_host(off: dict):
	"""The l2bnmw spread end to end through the wheel: it moves the calibration (thresholds), the
	trainer digest and the search scorer row, and the CPU and Metal search rows agree (CTRL-54)."""
	pp = os.environ.get("PYTHONPATH", H._here())
	gpu, cpu = _dump_in(pp, "spread"), _dump_in(pp, "spread", cpu=True)
	for key in ("thresholds", "train_digest", "scorer_rows"):
		assert gpu[key] != off[key], f"the spread did not move {key}"
	assert gpu["thresholds"] == cpu["thresholds"] and gpu["train_digest"] == cpu["train_digest"]
	for k in ("mean_attitude_error_deg", "stable_rate", "mean_altitude_error_m"):
		g, c = float(gpu["scorer_rows"][0][k]), float(cpu["scorer_rows"][0][k])
		assert abs(g - c) <= max(1e-3, 2e-2 * abs(c)), f"search row {k}: gpu {g} cpu {c}"
	assert gpu["prov"]["train_asym"] == "per_episode:0.3" and gpu["prov"]["train_lag"] == "uniform:0.15,0.3"
	print(f"✓ spread (wear ±0.30, lag U(0.15, 0.30)): thresholds, train digest and search row move; search err "
	      f"gpu {float(gpu['scorer_rows'][0]['mean_attitude_error_deg']):.4f} cpu "
	      f"{float(cpu['scorer_rows'][0]['mean_attitude_error_deg']):.4f} (off {float(off['scorer_rows'][0]['mean_attitude_error_deg']):.4f})")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--dump":
		import contextlib, io
		with contextlib.redirect_stdout(io.StringIO()):
			d = dump(sys.argv[2])
		print(json.dumps(d))
		sys.exit(0)
	off = test_flag_off_is_previous_build()
	test_nominal_spelled_out()
	test_spread_reaches_every_host(off)
	print("\nCTRL-87: cf21 flag-off pin vs the previous build passed.")
