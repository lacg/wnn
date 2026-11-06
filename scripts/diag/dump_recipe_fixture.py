"""CTRL-5 lag imitation probe (06/11/2026): dump the EXACT recipe controller
construction + RewardGatedConfigPacked that a phased_ga run hands
dagger_train_batch_inplace for its first (grid) batch, then exit before any
training. Writes a flat text fixture the Rust #[ignore] test
imitation_probe::diag_imitation_probe_recipe reads (WNN_IMIT_FIXTURE=<path>).

Usage: python scripts/diag/dump_recipe_fixture.py OUT.txt <phased_ga argv...>
Runs no training (threshold calibration rollouts only); not `-m phased_ga`.
"""
import os
import sys

out = sys.argv[1]
sys.argv = ["phased_ga"] + sys.argv[2:]
import wnn.control._accel as A  # noqa: E402


def _fmt(v):
	if isinstance(v, bool):
		return "true" if v else "false"
	if v is None:
		return "none"
	if isinstance(v, (bytes, bytearray)):
		return " ".join(str(x) for x in v)
	if isinstance(v, (list, tuple)):
		return " ".join(_fmt(x) for x in v)
	return repr(v)


def hook(**kw):
	cfg = kw.pop("cfg")
	kw.pop("init_cells_per_genome")
	with open(out, "w") as f:
		for k in sorted(dir(cfg)):
			if k.startswith("_"):
				continue
			v = getattr(cfg, k)
			if callable(v):
				continue
			f.write(f"cfg.{k}={_fmt(v)}\n")
		for k, v in sorted(kw.items()):
			if k.endswith("_per_genome") or k == "fold_seeds":
				v = v[0]
			f.write(f"kw.{k}={_fmt(v)}\n")
	print(f"[dump_recipe_fixture] wrote {out}", flush=True)
	os._exit(0)


A.dagger_train_batch_inplace = hook
from wnn.control import phased_ga  # noqa: E402

phased_ga.main()
