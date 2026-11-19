"""Fixed-width redo of unswr configuration C: clone C VERBATIM per seed r20421-23 and override exactly
max_bits 64->W, min_bits 4->W, bits_grid [W]. Asserts the diff is those three keys, every key is in KNOWN_PARAMS,
worker ABI 29. Created pending -> experiments verified -> PATCH queued.
Users: IDS62 (W=80, docs/ids62_unswr_fixed80_prereg.md), IDS63 (W=96, docs/ids63_unswr_fixed96_prereg.md).
Usage: python scripts/queue_ids_unswr_fixed_width.py --ids 63 --width 96 [--dry-run]"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402
from queue_ids60_unswt_ce20 import diff  # noqa: E402

CONTROLS = {20421: 7567, 20422: 7568, 20423: 7569}
WORKER_ABI = 29
PREFIX = "IDS{ids}-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb{width}-encoff"
PREREG = {62: "docs/ids62_unswr_fixed80_prereg.md", 63: "docs/ids63_unswr_fixed96_prereg.md"}


def control_params(db, seed: int, ctl: int) -> tuple[dict, str]:
	"""Load control C for one seed and assert it is the unswr configuration C every fixed-width arm clones."""
	name, cfg = db.execute("select name, config_json from flows where id=?", (ctl,)).fetchone()
	cfg = json.loads(cfg)
	base = cfg["params"]
	assert name.startswith("IDS58-unswr-") and name.endswith(f"-nb02-b64-encoff-r{seed}") and base["seed"] == seed
	assert base["ids_split"] == "random_3way" and base["ids_k_folds"] == 5 and base["ids_kfold_per_gen"] == 5
	assert base["ids_write_rule"] == "vote" and base["ids_vote_share"] == 0.02 and base["max_bits"] == 64
	return base, cfg["template"]


def plan(db, ids: int, width: int) -> list[tuple[str, str, dict, str]]:
	override = {"max_bits": width, "min_bits": width, "bits_grid": [width]}
	out = []
	for seed, ctl in CONTROLS.items():
		base, template = control_params(db, seed, ctl)
		new = {**base, **override}
		d = diff(base, new)
		assert sorted(d) == sorted(override), d
		Q.check(f"IDS{ids}", new)
		desc = (f"IDS{ids} (prereg {PREREG[ids]}): fixed {width}-bit width on unswr config C; clone of {ctl} (r{seed}) "
			f"with only {json.dumps(d)}. Worker ABI {WORKER_ABI}.")
		out.append((f"{PREFIX.format(ids=ids, width=width)}-r{seed}", template, new, desc))
	return out


def create(db, name: str, template: str, params: dict, desc: str) -> int:
	f = Q.call("POST", Q.API, {"name": name, "description": desc,
		"config": {"template": template, "params": params}, "experiments": Q.EXPS})
	n = db.execute("select count(*) from experiments where flow_id=?", (f["id"],)).fetchone()[0]
	if n != len(Q.EXPS):
		raise SystemExit(f"flow {f['id']}: {n} experiments, expected {len(Q.EXPS)}; left pending")
	Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
	return f["id"]


def main(ids: int, width: int, dry_run: bool):
	import ram_accelerator
	if int(ram_accelerator.ABI_VERSION) != WORKER_ABI:
		raise SystemExit(f"installed ABI {ram_accelerator.ABI_VERSION} != {WORKER_ABI}")
	db = sqlite3.connect(Q.DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	for name, template, params, desc in plan(db, ids, width):
		if name in have:
			print("exists", name)
		elif dry_run:
			print("[dry]", name, template, params["min_bits"], params["max_bits"], params["bits_grid"], params["seed"])
		else:
			print(create(db, name, template, params, desc), name, "queued")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--ids", type=int, required=True, choices=sorted(PREREG))
	ap.add_argument("--width", type=int, required=True)
	ap.add_argument("--dry-run", action="store_true")
	a = ap.parse_args()
	main(a.ids, a.width, a.dry_run)
