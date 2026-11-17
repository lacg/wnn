"""IDS61 — max_bits 64 -> 80 on the adopted unswr configuration C (vote nb02 + 64b; Luiz approved 16/11/2026 after the
VAL widening gate passed; docs/ids61_unswr_b80_prereg.md). Per seed r20421-23: clone the IDS58 C flow VERBATIM and
override exactly `max_bits` 64 -> 80. Asserts the diff is that one key, every key is in KNOWN_PARAMS, worker ABI 29.
Created pending -> experiments verified -> PATCH queued (behind IDS60 7577-7581 under FIFO).
Usage: python scripts/queue_ids61_unswr_b80.py [--dry-run]"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402
from queue_ids60_unswt_ce20 import diff  # noqa: E402

CONTROLS = {20421: 7567, 20422: 7568, 20423: 7569}
OVERRIDE = {"max_bits": 80}
WORKER_ABI = 29
PREFIX = "IDS61-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-b80-encoff"


def plan(db):
	out = []
	for seed, ctl in CONTROLS.items():
		name, cfg = db.execute("select name, config_json from flows where id=?", (ctl,)).fetchone()
		cfg = json.loads(cfg)
		base = cfg["params"]
		assert name.startswith("IDS58-unswr-") and name.endswith(f"-nb02-b64-encoff-r{seed}") and base["seed"] == seed
		assert base["ids_split"] == "random_3way" and base["ids_k_folds"] == 5 and base["ids_kfold_per_gen"] == 5
		assert base["ids_write_rule"] == "vote" and base["ids_vote_share"] == 0.02 and base["max_bits"] == 64
		new = {**base, **OVERRIDE}
		d = diff(base, new)
		assert sorted(d) == ["max_bits"], d
		Q.check(name, new)
		desc = (f"IDS61 (prereg docs/ids61_unswr_b80_prereg.md): max_bits 80 on unswr config C; clone of {ctl} (r{seed}) "
			f"with only {json.dumps(d)}. Worker ABI {WORKER_ABI}.")
		out.append((f"{PREFIX}-r{seed}", cfg["template"], new, desc))
	return out


def main(dry_run: bool):
	import ram_accelerator
	if int(ram_accelerator.ABI_VERSION) != WORKER_ABI:
		raise SystemExit(f"installed ABI {ram_accelerator.ABI_VERSION} != {WORKER_ABI}")
	db = sqlite3.connect(Q.DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	for name, template, params, desc in plan(db):
		if name in have:
			print("exists", name)
			continue
		if dry_run:
			print("[dry]", name, template, params["max_bits"], params["seed"])
			continue
		f = Q.call("POST", Q.API, {"name": name, "description": desc,
			"config": {"template": template, "params": params}, "experiments": Q.EXPS})
		n = db.execute("select count(*) from experiments where flow_id=?", (f["id"],)).fetchone()[0]
		if n != len(Q.EXPS):
			raise SystemExit(f"flow {f['id']}: {n} experiments, expected {len(Q.EXPS)}; left pending")
		Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
		print(f["id"], name, "queued")


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
