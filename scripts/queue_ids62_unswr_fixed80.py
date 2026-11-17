"""IDS62 — FIXED 80-bit width on unswr configuration C (redo of the VOID IDS61; Luiz 17/11/2026; the
GA drifted to <=64 bits under max_bits 80; docs/ids62_unswr_fixed80_prereg.md). Per seed r20421-23: clone C VERBATIM and
override exactly max_bits 64->80, min_bits 4->80, bits_grid [80]. Asserts the diff is those three keys, every key is in KNOWN_PARAMS, worker ABI 29.
Created pending -> experiments verified -> PATCH queued (runs next; everything else is paused).
Usage: python scripts/queue_ids62_unswr_fixed80.py [--dry-run]"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402
from queue_ids60_unswt_ce20 import diff  # noqa: E402

CONTROLS = {20421: 7567, 20422: 7568, 20423: 7569}
OVERRIDE = {"max_bits": 80, "min_bits": 80, "bits_grid": [80]}
WORKER_ABI = 29
PREFIX = "IDS62-unswr-WbCTRL-desir-BINARY-2m-max250-xo_blend_cluster-rate0.25-rklegacy-nb02-fixb80-encoff"


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
		assert sorted(d) == sorted(OVERRIDE), d
		Q.check(name, new)
		desc = (f"IDS62 (prereg docs/ids62_unswr_fixed80_prereg.md): fixed 80-bit width on unswr config C (redo of VOID IDS61); clone of {ctl} (r{seed}) "
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
			print("[dry]", name, template, params["min_bits"], params["max_bits"], params["bits_grid"], params["seed"])
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
