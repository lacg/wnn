"""IDS60 — CE20 fitness weights on the adopted unswt recipe (Luiz approved 16/11/2026; docs/ids60_unswt_ce20_prereg.md).
Per seed: clone the IDS-56 ms01-encon control VERBATIM and override exactly the four fitness weights
(ce/acc/f1/fpr 0.10/0.20/0.35/0.35 -> 0.20/0.10/0.30/0.40, the IDSZ CE20 definition). Asserts the diff is exactly
those four keys, every key is in KNOWN_PARAMS, and that the worker ABI is 29. Created pending -> experiments
verified -> PATCH queued (runs next under FIFO; the paused flows stay paused).
Usage: python scripts/queue_ids60_unswt_ce20.py [--dry-run]"""
import json, sqlite3, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

CONTROLS = {20419: 7484, 20420: 7488, 20421: 7492, 20422: 7496, 20423: 7500}
CE20 = {"fitness_weight_ce": 0.20, "fitness_weight_acc": 0.10, "fitness_weight_f1": 0.30, "fitness_weight_fpr": 0.40}
WORKER_ABI = 29
PREFIX = "IDS60-unswt-BINARY-2m-max250-xo_blend_neuron-16bCE20-rate0.25-ms01-encon"


def diff(a: dict, b: dict) -> dict:
	return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def plan(db) -> list[tuple[str, str, dict, str]]:
	out = []
	for seed, ctl in CONTROLS.items():
		name, cfg = db.execute("select name, config_json from flows where id=?", (ctl,)).fetchone()
		cfg = json.loads(cfg)
		base = cfg["params"]
		assert name.endswith(f"-ms01-encon-r{seed}") and base["seed"] == seed, (ctl, name)
		assert base["ids_split"] == "temporal_3way" and base["ids_k_folds"] == 5 and base["ids_kfold_per_gen"] == 5
		assert base["ids_write_rule"] == "vote" and base["ids_vote_share"] == 0.1 and base["ids_encoding_dedupe"] == "on"
		new = {**base, **CE20}
		d = diff(base, new)
		assert sorted(d) == sorted(CE20), d
		Q.check(name, new)
		desc = (f"IDS60 (prereg docs/ids60_unswt_ce20_prereg.md): CE20 fitness weights on the unswt ms01-encon recipe; "
			f"clone of control {ctl} (r{seed}) with only the 4 fitness weights changed {json.dumps(d)}. Worker ABI {WORKER_ABI}.")
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
			print("[dry]", name, template, {k: params[k] for k in CE20})
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
