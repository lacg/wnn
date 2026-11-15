"""IDS59 — pre-registered A1 (64 bits) on cicids (docs/ids59_cicids_b64_prereg.md, Luiz 15/11/2026).

Each arm flow clones ITS OWN seed's control (7531 / 7533 / 7535, IDS56C cicids lattice encon r20419-21) and changes
exactly one key: max_bits 34 -> 64. The script prints the diff and refuses to queue if the arm differs from its
control in anything else. Controls are NOT re-flown. Flows are created QUEUED (Luiz 15/11).

Usage: python scripts/queue_ids59_cicids_b64.py [--dry-run]"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

MAX_BITS = 64
CONTROLS = {20419: 7531, 20420: 7533, 20421: 7535}
PREFIX = "IDS59-cicids-CE20-zs-BINARY-2m-max250-xo_blend_neuron-rate0.25-lattice-b64-encon"
# Asserted on the control, never overridden: a drift in a control's params cannot silently change the arm.
PINNED = {"ids_dataset": "cicids2017", "ids_split": "random_3way", "ids_classification": "binary",
          "memory_mode": "BINARY", "ids_single_cluster": False, "ids_receptive_fields": "off",
          "ids_write_rule": "lattice", "ids_vote_share": None, "ids_encoding_dedupe": "on",
          "ids_feature_ranking": "train_only", "neuron_sample_rate": 0.25, "max_neurons": 250, "max_bits": 34,
          "ids_n_bits": 96, "ids_encoded_storage": "memory", "ids_k_folds": 5, "ids_kfold_per_gen": 5,
          "ids_offspring_folds": "all", "fitness_aggregation": "zscore", "ga_crossover_bits": "blend_neuron",
          "ga_generations": 250, "patience": 5, "population_size": 50, "balance_classes": True}


def control_params(db, flow_id: int, seed: int) -> tuple[str, dict]:
	cfg = json.loads(db.execute("select config_json from flows where id=?", (flow_id,)).fetchone()[0])
	p = cfg["params"]
	bad = {k: (p.get(k), v) for k, v in PINNED.items() if p.get(k) != v}
	if bad or int(p["seed"]) != seed:
		raise SystemExit(f"control {flow_id}: params drifted from the pre-registration {bad} seed={p.get('seed')}")
	return cfg["template"], p


def one_key_diff(name: str, arm: dict, control: dict):
	diff = {k: (control.get(k), arm.get(k)) for k in set(arm) | set(control) if arm.get(k) != control.get(k)}
	print(f"  diff vs control: {diff}")
	if diff != {"max_bits": (34, MAX_BITS)}:
		raise SystemExit(f"{name}: arm must differ from its control in max_bits only")


def verify_experiments(flow_id: int):
	db = sqlite3.connect(Q.DB, uri=True)
	n = db.execute("select count(*) from experiments where flow_id=?", (flow_id,)).fetchone()[0]
	if n != len(Q.EXPS):
		raise SystemExit(f"flow {flow_id}: {n} experiments, expected {len(Q.EXPS)} — do NOT let it run")


def main(dry_run: bool):
	db = sqlite3.connect(Q.DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	if not dry_run:
		import ram_accelerator
		if int(ram_accelerator.ABI_VERSION) != 29:  # prereg §5: arm must share its controls' ABI (7531 = 29)
			raise SystemExit(f"worker ABI {ram_accelerator.ABI_VERSION} != 29: stop and ask Luiz")
	for seed, control in CONTROLS.items():
		name = f"{PREFIX}-r{seed}"
		if name in have:
			print(f"[skip] {name} exists")
			continue
		template, cp = control_params(db, control, seed)
		p = {**cp, "max_bits": MAX_BITS}
		print(f"{'[dry] ' if dry_run else ''}{name}  control {control}")
		Q.check(name, p)
		one_key_diff(name, p, cp)
		if dry_run:
			continue
		f = Q.call("POST", Q.API, {"name": name, "config": {"template": template, "params": p}, "experiments": Q.EXPS,
			"description": f"IDS59 (pre-registered, docs/ids59_cicids_b64_prereg.md): clone {control} with max_bits "
			               f"34 -> {MAX_BITS} only; paired vs {control} (same seed, 34-bit lattice encon)."})
		verify_experiments(f["id"])
		Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
		print(f["id"], name)


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
