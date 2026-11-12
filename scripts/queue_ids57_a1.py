"""IDS-57 A1 — pre-registered b64 arm on unswr (docs/ids57_a1_prereg.md, Luiz 12/11/2026).

Each arm flow clones ITS OWN seed's control (7509 / 7513 / 7517, unswr lattice encoff r20421-23) and changes
exactly one key: max_bits 34 -> 64. The script prints the diff and refuses to queue if the arm differs from its
control in anything else. Controls are NOT re-flown. Extension seeds r20419-20 (controls 7501 / 7505) only via
--extension, and only when the pre-registration's AMBIGUOUS rule fires.

Usage: python scripts/queue_ids57_a1.py [--dry-run] [--extension]"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_ids56_vote_share_ab as Q  # noqa: E402

MAX_BITS = 64
CONTROLS_N3 = {20421: 7509, 20422: 7513, 20423: 7517}
CONTROLS_EXT = {20419: 7501, 20420: 7505}
PREFIX = "IDS57-unswr-WbCTRL-desir-BINARY-2m-max250-b64-xo_blend_cluster-rate0.25-rklegacy-lattice-encoff"
# Asserted on the control, never overridden: a drift in a control's params cannot silently change the arm.
PINNED = {"ids_dataset": "unsw-nb15", "ids_split": "random_3way", "ids_classification": "binary",
          "memory_mode": "BINARY", "ids_single_cluster": False, "ids_receptive_fields": "off",
          "ids_write_rule": "lattice", "ids_encoding_dedupe": "off", "ids_feature_ranking": "legacy",
          "neuron_sample_rate": 0.25, "max_neurons": 250, "max_bits": 34,
          "ids_k_folds": 5, "ids_kfold_per_gen": 5, "ids_offspring_folds": "all",
          "fitness_aggregation": "desirability", "ga_generations": 250, "patience": 5, "balance_classes": True}


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


def main(dry_run: bool, extension: bool):
	db = sqlite3.connect(Q.DB, uri=True)
	have = {r[0] for r in db.execute("select name from flows")}
	for seed, control in (CONTROLS_EXT if extension else CONTROLS_N3).items():
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
			"description": f"IDS-57 A1 (pre-registered, docs/ids57_a1_prereg.md): clone {control} with max_bits "
			               f"34 -> {MAX_BITS} only; paired vs {control} (same seed, 34-bit lattice encoff)."})
		verify_experiments(f["id"])
		Q.call("PATCH", f"{Q.API}/{f['id']}", {"status": "queued"})
		print(f["id"], name)


if __name__ == "__main__":
	main("--dry-run" in sys.argv, "--extension" in sys.argv)
