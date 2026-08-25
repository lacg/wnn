"""Queue the MCS multiclass screening trio (docs/MULTICLASS_DESIGN.md §6 step 2).

3 arms x 5 seeds = 15 flows, UNSW-NB15 temporal_3way quad 16b top20, IDSZ caps.
The ONLY variable across arms is ids_classification (+ its required
ids_single_cluster setting) — production Wb weights on every arm:

  MCS-multi    ids_classification=multi         single_cluster=False  K=10 clusters
  MCS-hier     ids_classification=hierarchical  single_cluster=False  S0 binary -> S1 9-class
  MCS-binary   ids_classification=binary        single_cluster=True   production control

Per CLAUDE.md Rule 2: created through the dashboard POST /api/flows, never by
SQL insert. Flows land at the FIFO tail — they run only after the IDSX cohort
drains (admit() takes min(id)). Idempotent: existing MCS- names are skipped.

PRE-REGISTERED READ (fixed before any flow runs): primary = macro-F1 and
benign-FPR on the val-calibrated held-out modes; the per-class recall table is
mandatory in the readout. Comparators: XGB multiclass macro-F1 0.523 /
benign-FPR 23.1% on the identical protocol (docs/multiclass_baselines/).
"""
import json
import sqlite3
import sys
import time

import requests
import urllib3

urllib3.disable_warnings()

DASHBOARD = "https://localhost:3000"
DB_PATH = "/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db"
SEEDS = [20401, 20402, 20403, 20404, 20405]
ARMS = [
	("multi", "multi", False),
	("hier", "hierarchical", False),
	("binary", "binary", True),
]
EXPERIMENTS = [
	{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
	{"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"},
]

BASE_PARAMS = {
	"architecture_type": "ids",
	"ids_dataset": "unsw-nb15",
	"ids_split": "temporal_3way",
	"ids_feature_selection": "top20",
	"ids_n_bits": 16,
	"ids_k_folds": 5,
	"ids_kfold_per_gen": 5,
	"ids_num_parts": 5,
	"ids_val_fraction": 0.25,
	"ids_encoded_storage": "memory",
	"fitness_aggregation": "zscore",
	"fitness_zrank_clamp": 3.0,
	"fitness_calculator": "harmonic_rank",
	# Production Wb weights — identical on every arm so classification is the only variable.
	"fitness_weight_ce": 0.10,
	"fitness_weight_acc": 0.20,
	"fitness_weight_f1": 0.35,
	"fitness_weight_fpr": 0.35,
	"fitness_percentile": 0.75,
	"ga_generations": 250,
	"patience": 5,
	"population_size": 50,
	"min_neurons": 5, "max_neurons": 500,
	"min_bits": 4, "max_bits": 34,
	"neuron_sample_rate": 0.25,
	"wnn_order_independent_train": True,
	"balance_classes": True,
	"assortative_mating_ratio": 0.85,
	"cluster_crossover_ratio": 0.5,
	"pool_shuffle_ratio": 0.8,
	"neighbors_per_iter": 50,
	"adaptation_iterations": 50,
	"phase_order": "neurons_first",
	"context_size": 4,
	"threshold_start": 0, "threshold_step": 1,
	"min_accuracy_floor": 0,
}


def ro():
	return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def main() -> int:
	con = ro()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'MCS-%'")}
	backlog = con.execute(
		"SELECT COUNT(*), COALESCE(MAX(id),0) FROM flows WHERE status IN ('queued','running')").fetchone()
	con.close()

	planned = [(arm, cls, single, seed) for seed in SEEDS for arm, cls, single in ARMS]  # seed-major
	print(f"MCS screening: {len(planned)} flows (3 arms x {len(SEEDS)} seeds), "
	      f"runs LAST behind {backlog[0]} queued/running (max id {backlog[1]})")
	if existing:
		print(f"  skipping {len(existing)} already present")
	if "--dry-run" in sys.argv:
		print("DRY RUN — nothing created."); return 0

	created = []
	for arm, cls, single, seed in planned:
		name = f"MCS-unswt-quad-16b-{arm}-s{seed}"
		if name in existing:
			continue
		p = dict(BASE_PARAMS)
		p["ids_classification"] = cls
		p["ids_single_cluster"] = single
		p["seed"] = seed
		body = {
			"name": name,
			"description": f"Multiclass screening arm {arm} ({cls}), seed {seed} — "
			               f"reviewer-A ask; only variable across arms is ids_classification",
			"config": {"template": "ids-binary-2-phase", "params": p},
			"experiments": EXPERIMENTS,
		}
		r = requests.post(f"{DASHBOARD}/api/flows", json=body, verify=False, timeout=30)
		if r.status_code not in (200, 201):
			print(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}"); return 2
		created.append((r.json()["id"], name))
		print(f"  + {r.json()['id']:>5}  {name}")
		time.sleep(0.2)

	print(f"flipping {len(created)} pending -> queued (FIFO tail)...")
	for fid, _ in created:
		requests.post(f"{DASHBOARD}/api/flows/{fid}/restart", json={}, verify=False, timeout=15)
		time.sleep(0.12)

	con = ro(); bad = 0
	for fid, name in created:
		st = con.execute("SELECT status FROM flows WHERE id=?", (fid,)).fetchone()[0]
		ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
		at = {r[0] for r in con.execute("SELECT architecture_type FROM experiments WHERE flow_id=?", (fid,))}
		q = json.loads(con.execute("SELECT config_json FROM flows WHERE id=?", (fid,)).fetchone()[0])["params"]
		ok = (ne == 2 and st == "queued" and at == {"ids"}
		      and q.get("patience") == 5 and q.get("ga_generations") == 250
		      and q.get("fitness_aggregation") == "zscore"
		      and q.get("ids_classification") in ("multi", "hierarchical", "binary"))
		if not ok:
			print(f"  x id={fid} {name} status={st} exps={ne} arch={at} "
			      f"cls={q.get('ids_classification')}"); bad += 1
	ahead = con.execute(
		"SELECT COUNT(*) FROM flows WHERE status IN ('queued','running') AND id<?",
		(min(f for f, _ in created),)).fetchone()[0] if created else 0
	con.close()
	print(f"verified {len(created) - bad}/{len(created)} OK; {ahead} flows ahead in the FIFO")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
