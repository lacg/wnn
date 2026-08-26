"""Queue the IDS DESIRABILITY A/B (26/08/2026; docs/DESIRABILITY_FITNESS_SHAPES.md).

2 arms x 5 seeds = 10 flows, UNSW-NB15 temporal_3way quad 16b top20 — the cell
where a flow costs ~5 min, so the whole A/B is ~1 h of worker time. Production
Wb weights (.10/.20/.35/.35) on BOTH arms; the ONLY variable is the aggregation:

  IDSD-...-zsc-sNNNNN    fitness_aggregation=zscore        (shipped control)
  IDSD-...-desir-sNNNNN  fitness_aggregation=desirability  (ABI-8 worker wheel)

Under desirability the SAME numbers act as exponents (relative half-life
importance) — that is the point: the A/B isolates the aggregation, weight
questions come after (mirrors the controller stage-A A/B contract).

⚠️ DEPLOY GATE: desir flows CRASH on a pre-ABI-8 worker (wnn.accel has no
desirability_fitness_combine and the calculator refuses to silently drop it).
Run scripts/deploy_ids_desir_worker.sh at WORKER IDLE first; this script
refuses to queue if the installed wheel is stale.

PRE-REGISTERED READ (fixed now, before any flow runs): primary = val_cal
held-out F1 and FPR, paired per seed, control first in absolutes then deltas;
full Rule-7 five tables via the ids-security agent; winner = paired majority
across 5 seeds, read ONCE. NEVER report during-search k-fold numbers.
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
SEEDS = [20501, 20502, 20503, 20504, 20505]
ARMS = [("zsc", "zscore"), ("desir", "desirability")]
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
	"fitness_zrank_clamp": 3.0,
	"fitness_calculator": "harmonic_rank",
	# Production Wb weights — identical on both arms; aggregation is the only variable.
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


def worker_wheel_ready() -> bool:
	try:
		import ram_accelerator as a
		return getattr(a, "ABI_VERSION", 0) >= 8 and hasattr(a, "desirability_fitness_combine")
	except ImportError:
		return False


def main() -> int:
	if not worker_wheel_ready() and "--force" not in sys.argv:
		print("REFUSED: installed ram_accelerator lacks ABI 8 / desirability_fitness_combine.\n"
		      "Run scripts/deploy_ids_desir_worker.sh at worker idle first "
		      "(--force only if you know the worker venv differs from this one).")
		return 3
	con = ro()
	existing = {r[0] for r in con.execute("SELECT name FROM flows WHERE name LIKE 'IDSD-%'")}
	backlog = con.execute(
		"SELECT COUNT(*), COALESCE(MAX(id),0) FROM flows WHERE status IN ('queued','running')").fetchone()
	con.close()

	planned = [(arm, agg, seed) for seed in SEEDS for arm, agg in ARMS]  # seed-major, pairs adjacent
	print(f"IDSD A/B: {len(planned)} flows (2 arms x {len(SEEDS)} seeds), "
	      f"runs behind {backlog[0]} queued/running (max id {backlog[1]})")
	if existing:
		print(f"  skipping {len(existing)} already present")
	if "--dry-run" in sys.argv:
		print("DRY RUN — nothing created."); return 0

	created = []
	for arm, agg, seed in planned:
		name = f"IDSD-unswt-quad-16b-{arm}-s{seed}"
		if name in existing:
			continue
		p = dict(BASE_PARAMS)
		p["fitness_aggregation"] = agg
		p["seed"] = seed
		body = {
			"name": name,
			"description": f"IDS desirability A/B arm {arm} ({agg}), seed {seed} — "
			               f"Wb weights on both arms; aggregation is the only variable",
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
		      and q.get("fitness_aggregation") in ("zscore", "desirability"))
		if not ok:
			print(f"  ! VERIFY FAILED {name}: status={st} exps={ne} arch={at}"); bad += 1
	con.close()
	print("ALL VERIFIED" if not bad else f"{bad} flows FAILED verification")
	return 1 if bad else 0


if __name__ == "__main__":
	sys.exit(main())
