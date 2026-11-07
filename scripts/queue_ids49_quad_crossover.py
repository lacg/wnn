"""IDS-49 (Luiz 07/11/2026): QUAD × GA crossover rule on cicids + ciciot.
WHAT IT TESTS: under the QUAD production recipe, does blend_cluster or blend_neuron beat the default
  `inherit` crossover (ga_crossover_bits)? The 3 existing IDS47 QUAD seeds per dataset (r20419-21) ran
  `inherit` (param absent = default) and ARE the control — not re-flown (Luiz).
NEW FLOWS: clone of each IDS47 QUAD source flow (same dataset, seed, budget, fitness), ONLY
  ga_crossover_bits changed: 2 rules x 2 datasets x 3 seeds = 12 flows.
  IDS49-<ds>-QUAD-xo_<rule>-r<seed>
READ: per dataset, each rule vs inherit at its own best genome/threshold mode (Luiz 06/11 rule: each config
  at the mode it shines), n=3, F1/FPR/Acc mean±SD.
Usage: python scripts/queue_ids49_quad_crossover.py [--dry-run]. Idempotent (skips existing names).
"""
import json
import sys
import time

import requests

sys.path.insert(0, "/Users/lacg/wnn/scripts")
import queue_param_clone as q  # noqa: E402

DATASETS = {"cicids": "IDS47-cicids-CE20-zs-QUAD-1m-r%", "ciciot": "IDS47-ciciot-B15AC-desir-QUAD-1m-r%"}
SEEDS = (20419, 20420, 20421)
RULES = ("blend_cluster", "blend_neuron")


def source(con, like: str, seed: int):
	for name, cj in con.execute("SELECT name, config_json FROM flows WHERE name LIKE ? AND status != 'cancelled' ORDER BY id", (like,)):
		cfg = json.loads(cj)
		if int(cfg["params"]["seed"]) == seed:
			return name, cfg
	raise SystemExit(f"no source for {like} seed {seed}")


def main(dry: bool) -> None:
	con = q.ro()
	existing = {n for (n,) in con.execute("SELECT name FROM flows WHERE name LIKE 'IDS49-%' AND status != 'cancelled'")}
	made = []
	for ds, like in DATASETS.items():
		for seed in SEEDS:
			sname, cfg = source(con, like, seed)
			for rule in RULES:
				name = f"IDS49-{ds}-QUAD-xo_{rule}-r{seed}"
				if name in existing:
					print(f"  = skip {name}"); continue
				params = dict(cfg["params"]); params["ga_crossover_bits"] = rule
				body = {"name": name,
					"description": f"IDS-49 QUAD crossover: clone of {sname} (inherit control) with ga_crossover_bits={rule}.",
					"config": {"template": cfg.get("template", "ids-binary-2-phase"), "params": params},
					"experiments": q.EXPERIMENTS}
				if dry:
					print(f"  ~ {name} <- {sname}"); continue
				r = q.post_with_retry(f"{q.DASHBOARD}/api/flows", body)
				if r.status_code not in (200, 201):
					raise SystemExit(f"  x FAILED {name} ({r.status_code}) {r.text[:200]}")
				fid = r.json()["id"]
				for _ in range(5):
					if requests.patch(f"{q.DASHBOARD}/api/flows/{fid}", json={"status": "queued"}, verify=False, timeout=30).status_code == 200:
						break
					time.sleep(2)
				made.append((fid, name)); print(f"  + {fid} {name}")
	con.close()
	if made:
		con = q.ro()
		for fid, name in made:
			st, cj = con.execute("SELECT status, config_json FROM flows WHERE id=?", (fid,)).fetchone()
			ne = con.execute("SELECT COUNT(*) FROM experiments WHERE flow_id=?", (fid,)).fetchone()[0]
			rule = json.loads(cj)["params"].get("ga_crossover_bits")
			ok = ne == 2 and st in ("queued", "running") and name.endswith(f"xo_{rule}-r{json.loads(cj)['params']['seed']}")
			print(f"  {'ok' if ok else '!! VERIFY FAILED'} {fid} {name} status={st} exps={ne} xo={rule}")


if __name__ == "__main__":
	main("--dry-run" in sys.argv)
