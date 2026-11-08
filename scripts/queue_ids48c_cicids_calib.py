import json, sqlite3, time, urllib.request, ssl
DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API = "https://127.0.0.1:3000/api/flows"
CTX = ssl._create_unverified_context()
EXPS = [{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
        {"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"}]
def call(method, url, body=None):
	for _ in range(8):
		try:
			r = urllib.request.Request(url, method=method, headers={"Content-Type": "application/json"}, data=None if body is None else json.dumps(body).encode())
			return json.load(urllib.request.urlopen(r, context=CTX, timeout=60))
		except Exception as e:
			err = e; time.sleep(2)
	raise err
base = json.loads(sqlite3.connect(DB, uri=True).execute("select config_json from flows where id=7343").fetchone()[0])
print("base template", base["template"], "ranking/dedupe in base:", base["params"].get("ids_feature_ranking"), base["params"].get("ids_encoding_dedupe"))
for s in range(20419, 20424):
	for arm, extra in (("calibfull", {}), ("calibsub200k", {"ids_calib_sample": 200000})):
		p = dict(base["params"]); p.update({"seed": s, "ids_encoding_dedupe": "off", "ids_feature_ranking": "train_only"}); p.update(extra)
		name = (f"IDS52-cicids-CE20-zs-QUAD-1m-rktrain-encoff-calibfull-r{s}" if arm == "calibfull"
		        else f"IDS48c-cicids-CE20-zs-QUAD-1m-rktrain-encoff-calibsub200k-r{s}")
		f = call("POST", API, {"name": name, "description": "IDS-48 moved to cicids (Luiz 08/11): calib full vs 200k subsample, train_only ranking, dedupe off; the calibfull flows double as the IDS-52 dedupe control. Clone of 7343.",
		                       "config": {"template": base["template"], "params": p}, "experiments": EXPS})
		call("PATCH", f"{API}/{f['id']}", {"status": "queued"})
		print(f["id"], name)
