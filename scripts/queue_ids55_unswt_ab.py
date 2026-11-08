"""IDS-55 receptive fields, quick A/B on unswt BINARY (Luiz 08/11): base 7125 (unswt BINARY lattice 2m max250
blend_neuron rate 0.25), seeds 20419-20423, paired off vs family (ids_rf_cross 0.2). Dedupe off, legacy ranking
(unswt has no leak), all set explicitly. Run on worker ABI 27 for both arms (7125 itself ran on 26: never pair)."""
import json, sqlite3, ssl, time, urllib.request

DB = "file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro"
API = "https://127.0.0.1:3000/api/flows"
CTX = ssl._create_unverified_context()
EXPS = [{"name": "Grid Search (neurons x bits)", "experiment_type": "grid_search", "phase_type": "grid_search"},
        {"name": "GA Neurons", "experiment_type": "ga", "phase_type": "ga_neurons"}]


def call(method, url, body=None):
	err = None
	for _ in range(8):
		try:
			r = urllib.request.Request(url, method=method, headers={"Content-Type": "application/json"},
			                           data=None if body is None else json.dumps(body).encode())
			return json.load(urllib.request.urlopen(r, context=CTX, timeout=60))
		except Exception as e:
			err = e
			time.sleep(2)
	raise err


def main():
	db = sqlite3.connect(DB, uri=True)
	cfg = json.loads(db.execute("select config_json from flows where id=7125").fetchone()[0])
	have = {r[0] for r in db.execute("select name from flows")}
	for s in range(20419, 20424):
		for arm, extra in (("rfoff", {"ids_receptive_fields": "off"}), ("rffam02", {"ids_receptive_fields": "family", "ids_rf_cross": 0.2})):
			name = f"IDS55-unswt-BINARY-lattice-2m-max250-xo_blend_neuron-16bWb-rate0.25-encoff-{arm}-r{s}"
			if name in have:
				continue
			p = dict(cfg["params"])
			p.update({"seed": s, "ids_encoding_dedupe": "off", "ids_feature_ranking": "legacy"})
			p.update(extra)
			f = call("POST", API, {"name": name, "description": "IDS-55 receptive fields quick A/B (Luiz 08/11), clone of 7125, worker ABI 27.",
			                       "config": {"template": cfg["template"], "params": p}, "experiments": EXPS})
			call("PATCH", f"{API}/{f['id']}", {"status": "queued"})
			print(f["id"], name)


if __name__ == "__main__":
	main()
