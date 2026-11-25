"""IDS-80 TEST read AFTER the locked VAL verdict (FAIL, commit 80026ef4). Same 10 random genomes per arm
(seeds 1000-1009) as gate_hybrid.py. Reporting cell: train on TRAIN, margin_val_cal tau fitted on VAL, applied once
to TEST (the worker's evaluate_multiclass_at_thresholds, Protocol v2)."""
import sys, json, gc, sqlite3, time
sys.path.insert(0, "/Users/lacg/wnn/logs/ids/ids80")
sys.argv = [sys.argv[0]]
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("G", "/Users/lacg/wnn/logs/ids/ids80/gate_hybrid.py")
src = open(spec.origin).read().split('out = open(')[0]  # defs only, no VAL loop
G = {}
exec(compile(src, spec.origin, "exec"), G)
H, factory, db, draw, HYBRID, SEEDS = G["H"], G["factory"], G["db"], G["draw"], G["HYBRID"], G["SEEDS"]
out = open("/Users/lacg/wnn/logs/ids/ids80/gate_hybrid_test.jsonl", "a")
for arm, flow, hyb in (("E", 7591, False), ("H", 7591, True), ("B", 7594, False)):
	HYBRID["on"] = hyb
	params = H.flow_params(db, flow)
	dataset = factory.load(H.settings_for(params))
	T = int(dataset.X_train.total_bits)
	print(f"[ARM] {arm} total_bits={T}", flush=True)
	H.apply_production_env(params); s = H.settings_for(params)
	opt, ev = factory.flat(dataset, s)
	for seed in SEEDS:
		t0 = time.time(); g = draw(seed, T)
		res = ev.evaluate_multiclass_at_thresholds(g)
		m = res["modes"]["margin_val_cal"]; fp = res["footprint"]
		rec = dict(arm=arm, bits=T, seed=seed, secs=round(time.time() - t0, 1), partition="test", mode="margin_val_cal",
			tau=m["tau"], f1=m["macro_f1"] * 100, fpr=m["fpr"] * 100, acc=m["acc"] * 100, confusion=m["confusion"],
			logical_cells=fp.logical_cells, logical_neurons=fp.logical_neurons)
		print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
	del opt, ev, dataset; gc.collect()
