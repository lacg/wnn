import json, numpy as np
OUT="/Users/lacg/wnn/logs/ids/ids77"
w=[json.loads(l) for l in open(f"{OUT}/wnn_val_diag.jsonl")]
r=[json.loads(l) for l in open(f"{OUT}/rf_val_cat.jsonl")]
rows={"WNN E GA (3)":[x for x in w if x["phase"]=="ga_neurons"],"WNN E grid (3)":[x for x in w if x["phase"]=="grid_search"],
	"RF raw (3)":[x for x in r if x["model"]=="rf_raw"],"RF on E bits (1)":[x for x in r if x["model"]=="rf_E"],"XGB raw (1)":[x for x in r if x["model"]=="xgb_raw"]}
cats=list(w[0]["cat"].keys())
print("VAL, threshold = each model's own F1-optimal VAL threshold; detected/support (mean over seeds); Normal = FP")
print(f"{'category':15s} {'support':>7s} | "+" | ".join(f"{k:>16s}" for k in rows))
for c in cats:
	sup=w[0]["cat"][c][1]
	cells=[]
	for k,xs in rows.items():
		d=np.mean([x["cat"][c][0] for x in xs]); cells.append(f"{d:7.1f} ({(d/sup*100 if c!='Normal' else d/sup*100):6.2f}%)")
	print(f"{c:15s} {sup:7d} | "+" | ".join(f"{s:>16s}" for s in cells))
fn={k:np.mean([sum(x['cat'][c][1]-x['cat'][c][0] for c in cats if c!='Normal') for x in xs]) for k,xs in rows.items()}
fz={k:np.mean([x['cat']['Fuzzers'][1]-x['cat']['Fuzzers'][0] for x in xs]) for k,xs in rows.items()}
print("total FN:", {k:round(v,1) for k,v in fn.items()}); print("Fuzzers FN:", {k:round(v,1) for k,v in fz.items()})
print("VAL F1:", {k:round(np.mean([x['val_f1'] for x in xs]),3) for k,xs in rows.items()})
print("VAL FPR:", {k:round(np.mean([x['val_fpr'] for x in xs]),3) for k,xs in rows.items()})
imp=np.mean([list(x["importances"].values()) for x in rows["RF raw (3)"]],axis=0); names=list(rows["RF raw (3)"][0]["importances"].keys())
meta=json.load(open(f"{OUT}/meta.json")); bpf=meta["bits_per_feature"]
g=[x for x in w if x["phase"]=="ga_neurons"]
conn={f:np.mean([x["feature_conn_counts"].get(f,0) for x in g]) for f in bpf}
tot=sum(conn.values())
xg=rows["XGB raw (1)"][0]["gain"]; xt=sum(xg.values())
print(f"\n{'feature':18s} RFimp%  XGBgain%  E-bits  bits%   WNN-conn%  (GA genomes, mean of 3)")
for i in np.argsort(-imp):
	f=names[i]; key=f"f{i}"
	xgv=xg.get(f, xg.get(key,0))/xt*100
	print(f"{f:18s} {imp[i]*100:6.2f}  {xgv:7.2f}  {bpf[f]:6d}  {bpf[f]/1104*100:5.2f}  {conn[f]/tot*100:8.2f}")
print("distinct bits used by GA genomes:", [x["distinct_bits_used"] for x in g], "of 1104")
