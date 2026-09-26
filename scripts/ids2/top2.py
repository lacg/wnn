import pickle, sys, collections, json, re, sqlite3, statistics as st
con = sqlite3.connect("file:/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db?mode=ro", uri=True)
def shape(gh, eid):
	r = con.execute("SELECT tiers_json, total_neurons FROM genomes WHERE experiment_id=? AND (genome_hash=? OR config_hash=?) LIMIT 1", (eid, gh, gh)).fetchone()
	if not r: return "n/a"
	try: t = json.loads(r[0])
	except Exception: return f"{r[1]}n ?b"
	if isinstance(t, dict):
		b = t.get("bits_per_neuron") or []
		return f"{len(b)}n x {st.mean(b):.1f}b(max {max(b)})" if b else "?"
	if isinstance(t, list) and t and isinstance(t[0], dict):
		return f"{sum(x.get('neurons',0)*x.get('clusters',1) for x in t)}n x {max(x.get('bits',0) for x in t)}b"
	return "?"
ORF="2026-08-30T02:27"; QSRF="2026-09-19T15:40"
def flags(r, sh):
	f=[]
	f.append("v2" if ((r["split"] or "").endswith("_3way") and r["start"]>="2026-07-11") else ("3WAY-preV2-oracle" if (r["split"] or "").endswith("_3way") else "2WAY-oracle"))
	if r["start"]<"2026-05-28T12" and r["ph"]=="GA": f.append("PRE-DUALFIX-LEAK")
	mb = int(re.search(r"max (\d+)", sh).group(1)) if "max" in sh else (int(re.search(r"x (\d+)b", sh).group(1)) if re.search(r"x (\d+)b", sh) else (r["maxb"] or 0))
	if mb>64 and r["start"]<ORF: f.append("PRE-ORFOLD")
	if r["mem"]=="QSR" and r["start"]<QSRF: f.append("PRE-QSR")
	if r["name"].startswith("zINVALID"): f.append("INVALID")
	if r["flip"]: f.append("FLIP")
	return ",".join(f)
rows=pickle.load(open("rows.pkl","rb"))
mode = sys.argv[1]
if mode=="top" and len(sys.argv)>4:
	rows=[r for r in rows if flags(r, "")=="v2" or flags(r,"").startswith("v2")]
if mode=="top":
	for ds in sys.argv[2].split(","):
		R=sorted([r for r in rows if r["ds"]==ds], key=lambda r:(-r["f1"], r["fpr"]))
		print(f"\n=== {ds}  vs_rows={len(R)} flows={len({r['fid'] for r in R})}")
		seen=set(); k=0
		for r in R:
			key=(r["fid"],r["ph"],r["gh"])
			if key in seen: continue
			seen.add(key); k+=1
			same=[x for x in R if (x["fid"],x["ph"],x["gh"])==key and abs(x["f1"]-r["f1"])<1e-9]
			sh=shape(r["gh"],r["eid"])
			gts='/'.join(sorted({x['gt'].replace('best_','') for x in same})); mds='/'.join(dict.fromkeys(x['md'] for x in same))
			print(f"{r['fid']} {r['name']} | {r['split']} {r['mem']} nb={r['nb']} w(ce/acc/f1/fpr)={r['w']} {r['agg']} {r['feat']} | {r['ph']} {gts} {mds} | {r['f1']:.2f}/{r['fpr']:.2f}/{r['acc']:.2f} | {sh} | {r['dur']:.2f}h | {r['start'][:16]} | {flags(r,sh)}")
			if k>=int(sys.argv[3]): break
elif mode=="cfg":
	G=collections.defaultdict(list)
	for r in rows:
		if r["md"]!="val_cal" or r["gt"]!="best_f1": continue
		G[(r["fid"])].append(r)
	C=collections.defaultdict(list)
	for fid, rs in G.items():
		ph = "GA" if any(x["ph"]=="GA" for x in rs) else "GS"
		r = [x for x in rs if x["ph"]==ph][0]
		base = re.sub(r"-(r|s)\d+(?=-|$)", "", r["name"])
		C[(r["ds"], base, r["split"], r["mem"], r["nb"], ph)].append(r)
	out=[]
	for k, rs in C.items():
		if len(rs)<3: continue
		f=[x["f1"] for x in rs]; p=[x["fpr"] for x in rs]; a=[x["acc"] for x in rs]; d=[x["dur"] for x in rs]
		out.append((k, len(rs), st.mean(f), st.stdev(f), st.mean(p), st.stdev(p), st.mean(a), st.mean(d), min(x["start"] for x in rs), max(x["start"] for x in rs), rs))
	for ds in sys.argv[2].split(","):
		O=sorted([o for o in out if o[0][0]==ds], key=lambda o:-o[2])
		print(f"\n=== {ds} configs n>=3: {len(O)}")
		for o in O[:int(sys.argv[3])]:
			k=o[0]; rs=o[-1]
			shs=[shape(x["gh"],x["eid"]) for x in rs]
			fl=collections.Counter(flags(x,s) for x,s in zip(rs,shs))
			print(f"{k[1]} | {k[2]} {k[3]} nb={k[4]} {k[5]} | n={o[1]} F1 {o[2]:.2f}±{o[3]:.2f} FPR {o[4]:.2f}±{o[5]:.2f} Acc {o[6]:.2f} | dur {o[7]:.2f}h (min {min(x['dur'] for x in rs):.2f} max {max(x['dur'] for x in rs):.2f}) | {o[8][:10]}..{o[9][:10]} | {dict(fl)} | w={rs[0]['w']} {rs[0]['agg']} | shapes {shs[:3]}")
