import json, numpy as np, collections, sys
for w in ('unswr','unswt'):
    try: rows=[json.loads(l) for l in open(f'/tmp/ids57/probe3_{w}.jsonl') if l.startswith('{')]
    except FileNotFoundError: continue
    agg=collections.OrderedDict()
    for r in rows:
        if r['part']!='val': print(r); continue
        agg.setdefault((r['npc'],r['bits']),[]).append(r)
    for k,v in agg.items():
        line='%s %3d %3d |'%(w,k[0],k[1])
        for name in ('lattice','u002','lattice_rate1','ms01'):
            xs=[x[name] for x in v if name in x]
            if not xs:
                errs=[x.get(name+'_error') for x in v if name+'_error' in x]
                if errs: line+=' %s ERR %s |'%(name,errs[0][:60])
                continue
            A=np.array(xs); line+=' %s F1 %.2f [%.2f-%.2f] FPR %.2f TPR %.1f cells %.0fK (n=%d) |'%(name,100*A[:,0].mean(),100*A[:,0].min(),100*A[:,0].max(),100*A[:,1].mean(),100*A[:,2].mean(),A[:,3].mean()/1e3,len(xs))
        print(line)
