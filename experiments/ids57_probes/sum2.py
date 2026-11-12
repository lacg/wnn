import json, numpy as np, collections
rows=[json.loads(l) for l in open('/tmp/ids57/probe2.jsonl') if l.startswith('{')]
agg=collections.OrderedDict()
for r in rows:
    if r['part']=='xgb_at_wnn_fpr':
        for k,v in r['rows'].items(): print('XGB at FP=%s: FPR %.2f TPR %.2f F1 %.2f'%(k,100*v['fpr'],100*v['tpr'],100*v['f1']))
    elif r['part']=='distinct_bits': print('distinct',r['total'],r['per_feature'])
    elif 'error' in r: print('ERR',r)
    else: agg.setdefault((r['kind'],r['npc'],r['bits']),[]).append(r)
print('kind npc bits | lattice F1 mean [min-max] FPR TPR | u0.02 F1 mean [min-max] FPR TPR | n at pointB(lat)')
for k,v in agg.items():
    L=np.array([x['lattice'] for x in v])*100; U=np.array([x['u002'] for x in v])*100
    print('%-3s %3d %2d | %.2f [%.2f-%.2f] %.2f %.1f | %.2f [%.2f-%.2f] %.2f %.1f | %d/%d'%(k[0],k[1],k[2],L[:,0].mean(),L[:,0].min(),L[:,0].max(),L[:,1].mean(),L[:,2].mean(),U[:,0].mean(),U[:,0].min(),U[:,0].max(),U[:,1].mean(),U[:,2].mean(),(L[:,1]<0.5).sum(),len(v)))
