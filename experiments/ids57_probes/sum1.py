import json
rows=[json.loads(l) for l in open('/tmp/ids57/probe1.jsonl') if l.startswith('{')]
for r in rows:
    if r['part']=='support': print(r)
    if r['part']=='xgb':
        print('XGB depth',r['depth'],'F1 %.2f FPR %.2f TPR %.2f'%(100*r['f1'],100*r['fpr'],100*r['tpr']),r['conf'],'thr',r['thr'],r['secs'],'s')
        print('  percat',{k:round(100*v,1) for k,v in r['percat'].items()})
        print('  tpr@fpr',{k:round(100*v,1) for k,v in r['tpr_at_fpr'].items()})
        if 'gain_share_by_feature' in r: print('  gain',{k:round(100*v,1) for k,v in sorted(r['gain_share_by_feature'].items(),key=lambda x:-x[1])}, r['n_bits_used'])
    if r['part']=='wnn':
        print('WNN',r['flow'],r['neurons'],'bits',r['bits_hist'],'F1 %.2f FPR %.2f'%(100*r['f1'],100*r['fpr']),r['conf'],r['secs'])
        print('  modes',{k:[round(100*x,2) for x in v] for k,v in r['modes'].items()})
        print('  sub',{k:(v['n'],round(100*v['correct'],1)) for k,v in r['subsets'].items()})
        for c in ('cluster0','cluster1'):
            print('  conn',c,r['conn_share'][c+'_distinct_bits'],{k:round(100*v,1) for k,v in sorted(r['conn_share'][c].items(),key=lambda x:-x[1])})
