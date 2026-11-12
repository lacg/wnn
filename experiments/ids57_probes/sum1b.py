import json, numpy as np
rows=[json.loads(l) for l in open('/tmp/ids57/probe1.jsonl') if l.startswith('{')]
sup=[r for r in rows if r['part']=='support'][0]['test']
x={r['depth']:r for r in rows if r['part']=='xgb'}
w=[r for r in rows if r['part']=='wnn']; B=[r for r in w if r['fpr']<0.005]; A=[r for r in w if r['fpr']>=0.005]
print('pointB flows',[r['flow'] for r in B],'pointA',[r['flow'] for r in A])
print('cat | n | XGB6 | XGB1 | WNN-B mean [min-max] | WNN-A | extra misses vs XGB (rows)')
for c,n in sup.items():
    v=np.array([r['subsets']['cat:'+c]['correct'] for r in B])*100
    print('%-15s %6d | %5.1f | %5.1f | %5.1f [%.1f-%.1f] | %5.1f | %+.0f'%(c,n,100*x[6]['percat'][c],100*x[1]['percat'][c],v.mean(),v.min(),v.max(),100*A[0]['subsets']['cat:'+c]['correct'],(x[6]['percat'][c]-v.mean()/100)*n))
for k in ('ov:ben_xgbOK','ov:ben_xgbFP','ov:att_xgbOK','ov:att_xgbFN'):
    n=B[0]['subsets'][k]['n']; v=np.array([r['subsets'][k]['correct'] for r in B])
    print(k,n,'WNN-B correct %.1f%% -> wrong rows %.0f [%.0f-%.0f]'%(100*v.mean(),(1-v.mean())*n,(1-v.max())*n,(1-v.min())*n),'| WNN-A correct %.1f%% wrong %.0f'%(100*A[0]['subsets'][k]['correct'],(1-A[0]['subsets'][k]['correct'])*n))
print('WNN-B conf mean', np.mean([np.array(r['conf']).ravel() for r in B],axis=0))
print('modes B mean', {k: np.round(np.mean([np.array(r['modes'][k])*100 for r in B],axis=0),2).tolist() for k in ('margin_val_cal','margin_classnorm','argmax_classnorm')})
for d in (6,3,2,1): print('xgb depth',d,'%.2f / %.2f / TPR %.1f'%(100*x[d]['f1'],100*x[d]['fpr'],100*x[d]['tpr']), {k:round(100*v,1) for k,v in x[d]['tpr_at_fpr'].items()})
