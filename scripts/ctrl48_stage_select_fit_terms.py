"""CTRL-48 2x2 s1 diagnosis: recompute the stage-select ZRank fitness terms (robust z = (x-median)/(1.4826 MAD),
clamp 3; columns reward, stable, steady, pos with the recipe weights) from the VAL stage-table values printed in a
run .out, to see which column decides the headline. Rounded inputs, so it reproduces the logged fit to ~0.05."""
import numpy as np, sys
W={'rw':0.2557,'stable':0.2045,'steady':0.358,'pos':0.1818}
def rz(x, higher):
	med=np.median(x); mad=1.4826*np.median(np.abs(x-med))
	if mad==0: z=np.where(x==med,0.0,np.where(x>med,3.0,-3.0))
	else: z=np.clip((x-med)/mad,-3,3)
	return -z if higher else z
def run(name,C):
	n=[c[0] for c in C]; A=np.array([c[1:] for c in C],float)
	st,rw,sd,pos,fit=A[:,0],A[:,1],A[:,2],A[:,3],A[:,4]
	t={'rw':W['rw']*rz(rw,True),'stable':W['stable']*rz(st,True),'steady':W['steady']*rz(sd,False),'pos':W['pos']*rz(pos,False)}
	tot=sum(t.values())
	print(name)
	for i in range(len(n)): print(f"  {n[i]:10s} log {fit[i]:+.3f} recomputed {tot[i]:+.3f} | "+" ".join(f"{k} {v[i]:+.3f}" for k,v in t.items()))
# (name, stable, rw, steady, pos, fit) from the .out stage tables (VAL, 20 units)
run("L1+L3 s1",[("GRID#0",100,-2.04,.60,.308,.3639),("GRID#1",100,-2.90,1.14,.312,2.3865),("GRID#2",100,-1.83,.63,.307,-.1119),
("NEURONS#0",99.9,-1.84,.53,.308,.3383),("NEURONS#1",99.9,-1.84,.53,.308,.3383),("NEURONS#2",100,-2.03,.67,.307,.3921),
("MEMORY#0",100,-1.95,.44,.312,.9136),("MEMORY#1",100,-1.98,.45,.313,.3786),("MEMORY#2",100,-1.95,.44,.313,.3004)])
run("L1 s1",[("GRID#0",100,-1.84,.56,.337,.2045),("GRID#1",100,-2.71,1.14,.339,2.4365),("GRID#2",100,-1.98,.57,.341,.7536),
("NEURONS#0",100,-1.73,.45,.339,-.3878),("NEURONS#1",100,-1.87,.54,.341,.4528),("NEURONS#2",100,-1.71,.49,.339,-.2245),
("MEMORY#0",100,-1.73,.45,.339,-.3878),("MEMORY#1",100,-1.78,.45,.348,.2085),("MEMORY#2",100,-1.90,.44,.349,.3689)])
