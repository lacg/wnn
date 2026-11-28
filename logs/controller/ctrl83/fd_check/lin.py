# Read-only linear ANALYSIS of the roll axis of MPCOF>rate (ctbr_axis_mpc.rs model, transcribed for margins only).
import numpy as np, math
dt=1e-3; dec=2; dtr=1/500.; n_o=10
kp,ki,kd=0.03497110216713654,0.06994220433427308,0.00043713877708920673
L=0.07071067811865477; Jx=3.0039824567715368e-05; c=L/Jx
fr=500/30.; ohm=math.tan(math.pi/fr); cos45=math.cos(math.pi/4); cc=1+2*cos45*ohm+ohm*ohm
b0=ohm*ohm/cc; b1=2*b0; b2=b0; a1=2*(ohm*ohm-1)/cc; a2=(1-2*cos45*ohm+ohm*ohm)/cc
alpha=lambda T: 4*dt/max(T,4*dt)
TH,W,UA,IN,F1,F2,PM,FH=range(8)
def tick(s,sp,al,upd,d=0.0):
	n=s.copy()
	if upd:
		e=sp-s[W]; n[IN]=s[IN]+e*dtr; raw=-(s[W]-s[PM])/dtr
		d0=raw-s[F1]*a1-s[F2]*a2; n[F2]=s[F1]; n[F1]=d0; der=d0*b0+s[F1]*b1+s[F2]*b2
		n[PM]=s[W]; n[FH]=kp*e+ki*n[IN]+kd*der
	n[UA]=s[UA]+al*(n[FH]-s[UA]); acc=c*n[UA]+d
	n[W]=s[W]+dt*acc; n[TH]=s[TH]+dt*s[W]+0.5*dt*dt*acc; return n
def outer(al):
	run=lambda s,sp: [s:=tick(s,sp,al,j%dec==0) for j in range(n_o)][-1]
	A=np.column_stack([run(np.eye(8)[j],0.) for j in range(8)]); B=run(np.zeros(8),1.)
	return A,B[:,None]
def lqr(A,B):
	Q=np.diag([100,.5,0,0,0,0,0,0.]); R=np.array([[1.]]); P=Q.copy()
	for _ in range(20000):
		Pn=Q+A.T@P@A-A.T@P@B@np.linalg.solve(R+B.T@P@B,B.T@P@A)
		if np.abs(Pn-P).max()<1e-12: break
		P=Pn
	K=-np.linalg.solve(R+B.T@P@B,B.T@P@A); return K[0],np.linalg.cond(P)
# augmented 1 kHz chain: z=[plant 8, ua_twin, dhat, prev_w]; controller uses x̃=[th,w,uatwin,I,f1,f2,pm,Fh]-xss(dhat)
lg=0.02
def chain_tick(z,sp,ap,ad,upd,d=0.0):
	s=tick(z[:8],sp,ap,upd,d); n=np.zeros(11); n[:8]=s
	n[8]=z[8]+ad*(s[FH]-z[8])   # twin of the lagged force at the design lag
	res=(s[W]-z[10])/dt - c*n[8]-z[9]; n[9]=z[9]+lg*res; n[10]=s[W]; return n
def ctrl_map(K):
	M=np.zeros((8,11)); M[:,:8]=np.eye(8); M[UA,:]=0; M[UA,8]=1
	# xss(dhat): UA=FH=-dhat/c, IN=-dhat/(c ki)
	M[UA,9]+=1/c; M[FH,9]+=1/c; M[IN,9]+=1/(c*ki); return K@M
def analyse(Tp,Td,Tlin_design=None):
	ap,ad=alpha(Tp),alpha(Td); A,B=outer(ad); K,condP=lqr(A,B); G=ctrl_map(K)
	run=lambda z,sp:[z:=chain_tick(z,sp,ap,ad,j%dec==0) for j in range(n_o)][-1]
	Az=np.column_stack([run(np.eye(11)[j],0.) for j in range(11)]); Bz=run(np.zeros(11),1.)
	cl=np.abs(np.linalg.eigvals(Az+np.outer(Bz,G))).max()
	ws=np.logspace(-1,math.log10(math.pi/(n_o*dt))-1e-3,6000); Lw=[]
	for w in ws:
		zz=np.exp(1j*w*n_o*dt); Lw.append(-G@np.linalg.solve(zz*np.eye(11)-Az,Bz))
	Lw=np.array(Lw); mag=np.abs(Lw); i=np.where(np.diff(np.sign(mag-1)))[0]
	wc=ws[i[0]] if len(i) else float('nan'); pm=180+np.degrees(np.angle(Lw[i[0]])) if len(i) else float('nan')
	pm=(pm+180)%360-180
	ph=np.unwrap(np.angle(Lw)); j=np.where(np.diff(np.sign(ph+np.pi)))[0]; gm=-20*np.log10(mag[j[0]]) if len(j) else float('inf')
	# step: 5 deg tilt at rest, linear, sp clamp 4.2 rad/s; IAE over 2 s; steady last 20%
	z=np.zeros(11); z[TH]=math.radians(5); th=[];sps=[];fh=[]
	for k in range(200):
		sp=float(np.clip(G@z,-4.2,4.2)); sps.append(sp)
		for j in range(n_o): z=chain_tick(z,sp,ap,ad,j%dec==0); th.append(z[TH]); fh.append(z[FH])
	th=np.degrees(np.array(th)); 
	t10=np.argmax(np.abs(th)<0.5)*dt
	return dict(K_th=K[0],K_w=K[1],condP=condP,rho=cl,wc=wc,pm=pm,gm=gm,iae=np.abs(th).mean(),ovs=-th.min(),t_to_0p5=t10,spmax=max(np.abs(sps)),fhmax=max(np.abs(fh)),jerk=np.abs(np.diff(sps)).sum())
print("Tp  Td   K_th    K_w   condP    rho   wc(r/s) PM(deg) GM(dB)  meanErr(deg) ovs(deg) t<0.5deg  |sp|max |Fh|max(N)")
for Tp in (0.2,0.3,0.4):
	for Td in (0.1,0.15,0.2,0.25,0.3,0.4):
		r=analyse(Tp,Td); print(f"{Tp:.2f} {Td:.2f} {r['K_th']:7.2f} {r['K_w']:6.3f} {r['condP']:.1e} {r['rho']:.3f} {r['wc']:7.2f} {r['pm']:7.1f} {r['gm']:6.1f}   {r['iae']:.4f}     {r['ovs']:.3f}   {r['t_to_0p5']:.3f}   {r['spmax']:.2f}  {r['fhmax']:.4f}")
