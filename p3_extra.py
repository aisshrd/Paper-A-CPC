"""Paper 3 (v1.7.0) extra checks: (a) growth term at 300 Td vs BOLOS; (b) Maxwellian/constant-nu check of the energy transport
coefficients (mu_eps/mu = D_eps/D = 5/3); (c) total-energy lookup: nu_iz(r) predicted from the f0(eps_tot) of one cell vs kinetic;
(d) table floor of the LMEA lookup (400 vs 800 cells)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, os, scipy.constants as co
os.environ.setdefault('OMP_NUM_THREADS','1')
from ekin import load_argon, EnergyGridKinetics
from ekin_mix import MixtureKinetics
from mix_scan import SP
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; e=co.e; me=co.m_e; out={}
# (a) growth term
K=MixtureKinetics([(SP['Ar'],1.0)],umax=80.,nu=400,L=1,Nh=0,ng=ng,growth=True); f=K.solve(300e-21*ng); r=K.rates(f)
Kn=MixtureKinetics([(SP['Ar'],1.0)],umax=80.,nu=400,L=1,Nh=0,ng=ng,growth=False); fn=Kn.solve(300e-21*ng); rn=Kn.rates(fn)
kiz=sum(x[4] for x in r if x[0]=='iz'); kizn=sum(x[4] for x in rn if x[0]=='iz')
print('300 Td: growth on <u>=%.3f k_iz=%.4g ; growth off <u>=%.3f k_iz=%.4g  (BOLOS 8.510 / 1.29e-15)'%(K.mean_energy(f),kiz,Kn.mean_energy(fn),kizn)); out['growth']=dict(u_on=K.mean_energy(f),kiz_on=kiz,u_off=Kn.mean_energy(fn),kiz_off=kizn)
# (b) Maxwellian check with sigma_m = c/sqrt(u) (constant collision frequency): mu_eps/mu and D_eps/D from the integral definitions
u=np.linspace(1e-4,60,200001); T=3.0; F0=np.exp(-u/T); F0/=np.trapezoid(np.sqrt(u)*F0,u); sig=1e-20/np.sqrt(u)
dF=np.gradient(F0,u); ub=np.trapezoid(u**1.5*F0,u)
muN=-np.trapezoid(u/sig*dF,u)/3; DN=np.trapezoid(u/sig*F0,u)/3; mueN=-np.trapezoid(u**2/sig*dF,u)/3/ub; DeN=np.trapezoid(u**2/sig*F0,u)/3/ub
print('Maxwellian, constant nu: mu_eps/mu=%.4f D_eps/D=%.4f (exact 5/3=1.6667); D/mu=%.4f (T=%.1f)'%(mueN/muN,DeN/DN,DN/muN,T)); out['maxwell']=dict(mue_mu=mueN/muN,De_D=DeN/DN,D_mu=DN/muN)
# (c) total-energy lookup
s=pickle.load(open('results/te_reffields_ne120_m16_L1_o2.pkl','rb')); F0=np.real(s['F0']); uk=s['ukin']; w=s['w']; X=s['X']; phi=s['phi']; ec=s['ec']
sg=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1],right=d[-1,1]); thi,di=xs['iz']; nuiz_u=lambda U: ng*sg(di,U)*np.sqrt(2*e*np.maximum(U,0)/me)*(U>thi)
res={}
for jref in (8,15):      # f0(eps_tot) taken from r/R=0.72 and from the axis
    g=F0[:,jref]; pred=[]
    for j in range(len(X)):
        ok=uk[:,j]>0; n=np.sum(w[ok,j]*g[ok]); pred.append(np.sum(w[ok,j]*nuiz_u(uk[ok,j])*g[ok])/n)
    pred=np.array(pred); res[jref]=pred; print('TE lookup from cell %d (r/R=%.2f): ratio pred/kin'%(jref,X[jref]),np.array2string(pred/s['nuiz'],precision=2,max_line_width=200))
out['te_lookup']={str(k):(v/s['nuiz']).tolist() for k,v in res.items()}
# (d) LMEA table floor
for nu in (400,800):
    Kd=EnergyGridKinetics(xs,umax=50.,nu=nu,L=1,Nh=1,ng=ng); tab=[]
    for E0 in np.logspace(0.0,3.2,40):
        g=Kd.solve(E0,2*np.pi*13.56e6); tab.append((Kd.mean_energy(g),Kd.rate_iz(g)/ng))
    tab=np.array(tab); print('nu=%d: mean energy / k_iz at low fields'%nu,np.array2string(tab[:14],precision=3,max_line_width=250))
    out[f'floor_{nu}']=tab.tolist()
json.dump(out,open('results/p3_extra.json','w'),indent=1,default=float)
