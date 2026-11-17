import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ekin import *
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for p,c in [(13.3,'C0'),(40.,'C1'),(133.,'C2')]:
    rows=np.array(pickle.load(open(f'results/hf_table_p{p}.pkl','rb')),dtype=float)
    E=rows[:,0]
    ax[0].plot(E,rows[:,1],'-',color=c,label=f'{p} Pa: harmonic balance'); ax[0].plot(E,rows[:,3],'--',color=c,label=f'{p} Pa: BOLSIG+-HF (E_eff)')
    m=np.isfinite(rows[:,6]); ax[0].plot(E[m],rows[m,6],'o',color=c,label=f'{p} Pa: Monte Carlo' if m.any() else None)
    ok=rows[:,2]>0; ax[1].semilogy(E[ok],rows[ok,2],'-',color=c); ok2=rows[:,4]>0; ax[1].semilogy(E[ok2],rows[ok2,4],'--',color=c)
    m2=m&(rows[:,7]>0); ax[1].semilogy(E[m2],rows[m2,7],'o',color=c)
ax[0].set(xlabel='E0 (RF amplitude), V/m',ylabel='<energy>, eV',title='0D RF: mean energy'); ax[0].legend(fontsize=6); ax[0].grid(alpha=.3)
ax[1].set(xlabel='E0, V/m',ylabel='nu_iz, 1/s',title='0D RF: ionization frequency (solid HB, dashed BOLSIG-HF, dots MC)'); ax[1].grid(alpha=.3)
# EEDF at 13.3 Pa, 400 V/m: HB vs MC
ng=3.2e21; K=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=2,ng=ng); f=K.solve(400.,omega); 
mc=pickle.load(open(f'results/mc_hf_ng{ng:.1e}.pkl','rb'))[400.0]; uc,fe=mc['eedf']
ax[2].semilogy(K.uc,np.maximum(K.eedf(f),1e-12),'-',label='harmonic balance L=1'); ax[2].semilogy(uc,np.maximum(fe,1e-12),'o',ms=3,label='Monte Carlo')
ax[2].set(xlabel='energy, eV',ylabel='EEDF, eV^-3/2',title='13.3 Pa, 400 V/m',xlim=(0,45),ylim=(1e-7,1)); ax[2].legend(); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig('results/figR1_0d_comparison.png',dpi=130)
# DC vs BOLOS
import scipy.integrate as _si; _si.simps = getattr(_si, 'simps', None) or _si.simpson   # BOLOS 0.2 calls simps, removed in SciPy 1.14
from bolos import parser, solver, grid; import scipy.constants as co, warnings; warnings.filterwarnings('ignore')
with open('data/Ar_LXCat.txt') as fp: procs=parser.parse(fp)
gr=grid.QuadraticGrid(0,80.,400); bs=solver.BoltzmannSolver(gr); bs.load_collisions(procs); bs.target['Ar'].density=1.0; bs.kT=300*co.k/co.eV
Kd=EnergyGridKinetics(xs,umax=80.,nu=400,L=1,Nh=0,ng=ng); ENs=[1,2,3,5,10,20,30,50,100,200,300]; rows=[]
for EN in ENs:
    bs.EN=EN*1e-21; bs.init(); fb=bs.converge(bs.maxwell(2.0),maxn=400,rtol=1e-6); f=Kd.solve(EN*1e-21*ng)
    rows.append((EN,Kd.mean_energy(f),bs.mean_energy(fb),Kd.rate_iz(f)/ng,bs.rate(fb,"Ar -> Ar+"),abs(Kd.drift(f))/(EN*1e-21*ng)*ng,bs.mobility(fb)))
rows=np.array(rows); pickle.dump(rows,open('results/dc_table.pkl','wb'))
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
ax[0].semilogx(rows[:,0],rows[:,1],'o-',label='this work'); ax[0].semilogx(rows[:,0],rows[:,2],'s--',label='BOLOS (BOLSIG+ algorithm)'); ax[0].set(xlabel='E/N, Td',ylabel='<energy>, eV'); ax[0].legend(); ax[0].grid(alpha=.3)
ok=rows[:,3]>1e-25; ax[1].loglog(rows[ok,0],rows[ok,3],'o-'); ok=rows[:,4]>1e-25; ax[1].loglog(rows[ok,0],rows[ok,4],'s--'); ax[1].set(xlabel='E/N, Td',ylabel='k_iz, m^3/s'); ax[1].grid(alpha=.3)
ax[2].semilogx(rows[:,0],rows[:,5],'o-'); ax[2].semilogx(rows[:,0],rows[:,6],'s--'); ax[2].set(xlabel='E/N, Td',ylabel='mu N, 1/(m V s)'); ax[2].grid(alpha=.3)
fig.suptitle('DC swarm parameters, Ar (IST-Lisbon set): energy-grid solver vs BOLOS'); fig.tight_layout(); fig.savefig('results/figR0_dc_bolos.png',dpi=130)
for r in rows: print('E/N=%4d: <u> %.3f/%.3f  k_iz %.2e/%.2e  muN %.3e/%.3e'%tuple(r))
