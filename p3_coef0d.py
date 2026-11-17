"""Paper 3: transport and rate coefficients from the energy-grid solver vs BOLOS (BOLSIG+ algorithm), DC field, argon.
Outputs results/p3_coef0d.pkl: for each E/N: <u>, muN, DN, D/mu, muEN, DEN, k_iz, k_exc, and solve times."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, time, warnings, scipy.constants as co
from ekin import load_argon, EnergyGridKinetics, GAMMA
import scipy.integrate as _si; _si.simps = getattr(_si, 'simps', None) or _si.simpson   # BOLOS 0.2 calls simps, removed in SciPy 1.14
from bolos import parser, solver, grid
warnings.filterwarnings('ignore')
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21
with open('data/Ar_LXCat.txt') as fp: procs=parser.parse(fp)
gr=grid.QuadraticGrid(0,80.,400); bs=solver.BoltzmannSolver(gr); bs.load_collisions(procs); bs.target['Ar'].density=1.0; bs.kT=300*co.k/co.eV
K=EnergyGridKinetics(xs,umax=80.,nu=400,L=1,Nh=0,ng=ng)
sig=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1],right=d[-1,1]); sm=sig(xs['elastic'],K.uc)
ENs=[1,2,3,5,7,10,15,20,30,50,70,100,150,200,300]; rows=[]
for EN in ENs:
    E=EN*1e-21*ng
    t=time.time(); f=K.solve(E); t_ours=time.time()-t
    F0=np.real(f[0,0]); F1=np.real(f[0,1]); u=K.uc; du=K.du; v=GAMMA*np.sqrt(u)
    ubar=K.mean_energy(f); W=abs(np.sum(K.w*v*F1))/3; muN=W/E*ng
    DN=(GAMMA/3)*np.sum(u/sm*F0*du)                       # two-term reduced diffusion coefficient, ng cancels: (gamma/3) int u/sigma_m F0 du
    Ge=abs(np.sum(K.w*u*v*F1))/3; muEN=Ge/(E*ubar)*ng         # energy mobility from the energy flux
    DEN=(GAMMA/3)*np.sum(u**2/sm*F0*du)/ubar
    kiz=K.rate_iz(f)/ng; kex=K.rate_exc_total(f)/ng
    t=time.time(); bs.EN=EN*1e-21; bs.init(); fb=bs.converge(bs.maxwell(2.0),maxn=400,rtol=1e-6); t_b=time.time()-t
    kex_b=sum(bs.rate(fb,pr) for tg,pr in bs.iter_inelastic() if pr.kind=='EXCITATION')
    row=dict(EN=EN,u=ubar,u_b=bs.mean_energy(fb),muN=muN,muN_b=bs.mobility(fb),DN=DN,DN_b=bs.diffusion(fb),muEN=muEN,DEN=DEN,kiz=kiz,kiz_b=bs.rate(fb,'Ar -> Ar+'),kex=kex,kex_b=kex_b,t_ours=t_ours,t_b=t_b)
    rows.append(row); print(f"E/N={EN:4d}: <u> {ubar:.3f}/{row['u_b']:.3f}  muN {muN:.3e}/{row['muN_b']:.3e}  DN {DN:.3e}/{row['DN_b']:.3e}  D/mu {DN/muN:.3f}/{row['DN_b']/row['muN_b']:.3f}  kiz {kiz:.2e}/{row['kiz_b']:.2e}  kex {kex:.2e}/{kex_b:.2e}  t {t_ours:.2f}/{t_b:.2f}s")
pickle.dump(rows,open('results/p3_coef0d.pkl','wb'))
