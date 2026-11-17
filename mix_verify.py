"""Homogeneous DC check of the mixture solver (ekin_mix.MixtureKinetics, growth on, CAR off) against BOLOS for N2 and air (79/21)
at 50, 100, 150 Td; prints mean energy, reduced mobility, ionization and attachment rate coefficients (this code / BOLOS).
Run from the repository root; cross sections from data/ or LXCAT_DIR (N2, O2 IST-Lisbon sets)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import sys, numpy as np, os

import scipy.integrate as _si; _si.simps=getattr(_si,'simps',None) or _si.simpson
from ekin_mix import MixtureKinetics, load_species
from bolos import parser, solver, grid
ng=3.21e21
SP={s:load_species(s) for s in ('N2','O2')}
for comp in ({'N2':1.0},{'N2':0.79,'O2':0.21}):
    K=MixtureKinetics([(SP[s],x) for s,x in comp.items()],umax=60.,nu=300,L=1,Nh=0,ng=ng,growth=True,car=False)
    bs=solver.BoltzmannSolver(grid.QuadraticGrid(0,60,300))
    for s,x in comp.items():
        path=SP[s].get('file') or os.path.join(os.environ.get('LXCAT_DIR','dl/LoKI/Code/Input/'),{'N2':'Nitrogen/N2_LXCat.txt','O2':'Oxygen/O2_LXCat.txt'}[s])
        with open(path) as fp: procs=parser.parse(fp)
        bs.load_collisions(procs)
        bs.target[s].density=x
    bs.kT=300*1.380649e-23/1.602176634e-19
    for EN in (50,100,150):
        f=K.solve(EN*1e-21*ng)
        u=K.mean_energy(f); mu=K.mobility_N(f,EN*1e-21*ng)
        bs.EN=EN*1e-21; bs.init()
        f0=bs.maxwell(2.0); f1=bs.converge(f0,maxn=200,rtol=1e-5)
        ub=bs.mean_energy(f1); mub=bs.mobility(f1)
        R={}; 
        for kind,sn,prod,thr,r in K.rates(f): R[kind]=R.get(kind,0)+comp[sn]*r if False else R.get(kind,0)+r*comp[sn]
        kizb=0; katb=0
        for tg,t in bs.iter_all():
            k=bs.rate(f1,t)
            if t.kind=='IONIZATION': kizb+=k*t.target.density
            if t.kind=='ATTACHMENT': katb+=k*t.target.density
        print(comp,EN,'u %.3f/%.3f (%.1f%%)'%(u,ub,100*(u/ub-1)),'muN %.3e/%.3e (%.1f%%)'%(mu,mub,100*(mu/mub-1)),'kiz %.3e/%.3e'%(R.get('iz',0),kizb),'katt %.3e/%.3e'%(R.get('att',0),katb))
