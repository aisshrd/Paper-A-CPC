import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, scipy.constants as co, time, pickle, sys
from ekin import load_argon; from ekin_cyl import *
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6; R=0.1
p=float(sys.argv[1]); n0=float(sys.argv[2]); Bext_G=float(sys.argv[3]); tag=sys.argv[4]
iters=int(sys.argv[5]) if len(sys.argv)>5 else 10
ng=p/(co.k*300.)
kin=CylKinetics(xs,R,ng,omega,L=2,Nh=1,nu=50,umax=50.,m=8)
Bext=np.zeros(3,complex); Bext[0]=Bext[2]=Bext_G*1e-4/2
d=Discharge(kin,Bext,n0_guess=n0)
print(f'p={p} Pa n0={n0:.1e} Bext={Bext_G} G -> E_vac(wall)={abs(Bext[2])*2*omega*R/2:.0f} V/m',flush=True)
t=time.time(); d.run(iters=iters,relax=0.6)
out=dict(p=p,n0=n0,Bext_G=Bext_G,X=kin.X,r=kin.r,ne=d.ne,ni=d.ni,Te=d.Te,nuiz=d.nuiz,phi=d.phi,E=d.E,B=d.B,hist=d.hist,balance=d.balance,
         f0=np.real(d.f[kin.Nh,0]),uc=kin.uc,w=kin.w,time=time.time()-t,Evac=abs(Bext[2])*omega*R)
pickle.dump(out,open(f'results/discharge_{tag}.pkl','wb')); print('done',time.time()-t,'balance',d.balance)
