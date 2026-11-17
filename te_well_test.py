"""Prescribed parabolic well phi0*(1-X^2) + uniform RF field 200 V/m, no B: density compression (final solver)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, sys, pickle
from ekin import load_argon
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1; m=16
phi0=float(sys.argv[1]); E0=200.
tmp=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=1,Nh=1,n_e=10,umax=5.,m=m); X=tmp.X
kin=CylKineticsTE(xs,R,ng,omega,phi0*(1-X**2),L=1,Nh=1,n_e=120,umax=45.,m=m,order=2)
E=np.zeros((3,m),complex); E[2]=E0/2; E[0]=E0/2; B=np.zeros((3,m),complex); f=kin.solve(E,B)
n=kin.density(f); Te=kin.mean_energy(f)*2/3
# compression: density on the axis cell / density in the wall cell (same definition as the MC: ring averages, inner/outer ring)
print(f'well {phi0} V: lam={kin.lam.real:.1e} n_axis/n_wall={n[-1]/n[0]:.3f}  Te wall/axis {Te[0]:.2f}/{Te[-1]:.2f}', flush=True)
pickle.dump(dict(X=X,n=n,Te=Te),open(f'results/te_well_{phi0}.pkl','wb'))
