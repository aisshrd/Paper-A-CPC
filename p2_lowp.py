"""Paper 2: low-pressure test of the harmonic truncation. Uniform RF field E0 + parabolic well phi0 (no B) at pressure p;
solver with Nh time harmonics and L angular harmonics. usage: python p2_lowp.py <p_Pa> <E0> <phi0> <Nh> <L>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, sys, pickle, time, scipy.constants as co
from ekin import load_argon
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6; R=0.1; m=16
p=float(sys.argv[1]); E0=float(sys.argv[2]); phi0=float(sys.argv[3]); Nh=int(sys.argv[4]); L=int(sys.argv[5]); ng=p/(co.k*300.); n_e=int(sys.argv[6]) if len(sys.argv)>6 else 120
tmp=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=1,Nh=1,n_e=10,umax=5.,m=m); X=tmp.X
t=time.time(); kin=CylKineticsTE(xs,R,ng,omega,phi0*(1-X**2),L=L,Nh=Nh,n_e=n_e,umax=45.,m=m,order=2)
H=2*Nh+1; E=np.zeros((H,m),complex); E[Nh+1]=E0/2; E[Nh-1]=E0/2; B=np.zeros((H,m),complex); f=kin.solve(E,B)
n=kin.density(f); Te=kin.mean_energy(f)*2/3; nuiz=kin.nu_iz(f); F0=np.real(f[Nh,0])
# harmonic content: |f_n0|/|f_00| for the isotropic part, density-weighted
hc=[float(np.sum(np.abs(kin.w*f[Nh+k,0]))/np.sum(np.abs(kin.w*f[Nh,0]))) for k in range(1,Nh+1)]
print(f'p={p} Pa E0={E0} well={phi0} Nh={Nh} L={L} ({time.time()-t:.0f}s, {kin.nTot} unknowns, lam={kin.lam.real:.1e}): Te',np.round(Te,3)); print('  n/n_wall',np.round(n/n[0],3)); print('  nuiz',np.array2string(nuiz,precision=1,max_line_width=250)); print('  harmonic content |f_n|/|f_0|:',np.round(hc,4))
pickle.dump(dict(p=p,E0=E0,phi0=phi0,Nh=Nh,L=L,X=X,Te=Te,n=n,nuiz=nuiz,F0=F0,ukin=kin.ukin,w=kin.w,ec=kin.ec,hc=hc,time=time.time()-t,nTot=kin.nTot),open(f'results/p2_lowp_p{p:g}_Nh{Nh}_L{L}{"_ne"+str(n_e) if n_e!=120 else ""}.pkl','wb'))
