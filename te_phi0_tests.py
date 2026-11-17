import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, json, pickle, time, sys
from ekin import load_argon
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1; m=16
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
tmp=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=1,Nh=1,n_e=10,umax=5.,m=m); X=tmp.X
def interp(key,ang=None):
    a=np.array(ref[key]); return np.interp(X,rr,a) if ang is None else np.interp(X,rr,a)*np.exp(1j*np.interp(X,rr,np.array(ref[ang])))
E1=interp('E_theta_amplitude_V_per_m','E_theta_phase_rad')/2; B1=interp('B_z_amplitude_T','B_z_phase_rad')/2
mode=sys.argv[1]
Nh=2 if mode=='Nh2' else 1; H=2*Nh+1
E=np.zeros((H,m),complex); E[Nh+1]=E1; E[Nh-1]=np.conj(E1); B=np.zeros((H,m),complex)
if mode!='noB': B[Nh+1]=B1; B[Nh-1]=np.conj(B1)
L=2 if mode=='L2' else (3 if mode=='L3' else 1)
t=time.time(); kin=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=L,Nh=Nh,n_e=(240 if mode=='ne240' else 120),umax=45.,m=m,order=2); f=kin.solve(E,B,limiter_iters=2)
print(f'TE phi=0 mode={mode} ({time.time()-t:.0f}s): Te',np.round(kin.mean_energy(f)*2/3,2)); print('   nuiz',np.array2string(kin.nu_iz(f),precision=0,max_line_width=250))
pickle.dump(dict(X=X,Te=kin.mean_energy(f)*2/3,nuiz=kin.nu_iz(f),ne=kin.density(f)),open(f'results/te_phi0_{mode}.pkl','wb'))
