"""Two-term vs multi-term vs local in a prescribed skin-layer field (no potential), final total-energy solver (generation IV)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, time, pickle, sys
from scipy.special import iv
from ekin import load_argon, EnergyGridKinetics
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1; m=16
L=int(sys.argv[1])
kin=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=L,Nh=1,n_e=120,umax=45.,m=m,order=2)
kk=(1+1j)/0.033; Ew=250.0; E1=Ew/2*iv(1,kk*kin.r)/iv(1,kk*R); E=np.zeros((3,m),complex); E[2]=E1; E[0]=np.conj(E1)
B1=-(1/(1j*omega))*(np.gradient(kin.r*E1,kin.r)/kin.r); B=np.zeros((3,m),complex); B[2]=B1; B[0]=np.conj(B1)
t=time.time(); f=kin.solve(E,B); out=dict(X=kin.X,Te=kin.mean_energy(f)*2/3,nuiz=kin.nu_iz(f),ne=kin.density(f),time=time.time()-t,lam=kin.lam)
print(f'L={L} ({out["time"]:.0f}s, lam={kin.lam.real:.1e}): Te',np.round(out['Te'],3),' nuiz',np.array2string(out['nuiz'],precision=3),flush=True)
if L==1:
    K=EnergyGridKinetics(xs,umax=50.,nu=200,L=1,Nh=1,ng=ng); loc=[]
    for e1 in np.abs(E1):
        g=K.solve(2*e1,omega); loc.append((K.mean_energy(g)*2/3,K.rate_iz(g)))
    loc=np.array(loc); out['local']=dict(Te=loc[:,0],nuiz=loc[:,1]); print('local: Te',np.round(loc[:,0],3),' nuiz',np.array2string(loc[:,1],precision=3))
    out['E1']=E1
pickle.dump(out,open(f'results/twoterm_te_L{L}.pkl','wb'))
