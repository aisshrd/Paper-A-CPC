import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, json, scipy.constants as co
from ekin import load_argon
from ekin_cyl_te import CylKineticsTE
from ekin_cyl_fv import CylKineticsFV
from ekin_cyl import Discharge
xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1; m=16; e=co.e
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
tmp=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=1,Nh=1,n_e=10,umax=5.,m=m); X=tmp.X
def interp(key,ang=None):
    a=np.array(ref[key]); return np.interp(X,rr,a) if ang is None else np.interp(X,rr,a)*np.exp(1j*np.interp(X,rr,np.array(ref[ang])))
E1=interp('E_theta_amplitude_V_per_m','E_theta_phase_rad')/2; B1=interp('B_z_amplitude_T','B_z_phase_rad')/2; phi=interp('phi_V')
E=np.zeros((3,m),complex); E[2]=E1; E[0]=np.conj(E1); B=np.zeros((3,m),complex); B[2]=B1; B[0]=np.conj(B1)
kin=CylKineticsTE(xs,R,ng,omega,phi-phi.min(),L=1,Nh=1,n_e=80,umax=45.,m=m); f=kin.solve(E,B)
vth=kin.ang.unit_vectors()[0][1]; c=np.einsum('aij,ij,ij->a',kin.ang.Y,vth,kin.ang.W)
j1=-e*np.einsum('a,ik,aik->k',c,kin.w*kin.v,f[2])*1e17/np.sqrt(4*np.pi); n=kin.density(f)
sig=j1/E1
fv=CylKineticsFV(xs,R,ng,omega,L=1,Nh=1,nu=44,umax=50.,m=m,cluster=2.0); d=Discharge(fv,np.zeros(3,complex),1e17); ff=d.electrons(E,B,phi); jf=fv.current_theta(ff,1)*1e17; nf=fv.density(ff); sf=jf/E1
print(' X     |E1|   n_TE   |sig_TE| arg   |sig_FV| arg   Drude |sig| (n_TE, num at Te)')
for k in range(m):
    print(f'{X[k]:.2f} {abs(E1[k])*2:7.1f} {n[k]:.3f} {abs(sig[k]):8.3f} {np.angle(sig[k]):6.2f} {abs(sf[k]):8.3f} {np.angle(sf[k]):6.2f}')
