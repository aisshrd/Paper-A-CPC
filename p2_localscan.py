"""Paper 2: nonlocal (TE solver, L=1) vs local field approximation in a prescribed skin-layer field (delta=3.3 cm, no potential)
at several pressures. usage: python p2_localscan.py <p_Pa> <E_wall V/m>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, time, pickle, sys, os, scipy.constants as co
NH=int(os.environ.get('NH','1')); H=2*NH+1
from scipy.special import iv
from ekin import load_argon, EnergyGridKinetics
from ekin_cyl_te import CylKineticsTE
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6; R=0.1; m=16
p=float(sys.argv[1]); Ew=float(sys.argv[2]); ng=p/(co.k*300.)
kin=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=1,Nh=NH,n_e=120,umax=45.,m=m,order=2)
kk=(1+1j)/0.033; E1=Ew/2*iv(1,kk*kin.r)/iv(1,kk*R); E=np.zeros((H,m),complex); E[NH+1]=E1; E[NH-1]=np.conj(E1)
B1=-(1/(1j*omega))*(np.gradient(kin.r*E1,kin.r)/kin.r); B=np.zeros((H,m),complex); B[NH+1]=B1; B[NH-1]=np.conj(B1)
t=time.time(); f=kin.solve(E,B); out=dict(p=p,Ew=Ew,Nh=NH,X=kin.X,E1=E1,Te=kin.mean_energy(f)*2/3,nuiz=kin.nu_iz(f),ne=kin.density(f),time=time.time()-t)
K=EnergyGridKinetics(xs,umax=50.,nu=200,L=1,Nh=NH,ng=ng); loc=[]
t=time.time()
for e1 in np.abs(E1):
    g=K.solve(2*e1,omega); loc.append((K.mean_energy(g)*2/3,K.rate_iz(g)))
loc=np.array(loc); out['local']=dict(Te=loc[:,0],nuiz=loc[:,1],time=time.time()-t)
lam=1/(ng*np.interp(5.0,xs['elastic'][:,0],xs['elastic'][:,1])); out['lam_eps']=lam*np.sqrt(1/(6*xs['mM']))
print(f'p={p} Pa Ew={Ew}: lam_eps={out["lam_eps"]*100:.1f} cm; nonlocal Te',np.round(out['Te'],2)); print('  local Te',np.round(loc[:,0],2))
print('  nonlocal nuiz',np.array2string(out['nuiz'],precision=2,max_line_width=250)); print('  local nuiz',np.array2string(loc[:,1],precision=2,max_line_width=250))
pickle.dump(out,open(f'results/p2_local_p{p:g}{"_nh2" if NH==2 else ""}.pkl','wb'))
