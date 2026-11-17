"""Paper 3: spatially resolved coefficients in the cylinder (13.3 Pa, prescribed self-consistent fields):
nonlocal (TE solver) vs local-field approximation (0D RF solver at the local field) vs local-mean-energy approximation
(0D DC table looked up at the local mean energy). Output results/p3_cylcoef.pkl"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, json, pickle, time, scipy.constants as co
from ekin import load_argon, EnergyGridKinetics, GAMMA
from ekin_cyl_te import CylKineticsTE
e=co.e; xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1; m=16; n0=1e17
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
tmp=CylKineticsTE(xs,R,ng,omega,np.zeros(m),L=1,Nh=1,n_e=10,umax=5.,m=m); X=tmp.X
def interp(key,ang=None):
    a=np.array(ref[key]); return np.interp(X,rr,a) if ang is None else np.interp(X,rr,a)*np.exp(1j*np.interp(X,rr,np.array(ref[ang])))
E1=interp('E_theta_amplitude_V_per_m','E_theta_phase_rad')/2; B1=interp('B_z_amplitude_T','B_z_phase_rad')/2; phi=interp('phi_V'); phi=phi-phi.min()
E=np.zeros((3,m),complex); E[2]=E1; E[0]=np.conj(E1); B=np.zeros((3,m),complex); B[2]=B1; B[0]=np.conj(B1)
t=time.time(); kin=CylKineticsTE(xs,R,ng,omega,phi,L=1,Nh=1,n_e=120,umax=45.,m=m,order=2); f=kin.solve(E,B); t_nl=time.time()-t
n=kin.density(f); n=n/np.sum(kin.wdisk*n); Te=kin.mean_energy(f)*2/3; nuiz=kin.nu_iz(f)
# total excitation frequency per cell: sum over inelastic processes (parent cells) -- use the solver's inelastic frequencies if stored
sigx=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1],right=d[-1,1])
uk=np.maximum(kin.ukin,0); nuex=sum(ng*sigx(d,uk)*GAMMA*np.sqrt(uk)*(uk>th) for th,d,_ in xs['inel'])
F0=np.real(f[1,0]); dens=np.einsum('ik,ik->k',kin.w,F0); nuex_r=np.einsum('ik,ik->k',kin.w*nuex,F0)/dens
vth=kin.ang.unit_vectors()[0][1]; c=np.einsum('aij,ij,ij->a',kin.ang.Y,vth,kin.ang.W)
j1=-e*np.einsum('a,ik,aik->k',c,kin.w*kin.v,f[2])*n0/np.sqrt(4*np.pi)   # current density amplitude; f normalised by sum w f_00 = n, physical density = sqrt(4pi) * that
sigma=j1/E1                                                                              # local kinetic conductivity (complex), n0*n(r) electrons
# --- LFA: 0D RF solver at the local field amplitude
K=EnergyGridKinetics(xs,umax=50.,nu=200,L=1,Nh=1,ng=ng); lfa=[]; t=time.time()
for k in range(m):
    g=K.solve(2*abs(E1[k]),omega); mu=K.drift(g,1)/(abs(E1[k])+1e-12)
    lfa.append((K.mean_energy(g)*2/3,K.rate_iz(g),K.rate_exc_total(g),abs(e*n0*n[k]*mu),np.angle(mu)))
lfa=np.array(lfa); t_lfa=time.time()-t
# --- LMEA: DC table (mean energy -> rates), looked up at the nonlocal local mean energy 1.5*Te(r)
Kd=EnergyGridKinetics(xs,umax=50.,nu=200,L=1,Nh=0,ng=ng); tab=[]
for EN in np.logspace(-0.5,2.6,40):
    g=Kd.solve(EN*1e-21*ng); tab.append((Kd.mean_energy(g),Kd.rate_iz(g),Kd.rate_exc_total(g)))
tab=np.array(tab); o=np.argsort(tab[:,0]); tab=tab[o]
ub=1.5*Te; lmea_iz=np.exp(np.interp(ub,tab[:,0],np.log(np.maximum(tab[:,1],1e-30)))); lmea_ex=np.exp(np.interp(ub,tab[:,0],np.log(np.maximum(tab[:,2],1e-30))))
# --- LMEA with an RF table (uniform RF field at 13.56 MHz and 13.3 Pa, harmonic balance), which is the appropriate table for an RF discharge
tabrf=[]
for E0 in np.logspace(0.0,3.2,50):
    g=K.solve(E0,omega); tabrf.append((K.mean_energy(g),K.rate_iz(g),K.rate_exc_total(g)))
tabrf=np.array(tabrf); o=np.argsort(tabrf[:,0]); tabrf=tabrf[o]
lmea_iz_rf=np.exp(np.interp(ub,tabrf[:,0],np.log(np.maximum(tabrf[:,1],1e-30)))); lmea_ex_rf=np.exp(np.interp(ub,tabrf[:,0],np.log(np.maximum(tabrf[:,2],1e-30))))
# --- Monte Carlo reference
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
out=dict(X=X,E1=E1,phi=phi,n=n,Te=Te,nuiz=nuiz,nuex=nuex_r,sigma=sigma,t_nl=t_nl,lfa=dict(Te=lfa[:,0],nuiz=lfa[:,1],nuex=lfa[:,2],sig_abs=lfa[:,3],sig_arg=lfa[:,4],time=t_lfa),
         lmea=dict(nuiz=lmea_iz,nuex=lmea_ex,table=tab),lmea_rf=dict(nuiz=lmea_iz_rf,nuex=lmea_ex_rf,table=tabrf),mc=dict(r=m1['r_over_R'],Te=(m1['Te_eV']+m2['Te_eV'])/2,nuiz=(m1['nu_iz']+m2['nu_iz'])/2))
for k in range(m): print(f"X={X[k]:.2f} |E|={2*abs(E1[k]):6.1f} Te NL/LFA {Te[k]:.2f}/{lfa[k,0]:.2f}  nuiz NL/LFA/LMEA(DC)/LMEA(RF) {nuiz[k]:.3g}/{lfa[k,1]:.3g}/{lmea_iz[k]:.3g}/{lmea_iz_rf[k]:.3g}  nuex NL/LFA/LMEA {nuex_r[k]:.3g}/{lfa[k,2]:.3g}/{lmea_ex[k]:.3g}  |sigma| NL/LFA {abs(sigma[k]):.2f}/{lfa[k,3]:.2f} arg {np.angle(sigma[k]):.2f}/{lfa[k,4]:.2f}")
pickle.dump(out,open('results/p3_cylcoef.pkl','wb'))
