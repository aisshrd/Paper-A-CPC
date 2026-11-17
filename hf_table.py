"""0D RF comparison at p: (a) energy-grid multi-term (ours, L=1 and L=2 via ekin Allis for L=1), (b) BOLSIG+-type effective-field DC
   (E_eff = E_rms * nu_m/sqrt(nu_m^2+w^2), evaluated with mean-energy-dependent nu_m, iterated), (c) Monte Carlo."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, scipy.constants as co, sys, os
from ekin import *
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6
def bolsig_hf(K,E0):
    """BOLSIG+ 'HF' convention: solve DC with E_eff^2 = (E0^2/2) * nu_m^2/(nu_m^2+w^2) using the EEDF-averaged momentum-transfer frequency; iterate."""
    Eeff=E0/np.sqrt(2); 
    for _ in range(30):
        f=K.solve(Eeff); f0=np.real(f[0,0]); num=np.sum(K.w*K.nu_m_c*f0)   # <nu_m>
        Enew=E0/np.sqrt(2)*num/np.sqrt(num**2+omega**2)
        if abs(Enew-Eeff)<1e-6*E0: break
        Eeff=0.5*(Eeff+Enew)
    return f,Eeff
for p in [13.3,40.,133.]:
    ng=p/(co.k*300.)
    K1=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=2,ng=ng); Kdc=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=0,ng=ng)
    mcf=f'results/mc_hf_ng{ng:.1e}.pkl'; mc=pickle.load(open(mcf,'rb')) if os.path.exists(mcf) else {}
    print(f'=== p={p} Pa, ng={ng:.2e}, nu_m/omega(5eV)~{ng*1e-19*GAMMA*np.sqrt(5)/omega:.1f}')
    rows=[]
    for E0 in [100.,200.,400.,700.,1000.]:
        f=K1.solve(E0,omega); fb,Eeff=bolsig_hf(Kdc,E0)
        r=mc.get(E0,None)
        rows.append((E0,K1.mean_energy(f),K1.rate_iz(f),Kdc.mean_energy(fb),Kdc.rate_iz(fb),Eeff, r['mean_energy'] if r else np.nan, r['nu_iz'] if r else np.nan))
        print(f'E0={E0:6.0f}: ours(HB,L1) <u>={rows[-1][1]:.3f} nuiz={rows[-1][2]:.3e} | BOLSIG-HF(Eeff={Eeff:.0f}) <u>={rows[-1][3]:.3f} nuiz={rows[-1][4]:.3e} | MC <u>={rows[-1][6]:.3f} nuiz={rows[-1][7]:.3e}')
    pickle.dump(rows,open(f'results/hf_table_p{p}.pkl','wb'))
