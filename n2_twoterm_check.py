"""N2, uniform RF field 13.56 MHz, 13.3 Pa: mean energy and ionization frequency vs field amplitude from the 0D two-term solver,
the cylinder solver with L = 1, 2 (uniform field, no potential) and Monte Carlo.  -> results/n2_twoterm.pkl"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, subprocess, os, sys, time
from ekin_cyl_te import CylKineticsTE
from ekin_cyl_mix import mixture_xs
from ekin_mix import MixtureKinetics, load_species
om=2*np.pi*13.56e6; ng=3.21e21; xs=mixture_xs([('N2',1.0)]); out={}
E0s=[float(x) for x in (sys.argv[1].split(',') if len(sys.argv)>1 else ['141','200','283','400','566'])]
for E0 in E0s:
    r={}
    K=MixtureKinetics([(load_species('N2'),1.0)],umax=45.,nu=600,L=1,Nh=1,ng=ng,growth=False); f0=K.solve(E0,om); r['0D']=(K.mean_energy(f0),K.rate_iz(f0))
    mm=4; E=np.zeros((3,mm),complex); E[2]=E[0]=E0/2; B=np.zeros((3,mm),complex)
    for L,ne_ in ((1,320),(2,240),(3,200)):
        t=time.time(); k=CylKineticsTE(xs,0.1,ng,om,np.zeros(mm),L=L,Nh=1,n_e=ne_,umax=30.,m=mm); f=k.solve(E,B); r[f'L{L}']=(k.mean_energy(f).mean(),k.nu_iz(f).mean(),time.time()-t)
        print(E0,L,r[f'L{L}'],flush=True)
    fn=f'results/mc_n2_uniform_E{E0:g}.pkl'
    if not os.path.exists(fn):
        env=dict(os.environ,MCGAS='N2',MCTEST='0',MCE=str(E0),MCNG=str(ng),MCB0='0',MCOUT=fn); subprocess.run(['python3','mc_cyl_mag.py','3000','1','4e-5'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    m=pickle.load(open(fn,'rb')); w=m['ne_rel']*m['r_over_R']; r['MC']=(1.5*np.sum(w*m['Te_eV'])/np.sum(w),np.sum(w*m['nu_iz'])/np.sum(w))
    out[E0]=r; print(E0,{k:(round(v[0],3),round(v[1],1)) for k,v in r.items()},flush=True)
    pickle.dump(out,open('results/n2_twoterm.pkl','wb'))
