"""N2 uniform field: cylinder solver with time harmonics Nh = 1, 2 and L = 1, 2 -> results/n2_nh.pkl"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, time, sys, os
from ekin_cyl_te import CylKineticsTE
from ekin_cyl_mix import mixture_xs
om=2*np.pi*13.56e6; ng=3.21e21; xs=mixture_xs([('N2',1.0)]); out=pickle.load(open('results/n2_nh.pkl','rb')) if os.path.exists('results/n2_nh.pkl') else {}
for E0 in [float(x) for x in sys.argv[1].split(',')]:
    for L,Nh,ne_ in [tuple(int(x) for x in t.split(':')) for t in os.environ.get('NHSET','1:2:320,2:2:240').split(',')]:
        if (E0,L,Nh) in out: continue
        mm=4; H=2*Nh+1; E=np.zeros((H,mm),complex); E[Nh+1]=E[Nh-1]=E0/2; B=np.zeros((H,mm),complex)
        t=time.time(); k=CylKineticsTE(xs,0.1,ng,om,np.zeros(mm),L=L,Nh=Nh,n_e=ne_,umax=30.,m=mm); f=k.solve(E,B)
        out[(E0,L,Nh)]=(float(k.mean_energy(f).mean()),float(k.nu_iz(f).mean()),time.time()-t); print(E0,L,Nh,out[(E0,L,Nh)],flush=True)
        pickle.dump(out,open('results/n2_nh.pkl','wb'))
