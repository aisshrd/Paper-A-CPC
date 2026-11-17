import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, time, sys, os
from ekin import *; from mc_real import MCReal
xs=load_argon('data/Ar_LXCat.txt'); ng=float(sys.argv[1]); omega=2*np.pi*13.56e6
out_path=f'results/mc_hf_ng{ng:.1e}.pkl'; out=pickle.load(open(out_path,'rb')) if os.path.exists(out_path) else {}
mc=MCReal(xs,ng)
for E0 in [float(x) for x in sys.argv[2:]]:
    if E0 in out: continue
    t=time.time(); r=mc.run(E0,omega,N=2000,t_total=4e-5,t_burn=2e-5,n_samples=270)
    r['time']=time.time()-t; out[E0]=r; pickle.dump(out,open(out_path,'wb'))
    print(f'ng={ng:.1e} E0={E0}: <u>={r["mean_energy"]:.3f} eV nu_iz={r["nu_iz"]:.4e} nu_ex={r["nu_ex"]:.4e} ({r["time"]:.0f}s)',flush=True)
