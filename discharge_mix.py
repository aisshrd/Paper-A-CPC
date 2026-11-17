"""Self-consistent ICP in a gas mixture (DischargeTE with the composite cross sections of ekin_cyl_mix).
usage: python3 discharge_mix.py <gas> <p_Pa> <Bext_G> [iters]   gas: 'N2' or 'Ar:0.9,N2:0.1'
env: MIX_NE (energy cells, 400), MIX_UMAX (30), MIX_N0 (1e17), MIX_START (pkl with E,B,phi,B_ext to start from), RELAX, PHIRELAX, TAG"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time, scipy.constants as co
os.environ.setdefault('OMP_NUM_THREADS','1')
from discharge_te import DischargeTE
from ekin_cyl_mix import mixture_xs, ion_params, ion_mobility
gas=sys.argv[1]; p=float(sys.argv[2]); Bg=float(sys.argv[3]); iters=int(sys.argv[4]) if len(sys.argv)>4 else 25
mix=[(t.split(':')[0],float(t.split(':')[1]) if ':' in t else 1.0) for t in gas.split(',')]
xs=mixture_xs(mix); M_amu,sig_i=ion_params(mix)
ion,M_amu,K0=ion_mobility(mix); os.environ.setdefault('ION_K0',str(K0)); print(f'dominant ion {ion}+ M={M_amu} K0={K0:.3f} cm2/Vs (ION_K0={os.environ["ION_K0"]}) ion_model={os.environ.get("ION_MODEL","kin")} teff_v2={os.environ.get("TEFF_V2","1")}',flush=True)
ng=p/(co.k*300); omega=2*np.pi*13.56e6; R=0.1; n0=float(os.environ.get('MIX_N0','1e17'))
n_e=int(os.environ.get('MIX_NE','400')); umax=float(os.environ.get('MIX_UMAX','30'))
Bext=np.zeros(3,complex); Bext[0]=Bext[2]=Bg*1e-4/2
d=DischargeTE(xs,R,ng,omega,Bext,n0,L=1,n_e=n_e,umax=umax,m=int(os.environ.get('MIX_M','16')),M_amu=M_amu,sigma_i=sig_i)
tag=os.environ.get('TAG',''); fn=f"results/discharge_mix_{gas.replace(':','').replace(',','_')}_p{p:g}_B{Bg:g}{tag}.pkl"
d.ckpt=fn.replace('.pkl','_ckpt.pkl'); d.ckpt_B=Bg
ref=pickle.load(open(os.environ.get('MIX_START','results/discharge_te_B0.76_n1e+17_L1_m16_o2.pkl'),'rb'))
sc=Bg/ref['B_ext']; phi0=ref['phi']*float(os.environ.get('MIX_PHISCALE','1.0')); E_st=ref['E']*sc; B_st=ref['B']*sc
if len(phi0)!=d.m:                                   # start file on another radial grid: interpolate onto the present nodes (wall first)
    from ekin_cyl_te import CylKineticsTE; Xs=ref['X'] if 'X' in ref else CylKineticsTE.grid(len(phi0)); X=d.k.X
    ip=lambda a: np.interp(X[::-1],Xs[::-1],a[::-1])[::-1]
    phi0=ip(phi0); E_st=np.array([ip(np.real(r))+1j*ip(np.imag(r)) for r in E_st]); B_st=np.array([ip(np.real(r))+1j*ip(np.imag(r)) for r in B_st]); print('start fields interpolated from m=%d to m=%d'%(len(Xs),d.m))
t=time.time(); d.run(iters=iters,relax=float(os.environ.get('RELAX','0.5')),phi_relax=float(os.environ.get('PHIRELAX','0.35')),E0=E_st,B0=B_st,phi0=phi0)
print('total %.0f s'%(time.time()-t),flush=True)
k=d.kin; f=d.f
out=dict(gas=gas,mix=mix,p=p,ng=ng,B_ext=Bg,n0=n0,n_e=n_e,umax=umax,X=k.X,E=d.E,B=d.B,phi=d.phi,ne=d.ne,ni=d.ni,Te=d.Te,nuiz=d.nuiz,nuiz_gross=k.nu_iz(f),nuatt=k.nu_att(f),hist=d.hist,balance=d.balance,converged=d.converged,n_iter=d.n_iter,
         F0=np.real(f[1,0]),ec=k.ec,ukin=k.ukin,w=k.w,wdisk=k.wdisk,n_neg=d.n_neg,f0_min_rel=d.f0_min_rel,
         rates={name:k.rate(f,nu) for (th,nu),(_,_,name) in zip(k.inel,xs['inel'])}, jt=d.current_theta(f,1)*n0, ion=ion, K0=K0, mu_i=d.mu_i, Teff=getattr(d,'Teff',None), ion_model=os.environ.get('ION_MODEL','kin'), m=d.m)
pickle.dump(out,open(fn,'wb')); print('saved',fn)
if os.path.exists(d.ckpt): os.remove(d.ckpt)
