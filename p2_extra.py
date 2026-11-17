"""Paper 2 (v1.7.0) auxiliary numbers: (1) second time harmonic in a uniform RF field, nu_iz(N_t=2)/nu_iz(N_t=1) and the effective-field
value; (2) EEPF of the kinetic solution in the reference fields vs the uniform-RF EEPF with the same mean energy (-> results/p3_eepf_cmp.pkl);
(3) slope temperatures of f0 in energy bands and tail fractions at four radii; (4) characteristic lengths per pressure; (5) e-e estimates.
usage: python3 p2_extra.py   -> results/p2_extra.json and printed tables"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, os, scipy.constants as co
os.environ.setdefault('OMP_NUM_THREADS','1')
from ekin import load_argon, EnergyGridKinetics
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6; e=co.e; me=co.m_e; M=39.95*co.atomic_mass; out={}
sig=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1],right=d[-1,1])
# ---------- (1) N_t = 1 vs 2 in a uniform RF field
rows=[]
for p,E0 in [(13.3,200),(13.3,400),(40,400),(133,700),(133,1000)]:
    ng=p/(co.k*300); r=[]
    for Nh in (1,2):
        K=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=Nh,ng=ng); f=K.solve(E0,omega); r.append((K.mean_energy(f),K.rate_iz(f)))
    # effective field: DC solve at E_eff with <nu_m> from the N_t=1 EEDF
    K1=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=1,ng=ng); f1=K1.solve(E0,omega); f0=np.real(f1[K1.Nh,0]); num=ng*sig(xs['elastic'],K1.uc)*np.sqrt(2*e*K1.uc/me)
    nm=float(np.sum(K1.w*num*f0)/np.sum(K1.w*f0)); Eeff=E0*nm/np.sqrt(2*(nm**2+omega**2)); Kdc=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=0,ng=ng); fd=Kdc.solve(Eeff,0.0)
    rows.append(dict(p=p,E0=E0,u1=r[0][0],u2=r[1][0],nu1=r[0][1],nu2=r[1][1],ratio=r[1][1]/r[0][1],uEF=Kdc.mean_energy(fd),nuEF=Kdc.rate_iz(fd),num_over_omega=nm/omega))
    print(f"p={p} E0={E0}: <u> Nt1/Nt2/EF {r[0][0]:.3f}/{r[1][0]:.3f}/{Kdc.mean_energy(fd):.3f}  nu_iz Nt1/Nt2/EF {r[0][1]:.4g}/{r[1][1]:.4g}/{Kdc.rate_iz(fd):.4g}  ratio Nt2/Nt1 {r[1][1]/r[0][1]:.3f}  <nu_m>/omega {nm/omega:.2f}",flush=True)
out['nt']=rows
# ---------- (2) EEPF comparison in the reference fields (13.3 Pa)
s=pickle.load(open('results/te_reffields_ne120_m16_L1_o2.pkl','rb')); F0=np.real(s['F0']); u=s['ukin']; w=s['w']; X=s['X']; phi=s['phi']; ng=3.21e21
K=EnergyGridKinetics(xs,umax=50.,nu=250,L=1,Nh=1,ng=ng); cmp=[]
def hom_at_energy(ub):
    lo,hi=1.0,2000.0
    for _ in range(40):
        mid=np.sqrt(lo*hi); f=K.solve(mid,omega); um=K.mean_energy(f)
        if um<ub: lo=mid
        else: hi=mid
    f=K.solve(mid,omega); g0=np.real(f[K.Nh,0]); n=np.sum(K.w*g0); return mid,K.uc,g0/n,K.mean_energy(f)
for j in range(len(X)):
    ok=u[:,j]>0; n=np.sum(w[ok,j]*F0[ok,j]); ub=np.sum(w[ok,j]*u[ok,j]*F0[ok,j])/n
    if j in (0,8,15):
        E0,uc,g0,um=hom_at_energy(ub); cmp.append(dict(j=j,X=float(X[j]),ub=float(ub),E0=float(E0),u_kin=u[ok,j],eepf_kin=F0[ok,j]/n,u_hom=uc,eepf_hom=g0,um=float(um)))
        print(f'EEPF cmp cell {j} r/R={X[j]:.2f}: <u>={ub:.3f} eV, uniform field E0={E0:.1f} V/m (<u>={um:.3f})',flush=True)
    else: cmp.append(dict(j=j,X=float(X[j]),ub=float(ub)))
pickle.dump(cmp,open('results/p3_eepf_cmp.pkl','wb'))
# ---------- (3) slope temperatures and tail fractions
bands=[(1,5),(5,11),(11.5,15.7),(16,22)]; slopes=[]
for j in (0,4,7,15):
    ok=u[:,j]>0; uu=u[ok,j]; f=F0[ok,j]; n=np.sum(w[ok,j]*f); row=dict(X=float(X[j]),phi=float(phi[j]))
    for a,b in bands:
        m=(uu>=a)&(uu<=b)&(f>0); c=np.polyfit(uu[m],np.log(f[m]),1); row[f'T_{a}_{b}']=float(-1/c[0])
    row['n_gt_11.55']=float(np.sum(w[ok,j][uu>11.55]*f[uu>11.55])/n); row['n_gt_15.76']=float(np.sum(w[ok,j][uu>15.76]*f[uu>15.76])/n)
    slopes.append(row); print('slopes',{k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items()})
out['slopes']=slopes
# ---------- (4) characteristic lengths
lens=[]
for p in (2,13.3,40,133):
    ng=p/(co.k*300); lam=lambda U: 1/(ng*sig(xs['elastic'],U));
    stot=lambda U: sig(xs['elastic'],U)+sum(sig(d,U)*(U>th) for th,d,_ in xs['inel'])+sig(xs['iz'][1],U)*(U>xs['iz'][0])
    sinel=lambda U: sum(sig(d,U)*(U>th) for th,d,_ in xs['inel'])+sig(xs['iz'][1],U)*(U>xs['iz'][0])
    le=lambda U: lam(U)*np.sqrt(M/(6*me)); lstar=1/(ng*stot(16.0)); linel=1/(ng*sinel(16.0)); lse=np.sqrt(lstar*linel/3)
    num5=ng*sig(xs['elastic'],5.0)*np.sqrt(2*e*5.0/me)
    lens.append(dict(p=p,lam3=lam(3.0)*100,le1=le(1.0)*100,le3=le(3.0)*100,le5=le(5.0)*100,lstar16=lstar*100,linel16=linel*100,lse16=lse*100,num_omega=num5/omega))
    print(f"p={p}: lam(3eV)={lam(3.0)*100:.2f} cm, lam_eps(1/3/5 eV)={le(1.0)*100:.0f}/{le(3.0)*100:.0f}/{le(5.0)*100:.0f} cm, lam*(16)={lstar*100:.2f} cm, lam_inel(16)={linel*100:.2f}, lam*_eps(16)={lse*100:.2f} cm, nu_m(5eV)/omega={num5/omega:.2f}")
out['lengths']=lens
# ---------- (5) e-e vs elastic/inelastic frequencies at n=1e17
n0=1e17; lnL=12.0; ee=[]
for U in (0.3,0.5,1,2,3,5,8,12,16,20):
    v=np.sqrt(2*e*U/me); nuee=n0*e**4*lnL/(4*np.pi*co.epsilon_0**2*me**2*v**3)   # Spitzer-type estimate, s^-1
    ng=3.21e21; nel=2*me/M*ng*sig(xs['elastic'],U)*v; ninel=ng*sinel(U)*v
    ee.append(dict(U=U,nu_ee=nuee,nu_el_energy=nel,nu_inel=ninel)); print(f'U={U}: nu_ee={nuee:.3g} (2m/M)nu_m={nel:.3g} nu_inel={ninel:.3g}')
out['ee']=ee
json.dump(out,open('results/p2_extra.json','w'),indent=1,default=float)
