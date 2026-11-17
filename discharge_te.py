"""Self-consistent discharge with the TOTAL-ENERGY electron solver (ekin_cyl_te.CylKineticsTE):
Picard loop  electrons(E,B,phi) -> sigma -> Maxwell(E,B) -> ambipolar ions (balance) -> phi (quasineutrality), at fixed n0.
The kinetic object is rebuilt each iteration because the eps_tot grid depends on phi (0.5 s)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, scipy.constants as co, json, pickle, sys, time, os
from ekin import load_argon
from ekin_cyl import Discharge
from ekin_cyl_fv import CylKineticsFV
from ekin_cyl_te import CylKineticsTE
e=co.e

class DischargeTE(Discharge):
    def __init__(self,xs,R,ng,omega,B_ext,n0,L=1,n_e=120,umax=45.,m=16,cluster=2.0,wall_absorb=False,order=2,**kw):
        xs_h=xs if not isinstance(xs['iz'],list) else load_argon('data/Ar_LXCat.txt')   # the helper only supplies the radial grid and FD matrices
        helper=CylKineticsFV(xs_h,R,ng,omega,L=1,Nh=1,nu=8,umax=10.,m=m,cluster=cluster)     # radial grid + FD matrices for Maxwell / phi'
        super().__init__(helper,B_ext,n0,**kw)
        self.xs,self.R,self.ngas,self.omega=xs,R,ng,omega; self.L,self.n_e,self.umax,self.m,self.cluster,self.wall_absorb,self.order=L,n_e,umax,m,cluster,wall_absorb,order
        self.kin=None
    def electrons(self,E,B,phi):
        self.kin=CylKineticsTE(self.xs,self.R,self.ngas,self.omega,phi-phi.min(),L=self.L,Nh=1,n_e=self.n_e,umax=self.umax,m=self.m,cluster=self.cluster,wall_absorb=self.wall_absorb,order=self.order)
        assert np.allclose(self.kin.X,self.k.X)
        if self.order==2 and getattr(self,'prev_f',None) is not None:        # limiter frozen from the previous Picard iterate: one LU per iteration
            self.kin._theta=self.kin.limiter_theta(self.prev_f); f=self.kin._solve(E,B); self.kin._theta=None
        else: f=self.kin.solve(E,B,limiter_iters=(2 if self.order==2 else 0))
        self.prev_f=f; return f
    def moments(self,f):
        k=self.kin; return k.density(f),k.mean_energy(f)*2/3,k.nu_iz(f)-(k.nu_att(f) if np.any(k.nuatt) else 0.0)   # net ionization (attachment as loss; negative ions not transported)
    def current_theta(self,f,n=1):
        k=self.kin; vth=k.ang.unit_vectors()[0][1]; c=np.einsum('aij,ij,ij->a',k.ang.Y,vth,k.ang.W)
        return -e*np.einsum('a,ik,aik->k',c,k.w*k.v,f[k.Nh+n])/np.sqrt(4*np.pi)   # f normalised by sum_i w_i f_00 = n (relative): physical density = sqrt(4 pi) * that
    def T_eff(self,f):
        """Effective temperature of the electron response at fixed total-energy distribution: T_eff = n_e / (dn_e/dphi).
        TEFF_V2=1 (default from v1.7.0): exact cell integrals of F0(eps_tot) over the kinetic energy, continuous in phi.
        TEFF_V2=0: midpoint rule with the cell mask u > de/2 (v1, discontinuous when the lowest allowed cell switches -> two-cycle, Paper 4)."""
        k=self.kin; F0=np.real(f[k.Nh,0]); u=k.ukin; w=k.w
        if os.environ.get('TEFF_V2','1')=='1':
            de=k.de; up=np.sqrt(np.maximum(u+0.5*de,0.0)); lo=np.sqrt(np.maximum(u-0.5*de,0.0))
            num=np.sum(F0*(2.0/3.0)*(up**3-lo**3),axis=0); den=np.sum(F0*(up-lo),axis=0)
            return np.clip(num/np.maximum(den,1e-300),0.05,30.)
        ok=u>0.5*k.de
        num=np.sum(np.where(ok,w*F0,0.0),axis=0); den=np.sum(np.where(ok,w*F0/(2*np.maximum(u,1e-9)),0.0),axis=0)
        return np.clip(num/np.maximum(den,1e-300),0.05,30.)
    def closure(self,ne,Te,nuiz,phi,f=None):
        """Ion profile + potential increment. ION_MODEL=kin (default from v1.7.0): corrected closure (ambipolar_kin + Newton step with T_eff);
        ION_MODEL=old: v1.6 closure (ambipolar diffusion with the local T_e, dphi = T_e ln(n_i/n_e))."""
        k=self.k
        if os.environ.get('ION_MODEL','kin') not in ('kin','old'): raise ValueError('ION_MODEL must be kin or old')
        if os.environ.get('ION_MODEL','kin')=='kin' and f is not None:
            Teff=self.T_eff(f); self.Teff=Teff
            ni,lam=self.ambipolar_kin(ne,Teff,nuiz,phi); ni=np.maximum(ni,1e-4); ni=ni/(k.wdisk@ni)
            dphi=np.clip(Teff*np.log(ni/ne),-2.0,2.0); dphi-=dphi[0]
            if os.environ.get('DEBUG_PHI'): print('    DEBUG_PHI Teff wall..axis',np.array2string(Teff[::max(1,len(Teff)//8)],precision=2),' ni/ne wall',f'{ni[0]/ne[0]:.3f}',' dphi max',f'{np.max(np.abs(dphi)):.3f}',flush=True)
        else:
            ni,lam=self.ambipolar(ne,Te,nuiz); ni=np.maximum(ni,1e-4); ni=ni/(k.wdisk@ni)
            dphi=np.clip(Te*np.log(ni/ne),-2.0,2.0); dphi-=dphi[0]
        return ni,lam,dphi
    def run(self,iters=12,relax=0.6,verbose=True,phi_relax=0.5,phi_max=40.,E0=None,B0=None,phi0=None):
        k=self.k; m=k.m; H,Nh=k.H,k.Nh
        E=np.zeros((H,m),complex); B=np.zeros((H,m),complex)
        for n in (-1,1): B[n+Nh]=self.B_ext[n+Nh]; E[n+Nh]=-1j*n*k.omega*k.r*self.B_ext[n+Nh]/2
        if E0 is not None: E=E0.copy(); B=B0.copy()
        phi=np.zeros(m) if phi0 is None else phi0.copy(); self.Te_wall=3.0; hist=[]; bal=np.nan
        import os; mA=int(os.environ.get('ANDERSON','0')); Xh=[]; Gh=[]      # Anderson acceleration memory (type II, damped)
        def pack(E,B,phi): return np.concatenate([E.ravel().view(float)/self.sc[0],B.ravel().view(float)/self.sc[1],phi/self.sc[2]])
        def unpack(x):
            nE=2*E.size; Ev=(x[:nE]*self.sc[0]).view(complex).reshape(E.shape); Bv=(x[nE:2*nE]*self.sc[1]).view(complex).reshape(B.shape); return Ev,Bv,x[2*nE:]*self.sc[2]
        self.sc=(max(np.max(np.abs(E)),1e-30),max(np.max(np.abs(B)),1e-30),max(np.max(np.abs(phi)),1.0))
        for it in range(iters):
            t=time.time(); f=self.electrons(E,B,phi)
            ne,Te,nuiz=self.moments(f); ne=np.maximum(ne,1e-6); Te=np.clip(Te,0.03,30.); nuiz=np.maximum(nuiz,0)
            j1=self.current_theta(f,1)*self.n0
            sigma=j1/np.where(np.abs(E[Nh+1])>1e-12,E[Nh+1],1e-12)
            sigma=np.where(np.abs(E[Nh+1])>1e-3*np.max(np.abs(E[Nh+1])),sigma,sigma[0])
            Enew,Bnew=self.maxwell(sigma)
            self.Te_wall=max(Te[0],0.5)
            ni,lam,dphi=self.closure(ne,Te,nuiz,phi,f)
            bal=1.0/lam
            dE=float(np.max(np.abs(Enew-E))/max(np.max(np.abs(Enew)),1e-30))
            if mA>0:
                x=pack(E,B,phi); g=pack(Enew,Bnew,np.clip(phi+dphi,-phi_max,phi_max)); r=g-x; Xh.append(x); Gh.append(g)
                if len(Xh)>mA+1: Xh.pop(0); Gh.pop(0)
                if len(Xh)>1:
                    dR=np.array([ (Gh[i+1]-Xh[i+1])-(Gh[i]-Xh[i]) for i in range(len(Xh)-1)]).T
                    dX=np.array([Xh[i+1]-Xh[i] for i in range(len(Xh)-1)]).T
                    gam=np.linalg.lstsq(dR,r,rcond=None)[0]; xn=x+relax*r-(dX+relax*dR)@gam
                else: xn=x+relax*r
                E,B,phi=unpack(xn)
            else:
                phi=np.clip(phi+phi_relax*dphi,-phi_max,phi_max)
                E=(1-relax)*E+relax*Enew; B=(1-relax)*B+relax*Bnew
            hist.append((it,dE,bal,Te[0],Te[-1],float(np.max(np.abs(dphi)))))
            if getattr(self,'ckpt',None): pickle.dump(dict(E=E,B=B,phi=phi,it=it,hist=hist,B_ext=getattr(self,'ckpt_B',None)),open(self.ckpt,'wb'))
            if verbose: print(f'  it{it} ({time.time()-t:.0f}s): dE={dE:.3f} dphi={np.max(np.abs(dphi)):.2f} Te wall/max/axis {Te[0]:.2f}/{Te.max():.2f}/{Te[-1]:.2f} eV, nuiz wall/axis {nuiz[0]:.2e}/{nuiz[-1]:.2e}, |E|wall {abs(E[Nh+1][0])*2:.0f} V/m, ne axis/wall {ne[-1]/ne[0]:.2f}, ni axis/wall {ni[-1]/ni[0]:.2f}, phi axis {phi[-1]:.2f} V, balance {bal:.3f}, lam_cons {self.kin.lam.real:.1e}',flush=True)
            if it>3 and dE<2e-3 and np.max(np.abs(dphi))<0.05: break
        self.converged=bool(dE<2e-3 and np.max(np.abs(dphi))<0.05); self.n_iter=it+1
        # final solve at the converged fields with the limiter re-evaluated on the current solution (no frozen theta, no clipping)
        self.kin=CylKineticsTE(self.xs,self.R,self.ngas,self.omega,phi-phi.min(),L=self.L,Nh=1,n_e=self.n_e,umax=self.umax,m=self.m,cluster=self.cluster,wall_absorb=self.wall_absorb,order=self.order)
        f=self.kin.solve(E,B,limiter_iters=3); ne,Te,nuiz=self.moments(f); F0=np.real(f[1,0]); al=self.kin.allowed
        self.f0_min_rel=float((F0/np.max(F0,axis=0)).min()); self.n_neg=int(np.sum(F0[al]<0)); self.nuiz_raw=nuiz.copy()
        ni,lam,dphi=self.closure(np.maximum(ne,1e-6),np.clip(Te,0.03,30.),np.maximum(nuiz,0),phi,f); bal=1.0/lam
        if verbose: print(f'  final re-limited solve: Te wall/max/axis {Te[0]:.2f}/{Te.max():.2f}/{Te[-1]:.2f}, nuiz wall {nuiz[0]:.2e} {nuiz[1]:.2e}, f0<0 cells {self.n_neg}, min f0/max {self.f0_min_rel:.1e}, balance {bal:.3f}, converged {self.converged} after {self.n_iter} it',flush=True)
        self.f,self.E,self.B,self.phi,self.ni,self.ne,self.Te,self.nuiz,self.hist,self.balance=f,E,B,phi,ni,ne,Te,nuiz,hist,bal
        return self

if __name__=='__main__':
    os.environ.setdefault('ION_K0','1.53')   # argon: measured reduced mobility of Ar+ (Ellis et al. 1976), used for all published runs; ION_K0= (empty) selects the crude estimate
    xs=load_argon('data/Ar_LXCat.txt'); ng=3.21e21; omega=2*np.pi*13.56e6; R=0.1
    Bg=float(sys.argv[1]) if len(sys.argv)>1 else 1.3; n0=float(sys.argv[2]) if len(sys.argv)>2 else 1e17; L=int(sys.argv[3]) if len(sys.argv)>3 else 1; m=int(sys.argv[4]) if len(sys.argv)>4 else 16; n_e=int(sys.argv[5]) if len(sys.argv)>5 else 120
    Bext=np.zeros(3,complex); Bext[0]=Bext[2]=Bg*1e-4/2
    d=DischargeTE(xs,R,ng,omega,Bext,n0,L=L,n_e=n_e,m=m)
    # start from the reference (kinetic-energy) self-consistent fields to shorten the Picard path
    ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R']); X=d.k.X
    def interp(key,ang=None):
        a=np.array(ref[key]);
        return np.interp(X,rr,a) if ang is None else np.interp(X,rr,a)*np.exp(1j*np.interp(X,rr,np.array(ref[ang])))
    E1=interp('E_theta_amplitude_V_per_m','E_theta_phase_rad')/2*Bg/1.3; B1=interp('B_z_amplitude_T','B_z_phase_rad')/2*Bg/1.3; phi=interp('phi_V')
    E0=np.zeros((3,m),complex); E0[2]=E1; E0[0]=np.conj(E1); B0=np.zeros((3,m),complex); B0[2]=B1; B0[0]=np.conj(B1)
    if os.environ.get('START'):                      # warm start from a previous self-consistent run (fields scaled by the coil-current ratio)
        s=pickle.load(open(os.environ['START'],'rb')); fac=Bg/s['B_ext']; E0=s['E']*fac; B0=s['B']*fac; phi=s['phi'].copy(); print('warm start from',os.environ['START'],'scale',fac)
    if os.environ.get('COLD')=='1': E0=B0=phi=None; print('cold start (vacuum fields, phi=0)')
    d.ckpt=f'results/ckpt_te_B{Bg}{os.environ.get("TAG","")}.pkl'; d.ckpt_B=Bg
    t=time.time(); d.run(iters=int(sys.argv[6]) if len(sys.argv)>6 else 30,relax=float(os.environ.get('RELAX','0.5')),phi_relax=float(os.environ.get('PHIRELAX','0.35')),E0=E0,B0=B0,phi0=phi); print('total',time.time()-t,'s')
    pickle.dump(dict(X=d.k.X,E=d.E,B=d.B,phi=d.phi,ne=d.ne,ni=d.ni,Te=d.Te,nuiz=d.nuiz,hist=d.hist,balance=d.balance,converged=d.converged,n_iter=d.n_iter,n_neg=d.n_neg,f0_min_rel=d.f0_min_rel,nuiz_raw=d.nuiz_raw,B_ext=Bg,n0=n0,L=L,m=m,n_e=n_e,
                     F0=np.real(d.f[1,0]),ec=d.kin.ec,ukin=d.kin.ukin,w=d.kin.w,Teff=getattr(d,'Teff',None),ion_model=os.environ.get('ION_MODEL','kin'),teff_v2=os.environ.get('TEFF_V2','1'),mu_i=d.mu_i),open(f'results/discharge_te_B{Bg}_n{n0:.0e}_L{L}_m{m}{"_o2" if d.order==2 else ""}{"_and"+os.environ["ANDERSON"] if os.environ.get("ANDERSON") else ""}{os.environ.get("TAG","")}.pkl','wb'))
