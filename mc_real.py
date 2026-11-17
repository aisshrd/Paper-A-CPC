"""Electron Monte Carlo with real argon cross sections, 3D velocity, uniform E(t)=E0 cos(wt) along z (0D).
Null-collision method with constant nu_max; free flight in E(t) is analytic. Isotropic scattering,
equal energy sharing at ionization, no secondary electrons (density growth ignored). Synchronous sampling."""
import numpy as np, scipy.constants as co
from ekin import load_argon, GAMMA
e,me=co.e,co.m_e
class MCReal:
    def __init__(self,xs,ng,Tg=300.,seed=1,umax_nu=60.):
        self.xs=xs; self.ng=ng; self.rng=np.random.default_rng(seed); self.mM=xs['mM']
        u=np.concatenate([[0],np.logspace(-3,2.3,1500)]); self.ugrid=u
        sig=lambda d,x: np.interp(x,d[:,0],d[:,1],left=d[0,1],right=d[-1,1])
        self.s_el=sig(xs['elastic'],u); self.inel=[(th,sig(d,u)*(u>th)) for th,d,_ in xs['inel']]
        thi,di=xs['iz']; self.th_iz=thi; self.s_iz=sig(di,u)*(u>thi)
        self.s_tot=self.s_el+sum(s for _,s in self.inel)+self.s_iz
        m=u<=umax_nu; self.nu_max=np.max((self.ng*self.s_tot*GAMMA*np.sqrt(u))[m])*1.1
        self.cum=[self.s_el]+[s for _,s in self.inel]+[self.s_iz]; self.ths=[None]+[th for th,_ in self.inel]+[self.th_iz]
    def _free_flight(self,v,t0,tau,E0,omega):
        if omega>0: v[:,2]+= -(e/me)*E0*(np.sin(omega*(t0+tau))-np.sin(omega*t0))/omega
        else:       v[:,2]+= -(e/me)*E0*tau
    def run(self,E0,omega,N=4000,t_total=None,t_burn=None,n_samples=400,seed=None):
        rng=self.rng if seed is None else np.random.default_rng(seed)
        v=rng.normal(0,np.sqrt(e*1.0/me),size=(N,3)); t=np.zeros(N)
        if t_total is None: t_total=(60*2*np.pi/omega) if omega>0 else 1e-5
        if t_burn is None: t_burn=0.5*t_total
        ts=np.linspace(t_burn,t_total,n_samples); S=dict(u=0.0,viz=0.0,vex=0.0,n=0); hist=np.zeros(300); hed=np.linspace(0,60,301)
        tnext=rng.exponential(1/self.nu_max,N)
        nb=max(int(n_samples*t_burn/(t_total-t_burn)),1)
        for ts_k in np.concatenate([np.linspace(0,t_burn,nb+1)[1:],ts]):
            while True:
                act=tnext<ts_k
                if not act.any(): break
                idx=np.where(act)[0]
                vv=v[idx]; self._free_flight(vv,t[idx],tnext[idx]-t[idx],E0,omega); v[idx]=vv; t[idx]=tnext[idx]
                self._collide(v,idx,rng)
                tnext[idx]=t[idx]+rng.exponential(1/self.nu_max,idx.size)
            self._free_flight(v,t,ts_k-t,E0,omega); t[:]=ts_k
            if ts_k>=t_burn:
                u=0.5*me*np.sum(v*v,1)/e; vel=np.sqrt(np.sum(v*v,1))
                S['u']+=u.mean(); S['n']+=1
                S['viz']+=np.mean(self.ng*np.interp(u,self.ugrid,self.s_iz)*vel)
                S['vex']+=np.mean(self.ng*sum(np.interp(u,self.ugrid,s) for _,s in self.inel)*vel)
                hist+=np.histogram(u,bins=hed)[0]
        n=S['n']; uc=0.5*(hed[1:]+hed[:-1]); eedf=hist/(n*N)/np.diff(hed)/np.sqrt(uc)
        return dict(mean_energy=S['u']/n,nu_iz=S['viz']/n,nu_ex=S['vex']/n,eedf=(uc,eedf))
    def _collide(self,v,idx,rng):
        u=0.5*me*np.sum(v[idx]**2,1)/e; vel=np.sqrt(np.sum(v[idx]**2,1))
        r=rng.random(idx.size)*self.nu_max; acc=np.zeros(idx.size); done=np.zeros(idx.size,bool)
        for k,(s,th) in enumerate(zip(self.cum,self.ths)):
            acc=acc+self.ng*np.interp(u,self.ugrid,s)*vel
            m=(~done)&(r<acc)
            if m.any():
                sel=idx[m]; n=sel.size
                if k==0:
                    cosx=1-2*rng.random(n); vnew=vel[m]*np.sqrt(np.maximum(1-2*self.mM*(1-cosx),0))
                else:
                    share=0.5 if th==self.th_iz else 1.0
                    vnew=np.sqrt(2*e*np.maximum(u[m]-th,0)*share/me)
                cth=1-2*rng.random(n); sth=np.sqrt(1-cth**2); ph=2*np.pi*rng.random(n)
                v[sel,0]=vnew*sth*np.cos(ph); v[sel,1]=vnew*sth*np.sin(ph); v[sel,2]=vnew*cth
            done|=m
