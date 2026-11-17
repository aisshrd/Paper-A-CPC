"""Figures for the Lobachevskii J. Math. manuscript: one figure per EPS file, 17 cm wide, distinguishable in black-and-white."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, os, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1] if len(sys.argv)>1 else 'ljm/figs'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,
                     'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,
                     'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name):
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
# ---------------- Fig. 1: DC swarm parameters vs BOLOS
rows=pickle.load(open('results/dc_table.pkl','rb'))
fig,ax=plt.subplots(1,3,figsize=(W,2.2))
st1=dict(color='k',ls='-',marker='o',ms=3); st2=dict(color='0.45',ls='--',marker='s',mfc='none',ms=5)
ax[0].semilogx(rows[:,0],rows[:,1],label='this work',**st1); ax[0].semilogx(rows[:,0],rows[:,2],label='BOLOS (BOLSIG+ algorithm)',**st2)
ax[0].set(xlabel=r'$E/N$, Td',ylabel=r'$\langle\epsilon\rangle$, eV'); ax[0].legend(loc='upper left')
ok=rows[:,3]>1e-25; ax[1].loglog(rows[ok,0],rows[ok,3],**st1); ok=rows[:,4]>1e-25; ax[1].loglog(rows[ok,0],rows[ok,4],**st2)
ax[1].set(xlabel=r'$E/N$, Td',ylabel=r'$k_{\rm iz}$, m$^3$ s$^{-1}$')
ax[2].semilogx(rows[:,0],rows[:,5]/1e24,**st1); ax[2].semilogx(rows[:,0],rows[:,6]/1e24,**st2); ax[2].set(xlabel=r'$E/N$, Td',ylabel=r'$\mu N$, $10^{24}$ (m V s)$^{-1}$')
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig1')
# ---------------- Fig. 2: uniform RF field, 0D
fig,ax=plt.subplots(1,3,figsize=(W,2.8))
for p,mk,col in [(13.3,'o','k'),(40.,'s','0.35'),(133.,'^','0.6')]:
    r=np.array(pickle.load(open(f'results/hf_table_p{p}.pkl','rb')),dtype=float); E=r[:,0]
    ax[0].plot(E,r[:,1],'-',color=col,label=f'{p:g} Pa, harmonic balance'); ax[0].plot(E,r[:,3],'--',color=col,label=f'{p:g} Pa, effective field')
    ok=r[:,2]>0; ax[1].semilogy(E[ok],r[ok,2],'-',color=col); ok=r[:,4]>0; ax[1].semilogy(E[ok],r[ok,4],'--',color=col)
    mcf={13.3:'results/mc_hf_ng3.2e+21.pkl',40.:'results/mc_hf_ng9.7e+21.pkl'}.get(p)
    if mcf:
        mcd=pickle.load(open(mcf,'rb')); Em=np.array(sorted(mcd)); um=np.array([mcd[e]['mean_energy'] for e in Em]); nm_=np.array([mcd[e]['nu_iz'] for e in Em])
        ax[0].plot(Em,um,mk,color=col,mfc='none',ls='none',label=f'{p:g} Pa, Monte Carlo'); okm=nm_>0; ax[1].semilogy(Em[okm],nm_[okm],mk,color=col,mfc='none',ls='none')
ax[0].set(xlabel=r'$E_0$, V/m',ylabel=r'$\langle\epsilon\rangle$, eV')
ax[1].set(xlabel=r'$E_0$, V/m',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$')
sys.path.insert(0,'.'); from ekin import load_argon, EnergyGridKinetics
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6; ng=3.2e21
K=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=2,ng=ng); f=K.solve(400.,omega)
mc=pickle.load(open(f'results/mc_hf_ng{ng:.1e}.pkl','rb'))[400.0]; uc,fe=mc['eedf']
ax[2].semilogy(K.uc,np.maximum(K.eedf(f),1e-12),'k-',label='harmonic balance'); ax[2].semilogy(uc,np.maximum(fe,1e-12),'o',color='0.4',mfc='none',ms=2.5,label='Monte Carlo')
ax[2].set(xlabel=r'$\epsilon$, eV',ylabel=r'EEPF, eV$^{-3/2}$',xlim=(0,45),ylim=(1e-7,1)); ax[2].legend()
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
h,lb=ax[0].get_legend_handles_labels(); fig.legend(h,lb,loc='lower center',ncol=4,fontsize=6,frameon=False)
fig.tight_layout(pad=0.4,rect=(0,0.17,1,1)); fig.savefig(f'{OUT}/Fig2.eps'); fig.savefig(f'{OUT}/Fig2.pdf'); fig.savefig(f'{OUT}/Fig2.png',dpi=200); plt.close(fig)
# ---------------- Fig. 3: two-term / multi-term / local in a prescribed skin layer
d1=pickle.load(open('results/twoterm_te_L1.pkl','rb')); d2=pickle.load(open('results/twoterm_te_L2.pkl','rb')); X=d1['X']
fig,ax=plt.subplots(1,2,figsize=(W,2.5))
ax[0].plot(X,d1['Te'],'k-',label=r'nonlocal, $L=1$'); ax[0].plot(X,d2['Te'],'--',color='0.4',label=r'nonlocal, $L=2$'); ax[0].plot(X,d1['local']['Te'],'k:',lw=1.6,label='local (pointwise 0D)')
ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e=2\langle\epsilon\rangle/3$, eV'); ax[0].legend()
ax[1].semilogy(X,d1['nuiz'],'k-'); ax[1].semilogy(X,d2['nuiz'],'--',color='0.4'); ax[1].semilogy(X,np.maximum(d1['local']['nuiz'],1e-8),'k:',lw=1.6)
ax[1].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1e-7,1e5))
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig5')
# ---------------- Fig. 4: cylinder in the reference fields vs Monte Carlo
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
rm=m1['r_over_R']; Tm=(m1['Te_eV']+m2['Te_eV'])/2; num=(m1['nu_iz']+m2['nu_iz'])/2; nm=(m1['ne_rel']+m2['ne_rel'])/2; nm=nm/np.interp(0.5,rm,nm)
def ld(p):
    d=pickle.load(open(p,'rb')); return d['X'],d['Te'],d['ne']/np.interp(0.5,d['X'][::-1],d['ne'][::-1]),d['nuiz']
runs=[('total energy, 2nd order, $L=1$','results/te_reffields_ne120_m16_L1_o2.pkl',dict(color='k',ls='-')),
      ('total energy, 2nd order, $L=2$','results/te_reffields_ne120_m16_L2_o2.pkl',dict(color='0.45',ls='--')),
      ('total energy, 1st order','results/te_reffields_ne120_m16_L1.pkl',dict(color='0.3',ls='-.')),
      ('kinetic energy, absorbing wall','results/fv_reffields_nu44_m16_L2_abs.pkl',dict(color='0.55',ls=':',lw=1.6))]
fig,ax=plt.subplots(1,3,figsize=(W,2.4))
dT=np.abs(m1['Te_eV']-m2['Te_eV'])/2; dn=np.abs(m1['ne_rel']-m2['ne_rel'])/2/np.interp(0.5,rm,(m1['ne_rel']+m2['ne_rel'])/2); dnu=np.abs(m1['nu_iz']-m2['nu_iz'])/2
ax[0].errorbar(rm,Tm,yerr=dT,fmt='ko',mfc='w',capsize=2,label='Monte Carlo (two runs)',zorder=5); ax[1].errorbar(rm,nm,yerr=dn,fmt='ko',mfc='w',capsize=2,zorder=5); ok=num>0; ax[2].errorbar(rm[ok],num[ok],yerr=np.minimum(dnu[ok],0.999*num[ok]),fmt='ko',mfc='w',capsize=2,zorder=5)
for lab,p,st in runs:
    X,T,n,nu=ld(p); ax[0].plot(X,T,label=lab,**st); ax[1].plot(X,n,**st); ax[2].semilogy(X,np.maximum(nu,1),**st)
ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV',ylim=(2.0,3.9)); ax[0].legend(loc='lower left',fontsize=5.5)
ax[1].set(xlabel=r'$r/R$',ylabel=r'$n_e/n_e(0.5R)$'); ax[2].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1,1e5))
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig3')
# ---------------- Fig. 5: EEPF at three radii and f0 versus total energy
te=pickle.load(open('results/te_reffields_ne120_m16_L1_o2.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(W,2.5))
uc,h=m1['eedf']; h=(h+m2['eedf'][1])/2; du=0.5; X=te['X']; F0=te['F0']; uk=te['ukin']
for b,lab,sel,mk,col in [(0,'wall, $0.9<r/R<1$',X>0.9,'o','k'),(1,'mid, $0.4<r/R<0.6$',(X>0.4)&(X<0.6),'s','0.35'),(2,'axis, $r/R<0.2$',X<0.2,'^','0.6')]:
    fm=h[b]/np.sum(h[b])/du/np.sqrt(uc); ax[0].semilogy(uc,np.maximum(fm,1e-9),mk,color=col,mfc='none',ms=2.5,ls='none',label='MC, '+lab)
    out=np.zeros_like(uc); cnt=np.zeros_like(uc)
    for k in np.where(sel)[0]:
        okk=uk[:,k]>0; inside=(uc>=uk[okk,k].min())&(uc<=uk[okk,k].max()); out+=np.where(inside,np.interp(uc,uk[okk,k],F0[okk,k]),0); cnt+=inside
    out=np.where(cnt>0,out/np.maximum(cnt,1),np.nan); out/=np.nansum(out*np.sqrt(uc)*du); ax[0].semilogy(uc,np.maximum(out,1e-9),'-',color=col,label='solver, '+lab)
ax[0].set(xlabel=r'$\epsilon$, eV',ylabel=r'EEPF, eV$^{-3/2}$',xlim=(0,25),ylim=(1e-7,1)); ax[0].legend(fontsize=5.5)
ec=te['ec']
for k,st in [(len(X)-1,dict(color='k',ls='-')),(len(X)//2,dict(color='0.35',ls='--')),(3,dict(color='0.5',ls='-.')),(0,dict(color='0.65',ls=':',lw=1.6))]:
    okk=uk[:,k]>0; ax[1].semilogy(ec[okk],np.maximum(F0[okk,k],1e-12),label=rf'$r/R={X[k]:.2f}$, $\varphi={te["phi"][k]:.1f}$ V',**st)
ax[1].set(xlabel=r'$\varepsilon=\epsilon-\varphi(r)$, eV',ylabel=r'$f_0$ (arb. units)',xlim=(-10,25),ylim=(1e-8,1)); ax[1].legend(fontsize=5.5)
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig4')
# ---------------- Fig. 6: self-consistent operating point
S=json.load(open('results/te_sc_o2.json')); d=pickle.load(open(f"results/discharge_te_B{S['Bop']}_n1e+17_L1_m16_o2.pkl",'rb')); X=d['X']
fig,ax=plt.subplots(1,4,figsize=(W,2.2))
ax[0].plot(X,np.abs(d['E'][2])*2,'k-'); ax[0].set(xlabel=r'$r/R$',ylabel=r'$|E_\theta|$, V/m')
ax[1].plot(X,d['Te'],'k-'); ax[1].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV')
ax[2].plot(X,d['phi']-d['phi'].min(),'k-'); ax[2].set(xlabel=r'$r/R$',ylabel=r'$\varphi$, V')
ax[3].plot(X,d['ne']/d['ne'][-1],'k-',label=r'$n_e$'); ax[3].plot(X,d['ni']/d['ni'][-1],'--',color='0.45',label=r'$n_i$'); ax[3].set(xlabel=r'$r/R$',ylabel=r'$n/n(0)$'); ax[3].legend()
for a,l in zip(ax,'abcd'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig6')
print('figures written to',OUT)
