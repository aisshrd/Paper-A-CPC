"""Paper 1 (LJM) figures, revision 1.7.0: Fig1 cylinder vs Monte Carlo (refined-step reference as circles, standard-step runs as crosses),
Fig2 EEPF / f0(eps_tot), Fig3 the prescribed test fields.  usage: python3 p1_figs17.py <outdir>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, os, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1] if len(sys.argv)>1 else 'papers/p1_v17'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name,rect=None):
    fig.tight_layout(pad=0.4,rect=rect) if rect else fig.tight_layout(pad=0.4)
    for ext in ('eps','pdf','png'): fig.savefig(f'{OUT}/{name}.{ext}',dpi=200)
    plt.close(fig)
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb')); m4=pickle.load(open('results/mc_cyl_real_N3000_dt4_80us_s41.pkl','rb'))
rm=m1['r_over_R']; Tm=(m1['Te_eV']+m2['Te_eV'])/2; num=(m1['nu_iz']+m2['nu_iz'])/2; nm=(m1['ne_rel']+m2['ne_rel'])/2; nm=nm/np.interp(0.5,rm,nm)
n4=m4['ne_rel']/np.interp(0.5,rm,m4['ne_rel'])
def ld(p):
    d=pickle.load(open(p,'rb')); return d['X'],d['Te'],d['ne']/np.interp(0.5,d['X'][::-1],d['ne'][::-1]),d['nuiz']
runs=[('total energy, 2nd order, $L=1$','results/te_reffields_ne120_m16_L1_o2.pkl',dict(color='k',ls='-')),
      ('total energy, 2nd order, $L=2$','results/te_reffields_ne120_m16_L2_o2.pkl',dict(color='0.45',ls='--')),
      ('total energy, 1st order','results/te_reffields_ne120_m16_L1.pkl',dict(color='0.3',ls='-.')),
      ('kinetic energy, absorbing wall','results/fv_reffields_nu44_m16_L2_abs.pkl',dict(color='0.55',ls=':',lw=1.6))]
# ---------------- Fig. 1
fig,ax=plt.subplots(1,3,figsize=(W,2.5))
dT=np.abs(m1['Te_eV']-m2['Te_eV'])/2; dn=np.abs(m1['ne_rel']-m2['ne_rel'])/2/np.interp(0.5,rm,(m1['ne_rel']+m2['ne_rel'])/2); dnu=np.abs(m1['nu_iz']-m2['nu_iz'])/2
ax[0].plot(rm,m4['Te_eV'],'ko',mfc='w',label=r'Monte Carlo, $\Delta t=0.125/\nu_{\max}$, 80 $\mu$s',zorder=6); ax[0].errorbar(rm,Tm,yerr=dT,fmt='kx',ms=4,capsize=2,label=r'Monte Carlo, $\Delta t=0.5/\nu_{\max}$ (two runs)',zorder=5)
ax[1].plot(rm,n4,'ko',mfc='w',zorder=6); ax[1].errorbar(rm,nm,yerr=dn,fmt='kx',ms=4,capsize=2,zorder=5)
ok=m4['nu_iz']>0; ax[2].semilogy(rm[ok],m4['nu_iz'][ok],'ko',mfc='w',zorder=6); ok=num>0; ax[2].errorbar(rm[ok],num[ok],yerr=np.minimum(dnu[ok],0.999*num[ok]),fmt='kx',ms=4,capsize=2,zorder=5)
for lab,p,st in runs:
    X,T,n,nu=ld(p); ax[0].plot(X,T,label=lab,**st); ax[1].plot(X,n,**st); ax[2].semilogy(X,np.maximum(nu,1),**st)
ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV',ylim=(2.0,3.9)); ax[1].set(xlabel=r'$r/R$',ylabel=r'$n_e/n_e(0.5R)$'); ax[2].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1,1e5))
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
h,lb=ax[0].get_legend_handles_labels(); fig.legend(h,lb,loc='lower center',ncol=3,fontsize=6,frameon=False)
save(fig,'Fig1',rect=(0,0.16,1,1))
# ---------------- Fig. 2: EEPF at three radii and f0 versus total energy
te=pickle.load(open('results/te_reffields_ne120_m16_L1_o2.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(W,2.5))
uc,h=m4['eedf']; du=uc[1]-uc[0]; X=te['X']; F0=te['F0']; uk=te['ukin']
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
save(fig,'Fig2')
# ---------------- Fig. 3: the prescribed test fields
ref=json.load(open('data/reference_profiles_p13_n1e17_B1.3G.json')); rr=np.array(ref['r_over_R'])
fig,ax=plt.subplots(1,3,figsize=(W,2.3))
ax[0].plot(rr,ref['E_theta_amplitude_V_per_m'],'k-',label=r'$|E_\theta|$'); ax[0].plot(rr,np.array(ref['B_z_amplitude_T'])*1e4*100,'--',color='0.45',label=r'$100\,|B_z|$, G'); ax[0].set(xlabel=r'$r/R$',ylabel='amplitude (V/m, G)'); ax[0].legend(loc='upper left')
ax[1].plot(rr,np.degrees(ref['E_theta_phase_rad']),'k-',label=r'$E_\theta$'); ax[1].plot(rr,np.degrees(ref['B_z_phase_rad']),'--',color='0.45',label=r'$B_z$'); ax[1].set(xlabel=r'$r/R$',ylabel='phase, degrees'); ax[1].legend()
ax[2].plot(rr,ref['phi_V'],'k-'); ax[2].plot(te['X'],te['phi'],'s',color='0.4',mfc='none',ms=3,label='solver cells'); ax[2].set(xlabel=r'$r/R$',ylabel=r'$\varphi$, V'); ax[2].legend()
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig3')
print('ok', 'well', max(ref['phi_V'])-min(ref['phi_V']), 'Ew', max(ref['E_theta_amplitude_V_per_m']), 'Eax', ref['E_theta_amplitude_V_per_m'][0], 'Bw', np.array(ref['B_z_amplitude_T']).max()*1e4)
