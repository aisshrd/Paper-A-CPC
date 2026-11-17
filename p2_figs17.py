"""Figures for Paper 2, revision 1.7.0 (06.11.2026). Writes Fig1..Fig6 (eps/pdf/png) into the directory given as argument.
Fig1 uniform RF (HB / EF / MC); Fig2 the three field settings; Fig3 pressure scan NL vs LFA (N_t=2 at 40 and 133 Pa when available);
Fig4 ionization LFA/LMEA/kinetic/MC + local EEPFs vs kinetic and vs total energy; Fig5 2 Pa truncation test; Fig6 self-consistent operating point.
usage: python3 p2_figs17.py <outdir>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, os, sys, glob, matplotlib, scipy.constants as co
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1] if len(sys.argv)>1 else 'papers/pkg/ARTICLES/p2'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,
                     'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,
                     'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name,rect=None):
    fig.tight_layout(pad=0.4,rect=rect) if rect else fig.tight_layout(pad=0.4)
    fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
sys.path.insert(0,'.'); from ekin import load_argon, EnergyGridKinetics
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6
# ---------------- Fig. 1: uniform RF field, 0D
fig,ax=plt.subplots(1,3,figsize=(W,2.8))
for p,mk,col in [(13.3,'o','k'),(40.,'s','0.35'),(133.,'^','0.6')]:
    r=np.array(pickle.load(open(f'results/hf_table_p{p}.pkl','rb')),dtype=float); E=r[:,0]
    ax[0].plot(E,r[:,1],'-',color=col,label=f'{p:g} Pa, harmonic balance'); ax[0].plot(E,r[:,3],'--',color=col,label=f'{p:g} Pa, effective field')
    ok=r[:,2]>0; ax[1].semilogy(E[ok],r[ok,2],'-',color=col); ok=r[:,4]>0; ax[1].semilogy(E[ok],r[ok,4],'--',color=col)
    mcf={13.3:'results/mc_hf_ng3.2e+21.pkl',40.:'results/mc_hf_ng9.7e+21.pkl'}.get(p)
    if mcf:
        mcd=pickle.load(open(mcf,'rb')); Em=np.array(sorted(mcd)); um=np.array([mcd[e]['mean_energy'] for e in Em]); nm_=np.array([mcd[e]['nu_iz'] for e in Em])
        ax[0].plot(Em,um,mk,color=col,mfc='none',ls='none',label=f'{p:g} Pa, Monte Carlo'); okm=nm_>0; ax[1].semilogy(Em[okm],nm_[okm],mk,color=col,mfc='none',ls='none')
ax[0].set(xlabel=r'$E_0$, V/m',ylabel=r'$\langle\epsilon\rangle$, eV'); ax[1].set(xlabel=r'$E_0$, V/m',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1,1e7))
ng=3.2e21; K=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=2,ng=ng); f=K.solve(400.,omega)
mc=pickle.load(open(f'results/mc_hf_ng{ng:.1e}.pkl','rb'))[400.0]; uc,fe=mc['eedf']
ax[2].semilogy(K.uc,np.maximum(K.eedf(f),1e-12),'k-',label='harmonic balance'); ax[2].semilogy(uc,np.maximum(fe,1e-12),'o',color='0.4',mfc='none',ms=2.5,label='Monte Carlo')
ax[2].set(xlabel=r'$\epsilon$, eV',ylabel=r'EEPF, eV$^{-3/2}$',xlim=(0,45),ylim=(1e-7,1)); ax[2].legend()
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
h,lb=ax[0].get_legend_handles_labels(); fig.legend(h,lb,loc='lower center',ncol=4,fontsize=6,frameon=False)
save(fig,'Fig1',rect=(0,0.17,1,1))
# ---------------- Fig. 2: the three field settings
from scipy.special import iv
d3=pickle.load(open('results/p3_cylcoef.pkl','rb')); Xa=d3['X']; Ea=np.abs(d3['E1'])*2; pha=d3['phi']
sc=pickle.load(open(os.environ.get('P2_SC','results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_teff2.pkl'),'rb')); Xs=sc['X']; Es=np.abs(sc['E'][2])*2; phs=sc['phi']-sc['phi'].min()
kk=(1+1j)/0.033; Xb=np.linspace(0,1,200); Eb=250*np.abs(iv(1,kk*Xb*0.1)/iv(1,kk*0.1))
fig,ax=plt.subplots(1,2,figsize=(W*0.72,2.4))
ax[0].plot(Xa,Ea,'k-',label='Sec. IV A, IV C (prescribed, $\\delta=3.3$ cm)'); ax[0].plot(Xb,Eb,':',color='0.4',lw=1.6,label='Sec. IV B (prescribed $I_1$, $\\delta=3.3$ cm, 13.3 Pa)')
ax[0].plot(Xs,Es,'--',color='0.45',label='Sec. V (self-consistent, 0.74 G)'); ax[0].set(xlabel=r'$r/R$',ylabel=r'$|E_\theta|$, V/m'); ax[0].legend(fontsize=5.5,loc='upper left')
ax[1].plot(Xa,pha,'k-',label='Sec. IV A, IV C'); ax[1].plot(Xs,phs,'--',color='0.45',label='Sec. V'); ax[1].plot([0,1],[0,0],':',color='0.4',lw=1.6,label='Sec. IV B (no potential)')
ax[1].set(xlabel=r'$r/R$',ylabel=r'$\varphi$, V'); ax[1].legend(fontsize=5.5,loc='lower left')
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig2')
# ---------------- Fig. 3: pressure scan NL vs LFA
fig,ax=plt.subplots(1,3,figsize=(W,2.4))
for p,col in [(13.3,'k'),(40,'0.4'),(133,'0.65')]:
    fn=f'results/p2_local_p{p:g}_nh2.pkl' if os.path.exists(f'results/p2_local_p{p:g}_nh2.pkl') else f'results/p2_local_p{p:g}.pkl'
    d=pickle.load(open(fn,'rb')); X=d['X']; print('Fig3',p,fn)
    ax[0].plot(X,d['Te'],'-',color=col,label=f'{p:g} Pa'); ax[0].plot(X,d['local']['Te'],':',color=col,lw=1.6)
    ax[1].semilogy(X,d['nuiz'],'-',color=col); ax[1].semilogy(X,np.maximum(d['local']['nuiz'],1e-8),':',color=col,lw=1.6)
    ax[2].semilogy(X,d['nuiz']/np.maximum(d['local']['nuiz'],1e-30),'-',color=col,label=f'{p:g} Pa')
ax[0].plot([],[],'k-',label='kinetic'); ax[0].plot([],[],'k:',lw=1.6,label='local field approx.')
ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV',ylim=(1.4,4.3)); ax[0].legend(loc='lower left')
ax[1].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1e-6,3e4))
ax[2].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}^{\rm kin}/\nu_{\rm iz}^{\rm LFA}$',ylim=(0.3,1e5)); ax[2].axhline(1,color='0.7',lw=0.8); ax[2].legend(loc='upper right')
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig3')
# ---------------- Fig. 4: ionization in the prescribed fields + local EEPFs
m4=pickle.load(open('results/mc_cyl_real_N3000_dt4_80us_s41.pkl','rb')); rm=m4['r_over_R']; num=m4['nu_iz']
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb')); num0=(m1['nu_iz']+m2['nu_iz'])/2
fig,ax=plt.subplots(1,3,figsize=(W,3.4))
X=Xa; ax[0].semilogy(X,np.maximum(d3['nuiz'],1e-3),'k-',label='kinetic solution'); ok=num>0
ax[0].plot(rm[ok],num[ok],'ko',mfc='w',label='Monte Carlo ($\\Delta t/4$, 80 $\\mu$s)',zorder=5); ok0=num0>0; ax[0].plot(rm[ok0],num0[ok0],'kx',ms=4,label='Monte Carlo (two runs, 40 $\\mu$s)',zorder=5)
ax[0].semilogy(X,np.maximum(d3['lfa']['nuiz'],1e-3),'k:',lw=1.6,label='local field approx.')
ax[0].semilogy(X,np.maximum(d3['lmea_rf']['nuiz'],1e-3),'--',color='0.4',label='local mean energy approx. (RF table)')
ax[0].semilogy(X,np.maximum(d3['lmea_hot']['nuiz'],1e-3),'-.',color='0.55',label='LMEA at the temperature of the electrons above 1 eV')
ax[0].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1e-3,3e4))
cmp=pickle.load(open('results/p3_eepf_cmp.pkl','rb'))
for j,col,ls,lab in [(0,'k','-','wall, $r/R=1.0$'),(8,'0.4','-','$r/R=0.72$'),(15,'0.65','-','axis')]:
    c=cmp[j]; ax[1].semilogy(c['u_kin'],np.maximum(c['eepf_kin'],1e-12),ls,color=col,label=lab+' (kinetic)')
    ax[1].semilogy(c['u_hom'],np.maximum(c['eepf_hom'],1e-12),':',color=col,lw=1.5,label=f'uniform RF, $E_0={c["E0"]:.0f}$ V/m (same $\\langle\\epsilon\\rangle$)')
ax[1].axvline(11.55,color='0.8',lw=0.8); ax[1].axvline(15.76,color='0.8',lw=0.8)
ax[1].set(xlabel=r'$\epsilon$, eV',ylabel=r'EEPF, eV$^{-3/2}$',xlim=(0,25),ylim=(1e-8,1))
s=pickle.load(open('results/te_reffields_ne120_m16_L1_o2.pkl','rb')); F0=np.real(s['F0']); u=s['ukin']; phi=s['phi']; Xr=s['X']
for j,col in [(0,'k'),(4,'0.3'),(7,'0.5'),(15,'0.7')]:
    ok=u[:,j]>0; ax[2].semilogy(u[ok,j]-phi[j],F0[ok,j],'-',color=col,label=f'$r/R={Xr[j]:.2f}$, $\\varphi={phi[j]:.1f}$ V')
ax[2].axvline(0,color='0.8',lw=0.8); ax[2].set(xlabel=r'total energy $\varepsilon=\epsilon-\varphi(r)$, eV',ylabel=r'$f_0(\varepsilon,r)$, eV$^{-3/2}$',xlim=(-10,25),ylim=(1e-12,1))
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
for k,x0 in zip(range(3),(0.0,0.345,0.69)):
    h_,l_=ax[k].get_legend_handles_labels(); fig.legend(h_,l_,loc='upper left',bbox_to_anchor=(x0,0.27),ncol=1,fontsize=5.6,frameon=False)
save(fig,'Fig4',rect=(0,0.27,1,1))
# ---------------- Fig. 5: 2 Pa truncation test
runs=[(1,1,'k','-'),(2,1,'0.35','--'),(1,2,'k','-.'),(1,3,'0.55',':')]
ms=[pickle.load(open(f'results/mc_lowp_p2_s{s_}.pkl','rb')) for s_ in (1,2) if os.path.exists(f'results/mc_lowp_p2_s{s_}.pkl')]
rm=ms[0]['r_over_R']; Tm=np.mean([m['Te_eV'] for m in ms],0); num=np.mean([m['nu_iz'] for m in ms],0)
uc,h=ms[0]['eedf']; h=np.mean([m['eedf'][1] for m in ms],0); du=uc[1]-uc[0]
fig,ax=plt.subplots(1,3,figsize=(W,2.5))
ax[0].plot(rm,Tm,'ko',mfc='w',label='Monte Carlo',zorder=5); ax[1].semilogy(rm,num,'ko',mfc='w',zorder=5)
b=2; fm=h[b]/np.sum(h[b])/du/np.sqrt(uc); ax[2].semilogy(uc,np.maximum(fm,1e-9),'ko',mfc='w',ms=2.5,ls='none',label='Monte Carlo, $r/R<0.2$')
for Nh,L,col,ls in runs:
    p=f'results/p2_lowp_p2_Nh{Nh}_L{L}.pkl' if L<3 else 'results/p2_lowp_p2_Nh1_L3_ne80.pkl'
    if not os.path.exists(p): continue
    d=pickle.load(open(p,'rb')); X=d['X']; lab=f'$N_t={Nh}$, $L={L}$'+(' ($n_\\varepsilon=80$)' if L==3 else '')
    ax[0].plot(X,d['Te'],ls,color=col,label=lab); ax[1].semilogy(X,d['nuiz'],ls,color=col)
    uk=d['ukin']; F0_=d['F0']; sel=np.where(X<0.2)[0]; out=np.zeros_like(uc); cnt=np.zeros_like(uc)
    for k in sel:
        ok=uk[:,k]>0; inside=(uc>=uk[ok,k].min())&(uc<=uk[ok,k].max()); out+=np.where(inside,np.interp(uc,uk[ok,k],F0_[ok,k]),0); cnt+=inside
    out=np.where(cnt>0,out/np.maximum(cnt,1),np.nan); out/=np.nansum(out*np.sqrt(uc)*du); ax[2].semilogy(uc,np.maximum(out,1e-9),ls,color=col,label=lab)
ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[0].legend(loc='lower center'); ax[1].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$')
ax[2].set(xlabel=r'$\epsilon$, eV',ylabel=r'EEPF, eV$^{-3/2}$',xlim=(0,35),ylim=(1e-7,1)); ax[2].legend(loc='lower left',fontsize=5.5)
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig5')
# ---------------- Fig. 6: self-consistent operating point (new closure, 0.74 G)
d=sc; X=d['X']; r=X*0.1; B1=d['B'][2]; j1=-np.gradient(B1,r)/co.mu_0; sig=np.abs(j1/d['E'][2])
fig,ax=plt.subplots(1,4,figsize=(W,2.3))
ax[0].plot(X,np.abs(d['E'][2])*2,'k-',label=r'$|E_\theta|$, V/m'); ax[0].plot(X[:-1],20*sig[:-1],'--',color='0.5',label=r'$20|\sigma|$, S/m'); ax[0].set(xlabel=r'$r/R$'); ax[0].legend(fontsize=6,loc='upper left')
ax[1].plot(X,d['Te'],'k-'); ax[1].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV')
ax[2].plot(X,d['phi']-d['phi'].min(),'k-'); ax[2].set(xlabel=r'$r/R$',ylabel=r'$\varphi$, V')
ax[3].semilogy(X,d['ne']/d['ne'][-1],'k-',label=r'$n_e/n_e(0)$'); ax[3].semilogy(X,d['ni']/d['ni'][-1],'--',color='0.45',label=r'$n_i/n_i(0)$'); ax[3].semilogy(X,d['nuiz']/d['nuiz'].max(),':',color='0.3',lw=1.6,label=r'$\nu_{\rm iz}/\max\nu_{\rm iz}$')
ax[3].set(xlabel=r'$r/R$',ylim=(1e-4,2)); ax[3].legend(fontsize=5.5,loc='lower left')
for a,l in zip(ax,'abcd'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig6'); print('figures written to',OUT)
