"""Figures for Paper 2 (physics): Fig3 pressure scan local vs nonlocal, Fig4 LMEA/LFA/kinetic/MC ionization, Fig6 low-pressure test.
usage: python p2_figs.py <outdir> [fig3|fig4|fig6 ...]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True); which=sys.argv[2:] or ['fig3','fig4','fig6']
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,
                     'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,
                     'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name):
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
if 'fig3' in which:
    fig,ax=plt.subplots(1,3,figsize=(W,2.4))
    for p,col in [(13.3,'k'),(40,'0.4'),(133,'0.65')]:
        d=pickle.load(open(f'results/p2_local_p{p:g}.pkl','rb')); X=d['X']
        ax[0].plot(X,d['Te'],'-',color=col,label=f'{p:g} Pa'); ax[0].plot(X,d['local']['Te'],':',color=col,lw=1.6)
        ax[1].semilogy(X,d['nuiz'],'-',color=col); ax[1].semilogy(X,np.maximum(d['local']['nuiz'],1e-8),':',color=col,lw=1.6)
        ax[2].semilogy(X,d['nuiz']/np.maximum(d['local']['nuiz'],1e-30),'-',color=col,label=f'{p:g} Pa')
    ax[0].plot([],[],'k-',label='nonlocal'); ax[0].plot([],[],'k:',lw=1.6,label='local field approx.')
    ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV',ylim=(1.4,4.3)); ax[0].legend(loc='lower left')
    ax[1].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1e-6,3e4))
    ax[2].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}^{\rm NL}/\nu_{\rm iz}^{\rm LFA}$',ylim=(0.3,1e5)); ax[2].axhline(1,color='0.7',lw=0.8); ax[2].legend(loc='upper right')
    for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
    save(fig,'Fig3')
if 'fig4' in which:
    d=pickle.load(open('results/p3_cylcoef.pkl','rb')); X=d['X']
    m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
    rm=m1['r_over_R']; num=(m1['nu_iz']+m2['nu_iz'])/2; dnu=np.abs(m1['nu_iz']-m2['nu_iz'])/2
    fig,ax=plt.subplots(1,1,figsize=(W*0.6,2.6))
    ax.semilogy(X,np.maximum(d['nuiz'],1e-3),'k-',label='kinetic solution'); ok=num>0
    ax.errorbar(rm[ok],num[ok],yerr=np.minimum(dnu[ok],0.999*num[ok]),fmt='ko',mfc='w',capsize=2,label='Monte Carlo',zorder=5)
    ax.semilogy(X,np.maximum(d['lfa']['nuiz'],1e-3),'k:',lw=1.6,label='local field approx.')
    ax.semilogy(X,np.maximum(d['lmea_rf']['nuiz'],1e-3),'--',color='0.4',label='local mean energy approx. (RF table)')
    ax.set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1e-3,3e4)); ax.legend(loc='lower right')
    save(fig,'Fig4')
if 'fig6' in which:
    runs=[(1,1,'k','-'),(2,1,'0.35','--'),(1,2,'k','-.'),(1,3,'0.55',':')]
    ms=[pickle.load(open(f'results/mc_lowp_p2_s{s}.pkl','rb')) for s in (1,2) if os.path.exists(f'results/mc_lowp_p2_s{s}.pkl')]
    rm=ms[0]['r_over_R']; Tm=np.mean([m['Te_eV'] for m in ms],0); nm=np.mean([m['ne_rel'] for m in ms],0); num=np.mean([m['nu_iz'] for m in ms],0)
    uc,h=ms[0]['eedf']; h=np.mean([m['eedf'][1] for m in ms],0); du=uc[1]-uc[0]
    fig,ax=plt.subplots(1,3,figsize=(W,2.5))
    ax[0].plot(rm,Tm,'ko',mfc='w',label='Monte Carlo',zorder=5); ax[1].semilogy(rm,num,'ko',mfc='w',zorder=5)
    b=2; fm=h[b]/np.sum(h[b])/du/np.sqrt(uc); ax[2].semilogy(uc,np.maximum(fm,1e-9),'ko',mfc='w',ms=2.5,ls='none',label='Monte Carlo, $r/R<0.2$')
    for Nh,L,col,ls in runs:
        p=f'results/p2_lowp_p2_Nh{Nh}_L{L}.pkl' if L<3 else 'results/p2_lowp_p2_Nh1_L3_ne80.pkl'
        if not os.path.exists(p): continue
        d=pickle.load(open(p,'rb')); X=d['X']; lab=f'$N_t={Nh}$, $L={L}$'+(' ($n_\\varepsilon=80$)' if L==3 else '')
        ax[0].plot(X,d['Te'],ls,color=col,label=lab); ax[1].semilogy(X,d['nuiz'],ls,color=col)
        uk=d['ukin']; F0=d['F0']; sel=np.where(X<0.2)[0]; out=np.zeros_like(uc); cnt=np.zeros_like(uc)
        for k in sel:
            ok=uk[:,k]>0; inside=(uc>=uk[ok,k].min())&(uc<=uk[ok,k].max()); out+=np.where(inside,np.interp(uc,uk[ok,k],F0[ok,k]),0); cnt+=inside
        out=np.where(cnt>0,out/np.maximum(cnt,1),np.nan); out/=np.nansum(out*np.sqrt(uc)*du); ax[2].semilogy(uc,np.maximum(out,1e-9),ls,color=col,label=lab)
    ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[0].legend(loc='lower center')
    ax[1].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$')
    ax[2].set(xlabel=r'$\epsilon$, eV',ylabel=r'EEPF, eV$^{-3/2}$',xlim=(0,35),ylim=(1e-7,1)); ax[2].legend(loc='lower left',fontsize=5.5)
    for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
    save(fig,'Fig6')
print('done',which)
