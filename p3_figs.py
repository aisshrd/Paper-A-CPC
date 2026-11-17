"""Figures for Paper 3 (transport coefficients). usage: python p3_figs.py <outdir>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,
                     'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,
                     'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name):
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
rows=pickle.load(open('results/p3_coef0d.pkl','rb')); EN=np.array([r['EN'] for r in rows])
g=lambda k: np.array([r[k] for r in rows])
# Fig. 1: coefficients vs E/N and relative differences
fig,ax=plt.subplots(1,3,figsize=(W,2.5))
ax[0].loglog(EN,g('muN'),'k-',label=r'$\mu N$, this work'); ax[0].loglog(EN,g('muN_b'),'ks',mfc='none',label=r'$\mu N$, BOLOS')
ax[0].loglog(EN,g('DN'),'-',color='0.45',label=r'$DN$, this work'); ax[0].loglog(EN,g('DN_b'),'s',color='0.45',mfc='none',label=r'$DN$, BOLOS')
ax[0].loglog(EN,g('muEN'),'k--',label=r'$\mu_\varepsilon N$'); ax[0].loglog(EN,g('DEN'),'--',color='0.45',label=r'$D_\varepsilon N$')
ax[0].set(xlabel=r'$E/N$, Td',ylabel=r'(m V s)$^{-1}$, (m s)$^{-1}$',ylim=(3e23,3e25)); ax[0].legend(ncol=2,loc='upper right',fontsize=5.5)
ok=g('kiz')>1e-25; ax[1].loglog(EN[ok],g('kiz')[ok],'k-',label=r'$k_{\rm iz}$, this work'); okb=g('kiz_b')>1e-25; ax[1].loglog(EN[okb],g('kiz_b')[okb],'ks',mfc='none',label=r'$k_{\rm iz}$, BOLOS')
ok=g('kex')>1e-25; ax[1].loglog(EN[ok],g('kex')[ok],'-',color='0.45',label=r'$k_{\rm ex}$, this work'); okb=g('kex_b')>1e-25; ax[1].loglog(EN[okb],g('kex_b')[okb],'s',color='0.45',mfc='none',label=r'$k_{\rm ex}$, BOLOS')
ax[1].set(xlabel=r'$E/N$, Td',ylabel=r'm$^3$ s$^{-1}$',ylim=(1e-22,1e-14)); ax[1].legend(loc='lower right')
ax[2].semilogx(EN,100*(g('u')/g('u_b')-1),'k-',label=r'$\langle\epsilon\rangle$'); ax[2].semilogx(EN,100*(g('muN')/g('muN_b')-1),'k--',label=r'$\mu N$'); ax[2].semilogx(EN,100*(g('DN')/g('DN_b')-1),'-',color='0.45',label=r'$DN$')
ok=(g('kiz')>1e-21)&(g('kiz_b')>1e-21); ax[2].semilogx(EN[ok],100*(g('kiz')[ok]/g('kiz_b')[ok]-1),'k:',lw=1.6,label=r'$k_{\rm iz}$')
ok=(g('kex')>1e-19)&(g('kex_b')>1e-19); ax[2].semilogx(EN[ok],100*(g('kex')[ok]/g('kex_b')[ok]-1),':',color='0.45',lw=1.6,label=r'$k_{\rm ex}$')
ax[2].set(xlabel=r'$E/N$, Td',ylabel='difference to BOLOS, %',ylim=(-12,12)); ax[2].legend(ncol=2)
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig1')
# Fig. 2: conductivity in the cylinder
d=pickle.load(open('results/p3_cylcoef.pkl','rb')); X=d['X']
fig,ax=plt.subplots(1,2,figsize=(W*0.75,2.4))
ax[0].semilogy(X,np.abs(d['sigma']),'k-',label='kinetic'); ax[0].semilogy(X,d['lfa']['sig_abs'],'k:',lw=1.6,label='local field approx.'); ax[0].semilogy(X,d['lmea_rf']['sig_abs'],'--',color='0.45',label='local mean energy approx.')
ax[0].set(xlabel=r'$r/R$',ylabel=r'$|\sigma|$, S/m'); ax[0].legend()
ax[1].plot(X,-np.angle(d['sigma'])*180/np.pi,'k-'); ax[1].plot(X,180-d['lfa']['sig_arg']*180/np.pi,'k:',lw=1.6); ax[1].plot(X,-d['lmea_rf']['sig_arg']*180/np.pi,'--',color='0.45'); ax[1].set(xlabel=r'$r/R$',ylabel=r'$-\arg\sigma$, deg',ylim=(0,50))
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig3')
# Fig. 3: rate constants versus local temperature along the radius
ng=3.21e21; tab=d['lmea_rf']['table']; Te0=tab[:,0]*2/3
fig,ax=plt.subplots(1,2,figsize=(W*0.75,2.5))
for a,key,lab in [(ax[0],'nuiz',r'$k_{\rm iz}$'),(ax[1],'nuex',r'$k_{\rm ex}$')]:
    col=1 if key=='nuiz' else 2
    a.semilogy(Te0,np.maximum(tab[:,col]/ng,1e-30),'-',color='0.5',label='homogeneous RF curve (LMEA)')
    a.semilogy(d['Te'],np.maximum(d[key]/ng,1e-30),'ko',mfc='w',label='kinetic solution, radial cells')
    a.semilogy(d['lfa']['Te'],np.maximum(d['lfa'][key]/ng,1e-30),'k^',ms=3,label='local field approx.')
    for k in [0,4,7,10,13,15]: a.annotate(f'{X[k]:.2f}',(d['Te'][k],d[key][k]/ng),fontsize=5.5,xytext=(3,2),textcoords='offset points')
    a.set(xlabel=r'$T_e$, eV',ylabel=lab+r', m$^3$ s$^{-1}$',xlim=(2.2,4.3))
ax[0].set(ylim=(1e-22,1e-14)); ax[1].set(ylim=(1e-20,1e-14)); ax[0].legend(loc='upper left')
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig2')
print('done')
