"""Figures for Paper 4: convergence histories and balance vs inductor field. usage: python p4_figs.py <outdir>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, os, sys, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name):
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
runs=[('results/discharge_te_B0.7_n1e+17_L1_m16_o2_r0605.pkl',r'Picard, $\rho=0.6$, $\alpha=0.5$','k',':'),
      ('results/discharge_te_B0.7_n1e+17_L1_m16_o2.pkl',r'Picard, $\rho=0.5$, $\alpha=0.35$','k','-'),
      ('results/discharge_te_B0.7_n1e+17_L1_m16_o2_and4.pkl',r'Anderson, $m=4$, $\rho=0.5$','0.45','--')]
fig,ax=plt.subplots(1,3,figsize=(W,2.5))
for p,lab,col,ls in runs:
    if not os.path.exists(p): continue
    h=np.array(pickle.load(open(p,'rb'))['hist']); it=h[:,0]+1
    ax[0].semilogy(it,h[:,1],ls,color=col,label=lab); ax[1].semilogy(it,h[:,5],ls,color=col); ax[2].plot(it,h[:,2],ls,color=col)
ax[0].axhline(2e-3,color='0.6',lw=0.8); ax[1].axhline(0.05,color='0.6',lw=0.8); ax[2].axhline(1.14,color='0.6',lw=0.8)
ax[0].set(xlabel='iteration',ylabel=r'$\|E_1^{\prime}-E_1\|_\infty/\|E_1^{\prime}\|_\infty$'); ax[0].legend()
ax[1].set(xlabel='iteration',ylabel=r'$\|\Delta\varphi\|_\infty$, V'); ax[2].set(xlabel='iteration',ylabel=r'balance $\beta$',ylim=(0,1.4))
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig1')
S=json.load(open('results/te_sc_o2.json')); B=np.array([b for b,_ in S['scan']]); bal=np.array([x for _,x in S['scan']])
fig,ax=plt.subplots(1,2,figsize=(W*0.7,2.4))
ax[0].semilogy(B,bal,'ko-'); ax[0].axhline(1,color='0.6',lw=0.8); ax[0].axvline(S['Bstar'],color='0.6',lw=0.8,ls='--'); ax[0].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$\beta$')
rows=np.array(S['scan_rows']); ax[1].plot(B,rows[:,2],'ko-',label=r'$T_e$ max'); ax[1].plot(B,rows[:,3],'ks--',mfc='w',label=r'$T_e$ axis'); ax[1].plot(B,rows[:,1],'k^:',label=r'$T_e$ wall'); ax[1].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$T_e$, eV'); ax[1].legend()
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig2'); print('done')
