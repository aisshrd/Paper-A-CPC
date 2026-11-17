"""Paper 7 Fig6: kinetic vs local model (p7_local results) for Ar, 90/10, N2.  usage: python3 p7_figs_local17.py <outdir>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, os, sys, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1]; J=json.load(open('results/p7_numbers17.json'))
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54; fig,ax=plt.subplots(1,2,figsize=(W*0.75,2.5))
for key,lab,c in (('Ar','Ar','k'),('Ar0.9_N20.1',r'Ar/N$_2$ 90/10','0.4'),('N2',r'N$_2$','0.65'),('N20.79_O20.21','air','0.8')):
    if key not in J or not J[key].get('local'): continue
    i0=int(np.argmin(np.abs(np.array(J[key]['beta'])-1))); f=J[key]['files'][i0]; L=None
    for c_ in (f'results/p7_local_{os.path.basename(f)}',):
        if os.path.exists(c_): L=pickle.load(open(c_,'rb'))
    if L is None: continue
    d=pickle.load(open(f,'rb')); X=d['X']; nl=np.array(L['nuiz_loc'])-(np.array(L['nuatt_loc']) if L.get('nuatt_loc') is not None else 0)
    ax[0].plot(X,d['Te'],'-',color=c,label=lab); ax[0].plot(X,L['Te_loc'],':',color=c,lw=1.6)
    ax[1].semilogy(X,np.maximum(d['nuiz'],1e-1),'-',color=c); ax[1].semilogy(X,np.maximum(nl,1e-1),':',color=c,lw=1.6)
ax[0].plot([],[],'k-',label='kinetic'); ax[0].plot([],[],'k:',lw=1.6,label='local model'); ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[0].legend(fontsize=5.5)
ax[1].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm net}$, s$^{-1}$',ylim=(1e-1,1e5))
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
fig.tight_layout(pad=0.4)
for ext in ('eps','pdf','png'): fig.savefig(f'{OUT}/Fig6.{ext}',dpi=200)
print('Fig6 ok')
