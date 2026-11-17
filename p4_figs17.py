"""Paper 4 figures, revision 1.7.0 (07.11.2026): Fig1 convergence histories at 0.74 G (kin closure), Fig2 scan of beta(B_ext) and T_e,
Fig3 the two-cycle (consecutive iterates) and the strong-coupling test.  usage: python3 p4_figs17.py <outdir>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, glob, matplotlib, json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=sys.argv[1] if len(sys.argv)>1 else 'papers/p4_v17'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,
                     'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3,
                     'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name,rect=None):
    fig.tight_layout(pad=0.4,rect=rect) if rect else fig.tight_layout(pad=0.4)
    fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
def ld(tag,n0='1e+17'): return pickle.load(open(f'results/discharge_te_B0.74_n{n0}_L1_m16_o2_{tag}.pkl','rb'))
runs=[('kin','Picard, $\\rho=0.5$, $\\alpha=0.35$, warm start','k','-'),('kin_r0605','Picard, $\\rho=0.6$, $\\alpha=0.5$, warm start','0.45','--'),
      ('kin_pr02','Picard, $\\rho=0.5$, $\\alpha=0.2$, warm start','0.6','-.'),('kin_cold','Picard, $\\rho=0.5$, $\\alpha=0.35$, cold start ($E=B=\\varphi=0$)','k',':'),
      ('and4_kin','Anderson $m=4$, $\\rho=0.5$, warm start','0.3','-')]
# ---------------- Fig. 1
fig,ax=plt.subplots(1,3,figsize=(W,2.6)); info={}
for tag,lab,col,ls in runs:
    d=ld(tag); h=np.array(d['hist'],dtype=float); it=np.arange(1,len(h)+1); lw=1.6 if tag=='and4_kin' else 1.1
    ax[0].semilogy(it,h[:,1],ls,color=col,lw=lw,label=lab,marker='o' if tag=='and4_kin' else None,ms=2)
    ax[1].semilogy(it,h[:,5],ls,color=col,lw=lw,marker='o' if tag=='and4_kin' else None,ms=2); ax[2].plot(it,h[:,2],ls,color=col,lw=lw,marker='o' if tag=='and4_kin' else None,ms=2)
    info[tag]=dict(n_iter=int(d['n_iter']),conv=bool(d['converged']),beta_last=float(h[-1,2]),beta_mean4=float(h[-4:,2].mean()),beta_min6=float(h[-6:,2].min()),beta_max6=float(h[-6:,2].max()),
                   Te=[float(d['Te'][0]),float(d['Te'].max()),float(d['Te'][-1])],well=float(d['phi'].max()-d['phi'].min()),compr=float(d['ne'][-1]/d['ne'][0]),Ew=float(abs(d['E'][2][0])*2),
                   dphi_last6=[float(x) for x in h[-6:,5]],dE_last6=[float(x) for x in h[-6:,1]])
ax[0].axhline(2e-3,color='0.7',lw=0.8); ax[1].axhline(0.05,color='0.7',lw=0.8); ax[2].axhline(1.0,color='0.7',lw=0.8)
ax[0].set(xlabel='iteration',ylabel=r'$\|E_1^{k+1}-E_1^k\|_\infty/\|E_1^{k+1}\|_\infty$',ylim=(1e-5,1)); ax[1].set(xlabel='iteration',ylabel=r'$\|\Delta\varphi\|_\infty$, V',ylim=(1e-2,5)); ax[2].set(xlabel='iteration',ylabel=r'$\beta$',ylim=(0.6,1.4))
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
h_,l_=ax[0].get_legend_handles_labels(); fig.legend(h_,l_,loc='lower center',ncol=3,fontsize=6,frameon=False); save(fig,'Fig1',rect=(0,0.13,1,1))
# ---------------- Fig. 2: scan
Bs=[0.7,0.74,0.76,0.8]; bal=[]; Te=[]; well=[]; comp=[]; Ew=[]
for B in Bs:
    d=pickle.load(open(f'results/discharge_te_B{B:g}_n1e+17_L1_m16_o2_kin.pkl','rb')); h=np.array(d['hist'],dtype=float)
    bal.append(h[-4:,2].mean()); Te.append([d['Te'][0],d['Te'].max(),d['Te'][-1]]); well.append(d['phi'].max()-d['phi'].min()); comp.append(d['ne'][-1]/d['ne'][0]); Ew.append(abs(d['E'][2][0])*2)
Bs=np.array(Bs); bal=np.array(bal); Te=np.array(Te); Bstar=np.exp(np.interp(0,np.log(bal),np.log(Bs)))
fig,ax=plt.subplots(1,2,figsize=(W*0.7,2.4))
ax[0].loglog(Bs,bal,'ko-'); ax[0].axhline(1,color='0.7',lw=0.8); ax[0].axvline(Bstar,color='0.5',ls='--',lw=0.9)
bb=np.linspace(0.68,0.82,50); ax[0].loglog(bb,(bb/Bstar)**4.5,':',color='0.5',label=r'$(B_{\rm ext}/B^*)^{4.5}$'); ax[0].legend(loc='upper left')
ax[0].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$\beta$',xticks=[0.7,0.74,0.76,0.8],xticklabels=['0.70','0.74','0.76','0.80'],yticks=[0.8,1,1.2,1.4],yticklabels=['0.8','1.0','1.2','1.4']); ax[0].minorticks_off()
ax[1].plot(Bs,Te[:,0],'ko-',label='wall'); ax[1].plot(Bs,Te[:,1],'ks--',mfc='w',label='maximum ($r/R=0.72$)'); ax[1].plot(Bs,Te[:,2],'k^:',label='axis'); ax[1].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$T_e$, eV'); ax[1].legend(loc='center right')
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig2')
# ---------------- Fig. 3: the two-cycle and the strong-coupling test
d=ld('and4_kin'); h=np.array(d['hist'],dtype=float)
fig,ax=plt.subplots(1,3,figsize=(W,2.5))
it=np.arange(1,len(h)+1); ax[0].plot(it,h[:,2],'k.-',label=r'$\beta$'); ax[0].set(xlabel='iteration',ylabel=r'$\beta$ (Anderson $m=4$)',ylim=(0.95,1.04)); a2=ax[0].twinx(); a2.plot(it,h[:,5],'.--',color='0.5'); a2.set_ylabel(r'$\|\Delta\varphi\|_\infty$, V (dashed)',fontsize=7); a2.grid(False); a2.tick_params(labelsize=7)
d3=ld('kin_n3e17','3e+17'); h3=np.array(d3['hist'],dtype=float); it3=np.arange(1,len(h3)+1)
ax[1].semilogy(it3,h3[:,1],'k-',label=r'$n_0=3\cdot10^{17}$ m$^{-3}$, field residual'); ax[1].semilogy(it3,h3[:,5],'--',color='0.45',label=r'$n_0=3\cdot10^{17}$ m$^{-3}$, $\|\Delta\varphi\|_\infty$, V')
dc=ld('kin_cold'); hc=np.array(dc['hist'],dtype=float); ax[1].semilogy(np.arange(1,len(hc)+1),hc[:,1],'k:',label=r'$n_0=10^{17}$ m$^{-3}$, field residual (cold start)')
ax[1].axhline(2e-3,color='0.7',lw=0.8); ax[1].set(xlabel='iteration',ylim=(1e-3,10)); ax[1].legend(fontsize=5.5,loc='upper right')
ax[2].plot(it3,h3[:,2],'k-'); ax[2].set(xlabel='iteration',ylabel=r'$\beta$ at $n_0=3\cdot10^{17}$ m$^{-3}$, $B_{\rm ext}=0.74$ G',ylim=(0,0.5))
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig3')
info['scan']=dict(B=list(Bs),beta=list(map(float,bal)),Bstar=float(Bstar),Te=Te.tolist(),well=list(map(float,well)),compr=list(map(float,comp)),Ew=list(map(float,Ew)),slopes=list(map(float,np.diff(np.log(bal))/np.diff(np.log(Bs)))))
d3i=dict(n_iter=int(d3['n_iter']),beta_mean4=float(h3[-4:,2].mean()),beta_min6=float(h3[-6:,2].min()),beta_max6=float(h3[-6:,2].max()),dE_last6=[float(x) for x in h3[-6:,1]],Te=[float(d3['Te'][0]),float(d3['Te'].max()),float(d3['Te'][-1])],well=float(d3['phi'].max()-d3['phi'].min()),compr=float(d3['ne'][-1]/d3['ne'][0]),Ew=float(abs(d3['E'][2][0])*2))
info['n3e17']=d3i; json.dump(info,open(f'{OUT}/p4_numbers.json','w'),indent=1); print(json.dumps(info,indent=1))
