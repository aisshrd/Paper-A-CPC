"""Corrected (sigma_{-1} = conj sigma_{+1}, 4.11.2026) self-consistent results for Papers 2 and 4:
numbers -> results/te_sc_conjfix.json, figures -> papers/p4/Fig1,Fig2 and papers/p2/Fig5.  usage: python3 p24_conjfix.py"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, glob, os, scipy.constants as co, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54; SUF='_conjfix'
def save(fig,out,name):
    os.makedirs(out,exist_ok=True); fig.tight_layout(pad=0.4); fig.savefig(f'{out}/{name}.eps'); fig.savefig(f'{out}/{name}.pdf'); fig.savefig(f'{out}/{name}.png',dpi=200); plt.close(fig)
# ---- scan table
pts=[]
for p in sorted(glob.glob(f'results/discharge_te_B*_n1e+17_L1_m16_o2{SUF}.pkl')):
    d=pickle.load(open(p,'rb')); pts.append((d['B_ext'],d['balance'],d))
pts.sort(); S={'scan':[(B,bal) for B,bal,_ in pts]}
def delta(d):
    X=d['X']; E=np.abs(d['E'][2]); y=E/E[0]; return float((1-np.interp(np.exp(-1),y[::-1],X[::-1]))*10)
S['scan_rows']=[(float(abs(d['E'][2][0])*2),float(d['Te'][0]),float(d['Te'].max()),float(d['Te'][-1]),float(d['phi'].max()-d['phi'].min()),float(d['ne'][-1]/d['ne'][0]),int(d['n_iter']),delta(d),int(d['n_neg']),float(d['f0_min_rel'])) for _,_,d in pts]
Bs=np.array([B for B,_,_ in pts]); bal=np.array([b for _,b,_ in pts]); S['Bstar']=float(np.exp(np.interp(0.0,np.log(bal),np.log(Bs))))
S['slopes']=list(np.round(np.diff(np.log(bal))/np.diff(np.log(Bs)),2))
B,b,d=min(pts,key=lambda t:abs(t[1]-1)); X=d['X']
S.update(Bop=float(B),bal_op=float(b),well=float(d['phi'].max()-d['phi'].min()),compr=float(d['ne'][-1]/d['ne'][0]),Te_wall=float(d['Te'][0]),Te_max=float(d['Te'].max()),Te_axis=float(d['Te'][-1]),
         X_Temax=float(X[np.argmax(d['Te'])]),E_wall=float(abs(d['E'][2][0])*2),IL=float(B*1e-4/co.mu_0),IL_star=float(S['Bstar']*1e-4/co.mu_0),nuiz_max=float(d['nuiz'].max()),X_numax=float(X[np.argmax(d['nuiz'])]),nuiz_axis=float(d['nuiz'][-1]),
         nuiz_wall=[float(x) for x in d['nuiz'][:3]],ni_ne_dev=float(np.max(np.abs(d['ni']/d['ni'][-1]-d['ne']/d['ne'][-1]))),sigma_abs=[float(x) for x in np.abs(d['j0t'] if 'j0t' in d else 0*X)])
# conductivity in the heated layer: sigma = j1*n0/E1 from stored fields -> recompute via kernel-free estimate: |j_theta| not stored in discharge_te pkl; skip
json.dump(S,open(f'results/te_sc{SUF}.json','w'),indent=1); print(json.dumps({k:v for k,v in S.items() if k!='sigma_abs'},indent=1))
# ---- Paper 4 Fig1: convergence histories at 0.7 G
runs=[(f'results/discharge_te_B0.7_n1e+17_L1_m16_o2{SUF}_r0605.pkl',r'Picard, $\rho=0.6$, $\alpha=0.5$','k',':'),
      (f'results/discharge_te_B0.7_n1e+17_L1_m16_o2{SUF}.pkl',r'Picard, $\rho=0.5$, $\alpha=0.35$','k','-'),
      (f'results/discharge_te_B0.7_n1e+17_L1_m16_o2_and4{SUF}.pkl',r'Anderson, $m=4$, $\rho=0.5$','0.45','--')]
fig,ax=plt.subplots(1,3,figsize=(W,2.5)); bal07=[b for B,b in S['scan'] if abs(B-0.7)<1e-6][0]
for p,lab,col,ls in runs:
    if not os.path.exists(p): print('missing',p); continue
    dd=pickle.load(open(p,'rb')); h=np.array(dd['hist']); it=h[:,0]+1
    ax[0].semilogy(it,h[:,1],ls,color=col,label=lab); ax[1].semilogy(it,h[:,5],ls,color=col); ax[2].plot(it,h[:,2],ls,color=col)
    print(os.path.basename(p),'n_iter',dd['n_iter'],'conv',dd['converged'],'balance %.3f'%dd['balance'],'Te axis %.2f'%dd['Te'][-1])
    print('   dE   ',np.round(h[:,1],3)); print('   dphi ',np.round(h[:,5],2)); print('   bal  ',np.round(h[:,2],3)); print('   Teax ',np.round(h[:,4],2)); print('   Tew  ',np.round(h[:,3],2))
ax[0].axhline(2e-3,color='0.6',lw=0.8); ax[1].axhline(0.05,color='0.6',lw=0.8); ax[2].axhline(bal07,color='0.6',lw=0.8)
ax[0].set(xlabel='iteration',ylabel=r'$\|E_1^{\prime}-E_1\|_\infty/\|E_1^{\prime}\|_\infty$'); ax[0].legend()
ax[1].set(xlabel='iteration',ylabel=r'$\|\Delta\varphi\|_\infty$, V'); ax[2].set(xlabel='iteration',ylabel=r'balance $\beta$',ylim=(0,1.4))
for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'papers/p4','Fig1')
# ---- Paper 4 Fig2: balance and Te vs B_ext
B=np.array([b for b,_ in S['scan']]); bal=np.array([x for _,x in S['scan']]); rows=np.array(S['scan_rows'])
fig,ax=plt.subplots(1,2,figsize=(W*0.7,2.4))
ax[0].semilogy(B,bal,'ko-'); ax[0].axhline(1,color='0.6',lw=0.8); ax[0].axvline(S['Bstar'],color='0.6',lw=0.8,ls='--'); ax[0].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$\beta$')
ax[1].plot(B,rows[:,2],'ko-',label=r'$T_e$ max'); ax[1].plot(B,rows[:,3],'ks--',mfc='w',label=r'$T_e$ axis'); ax[1].plot(B,rows[:,1],'k^:',label=r'$T_e$ wall'); ax[1].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$T_e$, eV'); ax[1].legend()
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'papers/p4','Fig2')
# ---- Paper 2 Fig5: operating point profiles
fig,ax=plt.subplots(1,4,figsize=(W,2.2))
ax[0].plot(X,np.abs(d['E'][2])*2,'k-'); ax[0].set(xlabel=r'$r/R$',ylabel=r'$|E_\theta|$, V/m')
ax[1].plot(X,d['Te'],'k-'); ax[1].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV')
ax[2].plot(X,d['phi']-d['phi'].min(),'k-'); ax[2].set(xlabel=r'$r/R$',ylabel=r'$\varphi$, V')
ax[3].plot(X,d['ne']/d['ne'][-1],'k-',label=r'$n_e$'); ax[3].plot(X,d['ni']/d['ni'][-1],'--',color='0.45',label=r'$n_i$'); ax[3].set(xlabel=r'$r/R$',ylabel=r'$n/n(0)$'); ax[3].legend()
for a,l in zip(ax,'abcd'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'papers/p2','Fig5'); print('figures written')
