"""Paper 4 (v1.7.0): figures Fig1 (scan), Fig2 (convergence 2x2) and the two tables -> papers/p4_v17/build/Paper4_selfconsistent.tex
usage: python3 p4_build17.py"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT='papers/p4_v17'; BLD=OUT+'/build'; os.makedirs(BLD,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3,'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name,rect=None):
    fig.tight_layout(pad=0.4,rect=rect) if rect else fig.tight_layout(pad=0.4)
    for ext in ('eps','pdf','png'): fig.savefig(f'{BLD}/{name}.{ext}',dpi=200)
    plt.close(fig)
def ld(tag,B=0.74,n0='1e+17'):
    f=f'results/discharge_te_B{B:g}_n{n0}_L1_m16_o2_{tag}.pkl'; return pickle.load(open(f,'rb')) if os.path.exists(f) else None
def hist(d): return np.array(d['hist'],dtype=float)
def skin(d):
    X=d['X']; E=np.abs(d['E'][2]); r=X[::-1]*0.1; e=E[::-1]/E[0]; return 100*(0.1-np.interp(np.exp(-1),e,r))
# ---------------- scan table and Fig. 1
Bs=[0.7,0.74,0.76,0.8]; rows=[]; info={}
for B in Bs:
    d=ld('kin_teff2',B)
    if d is None: continue
    h=hist(d); rows.append(dict(B=B,Ew=2*abs(d['E'][2][0]),delta=skin(d),Te=(d['Te'][0],d['Te'].max(),d['Te'][-1]),well=d['phi'][-1]-d['phi'][0],compr=d['ne'][-1]/d['ne'][0],beta=d['balance'],it=int(d['n_iter']),conv=bool(d['converged']),izmax=float(d['X'][np.argmax(d['nuiz'])])))
Bv=np.array([r['B'] for r in rows]); bv=np.array([r['beta'] for r in rows]); slopes=np.diff(np.log(bv))/np.diff(np.log(Bv)); Bstar=float(np.exp(np.interp(0,np.log(bv),np.log(Bv))))
tab=''
for i,r in enumerate(rows):
    tab+=f"{r['B']:.2f} & {r['Ew']:.0f} & {r['delta']:.1f} & {r['Te'][0]:.2f} / {r['Te'][1]:.2f} / {r['Te'][2]:.2f} & {r['well']:.1f} & {r['compr']:.1f} & {r['beta']:.3f} & {slopes[i]:.1f} & {r['it']}\\\\\n" if i<len(slopes) else f"{r['B']:.2f} & {r['Ew']:.0f} & {r['delta']:.1f} & {r['Te'][0]:.2f} / {r['Te'][1]:.2f} / {r['Te'][2]:.2f} & {r['well']:.1f} & {r['compr']:.1f} & {r['beta']:.3f} & -- & {r['it']}\\\\\n"
info['scan']=dict(rows=rows,slopes=slopes.tolist(),Bstar=Bstar,Im=Bstar*1e-4/(4e-7*np.pi))
old=[(0.60,0.31),(0.70,0.67),(0.76,0.93),(0.80,1.14),(1.00,2.77)]
fig,ax=plt.subplots(1,2,figsize=(W*0.75,2.5))
ax[0].loglog(Bv,bv,'ks-',label='present closure'); ax[0].loglog([b for b,_ in old],[x for _,x in old],'o--',color='0.5',mfc='w',label='first version')
ax[0].axhline(1,color='0.7',lw=0.8); ax[0].axvline(Bstar,color='k',ls='--',lw=0.8); ax[0].axvline(0.77,color='0.5',ls='--',lw=0.8)
bb=np.linspace(0.6,1.0,50); ax[0].loglog(bb,(bb/Bstar)**4,':',color='0.3',lw=0.9,label=r'$(B_{\rm ext}/B^*)^{4}$'); ax[0].legend(loc='upper left',fontsize=6)
ax[0].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$\beta$',xticks=[0.6,0.7,0.8,1.0],xticklabels=['0.6','0.7','0.8','1.0'],yticks=[0.3,0.5,1,2,3],yticklabels=['0.3','0.5','1','2','3']); ax[0].minorticks_off()
Te=np.array([r['Te'] for r in rows]); oldTe={0.60:(2.14,3.59,3.16),0.70:(2.25,3.69,3.39),0.76:(2.32,3.74,3.38),0.80:(2.37,3.78,3.40),1.00:(2.57,3.95,3.49)}
ob=sorted(oldTe); oT=np.array([oldTe[b] for b in ob])
for j,(lab,mk) in enumerate([('wall','o'),('maximum','s'),('axis','^')]):
    ax[1].plot(Bv,Te[:,j],'k'+mk+'-',label=lab+' (present)'); ax[1].plot(ob,oT[:,j],mk+'--',color='0.55',mfc='w',label=lab+' (first version)')
ax[1].set(xlabel=r'$B_{\rm ext}$, G',ylabel=r'$T_e$, eV'); h_,l_=ax[1].get_legend_handles_labels(); fig.legend(h_,l_,loc='lower center',ncol=3,fontsize=6,frameon=False)
for a,l in zip(ax,'ab'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig1',rect=(0,0.14,1,1))
# ---------------- runs table and Fig. 2
RUNS=[('kin_v1warm','midpoint, warm, (0.5, 0.35)'),('kin_v1_r0605','midpoint, warm, (0.6, 0.5)'),('kin_v1_pr02','midpoint, warm, (0.5, 0.2)'),('kin_v1cold','midpoint, cold, (0.5, 0.35)'),('and4_kin_v1','midpoint, warm, Anderson $m=4$'),
      ('kin_teff2','exact, warm, (0.5, 0.35)'),('kin_teff2_cold','exact, cold, (0.5, 0.35)'),('and4_kin_teff2','exact, from the converged state, Anderson $m=4$')]
RUNS3=[('kin_v1','midpoint, cold, (0.5, 0.35), $3\\cdot10^{17}$'),('kin_teff2','exact, cold, (0.5, 0.35), $3\\cdot10^{17}$')]
def row(tag,lab,n0='1e+17'):
    d=ld(tag,0.74,n0)
    if d is None: return f"{lab} & missing \\\\\n", None
    h=hist(d); b4=h[-4:,2].mean(); r10=h[-6:,2]; p10=h[-6:,5]
    return f"{lab} & {int(d['n_iter'])} & {'yes' if d['converged'] else 'no'} & {b4:.3f} & {r10.min():.3f}--{r10.max():.3f} & {p10.min():.2f}--{p10.max():.2f} & {d['Te'][0]:.2f} / {d['Te'].max():.2f} / {d['Te'][-1]:.2f} & {d['phi'][-1]-d['phi'][0]:.2f}\\\\\n", d
tr=''; D={}
for tag,lab in RUNS: s,d=row(tag,lab); tr+=s; D[tag]=d
for tag,lab in RUNS3: s,d=row(tag,lab,'3e+17'); tr+=s; D[tag+'_3e17']=d
fig,ax=plt.subplots(2,2,figsize=(W*0.85,5.0)); ax=ax.ravel()
st={'kin_v1warm':('k','-','(0.5, 0.35)'),'kin_v1_r0605':('0.45','--','(0.6, 0.5)'),'kin_v1_pr02':('0.6','-.','(0.5, 0.2)'),'and4_kin_v1':('0.3',':','Anderson $m=4$')}
for tag,(c,ls,lab) in st.items():
    d=D.get(tag)
    if d is None: continue
    h=hist(d); ax[0].semilogy(np.arange(1,len(h)+1),h[:,5],ls,color=c,label=lab,marker='.' if 'and4' in tag else None,ms=3)
ax[0].axhline(0.05,color='0.7',lw=0.8); ax[0].set(xlabel='iteration',ylabel=r'$\|\Delta\varphi\|_\infty$, V',ylim=(1e-2,3)); ax[0].legend(fontsize=6,title='midpoint quadrature',title_fontsize=6)
d=D.get('and4_kin_v1')
if d is not None:
    h=hist(d); it=np.arange(1,len(h)+1); ax[1].semilogy(it,h[:,1],'k-',label='field residual'); ax[1].semilogy(it,h[:,5],'--',color='0.5',label=r'$\|\Delta\varphi\|_\infty$, V')
ax[1].axhline(2e-3,color='0.7',lw=0.8); ax[1].axhline(0.05,color='0.7',lw=0.8); ax[1].set(xlabel='iteration',ylim=(1e-5,3)); ax[1].legend(fontsize=6,title='Anderson, midpoint quadrature',title_fontsize=6)
for key,c,lab in (('kin_v1_3e17','0.55','midpoint'),('kin_teff2_3e17','k','exact')):
    d=D.get(key)
    if d is None: continue
    h=hist(d); ax[2].plot(np.arange(1,len(h)+1),h[:,2],'-',color=c,label=lab)
ax[2].set(xlabel='iteration',ylabel=r'$\beta$',ylim=(0,0.4)); ax[2].legend(fontsize=6,title=r'$n_0=3\cdot10^{17}$ m$^{-3}$, cold start',title_fontsize=6)
for key,c,ls,lab,n0 in (('kin_teff2','k','-','warm start','1e+17'),('kin_teff2_cold','0.45','--','cold start','1e+17'),('and4_kin_teff2','0.3',':','Anderson from the converged state','1e+17'),('kin_teff2','0.6','-.',r'cold start, $3\cdot10^{17}$ m$^{-3}$','3e+17')):
    d=ld(key,0.74,n0)
    if d is None: continue
    h=hist(d); ax[3].semilogy(np.arange(1,len(h)+1),np.maximum(h[:,5],1e-4),ls,color=c,label=lab,marker='.' if 'and4' in key else None,ms=3)
ax[3].axhline(0.05,color='0.7',lw=0.8); ax[3].set(xlabel='iteration',ylabel=r'$\|\Delta\varphi\|_\infty$, V',ylim=(5e-4,3)); ax[3].legend(fontsize=6,title='exact quadrature',title_fontsize=6)
for a,l in zip(ax,'abcd'): a.set_title(f'({l})',loc='left',fontsize=8)
save(fig,'Fig2')
# ---------------- fill the tex
s=open(f'{OUT}/Paper4_selfconsistent.tex').read().replace('%%TAB_SCAN%%',tab).replace('%%TAB_RUNS%%',tr)
open(f'{BLD}/Paper4_selfconsistent.tex','w').write(s)
info['runs']={k:(dict(n_iter=int(v['n_iter']),conv=bool(v['converged']),beta=float(v['balance']),well=float(v['phi'][-1]-v['phi'][0]),compr=float(v['ne'][-1]/v['ne'][0]),Te=[float(v['Te'][0]),float(v['Te'].max()),float(v['Te'][-1])],Ew=float(2*abs(v['E'][2][0])),dphi=[float(x) for x in hist(v)[:,5]],beta_hist=[float(x) for x in hist(v)[:,2]],dE=[float(x) for x in hist(v)[:,1]]) if v is not None else None) for k,v in D.items()}
json.dump(info,open(f'{BLD}/p4_numbers.json','w'),indent=1,default=float); print(tab); print(tr); print('B*',Bstar,'I',info['scan']['Im'],'slopes',slopes)
