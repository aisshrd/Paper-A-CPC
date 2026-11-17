"""Figures for Paper 6 (magnetized ICP) -> papers/p6/Fig1..Fig4.  usage: python3 p6_figs.py [fig1 fig2 fig3 fig4]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, glob, matplotlib, scipy.constants as co
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ekin import load_argon, GAMMA
OUT=os.environ.get('P6_OUT','papers/p6'); os.makedirs(OUT,exist_ok=True); which=sys.argv[1:] or ['fig1','fig2','fig3','fig4']
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name):
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
S=pickle.load(open('results/lowp_scan_fix.pkl','rb'))
for k in [k for k in S if k[2]=='jr0k']: S[(k[0],k[1],'jr0',k[3])]=S[k]
S={k:v for k,v in S.items() if not (k[2]=='jr0' and np.max(np.abs(v['jr']))>0.03*np.max(np.abs(v['jt'])))}
REF=pickle.load(open('results/discharge_te_B0.76_n1e+17_L1_m16_o2.pkl','rb')); Et=np.abs(REF['E'][2])
def tot(v): return np.sum(v['wdisk']*v['P'])
def iz(v): return np.sum(v['wdisk']*v['ne']*v['nuiz'])/np.sum(v['wdisk']*v['ne'])
xs=load_argon('data/Ar_LXCat.txt'); om=2*np.pi*13.56e6; Bres=om*co.m_e/co.e*1e4
def local_kin(B,p,mode,Te):
    u=np.linspace(0.01,60,6000); sig=np.interp(u,xs['elastic'][:,0],xs['elastic'][:,1]); nu=p/(co.k*300)*sig*GAMMA*np.sqrt(u); Wt=u**1.5*np.exp(-u/Te)
    def P(b):
        wc=co.e*b*1e-4/co.m_e; z=nu+1j*om; D=z**2+wc**2; a=np.trapezoid(Wt*z/D,u); c=np.trapezoid(Wt*wc/D,u)
        return np.real(a) if mode=='Er0' else np.real(a+c**2/a)
    return np.array([P(b) for b in np.atleast_1d(B)])/P(0.0)
def cold(B,nu):
    wc=co.e*B*1e-4/co.m_e; s=0.5*(nu/(nu**2+(om-wc)**2)+nu/(nu**2+(om+wc)**2)); return s/(nu/(nu**2+om**2))
def nu_m(p,u=3.0): return p/(co.k*300)*np.interp(u,xs['elastic'][:,0],xs['elastic'][:,1])*GAMMA*np.sqrt(u)
GR={0:'k',3:'0.15',4.84:'0.3',10:'0.45',20:'0.6',30:'0.7',40:'0.8'}
if 'fig1' in which:   # 13.3 Pa, prescribed fields
    p=13.3; b0=S[(p,0.0,'Er0',1)]; P0=tot(b0); n0=iz(b0)
    fig,ax=plt.subplots(1,3,figsize=(W,2.4))
    for B in (0,10,20,40):
        k=(p,float(B),'Er0',1); ax[0].plot(S[k]['X'],S[k]['Te'],'-',color=GR[B],label=f'{B} G')
        k=(p,float(B),'jr0',1)
        if k in S: ax[0].plot(S[k]['X'],S[k]['Te'],'--',color=GR[B])
    ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[0].legend(title=r'$B_0$',loc='lower left')
    for mode,ls,mk in (('Er0','-','o'),('jr0','--','s')):
        ks=sorted([k for k in S if k[0]==p and k[2]==mode and k[1]<=40],key=lambda k:k[1]); Bs=[0]+[k[1] for k in ks] if mode=='jr0' else [k[1] for k in ks]
        Pv=([1]+[tot(S[k])/P0 for k in ks]) if mode=='jr0' else [tot(S[k])/P0 for k in ks]; Iv=([1]+[iz(S[k])/n0 for k in ks]) if mode=='jr0' else [iz(S[k])/n0 for k in ks]
        ax[1].plot(Bs,Pv,ls,marker=mk,color='k',label=('power' if mode=='Er0' else None)); ax[1].plot(Bs,Iv,ls,marker=mk,mfc='w',color='0.5',label=('ionization' if mode=='Er0' else None))
    ax[1].plot([],[],'k-',label=r'$E_r=0$'); ax[1].plot([],[],'k--',label=r'$j_r=0$')
    ax[1].set(xlabel=r'$B_0$, G',ylabel='relative to $B_0=0$'); ax[1].legend(fontsize=5.5); ax[1].axvline(Bres,color='0.7',lw=0.8,ls=':')
    for B in (0,10,20,40):
        k=(p,float(B),'Er0',1); ax[2].semilogy(S[k]['X'],np.maximum(S[k]['nuiz'],1e-2),'-',color=GR[B],label=f'{B} G')
        k=(p,float(B),'jr0',1)
        if k in S: ax[2].semilogy(S[k]['X'],np.maximum(S[k]['nuiz'],1e-2),'--',color=GR[B])
    ax[2].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1,3e4)); ax[2].legend(title=r'$B_0$',fontsize=5.5,loc='lower left')
    for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
    save(fig,'Fig1')
if 'fig2' in which:   # self-consistent
    BX=os.environ.get('P6_BEXT','0.74'); TG=os.environ.get('P6_TAG','_kin'); files=[(b,f'results/discharge_mag_B0{b}_Bext{BX}{TG}.pkl') for b in (0,10,20,40)]
    D=[(b,pickle.load(open(f,'rb'))) for b,f in files if os.path.exists(f)]
    fig,ax=plt.subplots(1,4,figsize=(W,2.3))
    for b,d in D:
        X=d['X']; c=GR[b]; ax[0].plot(X,d['Te'],'-',color=c,label=f'{b} G'); ax[1].plot(X,d['phi']-d['phi'][0],'-',color=c)
        ax[2].plot(X,d['ne']/d['ne'][-1],'-',color=c); ax[3].semilogy(X,np.maximum(d['nuiz'],1),'-',color=c)
    ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[0].legend(title=r'$B_0$',fontsize=5.5); ax[1].set(xlabel=r'$r/R$',ylabel=r'$\varphi$, V'); ax[2].set(xlabel=r'$r/R$',ylabel=r'$n_e/n_e(0)$')
    ax[3].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(10,5e4))
    for a,l in zip(ax,'abcd'): a.set_title(f'({l})',loc='left',fontsize=8)
    save(fig,'Fig2')
if 'fig3' in which:   # ECR scan
    fig,ax=plt.subplots(1,2,figsize=(W,2.6),sharey=True); Bf=np.linspace(0,20,300); TE={2.0:3.6,5.0:3.7,13.3:3.5}; col={2.0:'k',5.0:'0.4',13.3:'0.65'}
    for p in (2.0,5.0,13.3):
        b0=S[(p,0.0,'Er0',1)]; P0=tot(b0)
        ks=sorted([k for k in S if k[0]==p and k[2]=='Er0' and k[1]<=20],key=lambda k:k[1]); ax[0].plot([k[1] for k in ks],[tot(S[k])/P0 for k in ks],'o-',color=col[p],label=f'{p:g} Pa')
        ax[0].plot(Bf,local_kin(Bf,p,'Er0',TE[p]),'--',color=col[p],lw=0.9); ax[0].plot(Bf,cold(Bf,nu_m(p)),':',color=col[p],lw=0.9)
        ks=sorted([k for k in S if k[0]==p and k[2]=='jr0' and k[1]<=20],key=lambda k:k[1]); ax[1].plot([0]+[k[1] for k in ks],[1]+[tot(S[k])/P0 for k in ks],'s-',color=col[p],label=f'{p:g} Pa')
        ax[1].plot(Bf,local_kin(Bf,p,'jr0',TE[p]),'--',color=col[p],lw=0.9)
    ax[1].axhline(1,color='k',lw=0.8,ls=':')
    for a in ax: a.axvline(Bres,color='0.7',lw=0.8,ls='-.'); a.set(xlabel=r'$B_0$, G')
    ax[0].set(ylabel=r'$P(B_0)/P(0)$',ylim=(0,2)); ax[0].legend(); ax[1].legend()
    ax[0].set_title('(a) $E_r=0$',loc='left',fontsize=8); ax[1].set_title('(b) $j_r=0$',loc='left',fontsize=8)
    save(fig,'Fig3')
if 'fig4' in which:   # Monte Carlo: 13.3 Pa with B0 and 2 Pa resonance
    F=pickle.load(open('results/lowp_scan_fix_ne240.pkl','rb'))
    fig,ax=plt.subplots(1,3,figsize=(W,2.4)); p=2.0
    for B,f,c,lab in ((0,'results/mcfix_13_B0.pkl','k','0'),(20,'results/mcfix_13_B20_Er.pkl','0.4','20'),(40,'results/mcfix_13_B40_Er.pkl','0.7','40')):
        m=pickle.load(open(f,'rb')); k=(13.3,float(B),'Er0' if B==0 else 'jr0k',1); v=F[k]
        ax[0].plot(v['X'],v['Te'],'-',color=c,label=f'{lab} G'); ax[0].plot(m['r_over_R'],m['Te_eV'],'o',color=c,mfc='w')
    ax[0].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[0].legend(title=r'13.3 Pa, $B_0$',fontsize=5.5)
    for B,c in ((0,'k'),(4.84,'0.35'),(20,'0.65')):
        k=(p,float(B),'Er0',1); ax[1].plot(S[k]['X'],S[k]['Te'],'-',color=c,label=f'{B:g} G')
        k=(p,float(B),'jr0',1)
        if k in S: ax[1].plot(S[k]['X'],S[k]['Te'],'--',color=c)
    ax[1].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[1].legend(title=r'2 Pa, $B_0$',fontsize=5.5)
    for B,f,c in ((0,'results/mcfix_lowp_B0.pkl','k'),(4.84,'results/mcfix_lowp_B4.84.pkl','0.45')):
        m=pickle.load(open(f,'rb')); k=(p,float(B),'Er0',1); ax[2].plot(S[k]['X'],S[k]['Te'],'-',color=c,label=f'{B:g} G'); ax[2].plot(m['r_over_R'],m['Te_eV'],'o',color=c,mfc='w')
    ax[2].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV'); ax[2].legend(title=r'2 Pa, $B_0$',fontsize=5.5)
    for a,l in zip(ax,'abc'): a.set_title(f'({l})',loc='left',fontsize=8)
    save(fig,'Fig4')
print('done',which)
