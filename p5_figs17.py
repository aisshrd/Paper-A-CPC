"""Paper 5 figures (07.11.2026): Fig1 data flow of the package, Fig2 example outputs (verification in the target regime), Fig3 cost.
usage: python3 p5_figs17.py <outdir>"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, matplotlib, scipy.constants as co
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=sys.argv[1] if len(sys.argv)>1 else 'papers/p5_v17'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'lines.linewidth':1.1,
                     'ps.fonttype':42,'pdf.fonttype':42,'axes.grid':True,'grid.color':'0.85','grid.linewidth':0.5,'lines.markersize':3.5,
                     'font.family':'DejaVu Sans','mathtext.fontset':'cm','legend.framealpha':1.0,'legend.edgecolor':'0.7'})
W=17/2.54
def save(fig,name,rect=None):
    fig.tight_layout(pad=0.4,rect=rect) if rect else fig.tight_layout(pad=0.4)
    fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png',dpi=200); plt.close(fig)
# ---------------- Fig. 1: data flow
fig,ax=plt.subplots(figsize=(W,3.0)); ax.axis('off'); ax.set(xlim=(0,17),ylim=(0,7.2))
def box(x,y,w,h,text,fc='white',fs=6.5,lw=0.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=fc,ec='0.2',lw=lw)); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
def arrow(x1,y1,x2,y2,text='',ts=5.5,dx=0,dy=0.15):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,lw=0.8,color='0.2'))
    if text: ax.text((x1+x2)/2+dx,(y1+y2)/2+dy,text,ha='center',va='bottom',fontsize=ts,color='0.2')
box(0.3,5.4,3.6,1.3,'LXCat files\n(Ar, N$_2$, O$_2$, He)\n ekin.load_argon / ekin_mix.load_species',fc='0.95')
box(0.3,3.2,3.6,1.3,'ekin.py / ekin_mix.py\nEnergyGridKinetics, MixtureKinetics\n(0D energy-grid operators, HB in time)')
box(0.3,1.0,3.6,1.3,'ekin_cyl.py\nAngularBasis ($Y_{lm}$, force and\nrotation operators)')
box(5.3,2.6,4.2,2.3,'ekin_cyl_te.py  CylKineticsTE\n(+ ekin_cyl_mag.py CylKineticsMag)\n\ntotal-energy grid, $S_N$ radial scheme,\nbordered sparse LU\n$f_{nlm}(\\varepsilon,r)$, $n_e$, $T_e$, $\\nu_{\\rm iz}$, $j_1$, $T_{\\rm eff}$',fc='0.93')
box(10.8,4.6,5.9,2.2,'discharge_te.py  DischargeTE (+ _mix, _mag)\nPicard / Anderson loop:\nMaxwell ($\\sigma=j_1/E_1$) $\\to$ $E_1,B_1$\nion eigenproblem (ION_MODEL=kin) $\\to$ $n_i,\\beta$\nNewton step $\\Delta\\varphi=T_{\\rm eff}\\ln(n_i/n_e)$',fc='0.93')
box(10.8,2.0,5.9,1.9,'ekin_nlsigma.py  kernels\nlinear-response kernels $K_{cE},K_{cB},K_{cR}$\n$\\to$ radial RF field of the insulated column',fc='white')
box(5.3,0.2,4.2,1.5,'mc_real.py, mc_cyl_real.py, mc_cyl_mag.py\nelectron Monte Carlo references\n(same cross sections and fields)',fc='0.97')
box(10.8,0.2,5.9,1.2,'scripts/p1..p7_*.py, te_tests.py\ntables, figures, verification tests',fc='0.97')
arrow(2.1,5.4,2.1,4.5); arrow(3.9,3.85,5.3,3.75,'collision\noperators',dy=0.05); arrow(3.9,1.65,5.3,3.2,'angular\nmatrices',dx=-0.5,dy=-0.5)
arrow(9.5,4.1,10.8,5.0,'moments',dx=0.1,dy=0.1); arrow(10.8,5.3,9.5,4.5,'$E_1,B_1,\\varphi$',dx=0.2,dy=-0.6)
arrow(9.5,3.3,10.8,3.0,'$A_{11},f_0$',dy=0.05); arrow(10.8,3.6,9.5,3.7,'$E_r$',dy=0.05)
arrow(7.4,2.6,7.4,1.7,'prescribed fields',dx=1.2,dy=-0.1); arrow(13.7,2.0,13.7,1.4); arrow(13.7,4.6,13.7,3.9)
fig.savefig(f'{OUT}/Fig1.pdf',bbox_inches='tight'); fig.savefig(f'{OUT}/Fig1.eps',bbox_inches='tight'); fig.savefig(f'{OUT}/Fig1.png',dpi=200,bbox_inches='tight'); plt.close(fig)
# ---------------- Fig. 2: verification in the target regime
sys.path.insert(0,'.'); from ekin import load_argon, EnergyGridKinetics
xs=load_argon('data/Ar_LXCat.txt'); omega=2*np.pi*13.56e6; ng=3.2e21
fig,ax=plt.subplots(1,3,figsize=(W,2.5))
K=EnergyGridKinetics(xs,umax=70.,nu=350,L=1,Nh=2,ng=ng); f=K.solve(400.,omega); mc=pickle.load(open(f'results/mc_hf_ng{ng:.1e}.pkl','rb'))[400.0]; uc,fe=mc['eedf']
ax[0].semilogy(K.uc,np.maximum(K.eedf(f),1e-12),'k-',label='harmonic balance, $N_t=2$'); ax[0].semilogy(uc,np.maximum(fe,1e-12),'o',color='0.4',mfc='none',ms=2.5,label='Monte Carlo')
ax[0].set(xlabel=r'$\epsilon$, eV',ylabel=r'EEPF, eV$^{-3/2}$',xlim=(0,45),ylim=(1e-7,1),title='uniform RF field, 13.3 Pa, 400 V/m'); ax[0].legend(); ax[0].title.set_fontsize(7)
d3=pickle.load(open('results/p3_cylcoef.pkl','rb')); m4=pickle.load(open('results/mc_cyl_real_N3000_dt4_80us_s41.pkl','rb'))
ax[1].plot(d3['X'],d3['Te'],'k-',label='kinetic solver'); ax[1].plot(m4['r_over_R'],m4['Te_eV'],'ko',mfc='w',label='Monte Carlo (ring averages)'); ax[1].set(xlabel=r'$r/R$',ylabel=r'$T_e$, eV',title='cylinder, prescribed fields, 13.3 Pa'); ax[1].legend(loc='lower left'); ax[1].title.set_fontsize(7)
ok=m4['nu_iz']>0; ax[2].semilogy(d3['X'],np.maximum(d3['nuiz'],1e-2),'k-',label='kinetic solver (cells)'); ax[2].semilogy(m4['r_over_R'],np.maximum(d3['nuiz_ring'],1e-2),'ks',mfc='none',label='kinetic solver (ring averages)'); ax[2].semilogy(m4['r_over_R'][ok],m4['nu_iz'][ok],'ko',mfc='w',label='Monte Carlo')
ax[2].set(xlabel=r'$r/R$',ylabel=r'$\nu_{\rm iz}$, s$^{-1}$',ylim=(1,3e4),title='cylinder, prescribed fields, 13.3 Pa'); ax[2].legend(loc='lower right',fontsize=5.5); ax[2].title.set_fontsize(7)
for a,l in zip(ax,'abc'): a.set_title(f'({l}) '+a.get_title(),loc='left',fontsize=7)
save(fig,'Fig2')
# ---------------- Fig. 3: cost
unk=np.array([23040,34560,46080,46080,51840,69120]); tm=np.array([70,300,1140,840,480,1500]); lab=['default','$N_r=24$','$N_r=32$','$n_\\varepsilon=240$','$L=2$','$n_\\varepsilon=360$']
fig,ax=plt.subplots(figsize=(W*0.45,2.4)); ax.loglog(unk,tm,'ko');
for u,t,l in zip(unk,tm,lab): ax.annotate(l,(u,t),textcoords='offset points',xytext=(4,-2),fontsize=6)
uu=np.linspace(2e4,8e4,10); ax.loglog(uu,70*(uu/23040)**2,':',color='0.5',label='$\\propto N^2$'); ax.loglog(uu,70*(uu/23040)**1.5,'--',color='0.5',label='$\\propto N^{3/2}$')
ax.set(xlabel='unknowns $N$',ylabel='time of one solution, s'); ax.legend(loc='upper left'); save(fig,'Fig3'); print('ok')
