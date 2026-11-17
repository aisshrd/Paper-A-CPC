"""Paper 2 (v1.7.0): figures + tables -> papers/p2_v17/build, compile.  usage: python3 p2_build17.py [nofig]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, subprocess, json, shutil
OUT='papers/p2_v17'; BLD=OUT+'/build'; os.makedirs(BLD,exist_ok=True)
if 'nofig' not in sys.argv: subprocess.run(['python3','p2_figs17.py',BLD],check=True,env=dict(os.environ,OMP_NUM_THREADS='1'))
r=subprocess.run(['python3','p2_tab.py'],capture_output=True,text=True,check=True).stdout
rows=[l for l in r.splitlines() if l.endswith('\\\\') and l[0].isdigit()]; tab_local='\n'.join(rows)+'\n'
# self-consistent table
def skin(d):
    X=d['X']; E=np.abs(d['E'][2]); r=X[::-1]*0.1; e=E[::-1]/E[0]; return 100*(0.1-np.interp(np.exp(-1),e,r))
tab_sc=''; info={}
for B in (0.7,0.74,0.76,0.8):
    f=f'results/discharge_te_B{B:g}_n1e+17_L1_m16_o2_kin_teff2.pkl'
    if not os.path.exists(f): continue
    d=pickle.load(open(f,'rb')); X=d['X']; edges=np.concatenate(([1.0],(X[1:]+X[:-1])/2,[0.0])); vol=edges[:-1]**2-edges[1:]**2; src=d['ne']*d['nuiz']*vol
    f06=src[X>0.6].sum()/src.sum(); f09=src[X>0.9].sum()/src.sum(); im=int(np.argmax(d['nuiz']))
    tab_sc+=f"{B:.2f} & {2*abs(d['E'][2][0]):.0f} & {2*abs(d['E'][2][-1]):.1f} & {skin(d):.1f} & {d['Te'][0]:.2f} / {d['Te'].max():.2f} / {d['Te'][-1]:.2f} & {d['phi'][-1]-d['phi'][0]:.1f} & {d['ne'][-1]/d['ne'][0]:.1f} & {int(round(d['nuiz'][im],-1))} & {X[im]:.2f} & {d['nuiz'][-1]:.0f} & {d['nuiz'][0]:.1f} & {f06:.2f} & {d['balance']:.3f}\\\\\n"
    info[B]=dict(Ew=float(2*abs(d['E'][2][0])),Eax=float(2*abs(d['E'][2][-1])),delta=float(skin(d)),Te=[float(d['Te'][0]),float(d['Te'].max()),float(d['Te'][-1])],Xmax=float(X[int(np.argmax(d['Te']))]),well=float(d['phi'][-1]-d['phi'][0]),compr=float(d['ne'][-1]/d['ne'][0]),beta=float(d['balance']),f06=float(f06),f09=float(f09),izmax=float(d['nuiz'][im]),Xiz=float(X[im]),phi_prof=[float(x) for x in (d['phi']-d['phi'][0])],X=[float(x) for x in X],ni_ne_max=float(np.max(np.abs(d['ni']/d['ne']-1))),Teff=[float(x) for x in d['Teff']] if d.get('Teff') is not None else None)
    r=X*0.1; B1=d['B'][2]; j1=-np.gradient(B1,r)/4e-7/np.pi; sig=np.abs(j1/d['E'][2]); info[B]['sigma']=[float(x) for x in sig]
s=open(f'{OUT}/Paper2_physics.tex').read().replace('%%TAB_LOCAL%%',tab_local).replace('%%TAB_SC%%',tab_sc)
open(f'{BLD}/Paper2_physics.tex','w').write(s); shutil.copy('papers/common/model_block.tex',BLD)
json.dump(info,open(f'{BLD}/p2_numbers.json','w'),indent=1); print(tab_sc); print({k:(v['well'],v['compr'],v['f06'],v['f09'],v['delta'],v['Xmax']) for k,v in info.items()})
for _ in range(2): subprocess.run(['pdflatex','-interaction=nonstopmode','Paper2_physics.tex'],cwd=BLD,stdout=subprocess.DEVNULL)
log=open(f'{BLD}/Paper2_physics.log',errors='ignore').read(); print('errors:',log.count('! '),'undefined refs:',log.count('undefined'),'overfull:',log.count('Overfull'))
