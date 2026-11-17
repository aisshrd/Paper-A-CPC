"""Paper 2, Table III rows (kinetic vs LFA at three pressures) and integral ratios; uses the N_t=2 files when present.
usage: python3 p2_tab.py  -> prints LaTeX rows and a json with the integral numbers (results/p2_tab3.json)"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import pickle, numpy as np, os, json
rows={}; integ={}
def load(p,nh):
    fn=f'results/p2_local_p{p:g}'+('_nh2' if nh==2 else '')+'.pkl'
    return pickle.load(open(fn,'rb')) if os.path.exists(fn) else None
def fmt(v):
    if v<=1e-6: return '$<10^{-6}$'
    if v>=1e4: return f'{int(round(v,-2)):d}'
    if v>=1e3: return f'{int(round(v,-1)):d}'
    if v>=100: return f'{v:.0f}'
    if v>=1: return f'{v:.3g}'
    e=int(np.floor(np.log10(v))); m=v/10**e; return f'${m:.1f}\\cdot10^{{{e}}}$'
for p in (13.3,40,133):
    for nh in (1,2):
        d=load(p,nh)
        if d is None: continue
        X=d['X']; edges=np.concatenate(([1.0],(X[1:]+X[:-1])/2,[0.0])); vol=edges[:-1]**2-edges[1:]**2
        n=d['ne']; nuL=np.maximum(np.asarray(d['local']['nuiz']),0)
        I=np.sum(n*d['nuiz']*vol); IL=np.sum(n*nuL*vol);
        frac_layer=np.sum((n*d['nuiz']*vol)[X>0.78])/I; fracL=np.sum((n*nuL*vol)[X>0.78])/IL
        integ[f'{p:g}_nh{nh}']=dict(I=I,IL=IL,ratio=IL/I,frac_layer_kin=frac_layer,frac_layer_lfa=fracL,Te_axis=d['Te'][-1],TeL_axis=d['local']['Te'][-1],time=d['time'])
        rows[(p,nh)]=d
        print(f'p={p} Nt={nh}: <nu n>_LFA/<nu n>_kin={IL/I:.3f}  layer(r>0.78R) share kin {frac_layer:.2f} lfa {fracL:.2f}  Te axis {d["Te"][-1]:.2f}/{d["local"]["Te"][-1]:.2f}  nuiz wall {d["nuiz"][0]:.3g}/{nuL[0]:.3g} time {d["time"]:.0f}s')
json.dump(integ,open('results/p2_tab3.json','w'),indent=1)
# LaTeX rows: r/R, then for each pressure (N_t best available): Te kin/LFA, nuiz kin/LFA
sel=[(p,2 if (p,2) in rows else 1) for p in (13.3,40,133)]
X=rows[sel[0]]['X']; idx=[0,2,4,6,8,10,12,14,15]
print('% columns: r/R | 13.3 Pa Te, nuiz | 40 Pa Te, nuiz | 133 Pa Te, nuiz   (kinetic / LFA); N_t used:',sel)
for i in idx:
    cells=[f'{X[i]:.2f}']
    for p,nh in sel:
        d=rows[(p,nh)]; nuL=max(float(np.asarray(d['local']['nuiz'])[i]),0.0)
        cells.append(f"{d['Te'][i]:.2f} / {d['local']['Te'][i]:.2f}"); cells.append(f"{fmt(d['nuiz'][i])} / {fmt(nuL)}")
    print(' & '.join(cells)+'\\\\')
