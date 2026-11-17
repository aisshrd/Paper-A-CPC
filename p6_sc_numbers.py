"""Paper 6 (v1.7.0): self-consistent magnetized discharge with the corrected closure: table rows at B_ext = 0.74 G, the second
current at each B0 and the operating point B*(B0) by log interpolation.  usage: python3 p6_sc_numbers.py -> results/p6_sc_kin.json"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import pickle, numpy as np, glob, json, os, scipy.constants as co
out = {}
for B0 in (0, 10, 20, 40):
    fs = sorted(glob.glob(f'results/discharge_mag_B0{B0}_Bext*_kin.pkl'))
    pts = []
    for f in fs:
        d = pickle.load(open(f, 'rb')); h = np.array(d['hist'], dtype=float); X = d['X']; E = 2 * np.abs(d['E'][2]); y = E / E[0]
        delta = float((1 - np.interp(np.exp(-1), y[::-1], X[::-1])) * 10)
        rec = dict(file=f, Bext=float(d['B_ext']), beta=float(np.mean(h[-6:, 2])), beta_range=[float(h[-6:, 2].min()), float(h[-6:, 2].max())], n_iter=int(d['n_iter']), conv=bool(d['converged']),
                   Te=[float(d['Te'][0]), float(d['Te'].max()), float(d['Te'][-1])], XTmax=float(X[np.argmax(d['Te'])]), well=float(d['phi'].max() - d['phi'].min()), compr=float(d['ne'][-1] / d['ne'][0]),
                   Ew=float(E[0]), delta=delta, Er_ratio=(float(np.max(np.abs(d['Er'][2])) / abs(d['E'][2][0])) if d.get('Er') is not None else None), Er_res=(float(d['Er_res'][-1]) if d.get('Er_res') else None),
                   numax=float(d['nuiz'].max()), Xnumax=float(X[np.argmax(d['nuiz'])]), nuax=float(d['nuiz'][-1]), dphi_last=[round(float(x), 3) for x in h[-6:, 5]])
        pts.append(rec)
    if not pts: continue
    Bs = np.array([p['Bext'] for p in pts]); bs = np.array([p['beta'] for p in pts]); o = np.argsort(bs)
    slope = float(np.diff(np.log(bs[o]))[0] / np.diff(np.log(Bs[o]))[0]) if len(pts) > 1 else None
    Bstar = float(np.exp(np.log(Bs[o][0]) - np.log(bs[o][0]) / slope)) if len(pts) > 1 else None   # log-linear interpolation/extrapolation through the two points
    out[str(B0)] = dict(points=pts, Bstar=Bstar, slope=slope)
    print(f'B0={B0} G:', [(p['Bext'], round(p['beta'], 3), p['n_iter'], p['conv']) for p in pts], 'B*', Bstar, 'slope', slope)
    for p in pts: print(f"   Bext={p['Bext']}: beta {p['beta']:.3f} ({p['beta_range'][0]:.3f}-{p['beta_range'][1]:.3f}) Te {p['Te'][0]:.2f}/{p['Te'][1]:.2f}@{p['XTmax']:.2f}/{p['Te'][2]:.2f} well {p['well']:.2f} compr {p['compr']:.1f} Ew {p['Ew']:.0f} delta {p['delta']:.1f} Er/Ew {p['Er_ratio']} Er_res {p['Er_res']} numax {p['numax']:.3g}@{p['Xnumax']:.2f} axis {p['nuax']:.3g} dphi {p['dphi_last']}")
json.dump(out, open('results/p6_sc_kin.json', 'w'), indent=1)
