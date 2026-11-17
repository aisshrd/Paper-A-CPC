"""Floating barrier: for given (p, Ea) find the static barrier Vb such that the layer is self-sustained (eta = 0):
electron loss to the surface = ionisation.  eta(Vb) is monotone (higher barrier -> smaller loss); secant on ln(Riz/Gs).
usage: python3 slab_float.py p Ea Vb0 mirror L Nh n_e m"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, sys, os, json
os.environ.setdefault('LIMITER', '1')
import slab_case
M_AR = 39.948 * 1.66053907e-27; E = 1.602176634e-19

def main():
    a = sys.argv; p, Ea, Vb, mir = float(a[1]), float(a[2]), float(a[3]), a[4]
    L, Nh, ne, m = int(a[5]), int(a[6]), int(a[7]), int(a[8]); tag = a[9] if len(a) > 9 else ''
    hist = []
    def F(Vb):
        out = slab_case.run(p, Ea, round(Vb, 4), mir, L, Nh, ne, m, tag='_fl' + tag, save=True, verbose=False)
        b = out['balance']; r = np.log(max(b['Riz'], 1e-300) / b['Gs'])
        H = out['H']; x = out['xc']; s0 = out['cfg']['s0']; n0 = np.real(H[0]['n']); en = np.real(H[0]['en'])
        ns = np.interp(s0, x, n0); Ts = 2 / 3 * np.interp(s0, x, en) / ns; uB = np.sqrt(E * Ts / M_AR)
        hist.append(dict(Vb=Vb, r=r, eta=out['eta'], Riz=b['Riz'], Gs=b['Gs'], bohm=b['Gs'] / (ns * uB), Ts=Ts, file=out['file']))
        print(f'  Vb={Vb:7.3f}  ln(Riz/Gs)={r:+.4f}  eta={out["eta"]:.3e}  Gamma/(n_s u_B)={hist[-1]["bohm"]:.3f}', flush=True)
        return r
    V0, r0 = Vb, F(Vb); V1 = Vb + (2.0 if r0 < 0 else -2.0); r1 = F(V1)
    for it in range(8):
        if abs(r1) < 0.01: break
        V2 = V1 - r1 * (V1 - V0) / (r1 - r0); V2 = float(np.clip(V2, V1 - 6, V1 + 6))
        V0, r0, V1, r1 = V1, r1, V2, F(V2)
    res = dict(p=p, Ea=Ea, Vb=V1, mirror=mir, hist=hist, grid=(L, Nh, ne, m))
    json.dump(res, open(f'results/float_p{p}_E{Ea:g}_{mir}_L{L}_N{Nh}_e{ne}_m{m}{tag}.json', 'w'), indent=1, default=float)
    print('FLOAT', p, Ea, V1)

if __name__ == '__main__':
    main()
