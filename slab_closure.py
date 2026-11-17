"""Closure of the near-surface layer model: find (Ea, Vb) such that
   (1) the layer is self-sustained:  ionisation = electron loss to the surface   (eta = 0)
   (2) the dc electron flux to the surface equals the Bohm ion flux at the time-averaged sheath edge x = s0:
         Gamma_e = n(s0) u_B,  u_B = sqrt(e Te(s0)/M),  Te = 2/3 <eps>   (ambipolar: Gamma_i = Gamma_e = ionisation in the half gap)
Unknowns y = (ln Ea, Vm), Vb = Ea*ell + Vm (ell = d + delta sqrt(pi)/2: RF potential amplitude per unit field).
Residuals r = (ln(Riz/Gs), ln(Gs/(n_s u_B))).  Broyden iteration with a finite-difference initial Jacobian.
usage: python3 slab_closure.py p Ea0 Vb0 mirror L Nh n_e m"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, sys, os, pickle, json
os.environ.setdefault('LIMITER', '1')
import slab_case
M_AR = 39.948 * 1.66053907e-27; E = 1.602176634e-19

def resid(out):
    H = out['H']; x = out['xc']; s0 = out['cfg']['s0']
    n0 = np.real(H[0]['n']); en = np.real(H[0]['en'])
    ns = np.interp(s0, x, n0); Ts = 2 / 3 * np.interp(s0, x, en) / ns
    uB = np.sqrt(E * Ts / M_AR); b = out['balance']
    return np.array([np.log(b['Riz'] / b['Gs']), np.log(b['Gs'] / (ns * uB))]), dict(ns=ns, Ts=Ts, uB=uB, Riz=b['Riz'], Gs=b['Gs'], eta=out['eta'])

def main():
    a = sys.argv; p = float(a[1]); Ea0 = float(a[2]); Vb0 = float(a[3]); mir = a[4]
    L, Nh, ne, m = int(a[5]), int(a[6]), int(a[7]), int(a[8])
    M = slab_case.model(p, 1, 1); ell = M['d'] + M['delta'] * np.sqrt(np.pi) / 2
    log = []
    def F(y):
        Ea = float(np.exp(y[0])); Vb = float(Ea * ell + y[1])
        out = slab_case.run(p, round(Ea, 3), round(Vb, 4), mir, L, Nh, ne, m, tag='_clos', save=True, verbose=False)
        r, info = resid(out); info.update(Ea=Ea, Vb=Vb, r=r.tolist(), file=out['file']); log.append(info)
        print(f'  Ea={Ea:8.1f} Vb={Vb:7.3f} (Vm={y[1]:6.3f})  r=({r[0]:+.4f},{r[1]:+.4f})  Te_s={info["Ts"]:.2f} eta={out["eta"]:.3e}', flush=True)
        return r
    y = np.array([np.log(Ea0), Vb0 - Ea0 * ell]); r = F(y)
    h = np.array([0.05, 0.5]); J = np.zeros((2, 2))
    for k in range(2):
        yk = y.copy(); yk[k] += h[k]; J[:, k] = (F(yk) - r) / h[k]
    for it in range(10):
        dy = -np.linalg.solve(J, r)
        dy[0] = np.clip(dy[0], -0.4, 0.4); dy[1] = np.clip(dy[1], -4, 4)
        yn = y + dy; rn = F(yn)
        J = J + np.outer((rn - r) - J @ dy, dy) / (dy @ dy)       # Broyden update
        y, r = yn, rn
        if np.max(np.abs(r)) < 3e-3: break
    res = dict(p=p, mirror=mir, Ea=float(np.exp(y[0])), Vb=float(np.exp(y[0]) * ell + y[1]), ell=ell, log=log, grid=(L, Nh, ne, m))
    json.dump(res, open(f'results/closure_p{p}_{mir}_L{L}_N{Nh}_e{ne}_m{m}.json', 'w'), indent=1, default=float)
    print('CLOSURE', json.dumps({k: res[k] for k in ('p', 'Ea', 'Vb')}))

if __name__ == '__main__':
    main()
