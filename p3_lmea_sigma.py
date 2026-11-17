"""Paper 3 additions (05.11.2026, after the logic audit): (a) LMEA conductivity (RF table: complex mobility vs mean energy, looked up
at the local kinetic mean energy); (b) LMEA by the temperature of the electrons above 1 eV (the heated group) for nu_iz and sigma;
(c) ring-averaged comparison of the kinetic nu_iz with the Monte Carlo rings. Appends to results/p3_cylcoef.pkl."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, scipy.constants as co, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from ekin import load_argon, EnergyGridKinetics
e = co.e; xs = load_argon('data/Ar_LXCat.txt'); ng = 3.21e21; omega = 2 * np.pi * 13.56e6; n0 = 1e17
d = pickle.load(open('results/p3_cylcoef.pkl', 'rb')); X = d['X']; n = d['n']; Te = d['Te']
K = EnergyGridKinetics(xs, umax=50., nu=200, L=1, Nh=1, ng=ng); tab = []
for E0 in np.logspace(0.0, 3.2, 60):
    g = K.solve(E0, omega); mu = K.drift(g, 1) / (E0 / 2); tab.append((K.mean_energy(g), K.rate_iz(g), K.rate_exc_total(g), abs(mu), np.angle(-mu)))
tab = np.array(tab); o = np.argsort(tab[:, 0]); tab = tab[o]
def look(ub):
    return (np.exp(np.interp(ub, tab[:, 0], np.log(np.maximum(tab[:, 1], 1e-30)))), np.exp(np.interp(ub, tab[:, 0], np.log(np.maximum(tab[:, 2], 1e-30)))),
            np.interp(ub, tab[:, 0], tab[:, 3]), np.interp(ub, tab[:, 0], tab[:, 4]))
ub = 1.5 * Te; iz1, ex1, mu1, ph1 = look(ub)
d['lmea_rf']['sig_abs'] = e * n0 * n * mu1; d['lmea_rf']['sig_arg'] = ph1     # phase of sigma (= -e n mu)
# (b) heated-group temperature from the stored kinetic solution
s = pickle.load(open('results/te_reffields_ne120_m16_L1_o2.pkl', 'rb')); F0 = s['F0']; u = np.maximum(s['ukin'], 0); w = s['w']
sel = u > 1.0; Th = 2 / 3 * np.sum(np.where(sel, w * F0 * u, 0), axis=0) / np.sum(np.where(sel, w * F0, 0), axis=0)
assert np.allclose(s['X'], X)
iz2, ex2, mu2, ph2 = look(1.5 * Th)
d['lmea_hot'] = dict(Th=Th, nuiz=iz2, nuex=ex2, sig_abs=e * n0 * n * mu2, sig_arg=ph2)
# (c) ring averages of the kinetic nu_iz (rings of the MC reference)
rm = d['mc']['r']; xf = np.linspace(0, 1, 4001); Xa = X[::-1]; nf = np.interp(xf, Xa, n[::-1]); nuf = np.exp(np.interp(xf, Xa, np.log(np.maximum(d['nuiz'][::-1], 1e-3))))
ring = []
for k in range(len(rm)):
    sl = (xf >= k / len(rm)) & (xf < (k + 1) / len(rm)); ring.append(np.sum(xf[sl] * nf[sl] * nuf[sl]) / np.sum(xf[sl] * nf[sl]))
d['nuiz_ring'] = np.array(ring)
pickle.dump(d, open('results/p3_cylcoef.pkl', 'wb'))
print('X     ', np.round(X, 2)); print('Te    ', np.round(Te, 2)); print('T_hot ', np.round(Th, 2))
print('|sigma| NL   ', np.round(np.abs(d['sigma']), 2)); print('|sigma| LFA  ', np.round(d['lfa']['sig_abs'], 2)); print('|sigma| LMEA ', np.round(d['lmea_rf']['sig_abs'], 2)); print('|sigma| LMEA(hot)', np.round(d['lmea_hot']['sig_abs'], 2))
print('-arg NL  ', np.round(-np.angle(d['sigma']) * 180 / np.pi, 1)); print('-arg LFA ', np.round(-np.angle(-np.exp(1j * d['lfa']['sig_arg'])) * 180 / np.pi, 1)); print('-arg LMEA', np.round(-ph1 * 180 / np.pi, 1)); print('-arg LMEA(hot)', np.round(-ph2 * 180 / np.pi, 1))
print('nuiz NL  ', np.array2string(d['nuiz'], precision=3)); print('nuiz LMEA(rf)', np.array2string(d['lmea_rf']['nuiz'], precision=3)); print('nuiz LMEA(hot)', np.array2string(iz2, precision=3))
print('rings', rm); print('nuiz MC   ', np.round(d['mc']['nuiz'], 0)); print('nuiz NL ring', np.round(d['nuiz_ring'], 0))
