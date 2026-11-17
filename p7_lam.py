"""Energy relaxation length lambda_eps = <v>/sqrt(3 nu_m nu_eps) and mean free path from the local isotropic distribution of a
converged discharge pkl, at the wall, mid-radius and axis.  nu_m: elastic momentum transfer (mixture); nu_eps: total energy-loss
rate (elastic+rotational eloss, inelastic thresholds, ionization threshold) divided by the mean energy.   usage: python3 p7_lam.py pkl..."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, scipy.constants as co
from ekin_cyl_mix import mixture_xs
sig = lambda s, x: np.interp(x, s[:, 0], s[:, 1], left=0.0, right=s[-1, 1])
for fn in sys.argv[1:]:
    d = pickle.load(open(fn, 'rb')); xs = mixture_xs(d['mix']); ng = d['ng']
    u = np.maximum(d['ukin'], 0); v = np.sqrt(2 * co.e * u / co.m_e); w = d['w']; F = d['F0']
    out = []
    for node in (0, len(d['X']) // 2, -1):
        uu, vv, ww, FF = u[:, node], v[:, node], w[:, node], F[:, node]; n = np.sum(ww * FF); um = np.sum(ww * FF * uu) / n
        num = np.sum(ww * FF * ng * sig(xs['elastic'], uu) * vv) / n                       # momentum transfer, s^-1
        Pel = np.sum(ww * FF * ng * sig(xs['eloss'], uu) * vv * uu) / n                   # elastic + rotational energy loss, eV/s (eloss = 2m/M sigma + rot)
        Pin = sum(np.sum(ww * FF * ng * sig(s, uu) * vv * (uu > th)) * th for th, s, _ in xs['inel']) / n
        izs = xs['iz'] if isinstance(xs['iz'], list) else [xs['iz']]
        Piz = sum(np.sum(ww * FF * ng * sig(s, uu) * vv * (uu > th)) * th for th, s in izs) / n
        nue = (Pel + Pin + Piz) / um; vm = np.sum(ww * FF * vv) / n
        out.append((d['X'][node], um, num, nue, vm / num * 100, vm / np.sqrt(3 * num * nue) * 100))
    print(fn.split('/')[-1], d['B_ext'], 'G')
    for X, um, num, nue, mfp, lam in out: print(f'   r/R={X:.2f}: <u>={um:.2f} eV  nu_m={num:.2e}  nu_eps={nue:.2e} s^-1  mfp={mfp:.2f} cm  lambda_eps={lam:.1f} cm')
