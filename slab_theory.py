"""Simple estimates for the 2w modulation of the ionising tail in the field-free region:
   d(df)/dt = D d2(df)/dx2 - nu* df  ->  df ~ exp(-k x + i 2w t),  k_n = sqrt((nu* + i n w)/D)
   ratio R_2/R_0 decays as exp(-(Re k_2 - k_0) x), its phase falls as -Im k_2 x.
D = v^2/(3 nu_tot), nu* = inelastic + ionisation frequency, evaluated at the energy that dominates ionisation."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np
from ekin import load_argon, GAMMA
xs = load_argon('data/Ar_LXCat.txt')
sig = lambda d, x: np.interp(x, d[:, 0], d[:, 1])
w = 2 * np.pi * 13.56e6

def ks(p, u):
    ng = p / (1.380649e-23 * 300); v = GAMMA * np.sqrt(u)
    nm = ng * sig(xs['elastic'], u) * v
    ns = ng * v * (sum(sig(d, u) * (u > th) for th, d, _ in xs['inel']) + sig(xs['iz'][1], u) * (u > xs['iz'][0]))
    D = v ** 2 / (3 * (nm + ns)); k0 = np.sqrt(ns / D); k2 = np.sqrt((ns + 2j * w) / D)
    return dict(nu_star=ns, nu_m=nm, D=D, k0=k0, k2=k2, ratio_decay_len=1 / max(k2.real - k0, 1e-30),
                phase_slope_deg_per_mm=np.degrees(k2.imag) * 1e-3, lag_deg=np.degrees(np.arctan(2 * w / ns)))

if __name__ == '__main__':
    for p in (13.3, 40, 133):
        for u in (17, 20, 24):
            r = ks(p, u)
            print(f'p={p:6} u={u}: nu*={r["nu_star"]:.2e} nu*/2w={r["nu_star"]/2/w:5.2f}  1/k0={1e3/r["k0"]:.2f} mm  '
                  f'ratio decay {r["ratio_decay_len"]*1e3:.1f} mm  phase slope {r["phase_slope_deg_per_mm"]:.1f} deg/mm  local lag {r["lag_deg"]:.0f} deg')
