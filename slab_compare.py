"""Fold the full-gap Monte Carlo onto the half gap (with the symmetry of the case) and take time harmonics,
in the normalisation of the solver (int_0^D n0 dx = 1)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle

def mc_harmonics(res, nmax=3):
    acc = res['acc'].copy(); nph, nbx = res['nph'], res['nbx']; Lx = res['Lx']; D = Lx / 2
    half = nbx // 2
    L = acc[:, :half, :]; R = acc[:, half:, :][:, ::-1, :].copy()
    R[:, :, 5] *= -1                                   # x-velocity flips under x -> 2D - x
    if res['mirror'] == 'cc': R = np.roll(R, -nph // 2, axis=0)   # f(2D-x, t+T/2) = f(x,t)
    A = 0.5 * (L + R)
    dxb = D / half; xb = (np.arange(half) + 0.5) * dxb
    tot = np.mean(np.sum(A[:, :, 0], axis=1)) * dxb   # time-averaged int n dx
    ph_t = (np.arange(nph) + 0.5) / nph * 2 * np.pi
    out = {}
    for n in range(nmax + 1):
        c = np.exp(-1j * n * ph_t)[:, None] / nph
        corr = 1.0 if n == 0 else (np.pi * n / nph) / np.sin(np.pi * n / nph)   # phase-bin averaging
        q = {}
        for k, name in enumerate(['n', 'iz', 'ex', 'em', 'en', 'vx']):
            q[name] = np.sum(A[:, :, k] * c, axis=0) * corr / tot
        out[n] = q
    # surface (fold both sides; right side shifted by half a period for 'cc')
    S0 = res['surf'][0]; S1 = res['surf'][1]
    if res['mirror'] == 'cc': S1 = np.roll(S1, -nph // 2, axis=0)
    S = 0.5 * (S0 + S1)
    nper = res['periods'] - res['burn']; T = 2 * np.pi / (2 * np.pi * 13.56e6)
    # S = absorbed / N(full gap) per phase bin, summed over periods; per half gap N/2 -> x2; samples of the density were per
    # sample_every steps, the absorbed count is per period -> rate = count/(nper T)
    surf = {}
    for n in range(nmax + 1):
        c = np.exp(-1j * n * ph_t)
        corr = 1.0 if n == 0 else (np.pi * n / nph) / np.sin(np.pi * n / nph)
        surf[n] = (2 * np.sum(S[:, 0] * c) / (nper * T) * corr, 2 * np.sum(S[:, 1] * c) / (nper * T) * corr)
    # density accumulators were sampled every sample_every steps: tot already normalises them
    return xb, out, surf, A, tot

def solver_profiles(out, nmax=None):
    H = out['H']; nmax = nmax if nmax is not None else max(H)
    return out['xc'], {n: H[n] for n in range(nmax + 1)}
