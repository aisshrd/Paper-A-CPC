"""Run one near-surface-layer case with the planar total-energy solver and save diagnostics.
usage: python3 slab_case.py p[Pa] Ea[V/m] Vb[V] mirror(cc|even) [L Nh n_e m tag]
Model (half gap [0,D], surface at x=0):
  phi0(x) = -Vb (1 - x/s0)^2 for x < s0 (time-averaged sheath barrier, uniform ion density), 0 beyond;
  E(x,t)  = Ea g(x) cos(w t),  g = 1 for x <= d, exp(-((x-d)/delta)^2) for x > d   (oscillation region of the layer);
  bulk x > d field-free.  Env: D_MM, S0_MM, D_OSC_MM, DELTA_MM, UMAX, CLUSTER, LIMITER."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, sys, os, pickle, time
os.environ.setdefault('PYPARDISO_MKL_RT', '/usr/local/lib/libmkl_rt.so.3')
from ekin import load_argon
from ekin_slab_te import SlabKineticsTE

def model(p, Ea, Vb):
    D = float(os.environ.get('D_MM', 8)) * 1e-3; s0 = float(os.environ.get('S0_MM', 1.0)) * 1e-3
    d = float(os.environ.get('D_OSC_MM', 2.0)) * 1e-3; dl = float(os.environ.get('DELTA_MM', 0.25)) * 1e-3
    phi0 = lambda x: -Vb * max(0.0, 1 - x / s0) ** 2
    g = lambda x: 1.0 if x <= d else np.exp(-((x - d) / dl) ** 2)
    return dict(D=D, s0=s0, d=d, delta=dl, phi0=phi0, g=g, Ea=Ea, Vb=Vb, p=p)

def run(p, Ea, Vb, mirror='cc', L=3, Nh=2, n_e=100, m=48, tag='', save=True, verbose=True):
    xs = load_argon('data/Ar_LXCat.txt')
    M = model(p, Ea, Vb); ng = p / (1.380649e-23 * 300.0); w = 2 * np.pi * 13.56e6
    umax = float(os.environ.get('UMAX', 45.0)); cl = float(os.environ.get('CLUSTER', 1.8))
    t0 = time.time()
    S = SlabKineticsTE(xs, D=M['D'], ng=ng, omega=w, phi0=M['phi0'], Eq={1: (lambda x: 0.5 * Ea * M['g'](x))},
                       L=L, Nh=Nh, n_e=n_e, umax=umax, m=m, cluster=cl, absorb=True, mirror=mirror, iz_mult=2)
    f = S.solve(passes=2, limiter_iters=int(os.environ.get('LIMITER', 2)), verbose=verbose)
    out = dict(cfg=dict(p=p, Ea=Ea, Vb=Vb, mirror=mirror, L=L, Nh=Nh, n_e=n_e, m=m, D=M['D'], s0=M['s0'], d=M['d'], delta=M['delta'], umax=umax),
               xc=S.xc, dx=S.dx, phi=S.phic, eta=float(S.eta), eta1=float(S.eta1), ec=S.ec, ukin=S.ukin, w=S.w)
    H = {}
    for n in range(0, Nh + 1):
        H[n] = dict(n=S.density(f, n), iz=S.rate(f, S.nuiz, n), ex=S.rate(f, S.nuex, n), em=S.rate(f, S.nu4p, n),
                    en=S.energy_density(f, n), flux=S.flux_cells(f, n), surf=S.surface_flux(f, n))
    out['H'] = H
    out['f0'] = np.real(f[Nh, 0]); out['f0_2'] = f[Nh + 2, 0] if Nh >= 2 else None; out['f0_1'] = f[Nh + 1, 0]
    # conservation check: eta = (R_iz - Gamma_surface)/N  (iz_mult = 2: +1 electron per ionisation)
    Ntot = np.sum(np.real(H[0]['n']) * S.dx); Riz = np.sum(np.real(H[0]['iz']) * S.dx); Gs = np.real(H[0]['surf'][0])
    out['balance'] = dict(N=Ntot, Riz=Riz, Gs=Gs, eta_check=(Riz - Gs) / Ntot)
    out['limiter_hist'] = getattr(S, 'limiter_hist', None); out['time'] = time.time() - t0; out['nTot'] = S.nTot
    if verbose:
        n0 = np.real(H[0]['n']); iz0 = np.real(H[0]['iz'])
        print(f'p={p} Ea={Ea} Vb={Vb} {mirror} L={L} Nh={Nh} ne={n_e} m={m}: eta={S.eta:.4e} (check {out["balance"]["eta_check"]:.4e}, '
              f'nu_iz_avg {Riz/Ntot:.3e}, loss {Gs/Ntot:.3e})  time {out["time"]:.0f}s  N={S.nTot}', flush=True)
    if save:
        fn = f'results/slab_p{p}_E{Ea:g}_V{Vb:g}_{mirror}_L{L}_N{Nh}_e{n_e}_m{m}{tag}.pkl'
        pickle.dump(out, open(fn, 'wb')); out['file'] = fn
    return out

if __name__ == '__main__':
    a = sys.argv
    p, Ea, Vb, mir = float(a[1]), float(a[2]), float(a[3]), a[4]
    L = int(a[5]) if len(a) > 5 else 3; Nh = int(a[6]) if len(a) > 6 else 2; ne = int(a[7]) if len(a) > 7 else 100
    m = int(a[8]) if len(a) > 8 else 48; tag = a[9] if len(a) > 9 else ''
    run(p, Ea, Vb, mir, L, Nh, ne, m, tag)
