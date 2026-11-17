"""Ion kinetics in the self-consistent near-sample layer (argon, sample in an RF-I jet).

1) deterministic: exact solution in total-energy variables for charge exchange with a constant cross section
   (constant mean free path lambda) in the period-averaged potential.  An ion born by charge exchange at x' (at rest, total
   energy e*phi(x')) flies to the sample along eps = const; it survives the path with probability exp(-(x'-x)/lambda).
       n_i(x)          = (Gamma/lambda) * int_x^inf exp(-(x'-x)/lambda) dx' / v(x,x'),   v^2 = 2e(phi(x')-phi(x))/M + v_g^2
       dGamma/dE (x=0) = (Gamma/lambda) * exp(-x'/lambda) / (e phi'(x')),                 E = e(phi(x')-phi(0))
   The potential beyond x = D is continued with the slope at D (the jet presheath), which is the same as injecting at D the
   drift-equilibrium distribution of a uniform field (half-Maxwellian in v_x with temperature e E_D lambda).
   v_g^2 = k T_g / M regularises the birth at rest where the field vanishes (thermal birth velocity of a charge-exchange ion).
2) Monte Carlo (reference): 3D ion velocities, Maxwellian neutrals at T_g, null-collision method with exact relative
   velocities, either charge exchange with sigma = const (same physics as the fluid and 1)) or the Phelps (1994) model
   (isotropic + backward in the CM frame), in the period-averaged field or in the full phi(x,t) of the self-consistent run.
usage: python3 ion_kin.py results/sc_....pkl [nion] [tag]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time
import scipy.constants as co
from numba import njit, prange
e, kB = co.e, co.k
M = 39.948 * 1.66053907e-27; SIG_CX = 5.5e-19; TG = 300.0

# ---------------------------------------------------------------- deterministic (total energy, sigma_cx = const)
def det_cx(xe, phf, lam, Gam, xt=None, Tg=TG, ext=30.0, nper=20, ebins=None):
    """xe, phf: faces and face potentials (phf[-1] = 0 at x = D, phf[0] = -V_dc).  Returns density at target points xt,
    the wall energy distribution and the flux-weighted mean energy."""
    D = xe[-1]; sD = max((phf[-1] - phf[-2]) / (xe[-1] - xe[-2]), 1.0)
    h0 = lam / nper
    X = [xe[0]]
    for j in range(len(xe) - 1):
        k = max(1, int(np.ceil((xe[j + 1] - xe[j]) / h0))); X += list(np.linspace(xe[j], xe[j + 1], k + 1)[1:])
    X += list(D + h0 * np.arange(1, int(ext * nper) + 1))
    X = np.array(X)
    if xt is None: xt = np.linspace(0, D, 301)
    X = np.unique(np.concatenate([X, np.asarray(xt)]))
    P = np.where(X <= D, np.interp(X, xe, phf), sD * (X - D))
    P = np.maximum.accumulate(P)                       # monotone potential (ions are accelerated towards the sample)
    vg2 = kB * Tg / M
    h = np.diff(X); xm = 0.5 * (X[1:] + X[:-1]); att = np.exp(-xm / lam)
    it = np.searchsorted(X, xt)
    n = np.zeros(len(xt))
    for a, i in enumerate(it):
        c = 2 * e * (P[i:-1] - P[i]) / M + vg2; b = 2 * e * np.diff(P[i:]) / (M * h[i:])
        I = np.where(b > 1e-12 * c / h[i:], 2 * (np.sqrt(c + b * h[i:]) - np.sqrt(c)) / np.maximum(b, 1e-300), h[i:] / np.sqrt(c))
        n[a] = Gam / lam * np.sum(np.exp(-(xm[i:] - X[i]) / lam) * I)
    # wall energy distribution (born at rest: all energy is normal energy)
    E = (0.5 * (P[1:] + P[:-1]) - P[0]); w = Gam / lam * att * h
    if ebins is None: ebins = np.linspace(0, np.max(E) * 1.02 + 0.5, 241)
    fE, _ = np.histogram(E, bins=ebins, weights=w); fE = fE / np.diff(ebins) / np.sum(w)
    Emean = np.sum(w * E) / np.sum(w)
    return dict(xt=np.asarray(xt), n=n, ebins=ebins, fE=fE, Emean=Emean, wsum=np.sum(w) / Gam, Tstar=sD * lam,
                frac10=np.sum(w[E > 10]) / np.sum(w), frac15=np.sum(w[E > 15]) / np.sum(w), Emax=float(E.max() if len(E) else 0))

# ---------------------------------------------------------------- Monte Carlo
@njit(cache=True)
def _sig(eps, mode):
    """(Q_iso, Q_back) in m^2; eps = lab energy of the ion with the neutral at rest (= relative-velocity energy M g^2/2)."""
    if mode == 0: return 0.0, SIG_CX
    x = max(eps, 1e-4)
    Qm = 1.15e-18 * x ** -0.1 * (1 + 0.015 / x) ** 0.6
    Qi = 2e-19 * x ** -0.5 / (1 + x) + 3e-19 * x / (1 + x / 3) ** 2.3
    return Qi, 0.5 * (Qm - Qi)

@njit(parallel=True, cache=True)
def _mc(xe, phf0, pr, pim, Nh, omega, rf, ng, Tg, mode, nion, dt, Tstar, seed, maxsteps, numax):
    m = len(xe) - 1; D = xe[-1]; nchunk = 64
    hist = np.zeros((nchunk, m)); Ew = -np.ones(nion); Exw = -np.ones(nion); ang = np.zeros(nion); phw = np.zeros(nion)
    ncoll = np.zeros(nion)
    vth = np.sqrt(kB * Tg / M); T = 2 * np.pi / omega
    for k in prange(nchunk):
        np.random.seed(seed + 7919 * k)
        for ip in range(k, nion, nchunk):
            x = D * (1 - 1e-9); vx = -np.sqrt(2 * Tstar * (-np.log(1 - np.random.random())) / M)
            vy = np.random.normal() * vth; vz = np.random.normal() * vth
            t = np.random.random() * T; tau = -np.log(1 - np.random.random()) / numax; j = m - 1; nc = 0
            for st in range(maxsteps):
                while j > 0 and x < xe[j]: j -= 1
                while j < m - 1 and x >= xe[j + 1]: j += 1
                dxj = xe[j + 1] - xe[j]
                E = -(phf0[j + 1] - phf0[j]) / dxj
                if rf:
                    for q in range(Nh):
                        c = np.cos((q + 1) * omega * t); s = np.sin((q + 1) * omega * t)
                        E -= 2 * ((pr[q, j + 1] - pr[q, j]) * c - (pim[q, j + 1] - pim[q, j]) * s) / dxj
                hist[k, j] += dt
                vx += e / M * E * dt; x += vx * dt; t += dt
                if x <= 0.0:
                    Ew[ip] = 0.5 * M * (vx * vx + vy * vy + vz * vz) / e; Exw[ip] = 0.5 * M * vx * vx / e
                    ang[ip] = np.arctan2(np.sqrt(vy * vy + vz * vz), -vx); phw[ip] = (t % T) / T; ncoll[ip] = nc
                    break
                if x >= D: break
                tau -= dt
                if tau <= 0.0:
                    tau = -np.log(1 - np.random.random()) / numax
                    ux = np.random.normal() * vth; uy = np.random.normal() * vth; uz = np.random.normal() * vth
                    gx = vx - ux; gy = vy - uy; gz = vz - uz; g = np.sqrt(gx * gx + gy * gy + gz * gz)
                    Qi, Qb = _sig(0.5 * M * g * g / e, mode)
                    if np.random.random() * numax < ng * (Qi + Qb) * g:
                        nc += 1
                        if np.random.random() * (Qi + Qb) < Qb:
                            vx, vy, vz = ux, uy, uz                       # charge exchange = backward scattering in the CM frame
                        else:
                            ct = 2 * np.random.random() - 1; st_ = np.sqrt(1 - ct * ct); ph = 2 * np.pi * np.random.random()
                            cx = 0.5 * (vx + ux); cy = 0.5 * (vy + uy); cz = 0.5 * (vz + uz)
                            vx = cx + 0.5 * g * ct; vy = cy + 0.5 * g * st_ * np.cos(ph); vz = cz + 0.5 * g * st_ * np.sin(ph)
    return hist.sum(axis=0), Ew, Exw, ang, phw, ncoll

def load_fields(fn):
    d = pickle.load(open(fn, 'rb')); xe, xc = d['xe'], d['xc']
    from slab_sc import face_interp
    phf = face_interp(xc, xe, d['phi'], -d['Vdc'], 0.0)
    Nh = d['cfg']['Nh']; Vrf = d['cfg']['Vrf']
    pq = [face_interp(xc, xe, d['phq'][q], 0.5 * Vrf if q == 1 else 0.0, 0.0) for q in range(1, Nh + 1)] if Vrf > 0 else []
    p = d['cfg']['p']; ng = p / (kB * TG)
    return d, xe, phf, pq, ng

def run_mc(xe, phf, pq, ng, Gam, mode=0, rf=False, nion=100000, omega=2 * np.pi * 13.56e6, seed=1):
    Nh = len(pq) if rf else 0
    pr = np.array([np.real(a) for a in pq]) if Nh else np.zeros((1, len(xe)))
    pim = np.array([np.imag(a) for a in pq]) if Nh else np.zeros((1, len(xe)))
    lam = 1 / (ng * SIG_CX); sD = max((phf[-1] - phf[-2]) / (xe[-1] - xe[-2]), 1.0)
    Vmax = -phf[0] + (2 * sum(np.abs(a[0]) for a in pq) if Nh else 0) + 10
    gmax = np.sqrt(2 * e * Vmax / M) + 8 * np.sqrt(kB * TG / M)
    gg = np.linspace(1.0, gmax, 2000); numax = 1.05 * ng * max(sum(_sig(0.5 * M * g * g / e, mode)) * g for g in gg)
    dt = min(2 * np.pi / omega / 200, 0.05 / numax)
    hist, Ew, Exw, ang, phw, nc = _mc(xe, phf, pr, pim, max(Nh, 1) if Nh else 0, omega, rf, ng, TG, mode, nion, dt,
                                      e * sD * lam, seed, 20_000_000, numax)
    ok = Ew >= 0; Nw = ok.sum()
    n = Gam / max(Nw, 1) * hist / np.diff(xe)
    return dict(n=n, xc=0.5 * (xe[1:] + xe[:-1]), E=Ew[ok], Ex=Exw[ok], ang=ang[ok], ph=phw[ok], ncoll=nc[ok], Nw=int(Nw), nion=nion,
                Emean=float(Ew[ok].mean()), Exmean=float(Exw[ok].mean()), frac10=float((Ew[ok] > 10).mean()), frac15=float((Ew[ok] > 15).mean()),
                angmean=float(np.degrees(ang[ok]).mean()), dt=dt, numax=numax, mode=mode, rf=rf)

if __name__ == '__main__':
    fn = sys.argv[1]; nion = int(sys.argv[2]) if len(sys.argv) > 2 else 100000; tag = sys.argv[3] if len(sys.argv) > 3 else ''
    d, xe, phf, pq, ng = load_fields(fn); lam = 1 / (ng * SIG_CX); Gam = float(d['Gf'][0])
    out = dict(src=fn, cfg=d['cfg'], Vdc=d['Vdc'], Gam=Gam, lam=lam, fluid=dict(xc=d['xc'], ni=d['ni'], Ei=0.5 * M * d['wf'][0] ** 2 / e))
    t0 = time.time(); out['det'] = det_cx(xe, phf, lam, Gam, xt=d['xc']); out['det_plot'] = det_cx(xe, phf, lam, Gam)
    print(f'det: Emean={out["det"]["Emean"]:.2f} eV  fluid Ei={out["fluid"]["Ei"]:.2f}  n(D) det/fluid={out["det"]["n"][-1]/d["ni"][-1]:.3f}  ({time.time()-t0:.1f}s)', flush=True)
    cases = [('cx_avg', 0, False)] + ([('cx_rf', 0, True), ('ph_rf', 1, True)] if d['cfg']['Vrf'] > 0 else [('ph_avg', 1, False)])
    for name, mode, rf in cases:
        t0 = time.time(); r = run_mc(xe, phf, pq, ng, Gam, mode=mode, rf=rf, nion=nion); out[name] = r
        print(f'{name}: Emean={r["Emean"]:.2f}  Ex={r["Exmean"]:.2f}  >10eV {r["frac10"]:.3f}  >15eV {r["frac15"]:.3f}  ang {r["angmean"]:.1f} deg  '
              f'Nw/N={r["Nw"]}/{nion}  coll {r["ncoll"].mean():.1f}  ({time.time()-t0:.0f}s)', flush=True)
    base = os.path.basename(fn).replace('sc_', 'ionkin_').replace('.pkl', f'{tag}.pkl')
    pickle.dump(out, open('results/' + base, 'wb'))
