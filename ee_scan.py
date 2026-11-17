"""Electron kinetics with e-e collisions in the group's working gases.
13.56 MHz: harmonic balance Nh = 6, e-e operator frozen at the period-averaged f0 (self-consistent).
1.76 MHz : quasi-static limit (f follows the instantaneous field), e-e operator from the period-averaged f0 (self-consistent).
n_e = 1e16, 1e17, 1e18, 1e19 m^-3.  Output: results/ee_scan.pkl  (merged on rerun)
usage: python3 ee_scan.py nproc [hb|qs]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time, warnings, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import scipy.constants as co
from multiprocessing import Pool
from ekin_ee import MixtureKineticsEE
from mix_scan import GASES, PRESS, SP, TG, group
NE = [1e16, 1e17, 1e18, 1e19]
ENR = np.geomspace(8, 250, 15)
NH = 6
OUT = 'results/ee_scan.pkl'
xq, wq = np.polynomial.legendre.leggauss(24); TH = 0.25 * np.pi * (xq + 1); WQ = wq / wq.sum()

def grouped(K, f, comp):
    Rg = {}
    for kind, sn, prod, thr, r in K.rates(f):
        g = group(kind, sn, prod); Rg[g] = Rg.get(g, 0.0) + comp[sn] * r
    Pg = {}
    for k, v in K.power_split(f).items():
        if k in ('elastic', 'rotation'): Pg[k] = v; continue
        kind, sn, prod = k.split(':', 2); g = group(kind, sn, prod); Pg[g] = Pg.get(g, 0.0) + v
    return Rg, Pg

def run_hb(args):
    gname, p, ne = args
    comp = GASES[gname]; ng = p / (co.k * TG); w = 2 * np.pi * 13.56e6
    K = MixtureKineticsEE([(SP[s], x) for s, x in comp.items()], ne=ne, umax=60., nu=300, L=1, Nh=NH, ng=ng, Tg=TG)
    th = np.linspace(0, 2 * np.pi, 97)[:-1]; out = []
    for EN in ENR:
        f = K.solve(np.sqrt(2) * EN * 1e-21 * ng, w)
        Rg, Pg = grouped(K, f, comp)
        Riz_h = [sum(comp[sn] * r for kind, sn, prod, thr, r in K.rates(f, harm=q) if kind == 'iz') for q in range(NH + 1)]
        Rt = np.real(Riz_h[0]) + sum(2 * np.real(Riz_h[q] * np.exp(1j * q * th)) for q in range(1, NH + 1))
        u0 = K.mean_energy(f); u2 = abs(complex(np.sum(K.w * K.uc * f[NH + 2, 0])))
        Ptot = sum(Pg.values())
        out.append(dict(EN=EN, u=u0, mod_u=2 * u2 / u0, rates=Rg, power=Pg, Riz_peak=float(Rt.max() / max(Rt.mean(), 1e-300)),
                        f0=np.real(f[NH, 0]).copy(), neg=float(np.min(np.real(f[NH, 0]))), lnL=K.lnL, it=K.it,
                        ee_cons=K.ee_power(f) / max(Ptot, 1e-300)))
    return (gname, p, 13.56, ne), dict(uc=K.uc, ng=ng, rows=out)

def run_qs(args):
    gname, p, ne = args
    comp = GASES[gname]; ng = p / (co.k * TG)
    K = MixtureKineticsEE([(SP[s], x) for s, x in comp.items()], ne=ne, umax=60., nu=300, L=1, Nh=0, ng=ng, Tg=TG)
    out = []
    for EN in ENR:
        Es = [max(np.sqrt(2) * EN * 1e-21 * ng * np.cos(t), 1e-3 * EN * 1e-21 * ng) for t in TH]
        if K._A is None: K.solve(EN * 1e-21 * ng)              # start the e-e operator from the DC solution at E_rms
        u_old = None
        for outer in range(30):
            fs = []
            for E in Es: fs.append(K.solve(E, freeze_ee=True, maxit=40))
            f0 = sum(wg * np.real(f[0, 0]) for wg, f in zip(WQ, fs)); u = float(np.sum(K.w * K.uc * f0))
            K.set_ee(f0)
            if u_old is not None and abs(u - u_old) < 2e-4 * u: break
            u_old = u
        Rg = {}; Pg = {}; Riz = []
        for wg, f in zip(WQ, fs):
            r_, p_ = grouped(K, f, comp)
            for k, v in r_.items(): Rg[k] = Rg.get(k, 0.0) + wg * v
            for k, v in p_.items(): Pg[k] = Pg.get(k, 0.0) + wg * v
            Riz.append(r_.get('iz', 0.0))
        Riz = np.array(Riz)
        out.append(dict(EN=EN, u=u, rates=Rg, power=Pg, Riz_peak=float(Riz.max() / max(np.sum(WQ * Riz), 1e-300)), f0=f0,
                        neg=float(f0.min()), lnL=K.lnL, it=outer))
    return (gname, p, 1.76, ne), dict(uc=K.uc, ng=ng, rows=out)

if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2; mode = sys.argv[2] if len(sys.argv) > 2 else 'hb'
    jobs = list(itertools.product(GASES, PRESS, NE)); t0 = time.time()
    jobs.sort(key=lambda j: -j[2])                      # slowest (high n_e) first
    res = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
    fn = run_hb if mode == 'hb' else run_qs
    with Pool(nproc) as pool:
        for key, val in pool.imap_unordered(fn, jobs):
            res[key] = val
            pickle.dump(res, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
            print(key, f'{time.time()-t0:.0f}s', 'neg', f"{min(r['neg'] for r in val['rows']):.1e}",
                  'it', max(r['it'] for r in val['rows']), flush=True)
