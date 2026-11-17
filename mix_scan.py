"""Electron kinetics in the group's working gases (Ar, N2, O2, air, Ar+N2, Ar+O2) under RF-I treatment conditions:
p = 13.3 / 26.6 / 133 Pa, f = 1.76 / 13.56 MHz, uniform RF field E(t) = sqrt(2) E_rms cos(wt), harmonic balance Nh = 6, two-term,
temporal growth, CAR rotations.  Output: results/mix_scan.pkl
usage: python3 mix_scan.py [nproc]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time, warnings, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import scipy.constants as co
from multiprocessing import Pool
from ekin_mix import MixtureKinetics, load_species
GASES = {'Ar': {'Ar': 1.0}, 'N2': {'N2': 1.0}, 'O2': {'O2': 1.0}, 'air': {'N2': 0.79, 'O2': 0.21},
         'Ar+N2 90/10': {'Ar': 0.9, 'N2': 0.1}, 'Ar+N2 50/50': {'Ar': 0.5, 'N2': 0.5},
         'Ar+O2 90/10': {'Ar': 0.9, 'O2': 0.1}, 'Ar+O2 50/50': {'Ar': 0.5, 'O2': 0.5}}
PRESS = [13.3, 26.6, 133.0]; FREQ = [1.76, 13.56]
ENRMS = np.geomspace(10, 1000, 19)
NH = 6; TG = 300.0
SP = {k: load_species(k) for k in ('Ar', 'N2', 'O2')}

def group(kind, sname, prod):
    p = (prod or '')
    if kind == 'iz': return 'iz'
    if kind == 'att': return 'att'
    if sname == 'O2':
        if p.startswith('O(3P)+O'): return 'diss_O2'
        if p.startswith('O2(v=') or p.startswith('O2 (v='): return 'vib'
        if 'a1Dg' in p: return 'O2(a1Dg)'
        if 'b1Sg' in p: return 'O2(b1Sg)'
        return 'el_O2'
    if sname == 'N2':
        if 'A3Su' in p: return 'N2(A)'
        if p.startswith('N2 (v=') or p.startswith('N2(v='): return 'vib'
        return 'el_N2'
    return 'exc_Ar'

def run(args):
    gname, p, fMHz = args
    comp = GASES[gname]; ng = p / (co.k * TG); w = 2 * np.pi * fMHz * 1e6
    K = MixtureKinetics([(SP[s], x) for s, x in comp.items()], umax=60., nu=300, L=1, Nh=NH, ng=ng, Tg=TG)
    KDC = MixtureKinetics([(SP[s], x) for s, x in comp.items()], umax=60., nu=300, L=1, Nh=0, ng=ng, Tg=TG)
    th = np.linspace(0, 2 * np.pi, 97)[:-1]
    out = []
    for EN in ENRMS:
        E0 = np.sqrt(2) * EN * 1e-21 * ng
        f = K.solve(E0, w)
        R = K.rates(f); Rg = {}
        for kind, sn, prod, thr, r in R:
            g = group(kind, sn, prod); Rg[g] = Rg.get(g, 0.0) + comp[sn] * r          # per molecule of the MIXTURE (x_s k_s)
        # time-resolved ionisation (sum of harmonics) -> peak/mean
        Riz_h = [sum(comp[sn] * r for kind, sn, prod, thr, r in K.rates(f, harm=q) if kind == 'iz') for q in range(NH + 1)]
        Rt = np.real(Riz_h[0]) + sum(2 * np.real(Riz_h[q] * np.exp(1j * q * th)) for q in range(1, NH + 1))
        P = K.power_split(f); Pg = {}
        for k, v in P.items():
            if k in ('elastic', 'rotation'): Pg[k] = v; continue
            kind, sn, prod = k.split(':', 2); g = group(kind, sn, prod); Pg[g] = Pg.get(g, 0.0) + v
        fd = KDC.solve(EN * 1e-21 * ng); RD = {}
        for kind, sn, prod, thr, r in KDC.rates(fd):
            g = group(kind, sn, prod); RD[g] = RD.get(g, 0.0) + comp[sn] * r
        out.append(dict(EN=EN, raw=[(k_, s_, p_, t_, r_) for k_, s_, p_, t_, r_ in R], u=K.mean_energy(f), u_dc=KDC.mean_energy(fd), rates=Rg, rates_dc=RD, Riz_t=Rt, Riz_peak=float(Rt.max() / max(Rt.mean(), 1e-300)),
                        power=Pg, nu_g=K.nu_g, vz1=K.drift_b(1), f0=np.real(f[NH, 0]).copy(), f0dc=np.real(fd[0, 0]).copy(),
                        neg=float(np.min(np.real(f[NH, 0]))), it=K.it))
    return (gname, p, fMHz), dict(uc=K.uc, ng=ng, w=w, rows=out)

if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else list(GASES)
    jobs = list(itertools.product(only, PRESS, FREQ)); t0 = time.time()
    res = pickle.load(open('results/mix_scan.pkl', 'rb'))['res'] if len(sys.argv) > 2 else {}
    with Pool(nproc) as pool:
        for key, val in pool.imap_unordered(run, jobs):
            res[key] = val; print(key, f'{time.time()-t0:.0f}s', 'neg min', min(r['neg'] for r in val['rows']), flush=True)
    pickle.dump(dict(res=res, GASES=GASES, PRESS=PRESS, FREQ=FREQ, ENRMS=ENRMS, NH=NH), open('results/mix_scan.pkl', 'wb'))
