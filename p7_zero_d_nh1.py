"""Homogeneous (0D two-term, harmonic balance) operating points with ONE time harmonic (L=1, N_h=1), i.e. the approximation of the
self-consistent runs of Paper 7, without electron-electron collisions and without the growth term (the convention of the cylinder
solver), with the ion mobility of ekin_cyl_mix.ion_mobility, for Lambda = 2 cm and R/2.405.  Root of (k_iz-k_att) N = mu_i T_e / Lambda^2
in ln(E/N) by the Illinois method.  -> results/p7_zero_d_nh1.pkl {(gas, p, lam): record}   usage: python3 p7_zero_d_nh1.py [growth]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time, scipy.constants as co
os.environ.setdefault('OMP_NUM_THREADS', '1')
from ekin_mix import MixtureKinetics
from mix_scan import SP, group
from ekin_cyl_mix import ion_mobility
growth = 'growth' in sys.argv
GASES = {'Ar': [('Ar', 1.0)], 'Ar+N2 90/10': [('Ar', 0.9), ('N2', 0.1)], 'Ar+N2 50/50': [('Ar', 0.5), ('N2', 0.5)], 'N2': [('N2', 1.0)], 'air': [('N2', 0.79), ('O2', 0.21)]}
p = 13.3; ng = p / (co.k * 300); om = 2 * np.pi * 13.56e6
OUT = 'results/p7_zero_d_nh1%s.pkl' % ('_growth' if growth else ''); res = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
def point(K, mix, EN):
    f = K.solve(np.sqrt(2) * EN * 1e-21 * ng, om); u = K.mean_energy(f); rates = {}; power = {}
    for kind, sname, prod, th, r in K.rates(f):
        if kind == 'iz': rates['iz'] = rates.get('iz', 0) + r * dict(mix)[sname]
        elif kind == 'att': rates['att'] = rates.get('att', 0) + r * dict(mix)[sname]
        else:
            g = group('exc', sname, prod); rates[g] = rates.get(g, 0) + r * dict(mix)[sname]; power[g] = power.get(g, 0) + r * dict(mix)[sname] * th * ng
            if g == 'diss_O2': rates['diss_O2'] = rates.get('diss_O2', 0) + 0  # kept under its group
    izth = min(th for kind, sname, prod, th, r in K.rates(f) if kind == 'iz')
    power['iz'] = rates.get('iz', 0) * ng * izth
    return dict(EN=EN, u=u, Te=2 * u / 3, rates=rates, power=power, f0=np.real(f[K.Nh, 0]).copy(), uc=K.uc, ng=ng)
for lam in (0.02, 0.1 / 2.405):
    for g, mix in GASES.items():
        key = (g, p, round(lam, 5))
        if key in res: continue
        t = time.time(); K = MixtureKinetics([(SP[s], x) for s, x in mix], umax=60., nu=300, L=1, Nh=1, ng=ng, Tg=300., growth=growth)
        ion, M, K0 = ion_mobility(mix); mu_i = K0 * 1e-4 * 2.6868e25 / ng
        def G(lnE):
            r = point(K, mix, float(np.exp(lnE))); loss = mu_i * r['Te'] / lam ** 2
            return ((r['rates'].get('iz', 0) - r['rates'].get('att', 0)) * ng - loss) / loss, r
        a = np.log(60.0); ga, ra = G(a); step = np.log(0.8) if ga > 0 else np.log(1.25); b, gb, rb = a, ga, ra
        for _ in range(15):
            a, ga, ra = b, gb, rb; b = a + step; gb, rb = G(b); step *= 1.4
            if np.sign(gb) != np.sign(ga): break
        for _ in range(20):
            c = (a * gb - b * ga) / (gb - ga); gc, rc = G(c)
            if abs(gc) < 1e-3 or abs(c - b) < 2e-4: break
            if gc * gb < 0: a, ga = b, gb
            else: ga = ga / 2
            b, gb = c, gc
        r = rc; r['Nh'] = 1; r['mu_i'] = mu_i; r['ion'] = ion; r['growth'] = growth; res[key] = r; pickle.dump(res, open(OUT, 'wb'))
        Pt = sum(r['power'].values()); net = r['rates'].get('iz', 0) - r['rates'].get('att', 0)
        print(f"{g:12s} Lambda {100*lam:.2f} cm: E/N {r['EN']:.1f} Td rms, Te {r['Te']:.2f} eV, cost {Pt/(net*ng):.0f} eV, frac {', '.join(f'{k} {100*v/Pt:.0f}' for k,v in sorted(r['power'].items(), key=lambda t:-t[1]) if v/Pt>0.01)} ({time.time()-t:.0f} s)", flush=True)
