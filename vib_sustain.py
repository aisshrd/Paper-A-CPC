"""Working point of the N2-containing gases versus the vibrational temperature T_v (superelastic collisions with N2(v), v <= 10).
13.56 MHz harmonic balance, same sustaining condition and root search as ee_sustain.py.  Output: results/vib_sust.pkl (merged)
usage: python3 vib_sustain.py nproc [ne]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time, warnings, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import scipy.constants as co
from multiprocessing import Pool
from ekin_vib import MixtureKineticsVib
from ee_sustain import point_hb
from mix_scan import GASES, PRESS, SP, TG
import mix_figs as MF
GV = ['N2', 'air', 'Ar+N2 90/10', 'Ar+N2 50/50']
TV = [300., 1000., 2000., 3000., 4000., 5000., 6000.]
OUT = os.environ.get('VIB_OUT', 'results/vib_sust.pkl')

def run(args):
    gname, p, Tv, ne = args
    comp = GASES[gname]; ng = p / (co.k * TG); Ki = MF.Kion(comp, ng)
    K = MixtureKineticsVib([(SP[s], x) for s, x in comp.items()], Tv=Tv, ne=ne, umax=60., nu=300, L=1, Nh=6, ng=ng, Tg=TG)
    s0 = MF.sustain(gname, p, 13.56); EN0 = s0['EN'] if s0 else 50.0
    hist = []
    def g(lnE):
        r = point_hb(K, float(np.exp(lnE)), comp); r['Pvib'] = K.vib_power_last; hist.append(r)
        loss = Ki * (2 / 3 * r['u']) / MF.LAM ** 2
        return ((r['rates'].get('iz', 0) - r['rates'].get('att', 0)) * ng - loss) / loss, r
    a = np.log(EN0); ga, ra = g(a)
    step = np.log(0.8) if ga > 0 else np.log(1.2)
    b, gb, rb = a, ga, ra
    for _ in range(14):
        a, ga, ra = b, gb, rb; b = a + step; gb, rb = g(b); step *= 1.5
        if np.sign(gb) != np.sign(ga): break
    c, gc, rc = b, gb, rb
    for _ in range(16):
        c = (a * gb - b * ga) / (gb - ga); gc, rc = g(c)
        if abs(gc) < 2e-3 or abs(c - b) < 1e-3: break
        if gc * gb < 0: a, ga = b, gb
        else: ga = ga / 2
        b, gb = c, gc
    return (gname, p, Tv, ne), dict(root=rc, g=float(gc), hist=hist, uc=K.uc, ng=ng, pop=K.pop)

# vib_power of the last solve, recorded through a thin wrapper of solve()
_solve = MixtureKineticsVib.solve
def _solve_rec(self, *a, **k):
    f = _solve(self, *a, **k); self.vib_power_last = self.vib_power(f); return f
MixtureKineticsVib.solve = _solve_rec

if __name__ == '__main__':
    nproc = int(sys.argv[1]); ne = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    jobs = [(gn, p, Tv, ne) for gn, p, Tv in itertools.product(GV, PRESS, TV)]
    jobs.sort(key=lambda j: -j[2]); t0 = time.time()
    with Pool(nproc) as pool:
        for key, val in pool.imap_unordered(run, jobs):
            res = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
            res[key] = val; pickle.dump(res, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
            r = val['root']
            print(key, f"{time.time()-t0:.0f}s EN {r['EN']:.1f} Te {2*r['u']/3:.3f} g {val['g']:.1e} n {len(val['hist'])} neg {r['neg']:.0e} Pvib {r['Pvib']:.3e}", flush=True)
