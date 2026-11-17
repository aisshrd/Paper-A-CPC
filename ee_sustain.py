"""Working (self-sustaining) point with e-e collisions: E_rms/N where (k_iz - k_att) N = K_i T_e / Lambda^2 (same condition and
constants as mix_figs.sustain), found by bracketing + Illinois iteration in ln(E/N), started at the point without e-e.
13.56 MHz: harmonic balance; 1.76 MHz: quasi-static.  Output: results/ee_sust.pkl (merged)
usage: python3 ee_sustain.py nproc hb13|hb176|qs176 [gases]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time, warnings, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import scipy.constants as co
from multiprocessing import Pool
from ekin_ee import MixtureKineticsEE
from ee_scan import grouped, NH, TH, WQ
from mix_scan import GASES, PRESS, SP, TG
import mix_figs as MF
NE = [float(x) for x in os.environ['EE_NE'].split(',')] if os.environ.get('EE_NE') else [1e16, 1e17, 1e18, 1e19]
OUT = os.environ.get('EE_OUT', 'results/ee_sust.pkl')

def point_hb(K, EN, comp, fMHz=13.56):
    ng = K.ng; w = 2 * np.pi * fMHz * 1e6; th = np.linspace(0, 2 * np.pi, 97)[:-1]
    f = K.solve(np.sqrt(2) * EN * 1e-21 * ng, w, tol=1e-5, maxit=150)
    Rg, Pg = grouped(K, f, comp)
    Riz_h = [sum(comp[sn] * r for kind, sn, prod, thr, r in K.rates(f, harm=q) if kind == 'iz') for q in range(NH + 1)]
    Rt = np.real(Riz_h[0]) + sum(2 * np.real(Riz_h[q] * np.exp(1j * q * th)) for q in range(1, NH + 1))
    u = K.mean_energy(f); u2 = abs(complex(np.sum(K.w * K.uc * f[NH + 2, 0])))
    return dict(EN=EN, u=u, mod_u=2 * u2 / u, rates=Rg, power=Pg, Riz_peak=float(Rt.max() / max(Rt.mean(), 1e-300)),
                f0=np.real(f[NH, 0]).copy(), neg=float(np.min(np.real(f[NH, 0]))), lnL=K.lnL, it=K.it, res=getattr(K, 'res', 0.0),
                ee_cons=K.ee_power(f) / max(sum(Pg.values()), 1e-300))

def point_qs(K, EN, comp, tol=1e-5, mAA=6, maxit=60):
    """quasi-static: f(t) = f_DC(E(t)) with ONE e-e operator built from the period-averaged f0; the map
    x = w*<f0> -> G(x) is solved by Anderson mixing (same reason as in MixtureKineticsEE.solve)."""
    ng = K.ng
    Es = [max(np.sqrt(2) * EN * 1e-21 * ng * np.cos(t), 1e-3 * EN * 1e-21 * ng) for t in TH]
    if K.ne <= 0: maxit = 0; fs = [K.solve(E) for E in Es]; f0 = sum(wg * np.real(f[0, 0]) for wg, f in zip(WQ, fs)); res = 0.0; outer = 0
    else:
        if K._A is None: K.solve(EN * 1e-21 * ng, tol=1e-5)
        x = np.diff(K._A[0]); X, Gs = [], []
    for outer in range(maxit):
        K._A = None; K.set_ee(np.maximum(x, 0) / K.w, relax=1.0)
        fs = [K.solve(E, freeze_ee=True, maxit=40, tol=1e-5) for E in Es]
        f0 = sum(wg * np.real(f[0, 0]) for wg, f in zip(WQ, fs)); gx = K.w * f0
        res = float(np.sum(np.abs(gx - x)))
        if res < tol: break
        X.append(x.copy()); Gs.append(gx.copy()); X, Gs = X[-mAA:], Gs[-mAA:]
        if len(X) == 1: x = gx
        else:
            R = np.array([g_ - x_ for x_, g_ in zip(X, Gs)]); dR = (R[1:] - R[:-1]).T; dG = (np.array(Gs[1:]) - np.array(Gs[:-1])).T
            x = Gs[-1] - dG @ np.linalg.lstsq(dR, R[-1], rcond=None)[0]; x = np.maximum(x, 0); x /= x.sum()
    u = float(np.sum(K.w * K.uc * f0))
    Rg = {}; Pg = {}; Riz = []
    for wg, f in zip(WQ, fs):
        r_, p_ = grouped(K, f, comp)
        for k, v in r_.items(): Rg[k] = Rg.get(k, 0.0) + wg * v
        for k, v in p_.items(): Pg[k] = Pg.get(k, 0.0) + wg * v
        Riz.append(r_.get('iz', 0.0))
    Riz = np.array(Riz)
    return dict(EN=EN, u=u, rates=Rg, power=Pg, Riz_peak=float(Riz.max() / max(np.sum(WQ * Riz), 1e-300)), f0=f0,
                neg=float(f0.min()), lnL=K.lnL, it=outer, res=res)

def run(args):
    gname, p, fMHz, ne, meth = args
    comp = GASES[gname]; ng = p / (co.k * TG); Ki = MF.Kion(comp, ng)
    K = MixtureKineticsEE([(SP[s], x) for s, x in comp.items()], ne=ne, umax=60., nu=300, L=1,
                          Nh=(NH if meth == 'hb' else 0), ng=ng, Tg=TG)
    pt = (lambda K_, E_, c_: point_hb(K_, E_, c_, fMHz)) if meth == 'hb' else point_qs
    s0 = MF.sustain(gname, p, fMHz); EN0 = s0['EN'] if s0 else 50.0
    hist = []
    def g(lnE):
        r = pt(K, float(np.exp(lnE)), comp); hist.append(r)
        loss = Ki * (2 / 3 * r['u']) / MF.LAM ** 2
        return ((r['rates'].get('iz', 0) - r['rates'].get('att', 0)) * ng - loss) / loss, r
    a = np.log(EN0); ga, ra = g(a)
    step = np.log(0.8) if ga > 0 else np.log(1.2)
    b, gb, rb = a, ga, ra
    for _ in range(12):                              # expanding steps until the sign changes
        a, ga, ra = b, gb, rb; b = a + step; gb, rb = g(b); step *= 1.5
        if np.sign(gb) != np.sign(ga): break
    for _ in range(14):                              # Illinois (regula falsi), bracket [a, b], ga * gb < 0
        c = (a * gb - b * ga) / (gb - ga); gc, rc = g(c)
        if abs(gc) < 2e-3 or abs(c - b) < 1e-3: break
        if gc * gb < 0: a, ga = b, gb
        else: ga = ga / 2
        b, gb = c, gc
    return (gname, p, fMHz, ne, meth), dict(root=rc, g=float(gc), hist=hist, EN0=EN0, uc=K.uc, ng=ng)

if __name__ == '__main__':
    nproc = int(sys.argv[1]); mode = sys.argv[2]
    gases = sys.argv[3].split(',') if len(sys.argv) > 3 else list(GASES)
    # mode: hb13 | hb176 | qs176
    fr = 13.56 if mode == 'hb13' else 1.76; meth = 'qs' if mode == 'qs176' else 'hb'
    jobs = [(gn, p, fr, ne, meth) for gn, p, ne in itertools.product(gases, PRESS, NE)]
    jobs.sort(key=lambda j: -j[3]); t0 = time.time()
    with Pool(nproc) as pool:
        for key, val in pool.imap_unordered(run, jobs):
            res = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
            res[key] = val; pickle.dump(res, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
            r = val['root']
            print(key, f'{time.time()-t0:.0f}s', f"EN {val['EN0']:.1f}->{r['EN']:.1f} Te {2*r['u']/3:.2f} g {val['g']:.1e} n {len(val['hist'])}",
                  f"neg {r['neg']:.0e} it {r['it']}", flush=True)
