"""Electron cyclotron resonance in the ICP column: static B0 scan at low pressure in prescribed ICP fields (the self-consistent
E_theta, B_z, phi of the 13.3 Pa / 1e17 m^-3 discharge, the same at every pressure).  Two limits for the RF radial current:
  Er = 0   ('open' / end-shorted: j_r closes through conducting end plates, no RF radial field)
  jr = 0   (infinite insulated column: E_r from j_r = 0, as in ekin_cyl_mag.solve_Er)
Output: results/lowp_scan.pkl, keys (p, B0_G, mode, L).  usage: python3 lowp_scan.py nproc p_list B0_list mode [L]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1')
import scipy.constants as co
from multiprocessing import Pool
from ekin import load_argon
from ekin_cyl_mag import CylKineticsMag
xs = load_argon('data/Ar_LXCat.txt')
REF = os.environ.get('LOWP_REF', 'results/discharge_te_B0.7_n1e+17_L1_m16_o2.pkl'); OUT = os.environ.get('LOWP_OUT', 'results/lowp_scan.pkl')
R = 0.1; om = 2 * np.pi * 13.56e6

def run(args):
    p, BG, mode, L = args
    d = pickle.load(open(REF, 'rb')); E, B, phi = d['E'], d['B'], d['phi']; ng = p / (co.k * 300.)
    t = time.time()
    k = CylKineticsMag(xs, R, ng, om, phi - phi.min(), L=L, Nh=1, n_e=int(os.environ.get('LOWP_NE', '120')), umax=45., m=16, cluster=2.0, order=2, B0=BG * 1e-4)
    f = (k.solve_Er(E, B, maxit=8) if mode == 'jr0' else k.solve_Er_kernel(E, B, verbose=True)) if (mode in ('jr0', 'jr0k') and BG > 0) else k.solve(E, B, limiter_iters=2)
    ne = k.density(f); Te = k.mean_energy(f) * 2 / 3; nz = k.nu_iz(f)
    jt = k.current_theta(f); jr = k.current_r(f); Er = getattr(k, 'Er', None)
    P = 2 * np.real(jt * np.conj(E[2]))                                   # <j.E> per unit n0 (W/m^3 per m^-3)
    if Er is not None: P = P + 2 * np.real(jr * np.conj(Er[2]))
    return (p, BG, mode, L), dict(p=p, B0=BG, mode=mode, L=L, X=k.X, wdisk=k.wdisk, ne=ne, Te=Te, nuiz=nz, P=P, jt=jt, jr=jr, Er=Er,
                                  Er_hist=getattr(k, 'Er_hist', None), F0=np.real(f[k.Nh, 0]), ukin=k.ukin, w=k.w, t=time.time() - t)

if __name__ == '__main__':
    nproc = int(sys.argv[1]); ps = [float(x) for x in sys.argv[2].split(',')]; Bs = [float(x) for x in sys.argv[3].split(',')]
    mode = sys.argv[4]; L = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    jobs = list(itertools.product(ps, Bs, [mode], [L])); t0 = time.time()
    with Pool(nproc) as pool:
        for key, v in pool.imap_unordered(run, jobs):
            res = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
            res[key] = v; pickle.dump(res, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
            ne = v['ne']; w = v['wdisk']
            print(key, f"{time.time()-t0:.0f}s Te wall/max/axis {v['Te'][0]:.2f}/{v['Te'].max():.2f}/{v['Te'][-1]:.2f} <nuiz> {np.sum(w*ne*v['nuiz'])/np.sum(w*ne):.3e} "
                  f"P {np.sum(w*v['P']):.3e} |jr|/|jt| {np.max(np.abs(v['jr']))/np.max(np.abs(v['jt'])):.3f}", flush=True)
