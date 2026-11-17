"""Static axial B0 in the self-consistent fields of the unmagnetised ICP (13.3 Pa, n0 = 1e17 m^-3, B_RF = 0.7 G): kinetic response
with the RF radial field closure (j_r = 0).  Output: results/mag_prescribed.pkl (merged).  usage: python3 mag_prescribed.py nproc B0list_G [noEr]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
from ekin import load_argon
from ekin_cyl_mag import CylKineticsMag
xs = load_argon('data/Ar_LXCat.txt')
REF = 'results/discharge_te_B0.7_n1e+17_L1_m16_o2.pkl'
OUT = os.environ.get('MAG_OUT', 'results/mag_prescribed.pkl')
ng = 3.21e21; R = 0.1; om = 2 * np.pi * 13.56e6; N0 = 1e17

def run(args):
    BG, withEr, L = args
    d = pickle.load(open(REF, 'rb')); E, B, phi = d['E'], d['B'], d['phi']
    t = time.time()
    k = CylKineticsMag(xs, R, ng, om, phi - phi.min(), L=L, Nh=1, n_e=120, umax=45., m=16, cluster=2.0, order=2, B0=BG * 1e-4)
    f = k.solve_Er(E, B) if withEr else k.solve(E, B, limiter_iters=3)
    ne = k.density(f); Te = k.mean_energy(f) * 2 / 3; nz = k.nu_iz(f)
    jt = k.current_theta(f) * N0; jr = k.current_r(f) * N0
    P = 0.5 * np.real(jt * np.conj(E[2])) * 2            # time-averaged j.E with E(t) = 2 Re(E_1 e^{iwt}), j likewise
    if withEr: P = P + 0.5 * np.real(jr * np.conj(k.Er[2])) * 2
    j0t = k.current_theta(f, 0) * N0                      # DC azimuthal (E x B) current
    out = dict(B0=BG, withEr=withEr, L=L, X=k.X, r=k.r, wdisk=k.wdisk, ne=ne, Te=Te, nuiz=nz, P=P, jt=jt, jr=jr, j0t=j0t,
               Er=getattr(k, 'Er', None), Er_hist=getattr(k, 'Er_hist', None), E=E, phi=phi, F0=np.real(f[k.Nh, 0]), ec=k.ec,
               ukin=k.ukin, w=k.w, lam=complex(k.lam), t=time.time() - t)
    return (BG, withEr, L), out

if __name__ == '__main__':
    nproc = int(sys.argv[1]); Bs = [float(x) for x in sys.argv[2].split(',')]
    withEr = not (len(sys.argv) > 3 and sys.argv[3] == 'noEr'); L = int(os.environ.get('MAG_L', '1'))
    jobs = [(b, withEr, L) for b in Bs]; t0 = time.time()
    with Pool(nproc) as pool:
        for key, val in pool.imap_unordered(run, jobs):
            res = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
            res[key] = val; pickle.dump(res, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
            ne, Te = val['ne'], val['Te']; nb = np.sum(val['wdisk'] * ne * val['nuiz']) / np.sum(val['wdisk'] * ne)
            print(key, f"{time.time()-t0:.0f}s Te wall/max/axis {Te[0]:.2f}/{Te.max():.2f}/{Te[-1]:.2f} ne ax/wall {ne[-1]/ne[0]:.1f} <nuiz> {nb:.3e} "
                  f"P {np.sum(val['wdisk']*val['P'])*np.pi*R**2:.3e} W/m Er_hist {val['Er_hist'][-1] if val['Er_hist'] else 0:.2e}", flush=True)
