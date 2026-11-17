"""Self-consistent ICP with a static axial B0 (DischargeTE + CylKineticsMag): Picard loop electrons(E_theta, B_z, B0, phi, E_r) ->
sigma_theta -> Maxwell -> ambipolar ions -> phi, with the RF radial field E_r updated inside every electron step (by default from the exact first-harmonic response kernels,
CylKineticsMag.solve_Er_kernel / ekin_nlsigma; MAG_ER_KERNEL=0 selects the diagonal Newton iteration solve_Er, which converges only at small B0).  Started from the converged unmagnetised discharge.
env MAG_NG (gas density, default 13.3 Pa), MAG_ER0=1 (E_r = 0 limit instead of j_r = 0), MAG_TAG.
usage: python3 discharge_mag.py B0_gauss Bext_gauss [iters]   -> results/discharge_mag_B0{B0}_Bext{Bext}.pkl"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, time, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from ekin import load_argon
from discharge_te import DischargeTE
from ekin_cyl_mag import CylKineticsMag

class DischargeMag(DischargeTE):
    def __init__(self, *a, B0=0.0, **kw):
        super().__init__(*a, **kw); self.B0 = B0; self.Er = None; self.J = None; self.Er_res = []
    def electrons(self, E, B, phi):
        self.kin = CylKineticsMag(self.xs, self.R, self.ngas, self.omega, phi - phi.min(), L=self.L, Nh=1, n_e=self.n_e, umax=self.umax,
                                  m=self.m, cluster=self.cluster, wall_absorb=self.wall_absorb, order=self.order, B0=self.B0)
        k = self.kin; Nh = k.Nh
        k.Er = np.zeros((k.H, k.m), complex) if self.Er is None else self.Er.copy()
        def sol():
            if self.order == 2 and getattr(self, 'prev_f', None) is not None and not os.environ.get('MAG_FRESH'):
                k._theta = k.limiter_theta(self.prev_f); f = k._solve(E, B); k._theta = None; return f
            return k.solve(E, B, limiter_iters=2)
        f = sol()
        if self.B0 > 0 and not os.environ.get('MAG_ER0'):
            if os.environ.get('MAG_ER_KERNEL', '1') == '1':             # E_r from the exact response kernels at frozen f0 (ekin_nlsigma), damped
                from ekin_nlsigma import kernels
                for _ in range(int(os.environ.get('ER_STEPS', '1'))):
                    th = k.limiter_theta(f) if k.order == 2 else None
                    K = kernels(k, f, theta=th)
                    Er1 = -np.linalg.solve(K['KrR'], K['KrE'] @ E[Nh + 1] + K['KrB'] @ B[Nh + 1])
                    rel = float(os.environ.get('ER_RELAX', '0.5')) if self.Er is not None else 1.0
                    k.Er[Nh + 1] += rel * (Er1 - k.Er[Nh + 1]); k.Er[Nh - 1] = np.conj(k.Er[Nh + 1]); f = sol()
            else:
                if self.J is None:                                        # local Jacobian dj_r/dE_r, once
                    dE = 0.02 * np.max(np.abs(E[Nh + 1])) * np.ones(k.m); Er_s = k.Er.copy()
                    k.Er[Nh + 1] += dE; k.Er[Nh - 1] = np.conj(k.Er[Nh + 1]); fp = sol(); k.Er = Er_s
                    self.J = (k.current_r(fp) - k.current_r(f)) / dE
                for _ in range(int(os.environ.get('ER_STEPS', '2'))):
                    jr = k.current_r(f); k.Er[Nh + 1] += -jr / self.J; k.Er[Nh - 1] = np.conj(k.Er[Nh + 1]); f = sol()
            self.Er_res.append(float(np.max(np.abs(k.current_r(f))) / np.max(np.abs(k.current_theta(f)))))
            self.Er = k.Er.copy()
        self.prev_f = f; return f
    def run(self, *a, **kw):
        import discharge_te as DT
        orig = DT.CylKineticsTE; me_ = self
        def factory(*aa, **kk):                       # the final re-limited solve of DischargeTE.run must also carry B0 and E_r
            kobj = CylKineticsMag(*aa, B0=me_.B0, **kk); kobj.Er = me_.Er; return kobj
        DT.CylKineticsTE = factory
        try: return super().run(*a, **kw)
        finally: DT.CylKineticsTE = orig

if __name__ == '__main__':
    os.environ.setdefault('ION_K0', '1.53')   # argon: measured reduced mobility (as for all published runs); ION_K0= (empty) selects the crude estimate
    xs = load_argon('data/Ar_LXCat.txt'); ng = float(os.environ.get('MAG_NG', '3.21e21')); omega = 2 * np.pi * 13.56e6; R = 0.1; n0 = 1e17
    B0g = float(sys.argv[1]); Bg = float(sys.argv[2]); iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    Bext = np.zeros(3, complex); Bext[0] = Bext[2] = Bg * 1e-4 / 2
    ref = pickle.load(open(os.environ.get('MAG_START', 'results/discharge_te_B0.7_n1e+17_L1_m16_o2.pkl'), 'rb'))
    d = DischargeMag(xs, R, ng, omega, Bext, n0, L=1, n_e=int(os.environ.get('MAG_NE', '120')), m=int(os.environ.get('MAG_M', '16')), B0=B0g * 1e-4)
    if os.environ.get('MAG_ER_START'):
        prev = pickle.load(open(os.environ['MAG_ER_START'], 'rb')); d.Er = prev.get('Er')
    t = time.time()
    d.run(iters=iters, relax=float(os.environ.get('RELAX', '0.5')), phi_relax=float(os.environ.get('PHIRELAX', '0.35')),
          E0=ref['E'] * (Bg / ref['B_ext']), B0=ref['B'] * (Bg / ref['B_ext']), phi0=ref['phi'])
    print('total', time.time() - t, 's', 'Er residuals', np.round(d.Er_res, 4))
    k = d.kin; f = d.f
    pickle.dump(dict(B0=B0g, B_ext=Bg, ng=ng, Er0_limit=bool(os.environ.get('MAG_ER0')), X=k.X, E=d.E, B=d.B, phi=d.phi, ne=d.ne, ni=d.ni, Te=d.Te, nuiz=d.nuiz, hist=d.hist, balance=d.balance,
                     converged=d.converged, n_iter=d.n_iter, Er=d.Er, Er_res=d.Er_res, F0=np.real(f[1, 0]), ec=k.ec, ukin=k.ukin, w=k.w,
                     jt=k.current_theta(f) * n0, jr=k.current_r(f) * n0, j0t=k.current_theta(f, 0) * n0, wdisk=k.wdisk),
                open(f'results/discharge_mag_B0{B0g:g}_Bext{Bg:g}{os.environ.get("MAG_TAG", "")}.pkl', 'wb'))
