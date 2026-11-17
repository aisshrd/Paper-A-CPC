"""Self-consistent near-sample layer in an RF plasma (sample in an RF-I plasma jet), planar, argon.

  electrons : kinetic, ekin_slab_te.SlabKineticsTE in driven mode (isotropic Maxwellian T_j, density n_D entering at x = D),
              absorbing sample at x = 0, harmonic balance |n| <= Nh, ionisation with both electrons
  ions      : cold fluid in the period-averaged field, charge-exchange drag (constant mean free path), ionisation source and
              ionisation drag; drift equilibrium at x = D
  potential : Poisson for the static part and for every RF harmonic, with a Boltzmann predictor (Newton step for a Boltzmann
              response, as in Paper 4):  eps0 phi0'' = -e n_i + e n_e^k exp((phi0-phi0^k)/T_e),
                                         eps0 phi_q'' - e n_e0/T_e phi_q = e (n_eq^k - n_e0/T_e phi_q^k)
              BC: phi(D,t) = 0 (plasma), phi(0,t) = -V_dc + V_rf cos(w t) (sample)
  sample    : floating: <Gamma_e> = Gamma_i at x = 0 (zero period-averaged current) -> V_dc
usage: python3 slab_sc.py p nD Vrf [Nh n_e m iters tag]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, sys, os, pickle, time
import scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.constants as co
os.environ.setdefault('PYPARDISO_MKL_RT', '/usr/local/lib/libmkl_rt.so.3')
from ekin import load_argon
from ekin_slab_te import SlabKineticsTE
e, me, eps0 = co.e, co.m_e, co.epsilon_0
M_AR = 39.948 * 1.66053907e-27; SIG_CX = 5.5e-19
XS = load_argon('data/Ar_LXCat.txt')

def face_interp(xc, xe, vc, left, right):
    """face values: interior by linear interpolation between centres, boundary values given"""
    vf = np.empty(len(xe), dtype=np.result_type(vc, complex) if np.iscomplexobj(vc) else float)
    vf[0], vf[-1] = left, right
    vf[1:-1] = vc[:-1] + (vc[1:] - vc[:-1]) * (xe[1:-1] - xc[:-1]) / (xc[1:] - xc[:-1])
    return vf

def laplacian(xc, xe):
    """eps0-free Laplacian on cell centres with Dirichlet face values; returns (L, bl, br): L phi + bl*phi(0) + br*phi(D)"""
    m = len(xc); dx = np.diff(xe)
    hl = np.empty(m); hr = np.empty(m)
    hl[0] = xc[0] - xe[0]; hl[1:] = xc[1:] - xc[:-1]
    hr[-1] = xe[-1] - xc[-1]; hr[:-1] = xc[1:] - xc[:-1]
    main = -(1 / hl + 1 / hr) / dx; lo = (1 / hl[1:]) / dx[1:]; up = (1 / hr[:-1]) / dx[:-1]
    L = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
    bl = np.zeros(m); br = np.zeros(m); bl[0] = 1 / hl[0] / dx[0]; br[-1] = 1 / hr[-1] / dx[-1]
    return L, bl, br

class SelfConsistentLayer:
    def __init__(self, p, nD, Vrf, Tj=3.0, D=6e-3, Nh=2, L=3, n_e=80, m=48, umax=45., cluster=2.0, w=2 * np.pi * 13.56e6):
        self.p, self.nD, self.Vrf, self.Tj, self.D, self.Nh, self.L, self.n_e, self.m, self.umax, self.cluster, self.w = p, nD, Vrf, Tj, D, Nh, L, n_e, m, umax, cluster, w
        self.ng = p / (co.k * 300.0); self.lam = 1 / (self.ng * SIG_CX)
        s = np.linspace(0, 1, m + 1); self.xe = D * s ** cluster; self.xc = 0.5 * (self.xe[1:] + self.xe[:-1]); self.dx = np.diff(self.xe)
        self.Lap, self.bl, self.br = laplacian(self.xc, self.xe)
        # initial guess: floating sheath of a Maxwellian, parabolic over ~5 Debye lengths
        lD = np.sqrt(eps0 * Tj / (e * nD)); self.Vdc = 4.7 * Tj + 1.0
        s0 = min(8 * lD, 0.5 * D)
        self.phi = -self.Vdc * np.maximum(0, 1 - self.xc / s0) ** 2
        self.phq = {q: np.zeros(m, complex) for q in range(1, Nh + 1)}
        self.hist = []

    def fields(self):
        phf = face_interp(self.xc, self.xe, self.phi, -self.Vdc, 0.0)
        Eq = {}
        for q, ph in self.phq.items():
            bc0 = 0.5 * self.Vrf if q == 1 else 0.0
            pf = face_interp(self.xc, self.xe, ph, bc0, 0.0)
            Eq[q] = -(pf[1:] - pf[:-1]) / self.dx
        return phf, Eq

    def kinetic(self):
        phf, Eq = self.fields(); xe = self.xe
        phi0 = lambda x: float(np.interp(x, xe, phf))
        S = SlabKineticsTE(XS, D=self.D, ng=self.ng, omega=self.w, phi0=phi0, Eq=Eq, L=self.L, Nh=self.Nh, n_e=self.n_e,
                           umax=self.umax, xgrid=xe, absorb=True, iz_mult=2, inflow=dict(T=self.Tj, n=self.nD))
        f = S.solve_driven(limiter_iters=int(os.environ.get('LIMITER', 1)))
        out = dict(ne={n: S.density(f, n) for n in range(self.Nh + 1)}, en0=np.real(S.energy_density(f, 0)),
                   iz=np.real(S.rate(f, S.nuiz, 0)), em={n: S.rate(f, S.nu4p, n) for n in range(self.Nh + 1)},
                   flux={n: S.flux_cells(f, n) for n in range(self.Nh + 1)},
                   surf={n: S.surface_flux(f, n) for n in range(self.Nh + 1)}, f0=np.real(f[self.Nh, 0]), w=S.w, ukin=S.ukin, ec=S.ec,
                   negmass=float(-np.sum(np.minimum(np.real(f[self.Nh, 0]), 0) * S.w * S.dx[None, :]) / np.sum(np.abs(np.real(f[self.Nh, 0])) * S.w * S.dx[None, :])))
        self.S = S
        return out

    def ions(self, K):
        """march from x = D to the sample; returns n_i at centres, flux at faces (positive towards the sample), w at faces"""
        m = self.m; phf = face_interp(self.xc, self.xe, self.phi, -self.Vdc, 0.0); R = np.maximum(K['iz'], 0.0)
        ne0 = np.maximum(np.real(K['ne'][0]), 1.0); Te = np.clip(2 / 3 * K['en0'] / ne0, 0.3, 20.0)
        uB = np.sqrt(e * Te[-1] / M_AR)
        a_last = -(e / M_AR) * (phf[-2] - phf[-1]) / self.dx[-1]     # acceleration towards the sample in the last cell
        G = ne0[-1] * max(np.sqrt(max(a_last, 0.0) * self.lam), 0.05 * uB)    # drift equilibrium at D (w^2 = a lambda)
        y = (G / ne0[-1]) ** 2
        ni = np.zeros(m); Gf = np.zeros(m + 1); wf = np.zeros(m + 1); Gf[m] = G; wf[m] = np.sqrt(y)
        for j in range(m - 1, -1, -1):
            a = -(e / M_AR) * (phf[j] - phf[j + 1]) / self.dx[j]
            Gn = G + R[j] * self.dx[j]
            kap = 1 / self.lam + R[j] / max(0.5 * (G + Gn), 1e-30)
            yeq = a / kap
            yh = yeq + (y - yeq) * np.exp(-kap * self.dx[j])          # half cell (2 kap ds / 2)
            y2 = yeq + (y - yeq) * np.exp(-2 * kap * self.dx[j])
            yh = max(yh, (0.01 * uB) ** 2); y = max(y2, (0.01 * uB) ** 2)
            ni[j] = 0.5 * (G + Gn) / np.sqrt(yh); G = Gn; Gf[j] = G; wf[j] = np.sqrt(y)
        return ni, Gf, wf, Te

    def poisson(self, K, ni, Te, alpha=0.6, alpha_rf=0.6):
        ne0 = np.maximum(np.real(K['ne'][0]), 0.0)
        phk = self.phi.copy(); phi = phk.copy()
        for it in range(30):          # Newton for eps0 L phi + bc = -e ni + e ne0 exp((phi-phk)/Te)
            ex = np.exp(np.clip((phi - phk) / Te, -30, 30))
            Fv = eps0 * (self.Lap @ phi + self.bl * (-self.Vdc) + self.br * 0.0) + e * ni - e * ne0 * ex
            J = eps0 * self.Lap - sp.diags(e * ne0 * ex / Te)
            d = spla.spsolve(J.tocsc(), -Fv); d = np.clip(d, -2 * Te, 2 * Te); phi += d
            if np.max(np.abs(d)) < 1e-6: break
        dphi = phi - phk; self.phi = phk + alpha * dphi
        drf = 0.0
        for q, phq in self.phq.items():
            neq = K['ne'][q]; bc0 = 0.5 * self.Vrf if q == 1 else 0.0
            A = eps0 * self.Lap - sp.diags(e * ne0 / Te)
            rhs = e * (neq - ne0 / Te * phq) - eps0 * self.bl * bc0
            new = spla.spsolve(A.tocsc().astype(complex), rhs.astype(complex))
            drf = max(drf, np.max(np.abs(new - phq))); self.phq[q] = phq + alpha_rf * (new - phq)
        return float(np.max(np.abs(dphi))), float(drf)

    def step(self, beta=0.5):
        t0 = time.time(); K = self.kinetic(); ni, Gf, wf, Te = self.ions(K)
        Ge = float(np.real(K['surf'][0][0])); Gi = float(Gf[0])
        # floating condition: <Gamma_e>(0) = Gamma_i(0); Boltzmann-type Newton step on V_dc
        jsh = np.argmin(np.abs(self.xc - 0.5 * self.D)); Ts = float(Te[jsh])
        dV = beta * Ts * np.log(max(Ge, 1e-30) / max(Gi, 1e-30)); dV = float(np.clip(dV, -3, 3))
        self.Vdc += dV
        dphi, drf = self.poisson(K, ni, Te)
        rec = dict(Vdc=self.Vdc, dV=dV, dphi=dphi, drf=drf, Ge=Ge, Gi=Gi, t=time.time() - t0, negmass=K['negmass'])
        self.hist.append(rec); self.K, self.ni, self.Gf, self.wf, self.Te = K, ni, Gf, wf, Te
        return rec

    def run(self, iters=40, tol=0.02, verbose=True):
        for it in range(iters):
            r = self.step()
            if verbose:
                print(f'  it {it:2d}: Vdc={r["Vdc"]:7.3f} (dV {r["dV"]:+.3f})  |dphi|={r["dphi"]:.3f}  |dphi_rf|={r["drf"]:.3f}  '
                      f'Ge/Gi={r["Ge"]/max(r["Gi"],1e-30):.3f}  Gi={r["Gi"]:.3e}  neg={r["negmass"]:.1e}  ({r["t"]:.0f}s)', flush=True)
            if it > 4 and abs(r['dV']) < tol and r['dphi'] < tol and r['drf'] < tol: break
        return self

    def summary(self):
        K, ni, Gf, wf, Te = self.K, self.ni, self.Gf, self.wf, self.Te
        ne0 = np.real(K['ne'][0]); xc = self.xc
        Ei = 0.5 * M_AR * wf[0] ** 2 / e
        # sheath edge: where (ni-ne)/ni first exceeds 0.1 going from the plasma; RF oscillation region: where |n_e1|/n_e0 > 0.3 & n_e0 > 0.01 nD
        rho = (ni - ne0) / np.maximum(ni, 1.0)
        idx = np.where(rho > 0.1)[0]; s_edge = float(xc[idx.max()]) if len(idx) else 0.0
        mod = np.abs(K['ne'][1]) / np.maximum(ne0, 1e-30)
        osc = np.where((mod > 0.3) & (ne0 > 1e-3 * self.nD))[0]; s_osc = float(xc[osc.max()]) if len(osc) else 0.0
        Ge = np.real(K['surf'][0][0]); Eflux = np.real(K['surf'][0][1])
        return dict(p=self.p, nD=self.nD, Vrf=self.Vrf, Vdc=self.Vdc, Ji=e * Gf[0], Ei=Ei, Ge=Ge, Gi=Gf[0], ee_surf=Eflux / max(Ge, 1e-30),
                    Te_D=float(Te[-1]), s_edge=s_edge, s_osc=s_osc, lam_i=self.lam, lD=float(np.sqrt(eps0 * Te[-1] / (e * self.nD))),
                    iz_total=float(np.sum(K['iz'] * self.dx)), Gi_D=float(Gf[-1]))

    def save(self, fn):
        d = dict(cfg=dict(p=self.p, nD=self.nD, Vrf=self.Vrf, Tj=self.Tj, D=self.D, Nh=self.Nh, L=self.L, n_e=self.n_e, m=self.m),
                 xc=self.xc, xe=self.xe, dx=self.dx, phi=self.phi, phq=self.phq, Vdc=self.Vdc, K=self.K, ni=self.ni, Gf=self.Gf, wf=self.wf,
                 Te=self.Te, hist=self.hist, summary=self.summary())
        pickle.dump(d, open(fn, 'wb'))

if __name__ == '__main__':
    a = sys.argv
    p, nD, Vrf = float(a[1]), float(a[2]), float(a[3])
    Nh = int(a[4]) if len(a) > 4 else 2; ne = int(a[5]) if len(a) > 5 else 80; m = int(a[6]) if len(a) > 6 else 48
    iters = int(a[7]) if len(a) > 7 else 40; tag = a[8] if len(a) > 8 else ''
    L = int(os.environ.get('SC_L', 3)); D = float(os.environ.get('SC_D_MM', 6)) * 1e-3; Tj = float(os.environ.get('SC_TJ', 3.0))
    t0 = time.time()
    X = SelfConsistentLayer(p, nD, Vrf, Tj=Tj, D=D, Nh=Nh, L=L, n_e=ne, m=m).run(iters)
    fn = f'results/sc_p{p:g}_n{nD:.0e}_V{Vrf:g}_N{Nh}_e{ne}_m{m}{tag}.pkl'; X.save(fn)
    s = X.summary(); print('SUMMARY', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, f'time {time.time()-t0:.0f}s')
