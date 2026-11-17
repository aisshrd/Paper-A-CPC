"""ekin_cyl_mag.py -- ICP with an additional STATIC axial magnetic field B0 in the total-energy cylinder solver (ekin_cyl_te).

What changes compared with the unmagnetised discharge:
  * B0 enters the zeroth harmonic of B_z: the existing rotation operator (v x B).grad_v couples every harmonic to itself;
  * with B0 the azimuthal RF field E_theta drives, through the Hall coupling, an RF RADIAL electron current j_r.  In an infinite
    axisymmetric cylinder div j = 0 forces r j_r = const = 0 (ions do not respond at 13.56 MHz), so an RF radial field E_r(r)
    builds up that cancels j_r.  For a cold local plasma it restores sigma_eff = n e^2 / (m (nu + i w)) exactly; omitting it
    would suppress the inductive heating artificially.  E_r (harmonics n = +-1) is found here by Newton iterations on j_r = 0,
    with a diagonal (local) Jacobian obtained from one perturbed solve.
The static radial (ambipolar) field is part of the total-energy variable as before.
"""
import numpy as np, scipy.sparse as sp, scipy.constants as co
from ekin_cyl_te import CylKineticsTE
e, me = co.e, co.m_e

class CylKineticsMag(CylKineticsTE):
    def __init__(self, *a, B0=0.0, **kw):
        super().__init__(*a, **kw); self.B0 = float(B0)

    def _Er_matrix(self, Er):
        """force (-e/m) E_r r_hat . grad_v f for harmonics q = +-1 of E_r, in the block layout of kinetic_matrix"""
        H, Nh, nlm, ne, m, ang = self.H, self.Nh, self.nlm, self.n_e, self.m, self.ang
        if not hasattr(self, '_cellblocks'):
            def cellblock(mats):
                rows = []; cols = []; vals = []
                for k, Mk in enumerate(mats):
                    Mk = Mk.tocoo(); rows.append(Mk.row * m + k); cols.append(Mk.col * m + k); vals.append(Mk.data)
                return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ne * m, ne * m))
            self._cellblocks = (cellblock(self.Dv), cellblock(self.D0), cellblock(self.Dvc))
        Dvb, D0b, Dvcb = self._cellblocks
        vinv = np.where(self.allowed, 1 / np.maximum(self.v, 1e-30), 0.0); Vinv = sp.diags(vinv.ravel())
        N = H * nlm * ne * m; rows = []; cols = []; vals = []
        def put(n1, a1, n2, a2, M):
            M = M.tocoo(); o1 = ((n1 + Nh) * nlm + a1) * ne * m; o2 = ((n2 + Nh) * nlm + a2) * ne * m
            rows.append(M.row + o1); cols.append(M.col + o2); vals.append(M.data)
        for q in range(-Nh, Nh + 1):
            Eq = Er[q + Nh]
            if np.max(np.abs(Eq)) < 1e-30: continue
            Ed = sp.diags(np.tile(-(e / me) * Eq, ne))
            for n in range(-Nh, Nh + 1):
                mm = n - q
                if abs(mm) > Nh: continue
                for a in range(nlm):
                    for b in range(nlm):
                        fv, fa = self.Fr_v[a, b], self.Fr_a[a, b]
                        if a == 0 and abs(fv) > 1e-12: put(n, a, mm, b, fv * (D0b @ Ed))
                        else:
                            Dsel = Dvb if ang.lm[b][0] == 0 else Dvcb
                            if abs(fv) > 1e-12: put(n, a, mm, b, fv * (Dsel @ Ed))
                            if abs(fa) > 1e-12: put(n, a, mm, b, fa * (Vinv @ Ed))
        if not rows: return None
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        forb = np.where(~self.allowed.ravel())[0]
        if len(forb):
            allrows = np.concatenate([((n + Nh) * nlm + a) * ne * m + forb for n in range(-Nh, Nh + 1) for a in range(nlm)])
            keep = np.ones(N); keep[allrows] = 0; M = sp.diags(keep) @ M
        return M

    def kinetic_matrix(self, Eth, Bz):
        Bs = np.array(Bz, complex).copy(); Bs[self.Nh] = Bs[self.Nh] + self.B0
        A = super().kinetic_matrix(Eth, Bs)
        Er = getattr(self, 'Er', None)
        if Er is not None:
            M = self._Er_matrix(Er)
            if M is not None: A = A + M
        return A

    # ------------------------------------------------------------------ currents (per unit n0, same normalisation as DischargeTE)
    def _cur(self, f, comp, n):
        vh = self.ang.unit_vectors()[0][comp]; c = np.einsum('aij,ij,ij->a', self.ang.Y, vh, self.ang.W)
        return -e * np.einsum('a,ik,aik->k', c, self.w * self.v, f[self.Nh + n]) / np.sqrt(4 * np.pi)
    def current_theta(self, f, n=1): return self._cur(f, 1, n)
    def current_r(self, f, n=1):     return self._cur(f, 0, n)

    def solve_Er(self, Eth, Bz, Er0=None, tol=0.02, maxit=6, limiter_iters=2, verbose=False):
        """find E_r^{(+-1)}(r) with j_r^{(1)} = 0 (relative to |j_theta|); returns f.  Diagonal Newton, Jacobian from one perturbed solve."""
        H, Nh, m = self.H, self.Nh, self.m
        self.Er = np.zeros((H, m), complex) if Er0 is None else Er0.copy()
        f = self.solve(Eth, Bz, limiter_iters=limiter_iters)
        if self.B0 == 0.0: self.Er_hist = [0.0]; return f
        jr = self.current_r(f); jt = self.current_theta(f); scale = np.max(np.abs(jt)); hist = [float(np.max(np.abs(jr)) / scale)]
        # local Jacobian dj_r/dE_r from a perturbation of 2 % of |E_theta| (complex, per radial cell)
        dE = 0.02 * np.max(np.abs(Eth[Nh + 1])) * np.ones(m)
        Ep = self.Er.copy(); Ep[Nh + 1] += dE; Ep[Nh - 1] += np.conj(dE); Er_save = self.Er; self.Er = Ep
        fp = self.solve(Eth, Bz, limiter_iters=0); self.Er = Er_save
        J = (self.current_r(fp) - self.current_r(self.solve(Eth, Bz, limiter_iters=0))) / dE
        for it in range(maxit):
            if hist[-1] < tol: break
            step = -jr / J
            self.Er[Nh + 1] += step; self.Er[Nh - 1] = np.conj(self.Er[Nh + 1])
            f = self.solve(Eth, Bz, limiter_iters=limiter_iters); jr = self.current_r(f); hist.append(float(np.max(np.abs(jr)) / scale))
            if verbose: print(f'    Er it{it}: max|j_r|/max|j_th| = {hist[-1]:.3e}', flush=True)
        self.Er_hist = hist
        return f

    def solve_Er_kernel(self, Eth, Bz, tol=0.02, maxit=10, limiter_iters=2, verbose=False, relax=0.5):
        """E_r from j_r = 0 with the exact first-harmonic response kernels at frozen f_0 (ekin_nlsigma.kernels):
        E_r = -K_rR^{-1} (K_rE E_theta + K_rB B_z); repeated with the updated f_0 until max|j_r|/max|j_theta| < tol."""
        from ekin_nlsigma import kernels
        H, Nh, m = self.H, self.Nh, self.m
        self.Er = np.zeros((H, m), complex); f = self.solve(Eth, Bz, limiter_iters=limiter_iters); hist = []
        for it in range(maxit):
            jr = self.current_r(f); jt = self.current_theta(f); hist.append(float(np.max(np.abs(jr)) / np.max(np.abs(jt))))
            if verbose: print(f'    Er(kernel) it{it}: max|j_r|/max|j_th| = {hist[-1]:.3e}', flush=True)
            if hist[-1] < tol or self.B0 == 0.0: break
            K = kernels(self, f)
            Er1 = -np.linalg.solve(K['KrR'], K['KrE'] @ Eth[Nh + 1] + K['KrB'] @ Bz[Nh + 1])
            Er1 = self.Er[Nh + 1] + (relax if it > 0 else 1.0) * (Er1 - self.Er[Nh + 1])
            self.Er = np.zeros((H, m), complex); self.Er[Nh + 1] = Er1; self.Er[Nh - 1] = np.conj(Er1)
            f = self.solve(Eth, Bz, limiter_iters=limiter_iters)
        self.Er_hist = hist
        return f
