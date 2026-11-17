"""ekin_ee.py -- electron-electron collisions for the mixture solver (ekin_mix.MixtureKinetics).

Isotropic Rosenbluth (Fokker-Planck) e-e operator in the energy-flux form of BOLSIG+ (Hagelaar & Pitchford 2005, eqs. 37-40):
    flux  G = W f0 - D df0/du,   added to the elastic/CAR flux coefficients on the cell boundaries u_j:
    W_ee = -3 a n_e A1(u),   D_ee = 2 a n_e [A2(u) + u^{3/2} A3(u)],
    A1 = int_0^u s^{1/2} f0 ds,   A2 = int_0^u s^{3/2} f0 ds,   A3 = int_u^inf f0 ds,
    a  = e^2 GAMMA lnL / (24 pi eps0^2)   (energies in volts, GAMMA = sqrt(2e/m)),
    lnL = ln[ 12 pi (eps0 Te / e)^{3/2} / sqrt(n_e) ],  Te = (2/3)<u>.
The operator conserves particles and energy and its zero-flux state is the Maxwellian at Te = (2/3)<u>.
It is nonlinear in f0; we freeze A1..A3 at the PERIOD-AVERAGED f0 (zeroth harmonic) and iterate to self-consistency together with
the growth rate nu_g.  Freezing in time is justified because nu_ee << omega here (nu_ee ~ 1e4-1e6 s^-1 vs omega = 1.1e7-8.5e7 s^-1),
so e-e collisions see the period average.  The same frozen operator acts on every harmonic.
"""
import numpy as np, scipy.constants as co
from ekin_mix import MixtureKinetics
from ekin import GAMMA, e

EPS0 = co.epsilon_0
A_EE = e ** 2 * GAMMA / (24 * np.pi * EPS0 ** 2)      # times lnL -> a  (m^3 s^-1 V^{3/2})

def coulomb_log(Te, ne):
    return float(np.log(12 * np.pi * (EPS0 * Te / e) ** 1.5 / np.sqrt(ne)))

class MixtureKineticsEE(MixtureKinetics):
    def __init__(self, mix, ne=0.0, relax=0.5, **kw):
        self.ne = float(ne); self.relax = relax; self._A = None; self.lnL = None
        super().__init__(mix, **kw)
        self.Wb0, self.Db0 = self.Wb.copy(), self.Db.copy()

    # ---------------------------------------------------------------- e-e coefficients
    def ee_coeffs(self, f0):
        """A1, A2, A3 on the cell boundaries ub[0..nu] from cell-centre f0 (normalised: sum w f0 = 1)."""
        f0 = np.maximum(np.real(f0), 0.0)
        c1 = np.concatenate([[0.0], np.cumsum(self.w * f0)])
        c2 = np.concatenate([[0.0], np.cumsum(self.w * self.uc * f0)])
        c3 = np.concatenate([np.cumsum((self.du * f0)[::-1])[::-1], [0.0]])
        return c1, c2, c3

    def set_ee(self, f0, relax=None):
        """(re)build the l = 0 flux operator with e-e coefficients from f0 (under-relaxed in A1..A3)."""
        if self.ne <= 0: return
        A = np.array(self.ee_coeffs(f0)); r = self.relax if relax is None else relax
        self._A = A if self._A is None else (1 - r) * self._A + r * A
        A1, A2, A3 = self._A
        Te = (2.0 / 3.0) * A2[-1] / max(A1[-1], 1e-300)
        self.lnL = coulomb_log(Te, self.ne); a = A_EE * self.lnL
        self.Wee = -3 * a * self.ne * A1
        self.Dee = 2 * a * self.ne * (A2 + self.ub ** 1.5 * A3)
        self.Wb, self.Db = self.Wb0 + self.Wee, self.Db0 + self.Dee
        self.Cel0 = self._flux_operator(self.Wb, self.Db)

    def clear_ee(self):
        self._A = None; self.Wb, self.Db = self.Wb0.copy(), self.Db0.copy(); self.Cel0 = self._flux_operator(self.Wb, self.Db)

    # ---------------------------------------------------------------- solve with nu_g and e-e iteration
    def solve(self, E0, omega=0.0, tol=1e-4, maxit=80, freeze_ee=False, warm=True, mAA=6):
        """Self-consistent e-e + growth.  Fixed-point map x -> G(x): x = (density-weighted f0, nu_g); G solves the linear problem
        with the e-e operator built from x.  When e-e dominates the energy balance the plain (Picard) map contracts very weakly,
        so it is accelerated by Anderson mixing (depth mAA).  Converged when the RESIDUAL |G(x) - x|_1 < tol (not when successive
        iterates stop moving) and nu_g is consistent.
        freeze_ee=True: keep the current e-e operator (quasi-static driver updates it from the period mean)."""
        if self.ne <= 0: return super().solve(E0, omega, tol=tol, maxit=maxit)
        if freeze_ee:
            self.nu_g = getattr(self, 'nu_g', 0.0)
            for it in range(maxit):
                f = self._solve_once(E0, omega); f0 = np.real(f[self.Nh, 0])
                nug = float(np.sum(self.w * (self.nu_iz_c - self.nu_att_c) * f0)) if self.growth else 0.0
                if abs(nug - self.nu_g) <= tol * abs(nug) + 1.0: break
                self.nu_g = 0.5 * (self.nu_g + nug) if it > 8 else nug
            self.it = it; return f
        if self._A is None or not warm:
            self.clear_ee(); f = super().solve(E0, omega); self._A = None
            x = np.concatenate([self.w * np.maximum(np.real(f[self.Nh, 0]), 0), [self.nu_g]])
        else:
            x = getattr(self, '_x', None)
            if x is None: x = np.concatenate([self.w * np.maximum(self._f0_from_A(), 0), [self.nu_g]])
        X, Gs = [], []
        sc = None
        for it in range(maxit):
            self._A = None; self.set_ee(np.maximum(x[:-1], 0) / self.w, relax=1.0); self.nu_g = x[-1]
            f = self._solve_once(E0, omega); f0 = np.real(f[self.Nh, 0])
            nug = float(np.sum(self.w * (self.nu_iz_c - self.nu_att_c) * f0)) if self.growth else 0.0
            gx = np.concatenate([self.w * f0, [nug]])
            if sc is None: sc = max(abs(nug), 1e3)          # scale of nu_g in the residual norm
            r = gx - x; res = float(np.sum(np.abs(r[:-1])) + abs(r[-1]) / sc)
            self.res = res
            if res < tol: break
            X.append(x.copy()); Gs.append(gx.copy()); X, Gs = X[-mAA:], Gs[-mAA:]
            if len(X) == 1: x = gx
            else:                                              # Anderson (type II) on the residuals, nu_g scaled
                W = np.ones_like(x); W[-1] = 1 / sc
                R = np.array([(g_ - x_) * W for x_, g_ in zip(X, Gs)])
                dR = (R[1:] - R[:-1]).T; dG = (np.array(Gs[1:]) - np.array(Gs[:-1])).T
                gam = np.linalg.lstsq(dR, R[-1], rcond=None)[0]
                x = Gs[-1] - dG @ gam
                x[:-1] = np.maximum(x[:-1], 0); x[:-1] /= np.sum(x[:-1])
        self._x = gx; self.it = it
        return f

    def _f0_from_A(self):
        return np.diff(self._A[0]) / self.w

    def ee_power(self, f):
        """energy exchanged by the e-e operator (eV/s per electron): ~0 by construction; used as a conservation check."""
        if self.ne <= 0: return 0.0
        Cee = self._flux_operator(self.Wee, self.Dee)
        return float(np.sum(self.w * self.uc * (Cee @ np.real(f[self.Nh, 0]))))

    def ee_particle(self, f):
        if self.ne <= 0: return 0.0
        Cee = self._flux_operator(self.Wee, self.Dee)
        return float(np.sum(self.w * (Cee @ np.real(f[self.Nh, 0]))))
