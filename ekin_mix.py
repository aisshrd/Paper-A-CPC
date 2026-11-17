"""ekin_mix.py -- the energy-grid x Legendre x time-harmonic electron kinetics of ekin.py, generalised to gas MIXTURES.

New compared with ekin.EnergyGridKinetics (argon only):
  * any number of species with mole fractions; ELASTIC or EFFECTIVE momentum-transfer cross sections
    (EFFECTIVE -> elastic by subtracting all inelastic, ionisation and attachment cross sections, as BOLSIG+ does);
  * several ionisation channels (equal energy sharing, two electrons out) and ATTACHMENT (loss, no gain);
  * net production handled by the temporal-growth model (BOLSIG+ default): f ~ exp(nu_g t), nu_g = <nu_iz> - <nu_att>,
    found by fixed-point iteration; nu_g enters every harmonic and every Legendre block;
  * rotational excitation of N2 and O2 by the continuous approximation (CAR, Frost & Phelps): energy loss rate 4 B sigma0 N v,
    sigma0 = 8 pi q^2 a0^2 / 15, added to the elastic energy-loss flux (with the matching gas-temperature diffusion term).
Everything else (finite volumes in energy, Scharfetter-Gummel elastic flux, particle-conserving inelastic gains, harmonic balance
for E(t) = E0 cos(w t)) is inherited unchanged.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.constants as co
from bolos import parser
from ekin import EnergyGridKinetics, GAMMA, e, me, kB

A0 = co.physical_constants['Bohr radius'][0]
CAR = {'N2': (2.4775e-4, 1.09), 'O2': (1.7825e-4, 1.4)}          # rotational constant B (eV), quadrupole moment q (e a0^2)
import os
LOKI = os.environ.get('LXCAT_DIR', 'dl/LoKI/Code/Input/')   # set LXCAT_DIR, or put the files in data/ (see data/README.md)
FILES = {'Ar': LOKI + 'Argon/Ar_LXCat.txt', 'N2': LOKI + 'Nitrogen/N2_LXCat.txt', 'O2': LOKI + 'Oxygen/O2_LXCat.txt',
         'He': LOKI + 'Helium/He_LXCat.txt'}
if not os.path.exists(FILES['Ar']): FILES = {k: f'data/{k}_LXCat.txt' for k in FILES}

def load_species(name, path=None):
    with open(path or FILES[name]) as fp: procs = parser.parse(fp)
    s = dict(name=name, el=None, eff=None, mM=None, inel=[], iz=[], att=[])
    for p in procs:
        d = np.array(p['data'], float); k = p['kind']
        if k == 'ELASTIC': s['el'] = d; s['mM'] = p.get('mass_ratio')
        elif k in ('EFFECTIVE', 'MOMENTUM'): s['eff'] = d; s['mM'] = p.get('mass_ratio')
        elif k == 'EXCITATION': s['inel'].append((float(p['threshold']), d, p.get('product')))
        elif k == 'IONIZATION': s['iz'].append((float(p['threshold']), d, p.get('product')))
        elif k == 'ATTACHMENT': s['att'].append((d, p.get('product')))
    return s

def _sig(d, x):
    return np.interp(x, d[:, 0], d[:, 1], left=d[0, 1], right=d[-1, 1])

class MixtureKinetics(EnergyGridKinetics):
    def __init__(self, mix, umax=60., nu=300, L=1, Nh=0, ng=3.2e21, Tg=300., quadratic=True, car=True, growth=True):
        """mix: list of (species dict from load_species, mole fraction)."""
        self.mix = [(s, x) for s, x in mix if x > 0]; self.L, self.Nh, self.H, self.ng, self.kTg = L, Nh, 2 * Nh + 1, ng, kB * Tg / e
        self.car, self.growth = car, growth
        sgrid = np.linspace(0, 1, nu + 1); self.ub = umax * (sgrid ** 2 if quadratic else sgrid)
        self.uc = 0.5 * (self.ub[1:] + self.ub[:-1]); self.du = np.diff(self.ub); self.nu = nu; u = self.uc
        self.w = np.sqrt(u) * self.du
        vc = GAMMA * np.sqrt(u)
        self.procs = []                      # (kind, species, product, threshold, nu_k(u) at centres incl. fraction)
        self.sig_el_c = np.zeros(nu); self.sig_el_b = np.zeros(nu + 1)
        Wb = np.zeros(nu + 1); Db = np.zeros(nu + 1); self.Pel = np.zeros(nu); self.Pcar = np.zeros(nu)
        for s, x in self.mix:
            sig_in_c = np.zeros(nu); sig_in_b = np.zeros(nu + 1)
            for th, d, prod in s['inel']:
                sc = _sig(d, u) * (u > th); sig_in_c += sc; sig_in_b += _sig(d, self.ub) * (self.ub > th)
                self.procs.append(('exc', s['name'], prod, th, x * ng * sc * vc))
            for th, d, prod in s['iz']:
                sc = _sig(d, u) * (u > th); sig_in_c += sc; sig_in_b += _sig(d, self.ub) * (self.ub > th)
                self.procs.append(('iz', s['name'], prod, th, x * ng * sc * vc))
            for d, prod in s['att']:
                sc = _sig(d, u); sig_in_c += sc; sig_in_b += _sig(d, self.ub)
                self.procs.append(('att', s['name'], prod, 0.0, x * ng * sc * vc))
            if s['el'] is not None: el_c, el_b = _sig(s['el'], u), _sig(s['el'], self.ub)
            else: el_c = np.maximum(_sig(s['eff'], u) - sig_in_c, 0); el_b = np.maximum(_sig(s['eff'], self.ub) - sig_in_b, 0)
            self.sig_el_c += x * el_c; self.sig_el_b += x * el_b
            a = 2 * s['mM']
            Wb += -a * self.ub ** 2 * x * el_b * ng * GAMMA; Db += a * self.kTg * self.ub ** 2 * x * el_b * ng * GAMMA
            self.Pel += a * u * x * ng * el_c * vc                             # elastic power per electron at u (eV/s)
            if car and s['name'] in CAR:
                B, q = CAR[s['name']]; s0 = 8 * np.pi * q ** 2 * A0 ** 2 / 15
                Wb += -4 * B * s0 * x * ng * GAMMA * self.ub; Db += 4 * B * s0 * x * ng * GAMMA * self.ub * self.kTg
                self.Pcar += 4 * B * s0 * x * ng * vc
        self.Wb, self.Db = Wb, Db
        self.nu_m_c = ng * self.sig_el_c * vc
        self.nu_tot_inel = sum(p[4] for p in self.procs) if self.procs else np.zeros(nu)
        self.nu_iz_c = sum((p[4] for p in self.procs if p[0] == 'iz'), np.zeros(nu))
        self.nu_att_c = sum((p[4] for p in self.procs if p[0] == 'att'), np.zeros(nu))
        self.nu_g = 0.0
        self._build_static()

    def _build_static(self):
        nu, L = self.nu, self.L
        self.Cel0 = self._flux_operator(self.Wb, self.Db)
        Cin = sp.lil_matrix((nu, nu))
        for kind, sname, prod, th, nuk in self.procs:
            Cin -= sp.diags(nuk)
            if kind == 'exc': Cin += self._shift_gain(th, nuk)
            elif kind == 'iz':     # equal sharing; with growth two electrons out per event, without growth particle-conserving
                Cin += (2.0 if self.growth else 1.0) * 2 * self._shift_gain_scaled(th, nuk, share=0.5)
        self.Cin0 = Cin.tocsr()
        self.Closs_l = sp.diags(self.nu_tot_inel); self.Cm = sp.diags(self.nu_m_c)
        self.Fdown = [None] * (L + 1); self.Fup = [None] * (L + 1)
        v = GAMMA * np.sqrt(self.uc); Dv = self._dv_matrix(v)
        for l in range(L + 1):
            if l >= 1: self.Fdown[l] = (l / (2 * l - 1)) * (sp.diags(v ** (l - 1)) @ Dv @ sp.diags(v ** (-(l - 1))))
            if l < L: self.Fup[l] = ((l + 1) / (2 * l + 3)) * (sp.diags(v ** (-(l + 2))) @ Dv @ sp.diags(v ** (l + 2)))
        self.Dv = Dv

    def assemble(self, E0, omega=0.0):
        """two-term (L = 1) harmonic balance on a STAGGERED grid: f0_n at cell centres, f1_n at interior cell boundaries.
        The field term in the f0 equation is an exact divergence (telescopes with the density weights), the f1 equation uses
        the two-point derivative of f0 across the boundary -> no odd-even decoupling, positive EEDF (as in BOLSIG+)."""
        assert self.L == 1, 'mixture solver is two-term'
        nu, Nh = self.nu, self.Nh; nb = nu - 1; a = e * E0 / me
        vc = GAMMA * np.sqrt(self.uc); vb = GAMMA * np.sqrt(self.ub[1:-1])
        g3 = 2.0 / GAMMA ** 3
        # f0 row i  <- f1 at boundaries i (left, index i-1 in the nb array) and i+1 (right, index i)
        Fup = sp.lil_matrix((nu, nb))
        for i in range(nu):
            if i < nu - 1: Fup[i, i] += g3 / (3 * self.w[i]) * vb[i] ** 2
            if i > 0:      Fup[i, i - 1] -= g3 / (3 * self.w[i]) * vb[i - 1] ** 2
        Fdn = sp.lil_matrix((nb, nu))
        for j in range(nb):
            h = vc[j + 1] - vc[j]; Fdn[j, j + 1] = 1 / h; Fdn[j, j] = -1 / h
        Fup, Fdn = Fup.tocsr(), Fdn.tocsr()
        nub = np.interp(self.ub[1:-1], self.uc, self.nu_m_c + self.nu_tot_inel)
        D1 = sp.diags(nub + self.nu_g)
        I0 = sp.identity(nu); I1 = sp.identity(nb)
        blocks = {}
        def add(r, c, M): blocks[(r, c)] = blocks[(r, c)] + M if (r, c) in blocks else M
        for n in range(-Nh, Nh + 1):
            r0, r1 = ('0', n), ('1', n)
            add(r0, r0, (1j * n * omega + self.nu_g) * I0 - (self.Cel0 + self.Cin0))
            add(r1, r1, 1j * n * omega * I1 + D1)
            for q, cq in ([(0, 1.0)] if omega == 0 else [(1, 0.5), (-1, 0.5)]):
                m = n - q
                if abs(m) > Nh: continue
                add(r0, ('1', m), -a * cq * Fup)
                add(r1, ('0', m), -a * cq * Fdn)
        keys = [(k, n) for n in range(-Nh, Nh + 1) for k in ('0', '1')]
        self._keys = keys
        return sp.bmat([[blocks.get((r, c)) for c in keys] for r in keys], format='lil')

    def _solve_once(self, E0, omega):
        A = self.assemble(E0, omega); nu, nb, Nh = self.nu, self.nu - 1, self.Nh
        off = {}; o = 0
        for k in self._keys: off[k] = o; o += nu if k[0] == '0' else nb
        r = off[('0', 0)] + nu // 3; A[r, :] = 0
        for i in range(nu): A[r, off[('0', 0)] + i] = self.w[i]
        b = np.zeros(A.shape[0], complex); b[r] = 1.0
        x = spla.spsolve(A.tocsc(), b)
        f = np.zeros((2 * Nh + 1, 2, nu), complex)
        for n in range(-Nh, Nh + 1):
            f[n + Nh, 0] = x[off[('0', n)]: off[('0', n)] + nu]
            fb = x[off[('1', n)]: off[('1', n)] + nb]
            f[n + Nh, 1] = np.interp(self.uc, self.ub[1:-1], fb.real) + 1j * np.interp(self.uc, self.ub[1:-1], fb.imag)
            setattr(self, f'_f1b_{n + Nh}', fb)
        self._f1b = [getattr(self, f'_f1b_{k}') for k in range(2 * Nh + 1)]
        return f

    def solve(self, E0, omega=0.0, tol=1e-4, maxit=40):
        self.nu_g = 0.0
        f = self._solve_once(E0, omega)
        if not self.growth: return f
        for it in range(maxit):
            f0 = np.real(f[self.Nh, 0]); nug = float(np.sum(self.w * (self.nu_iz_c - self.nu_att_c) * f0))
            if abs(nug - self.nu_g) <= tol * abs(nug) + 1.0: break
            self.nu_g = 0.5 * (self.nu_g + nug) if it > 8 else nug
            f = self._solve_once(E0, omega)
        self.it = it
        return f

    # ---------------------------------------------------------- diagnostics
    def rates(self, f, harm=0):
        """rate coefficients (m^3/s) per process: <nu_k>/(x_s N), time average (harm=0) or complex harmonic amplitude"""
        out = []
        fr = dict((s['name'], x) for s, x in self.mix)
        for kind, sname, prod, th, nuk in self.procs:
            r = complex(np.sum(self.w * nuk * f[self.Nh + harm, 0])) / (fr[sname] * self.ng)
            out.append((kind, sname, prod, th, r if harm else r.real))
        return out

    def power_split(self, f):
        """time-averaged power per electron (eV/s) into: elastic, rotation (CAR), each inelastic process (th * rate), ionisation"""
        f0 = np.real(f[self.Nh, 0])
        P = dict(elastic=float(np.sum(self.w * self.Pel * f0)), rotation=float(np.sum(self.w * self.Pcar * f0)))
        for kind, sname, prod, th, nuk in self.procs:
            if kind == 'att': continue
            key = f'{kind}:{sname}:{prod}'; P[key] = P.get(key, 0.0) + th * float(np.sum(self.w * nuk * f0))
        return P

    def drift_b(self, harm=0):
        """<v_z> (m/s) from the staggered f1: (1/3) int v f1 sqrt(u) du over the boundary nodes"""
        ubi = self.ub[1:-1]; wb = np.sqrt(ubi) * np.diff(self.uc)
        return complex(np.sum(wb * GAMMA * np.sqrt(ubi) * self._f1b[self.Nh + harm])) / 3

    def mobility_N(self, f, E0):
        return abs(self.drift_b(0)) / E0 * self.ng
