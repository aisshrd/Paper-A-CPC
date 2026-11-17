"""Nonlocal linear-response conductivity kernels of the electron kinetic solution (cylinder, total energy, Nh = 1).

For one RF harmonic (Nh = 1) the n = +1 block of the harmonic-balance system reads
    A_11 f_1 + G_E(E_theta^(1)) f_0 + G_B(B_z^(1)) f_0 + G_R(E_r^(1)) f_0 = 0 ,
with A_11 independent of the RF fields (collisions, streaming, i*omega, static B0 and the ambipolar field in the total-energy
variable) and G_X linear and diagonal in the radial node of the field.  At fixed f_0 the first-harmonic currents are therefore
EXACTLY linear functionals of the RF fields:
    j_c(r) = sum_r' [ K_cE(r,r') E_theta(r') + K_cB(r,r') B_z(r') + K_cR(r,r') E_r(r') ] ,   c = theta, r.
The kernels are obtained with one LU of A_11 and 3*m back-substitutions.  They replace the local sigma = j/E of the
Picard loop (singular where E_theta is small, unstable when transport is strongly nonlocal, e.g. at 2 Pa) and give E_r for
j_r = 0 exactly (no diagonal Newton).
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.constants as co
e = co.e

def kernels(k, f, theta=None):
    """k: CylKineticsMag (B0 set; k.Er is ignored here), f: its solution.  theta: limiter to freeze (default: from f).
    Returns dict of complex (m x m) kernels per unit n0, harmonic +1: KtE, KtB, KtR, KrE, KrB, KrR."""
    Nh, nlm, ne, m = k.Nh, k.nlm, k.n_e, k.m; N1 = nlm * ne * m
    if k.order == 2: k._theta = k.limiter_theta(f) if theta is None else theta
    Er_save = getattr(k, 'Er', None); k.Er = None
    Z = np.zeros((k.H, m), complex); O = Z.copy(); O[Nh + 1] = 1.0
    A0 = k.kinetic_matrix(Z, Z).tocsr()
    b1 = slice((1 + Nh) * N1, (2 + Nh) * N1); b0 = slice(Nh * N1, (Nh + 1) * N1)
    A11 = A0[b1, b1].tocsc()
    GE = (k.kinetic_matrix(O, Z).tocsr() - A0)[b1, b0]
    GB = (k.kinetic_matrix(Z, O).tocsr() - A0)[b1, b0]
    GR = k._Er_matrix(O).tocsr()[b1, b0]
    k.Er = Er_save; k._theta = None
    f0 = f[Nh].ravel(); P = sp.csr_matrix((np.ones(N1), (np.arange(N1), np.arange(N1) % m)), shape=(N1, m))
    lu = spla.splu(A11)
    # current weights: j_c(node) = sum over (a,i) of cw_c[a,i,node] * f1[a,i,node]
    def cweights(comp):
        vh = k.ang.unit_vectors()[0][comp]; c = np.einsum('aij,ij,ij->a', k.ang.Y, vh, k.ang.W)
        return -e * c[:, None, None] * (k.w * k.v)[None] / np.sqrt(4 * np.pi)
    cw = {'t': cweights(1), 'r': cweights(0)}
    out = {}
    for name, G in (('E', GE), ('B', GB), ('R', GR)):
        rhs = -(G @ sp.diags(f0) @ P).toarray()
        F1 = lu.solve(rhs).reshape(nlm, ne, m, m)                    # (a, i, node, basis)
        for c in ('t', 'r'): out['K' + c + name] = np.einsum('aik,aikb->kb', cw[c], F1)
    return out

def maxwell_nl(dis, K, n0, limit='jr0'):
    """Maxwell for harmonics +-1 with the nonlocal kernels (physical current = n0 * kernel * field).
    limit 'jr0': E_r eliminated from j_r = 0; 'Er0': E_r = 0.  Returns E, B, Er (H x m)."""
    kk = dis.k; H, Nh, m = kk.H, kk.Nh, kk.m; R = kk.R; w = kk.omega; mu0 = co.mu_0; r = kk.r
    Dodd = kk.grid['Dodd'] / R; Deven = kk.grid['Deven'] / R
    KtE, KtB = K['KtE'] * n0, K['KtB'] * n0
    if limit == 'jr0' and np.max(np.abs(K['KrR'])) > 0 and getattr(dis, 'B0', 0) > 0:
        Sr = -np.linalg.solve(K['KrR'], np.hstack([K['KrE'], K['KrB']]))   # E_r = Sr_E E + Sr_B B
        SE, SB = Sr[:, :m], Sr[:, m:]
        KtE = KtE + n0 * K['KtR'] @ SE; KtB = KtB + n0 * K['KtR'] @ SB
    else: SE = SB = np.zeros((m, m))
    E = np.zeros((H, m), complex); B = np.zeros((H, m), complex); Er = np.zeros((H, m), complex)
    for n in (-1, 1):
        cE, cB = (KtE, KtB) if n == 1 else (np.conj(KtE), np.conj(KtB))
        A = np.zeros((2 * m, 2 * m), complex); rhs = np.zeros(2 * m, complex)
        A[:m, :m] = np.diag(r) @ Dodd + np.eye(m); A[:m, m:] = 1j * n * w * np.diag(r)
        A[m:, m:] = Deven + mu0 * cB; A[m:, :m] = mu0 * cE
        A[m, :] = 0; A[m, m] = 1; rhs[m] = dis.B_ext[n + Nh]
        x = np.linalg.solve(A, rhs); E[n + Nh] = x[:m]; B[n + Nh] = x[m:]
    Er[Nh + 1] = SE @ E[Nh + 1] + SB @ B[Nh + 1]; Er[Nh - 1] = np.conj(Er[Nh + 1])
    return E, B, Er
