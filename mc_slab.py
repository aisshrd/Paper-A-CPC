"""Planar electron Monte Carlo (1D in x, 3D in velocity) for the near-surface-layer model of slab_case.py.
Full gap [0, 2D] with absorbing surfaces at both ends; fields:
   static  E0(x) = -dphi0/dx with phi0 symmetric about the midplane (barrier at each surface);
   RF      E(x,t) = Ea g(x) cos(w t)  on the left half;  right half: 'cc'  -> +Ea g(2D-x) cos(w t)   (antiphase sheaths)
                                                                    'even' -> -Ea g(2D-x) cos(w t)   (in-phase sheaths)
Real argon cross sections (same LXCat set), null-collision method with Poisson candidates per step, isotropic scattering,
ionisation with equal energy sharing and the secondary electron followed (both electrons counted, as iz_mult=2 in the solver).
Population control: random cloning/killing keeps N within +-20 % of N0 (samples the fundamental mode = bordered eta).
Diagnostics resolved in RF phase (nph bins) and x (nb bins over the full gap): density, ionisation, total excitation,
4p+ excitation (emission), kinetic energy, x-velocity; absorbed particles and energy per phase at each surface;
growth rate eta = (births - deaths)/(N dt) averaged.
usage: python3 mc_slab.py p Ea Vb mirror N0 periods seed [outfile]   (env D_MM, S0_MM, D_OSC_MM, DELTA_MM as slab_case.py)"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, sys, os, time, pickle
import numba as nb
import scipy.constants as co
from ekin import load_argon, GAMMA
from slab_case import model
e, me = co.e, co.m_e

@nb.njit(cache=True)
def _seed(s): np.random.seed(s)

@nb.njit(cache=True)
def advance(x, vx, vp, n, cap, t0, nsteps, dt, w, Lx, xf, dxf, E0f, E1f, du, nucum, ths, kiz, nu_max,
            nu_iz_t, nu_ex_t, nu_em_t, nph, nbx, acc, surf, sample_every, ctr):
    """advance n particles by nsteps; returns new n.  acc[ph, bx, q]; surf[side, ph, 0:count 1:energy]; ctr[0]=births ctr[1]=deaths"""
    T = 2 * np.pi / w; qm = -e / me; nproc = nucum.shape[0]; nu_tab = nucum.shape[1]
    lam = nu_max * dt; emlam = np.exp(-lam)
    for s in range(nsteps):
        t = t0 + s * dt; cw = np.cos(w * t + 0.5 * w * dt)      # field at mid-step
        ph = int(((t % T) / T) * nph) % nph
        do_sample = (s % sample_every) == 0
        nnew = n
        p = 0
        while p < n:
            xi = x[p]
            if xi < 0:      # dead marker
                p += 1; continue
            # field (linear interpolation on the fine grid)
            fx = xi / dxf; i0 = int(fx)
            if i0 >= xf.shape[0] - 1: i0 = xf.shape[0] - 2
            tt = fx - i0
            Ex = (1 - tt) * E0f[i0] + tt * E0f[i0 + 1] + cw * ((1 - tt) * E1f[i0] + tt * E1f[i0 + 1])
            vx[p] += qm * Ex * dt
            xi += vx[p] * dt
            if xi <= 0.0 or xi >= Lx:
                side = 0 if xi <= 0.0 else 1
                u = 0.5 * me * (vx[p] ** 2 + vp[p] ** 2) / e
                surf[side, ph, 0] += 1.0; surf[side, ph, 1] += u
                x[p] = -1.0; ctr[1] += 1; p += 1; continue
            x[p] = xi
            # collisions: Poisson number of null-collision candidates
            r = np.random.random(); k = 0; pr = emlam; cum = pr
            while r > cum and k < 10:
                k += 1; pr *= lam / k; cum += pr
            for c in range(k):
                v2 = vx[p] ** 2 + vp[p] ** 2; u = 0.5 * me * v2 / e; v = np.sqrt(v2)
                iu = int(u / du)
                if iu >= nu_tab - 1: iu = nu_tab - 2
                rr = np.random.random() * nu_max
                proc = -1
                for kk in range(nproc):
                    if rr < nucum[kk, iu] * v:
                        proc = kk; break
                if proc < 0: continue      # null collision
                if proc == 0:
                    cosx = 1 - 2 * np.random.random(); vnew = v * np.sqrt(max(1 - 2 * 1.371e-5 * (1 - cosx), 0.0))
                elif proc == kiz:
                    unew = 0.5 * (u - ths[proc])
                    if unew < 0: unew = 0.0
                    vnew = np.sqrt(2 * e * unew / me)
                    # secondary electron
                    if nnew < cap:
                        cth = 1 - 2 * np.random.random(); x[nnew] = xi; vx[nnew] = vnew * cth; vp[nnew] = vnew * np.sqrt(1 - cth * cth)
                        nnew += 1; ctr[0] += 1
                else:
                    unew = u - ths[proc]
                    if unew < 0: unew = 0.0
                    vnew = np.sqrt(2 * e * unew / me)
                cth = 1 - 2 * np.random.random(); vx[p] = vnew * cth; vp[p] = vnew * np.sqrt(1 - cth * cth)
            p += 1
        n = nnew
        if do_sample:
            for p in range(n):
                xi = x[p]
                if xi < 0: continue
                bx = int(xi / Lx * nbx)
                if bx >= nbx: bx = nbx - 1
                v2 = vx[p] ** 2 + vp[p] ** 2; u = 0.5 * me * v2 / e; v = np.sqrt(v2)
                iu = int(u / du)
                if iu >= nu_tab - 1: iu = nu_tab - 2
                acc[ph, bx, 0] += 1.0
                acc[ph, bx, 1] += nu_iz_t[iu] * v
                acc[ph, bx, 2] += nu_ex_t[iu] * v
                acc[ph, bx, 3] += nu_em_t[iu] * v
                acc[ph, bx, 4] += u
                acc[ph, bx, 5] += vx[p]
        # compact (remove dead)
        if s % 20 == 19 or s == nsteps - 1:
            j = 0
            for p in range(n):
                if x[p] >= 0:
                    x[j] = x[p]; vx[j] = vx[p]; vp[j] = vp[p]; j += 1
            n = j
    return n

@nb.njit(cache=True)
def eedf_sample(x, vx, vp, n, Lx, nbx, du_h, hist):
    for p in range(n):
        if x[p] < 0: continue
        bx = int(x[p] / Lx * nbx)
        if bx >= nbx: bx = nbx - 1
        u = 0.5 * me * (vx[p] ** 2 + vp[p] ** 2) / e; iu = int(u / du_h)
        if iu < hist.shape[1]: hist[bx, iu] += 1.0

def main():
    a = sys.argv
    p, Ea, Vb, mir = float(a[1]), float(a[2]), float(a[3]), a[4]
    N0 = int(a[5]); periods = int(a[6]); seed = int(a[7]); out = a[8] if len(a) > 8 else None
    M = model(p, Ea, Vb); D = M['D']; Lx = 2 * D; ng = p / (1.380649e-23 * 300.0); w = 2 * np.pi * 13.56e6
    xs = load_argon('data/Ar_LXCat.txt')
    # fields on a fine grid over the full gap
    nxf = 8001; xf = np.linspace(0, Lx, nxf); dxf = xf[1] - xf[0]
    xl = np.minimum(xf, Lx - xf)
    phi = np.array([M['phi0'](v) for v in xl])
    # exact derivative of the parabolic barrier: phi0 = -Vb (1-x/s0)^2 -> dphi0/dx = 2 Vb (1-x/s0)/s0 on the left
    s0 = M['s0']; dphi_l = np.where(xl < s0, 2 * Vb * (1 - xl / s0) / s0, 0.0)
    E0f = np.where(xf <= D, -dphi_l, +dphi_l)                # E = -dphi/dx (antisymmetric)
    g = np.array([M['g'](v) for v in xl]); sgn = 1.0 if mir == 'cc' else -1.0
    E1f = Ea * np.where(xf <= D, g, sgn * g)
    # cross-section tables (uniform in u): cumulative nu/v = N sigma
    du = 0.005; ut = np.arange(0, 120, du)
    sig = lambda d, x: np.interp(x, d[:, 0], d[:, 1], left=d[0, 1], right=d[-1, 1])
    procs = [(0.0, sig(xs['elastic'], ut))] + [(th, sig(d, ut) * (ut > th)) for th, d, _ in xs['inel']] + [(xs['iz'][0], sig(xs['iz'][1], ut) * (ut > xs['iz'][0]))]
    ths = np.array([th for th, _ in procs]); kiz = len(procs) - 1
    nucum = ng * np.cumsum(np.array([s for _, s in procs]), axis=0)
    nu_max = 1.05 * np.max(nucum[-1] * GAMMA * np.sqrt(ut)[None, :][0] * (ut <= 100))
    nu_iz_t = ng * procs[-1][1]; nu_ex_t = ng * sum(s for th, s in procs[1:-1]); nu_em_t = ng * sum(s for th, s in procs[1:-1] if th > 12.5)
    dt = 0.5 / nu_max; T = 2 * np.pi / w; steps_per_period = int(np.ceil(T / dt)); dt = T / steps_per_period
    _seed(seed); rng = np.random.default_rng(seed)
    cap = 3 * N0; x = -np.ones(cap); vx = np.zeros(cap); vp = np.zeros(cap)
    # initial: uniform in the bulk part, Maxwellian 3 eV
    x[:N0] = s0 + (Lx - 2 * s0) * rng.random(N0); vth = np.sqrt(e * 3.0 / me)
    v3 = rng.normal(0, vth, (N0, 3)); vx[:N0] = v3[:, 0]; vp[:N0] = np.hypot(v3[:, 1], v3[:, 2]); n = N0
    nph, nbx = 24, 64
    acc_tot = np.zeros((nph, nbx, 6)); surf_tot = np.zeros((2, nph, 2)); hist = np.zeros((nbx, 300)); du_h = 0.2
    burn = int(os.environ.get('MC_BURN', periods // 3)); t = 0.0; eta_num = 0.0; eta_den = 0.0; logW = 0.0
    t_start = time.time(); hist_eta = []
    for P in range(periods):
        acc = np.zeros((nph, nbx, 6)); surf = np.zeros((2, nph, 2)); ctr = np.zeros(2)
        n_start = n
        n = advance(x, vx, vp, n, cap, t, steps_per_period, dt, w, Lx, xf, dxf, E0f, E1f, du, nucum, ths, kiz, nu_max,
                    nu_iz_t, nu_ex_t, nu_em_t, nph, nbx, acc, surf, 4, ctr)
        t += steps_per_period * dt
        eta_P = (ctr[0] - ctr[1]) / (0.5 * (n_start + n) * T); hist_eta.append(eta_P)
        if P >= burn:
            norm = 0.5 * (n_start + n)          # remove the slow growth: per-period normalisation
            acc_tot += acc / norm; surf_tot += surf / norm; eta_num += ctr[0] - ctr[1]; eta_den += norm * T
            eedf_sample(x, vx, vp, n, Lx, nbx, du_h, hist)
        # population control
        if n > 1.2 * N0 or n < 0.8 * N0:
            idx = rng.choice(n, N0, replace=(n < N0))
            x[:N0] = x[idx]; vx[:N0] = vx[idx]; vp[:N0] = vp[idx]; x[N0:] = -1.0; n = N0
        if P % max(1, periods // 20) == 0:
            print(f'  period {P}/{periods}  n={n}  eta_P={eta_P:.3e}  ({time.time()-t_start:.0f}s)', flush=True)
    res = dict(p=p, Ea=Ea, Vb=Vb, mirror=mir, N0=N0, periods=periods, burn=burn, nph=nph, nbx=nbx, Lx=Lx, D=D,
               acc=acc_tot, surf=surf_tot, eta=eta_num / eta_den, eta_hist=np.array(hist_eta), hist=hist, du_h=du_h,
               dt=dt, time=time.time() - t_start, sample_every=4, steps_per_period=steps_per_period)
    fn = out or f'results/mc_slab_p{p}_E{Ea:g}_V{Vb:g}_{mir}_N{N0}_P{periods}_s{seed}.pkl'
    pickle.dump(res, open(fn, 'wb')); print('saved', fn, 'eta', res['eta'], 'time', res['time'])

if __name__ == '__main__':
    main()
