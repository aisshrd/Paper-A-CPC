"""Figures and table for the self-consistent near-sample layer."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, glob, os, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import i0
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
C = ['#1f5fa8', '#c0392b', '#2e8b57', '#8e44ad', '#d68910', '#555555']
OUT = 'results/sc_figs'; os.makedirs(OUT, exist_ok=True)
L = lambda f: pickle.load(open(f, 'rb'))
e = 1.602176634e-19; M = 39.948 * 1.66053907e-27

def F(p, V, n=1e16, Nh=None, tag='_prod'):
    Nh = Nh if Nh is not None else (1 if V == 0 else (3 if V >= 12 else 2))
    fn = f'results/sc_p{p:g}_n{n:.0e}_V{V:g}_N{Nh}_e80_m48{tag}.pkl'
    return fn if os.path.exists(fn) else None

def recon(H, x_idx, t, nmax):
    s = np.real(H[0][x_idx])[None, :] + 0 * t[:, None]
    for k in range(1, nmax + 1): s = s + 2 * np.real(H[k][x_idx][None, :] * np.exp(1j * k * t)[:, None])
    return s

def profiles(fn, name, title):
    d = L(fn); x = d['xc'] * 1e3; K = d['K']; ne0 = np.real(K['ne'][0]); ni = d['ni']; Nh = d['cfg']['Nh']
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
    a = ax[0]; a.plot(x, d['phi'], c=C[0], label='$\\varphi_0$ (среднее)')
    if Nh >= 1 and d['cfg']['Vrf'] > 0:
        a.plot(x, 2 * np.abs(d['phq'][1]), c=C[1], label='$2|\\varphi_1|$ (амплитуда ВЧ)')
    a.set_xlabel('x, мм'); a.set_ylabel('В'); a.legend(fontsize=7); a.set_title('потенциал')
    a = ax[1]; a.semilogy(x, ne0, c=C[0], label='$n_e$ (среднее)'); a.semilogy(x, ni, c=C[1], label='$n_i$')
    a.set_xlabel('x, мм'); a.set_ylabel('м$^{-3}$'); a.legend(fontsize=7); a.set_title('плотности'); a.set_ylim(1e13, 2e16)
    a = ax[2]; t = np.linspace(0, 2 * np.pi, 121); idx = np.arange(len(x))
    nt = recon(K['ne'], idx, t, Nh) / d['cfg']['nD']
    im = a.pcolormesh(x, t / (2 * np.pi), np.clip(nt, 0, 1.1), shading='auto', cmap='viridis', vmin=0, vmax=1)
    a.set_xlim(0, 3); a.set_xlabel('x, мм'); a.set_ylabel('t / T'); a.set_title('$n_e(x,t)/n_D$: колебание границы слоя')
    plt.colorbar(im, ax=a, shrink=0.85)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png'); plt.close(fig)

def edge_motion(fn, level=0.5):
    """position where n_e(x,t) = level * n_e,plasma, over the period: sheath-edge excursion"""
    d = L(fn); x = d['xc']; K = d['K']; Nh = d['cfg']['Nh']; t = np.linspace(0, 2 * np.pi, 241)
    nt = recon(K['ne'], np.arange(len(x)), t, Nh); nref = np.real(K['ne'][0])[-1]
    pos = []
    for row in nt:
        r = row / nref; k = np.where(r >= level)[0]
        pos.append(x[k.min()] if len(k) else np.nan)
    pos = np.array(pos); return np.nanmin(pos), np.nanmax(pos), t, pos

def table():
    rows = []
    for p in (13.3, 40, 133):
        for V in (0, 4, 8, 12):
            for Nh in ((1,) if V == 0 else (2, 3)):
                fn = F(p, V, Nh=Nh)
                if not fn: continue
                d = L(fn); s = d['summary']; lo, hi, _, _ = edge_motion(fn)
                rows.append(dict(p=p, V=V, Nh=Nh, n=1e16, Vdc=s['Vdc'], Ji=s['Ji'], Ei=s['Ei'], ee=s['ee_surf'], Te=s['Te_D'],
                                 lam=s['lam_i'] * 1e3, lD=s['lD'] * 1e3, edge_lo=lo * 1e3, edge_hi=hi * 1e3, it=len(d['hist'])))
    fn = F(13.3, 8, n=3e16)
    if fn:
        d = L(fn); s = d['summary']; lo, hi, _, _ = edge_motion(fn)
        rows.append(dict(p=13.3, V=8, Nh=2, n=3e16, Vdc=s['Vdc'], Ji=s['Ji'], Ei=s['Ei'], ee=s['ee_surf'], Te=s['Te_D'], lam=s['lam_i'] * 1e3,
                         lD=s['lD'] * 1e3, edge_lo=lo * 1e3, edge_hi=hi * 1e3, it=len(d['hist'])))
    return rows

def fig_vdc(rows):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for k, p in enumerate((13.3, 40, 133)):
        r = sorted([q for q in rows if q['p'] == p and q['n'] == 1e16 and (q['V'] == 0 or q['Nh'] == (3 if q['V'] >= 12 else 2))], key=lambda q: q['V'])
        if not r: continue
        V = np.array([q['V'] for q in r]); Vd = np.array([q['Vdc'] for q in r])
        ax.plot(V, Vd - Vd[0], 'o-', c=C[k], label=f'{p:g} Па')
        T = r[0]['Te']; vv = np.linspace(0, 12, 50); ax.plot(vv, T * np.log(i0(vv / T)), ':', c=C[k], lw=1)
    ax.set_xlabel('амплитуда ВЧ-потенциала $V_{rf}$, В'); ax.set_ylabel('$V_{dc}(V_{rf}) - V_{dc}(0)$, В')
    ax.set_title('выпрямление на слое: расчёт (точки) и $T_e\\ln I_0(V_{rf}/T_e)$'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_vdc.png'); plt.close(fig)

def fig_ions():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for k, p in enumerate((13.3, 40, 133)):
        fn = F(p, 8) or F(p, 0)
        if not fn: continue
        d = L(fn); xe = d['xe'] * 1e3; w = d['wf']; Eion = 0.5 * M * w ** 2 / e
        ax[0].plot(xe, Eion, c=C[k], label=f'{p:g} Па'); ax[1].plot(d['xc'] * 1e3, -d['phi'], c=C[k], label=f'{p:g} Па')
    ax[0].set_xlim(0, 3); ax[0].set_xlabel('x, мм'); ax[0].set_ylabel('энергия направленного движения иона, эВ'); ax[0].legend(fontsize=7)
    ax[0].set_title('ионы: набор энергии и перезарядка')
    ax[1].set_xlim(0, 3); ax[1].set_xlabel('x, мм'); ax[1].set_ylabel('$-\\varphi_0$, В'); ax[1].legend(fontsize=7); ax[1].set_title('средний потенциал')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_ions.png'); plt.close(fig)

if __name__ == '__main__':
    rows = table()
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    pickle.dump(rows, open(f'{OUT}/table.pkl', 'wb'))
    fig_vdc(rows); fig_ions()
    for p in (13.3, 40, 133):
        fn = F(p, 8)
        if fn: profiles(fn, f'fig_prof_p{p:g}', f'{p:g} Па, $n_D$ = 10$^{{16}}$ м$^{{-3}}$, $V_{{rf}}$ = 8 В')
