"""Figures for the near-surface layer report (item 8)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, sys, os, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from slab_compare import mc_harmonics
from slab_theory import ks
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
C = ['#1f5fa8', '#c0392b', '#2e8b57', '#8e44ad', '#d68910', '#555555']
OUT = 'results/slab_figs'; os.makedirs(OUT, exist_ok=True)
L = lambda f: pickle.load(open(f, 'rb'))
W = 2 * np.pi * 13.56e6
PROD = {13.3: 'results/slab_p13.3_E1920_V20_cc_L3_N3_e90_m40_prod.pkl',
        40: 'results/slab_p40_E5780_V20_cc_L3_N3_e90_m40_prod.pkl',
        133: 'results/slab_p133_E19200_V20_cc_L3_N3_e90_m40_prod.pkl'}
EVEN = 'results/slab_p13.3_E1920_V20_even_L3_N3_e90_m40_prod.pkl'

def recon(H, q, x_idx, t, nmax):
    """time signal X(t) = sum_n X_n e^{i n w t} at cells x_idx"""
    s = np.real(H[0][q][x_idx])[None, :] + 0 * t[:, None]
    for n in range(1, nmax + 1):
        s = s + 2 * np.real(H[n][q][x_idx][None, :] * np.exp(1j * n * t)[:, None])
    return s

def fig1_model():
    from slab_case import model
    M = model(13.3, 1920, 20); x = np.linspace(0, 8e-3, 800)
    phi0 = np.array([M['phi0'](v) for v in x]); g = np.array([M['g'](v) for v in x])
    # RF potential relative to the bulk: phi1(x) = int_x^inf E1 dx'
    E1 = 1920 * g; phi1 = np.cumsum((E1 * np.gradient(x))[::-1])[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    a = ax[0]
    for th, c in ((0, C[1]), (np.pi / 2, C[5]), (np.pi, C[0])):
        a.plot(x * 1e3, -(phi0 + phi1 * np.cos(th)), c=c, label=f'ωt = {int(round(np.degrees(th)))}°')
    a.plot(x * 1e3, -phi0, 'k--', lw=0.8, label='среднее (барьер)')
    a.set_xlabel('x, мм (поверхность при x = 0)'); a.set_ylabel('потенциальная энергия электрона, эВ'); a.legend(fontsize=7)
    a.set_title('барьер слоя и его колебания')
    a = ax[1]; a.plot(x * 1e3, E1, c=C[2]); a.set_xlabel('x, мм'); a.set_ylabel('амплитуда ВЧ-поля, В/м')
    a.axvspan(0, 2, color=C[2], alpha=0.08); a.text(0.15, 1500, 'область\nосцилляций', fontsize=8)
    a.text(4.5, 300, 'область без поля\n(нет нагрева)', fontsize=8); a.set_title('ВЧ-поле (13.3 Па, 600 Тд)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig1_model.png'); plt.close(fig)

def fig_mc(sfile, mfiles, name, title):
    S = L(sfile); x = S['xc'] * 1e3; H = S['H']
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.4))
    a = ax[0, 0]; a.plot(x, H[0]['n'].real, '-', c=C[0], lw=1.6, label='решатель')
    a2 = ax[0, 1]; a2.plot(x, 2 / 3 * H[0]['en'].real / H[0]['n'].real, '-', c=C[0], lw=1.6)
    a3 = ax[1, 0]; a4 = ax[1, 1]
    for q, cc, lab in (('n', C[2], '$|n_1|/n_0$'), ('iz', C[0], '$|R_{iz,2}|/R_{iz,0}$'), ('em', C[4], '$|R_{4p,2}|/R_{4p,0}$')):
        nn = 1 if q == 'n' else 2
        a4.plot(x, np.abs(H[nn][q]) / H[0][q].real, '-', c=cc, lw=1.6, label=lab)
    a3.semilogy(x, H[0]['ex'].real / H[0]['n'].real, '-', c=C[0], lw=1.6, label='решатель')
    mk = ['o', 's', '^']
    for k, (mf, lab) in enumerate(mfiles):
        R = L(mf); xb, M, Ms, A, tot = mc_harmonics(R); xm = xb * 1e3
        a.plot(xm, M[0]['n'].real, mk[k], ms=3, mfc='none', c=C[1 + k], label=lab)
        a2.plot(xm, 2 / 3 * M[0]['en'].real / M[0]['n'].real, mk[k], ms=3, mfc='none', c=C[1 + k])
        a3.semilogy(xm, M[0]['ex'].real / M[0]['n'].real, mk[k], ms=3, mfc='none', c=C[1 + k], label=lab)
        if k == len(mfiles) - 1:
            for q, cc in (('n', C[2]), ('iz', C[0]), ('em', C[4])):
                nn = 1 if q == 'n' else 2
                a4.plot(xm, np.abs(M[nn][q]) / M[0][q].real, mk[k], ms=2.5, c=cc)
    a.set_ylabel('$n_0(x)$, норм.'); a.legend(fontsize=7); a.set_title('плотность')
    a2.set_ylabel('$T_e = \\frac{2}{3}\\langle\\varepsilon\\rangle$, эВ'); a2.set_title('температура')
    a3.set_ylabel('$\\nu_{ex}$, с$^{-1}$'); a3.set_xlabel('x, мм'); a3.legend(fontsize=7); a3.set_title('частота возбуждения')
    a4.set_ylim(0, 1.05); a4.set_xlabel('x, мм'); a4.set_ylabel('глубина модуляции'); a4.legend(fontsize=7); a4.set_title('гармоники (точки: МК)')
    for b in ax.ravel(): b.axvline(S['cfg']['d'] * 1e3, c='gray', lw=0.8, ls=':')
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(f'{OUT}/{name}.png'); plt.close(fig)

def fig_maps():
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
    t = np.linspace(0, 2 * np.pi, 97)
    for a, (p, f) in zip(ax, PROD.items()):
        if not os.path.exists(f): continue
        S = L(f); H = S['H']; Nh = S['cfg']['Nh']; x = S['xc'] * 1e3; idx = np.arange(len(x))
        R = recon(H, 'iz', idx, t, Nh); R = np.maximum(R, 0) / np.max(R)
        im = a.pcolormesh(x, t / (2 * np.pi), R, shading='auto', cmap='magma', vmin=0, vmax=1)
        a.set_xlim(0, 8); a.set_xlabel('x, мм'); a.set_title(f'{p:g} Па'); a.axvline(2, c='w', lw=0.6, ls=':')
    ax[0].set_ylabel('t / T'); fig.colorbar(im, ax=ax, shrink=0.85, label='скорость ионизации (норм. на максимум)')
    fig.savefig(f'{OUT}/fig3_maps.png'); plt.close(fig)

def fig_modulation():
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for k, (p, f) in enumerate(PROD.items()):
        if not os.path.exists(f): continue
        S = L(f); H = S['H']; x = S['xc'] * 1e3
        m2 = np.abs(H[2]['iz']) / H[0]['iz'].real; ph = np.degrees(np.unwrap(np.angle(H[2]['iz'])))
        ok = H[0]['iz'].real > 1e-3 * H[0]['iz'].real.max()
        ax[0].plot(x[ok], m2[ok], c=C[k], label=f'{p:g} Па')
        ax[1].plot(x[ok], ph[ok] - ph[ok][np.argmin(abs(x[ok] - 2.5))], c=C[k], label=f'{p:g} Па')
        r = ks(p, 20); xx = np.linspace(2.5, 8, 20)
        ax[1].plot(xx, -r['phase_slope_deg_per_mm'] * (xx - 2.5), ':', c=C[k], lw=1)
    ax[0].set_xlabel('x, мм'); ax[0].set_ylabel('$|R_{iz,2}|/R_{iz,0}$'); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    ax[0].set_title('глубина модуляции ионизации на 2ω')
    ax[1].set_xlabel('x, мм'); ax[1].set_ylabel('фаза 2ω-гармоники, град (от x = 2.5 мм)')
    ax[1].set_title('запаздывание: расчёт (—) и оценка (···)')
    for a in ax: a.axvline(2, c='gray', lw=0.8, ls=':')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig4_modulation.png'); plt.close(fig)

def fig_symmetry(mc_cc=None, mc_even=None):
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for k, (f, mf, lab) in enumerate(((PROD[13.3], mc_cc, 'противофаза (ВЧЕ-зазор)'), (EVEN, mc_even, 'в фазе (образец в плазме)'))):
        if not os.path.exists(f): continue
        S = L(f); H = S['H']; x = S['xc'] * 1e3
        ax[0].plot(x, np.abs(H[1]['n']) / H[0]['n'].real, c=C[k], label=lab)
        ax[1].plot(x, np.abs(H[1]['iz']) / H[0]['iz'].real, c=C[k], label=lab)
        if mf and os.path.exists(mf):
            R = L(mf); xb, M, Ms, A, tot = mc_harmonics(R)
            ax[0].plot(xb * 1e3, np.abs(M[1]['n']) / M[0]['n'].real, 'o', ms=2.5, c=C[k])
    ax[0].set_ylabel('$|n_1|/n_0$ (частота ω)'); ax[1].set_ylabel('$|R_{iz,1}|/R_{iz,0}$ (частота ω)')
    for a in ax: a.set_xlabel('x, мм (середина зазора: 8 мм)'); a.legend(fontsize=7); a.axvline(8, c='gray', lw=0.8, ls=':')
    ax[0].set_title('плотность на частоте ω'); ax[1].set_title('ионизация на частоте ω')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig5_symmetry.png'); plt.close(fig)

def fig_surface(sfile, mfile):
    S = L(sfile); H = S['H']; Nh = S['cfg']['Nh']; t = np.linspace(0, 2 * np.pi, 200)
    G = np.real(H[0]['surf'][0]) + sum(2 * np.real(H[n]['surf'][0] * np.exp(1j * n * t)) for n in range(1, Nh + 1))
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(t / (2 * np.pi), G / np.real(H[0]['surf'][0]), c=C[0], label=f'решатель ($N_t$ = {Nh})')
    if mfile and os.path.exists(mfile):
        R = L(mfile); S0 = R['surf'][0][:, 0]; S1 = np.roll(R['surf'][1][:, 0], -R['nph'] // 2) if R['mirror'] == 'cc' else R['surf'][1][:, 0]
        s = S0 + S1; tt = (np.arange(R['nph']) + 0.5) / R['nph']
        ax.step(tt, s / s.mean(), where='mid', c=C[1], label='МК')
    ax.axhline(0, c='k', lw=0.5); ax.set_xlabel('t / T'); ax.set_ylabel('$\\Gamma_e(t)/\\overline{\\Gamma_e}$')
    ax.set_title('поток электронов на поверхность'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig6_surface.png'); plt.close(fig)

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'model'): fig1_model()
    if what in ('all', 'maps'): fig_maps()
    if what in ('all', 'mod'): fig_modulation()
