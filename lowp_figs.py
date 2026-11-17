"""Figures for the low-pressure ECR study -> results/mag_figs/fig_ecr_*.png"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import scipy.constants as co
from ekin import load_argon, GAMMA
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
S = pickle.load(open('results/lowp_scan.pkl', 'rb')); xs = load_argon('data/Ar_LXCat.txt')
om = 2 * np.pi * 13.56e6; Bres = om * co.m_e / co.e * 1e4
COL = {2.0: '#c0392b', 5.0: '#d68910', 13.3: '#1f5fa8'}
def tot(v, q): return np.sum(v['wdisk'] * q)
def nu_m(p, u=3.0):
    sig = np.interp(u, xs['elastic'][:, 0], xs['elastic'][:, 1]); return p / (co.k * 300) * sig * GAMMA * np.sqrt(u)
def cold(B, nu):
    wc = co.e * B * 1e-4 / co.m_e
    s = 0.5 * (nu / (nu ** 2 + (om - wc) ** 2) + nu / (nu ** 2 + (om + wc) ** 2)); return s / (nu / (nu ** 2 + om ** 2))

def fig_power():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    Bf = np.linspace(0, 20, 300)
    for p in (2.0, 5.0, 13.3):
        ks = sorted([k for k in S if k[0] == p and k[2] == 'Er0' and k[3] == 1], key=lambda k: k[1]); b0 = S[(p, 0.0, 'Er0', 1)]; P0 = tot(b0, b0['P'])
        ax[0].plot([k[1] for k in ks], [tot(S[k], S[k]['P']) / P0 for k in ks], 'o-', c=COL[p], ms=3.5, label=f'{p:g} Па, кинетика')
        ax[0].plot(Bf, cold(Bf, nu_m(p)), ':', c=COL[p], lw=1, label=f'{p:g} Па, холодная модель ($\\nu_m$ при 3 эВ)')
        ks = sorted([k for k in S if k[0] == p and k[2] == 'jr0' and k[3] == 1], key=lambda k: k[1])
        if ks: ax[1].plot([0] + [k[1] for k in ks], [1] + [tot(S[k], S[k]['P']) / P0 for k in ks], 's-', c=COL[p], ms=3.5, label=f'{p:g} Па, кинетика')
    for p, f in ((13.3, 'results/mag_prescribed.pkl'),):
        M = pickle.load(open(f, 'rb')); ks = sorted([k for k in M if k[1] and k[2] == 1 and k[0] <= 20], key=lambda k: k[0]); P0 = np.sum(M[(0.0, True, 1)]['wdisk'] * M[(0.0, True, 1)]['P'])
        ax[1].plot([k[0] for k in ks], [np.sum(M[k]['wdisk'] * M[k]['P']) / P0 for k in ks], 's-', c=COL[p], ms=3.5, label='13.3 Па, кинетика')
    ax[1].axhline(1, c='k', lw=0.8, ls=':', label='холодная модель: нет зависимости от $B_0$')
    for a in ax: a.axvline(Bres, c='0.5', lw=0.8, ls='--'); a.set_xlabel('$B_0$, Гс')
    ax[0].text(Bres + 0.3, 0.05, '$\\omega_c=\\omega$', fontsize=8, color='0.4')
    ax[0].set_ylabel('поглощённая мощность / при $B_0$ = 0'); ax[0].set_title('$E_r$ = 0: радиальный ток замыкается через торцы', fontsize=9)
    ax[1].set_title('$j_r$ = 0: изолированный столб, $E_r$ из условия $j_r$ = 0', fontsize=9)
    ax[0].legend(fontsize=6.5); ax[1].legend(fontsize=6.5); ax[0].set_ylim(0, 1.8)
    fig.tight_layout(); fig.savefig('results/mag_figs/fig_ecr_power.png'); plt.close(fig)

def fig_te():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    cm = plt.cm.plasma
    for a, (mode, title) in zip(ax, (('Er0', '2 Па, $E_r$ = 0'), ('jr0', '2 Па, $j_r$ = 0'), ('Er0', '13.3 Па, $E_r$ = 0'))):
        p = 13.3 if '13.3' in title else 2.0
        Bs = [0, 3, 4.84, 7, 10, 20] if mode == 'jr0' else [0, 2, 4, 4.84, 7, 10, 20]
        for i, b in enumerate(Bs):
            k = (p, float(b), 'Er0' if b == 0 else mode, 1)
            if k not in S: continue
            a.plot(S[k]['X'], S[k]['Te'], 'o-', ms=2, c=cm(0.05 + 0.85 * i / (len(Bs) - 1)), label=f'{b:g} Гс')
        a.set_title(title, fontsize=9); a.set_xlabel('$r/R$')
    ax[0].set_ylabel('$T_e$, эВ'); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig('results/mag_figs/fig_ecr_te.png'); plt.close(fig)

if __name__ == '__main__':
    fig_power(); fig_te()
    for p in (2.0, 5.0, 13.3): print(p, 'nu_m(3 eV) = %.2e s^-1, nu/omega = %.2f' % (nu_m(p), nu_m(p) / om))
    print('B_res = %.2f G' % Bres)
