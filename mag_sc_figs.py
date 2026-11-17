"""Self-consistent magnetised ICP: profiles and table -> results/mag_figs/fig_sc_*.png"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
OUT = 'results/mag_figs'
CASES = [(0, 'results/discharge_mag_B00_Bext0.7_ne240.pkl'), (10, 'results/discharge_mag_B010_Bext0.7.pkl'),
         (20, 'results/discharge_mag_B020_Bext0.7.pkl'), (40, 'results/discharge_mag_B040_Bext0.7_ne240.pkl')]
D = [(b, pickle.load(open(p, 'rb'))) for b, p in CASES if os.path.exists(p)]
cm = plt.cm.viridis
def col(b): return cm(0.05 + 0.85 * b / 40)

def fig():
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
    for b, d in D:
        X = d['X']; lab = f'$B_0$ = {b} Гс'
        ax[0, 0].plot(X, d['Te'], 'o-', ms=2.5, c=col(b), label=lab)
        w = d['wdisk']; ax[0, 1].plot(X, d['ne'] / np.sum(w * d['ne']), 'o-', ms=2.5, c=col(b))
        ax[1, 0].plot(X, d['phi'] - d['phi'][0], 'o-', ms=2.5, c=col(b))
        ax[1, 1].plot(X, 2 * np.abs(d['E'][2]), 'o-', ms=2.5, c=col(b), label=f'$|E_\\theta|$, {b} Гс')
        if d.get('Er') is not None: ax[1, 1].plot(X, 2 * np.abs(d['Er'][2]), '--', c=col(b))
    ax[0, 0].set_ylabel('$T_e$, эВ'); ax[0, 1].set_ylabel('$n_e/\\langle n_e\\rangle$'); ax[1, 0].set_ylabel('$\\varphi - \\varphi(R)$, В')
    ax[1, 1].set_ylabel('амплитуда поля, В/м (пунктир — $E_r$)'); ax[0, 0].legend(fontsize=7); ax[1, 1].set_yscale('log'); ax[1, 1].set_ylim(1, 500)
    for a in ax.ravel(): a.set_xlabel('$r/R$')
    fig.suptitle('Самосогласованный ИСП с постоянным $B_0$: 13.3 Па, 13.56 МГц, $n_0$ = 10$^{17}$ м$^{-3}$, ток индуктора как при $B_0$ = 0 ($B_{ВЧ}$ = 0.7 Гс)', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_sc_profiles.png'); plt.close(fig)

if __name__ == '__main__':
    fig()
    for b, d in D:
        Te = d['Te']; X = d['X']
        print(f"B0={b:3d} G: Te wall/max/axis {Te[0]:.2f}/{Te.max():.2f}/{Te[-1]:.2f} (max at r/R={X[np.argmax(Te)]:.2f}), phi well {d['phi'][-1]-d['phi'][0]:.2f} V, "
              f"ne ax/wall {d['ne'][-1]/d['ne'][0]:.1f}, |E_th| wall {2*abs(d['E'][2][0]):.0f} V/m, max|Er| {2*np.max(np.abs(d['Er'][2])) if d.get('Er') is not None else 0:.0f} V/m, "
              f"balance {d['balance']:.3f}, conv {d['converged']}, last dE {d['hist'][-1][1]:.4f}")
