"""Figures and tables for the T_v study (vib_sustain.py) -> results/vib_figs/"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mix_figs import COL, LAB
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
OUT = 'results/vib_figs'; os.makedirs(OUT, exist_ok=True)
S = pickle.load(open('results/vib_sust.pkl', 'rb'))
GV = ['N2', 'air', 'Ar+N2 90/10', 'Ar+N2 50/50']; PRESS = [13.3, 26.6, 133.0]
TV = sorted({k[2] for k in S})

def get(g, p, Tv, ne=0.0):
    v = S.get((g, p, Tv, ne))
    if v is None: return None
    r = v['root']; ng = v['ng']; P = sum(r['power'].values())
    kz = r['rates'].get('iz', 0)
    return dict(EN=r['EN'], Te=2 / 3 * r['u'], f0=r['f0'], uc=v['uc'], P=P, Pvib=r['power'].get('vib', 0.0),
                fr_vib=r['power'].get('vib', 0.0) / P, E_ion=P / (kz * ng), kA=r['rates'].get('N2(A)', 0.0), kiz=kz,
                kel=r['rates'].get('el_N2', 0.0), E_O=(P / (2 * r['rates']['diss_O2'] * ng) if r['rates'].get('diss_O2', 0) > 0 else np.nan),
                neg=r['neg'], g=v['g'], peak=r['Riz_peak'])

def fig_te():
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for a, p in zip(ax, PRESS):
        for g in GV:
            y = [get(g, p, t) for t in TV]
            a.plot(TV, [r['Te'] for r in y], 'o-', c=COL[g], ms=3.5, label=LAB.get(g, g))
        a.axhspan(1, 4, color='0.85', zorder=0, alpha=0.6); a.set_title(f'{p:g} Па, 13.56 МГц', fontsize=9); a.set_xlabel('$T_v$, К')
    ax[0].set_ylabel('$T_e$ в рабочей точке, эВ'); ax[0].legend(fontsize=7); ax[0].set_ylim(0.5, 4.2)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_te_tv.png'); plt.close(fig)

def fig_eedf():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for a, p in zip(ax, (13.3, 133.0)):
        cm = plt.cm.inferno
        for i, t in enumerate([300., 2000., 4000., 6000.]):
            r = get('N2', p, t)
            if r: a.semilogy(r['uc'], np.maximum(r['f0'], 1e-30), c=cm(0.1 + 0.22 * i), label=f'$T_v$ = {t:.0f} К, $T_e$ = {r["Te"]:.2f} эВ')
        a.set_xlim(0, 25); a.set_ylim(1e-12, 3); a.set_xlabel('энергия, эВ'); a.set_ylabel('ФРЭЭ, эВ$^{-3/2}$')
        a.set_title(f'N$_2$, {p:g} Па, 13.56 МГц, рабочая точка', fontsize=9); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_eedf_tv.png'); plt.close(fig)

def fig_balance():
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for g in GV:
        for p, ls in ((13.3, '-'), (133.0, '--')):
            y = [get(g, p, t) for t in TV]
            ax[0].plot(TV, [100 * r['fr_vib'] for r in y], 'o' + ls, c=COL[g], ms=3, label=f'{LAB.get(g, g)}, {p:g} Па')
            ax[1].semilogy(TV, [r['E_ion'] for r in y], 'o' + ls, c=COL[g], ms=3)
            ax[2].plot(TV, [r['kA'] / y[0]['kA'] for r in y], 'o' + ls, c=COL[g], ms=3)
    ax[0].set_ylabel('доля мощности на колебания (нетто), %'); ax[1].set_ylabel('эВ на ион'); ax[2].set_ylabel('$k$(N$_2$(A)) / $k$($T_v$ = 300 К)')
    for a in ax: a.set_xlabel('$T_v$, К')
    ax[0].legend(fontsize=6, ncol=1); fig.tight_layout(); fig.savefig(f'{OUT}/fig_balance_tv.png'); plt.close(fig)

if __name__ == '__main__':
    fig_te(); fig_eedf(); fig_balance()
    for p in PRESS:
        print(f'=== {p} Pa: Te (EN) vs Tv =', TV)
        for g in GV:
            print(f'{g:12s}', ' | '.join(f"{r['Te']:.2f} ({r['EN']:.0f})" for r in (get(g, p, t) for t in TV)))
    print('fr_vib, E_ion, kA ratio, kiz ratio at 300 / 4000 / 6000 K')
    for p in PRESS:
        for g in GV:
            a, b, c = get(g, p, 300.), get(g, p, 4000.), get(g, p, 6000.)
            print(f'{g:12s} {p:5g}: vib {100*a["fr_vib"]:.0f}/{100*b["fr_vib"]:.0f}/{100*c["fr_vib"]:.0f} %  Eion {a["E_ion"]:.0f}/{b["E_ion"]:.0f}/{c["E_ion"]:.0f}  kA x{b["kA"]/a["kA"]:.2f}/x{c["kA"]/a["kA"]:.2f}  kiz x{b["kiz"]/a["kiz"]:.2f}/x{c["kiz"]/a["kiz"]:.2f}  E_O {a["E_O"]:.1f}/{c["E_O"]:.1f}')
    rr = [get(*k) for k in S]; print('diag max|g|', max(abs(r['g']) for r in rr), 'min neg', min(r['neg'] for r in rr))
