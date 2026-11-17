"""Figures and tables for the e-e study (ee_sustain.py) -> results/ee_figs/"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import mix_figs as MF
from mix_figs import COL, LAB, GASES
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
OUT = 'results/ee_figs'; os.makedirs(OUT, exist_ok=True)
S = pickle.load(open('results/ee_sust.pkl', 'rb'))
NE = [1e16, 1e17, 1e18, 1e19]
PRESS = [13.3, 26.6, 133.0]

def costs(r, ng):
    P = sum(r['power'].values()); kiz = r['rates'].get('iz', 0); kd = r['rates'].get('diss_O2', 0)
    return dict(E_ion=P / (kiz * ng) if kiz > 0 else np.nan, E_O=P / (2 * kd * ng) if kd > 0 else np.nan, P=P,
                fr={k: v / P for k, v in r['power'].items()})

def get(g, p, f, ne):
    """working point dict: ne = 0 -> no e-e (mix scan), ne = 'M' -> Maxwellian limit, else e-e run"""
    if ne == 0:
        s = MF.sustain(g, p, f); ng = s['ng']; c = costs(s, ng)
        return dict(EN=s['EN'], Te=s['Te'], peak=s['peak'], f0=s['f0'], uc=s['uc'], **c)
    if ne == 'M':
        m = MF.maxwell_sustain(g, p)
        if m is None: return None
        P = sum(m['power'].values())
        return dict(EN=np.nan, Te=m['Te'], E_ion=m['E_ion'], E_O=m['E_O'], fr={k: v / P for k, v in m['power'].items()})
    k = (g, p, f, ne)
    if k not in S: return None
    v = S[k]; r = v['root']; c = costs(r, v['ng'])
    return dict(EN=r['EN'], Te=2 / 3 * r['u'], peak=r['Riz_peak'], f0=r['f0'], uc=v['uc'], neg=r['neg'], it=r['it'], g=v['g'],
                lnL=r['lnL'], mod_u=r.get('mod_u', np.nan), ee_cons=r.get('ee_cons', np.nan), **c)

def fig_te(f=13.56):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    x = np.array(NE)
    for k, p in enumerate((13.3, 133.0)):
        a = ax[k]
        for g in GASES:
            ys = [get(g, p, f, ne) for ne in NE]
            if any(y is None for y in ys): continue
            a.semilogx(x, [y['Te'] for y in ys], 'o-', c=COL[g], ms=3.5, label=LAB.get(g, g))
            a.plot([4e15], [get(g, p, f, 0)['Te']], '<', c=COL[g], ms=5)
            m = get(g, p, f, 'M')
            if m: a.plot([2.5e19], [m['Te']], '>', c=COL[g], ms=5)
        a.axhspan(1, 4, color='0.85', zorder=0, alpha=0.6)
        a.set_xlim(3e15, 3e19); a.set_xlabel('$n_e$, м$^{-3}$'); a.set_title(f'{p:g} Па, {f:g} МГц', fontsize=9)
    ax[0].set_ylabel('$T_e$ в рабочей точке, эВ'); ax[0].legend(fontsize=7, ncol=2, loc='upper right')
    fig.text(0.5, -0.04, '◀ слева — без e–e,  ▶ справа — максвелловский предел; серая полоса — измерения группы 1–4 эВ', ha='center', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_te_ne_{f:g}.png'); plt.close(fig)

def fig_en(f=13.56):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    x = np.array(NE)
    for k, p in enumerate((13.3, 133.0)):
        a = ax[k]
        for g in GASES:
            ys = [get(g, p, f, ne) for ne in NE]
            if any(y is None for y in ys): continue
            e0 = get(g, p, f, 0)['EN']
            a.semilogx(x, [y['EN'] / e0 for y in ys], 'o-', c=COL[g], ms=3.5, label=LAB.get(g, g))
        a.set_xlabel('$n_e$, м$^{-3}$'); a.set_title(f'{p:g} Па, {f:g} МГц', fontsize=9); a.axhline(1, c='k', lw=0.6)
    ax[0].set_ylabel('$E_{rms}/N$ с e–e  /  без e–e'); ax[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_en_ne_{f:g}.png'); plt.close(fig)

def fig_eedf(f=13.56):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, g in zip(ax, ('Ar', 'O2')):
        p = 13.3; cm = plt.cm.viridis
        z = get(g, p, f, 0); a.semilogy(z['uc'], z['f0'], c='k', lw=1.2, label='без e–e')
        for i, ne in enumerate(NE):
            y = get(g, p, f, ne)
            if y is None: continue
            a.semilogy(y['uc'], np.maximum(y['f0'], 1e-30), c=cm(0.15 + 0.22 * i), label=f'$n_e$ = 1e{int(np.log10(ne))}')
            if ne == NE[-1]:
                T = y['Te']; fM = np.exp(-y['uc'] / T); fM /= np.sum(np.sqrt(y['uc']) * np.diff(np.concatenate([[0], 0.5 * (y['uc'][1:] + y['uc'][:-1]), [2 * y['uc'][-1] - 0.5 * (y['uc'][-1] + y['uc'][-2])]])) * fM)
                a.semilogy(y['uc'], fM, c=cm(0.15 + 0.22 * i), ls=':', lw=1, label=f'Максвелл, {T:.1f} эВ')
        a.set_ylim(1e-9, 1); a.set_xlim(0, 40); a.set_xlabel('энергия, эВ'); a.set_ylabel('ФРЭЭ, эВ$^{-3/2}$')
        a.set_title(f'{LAB.get(g, g)}, {p:g} Па, {f:g} МГц, рабочая точка', fontsize=9); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_eedf_ee.png'); plt.close(fig)

def fig_cost(f=13.56):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.array(NE)
    for g in ('Ar', 'Ar+N2 90/10', 'Ar+O2 90/10', 'N2'):
        for p, ls in ((13.3, '-'), (133.0, '--')):
            ys = [get(g, p, f, ne) for ne in NE]
            if any(y is None for y in ys): continue
            ax[0].loglog(x, [y['E_ion'] for y in ys], 'o' + ls, c=COL[g], ms=3, label=f'{LAB.get(g, g)}, {p:g} Па')
    for g in ('O2', 'air', 'Ar+O2 90/10', 'Ar+O2 50/50'):
        for p, ls in ((13.3, '-'), (133.0, '--')):
            ys = [get(g, p, f, ne) for ne in NE]
            if any(y is None for y in ys): continue
            ax[1].semilogx(x, [y['E_O'] for y in ys], 'o' + ls, c=COL[g], ms=3, label=f'{LAB.get(g, g)}, {p:g} Па')
    ax[0].set_ylabel('эВ на ион'); ax[1].set_ylabel('эВ на атом O')
    for a in ax: a.set_xlabel('$n_e$, м$^{-3}$'); a.legend(fontsize=6.5, ncol=2)
    ax[0].set_title('цена иона', fontsize=9); ax[1].set_title('цена атома O', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_cost_ee.png'); plt.close(fig)

def table(f=13.56):
    lines = []
    for p in PRESS:
        for g in GASES:
            row = [get(g, p, f, 0)] + [get(g, p, f, ne) for ne in NE] + [get(g, p, f, 'M')]
            lines.append((g, p, row))
    return lines

if __name__ == '__main__':
    for f in (13.56, 1.76):
        if not any(k[2] == f for k in S): continue
        fig_te(f); fig_en(f)
        print(f'=== {f} MHz: Te (EN) at ne = 0, 1e16..1e19, Maxwell')
        for g, p, row in table(f):
            s = ' | '.join('—' if r is None else (f"{r['Te']:.2f}" + ('' if np.isnan(r['EN']) else f" ({r['EN']:.0f})")) for r in row)
            print(f'{g:12s} {p:5g}: {s}')
        print('diag: max |g|, min neg, max it, max mod_u, max |ee_cons|')
        rr = [get(*k) for k in S if k[2] == f]; rr = [r for r in rr if r]
        print(max(abs(r['g']) for r in rr), min(r['neg'] for r in rr), max(r['it'] for r in rr),
              np.nanmax([r['mod_u'] for r in rr]), np.nanmax([abs(r['ee_cons']) for r in rr]))
    fig_eedf(); fig_cost()
