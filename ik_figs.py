"""Figures and table for the ion kinetics (test-particle runs on the self-consistent fields, ion_kin.py)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
C = ['#1f5fa8', '#c0392b', '#2e8b57', '#8e44ad', '#d68910', '#555555']
OUT = 'results/ik_figs'; os.makedirs(OUT, exist_ok=True)
L = lambda f: pickle.load(open(f, 'rb'))
def F(p, V, n=1e16):
    Nh = 1 if V == 0 else (3 if V >= 12 else 2)
    fn = f'results/ionkin_p{p:g}_n{n:.0e}_V{V:g}_N{Nh}_e80_m48_prod.pkl'
    return fn if os.path.exists(fn) else None

def hist(E, bins):
    h, _ = np.histogram(E, bins=bins); return h / np.diff(bins) / len(E)

def fig_verify():
    d = L(F(13.3, 0)); fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
    a = ax[0]; dt = d['det']; b = dt['ebins']; ec = 0.5 * (b[1:] + b[:-1])
    a.plot(ec, dt['fE'], c=C[0], lw=1.6, label='решатель (полная энергия)')
    a.step(ec, hist(d['cx_avg']['E'], b), where='mid', c=C[1], lw=0.9, label='МК, та же физика')
    a.step(ec, hist(d['ph_avg']['E'], b), where='mid', c=C[2], lw=0.9, label='МК, Phelps + тепловые атомы')
    a.axvline(d['fluid']['Ei'], c='k', ls='--', lw=0.9); a.text(d['fluid']['Ei'] + 0.2, a.get_ylim()[1] * 0.8 if False else 0.0, '')
    a.axvline(dt['Emean'], c=C[0], ls=':', lw=1)
    a.set_xlabel('энергия иона на образце, эВ'); a.set_ylabel('доля потока, 1/эВ'); a.set_xlim(0, d['Vdc'] + 1)
    a.set_title(f'13.3 Па, без ВЧ: пунктир — жидкость ({d["fluid"]["Ei"]:.1f} эВ), точки — среднее ({dt["Emean"]:.1f} эВ)', fontsize=8)
    a.legend(fontsize=7)
    a = ax[1]; x = d['fluid']['xc'] * 1e3
    a.plot(x, d['fluid']['ni'], c='k', ls='--', label='холодная жидкость')
    a.plot(x, d['det']['n'], c=C[0], label='решатель (холодное рождение)')
    a.plot(x, d['cx_avg']['n'], c=C[1], label='МК, σ = const, тепловые атомы')
    a.plot(x, d['ph_avg']['n'], c=C[2], label='МК, Phelps')
    a.set_xlabel('x, мм'); a.set_ylabel('$n_i$, м$^{-3}$ (один и тот же поток)'); a.legend(fontsize=7); a.set_title('плотность ионов при одном потоке', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_ik_verify.png'); plt.close(fig)

def fig_pressure():
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for k, p in enumerate((13.3, 40, 133)):
        fn = F(p, 0)
        if not fn: continue
        d = L(fn); b = np.linspace(0, d['Vdc'] + 1, 121); ec = 0.5 * (b[1:] + b[:-1])
        ax[0].step(ec, hist(d['ph_avg']['E'], b), where='mid', c=C[k], label=f'{p:g} Па: среднее {d["ph_avg"]["Emean"]:.1f} эВ, $eV_{{dc}}$ = {d["Vdc"]:.1f} эВ')
        ax[0].axvline(d['Vdc'], c=C[k], ls=':', lw=0.8)
        ab = np.linspace(0, 90, 46); ac = 0.5 * (ab[1:] + ab[:-1])
        h, _ = np.histogram(np.degrees(d['ph_avg']['ang']), bins=ab); ax[1].step(ac, h / h.sum() / 2, where='mid', c=C[k], label=f'{p:g} Па: среднее {d["ph_avg"]["angmean"]:.0f}°')
    ax[0].set_yscale('log'); ax[0].set_xlabel('энергия иона на образце, эВ'); ax[0].set_ylabel('доля потока, 1/эВ'); ax[0].legend(fontsize=7)
    ax[0].set_title('распределение по энергиям (без ВЧ, Phelps)', fontsize=9)
    ax[1].set_yscale('log'); ax[1].set_xlabel('угол к нормали, град'); ax[1].set_ylabel('доля потока, 1/град'); ax[1].legend(fontsize=7)
    ax[1].set_title('распределение по углам', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_ik_pressure.png'); plt.close(fig)

def fig_rf():
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for k, (p, V) in enumerate(((13.3, 8), (13.3, 12), (40, 8))):
        fn = F(p, V)
        if not fn: continue
        d = L(fn); b = np.linspace(0, d['Vdc'] + V + 2, 121); ec = 0.5 * (b[1:] + b[:-1])
        ax[0].step(ec, hist(d['ph_rf']['E'], b), where='mid', c=C[k], label=f'{p:g} Па, $V_{{rf}}$ = {V} В, поле $\\varphi(x,t)$')
        ax[0].step(ec, hist(d['ph_avg']['E'], b), where='mid', c=C[k], lw=0.7, ls=':')
    ax[0].set_yscale('log'); ax[0].set_xlabel('энергия иона на образце, эВ'); ax[0].set_ylabel('доля потока, 1/эВ')
    ax[0].set_title('ВЧ: сплошные — с колебаниями поля, точки — среднее поле', fontsize=8); ax[0].legend(fontsize=7)
    fn = F(13.3, 12) or F(13.3, 8)
    if fn:
        d = L(fn); r = d['ph_rf']; hb = np.linspace(0, 1, 41); eb = np.linspace(0, d['Vdc'] + 14, 61)
        H, _, _ = np.histogram2d(r['ph'], r['E'], bins=(hb, eb)); H = H / H.sum(axis=1, keepdims=True).clip(1)
        im = ax[1].pcolormesh(eb, hb, np.log10(H + 1e-5), cmap='viridis', vmin=-4, vmax=-0.5, shading='auto')
        ax[1].set_xlabel('энергия, эВ'); ax[1].set_ylabel('фаза прихода t/T'); plt.colorbar(im, ax=ax[1], shrink=0.85, label='lg доли')
        ax[1].set_title(f'{d["cfg"]["p"]:g} Па, $V_{{rf}}$ = {d["cfg"]["Vrf"]:g} В: энергия по фазе прихода', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_ik_rf.png'); plt.close(fig)

def table():
    rows = []
    for fn in sorted(glob.glob('results/ionkin_*_prod.pkl')):
        d = L(fn); c = d['cfg']
        r = dict(p=c['p'], n=c['nD'], V=c['Vrf'], Vdc=d['Vdc'], lam=d['lam'] * 1e3, Ei_fluid=d['fluid']['Ei'], E_det=d['det']['Emean'],
                 E_cx=d['cx_avg']['Emean'], E_ph=d['ph_avg']['Emean'], f10=d['ph_avg']['frac10'], f15=d['ph_avg']['frac15'], ang=d['ph_avg']['angmean'],
                 nD_ratio=d['ph_avg']['n'][-1] / d['fluid']['ni'][-1])
        if 'ph_rf' in d: r.update(E_rf=d['ph_rf']['Emean'], f10rf=d['ph_rf']['frac10'], f15rf=d['ph_rf']['frac15'], Emax_rf=np.percentile(d['ph_rf']['E'], 99))
        rows.append(r)
    return rows

if __name__ == '__main__':
    rows = table()
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    pickle.dump(rows, open(f'{OUT}/table.pkl', 'wb'))
    if F(13.3, 0): fig_verify()
    fig_pressure(); fig_rf()
