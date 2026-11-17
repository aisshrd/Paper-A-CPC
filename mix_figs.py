"""Figures and tables for the working-gas scan (mix_scan.py)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import scipy.constants as co
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 160, 'savefig.bbox': 'tight'})
OUT = 'results/mix_figs'; os.makedirs(OUT, exist_ok=True)
D = pickle.load(open('results/mix_scan.pkl', 'rb')); R = D['res']; GASES = D['GASES']
COL = {'Ar': '#1f5fa8', 'N2': '#2e8b57', 'O2': '#c0392b', 'air': '#8e44ad', 'Ar+N2 90/10': '#5dade2', 'Ar+N2 50/50': '#82c785',
       'Ar+O2 90/10': '#f5a3a3', 'Ar+O2 50/50': '#d68910'}
LAB = {'air': 'воздух'}
K0 = {'Ar': 1.53, 'N2': 1.87, 'O2': 2.2}                 # reduced ion mobilities, cm^2/(V s), approximate (dominant parent ion)
N0 = 2.6868e25
LAM = 0.02                                               # effective diffusion length, m (assumption)

def Kion(comp, ng):
    return 1e-4 * N0 / ng / sum(x / K0[s] for s, x in comp.items())

QS = pickle.load(open('results/mix_qs.pkl', 'rb')) if os.path.exists('results/mix_qs.pkl') else {}
def hb_ok(g, p, f):
    return min(r['neg'] for r in R[(g, p, f)]['rows']) > -1e-6
def method(g, p, f):
    return 'ГБ' if (f != 1.76 or hb_ok(g, p, f) or (g, p, f) not in QS) else 'КС'
def series(g, p, f):
    d = R[(g, p, f)]
    if method(g, p, f) == 'КС':
        q = QS[(g, p, f)]; d = dict(d, rows=[dict(r, u_dc=r['u'], rates_dc=r['rates']) for r in q['rows']], uc=q['uc'])
    rows = d['rows']
    EN = np.array([r['EN'] for r in rows]); u = np.array([r['u'] for r in rows]); udc = np.array([r['u_dc'] for r in rows])
    rate = lambda k, dc=False: np.array([(r['rates_dc'] if dc else r['rates']).get(k, 0.0) for r in rows])
    return d, rows, EN, u, udc, rate

def sustain(g, p, f, lam=LAM):
    """E_rms/N where (k_iz - k_att) N = D_a / Lambda^2, D_a = K_i T_e (T_e = 2/3 <u>); log interpolation"""
    d, rows, EN, u, udc, rate = series(g, p, f); ng = d['ng']
    net = (rate('iz') - rate('att')) * ng; loss = Kion(GASES[g], ng) * (2 / 3 * u) / lam ** 2
    s = net - loss; i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if not len(i): return None
    i = i[0]; t = -s[i] / (s[i + 1] - s[i]); lg = np.log(EN[i]) + t * (np.log(EN[i + 1]) - np.log(EN[i]))
    def at(arr): return arr[i] + t * (arr[i + 1] - arr[i])
    r0, r1 = rows[i], rows[i + 1]
    keys = set(r0['rates']) | set(r1['rates'])
    rates = {k: np.exp(np.log(max(r0['rates'].get(k, 0), 1e-300)) + t * (np.log(max(r1['rates'].get(k, 0), 1e-300)) - np.log(max(r0['rates'].get(k, 0), 1e-300)))) for k in keys}
    pk = set(r0['power']) | set(r1['power']); power = {k: r0['power'].get(k, 0) + t * (r1['power'].get(k, 0) - r0['power'].get(k, 0)) for k in pk}
    f0 = r0['f0'] + t * (r1['f0'] - r0['f0'])
    return dict(EN=float(np.exp(lg)), u=float(at(u)), Te=float(2 / 3 * at(u)), rates=rates, power=power, peak=float(r0['Riz_peak'] + t * (r1['Riz_peak'] - r0['Riz_peak'])),
                f0=f0, uc=d['uc'], ng=ng, udc=float(at(udc)))

def fig_te():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for k, p in enumerate((13.3, 133.0)):
        a = ax[k]
        for g in GASES:
            d, rows, EN, u, udc, rate = series(g, p, 13.56)
            a.semilogx(EN, 2 / 3 * u, c=COL[g], label=LAB.get(g, g))
            a.semilogx(EN, 2 / 3 * udc, c=COL[g], ls=':', lw=0.8)
            s = sustain(g, p, 13.56)
            if s: a.plot(s['EN'], s['Te'], 'o', c=COL[g], ms=4)
        a.axhspan(1, 4, color='0.85', zorder=0, alpha=0.6)
        a.set_xlabel('$E_{rms}/N$, Тд'); a.set_ylabel('$T_e = \\frac{2}{3}\\langle\\varepsilon\\rangle$, эВ'); a.set_title(f'{p:g} Па, 13.56 МГц: пунктир — постоянное поле, точки — самоподдержание', fontsize=8)
        a.set_ylim(0, 9)
    ax[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(f'{OUT}/fig_te.png'); plt.close(fig)

def fig_kiz():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for k, p in enumerate((13.3, 133.0)):
        a = ax[k]
        for g in GASES:
            d, rows, EN, u, udc, rate = series(g, p, 13.56)
            a.loglog(EN, np.maximum(rate('iz'), 1e-22), c=COL[g], label=LAB.get(g, g))
            if rate('att').max() > 0: a.loglog(EN, np.maximum(rate('att'), 1e-22), c=COL[g], ls='--', lw=0.8)
            s = sustain(g, p, 13.56)
            if s: a.plot(s['EN'], s['rates']['iz'], 'o', c=COL[g], ms=4)
        a.set_ylim(1e-19, 5e-14); a.set_xlabel('$E_{rms}/N$, Тд'); a.set_ylabel('коэффициент скорости, м$^3$/с')
        a.set_title(f'{p:g} Па, 13.56 МГц: ионизация (—), прилипание (– –)', fontsize=9)
    ax[1].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(f'{OUT}/fig_kiz.png'); plt.close(fig)

PGRP = [('elastic', 'упругие', '#bbbbbb'), ('rotation', 'вращение', '#e0c9a6'), ('vib', 'колебания', '#7fb3d5'),
        ('O2(a1Dg)', 'O$_2$(a$^1\\Delta_g$)', '#f8c471'), ('O2(b1Sg)', 'O$_2$(b$^1\\Sigma_g^+$)', '#f0b27a'), ('diss_O2', 'диссоциация O$_2$', '#c0392b'),
        ('el_O2', 'прочие уровни O$_2$', '#e6b0aa'), ('N2(A)', 'N$_2$(A$^3\\Sigma_u^+$)', '#27ae60'), ('el_N2', 'прочие уровни N$_2$', '#a9dfbf'),
        ('exc_Ar', 'возбуждение Ar', '#5dade2'), ('iz', 'ионизация', '#1b2631')]

def fig_power(p=13.3, f=13.56):
    gs = [g for g in GASES if sustain(g, p, f)]
    fig, ax = plt.subplots(figsize=(8.6, 3.4)); left = np.zeros(len(gs))
    tot = np.array([sum(sustain(g, p, f)['power'].values()) for g in gs])
    for key, lab, c in PGRP:
        v = np.array([sustain(g, p, f)['power'].get(key, 0.0) for g in gs]) / tot
        if v.max() < 1e-3: continue
        ax.barh(range(len(gs)), v, left=left, color=c, label=lab, edgecolor='white', lw=0.5); left += v
    ax.set_yticks(range(len(gs))); ax.set_yticklabels([LAB.get(g, g) for g in gs]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel('доля мощности, переданной электронами'); ax.set_title(f'Куда уходит энергия электронов: {p:g} Па, {f:g} МГц, самоподдержание', fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.3)); fig.tight_layout(); fig.savefig(f'{OUT}/fig_power.png'); plt.close(fig)

def fig_eedf(p=13.3, f=13.56):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for k, gs in enumerate((['Ar', 'N2', 'O2', 'air'], ['Ar', 'Ar+N2 90/10', 'Ar+N2 50/50', 'Ar+O2 90/10', 'Ar+O2 50/50'])):
        a = ax[k]
        for g in gs:
            s = sustain(g, p, f)
            if not s: continue
            uc = s['uc']; f0 = np.maximum(s['f0'], 1e-30)
            a.semilogy(uc, f0, c=COL[g], label=f'{LAB.get(g, g)}: $T_e$ = {s["Te"]:.1f} эВ')
            T = s['Te']; fm = 2 / np.sqrt(np.pi) * T ** -1.5 * np.exp(-uc / T)
            a.semilogy(uc, fm, c=COL[g], ls=':', lw=0.8)
        a.set_xlim(0, 30); a.set_ylim(1e-7, 2); a.set_xlabel('энергия, эВ'); a.set_ylabel('ФРЭЭ, эВ$^{-3/2}$'); a.legend(fontsize=7)
        a.set_title(f'{p:g} Па, {f:g} МГц, самоподдержание (пунктир — Максвелл)', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_eedf.png'); plt.close(fig)

def fig_freq():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    gs = list(GASES); x = np.arange(len(gs)); wdt = 0.2
    for k, (p, f, c) in enumerate(((13.3, 1.76, '#c0392b'), (13.3, 13.56, '#1f5fa8'), (133.0, 1.76, '#f5a3a3'), (133.0, 13.56, '#85c1e9'))):
        pk = [sustain(g, p, f)['peak'] if sustain(g, p, f) else np.nan for g in gs]
        EN = [sustain(g, p, f)['EN'] if sustain(g, p, f) else np.nan for g in gs]
        ax[0].bar(x + (k - 1.5) * wdt, pk, wdt, color=c, label=f'{p:g} Па, {f:g} МГц')
        ax[1].bar(x + (k - 1.5) * wdt, EN, wdt, color=c, label=f'{p:g} Па, {f:g} МГц')
    for a in ax: a.set_xticks(x); a.set_xticklabels([LAB.get(g, g) for g in gs], rotation=30, ha='right', fontsize=7)
    ax[0].set_ylabel('пик / среднее скорости ионизации за период'); ax[0].set_title('пульсации ионизации на 2ω', fontsize=9); ax[0].legend(fontsize=6)
    ax[1].set_ylabel('$E_{rms}/N$ самоподдержания, Тд'); ax[1].set_yscale('log'); ax[1].set_title('поле самоподдержания ($\\Lambda$ = 2 см)', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_freq.png'); plt.close(fig)

def table():
    rows = []
    for g in GASES:
        for p in D['PRESS']:
            for f in D['FREQ']:
                s = sustain(g, p, f)
                if not s: rows.append(dict(g=g, p=p, f=f)); continue
                P = sum(s['power'].values()); ng = s['ng']; kiz = s['rates']['iz']
                kO = s['rates'].get('diss_O2', 0.0)
                rows.append(dict(g=g, p=p, f=f, method=method(g, p, f), EN=s['EN'], Te=s['Te'], Te_dc=2 / 3 * s['udc'], kiz=kiz, katt=s['rates'].get('att', 0.0), kdissO2=kO,
                                 kA=s['rates'].get('N2(A)', 0.0), ka1=s['rates'].get('O2(a1Dg)', 0.0),
                                 E_ion=P / (kiz * ng), E_O=(P / (2 * kO * ng) if kO > 0 else np.nan), peak=s['peak'],
                                 fr_vib=s['power'].get('vib', 0) / P, fr_iz=s['power'].get('iz', 0) / P, fr_el=s['power'].get('elastic', 0) / P,
                                 fr_diss=s['power'].get('diss_O2', 0) / P))
    return rows

if __name__ == '__main__':
    rows = table(); pickle.dump(rows, open(f'{OUT}/table.pkl', 'wb'))
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) and abs(v) > 1e-3 else (f'{v:.2e}' if isinstance(v, float) else v)) for k, v in r.items()})
    fig_te(); fig_kiz(); fig_power(); fig_eedf(); fig_freq()

# ---------------------------------------------------------------- Maxwellian limit (strong electron-electron collisions)
from ekin_mix import MixtureKinetics
from mix_scan import SP, group as _group
_KM = {}
def maxwell_point(g, p, Te):
    key = (g, p)
    if key not in _KM:
        comp = GASES[g]; ng = p / (co.k * 300.0)
        _KM[key] = MixtureKinetics([(SP[s], x) for s, x in comp.items()], umax=60., nu=300, L=1, Nh=0, ng=ng)
    K = _KM[key]; uc = K.uc
    f0 = 2 / np.sqrt(np.pi) * Te ** -1.5 * np.exp(-uc / Te); f0 /= np.sum(K.w * f0)
    f = np.zeros((1, 2, K.nu), complex); f[0, 0] = f0
    comp = GASES[g]; Rg = {}
    for kind, sn, prod, th, r in K.rates(f):
        gg = _group(kind, sn, prod); Rg[gg] = Rg.get(gg, 0.0) + comp[sn] * r
    Pg = {}
    for k, v in K.power_split(f).items():
        if k in ('elastic', 'rotation'): Pg[k] = v; continue
        kind, sn, prod = k.split(':', 2); gg = _group(kind, sn, prod); Pg[gg] = Pg.get(gg, 0.0) + v
    return Rg, Pg, K.ng

def maxwell_sustain(g, p, lam=LAM):
    Ts = np.linspace(0.5, 8, 151); s = []
    for T in Ts:
        Rg, Pg, ng = maxwell_point(g, p, T)
        s.append((Rg.get('iz', 0) - Rg.get('att', 0)) * ng - Kion(GASES[g], ng) * T / lam ** 2)
    s = np.array(s); i = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    if not len(i): return None
    i = i[0]; T = Ts[i] - s[i] * (Ts[i + 1] - Ts[i]) / (s[i + 1] - s[i])
    Rg, Pg, ng = maxwell_point(g, p, T); P = sum(Pg.values())
    return dict(Te=T, rates=Rg, power=Pg, E_ion=P / (Rg['iz'] * ng), E_O=(P / (2 * Rg['diss_O2'] * ng) if Rg.get('diss_O2', 0) > 0 else np.nan),
                fr_vib=Pg.get('vib', 0) / P, fr_diss=Pg.get('diss_O2', 0) / P, fr_iz=Pg.get('iz', 0) / P)
