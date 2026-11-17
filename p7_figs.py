"""Figures and tables for Paper 7 (working gases in the cylindrical nonlocal solver) -> papers/p7/Fig1..Fig5, table printed.
usage: python3 p7_figs.py [fig1 fig2 fig3 fig4 fig5 table]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, os, sys, glob, matplotlib, scipy.constants as co
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ekin_cyl_mix import mixture_xs
from mix_scan import group
OUT = os.environ.get('P7_OUT', 'papers/p7'); os.makedirs(OUT, exist_ok=True); which = sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'table']
TAG = os.environ.get('P7_TAG', '_opt'); P7_AR = os.environ.get('P7_AR', 'results/discharge_te_B0.76_n1e+17_L1_m16_o2.pkl'); P7_MC_N2 = os.environ.get('P7_MC_N2', 'results/mc_n2_disc_B2.4.pkl')
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 1.1, 'ps.fonttype': 42, 'pdf.fonttype': 42, 'axes.grid': True, 'grid.color': '0.85', 'grid.linewidth': 0.5, 'lines.markersize': 3.5, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'cm', 'legend.framealpha': 1.0, 'legend.edgecolor': '0.7'})
W = 17 / 2.54; R = 0.1; P = 13.3
def save(fig, name):
    fig.tight_layout(pad=0.4); fig.savefig(f'{OUT}/{name}.eps'); fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png', dpi=200); plt.close(fig)
# ---------------- gases: (key in file names, mixture, label, 0D name, line style, colour)
GAS = [('Ar', [('Ar', 1.0)], 'Ar', 'Ar', '-', 'k'),
       ('Ar0.9_N20.1', [('Ar', 0.9), ('N2', 0.1)], r'Ar/N$_2$ 90/10', 'Ar+N2 90/10', '--', '0.3'),
       ('Ar0.5_N20.5', [('Ar', 0.5), ('N2', 0.5)], r'Ar/N$_2$ 50/50', 'Ar+N2 50/50', '-.', '0.5'),
       ('N2', [('N2', 1.0)], r'N$_2$', 'N2', '-', '0.6'),
       ('N20.79_O20.21', [('N2', 0.79), ('O2', 0.21)], 'air', 'air', ':', '0.4')]
def load_point(key):
    """converged point with beta closest to 1 (argon: the conjfix reference)"""
    if key == 'Ar':
        d = pickle.load(open(P7_AR, 'rb')); d['mix'] = [('Ar', 1.0)]; d['ng'] = 3.21e21; d['p'] = P; d['wdisk'] = pickle.load(open('results/discharge_mag_B00_Bext0.77_fix.pkl', 'rb'))['wdisk']; return d
    fs = [g for g in glob.glob(f'results/discharge_mix_{key}_p{P:g}_B*{TAG}.pkl') if 'ckpt' not in g]
    if TAG == '_opt': fs += glob.glob(f'results/discharge_mix_{key}_p{P:g}_B*_opt2.pkl')
    if not fs: return None
    ds = [pickle.load(open(f, 'rb')) for f in fs]; return min(ds, key=lambda d: abs(d['balance'] - 1))
def rates_from_F0(d, xs):
    """per-electron rates (s^-1) of every process at every node from the stored isotropic part"""
    ng = d['ng']; u = np.maximum(d['ukin'], 0); v = np.sqrt(2 * co.e * u / co.m_e); w = d['w']; F = d['F0']; ne = np.sum(w * F, axis=0)
    sig = lambda s, x: np.interp(x, s[:, 0], s[:, 1], left=0.0, right=s[-1, 1])
    out = {}
    for th, s, name in xs['inel']: out[name] = np.sum(w * F * ng * sig(s, u) * v * (u > th), axis=0) / ne
    izs = xs['iz'] if isinstance(xs['iz'], list) else [xs['iz']]
    out['iz'] = sum(np.sum(w * F * ng * sig(s, u) * v * (u > th), axis=0) for th, s in izs) / ne
    out['att'] = sum((np.sum(w * F * ng * sig(s, u) * v, axis=0) for s in xs.get('att', [])), np.zeros_like(ne)) / ne
    return out, ne
def budget(d):
    """volume-integrated power fractions by group, cost per ion, mean quantities"""
    xs = mixture_xs(d['mix']); rt, ne = rates_from_F0(d, xs); w = d['wdisk'] * d['ne']; Ne = np.sum(w)
    Pg = {}; Rg = {}
    for th, s, name in xs['inel']:
        g = group('exc', name.split(':')[0], name.split(':', 1)[1]); Pg[g] = Pg.get(g, 0) + np.sum(w * rt[name]) * th; Rg[g] = Rg.get(g, 0) + np.sum(w * rt[name])
    izs = xs['iz'] if isinstance(xs['iz'], list) else [xs['iz']]; thiz = min(t for t, _ in izs)
    Niz = np.sum(w * rt['iz']); Natt = np.sum(w * rt['att']); Pg['iz'] = Niz * thiz; Pt = sum(Pg.values())
    return dict(frac={k: v / Pt for k, v in Pg.items()}, cost=Pt / max(Niz - Natt, 1e-300), cost_gross=Pt / Niz, att_iz=Natt / max(Niz, 1e-300),
                eV_per_O=(Pt / (2 * Rg['diss_O2']) if 'diss_O2' in Rg else None), nuiz=(Niz - Natt) / Ne, Te_n=np.sum(w * d['Te']) / Ne,
                Ew=2 * abs(d['E'][2][0]), EN=2 * abs(d['E'][2][0]) / np.sqrt(2) / d['ng'] * 1e21)
Z = pickle.load(open('results/p7_zero_d.pkl', 'rb')) if os.path.exists('results/p7_zero_d.pkl') else {}
def zero_d(g0, lam):
    r = Z.get((g0, P, round(lam, 5)))
    if r is None: return None
    Pt = sum(r['power'].values()); net = r['rates'].get('iz', 0) - r['rates'].get('att', 0)
    return dict(EN=r['EN'], Te=r['Te'], frac={k: v / Pt for k, v in r['power'].items()}, cost=Pt / (net * r['ng']),
                eV_per_O=(Pt / (2 * r['rates']['diss_O2'] * r['ng']) if 'diss_O2' in r['rates'] else None), f0=r['f0'], uc=r['uc'])
PTS = {k: load_point(k) for k, *_ in GAS}
# ---------------- Fig1: uniform field N2, two-term vs multi-term vs MC
if 'fig1' in which:
    T = pickle.load(open('results/n2_twoterm.pkl', 'rb')); NH = pickle.load(open('results/n2_nh.pkl', 'rb')) if os.path.exists('results/n2_nh.pkl') else {}
    E0 = np.array(sorted(T)); EN = E0 / np.sqrt(2) / 3.21e21 * 1e21
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.5))
    series = [('0D', r'0D, $L=1$, $N_h=1$', 'k', ':', 'none'), ('L1', r'cyl, $L=1$, $N_h=1$', 'k', '-', 'o'), ('L2', r'cyl, $L=2$, $N_h=1$', '0.4', '-', 's')]
    for key, lab, c, ls, mk in series:
        ax[0].plot(EN, [T[e][key][0] for e in E0], ls, marker=mk, color=c, mfc='w', label=lab)
        ax[1].semilogy(EN, [max(T[e][key][1], 1e-2) for e in E0], ls, marker=mk, color=c, mfc='w', label=lab)
    for (L, Nh), lab, c, mk in (((1, 2), r'cyl, $L=1$, $N_h=2$', '0.4', '^'), ((2, 2), r'cyl, $L=2$, $N_h=2$', '0.2', 'D')):
        es = sorted(e for (e, l, n) in NH if l == L and n == Nh)
        if es:
            en = np.array(es) / np.sqrt(2) / 3.21e21 * 1e21
            ax[0].plot(en, [NH[(e, L, Nh)][0] for e in es], '--', marker=mk, color=c, label=lab); ax[1].semilogy(en, [max(NH[(e, L, Nh)][1], 1e-2) for e in es], '--', marker=mk, color=c, label=lab)
    ax[0].plot(EN, [T[e]['MC'][0] for e in E0], 'k*', ms=6, label='Monte Carlo'); ax[1].semilogy(EN, [max(T[e]['MC'][1], 1e-2) for e in E0], 'k*', ms=6, label='Monte Carlo')
    ax[0].set(xlabel=r'$E_{\rm rms}/N$, Td', ylabel=r'$\langle u\rangle$, eV'); ax[1].set(xlabel=r'$E_{\rm rms}/N$, Td', ylabel=r'$\nu_{\rm iz}$, s$^{-1}$', ylim=(1, 1e5)); ax[0].legend(fontsize=5.5)
    for a, l in zip(ax, 'ab'): a.set_title(f'({l})', loc='left', fontsize=8)
    save(fig, 'Fig1')
# ---------------- Fig2: self-consistent profiles at the operating points
if 'fig2' in which:
    fig, ax = plt.subplots(1, 4, figsize=(W, 2.3))
    for key, mix, lab, g0, ls, c in GAS:
        d = PTS[key]
        if d is None: continue
        X = d['X']; ax[0].plot(X, d['Te'], ls, color=c, label=lab); ax[1].plot(X, d['phi'] - d['phi'][0], ls, color=c)
        ax[2].plot(X, d['ne'] / d['ne'][-1], ls, color=c)
        if 'nuatt' in d and np.any(d['nuatt'] > 0):      # electronegative gas: gross ionization and attachment
            ax[3].semilogy(X, np.maximum(d['nuiz_gross'], 1e-1), ls, color=c); ax[3].semilogy(X, np.maximum(d['nuatt'], 1e-1), ls, color=c, lw=0.6, label=lab + ', attachment')
        else: ax[3].semilogy(X, np.maximum(d['nuiz'], 1e-1), ls, color=c)
    ax[0].set(xlabel=r'$r/R$', ylabel=r'$T_e$, eV'); ax[0].legend(fontsize=5.5); ax[1].set(xlabel=r'$r/R$', ylabel=r'$\varphi$, V'); ax[2].set(xlabel=r'$r/R$', ylabel=r'$n_e/n_e(0)$')
    ax[3].set(xlabel=r'$r/R$', ylabel=r'$\nu_{\rm iz}$, $\nu_{\rm att}$, s$^{-1}$', ylim=(1, 1e5)); ax[3].legend(fontsize=5.5, loc='lower left')
    for a, l in zip(ax, 'abcd'): a.set_title(f'({l})', loc='left', fontsize=8)
    save(fig, 'Fig2')
# ---------------- Fig3: EEDFs at wall and axis (kinetic energy) for Ar, Ar/N2 90/10, N2, air
if 'fig3' in which:
    keys = [k for k in ('Ar', 'Ar0.9_N20.1', 'N2', 'N20.79_O20.21') if PTS.get(k) is not None]
    fig, ax = plt.subplots(1, len(keys), figsize=(W, 2.4), sharey=True)
    ax = np.atleast_1d(ax)
    for a, key in zip(ax, keys):
        d = PTS[key]; lab = [g[2] for g in GAS if g[0] == key][0]; g0 = [g[3] for g in GAS if g[0] == key][0]
        for j, (node, nl, c) in enumerate(((0, 'wall', 'k'), (len(d['X']) // 2, r'$r/R=%.2f$' % d['X'][len(d['X']) // 2], '0.4'), (-1, 'axis', '0.65'))):
            u = d['ukin'][:, node]; F = d['F0'][:, node]; ok = (u > 0) & (F > 0); n = np.sum(d['w'][:, node] * F)
            a.semilogy(u[ok], F[ok] / n, '-', color=c, label=nl)
        z = zero_d(g0, 0.02)
        if z: a.semilogy(z['uc'], np.maximum(z['f0'], 1e-30), ':', color='k', lw=0.9, label=r'0D, $\Lambda=2$ cm')
        a.set(xlabel=r'$u$, eV', xlim=(0, 25), ylim=(1e-7, 1), title=lab); a.title.set_fontsize(8)
    ax[0].set(ylabel=r'$f_0(u)$, eV$^{-3/2}$'); ax[0].legend(fontsize=5.5)
    save(fig, 'Fig3')
# ---------------- Fig4: energy budget, cylinder vs 0D
GROUPS = [('vib', r'N$_2$/O$_2$ vibration', '0.85'), ('N2(A)', r'N$_2$(A)', '0.65'), ('el_N2', r'N$_2$ electronic', '0.45'), ('exc_Ar', 'Ar excitation', '0.25'),
          ('diss_O2', r'O$_2$ dissociation', '0.55'), ('O2(a1Dg)', r'O$_2$(a,b)', '0.75'), ('O2(b1Sg)', None, '0.75'), ('el_O2', r'O$_2$ electronic', '0.35'), ('iz', 'ionization', '0.0'), ('elastic', 'elastic', '0.95'), ('rotation', None, '0.95')]
if 'fig4' in which:
    fig, ax = plt.subplots(figsize=(W, 3.0)); x = 0; ticks = []; done = set()
    for key, mix, lab, g0, ls, c in GAS:
        d = PTS[key]
        if d is None: continue
        cols = [('cyl', budget(d)['frac']), (r'0D $\Lambda$=2 cm', (zero_d(g0, 0.02) or {}).get('frac', {})), (r'0D $\Lambda$=4.2 cm', (zero_d(g0, R / 2.405) or {}).get('frac', {}))]
        for j, (cl, fr) in enumerate(cols):
            b = 0
            for gk, gl, gc in GROUPS:
                v = fr.get(gk, 0)
                if v <= 0: continue
                ax.bar(x, v, 0.8, bottom=b, color=gc, edgecolor='k', lw=0.4, label=(gl if gl and gk not in done else None)); done.add(gk); b += v
            ticks.append((x, cl)); x += 1
        ax.text(x - 2, 1.03, lab, ha='center', fontsize=7); x += 0.7
    ax.set(ylabel='power fraction', ylim=(0, 1.12), xticks=[t for t, _ in ticks], xticklabels=[l for _, l in ticks]); ax.tick_params(axis='x', labelsize=5.5, rotation=90)
    h, l = ax.get_legend_handles_labels(); ax.legend(h, l, fontsize=5.5, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.45)); ax.grid(False)
    save(fig, 'Fig4')
# ---------------- Fig5: MC check of the N2 discharge (and argon for reference)
if 'fig5' in which:
    fig, ax = plt.subplots(1, 2, figsize=(W * 0.7, 2.4))
    for key, mcf, c, lab in (('N2', P7_MC_N2, 'k', r'N$_2$'), ('Ar', 'results/mcfix_13_B0.pkl', '0.5', 'Ar')):
        d = PTS[key]
        if d is None or not os.path.exists(mcf): continue
        m = pickle.load(open(mcf, 'rb')); ax[0].plot(d['X'], d['Te'], '-', color=c, label=lab + ' solver'); ax[0].plot(m['r_over_R'], m['Te_eV'], 'o', color=c, mfc='w', label=lab + ' MC')
        ax[1].semilogy(d['X'], np.maximum(d['nuiz'], 1e-1), '-', color=c); ax[1].semilogy(m['r_over_R'], np.maximum(m['nu_iz'], 1e-1), 'o', color=c, mfc='w')
    ax[0].set(xlabel=r'$r/R$', ylabel=r'$T_e$, eV'); ax[0].legend(fontsize=5.5); ax[1].set(xlabel=r'$r/R$', ylabel=r'$\nu_{\rm iz}$, s$^{-1}$', ylim=(1, 1e5))
    for a, l in zip(ax, 'ab'): a.set_title(f'({l})', loc='left', fontsize=8)
    save(fig, 'Fig5')
# ---------------- table
if 'table' in which:
    print('\n%-14s %6s %6s %6s | %5s %5s %5s %6s %5s %5s | %8s %8s | %7s %7s' % ('gas', 'B*,G', 'Ew', 'EN_w', 'Te_w', 'Te_mx', 'Te_ax', '<Te>_n', 'well', 'n0/nR', 'cost_cyl', 'cost_0D', 'EN_0D', 'Te_0D'))
    for key, mix, lab, g0, ls, c in GAS:
        d = PTS[key]
        if d is None: print('%-14s missing' % key); continue
        b = budget(d); z = zero_d(g0, 0.02) or {}; z2 = zero_d(g0, R / 2.405) or {}
        print('%-14s %6.2f %6.0f %6.0f | %5.2f %5.2f %5.2f %6.2f %5.2f %5.1f | %8.0f %8.0f | %7s %7s   beta %.3f  frac %s' % (
            key, d['B_ext'], b['Ew'], b['EN'], d['Te'][0], d['Te'].max(), d['Te'][-1], b['Te_n'], d['phi'][-1] - d['phi'][0], d['ne'][-1] / d['ne'][0], b['cost'], z.get('cost', np.nan),
            '%.0f/%.0f' % (z.get('EN', np.nan), z2.get('EN', np.nan)), '%.2f/%.2f' % (z.get('Te', np.nan), z2.get('Te', np.nan)), d['balance'],
            ', '.join(f'{k} {100*v:.0f}' for k, v in sorted(b['frac'].items(), key=lambda t: -t[1]) if v > 0.005)))
        if b['eV_per_O']: print('      eV per O atom: cyl %.1f, 0D %.1f / %.1f; att/iz %.2f' % (b['eV_per_O'], z.get('eV_per_O') or np.nan, z2.get('eV_per_O') or np.nan, b['att_iz']))
print('done', which)
