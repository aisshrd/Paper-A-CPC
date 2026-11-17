"""Paper 7 analysis: summarize self-consistent mixture discharges (discharge_mix pkl) and compare with the homogeneous
two-term operating points of 3 Nov (mix_figs.sustain, ee_sust0.pkl).  usage: python3 p7_an.py [pkl ...]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, glob, os, scipy.constants as co
from ekin_cyl_mix import mixture_xs
from mix_scan import group
GAS0D = {'N2': 'N2', 'O2': 'O2', 'Ar': 'Ar', 'N20.79_O20.21': 'air', 'Ar0.9_N20.1': 'Ar+N2 90/10', 'Ar0.5_N20.5': 'Ar+N2 50/50',
         'Ar0.9_O20.1': 'Ar+O2 90/10', 'Ar0.5_O20.5': 'Ar+O2 50/50'}
R = 0.1

def summary(d):
    """global (volume-integrated) energy budget of a cylinder solution"""
    xs = mixture_xs(d['mix']); X = d['X']; w = d['wdisk']; ne = d['ne'] * d['n0']
    Pabs = 2 * np.real(d['jt'] * np.conj(d['E'][2]))            # W/m^3, E(t)=E1 e^{iwt}+cc
    Ptot = np.sum(w * Pabs)                                      # W per unit length / (pi R^2 normalisation of wdisk)
    rates = d['rates']; P = {}; Rg = {}
    for (th, s, name), (nm, nu) in zip(xs['inel'], rates.items()):
        assert nm == name
        kind, sn, prod = 'exc', name.split(':')[0], name.split(':', 1)[1]
        g = group(kind, sn, prod); P[g] = P.get(g, 0.0) + np.sum(w * ne * nu) * th * co.e; Rg[g] = Rg.get(g, 0.0) + np.sum(w * ne * nu)
    Niz = np.sum(w * ne * d['nuiz_gross']); Natt = np.sum(w * ne * d['nuatt']); Ne = np.sum(w * ne)
    thiz = min(t for t, _ in xs['iz']); P['iz'] = Niz * thiz * co.e
    Pin = sum(P.values())                                        # collisional losses (elastic + rotational ~1 % omitted)
    frac = {k: v / Pin for k, v in P.items()}
    Ew = abs(d['E'][2][0]) * 2; EN_rms = Ew / np.sqrt(2) / d['ng'] * 1e21
    out = dict(B=d['B_ext'], beta=d['balance'], Ew=Ew, EN_rms_wall=EN_rms, Te_wall=d['Te'][0], Te_max=d['Te'].max(), Te_axis=d['Te'][-1],
               X_Temax=X[np.argmax(d['Te'])], well=d['phi'][-1] - d['phi'][0], compr=ne[-1] / ne[0], Ptot=Ptot, Pin=Pin, frac=frac,
               eV_per_ion=Pin / (co.e * max(Niz - Natt, 1e-300)), eV_per_ion_gross=Pin / (co.e * Niz), att_iz=Natt / Niz,
               nuiz_mean=(Niz - Natt) / Ne, Te_mean=np.sum(w * ne * d['Te']) / Ne, Rg={k: v / Ne for k, v in Rg.items()}, P_Joule_over_loss=Ptot / Pin,
               eV_per_O=(Pin / (co.e * 2 * Rg['diss_O2'] * Ne) if 'diss_O2' in Rg else None), skin=skin(d))
    return out

def skin(d):
    X = d['X']; E = np.abs(d['E'][2]); y = E / E[0]
    return float((1 - np.interp(np.exp(-1), y[::-1], X[::-1])) * R * 100) if y[-1] < np.exp(-1) else np.inf

def zero_d(gname, p, lam):
    import mix_figs as MF
    s = MF.sustain(gname, p, 13.56, lam=lam)
    if s is None: return None
    Pt = sum(s['power'].values()); frac = {k: v / Pt for k, v in s['power'].items()}     # power in eV/s per electron
    net = s['rates'].get('iz', 0) - s['rates'].get('att', 0)
    return dict(EN=s['EN'], Te=s['Te'], u=s['u'], frac=frac, eV_per_ion=Pt / (net * s['ng']) if net > 0 else np.inf,
                eV_per_O=(Pt / (2 * s['rates']['diss_O2'] * s['ng']) if 'diss_O2' in s['rates'] else None), rates=s['rates'])

def fmt(o):
    f = ', '.join(f"{k} {100*v:.0f}%" for k, v in sorted(o['frac'].items(), key=lambda t: -t[1]) if v > 0.004)
    return f

if __name__ == '__main__':
    files = sys.argv[1:] or sorted(glob.glob('results/discharge_mix_*_opt.pkl'))
    for f in files:
        d = pickle.load(open(f, 'rb')); o = summary(d)
        gname = os.path.basename(f).split('_p')[0].replace('discharge_mix_', '')
        print(f"\n== {f}: {d['gas']} {d['p']} Pa, B_ext {o['B']:.3f} G, beta {o['beta']:.3f}")
        print(f"  E wall {o['Ew']:.0f} V/m ({o['EN_rms_wall']:.0f} Td rms), skin {o['skin']:.1f} cm; Te wall/max/axis {o['Te_wall']:.2f}/{o['Te_max']:.2f}/{o['Te_axis']:.2f} eV (max at r/R={o['X_Temax']:.2f}), <Te>_n {o['Te_mean']:.2f}; well {o['well']:.2f} V; n(0)/n(R) {o['compr']:.1f}")
        print(f"  Joule/loss {o['P_Joule_over_loss']:.3f}; <nu_iz,net> {o['nuiz_mean']:.3e} s^-1, cost {o['eV_per_ion']:.0f} eV/ion" + (f" (gross {o['eV_per_ion_gross']:.0f}, att/iz {o['att_iz']:.2f})" if o['att_iz'] > 0 else '') + (f", {o['eV_per_O']:.1f} eV per O atom" if o['eV_per_O'] else ''))
        print('  power:', fmt(o))
        g0 = GAS0D.get(gname)
        if g0:
            for lam, lab in ((0.02, 'Lambda=2 cm'), (R / 2.405, 'Lambda=R/2.405')):
                z = zero_d(g0, d['p'], lam)
                if z: print(f"  0D {lab}: E/N {z['EN']:.0f} Td rms, Te {z['Te']:.2f} eV, cost {z['eV_per_ion']:.0f} eV/ion" + (f", {z['eV_per_O']:.1f} eV/O" if z['eV_per_O'] else '') + '; power: ' + fmt(z))
