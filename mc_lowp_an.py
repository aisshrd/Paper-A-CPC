"""Power, ionisation and RF radial current from the low-pressure Monte-Carlo runs (results/mc_lowp_*.pkl).
P per particle ~ -<v.E>; E at bin centres from the reference fields (E_theta) and the E_r file used in the run."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os
d = pickle.load(open(os.environ.get('MCAN_REF', 'results/discharge_te_B0.7_n1e+17_L1_m16_o2.pkl'), 'rb'))
X = d['X'][::-1]; E1 = d['E'][2][::-1]
def field(xc, E):                                          # amplitude 2|E1| and phase at bin centres
    return np.interp(xc, np.r_[0, X], np.r_[0, 2 * np.abs(E)]), np.interp(xc, np.r_[0, X], np.r_[np.angle(E[0]), np.angle(E)])
def analyse(f, erfile=None, key=(4.84, True, 1)):
    m = pickle.load(open(f, 'rb')); j = m['j_rf']; xc = m['r_over_R']
    A, ph = field(xc, E1)
    Pt = -np.sum(A * (np.cos(ph) * j[:, 2] - np.sin(ph) * j[:, 3]))
    Pr = 0.0
    if erfile:
        Er = pickle.load(open(erfile, 'rb'))[key]['Er'][2][::-1]; Ar, phr = field(xc, Er)
        Pr = -np.sum(Ar * (np.cos(phr) * j[:, 0] - np.sin(phr) * j[:, 1]))
    w = m['ne_rel'] * xc; nz = np.sum(w * m['nu_iz']) / np.sum(w)
    jr = np.sum(np.hypot(j[:, 0], j[:, 1])) / np.sum(np.hypot(j[:, 2], j[:, 3]))
    return dict(P=Pt + Pr, Pt=Pt, Pr=Pr, nuiz=nz, jr=jr, Te=m['Te_eV'], x=xc)
if __name__ == '__main__':
    pre = os.environ.get('MCAN_PRE', 'mc_lowp'); b = analyse(f'results/{pre}_B0.pkl')
    for name, f, er in (('B0=0', f'results/{pre}_B0.pkl', None), ('4.84 G, Er=0', f'results/{pre}_B4.84.pkl', None),
                        ('4.84 G, Er(L=1)', f'results/{pre}_B4.84_Er.pkl', os.environ.get('MCAN_ERF', 'results/lowp_Er_for_mc.pkl')),
                        ('4.84 G, Er(L=2)', f'results/{pre}_B4.84_ErL2.pkl', 'results/lowp_Er_for_mc_L2.pkl')):
        try: r = analyse(f, er)
        except FileNotFoundError: continue
        print(f"{name:18s} P/P0 {r['P']/b['P']:.2f} (Pr share {r['Pr']/r['P']:+.2f})  nuiz/nuiz0 {r['nuiz']/b['nuiz']:.2f}  |jr|/|jt| {r['jr']:.3f}  Te axis/wall {r['Te'][0]:.2f}/{r['Te'][-1]:.2f}")
