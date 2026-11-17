"""Local (0D at the local field) model of a converged self-consistent discharge: at every radial node the homogeneous two-term
solver (L=1, N_h=1, no growth term) is run at the local RF field amplitude, and T_e(r), nu_iz(r) (and nu_att) are compared with the
kinetic solution. usage: python3 p7_local.py <pkl> [gas]   -> results/p7_local_<gas>.pkl and a printed table."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from ekin_mix import MixtureKinetics
from mix_scan import SP
fn = sys.argv[1]; d = pickle.load(open(fn, 'rb')); mix = d.get('mix', [('Ar', 1.0)]); gas = sys.argv[2] if len(sys.argv) > 2 else '_'.join(f'{s}{x:g}' for s, x in mix)
ng = d.get('ng', 3.21e21); om = 2 * np.pi * 13.56e6; X = d['X']; Eamp = 2 * np.abs(d['E'][2])
K = MixtureKinetics([(SP[s], x) for s, x in mix], umax=60., nu=300, L=1, Nh=1, ng=ng, Tg=300., growth=False)
Te_l = []; nuiz_l = []; nuatt_l = []
for k in range(len(X)):
    f = K.solve(max(Eamp[k], 1e-3), om); f0 = np.real(f[K.Nh, 0]); n = np.sum(K.w * f0)
    Te_l.append(2 / 3 * np.sum(K.w * K.uc * f0) / n); nuiz_l.append(np.sum(K.w * K.nu_iz_c * f0) / n); nuatt_l.append(np.sum(K.w * K.nu_att_c * f0) / n)
Te_l = np.array(Te_l); nuiz_l = np.array(nuiz_l); nuatt_l = np.array(nuatt_l)
w = d['wdisk'] * d['ne'] if 'wdisk' in d else None
print(f"{gas}: r/R, E/N(Td rms), Te kin/local, nu_iz kin(gross)/local, nu_att kin/local")
for k in range(len(X)):
    print(f"  {X[k]:.2f} {Eamp[k]/np.sqrt(2)/ng*1e21:6.1f}  {d['Te'][k]:.2f}/{Te_l[k]:.2f}  {d.get('nuiz_gross', d['nuiz'])[k]:.3g}/{nuiz_l[k]:.3g}  {d.get('nuatt', np.zeros_like(X))[k]:.3g}/{nuatt_l[k]:.3g}")
if w is not None:
    net_k = np.sum(w * d['nuiz']) / w.sum(); net_l = np.sum(w * (nuiz_l - nuatt_l)) / w.sum()
    print(f"  density-weighted net ionization: kinetic {net_k:.4g}, local {net_l:.4g} s^-1 (ratio {net_l/net_k:.3f}); Te_n kinetic {np.sum(w*d['Te'])/w.sum():.2f}, local {np.sum(w*Te_l)/w.sum():.2f} eV")
pickle.dump(dict(X=X, EN=Eamp / np.sqrt(2) / ng * 1e21, Te_kin=d['Te'], Te_loc=Te_l, nuiz_kin=d.get('nuiz_gross', d['nuiz']), nuiz_loc=nuiz_l, nuatt_kin=d.get('nuatt', None), nuatt_loc=nuatt_l, src=fn), open(f'results/p7_local_{gas}.pkl', 'wb'))
