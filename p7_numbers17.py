"""Paper 7 (v1.7.0) numbers from the kin runs: operating points (log interpolation), values at the point nearest beta=1 and interpolated
to beta=1, energy budget, Lambda_eff^2 = <D_a>/<nu_net>, local-model ratios (results/p7_local_*.pkl), lambda_eps, MC correction for N2.
usage: P7_TAG=_kin python3 p7_numbers17.py  -> results/p7_numbers17.json and printed LaTeX rows"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, glob, os, sys, json, scipy.constants as co
os.environ.setdefault('P7_TAG', '_kin'); os.environ.setdefault('P7_AR', 'results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_teff2.pkl')
sys.argv = ['p7_figs.py', 'none']; import runpy; F = runpy.run_path('p7_figs.py')
GAS, budget, mixture_xs, P = F['GAS'], F['budget'], F['mixture_xs'], F['P']
from ekin_cyl_mix import ion_mobility
sig = lambda s, x: np.interp(x, s[:, 0], s[:, 1], left=0.0, right=s[-1, 1])
def wmean(d, q): w = d['wdisk'] * d['ne']; return float(np.sum(w * q) / np.sum(w))
def lam_eps(d):
    xs = mixture_xs(d['mix']); ng = d['ng']; u = np.maximum(d['ukin'], 0); v = np.sqrt(2 * co.e * u / co.m_e); w = d['w']; Fz = d['F0']; out = {}
    for name, node in (('wall', 0), ('mid', len(d['X']) // 2), ('axis', len(d['X']) - 1)):
        uu, vv, ww, FF = u[:, node], v[:, node], w[:, node], Fz[:, node]; n = np.sum(ww * FF); um = np.sum(ww * FF * uu) / n
        num = np.sum(ww * FF * ng * sig(xs['elastic'], uu) * vv) / n
        Pel = np.sum(ww * FF * ng * sig(xs['eloss'], uu) * vv * uu) / n
        Pin = sum(np.sum(ww * FF * ng * sig(s, uu) * vv * (uu > th)) * th for th, s, _ in xs['inel']) / n
        izl = xs['iz'] if isinstance(xs['iz'], list) else [xs['iz']]; Piz = sum(np.sum(ww * FF * ng * sig(s, uu) * vv * (uu > th)) * th for th, s in izl) / n
        nue = (Pel + Pin + Piz) / um; vm = np.sum(ww * FF * vv) / n
        out[name] = dict(lam_eps_cm=float(vm / np.sqrt(3 * num * nue) * 100), mfp_cm=float(vm / num * 100), nu_m=float(num), nu_eps=float(nue))
    return out
out = {}; rows = []
for key, mix, lab, g0, ls, c in GAS:
    tag = os.environ['P7_TAG']
    if key == 'Ar':
        fs = sorted(glob.glob('results/discharge_te_B*_n1e+17_L1_m16_o2_kin_teff2.pkl'))
        ds = []
        for f in fs:
            d = pickle.load(open(f, 'rb')); d['mix'] = [('Ar', 1.0)]; d['ng'] = 3.21e21; d['p'] = P; d['wdisk'] = pickle.load(open('results/discharge_mag_B00_Bext0.77_fix.pkl', 'rb'))['wdisk']; ds.append(d)
    else:
        fs = [g for g in sorted(glob.glob(f'results/discharge_mix_{key}_p{P:g}_B*{tag}.pkl')) if 'ckpt' not in g]
        ds = [pickle.load(open(f, 'rb')) for f in fs]
    if not ds: print(key, 'missing'); continue
    Bs = np.array([d['B_ext'] for d in ds]); bs = np.array([float(np.mean(np.array(d['hist'])[-4:, 2])) for d in ds]); o = np.argsort(bs)
    Bstar = float(np.exp(np.interp(0.0, np.log(bs[o]), np.log(Bs[o])))) if len(ds) > 1 else float(Bs[0])
    slope = float((np.log(bs[o][-1]) - np.log(bs[o][0])) / (np.log(Bs[o][-1]) - np.log(Bs[o][0]))) if len(ds) > 1 and abs(np.log(Bs[o][-1] / Bs[o][0])) > 1e-6 else None
    i0 = int(np.argmin(np.abs(bs - 1))); d = ds[i0]; b = budget(d); X = d['X']; ion, M, K0 = ion_mobility(mix)
    mu_i = K0 * 1e-4 * 2.6868e25 / d['ng']; Teff = d.get('Teff'); Da = mu_i * (np.array(Teff) if Teff is not None else d['Te'])
    nu_net = d['nuiz']; Lam2 = wmean(d, Da) / wmean(d, nu_net); src = d['wdisk'] * d['ne'] * nu_net
    f07 = float(np.sum(src[X > 0.7]) / np.sum(src)); f05 = float(np.sum(src[X > 0.5]) / np.sum(src)); imax = int(np.argmax(nu_net))
    cands = [f'results/p7_local_{os.path.basename(fs[i0])}', f'results/p7_local_{"_".join(f"{s}{x:g}" for s, x in mix)}.pkl'] if fs else []
    loc_f = next((c for c in cands if os.path.exists(c)), None); loc = pickle.load(open(loc_f, 'rb')) if loc_f else None
    locr = None
    if loc is not None:
        w = d['wdisk'] * d['ne']; nl = loc['nuiz_loc'] - (loc['nuatt_loc'] if loc.get('nuatt_loc') is not None else 0); locr = dict(ratio=float(np.sum(w * nl) / np.sum(w * nu_net)), Te_loc_n=float(np.sum(w * loc['Te_loc']) / np.sum(w)), Te_loc=[float(x) for x in loc['Te_loc']], nuiz_loc=[float(x) for x in nl])
    # linear interpolation to beta=1 of the scalar quantities between the two bracketing points (if available)
    interp = None
    if len(ds) > 1:
        j = np.argsort(bs); lo = [k for k in j if bs[k] <= 1]; hi = [k for k in j if bs[k] > 1]
        if lo and hi:
            a, bb = ds[lo[-1]], ds[hi[0]]; t = (0 - np.log(bs[lo[-1]])) / (np.log(bs[hi[0]]) - np.log(bs[lo[-1]]))
            ba, bb_ = budget(a), budget(bb); li = lambda x, y: float(x + t * (y - x))
            interp = dict(Ew=li(ba['Ew'], bb_['Ew']), EN=li(ba['EN'], bb_['EN']), Te_w=li(a['Te'][0], bb['Te'][0]), Te_max=li(a['Te'].max(), bb['Te'].max()), Te_ax=li(a['Te'][-1], bb['Te'][-1]), Te_n=li(ba['Te_n'], bb_['Te_n']),
                          well=li(a['phi'][-1] - a['phi'][0], bb['phi'][-1] - bb['phi'][0]), compr=li(a['ne'][-1] / a['ne'][0], bb['ne'][-1] / bb['ne'][0]), cost=li(ba['cost'], bb_['cost']))
    rec = dict(key=key, label=lab, files=fs, B=Bs.tolist(), beta=bs.tolist(), Bstar=Bstar, Im=Bstar * 1e-4 / co.mu_0, slope=slope, nearest=dict(B=float(d['B_ext']), beta=float(d['balance']), n_iter=int(d['n_iter']), conv=bool(d['converged']), Ew=b['Ew'], EN=b['EN'],
                 Te_w=float(d['Te'][0]), Te_max=float(d['Te'].max()), XTmax=float(X[int(np.argmax(d['Te']))]), Te_ax=float(d['Te'][-1]), Te_n=b['Te_n'], well=float(d['phi'][-1] - d['phi'][0]), phimax=float(d['phi'].max() - d['phi'][0]), Xphimax=float(X[int(np.argmax(d['phi']))]),
                 compr=float(d['ne'][-1] / d['ne'][0]), cost=b['cost'], cost_gross=b['cost_gross'], att_iz=b['att_iz'], eV_per_O=b['eV_per_O'], frac={k: float(v) for k, v in b['frac'].items()}, numax=float(nu_net[imax]), Xnumax=float(X[imax]), nu_ax=float(nu_net[-1]), nu_w=float(nu_net[0]),
                 f07=f07, f05=f05, nu_n=wmean(d, nu_net), Da_n=wmean(d, Da), Lam_eff_cm=float(np.sqrt(Lam2) * 100), ion=ion, K0=K0, mu_i=mu_i, m=len(X), delta_cm=float((1 - np.interp(np.exp(-1), (np.abs(d['E'][2]) / abs(d['E'][2][0]))[::-1], X[::-1])) * 10),
                 nuatt_n=(wmean(d, d['nuatt']) if 'nuatt' in d and np.any(d['nuatt'] > 0) else 0.0), nuiz_gross_n=(wmean(d, d['nuiz_gross']) if 'nuiz_gross' in d else wmean(d, nu_net)), neg_inside=(float(X[np.where(nu_net < 0)[0]].max()) if np.any(nu_net < 0) else None)),
               lam=lam_eps(d), local=locr, interp=interp)
    out[key] = rec; n = rec['nearest']
    print(f"{lab:14s} B*={Bstar:.3f} G ({rec['Im']:.0f} A/m) slope {slope} | nearest B={n['B']:.3f} beta={n['beta']:.3f} ({n['n_iter']} it conv {n['conv']}) Ew {n['Ew']:.0f} ({n['EN']:.0f} Td) delta {n['delta_cm']:.1f} Te {n['Te_w']:.2f}/{n['Te_max']:.2f}@{n['XTmax']:.2f}/{n['Te_ax']:.2f} <Te>_n {n['Te_n']:.2f} well {n['well']:.2f} (max {n['phimax']:.2f}@{n['Xphimax']:.2f}) n0/nw {n['compr']:.1f} cost {n['cost']:.0f} f>0.7 {f07:.2f} Lam_eff {n['Lam_eff_cm']:.2f} cm <nu> {n['nu_n']:.0f} Da {n['Da_n']:.2f} ion {ion} mu {mu_i:.2f} local {locr['ratio'] if locr else None} lam {[(k, round(v['lam_eps_cm'], 2)) for k, v in rec['lam'].items()]}")
    print('   frac', {k: round(v, 3) for k, v in n['frac'].items() if v > 0.005}, 'interp', interp)
json.dump(out, open('results/p7_numbers17.json', 'w'), indent=1, default=float)
