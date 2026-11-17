"""Find the operating point (beta = 1) of a gas mixture discharge by secant iteration in ln B_ext on ln beta, each step warm-started
from the previous converged fields.  usage: python3 mix_opt.py <gas> <p_Pa> <B1> [B2] [tol=0.08] [max_runs=4]
env passed to discharge_mix.py: MIX_NE, MIX_UMAX, MIX_M, ITERS (first run), ITERS2 (warm runs)."""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import subprocess, sys, os, pickle, numpy as np
gas, p = sys.argv[1], float(sys.argv[2]); B1 = float(sys.argv[3]); B2 = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '-' else None
tol = float(sys.argv[5]) if len(sys.argv) > 5 else 0.08; maxr = int(sys.argv[6]) if len(sys.argv) > 6 else 4
tag = os.environ.get('MIX_TAG', '_opt'); gname = gas.replace(':', '').replace(',', '_')
def fname(B): return f"results/discharge_mix_{gname}_p{p:g}_B{B:g}{tag}.pkl"
def run(B, start=None, iters=None):
    env = dict(os.environ, TAG=tag); env['MIX_START'] = start or env.get('MIX_START', 'results/discharge_te_B0.76_n1e+17_L1_m16_o2_conjfix.pkl')
    if start: env['MIX_PHISCALE'] = '1.0'
    it = str(iters or env.get('ITERS', '25'))
    log = f"results/dmix_{gname}_p{p:g}_B{B:g}{tag}.log"
    with open(log, 'w') as lf: subprocess.run(['python3', 'discharge_mix.py', gas, str(p), str(B), it], stdout=lf, stderr=subprocess.STDOUT, env=env, check=True)
    d = pickle.load(open(fname(B), 'rb')); h = np.array(d['hist']); beta = float(np.mean(h[-4:, 2]))
    print(f"{gas} {p} Pa B={B:.3f} G: beta={beta:.3f} (final {d['balance']:.3f}, conv {d['converged']}, {d['n_iter']} it) Te axis {d['Te'][-1]:.2f} wall {d['Te'][0]:.2f} well {d['phi'][-1]-d['phi'][0]:.2f}", flush=True)
    return beta
pts = []
b = run(B1); pts.append((B1, b))
if B2 is None: B2 = B1 * b ** (-1 / 5.0)
B_prev = B1
for k in range(maxr - 1):
    if abs(pts[-1][1] - 1) < tol: break
    b = run(B2, start=fname(B_prev), iters=os.environ.get('ITERS2', '15')); pts.append((B2, b))
    if abs(b - 1) < tol: break
    (Ba, ba), (Bb, bb) = pts[-2], pts[-1]
    slope = (np.log(bb) - np.log(ba)) / (np.log(Bb) - np.log(Ba)) if abs(np.log(Bb / Ba)) > 1e-6 else 5.0
    slope = float(np.clip(slope, 2.0, 12.0)); B_prev = B2; B2 = float(Bb * bb ** (-1 / slope))
print('points', [(round(B, 3), round(b, 3)) for B, b in pts])
Bs = np.array([B for B, _ in pts]); bs = np.array([b for _, b in pts]); o = np.argsort(bs)
print('B* (log interp) = %.3f G' % np.exp(np.interp(0.0, np.log(bs[o]), np.log(Bs[o]))))
