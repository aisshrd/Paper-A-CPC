"""Paper 7 queue: operating points (beta = 1) of several gases at 13.3 Pa, one discharge_mix run at a time, resumable.
Each run: skipped if its pkl exists; resumed from its _ckpt.pkl if one exists; otherwise warm-started from the previous
converged point of the same gas, or cold-started from the argon solution (MIX_PHISCALE 0.45).  Secant in ln B on ln beta.
usage: python3 p7_queue.py [gas=B1 ...]   (default list below)   env: MIX_NE, MIX_UMAX, MIX_M, ITERS, ITERS2, MIX_TAG, P7_P"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import subprocess, sys, os, pickle, numpy as np, time, glob
p = float(os.environ.get('P7_P', '13.3')); tag = os.environ.get('MIX_TAG', '_opt'); tol = 0.08; maxr = 3
PLAN = [('N2', 2.4), ('Ar:0.9,N2:0.1', 1.42), ('Ar:0.5,N2:0.5', 1.65), ('N2:0.79,O2:0.21', 2.6)]
if sys.argv[1:]: PLAN = [(a.split('=')[0], float(a.split('=')[1])) for a in sys.argv[1:]]
KNOWN = {'Ar:0.9,N2:0.1': [(0.95, 0.039)]}          # converged earlier points not re-run
ARGON = os.environ.get('P7_ARGON', 'results/discharge_te_B0.76_n1e+17_L1_m16_o2.pkl')
def gname(gas): return gas.replace(':', '').replace(',', '_')
def fname(gas, B): return f"results/discharge_mix_{gname(gas)}_p{p:g}_B{B:g}{tag}.pkl"
def beta_of(fn):
    d = pickle.load(open(fn, 'rb')); h = np.array(d['hist']); return float(np.mean(h[-4:, 2])), d
def run(gas, B, prev):
    fn = fname(gas, B)
    if not os.path.exists(fn):
        env = dict(os.environ, TAG=tag); ck = fn.replace('.pkl', '_ckpt.pkl')
        if os.path.exists(ck): env['MIX_START'] = ck; env['MIX_PHISCALE'] = '1.0'; it = env.get('ITERS2', '14'); how = 'resume'
        elif prev: env['MIX_START'] = prev; env['MIX_PHISCALE'] = '1.0'; it = env.get('ITERS2', '14'); how = 'warm'
        elif os.environ.get('P7_FROM_OPT') and glob.glob(f"results/discharge_mix_{gname(gas)}_p{p:g}_B*{os.environ.get('P7_FROM_OPT_TAG','_opt')}.pkl"):
            cands = [c for c in glob.glob(f"results/discharge_mix_{gname(gas)}_p{p:g}_B*{os.environ.get('P7_FROM_OPT_TAG','_opt')}.pkl") if 'ckpt' not in c]
            best = min(cands, key=lambda c: abs(float(c.split('_B')[-1].split('_')[0]) - B)); env['MIX_START'] = best; env['MIX_PHISCALE'] = '1.0'; it = env.get('ITERS', '18'); how = 'warm from ' + os.path.basename(best)
        else: env['MIX_START'] = ARGON; env['MIX_PHISCALE'] = env.get('MIX_PHISCALE', '0.45'); it = env.get('ITERS', '18'); how = 'cold'
        print(f"[{time.strftime('%H:%M')}] {gas} B={B:.3f} G ({how}, {it} it)", flush=True)
        with open(f"results/dmix_{gname(gas)}_p{p:g}_B{B:g}{tag}.log", 'a') as lf:
            subprocess.run(['python3', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'discharge_mix.py'), gas, str(p), str(B), it], stdout=lf, stderr=subprocess.STDOUT, env=env, check=True)
    b, d = beta_of(fn)
    print(f"[{time.strftime('%H:%M')}] {gas} {p} Pa B={B:.3f} G: beta={b:.3f} (final {d['balance']:.3f}, conv {d['converged']}, {d['n_iter']} it) Te wall/max/axis {d['Te'][0]:.2f}/{d['Te'].max():.2f}/{d['Te'][-1]:.2f} well {d['phi'][-1]-d['phi'][0]:.2f}", flush=True)
    return b, fn
for gas, B1 in PLAN:
    pts = list(KNOWN.get(gas, [])); prev = None; B = B1
    for k in range(maxr):
        b, fn = run(gas, B, prev); pts.append((B, b)); prev = fn
        if abs(b - 1) < tol: break
        if len(pts) >= 2:
            (Ba, ba), (Bb, bb) = pts[-2], pts[-1]
            slope = (np.log(bb) - np.log(ba)) / (np.log(Bb) - np.log(Ba)) if abs(np.log(Bb / Ba)) > 1e-6 else 6.0
            slope = float(np.clip(slope, 3.0, 12.0))
        else: slope = 6.0
        B = float(round(B * b ** (-1 / slope), 3))
    Bs = np.array([x for x, _ in pts]); bs = np.array([y for _, y in pts]); o = np.argsort(bs)
    print(f"== {gas}: points {[(round(x,3), round(y,3)) for x, y in pts]}  B* (log interp) = {np.exp(np.interp(0.0, np.log(bs[o]), np.log(Bs[o]))):.3f} G", flush=True)
print('queue done')
