"""Refined homogeneous (0D two-term, harmonic balance) operating points for Paper 7 at 13.56 MHz, without e-e, for two diffusion
lengths: Lambda = 2 cm (as in the 3 Nov study) and Lambda = R/2.405 (fundamental mode of the R = 10 cm tube).
-> results/p7_zero_d.pkl  {(gas, p, lam): record with EN, u, Te, rates, power}.   usage: python3 p7_zero_d.py [p_Pa]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, sys, os, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import mix_figs as MF, ee_sustain as ES
p = float(sys.argv[1]) if len(sys.argv) > 1 else 13.3
GASES = ['N2', 'Ar+N2 90/10', 'Ar+N2 50/50', 'air', 'Ar', 'O2', 'Ar+O2 90/10']
OUT = 'results/p7_zero_d.pkl'; res = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
for lam in (0.02, 0.1 / 2.405):
    MF.LAM = lam
    for g in GASES:
        key = (g, p, round(lam, 5))
        if key in res: continue
        t = time.time(); _, val = ES.run((g, p, 13.56, 0.0, 'hb')); r = val['root']
        res[key] = dict(EN=r['EN'], u=r['u'], Te=2 * r['u'] / 3, rates=r['rates'], power=r['power'], f0=r['f0'], uc=val['uc'], ng=val['ng'], g=val['g'])
        pickle.dump(res, open(OUT, 'wb'))
        print(f"{g:12s} {p} Pa Lambda {100*lam:.2f} cm: E/N {r['EN']:.1f} Td rms, Te {2*r['u']/3:.2f} eV ({time.time()-t:.0f} s, {len(val['hist'])} solves)", flush=True)
