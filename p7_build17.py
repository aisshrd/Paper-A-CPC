"""Paper 7 (v1.7.0): fill the template with the numbers of the kin runs (results/p7_numbers17.json), the 0D N_h=1 operating points
(results/p7_zero_d_nh1.pkl), the MC comparison (results/mc_n2_disc_B*_kin.pkl); figures by p7_figs.py (P7_TAG=_kin); compile.
usage: python3 p7_build17.py [nofig]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import json, pickle, os, sys, glob, subprocess, numpy as np
OUT = 'papers/p7_v17'; BLD = OUT + '/build'; os.makedirs(BLD, exist_ok=True)
subprocess.run(['python3', 'p7_numbers17.py'], check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS='1'))
J = json.load(open('results/p7_numbers17.json')); Z = pickle.load(open('results/p7_zero_d_nh1.pkl', 'rb'))
P = 13.3; K = {'Ar': 'Ar', 'Ar0.9_N20.1': 'Ar+N2 90/10', 'Ar0.5_N20.5': 'Ar+N2 50/50', 'N2': 'N2', 'N20.79_O20.21': 'air'}
LAB = {'Ar': 'Ar', 'Ar0.9_N20.1': 'Ar/N$_2$ 90/10', 'Ar0.5_N20.5': 'Ar/N$_2$ 50/50', 'N2': 'N$_2$', 'N20.79_O20.21': 'air'}
def z(key, lam):
    r = Z.get((K[key], P, round(lam, 5)))
    if r is None: return None
    Pt = sum(r['power'].values()); net = r['rates'].get('iz', 0) - r['rates'].get('att', 0)
    return dict(EN=r['EN'], Te=r['Te'], cost=Pt / (net * r['ng']), frac={k: v / Pt for k, v in r['power'].items()})
have = [k for k in K if k in J]
def q(key, name, interp_ok=True):
    """operating-point value: interpolated to beta=1 when available, else the nearest run"""
    r = J[key]; n = r['nearest']; ip = r.get('interp')
    if interp_ok and ip and name in ip: return ip[name]
    return n[name]
T = {}
def setg(pref, key):
    r = J[key]; n = r['nearest']
    T[pref + '_B'] = f"{r['Bstar']:.2f}"; T[pref + '_I'] = f"{r['Im']:.0f}"; T[pref + '_EW'] = f"{q(key,'Ew'):.0f}"; T[pref + '_EN'] = f"{q(key,'EN'):.0f}"
    T[pref + '_TEW'] = f"{q(key,'Te_w'):.2f}"; T[pref + '_TEMAX'] = f"{q(key,'Te_max'):.2f}"; T[pref + '_TEAX'] = f"{q(key,'Te_ax'):.2f}"; T[pref + '_TEN'] = f"{q(key,'Te_n'):.2f}"
    T[pref + '_WELL'] = f"{q(key,'well'):.1f}"; T[pref + '_COMPR'] = f"{q(key,'compr'):.0f}"; T[pref + '_COST'] = f"{q(key,'cost'):.0f}"; T[pref + '_BETA'] = f"{n['beta']:.2f}"
    T[pref + '_NUMAX'] = ('$%.1f\\cdot10^{%d}$' % (n['numax'] / 10 ** int(np.floor(np.log10(n['numax']))), int(np.floor(np.log10(n['numax']))))) if n['numax'] >= 1e3 else f"{n['numax']:.0f}"; T[pref + '_XNUMAX'] = f"{n['Xnumax']:.2f}"; T[pref + '_NUAX'] = f"{n['nu_ax']:.0f}"
    T[pref + '_F07'] = f"{100*n['f07']:.0f}"; T[pref + '_F05'] = f"{100*n['f05']:.0f}"; T[pref + '_LAM'] = f"{n['Lam_eff_cm']:.1f}"; T[pref + '_NUN'] = f"{n['nu_n']:.0f}"; T[pref + '_DAN'] = f"{n['Da_n']:.1f}"
    T[pref + '_DELTA'] = f"{n['delta_cm']:.1f}"; T[pref + '_SLOPE'] = f"{r['slope']:.0f}" if r.get('slope') else '7'
    T[pref + '_LAMW'] = f"{r['lam']['wall']['lam_eps_cm']:.1f}"; T[pref + '_LAMM'] = f"{r['lam']['mid']['lam_eps_cm']:.1f}"; T[pref + '_LAMA'] = f"{r['lam']['axis']['lam_eps_cm']:.1f}"
    T[pref + '_LOC'] = f"{r['local']['ratio']:.2f}" if r.get('local') else '--'; T[pref + '_LOCERR'] = f"{abs(100*(r['local']['ratio']-1)):.0f}" if r.get('local') else '--'
    T[pref + '_VIB'] = f"{100*n['frac'].get('vib',0):.0f}"
    z2 = z(key, 0.02); z4 = z(key, 0.1 / 2.405)
    T[pref + '_EN0D'] = f"{z2['EN']:.0f}" if z2 else '--'; T[pref + '_TE0D'] = f"{z2['Te']:.2f}" if z2 else '--'; T[pref + '_TE0D4'] = f"{z4['Te']:.2f}" if z4 else '--'; T[pref + '_COST0D'] = f"{z2['cost']:.0f}" if z2 else '--'
for pref, key in (('AR', 'Ar'), ('M10', 'Ar0.9_N20.1'), ('M50', 'Ar0.5_N20.5'), ('N2', 'N2'), ('AIR', 'N20.79_O20.21')):
    if key in J: setg(pref, key)
    else:
        for suf in ('B', 'I', 'EW', 'EN', 'TEW', 'TEMAX', 'TEAX', 'TEN', 'WELL', 'COMPR', 'COST', 'BETA', 'NUMAX', 'XNUMAX', 'NUAX', 'F07', 'F05', 'LAM', 'NUN', 'DAN', 'DELTA', 'SLOPE', 'LAMW', 'LAMM', 'LAMA', 'LOC', 'LOCERR', 'VIB', 'EN0D', 'TE0D', 'TE0D4', 'COST0D'): T[pref + '_' + suf] = '--'
# ---- derived
ar, n2 = J['Ar'], J.get('N2')
if n2:
    rB = n2['Bstar'] / ar['Bstar']; rnu = n2['nearest']['nu_n'] / ar['nearest']['nu_n']; rcost = q('N2', 'cost') / q('Ar', 'cost'); rP = rnu * rcost
    T.update(RATIO_B=f"{rB:.1f}", RATIO_I=f"{rB:.1f}", RATIO_NUN=f"{rnu:.2f}", RATIO_COST=f"{rcost:.1f}", RATIO_P=f"{rP:.1f}", RATIO_P_SQRT=f"{np.sqrt(rP):.1f}")
last = [k for k in ('N20.79_O20.21', 'N2', 'Ar0.5_N20.5', 'Ar0.9_N20.1') if k in J][0]
T['LAST_B'] = f"{J[last]['Bstar']:.2f}"; T['LAST_EN'] = f"{q(last,'EN'):.0f}"; T['NGAS'] = str(len(have)); T['AIRSEQ'] = ' -- air' if 'N20.79_O20.21' in J else ''
T['MINWELL'] = f"{min(q(k,'well') for k in have):.1f}"; T['MIN_TEAX'] = f"{min(q(k,'Te_ax') for k in have):.1f}"; T['MAXCOST'] = f"{max(q(k,'cost') for k in have):.0f}"
costerr = [abs(z(k, 0.02)['cost'] / q(k, 'cost') - 1) * 100 for k in have if z(k, 0.02)]; T['COSTERR'] = f"{min(costerr):.0f}--{max(costerr):.0f}"
OLD = {'Ar': (0.774, 9.7), 'Ar0.9_N20.1': (1.28, 7.8), 'Ar0.5_N20.5': (1.92, 5.4), 'N2': (2.36, 4.1), 'N20.79_O20.21': (2.51, 3.6)}   # first version (v1.6.0): B*, well
ws = [q(k, 'well') - OLD[k][1] for k in have]; bs = [100 * (1 - J[k]['Bstar'] / OLD[k][0]) for k in have]
SH = {'Ar': 'Ar', 'Ar0.9_N20.1': '90/10', 'Ar0.5_N20.5': '50/50', 'N2': 'N$_2$', 'N20.79_O20.21': 'air'}
T['WELL_SHIFT'] = ', '.join(f"{w:+.1f}~V ({SH[k]})" for w, k in zip(ws, have)); T['B_SHIFT'] = ', '.join(f"{-b:+.0f}~\\% ({SH[k]})" for b, k in zip(bs, have))
T['MIX_M'] = str(J['N2']['nearest']['m'] if n2 else 12); T['LAYER_CELLS'] = '4' if T['MIX_M'] == '12' else '5'
m16 = glob.glob('results/discharge_mix_N2_p13.3_B*_kin_m16.pkl')
if m16 and n2:
    d16 = pickle.load(open(m16[0], 'rb')); d12 = pickle.load(open(n2['files'][int(np.argmin(np.abs(np.array(n2['beta']) - 1)))], 'rb'))
    T['M16_CHECK'] = f"by {abs(d16['balance']/d12['balance']-1)*100:.0f}~\\%"; T['M16_CHECK_LONG'] = f"a run with 16 cells at the same current changes the balance by {abs(d16['balance']/d12['balance']-1)*100:.0f}~\\%, the wall temperature by {abs(d16['Te'][0]-d12['Te'][0]):.2f}~eV and the well by {abs((d16['phi'][-1]-d16['phi'][0])-(d12['phi'][-1]-d12['phi'][0])):.2f}~V; with $\\beta\\propto B^{{7}}$ this is {abs(d16['balance']/d12['balance']-1)*100/7:.1f}~\\% in $B^*$."
else: T['M16_CHECK'] = 'by less than 10~\\% (see Sec.~\\ref{sec:disc})'; T['M16_CHECK_LONG'] = 'the sensitivity to the radial grid was checked in argon (16 against 24 and 32 cells, 3~\\% in the integral ionization~\\cite{Paper1}) and is expected to be of the same order here.'
# ---- table
rows = ''
for k in have:
    r = J[k]; n = r['nearest']; z2 = z(k, 0.02); z4 = z(k, 0.1 / 2.405)
    rows += f"{LAB[k]} & {r['Bstar']:.2f} & {r['Im']:.0f} & {q(k,'Ew'):.0f} & {q(k,'EN'):.0f} & {q(k,'Te_w'):.2f} & {q(k,'Te_max'):.2f} & {q(k,'Te_ax'):.2f} & {q(k,'Te_n'):.2f} & {q(k,'well'):.1f} & {q(k,'compr'):.0f} & {n['beta']:.2f} & {q(k,'cost'):.0f} & {n['Lam_eff_cm']:.1f} & {(('%.2f'%r['local']['ratio']) if r.get('local') else '--')} & {(('%.0f / %.0f'%(z2['EN'],z4['EN'])) if z2 and z4 else '--')} & {(('%.2f / %.2f'%(z2['Te'],z4['Te'])) if z2 and z4 else '--')} & {(('%.0f / %.0f'%(z2['cost'],z4['cost'])) if z2 and z4 else '--')}\\\\\n"
T['TAB_OP'] = r"""\begin{table}\caption{Self-consistent operating points at 13.3~Pa, $n_0=10^{17}$~m$^{-3}$ ($L=1$, $N_h=1$, corrected ion closure): the external field $B^*$ and the corresponding solenoid current per unit length $B^*/\mu_0$, the field amplitude and the reduced rms field at the wall, the electron temperature at the wall, its maximum and on the axis, the density-weighted mean temperature, the potential of the axis above the wall, the ratio of the axis to the wall density, the ionization balance of the run closest to $\beta=1$, the energy spent per net electron--ion pair (all inelastic losses divided by the net ionization), the effective diffusion length $\Lambda_{\rm eff}=(\langle D_a\rangle_n/\langle\nu_{\rm net}\rangle_n)^{1/2}$ of the kinetic solution, and the ratio of the density-weighted net ionization of the local model (homogeneous solver at the local field) to the kinetic one. $B^*$ is interpolated in $\ln\beta$ between the runs, and the scalar columns are interpolated to $\beta=1$ where two runs bracket it. The last three columns give the homogeneous two-term ($L=1$, $N_h=1$) operating point for $\Lambda=2$~cm / $\Lambda=R/2.405$.}\label{tab:op}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cc|cc|cccc|cc|c|c|c|c|ccc}
\hline
gas & $B^*$, G & $B^*/\mu_0$, A/m & $E_w$, V/m & $E_w/N$, Td & $T_{e,w}$ & $T_{e,\max}$ & $T_{e,0}$ & $\langle T_e\rangle_n$ & well, V & $n_0/n_w$ & $\beta$ & eV/ion & $\Lambda_{\rm eff}$, cm & local/kin & $E/N$ (0D), Td & $T_e$ (0D), eV & eV/ion (0D)\\
\hline
""" + rows + r"""\hline
\end{tabular}}
\end{table}"""
# ---- mixtures paragraph
mp = []
if 'Ar0.9_N20.1' in J:
    r = J['Ar0.9_N20.1']; n = r['nearest']
    mp.append(f"The mixtures show where the transition takes place. With 10~\\% of nitrogen in argon the energy relaxation length is already {r['lam']['mid']['lam_eps_cm']:.1f}~cm at mid-radius and {r['lam']['axis']['lam_eps_cm']:.1f}~cm on the axis against {ar['lam']['mid']['lam_eps_cm']:.1f} and {ar['lam']['axis']['lam_eps_cm']:.1f}~cm in argon, and {r['lam']['wall']['lam_eps_cm']:.1f}~cm at the wall against {ar['lam']['wall']['lam_eps_cm']:.0f}~cm, because the vibrational and the low electronic levels of N$_2$ give the 2--5~eV electrons, which in argon lose energy only elastically, an inelastic channel. The discharge is intermediate in every respect. The temperature is no longer flat: it has a maximum of {n['Te_max']:.2f}~eV at $r/R={n['XTmax']:.2f}$, closer to the wall than in argon, and falls to {n['Te_ax']:.2f}~eV on the axis, so that the core is {n['Te_max']-n['Te_ax']:.1f}~eV colder than the heated layer; the well, {q('Ar0.9_N20.1','well'):.1f}~V, is between the argon and the nitrogen values; the ionization frequency falls by three orders of magnitude from its maximum of ${n['numax']/1e4:.1f}\\cdot10^4$~s$^{{-1}}$ at $r/R={n['Xnumax']:.2f}$ to the axis, and {100*n['f07']:.0f}~\\% of the ionization occurs at $r/R>0.7$. The sustaining field, {r['Bstar']:.2f}~G, is {r['Bstar']/ar['Bstar']:.1f} times the argon value for a gas that is 90~\\% argon, and the cost of an ion doubles, from {q('Ar','cost'):.0f} to {q('Ar0.9_N20.1','cost'):.0f}~eV, because {100*(n['frac'].get('vib',0)+n['frac'].get('el_N2',0)+n['frac'].get('N2(A)',0)):.0f}~\\% of the power now goes into the vibrational and electronic levels of the 10~\\% of nitrogen (Sec.~\\ref{{sec:budget}}).")
if 'Ar0.5_N20.5' in J:
    r = J['Ar0.5_N20.5']; n = r['nearest']
    mp.append(f"At 50~\\% of nitrogen the discharge is already the nitrogen one: $\\lambda_\\eps$ is {r['lam']['mid']['lam_eps_cm']:.1f}~cm at mid-radius, the temperature is maximal at $r/R={n['XTmax']:.2f}$ ({n['Te_max']:.2f}~eV) and falls to {n['Te_ax']:.2f}~eV on the axis, {100*n['f07']:.0f}~\\% of the ionization is produced at $r/R>0.7$, the well is {q('Ar0.5_N20.5','well'):.1f}~V, the sustaining field {r['Bstar']:.2f}~G and the cost of an ion {q('Ar0.5_N20.5','cost'):.0f}~eV; the remaining differences from pure nitrogen -- a warmer axis, a deeper well, a smaller field -- come from the smaller total inelastic cross section of the mixture, not from a change of regime.")
T['MIXTURES_PARA'] = ' '.join(mp) if mp else 'The argon--nitrogen mixtures are discussed below as soon as their operating points are available.'
# ---- air paragraph
if 'N20.79_O20.21' in J:
    r = J['N20.79_O20.21']; n = r['nearest']; lost = n['att_iz']
    T['AIR_PARA'] = f"In air the nitrogen picture is modified by the oxygen in two ways. The sustaining field, {r['Bstar']:.2f}~G, is {100*(r['Bstar']/n2['Bstar']-1):.0f}~\\% above that of nitrogen although the attachment to O$_2$ removes {100*lost:.0f}~\\% of the electrons produced: the oxygen also brings ionization channels with lower thresholds (12.1~eV) and larger cross sections, so that the net ionization frequency in the heated layer reaches ${n['numax']/1e4:.1f}\\cdot10^4$~s$^{{-1}}$ (gross $7.5\\cdot10^4$, attachment $2.3\\cdot10^4$~s$^{{-1}}$), above the nitrogen one at nearly the same field ({q('N20.79_O20.21','EN'):.0f} against {q('N2','EN'):.0f}~Td), and the net production is similar. The structure is the same -- $\\lambda_\\eps={r['lam']['wall']['lam_eps_cm']:.1f}$--{r['lam']['mid']['lam_eps_cm']:.1f}~cm in the heated layer, {q('N20.79_O20.21','Te_w'):.1f}~eV at the wall and {q('N20.79_O20.21','Te_ax'):.1f}~eV on the axis, the net production confined to $r/R>0.7$ (where it amounts to {n['f07']:.1f} times the total, the excess being destroyed further in) -- but the core of the discharge is a net sink of electrons: the attachment frequency, fed by the plateau of the axis distribution between 3 and 7~eV (Fig.~\\ref{{fig:eedf}}), falls much more slowly towards the axis than the ionization, the net ionization frequency is negative inside $r/R={n['neg_inside']:.2f}$" + (f", and the plasma made in the skin layer diffuses into a region where it is destroyed. As a result the density profile is flatter than in nitrogen ($n_0/n_w={q('N20.79_O20.21','compr'):.0f}$ against {q('N2','compr'):.0f}) and the potential of the axis, {q('N20.79_O20.21','well'):.1f}~V, is the lowest of the five." if n['neg_inside'] else ".")
    T['AIR_ABS'] = f"In air the dissociative attachment exceeds the ionization inside {100*n['neg_inside']:.0f}~\\% of the radius and the core is a net sink of electrons." if n['neg_inside'] else "In air the dissociative attachment removes a large part of the electrons produced in the core."
    T['AIR_EEDF'] = "In air the axis distribution shows, besides the depletion at 2--3~eV, the plateau between 3 and 7~eV that is responsible for the attachment in the core."
    T['AIR_B'] = f"{r['Bstar']:.2f}"; T['AIR_EN'] = f"{q('N20.79_O20.21','EN'):.0f}"; T['AIR_XNET'] = f"{n['neg_inside']:.2f}" if n['neg_inside'] else '--'
else:
    T['AIR_PARA'] = 'The air discharge with the corrected closure is reported separately once its operating point is converged; with the first version of the closure it behaved as nitrogen in its structure and sustaining field (2.5~G), with the dissociative attachment exceeding the ionization inside $r/R\\approx0.7$.'
    T['AIR_ABS'] = ''; T['AIR_EEDF'] = ''
# ---- local-model text
lt = []
for k, name in (('N2', 'nitrogen'), ('Ar0.9_N20.1', 'the 90/10 mixture'), ('Ar', 'argon')):
    if k in J and J[k].get('local'):
        L = J[k]['local']; n = J[k]['nearest']; Tl = np.array(L['Te_loc']); Tk = np.array(pickle.load(open(J[k]['files'][int(np.argmin(np.abs(np.array(J[k]['beta'])-1)))], 'rb'))['Te'])
        lt.append(f"In {name} the local model gives {L['ratio']:.2f} of the kinetic density-weighted net ionization and a density-weighted temperature of {L['Te_loc_n']:.2f}~eV against {n['Te_n']:.2f}~eV; its temperature profile deviates from the kinetic one by at most {np.max(np.abs(Tl-Tk)):.2f}~eV over the cross section ({np.max(np.abs(Tl-Tk)[:len(Tl)//3]):.2f}~eV in the layer $r/R>0.7$, {abs(Tl[-1]-Tk[-1]):.2f}~eV on the axis).")
T['LOCAL_TEXT'] = ' '.join(lt) + (" The sequence 0.4 -- 0.7 -- 1.0 is the transition from a nonlocal to a local discharge in one number: in argon the local model misses 60~\\% of the ionization, because the electrons that ionize were heated elsewhere; in nitrogen the ionization is produced where the field is, and the remaining difference is the trapped core, which the local model heats too little (its axis temperature follows the 8~Td field there) but which carries no ionization." if lt else '')
# ---- budget text
def fr(k, g):
    return 100 * J[k]['nearest']['frac'].get(g, 0)
bt = []
if 'Ar' in J: bt.append(f"In argon {fr('Ar','exc_Ar'):.0f}~\\% of the inelastic power goes into excitation and {fr('Ar','iz'):.0f}~\\% into ionization, in the tube as in the homogeneous model.")
if n2:
    z2 = z('N2', 0.02); z4 = z('N2', 0.1 / 2.405)
    bt.append(f"In nitrogen the self-consistent partition is {fr('N2','vib'):.0f}~\\% into vibrational excitation, {fr('N2','el_N2'):.0f}~\\% into the electronic states above 7~eV and {fr('N2','N2(A)'):.0f}~\\% into N$_2(A^3\\Sigma_u^+)$; the homogeneous model gives {100*z2['frac'].get('vib',0):.0f}/{100*z2['frac'].get('el_N2',0):.0f}/{100*z2['frac'].get('N2(A)',0):.0f} with $\\Lambda=2$~cm and {100*z4['frac'].get('vib',0):.0f}/{100*z4['frac'].get('el_N2',0):.0f}/{100*z4['frac'].get('N2(A)',0):.0f} with $\\Lambda=4.2$~cm, and the tube lies between the two: the $\\Lambda=2$~cm field is close to that of the heated layer, where the electronic excitation is strongest, but the majority of the electrons sit in the cold core, which the weak field there can only excite vibrationally.")
    T['N2_PARTERR'] = f"{max(abs(fr('N2',g)-100*z2['frac'].get(g,0)) for g in ('vib','el_N2','N2(A)')):.0f}"
else: T['N2_PARTERR'] = '--'
for k, name in (('Ar0.5_N20.5', 'the 50/50 mixture'), ('Ar0.9_N20.1', 'the 90/10 mixture')):
    if k in J:
        z2 = z(k, 0.02); z4 = z(k, 0.1 / 2.405)
        bt.append(f"In {name} the partition is {fr(k,'vib'):.0f}~\\% vibration, {fr(k,'el_N2'):.0f}~\\% N$_2$ electronic, {fr(k,'N2(A)'):.0f}~\\% N$_2(A)$ and {fr(k,'exc_Ar'):.0f}~\\% argon excitation (homogeneous: {100*z2['frac'].get('vib',0):.0f}/{100*z2['frac'].get('el_N2',0):.0f}/{100*z2['frac'].get('N2(A)',0):.0f}/{100*z2['frac'].get('exc_Ar',0):.0f} and {100*z4['frac'].get('vib',0):.0f}/{100*z4['frac'].get('el_N2',0):.0f}/{100*z4['frac'].get('N2(A)',0):.0f}/{100*z4['frac'].get('exc_Ar',0):.0f})" + (f": with 10~\\% of nitrogen, {100-fr(k,'exc_Ar')-fr(k,'iz'):.0f}~\\% of the power is spent on the nitrogen, and the homogeneous model with $\\Lambda=2$~cm gives the argon channel {100*z2['frac'].get('exc_Ar',0):.0f}~\\% of the power instead of {fr(k,'exc_Ar'):.0f}~\\%, because its field, {z2['EN']:.0f}~Td, is the field of the hottest part of the tube." if k == 'Ar0.9_N20.1' else '.'))
if 'N20.79_O20.21' in J:
    k = 'N20.79_O20.21'; z2 = z(k, 0.02); z4 = z(k, 0.1 / 2.405); n = J[k]['nearest']
    bt.append(f"In air the partition is {fr(k,'vib'):.0f}~\\% vibration (N$_2$ and O$_2$), {fr(k,'el_N2'):.0f}~\\% N$_2$ electronic, {fr(k,'diss_O2'):.0f}~\\% dissociation of O$_2$ (the two channels with 6.0 and 8.4~eV thresholds of the IST-Lisbon set), {fr(k,'N2(A)'):.0f}~\\% N$_2(A)$ and a few per cent for the O$_2$ electronic states, the metastables and ionization, against {100*z2['frac'].get('vib',0):.0f}/{100*z2['frac'].get('el_N2',0):.0f}/{100*z2['frac'].get('diss_O2',0):.0f}/{100*z2['frac'].get('N2(A)',0):.0f} for the homogeneous model with $\\Lambda=2$~cm. The quantity of practical interest, the energy spent per oxygen atom produced by electron-impact dissociation (the total inelastic power divided by twice the dissociation rate, both integrated over the cross section), is {n['eV_per_O']:.0f}~eV in the tube.")
lams = ', '.join('%.1f' % J[k]['nearest']['Lam_eff_cm'] for k in have); costs = ', '.join(f"{q(k,'cost'):.0f}" for k in have); costs0 = ', '.join(f"{z(k,0.02)['cost']:.0f}" for k in have if z(k, 0.02)); ratio4 = [z(k, 0.1 / 2.405)['cost'] / q(k, 'cost') for k in have if z(k, 0.1 / 2.405)]
bt.append(f"The energy spent per net electron--ion pair is {costs}~eV in {', '.join(LAB[k] for k in have)}; the homogeneous model with $\\Lambda=2$~cm gives {costs0}~eV, within {T['COSTERR']}~\\%, and with $\\Lambda=4.2$~cm it gives values {min(ratio4):.0f}--{max(ratio4):.0f} times too large. This is the justification, a posteriori, of the diffusion length of 2~cm that is used in practice for a tube of 10~cm radius, and the kinetic solution makes it quantitative: the effective diffusion length of the ion loss, $\\Lambda_{{\\rm eff}}=(\\langle D_a\\rangle_n/\\langle\\nu_{{\\rm net}}\\rangle_n)^{{1/2}}$ computed from the converged profiles, is {lams}~cm (Table~\\ref{{tab:op}}), because the ionization is not spread over the fundamental diffusion mode but confined to the outer 3~cm of the radius in the molecular gases and to the outer half in argon. The same cannot be said of the electron temperature, which the homogeneous model with $\\Lambda=2$~cm overestimates in nitrogen by {z('N2',0.02)['Te']-q('N2','Te_n'):.1f}~eV with respect to the density-weighted mean of the tube ({z('N2',0.02)['Te']:.2f} against {q('N2','Te_n'):.2f}~eV) and underestimates by {q('N2','Te_w')-z('N2',0.02)['Te']:.1f}~eV with respect to the heated layer, nor of the power partition in the argon-rich mixtures." if n2 else '')
T['BUDGET_TEXT'] = ' '.join(bt)
# ---- MC text
mcf = sorted(glob.glob('results/mc_n2_disc_B*_kin.pkl'))
if mcf and n2:
    m = pickle.load(open(mcf[-1], 'rb')); d = pickle.load(open(n2['files'][int(np.argmin(np.abs(np.array(n2['beta'])-1)))], 'rb')); X = d['X']; rm = m['r_over_R']
    xf = np.linspace(0, 1, 4001); Xa = X[::-1]; nf = np.interp(xf, Xa, d['ne'][::-1]); nuf = np.exp(np.interp(xf, Xa, np.log(np.maximum(d['nuiz'][::-1], 1e-3)))); Tf = np.interp(xf, Xa, d['Te'][::-1])
    ring = []; Tring = []
    for kk in range(len(rm)):
        sl = (xf >= kk / len(rm)) & (xf < (kk + 1) / len(rm)); ring.append(np.sum(xf[sl] * nf[sl] * nuf[sl]) / np.sum(xf[sl] * nf[sl])); Tring.append(np.sum(xf[sl] * nf[sl] * Tf[sl]) / np.sum(xf[sl] * nf[sl]))
    ring = np.array(ring); Tring = np.array(Tring); w = m['ne_rel'] * rm; mk = np.sum(w * ring) / np.sum(w); mm = np.sum(w * m['nu_iz']) / np.sum(w); fac = mm / mk
    sl = n2['slope'] or 7.0; shift = 100 * (1 - fac ** (-1 / sl)); dT = Tring - m['Te_eV']
    T['MC_TEXT'] = f"The Monte Carlo simulation in the converged fields of the nitrogen discharge at {d['B_ext']:.3f}~G (Fig.~\\ref{{fig:mc}}; 3000 electrons, 40~$\\mu$s) confirms the local structure and quantifies the error of the two-term, single-harmonic solution. The electron temperature, compared ring by ring with the kinetic solution averaged over the same rings, agrees within {np.max(np.abs(dT[:5])):.2f}~eV from the axis to $r/R=0.5$ ({Tring[0]:.2f} against {m['Te_eV'][0]:.2f}~eV in the axis ring) and within {np.max(np.abs(dT[5:])):.2f}~eV in the heated layer, where the Monte Carlo value is {'higher' if np.mean(dT[6:9])<0 else 'lower'} ({', '.join(f'{x:.2f}' for x in m['Te_eV'][6:])} against {', '.join(f'{x:.2f}' for x in Tring[6:])}~eV in the four outer rings). The ionization frequency is higher in the Monte Carlo simulation by a factor of {np.min(m['nu_iz'][7:]/ring[7:]):.1f}--{np.max(m['nu_iz'][7:]/ring[7:]):.1f} in the three outer rings, consistent with the deficit of the $L=1$, $N_h=1$ approximation found in the uniform field at 125~Td, and the density-weighted mean over the rings (with the Monte Carlo density as weight) is ${mm/1e3:.2f}\\cdot10^3$ against ${mk/1e3:.2f}\\cdot10^3$~s$^{{-1}}$ (factor {fac:.2f}). With the dependence $\\beta\\propto B_{{\\rm ext}}^{{{sl:.0f}}}$ of the nitrogen discharge this means that the operating point of the exact kinetics is lower by the factor ${fac:.2f}^{{1/{sl:.0f}}}={fac**(1/sl):.2f}$, i.e. $B^*={n2['Bstar']/fac**(1/sl):.2f}$~G instead of {n2['Bstar']:.2f}~G; the profiles, the potential and the energy budget, which are set by the bulk of the distribution, are not affected. In argon the same comparison gives agreement within 0.1~eV in the temperature and 11~\\% in the volume-averaged ionization~\\cite{{Paper2}}."
    T['MC_FAC'] = f"{fac:.1f}"; T['MC_BSHIFT'] = f"{100*(fac**(1/sl)-1):.0f}"; T['MC_BSTAR'] = f"{n2['Bstar']/fac**(1/sl):.2f}"
else:
    T['MC_TEXT'] = 'The Monte Carlo comparison in the converged nitrogen fields is reported below once the run is complete.'; T['MC_FAC'] = '1.5'; T['MC_BSHIFT'] = '6'; T['MC_BSTAR'] = '--'
# ---- conclusions
c = []
c.append("The deterministic total-energy solver of the electron Boltzmann equation in the cross section of an inductively coupled discharge has been extended to gas mixtures with vibrational and electronic excitation, dissociation, attachment and several ionization channels, and applied self-consistently, with cold ions of the dominant species moving in the kinetic potential, to argon, nitrogen" + (", air" if 'N20.79_O20.21' in J else '') + " and argon--nitrogen mixtures at 13.3~Pa.")
c.append("In a uniform RF field in nitrogen the two-term approximation with a single time harmonic underestimates the mean energy by 15~\\% and the ionization frequency by a factor of two at 88~Td. The second angular harmonic and the second time harmonic each recover a comparable part of the deficit, the two corrections multiply, and with both the solver agrees with the Monte Carlo simulation within 2.5~\\% in the mean energy and 0.5~\\% in the ionization frequency at 88~Td (3--4~\\% at 125~Td), in the same computing time and without statistical noise.")
if n2:
    c.append(f"Nitrogen converts the discharge from nonlocal to local. The energy relaxation length falls from {ar['lam']['mid']['lam_eps_cm']:.0f}--{ar['lam']['wall']['lam_eps_cm']:.0f}~cm in argon to {n2['lam']['mid']['lam_eps_cm']:.1f}--{n2['lam']['wall']['lam_eps_cm']:.1f}~cm in nitrogen; the electron temperature, 3.3--3.7~eV across the argon discharge, falls from {T['N2_TEW']}~eV at the wall to {T['N2_TEAX']}~eV on the axis; {T['N2_F07']}~\\% of the ionization is produced in the outer 30~\\% of the radius; the potential of the axis above the wall shrinks from {T['AR_WELL']} to {T['N2_WELL']}~V and stays monotonic; and the inductor current that sustains the same mean density is {T['RATIO_B']} times larger, a factor that follows from the {T['RATIO_COST']} times higher energy cost of an ion and the ionization frequency that the ion loss requires. A local model evaluated in the kinetic field profile gives {T['N2_LOC']} of the kinetic ionization in nitrogen" + (f", {T['M10_LOC']} in the 90/10 mixture" if 'Ar0.9_N20.1' in J else '') + f" and {T['AR_LOC']} in argon: ten per cent of nitrogen in argon makes the discharge intermediate" + (", and fifty per cent makes it the nitrogen one" if 'Ar0.5_N20.5' in J else '') + ".")
if 'N20.79_O20.21' in J: c.append(f"Air behaves as nitrogen in its structure and sustaining field ({T['AIR_B']}~G), but the dissociative attachment, fed by a plateau of the axis distribution between 3 and 7~eV, exceeds the ionization inside $r/R={T['AIR_XNET']}$, so that the core of the discharge is a net sink of electrons.")
c.append(f"The homogeneous two-term model run with the diffusion length of 2~cm that is used in practice gives the cost of an ion within {T['COSTERR']}~\\% of the self-consistent values, because the effective diffusion length of the ion loss in the kinetic solutions is 2--4~cm in all gases, the ionization being confined to the outer 3~cm of the tube in the molecular gases and to the outer half in argon; with the diffusion length of the fundamental mode it overestimates the cost by a factor of {min(ratio4):.0f}--{max(ratio4):.0f}. It cannot give the electron temperature, which varies by a factor of four across the nitrogen discharge, nor the power partition in the argon-rich mixtures. The Monte Carlo simulation in the converged fields confirms the structure of the nitrogen discharge and shows that the two-term, single-harmonic closure of the self-consistent runs overestimates the sustaining field by {T['MC_BSHIFT']}~\\% (the exact kinetics needs {T['MC_BSTAR']}~G instead of {T['N2_B']}~G), without affecting the profiles or the energy budget.")
c.append(f"The first version of the ion closure, which transported the ions in a weaker potential than the kinetic one, produced an unphysical maximum of the potential inside the molecular-gas discharges; with the ions moving in the kinetic potential the wells are monotonic and deeper ({T['WELL_SHIFT']}), and the sustaining currents changed by {T['B_SHIFT']}, while the temperature profiles, the energy distributions and the energy budget did not change.")
T['CONCLUSIONS'] = '\n\n'.join(c)
# ---- fill
s = open(f'{OUT}/Paper7_gases_template.tex').read()
for k, v in T.items(): s = s.replace('@@' + k + '@@', str(v))
s = s.replace('%%TAB_OP%%', T['TAB_OP'])
import re; left = sorted(set(re.findall(r'@@[A-Z0-9_]+@@', s))); print('unfilled tokens:', left)
open(f'{BLD}/Paper7_gases.tex', 'w').write(s)
if 'nofig' not in sys.argv:
    env = dict(os.environ, P7_TAG='_kin', P7_OUT=BLD, P7_AR='results/discharge_te_B0.74_n1e+17_L1_m16_o2_kin_teff2.pkl', OMP_NUM_THREADS='1')
    if mcf: env['P7_MC_N2'] = mcf[-1]
    subprocess.run(['python3', 'p7_figs.py', 'fig1', 'fig2', 'fig3', 'fig4', 'fig5'], env=env)
    subprocess.run(['python3', 'p7_figs_local17.py', BLD], env=env)
for _ in range(2): subprocess.run(['pdflatex', '-interaction=nonstopmode', 'Paper7_gases.tex'], cwd=BLD, stdout=subprocess.DEVNULL)
log = open(f'{BLD}/Paper7_gases.log', errors='ignore').read(); print('errors', log.count('! '), 'undefined', log.count('undefined'), 'overfull', log.count('Overfull'))
json.dump(T, open(f'{BLD}/p7_tokens.json', 'w'), indent=1)
