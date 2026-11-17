"""Paper 6 (v1.7.0): self-consistent section from the kin runs + figures + compile. usage: python3 p6_build17.py [nofig]"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import json, os, sys, subprocess, numpy as np, shutil
OUT='papers/p6_v17'; BLD=OUT+'/build'; os.makedirs(BLD,exist_ok=True)
subprocess.run(['python3','p6_sc_numbers.py'],check=True,stdout=subprocess.DEVNULL)
J=json.load(open('results/p6_sc_kin.json'))
def pt(B0,Bext=0.74):
    for p in J.get(str(B0),{}).get('points',[]):
        if abs(p['Bext']-Bext)<1e-6: return p
    return None
def second(B0):
    ps=[p for p in J.get(str(B0),{}).get('points',[]) if abs(p['Bext']-0.74)>1e-6]; return ps[0] if ps else None
import pickle as _pk
def _ionfac(B0):
    d=_pk.load(open(f'results/discharge_mag_B0{B0}_Bext0.74_kin.pkl','rb')); w=d['wdisk']*d['ne']
    return float(np.sum(w*d['nuiz'])/np.sum(w)), float(d['ne'][0]/(np.sum(d['wdisk']*d['ne'])/np.sum(d['wdisk'])))
_IF={B0:_ionfac(B0) for B0 in (0,10,20,40)}
rows=''; have={}
for B0 in (0,10,20,40):
    p=pt(B0)
    if p is None: continue
    have[B0]=p; q=second(B0); Bs=J[str(B0)].get('Bstar')
    cur='--' if B0==0 else (f"{100*(Bs/0.742-1):+.0f}~\\%" if Bs else f"{100*((1/p['beta'])**(1/4.0)-1):+.0f}~\\%$^*$")
    rows+=f"{B0} & {p['beta']:.2f} ({p['beta_range'][0]:.2f}--{p['beta_range'][1]:.2f}) & {p['Te'][0]:.2f} / {p['Te'][1]:.2f} / {p['Te'][2]:.2f} & {p['XTmax']:.2f} & {p['well']:.1f} & {p['compr']:.1f} & {('%.2f'%p['Er_ratio']) if p['Er_ratio'] else '--'} & {int(round(p['numax'],-2))} / {p['nuax']:.0f} & {(('%.2f G (%.2f%s)'%(q['Bext'],q['beta'],'' if q.get('conv',True) else '$^\\dagger$')) if q else '--')} & {('%.3f'%Bs) if Bs else '--'} & {cur}\\\\\n"
b=have
def g(B,k,i=None):
    v=b[B][k]; return v if i is None else v[i]
Bst={B:J[str(B)].get('Bstar') for B in have}
red={B:(100*(1-Bst[B]/0.742) if Bst.get(B) else 100*(1-(1/b[B]['beta'])**0.25)) for B in have if B>0}
tab=f"""\\begin{{table}}[!ht]
\\caption{{Self-consistent discharge with the corrected ion closure at the unmagnetized operating current $B_{{\\rm ext}}=0.74$~G: ionization balance $\\beta$ (mean and range over the last six Picard iterations), electron temperature at the wall / maximum / axis and the position of the maximum, potential well, density compression $n_e(0)/n_e(r_1)$, maximum of the radial RF field relative to the wall value of $E_\\theta$, maximum and axis values of the ionization frequency, the second inductor current computed for the same $B_0$ with its balance, the operating point $B^*(B_0)$ by log-linear extrapolation through the two currents (both above balance; local exponents $\\beta\\propto B_{{\\rm ext}}^{{3.9}}$, $^{{3.0}}$ and $^{{3.5}}$ at 10, 20 and 40~G), and the resulting change of the inductor current that sustains $n_0=10^{{17}}$~m$^{{-3}}$ ($^\\dagger$: stopped after 40 iterations with $\\beta$ still rising by 0.1~\\% per iteration; the corresponding $B^*$ is uncertain by about 3~\\%).}}\\label{{tab:sc}}
\\scriptsize\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{ccccccccccc}}
\\hline
$B_0$, G & $\\beta$ & $T_e$, eV & $r/R$ of max & $\\Delta\\varphi$, V & $n_e(0)/n_e(r_1)$ & $\\frac{{\\max|E_r|}}{{|E_\\theta(R)|}}$ & $\\nu_\\iz^{{\\max}}$ / $\\nu_\\iz(0)$, s$^{{-1}}$ & 2nd $B_{{\\rm ext}}$ ($\\beta$) & $B^*$, G & $\\Delta I$\\\\
\\hline
{rows}\\hline
\\end{{tabular}}
\\end{{table}}
"""
def fmt_list(vals,f='{:.2f}'): return ', '.join(f.format(v) for v in vals)
Bl=[B for B in (10,20,40) if B in have]
txt=f"""The prescribed-field computation keeps the fields of the unmagnetized discharge. In the self-sustained discharge the field, the potential and the ion profile respond to the changed electron kinetics, and the question of practical interest is how the inductor current that sustains a given density changes with $B_0$. We compute the discharge of~\\cite{{Paper4}} (13.3~Pa, $n_0=10^{{17}}$~m$^{{-3}}$, corrected ion closure) at the unmagnetized operating current $B_{{\\rm ext}}=0.74$~G with $B_0=10$, 20 and 40~G in the insulated limit (I), in which the radial RF field is updated inside each Picard iteration from the response kernels, and at a second, lower current for each $B_0$ to locate the magnetized operating point. Figure~\\ref{{fig:sc}} and Table~\\ref{{tab:sc}} give the result. At $B_0=0$ the balance is $\\beta={b[0]['beta']:.3f}$, i.e. the current is at the operating point to within 0.5~\\%.

{tab}
\\begin{{figure}}[!ht]
\\includegraphics[width=\\textwidth]{{Fig2}}
\\caption{{Self-consistent discharge at 13.3~Pa, $n_0=10^{{17}}$~m$^{{-3}}$ and the fixed inductor field $B_{{\\rm ext}}=0.74$~G with $B_0=0$, 10, 20 and 40~G (insulated limit, corrected ion closure): (a) electron temperature, (b) ambipolar potential counted from the wall cell, (c) electron density normalized on the axis, (d) ionization frequency.}}\\label{{fig:sc}}
\\end{{figure}}

The density profile steepens: the compression is {b[0]['compr']:.1f} without field and {fmt_list([b[B]['compr'] for B in Bl],'{:.1f}')} at {', '.join(str(B) for B in Bl)}~G, because it is set by the transport of the unmagnetized ions in the kinetic potential and by the effective temperature of the electron response, which the field lowers in the core. The potential follows the temperature of the core: the well changes from {b[0]['well']:.1f}~V to {fmt_list([b[B]['well'] for B in Bl],'{:.1f}')}~V as the axis temperature falls from {b[0]['Te'][2]:.2f} to {fmt_list([b[B]['Te'][2] for B in Bl])}~eV, while the wall temperature rises from {b[0]['Te'][0]:.2f} to {fmt_list([b[B]['Te'][0] for B in Bl])}~eV and the maximum of $T_e$ moves from $r/R={b[0]['XTmax']:.2f}$ to {fmt_list([b[B]['XTmax'] for B in Bl])}. The RF field at the wall is unchanged to {max(abs(b[B]['Ew']/b[0]['Ew']-1) for B in Bl)*100:.0f}~\\% and the skin depth changes by at most {max(100*abs(1-b[B]['delta']/b[0]['delta']) for B in Bl):.0f}~\\% ({b[0]['delta']:.2f}~cm without field, {fmt_list([b[B]['delta'] for B in Bl],'{:.2f}')}~cm at {', '.join(str(B) for B in Bl)}~G, within the grid uncertainty of the layer conductivity), since the conductivity of the layer, which closes Maxwell's equations, is restored by the radial field, which reaches {fmt_list([b[B]['Er_ratio'] for B in Bl])} times the wall value of $E_\\theta$. What changes is the ionization: at the same inductor current $\\beta$ rises to {fmt_list([b[B]['beta'] for B in Bl])} at {', '.join(str(B) for B in Bl)}~G, i.e. the magnetized discharge produces {b[Bl[0]]['beta']:.1f}--{b[Bl[-1]]['beta']:.1f} times the ions it loses; its maximum rises from {int(round(b[0]['numax'],-2))} to {', '.join(str(int(round(b[B]['numax'],-2))) for B in Bl)}~s$^{{-1}}$ at $r/R={fmt_list([b[B]['Xnumax'] for B in Bl])}$ while the axis value is {fmt_list([b[B]['nuax'] for B in Bl],'{:.0f}')}~s$^{{-1}}$ against {b[0]['nuax']:.0f} without field. The second current computed for each $B_0$ (Table~\\ref{{tab:sc}}) lies, like the first one, above the magnetized operating point (both currents give $\\beta>1$; the local exponent of $\\beta(B_{{\\rm ext}})$ falls from 4.0 without field to 3.9, 3.0 and 3.5, and the 40~G point at 0.60~G was stopped after 40 iterations with $\\beta$ still creeping by 0.1~\\% per iteration), so the operating points are log-linear extrapolations with an uncertainty of a few per cent: $B^*\\approx{', '.join(('%.3f'%Bst[B]) if Bst.get(B) else '--' for B in Bl)}$~G at {', '.join(str(B) for B in Bl)}~G, i.e. the inductor current that sustains the density is {', '.join(('%.0f'%red[B]) for B in Bl)}~\\% lower than without the field{'' if all(Bst.get(B) for B in Bl) else ' (where the second point is missing, from the exponent 4 of the unmagnetized discharge)'}. The gain in $\\beta$ has two factors. The density-weighted ionization frequency per electron rises by {fmt_list([_IF[B][0]/_IF[0][0] for B in Bl])} at {', '.join(str(B) for B in Bl)}~G, i.e. by the same or slightly less than in the prescribed fields of Section~\\ref{{sec:presc}} (1.34, 1.70 and 2.41); the rest comes from the ion loss, which at fixed mean density falls by the factors {fmt_list([b[B]['beta']/b[0]['beta']/(_IF[B][0]/_IF[0][0]) for B in Bl])} relative to its unmagnetized value because the steeper density profile lowers the wall density relative to the mean ({_IF[0][1]:.3f} to {fmt_list([_IF[B][1] for B in Bl],'{:.3f}')}) while the unmagnetized ions leave at the Bohm speed. Part of the reduction of the inductor current is therefore a confinement effect that depends on holding the mean density fixed.

The Picard iteration with the radial field inside it converges more slowly than without the field and needs stronger damping as $B_0$ grows (relaxation {', '.join(['0.5/0.35','0.4/0.25','0.25/0.15'][:len(Bl)])} for the fields/potential at {', '.join(str(B) for B in Bl)}~G): the balance settles to the ranges given in Table~\\ref{{tab:sc}} within {', '.join(str(b[B]['n_iter']) for B in Bl)} iterations, and the residual radial current is {fmt_list([100*b[B]['Er_res'] for B in Bl],'{:.1f}')}~\\% of the azimuthal one."""
abs_txt=f"In the self-consistent discharge at fixed inductor current the RF field at the wall is unchanged and the skin depth shrinks by less than 10~\\%, the well shallows from {b[0]['well']:.1f} to {b[Bl[-1]]['well']:.1f}~V and the density compression nearly doubles ({b[0]['compr']:.0f} to {b[Bl[-1]]['compr']:.0f}) as the axis cools, while the ionization balance rises to {'--'.join('%.1f'%b[B]['beta'] for B in (Bl[0],Bl[-1]))} at {Bl[0]}--{Bl[-1]}~G, so that the inductor current that sustains the density falls by {red[Bl[0]]:.0f}--{red[Bl[-1]]:.0f}~\\%."
concl=f"3. In the self-sustained discharge at fixed inductor current the RF field at the wall does not change and the skin depth by less than 10~\\% (the radial RF field restores the conductivity of the layer), the density compression rises from {b[0]['compr']:.0f} to {b[Bl[-1]]['compr']:.0f} as the core cools (the ions are unmagnetized), the well shallows from {b[0]['well']:.1f} to {b[Bl[-1]]['well']:.1f}~V at {Bl[-1]}~G, and the ionization balance rises to {fmt_list([b[B]['beta'] for B in Bl])} at {', '.join(str(B) for B in Bl)}~G, so that the inductor current that sustains the density is {', '.join('%.0f'%red[B] for B in Bl)}~\\% lower, as extrapolated from a second current at each $B_0$; about half of the gain in $\\beta$ at 40~G comes from more ionization per electron and the rest from the lower wall density of the steeper profile."
s=open(f'{OUT}/Paper6_magnetized.tex').read().replace('%%SC_TEXT%%',txt).replace('%%SC_ABS%%',abs_txt).replace('%%SC_CONCL%%',concl)
open(f'{BLD}/Paper6_magnetized.tex','w').write(s)
if 'nofig' not in sys.argv: subprocess.run(['python3','p6_figs.py'],env=dict(os.environ,P6_OUT=BLD,OMP_NUM_THREADS='1'),check=True)
for _ in range(2): subprocess.run(['pdflatex','-interaction=nonstopmode','Paper6_magnetized.tex'],cwd=BLD,stdout=subprocess.DEVNULL)
log=open(f'{BLD}/Paper6_magnetized.log',errors='ignore').read(); print('errors',log.count('! '),'undefined',log.count('undefined'),'overfull',log.count('Overfull')); print(rows)
