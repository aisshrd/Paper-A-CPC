"""collect numbers for the article patch -> results/te_numbers.json"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import numpy as np, pickle, json, glob, sys
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
rm=m1['r_over_R']; Tm=(m1['Te_eV']+m2['Te_eV'])/2; num=(m1['nu_iz']+m2['nu_iz'])/2; nm=(m1['ne_rel']+m2['ne_rel'])/2; nm=nm/np.interp(0.5,rm,nm)
wm=2*rm; Im=np.sum(wm*nm*num)/np.sum(wm*nm)
def metrics(p):
    d=pickle.load(open(p,'rb')); X=d['X'][::-1]; T=d['Te'][::-1]; n=d['ne'][::-1]; n=n/np.interp(0.5,X,n); nu=d['nuiz'][::-1]
    ni=np.interp(rm,X,n); nui=np.interp(rm,X,nu); Ti=np.interp(rm,X,T)
    I=np.sum(wm*ni*nui)/np.sum(wm*ni); sel=rm>0.6; fr=np.sum((wm*ni*nui)[sel])/np.sum(wm*ni*nui)
    return dict(row=f'{T[-1]:.2f} / {T.max():.2f} / {T[0]:.2f}',n95=float(np.interp(0.95,X,n)),ratio=float(I/Im),fr=float(fr),nu25=float(np.interp(0.25,X,nu)),nu5=float(np.interp(0.05,X,nu)),
                nu85=float(np.interp(0.85,X,nu)),nu95=float(np.interp(0.95,X,nu)),dn=float(np.max(np.abs(ni-nm))),Te_wall=float(T[-1]),Te_axis=float(T[0]),Te_max=float(T.max()),
                t=float(d.get('t_solve',np.nan)+d.get('t_assemble',0)))
P={}
a=metrics('results/te_reffields_ne120_m16_L1.pkl'); b=metrics('results/te_reffields_ne160_m24_L1.pkl'); c=metrics('results/te_reffields_ne120_m16_L1_abs.pkl')
try: l2=metrics('results/te_reffields_ne120_m16_L2.pkl')
except Exception: l2=metrics('results/te_centred/te_reffields_ne120_m16_L2.pkl'); print('WARNING: using centred L=2 result')
P.update(r16=a['row'],n16=a['n95'],ratio_m16=a['ratio'],fr16=a['fr'],nu25_16=a['nu25'],nu5_16=a['nu5'],Te_w16=a['Te_wall'],t_m16=a['t'])
P.update(r24=b['row'],n24=b['n95'],ratio_m24=b['ratio'],fr24=b['fr'],nu25_24=b['nu25'],nu5_24=b['nu5'],t_m24=b['t'],dn_max=max(a['dn'],b['dn']))
P.update(rL2=l2['row'],nL2=l2['n95'],ratio_L2=l2['ratio'],frL2=l2['fr'],nu25_L2=l2['nu25'],t_L2=l2['t'],dL2_layer=100*(l2['nu85']/a['nu85']-1))
P.update(rabs=c['row'],nabs=c['n95'],ratio_abs=c['ratio'],frabs=c['fr'],nu25_abs=c['nu25'])
# self-consistent TE point
pts=[]
for p in sorted(glob.glob('results/discharge_te_B*_n1e+17_L1_m16.pkl')):
    d=pickle.load(open(p,'rb')); pts.append((d['B_ext'],d['balance'],d))
pts.sort(); P['scan']=[(B,bal) for B,bal,_ in pts]
Bs=np.array([B for B,_,_ in pts]); bal=np.array([b for _,b,_ in pts])
if len(pts)>=2:
    Bstar=float(np.interp(1.0,bal,Bs)); P['Bstar']=Bstar
    # operating point: the run closest to balance=1
    B,b,d=min(pts,key=lambda t:abs(t[1]-1)); X=d['X']
    import scipy.constants as co
    P.update(Bop=float(B),bal_op=float(b),well=float(d['phi'].max()-d['phi'].min()),compr=float(d['ne'][-1]/d['ne'][0]),Te_wall=float(d['Te'][0]),Te_max=float(d['Te'].max()),Te_axis=float(d['Te'][-1]),
             X_Temax=float(X[np.argmax(d['Te'])]),E_wall=float(abs(d['E'][2][0])*2),IL=float(B*1e-4/co.mu_0),nuiz_max=float(d['nuiz'].max()),X_numax=float(X[np.argmax(d['nuiz'])]),
             n95=float(np.interp(0.95,X[::-1],(d['ne']/np.interp(0.5,X[::-1],d['ne'][::-1]))[::-1])))
json.dump(P,open('results/te_numbers.json','w'),indent=1,default=float); print(json.dumps(P,indent=1,default=float))
