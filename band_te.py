"""band-averaged mean energies from EEDF histograms (MC) and from the TE solution, with and without the u<1 eV group"""
import os as _os, sys as _sys; _R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); _sys.path[:0] = [_os.path.join(_R, 'src'), _os.path.join(_R, 'scripts')]   # release layout
import os as _os, sys as _sys
_ROOT=_os.path.abspath(_os.path.join(_os.path.dirname(__file__),'..')); _sys.path.insert(0,_os.path.join(_ROOT,'src')); _os.chdir(_ROOT)
import numpy as np, pickle, sys
m1=pickle.load(open('results/mc_cyl_real_N3000_s1.pkl','rb')); m2=pickle.load(open('results/mc_cyl_real_N3000_s2.pkl','rb'))
uc,h=m1['eedf']; h=(h+m2['eedf'][1])/2; du=0.5
print('run',sys.argv[1])
d=pickle.load(open(sys.argv[1],'rb')); X=d['X']; ec=d['ec']; F0=d['F0']; uk=d['ukin']; w=d['w']; wd=(d['X']*0+1)
for b,lab,sel in [(0,'wall 0.9-1',X>0.9),(1,'mid 0.4-0.6',(X>0.4)&(X<0.6)),(2,'axis 0-0.2',X<0.2)]:
    hm=h[b]; Tm=np.sum(hm*uc)/np.sum(hm)*2/3; s=uc>1; Tm1=np.sum((hm*uc)[s])/np.sum(hm[s])*2/3; fr_m=np.sum(hm[~s])/np.sum(hm)
    ks=np.where(sel)[0]; num=0; den=0; num1=0; den1=0; 
    for k in ks:
        ok=uk[:,k]>0; ww=w[ok,k]*F0[ok,k]; u=uk[ok,k]; num+=np.sum(ww*u); den+=np.sum(ww); s1=u>1; num1+=np.sum((ww*u)[s1]); den1+=np.sum(ww[s1])
    print(f'{lab:12s} Te(all): MC {Tm:.2f}  TE {num/den*2/3:.2f} | Te(u>1eV): MC {Tm1:.2f}  TE {num1/den1*2/3:.2f} | fraction u<1eV: MC {fr_m:.3f} TE {1-den1/den:.3f}')
